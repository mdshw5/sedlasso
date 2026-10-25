"""A small, file-based workflow manager for the dashboard's heavy jobs.

Every job is a directory under ``data/jobs``: an immutable ``job.json`` spec, a mutable ``status.json`` the page
polls, and a ``job.log``. The dashboard only ever writes a spec and reads a status; the work itself runs in a
separate process started by :mod:`scripts.job_manager`, so a ball scan, a report build or a video export can never
hold the Streamlit script thread.

Why files instead of a broker: the project already uses atomically written status files for every scan, the
dashboard may run on a machine with no Redis/Celery, and a job must survive a Streamlit reload. A directory is
also the most debuggable queue there is - ``cat data/jobs/<id>/status.json`` tells the whole story.

The manager allows at most ``SOCCER_MAX_JOBS`` concurrent jobs (default 2) and treats the resources in a spec as
exclusive. Marking GPU work ``resources=["gpu"]`` keeps two scans off the same GPU while a CPU-only report build can
run beside one, which is the workflow the dashboard needs: start the long enrichment for one match, then build a
report for another.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOBS_ROOT = REPO_ROOT / "data" / "jobs"
MANAGER_SCRIPT = REPO_ROOT / "scripts" / "job_manager.py"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_job.py"

ACTIVE_STATES = frozenset({"queued", "running"})
TERMINAL_STATES = frozenset({"done", "error", "cancelled", "stale"})
# A manager that has not written its heartbeat for this long is treated as stopped. The manager writes every 2 s.
MANAGER_STALE_S = 20.0
DEFAULT_MAX_JOBS = 2
KEEP_PER_MATCH = 8
KEEP_TOTAL = 40


def jobs_root() -> Path:
    """Where job state lives; ``SOCCER_JOBS_ROOT`` makes it overridable for tests and shared deployments."""
    return Path(os.environ.get("SOCCER_JOBS_ROOT", DEFAULT_JOBS_ROOT)).expanduser()


def _plain(value):
    """JSON-safe values for a spec: Paths become strings, tuples become lists, recursively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def read_json(path: Path) -> dict:
    """A JSON object from ``path``, or ``{}`` when it is missing or half-written."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def pid_alive(pid: int | None) -> bool:
    """Whether ``pid`` names a live process (a PermissionError means it exists but is not ours)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def _terminate(pid: int, *, grace_s: float = 3.0) -> None:
    """Ask a process to stop, then insist if it does not."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(int(pid), sig)
        except ProcessLookupError:
            return
        except OSError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return
            time.sleep(0.1)


class JobStore:
    """Submit and inspect jobs; the actual execution belongs to ``scripts/job_manager.py``."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else jobs_root()

    # --- paths ----------------------------------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / str(job_id)

    def spec_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def status_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    # --- submitting -----------------------------------------------------------------------------------------
    def submit(
        self,
        *,
        kind: str,
        label: str,
        match_id: str,
        argv: Sequence[str] = (),
        watch: str | Path | None = None,
        params: dict | None = None,
        resources: Sequence[str] = (),
        requested_by: str = "dashboard",
    ) -> dict:
        """Queue one job and make sure a manager is awake to run it.

        The status is written before the manager is asked to start, so a manager that is slow to boot (or absent)
        still leaves the page with a truthful "queued" row instead of a job that vanished.
        """
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        job_id = f"{stamp}-{_slug(kind)}-{uuid.uuid4().hex[:6]}"
        spec = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "match_id": str(match_id),
            "created": now,
            "requested_by": requested_by,
            "argv": [str(part) for part in argv],
            "watch": None if watch is None else str(watch),
            "params": _plain(params or {}),
            "resources": [str(item) for item in resources],
        }
        status = {
            "job_id": job_id,
            "kind": kind,
            "label": label,
            "match_id": str(match_id),
            "state": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Queued - the job manager will start it.",
            "created": now,
            "updated": now,
            "requested_by": requested_by,
            "resources": spec["resources"],
            "watch": spec["watch"],
            "params": spec["params"],
            "log": str(self.log_path(job_id)),
        }
        _atomic_write(self.spec_path(job_id), spec)
        _atomic_write(self.status_path(job_id), status)
        self.prune()
        if not self.ensure_manager():
            status["message"] = (
                "Queued, but the job manager could not be started. Check that the repository's data/jobs "
                "directory is writable."
            )
            _atomic_write(self.status_path(job_id), status)
        return status

    # --- reading --------------------------------------------------------------------------------------------
    def status(self, job_id: str) -> dict:
        return read_json(self.status_path(job_id))

    def jobs(
        self,
        *,
        match_id: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """Statuses, newest first, across every match when ``match_id`` is omitted."""
        found: list[dict] = []
        if not self.root.exists():
            return []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            status = read_json(path / "status.json")
            if not status.get("job_id"):
                continue
            if match_id is not None and str(status.get("match_id")) != str(match_id):
                continue
            if active_only and status.get("state") not in ACTIVE_STATES:
                continue
            found.append(status)
        found.sort(key=lambda item: float(item.get("created") or 0.0), reverse=True)
        return found if limit is None else found[: max(0, int(limit))]

    def active(self, *, match_id: str | None = None) -> list[dict]:
        return self.jobs(match_id=match_id, active_only=True)

    def has_active(self, match_id: str, *, kinds: Iterable[str] | None = None) -> bool:
        wanted = None if kinds is None else set(kinds)
        return any(
            wanted is None or str(job.get("kind")) in wanted
            for job in self.active(match_id=match_id)
        )

    def manager_state(self) -> dict:
        return read_json(self.root / "manager.json")

    def manager_alive(self) -> bool:
        state = self.manager_state()
        return (
            pid_alive(int(state.get("pid") or 0))
            and time.time() - float(state.get("updated") or 0.0) < MANAGER_STALE_S
        )

    # --- control --------------------------------------------------------------------------------------------
    def retry(self, job_id: str) -> dict:
        """Queue a fresh copy of a finished job's spec; the old status stays as the record of what failed."""
        spec = read_json(self.spec_path(job_id))
        if not spec:
            return {}
        return self.submit(
            kind=str(spec.get("kind") or "command"),
            label=str(spec.get("label") or "Retry"),
            match_id=str(spec.get("match_id") or ""),
            argv=spec.get("argv") or (),
            watch=spec.get("watch"),
            params=spec.get("params") or {},
            resources=spec.get("resources") or (),
        )

    def cancel(self, job_id: str) -> dict:
        """Cancel a queued job immediately, or ask a running runner to stop its child."""
        status = self.status(job_id)
        if not status or status.get("state") in TERMINAL_STATES:
            return status
        status["cancel_requested"] = True
        status["message"] = "Cancelling..."
        status["updated"] = time.time()
        _atomic_write(self.status_path(job_id), status)
        if status.get("state") == "queued":
            status.update(
                state="cancelled",
                stage="cancelled",
                message="Cancelled before it started.",
                finished=time.time(),
            )
            _atomic_write(self.status_path(job_id), status)
        elif pid_alive(int(status.get("pid") or 0)):
            _terminate(int(status["pid"]))
        return status

    def clear_finished(self, *, match_id: str | None = None) -> int:
        """Delete terminal job directories, leaving active work alone; returns how many were removed."""
        removed = 0
        for status in self.jobs(match_id=match_id):
            if status.get("state") not in TERMINAL_STATES:
                continue
            directory = self.job_dir(str(status["job_id"]))
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
        return removed

    def prune(self) -> int:
        """Keep the queue small: the active jobs plus the newest few per match.

        Every dashboard poll reads the status files, so an ever-growing archive would slowly make the page's own
        job rows more expensive than the work they describe.
        """
        statuses = self.jobs()
        keep: set[str] = {str(job["job_id"]) for job in statuses if job.get("state") in ACTIVE_STATES}
        per_match: dict[str, int] = {}
        kept_total = 0
        for status in statuses:
            if status.get("state") in ACTIVE_STATES:
                continue
            match = str(status.get("match_id"))
            if per_match.get(match, 0) >= KEEP_PER_MATCH or kept_total >= KEEP_TOTAL:
                continue
            per_match[match] = per_match.get(match, 0) + 1
            kept_total += 1
            keep.add(str(status["job_id"]))
        removed = 0
        if self.root.exists():
            for path in self.root.iterdir():
                if path.is_dir() and path.name not in keep:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
        return removed

    # --- manager lifecycle ----------------------------------------------------------------------------------
    def _manager_version(self) -> float:
        try:
            return max(path.stat().st_mtime for path in (Path(__file__), MANAGER_SCRIPT, RUNNER_SCRIPT))
        except OSError:
            return 0.0

    def ensure_manager(self) -> bool:
        """Start the manager if it is not alive, or if it predates the current job code."""
        self.root.mkdir(parents=True, exist_ok=True)
        state = self.manager_state()
        pid = int(state.get("pid") or 0)
        fresh = time.time() - float(state.get("updated") or 0.0) < MANAGER_STALE_S
        current = float(state.get("version") or 0.0) >= self._manager_version() - 1e-6
        if pid_alive(pid) and fresh and current:
            return True
        if pid_alive(pid) and (not current or not fresh):
            # The manager reloads none of its Python after a deploy, and a manager that stopped heartbeating is not
            # going to pick up the new job; neither should be left holding the lock while a fresh one tries to start.
            _terminate(pid)
        if not MANAGER_SCRIPT.exists():
            return False
        try:
            subprocess.Popen(
                [sys.executable, str(MANAGER_SCRIPT), "--root", str(self.root)],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False
        return True


def _slug(text: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in str(text)).strip("-")[:24] or "job"


# The dashboard uses this shared store; tests construct their own with a temporary root.
JOBS = JobStore()

"""Run the dashboard's queued jobs, two at a time (or as configured), outside Streamlit.

The dashboard submits a job by writing its spec; this process is the only thing that starts workers. It is
started on demand by :func:`soccer_analytics.dashboard.jobs.JobStore.ensure_manager` and exits after a quiet
period, so a deploy that changes the runner is picked up by the next submission.

One manager per jobs root: an exclusive ``flock`` on ``manager.lock`` makes a second copy exit immediately. The
manager itself does almost nothing - a one-second loop over a handful of small JSON files - and it lowers each
worker's CPU priority, so the Streamlit process (and the page) keeps the machine's attention.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.dashboard import jobs  # noqa: E402

POLL_S = 1.0
HEARTBEAT_S = 2.0
IDLE_EXIT_S = 300.0
DEFAULT_MAX_JOBS = 2


def _lower_priority() -> None:
    """Drop the worker's CPU priority; the dashboard stays responsive even when a job is CPU-bound."""
    try:
        os.nice(int(os.environ.get("SOCCER_JOB_NICE", "10")))
    except (OSError, ValueError):
        pass


class Manager:
    def __init__(self, store: jobs.JobStore) -> None:
        self.store = store
        self.stop = False
        self.children: dict[str, subprocess.Popen] = {}
        self.started = time.time()
        self.last_activity = time.time()
        self.last_heartbeat = 0.0
        self.max_jobs = max(1, int(os.environ.get("SOCCER_MAX_JOBS", DEFAULT_MAX_JOBS)))

    def _signal(self, _signum, _frame) -> None:  # noqa: ANN001
        self.stop = True

    def _write_state(self, *, state: str = "running") -> None:
        running = [job.get("job_id") for job in self.store.active() if job.get("state") == "running"]
        queued = [job.get("job_id") for job in self.store.active() if job.get("state") == "queued"]
        jobs._atomic_write(
            self.store.root / "manager.json",
            {
                "pid": os.getpid(),
                "state": state,
                "started": self.started,
                "updated": time.time(),
                "version": self.store._manager_version(),
                "running": running,
                "queued": queued,
                "max_jobs": self.max_jobs,
            },
        )

    def _busy_resources(self) -> set[str]:
        busy: set[str] = set()
        for job in self.store.active():
            if job.get("state") != "running":
                continue
            busy.update(str(item) for item in job.get("resources") or [])
        return busy

    def _reap(self) -> None:
        for job_id, child in list(self.children.items()):
            code = child.poll()
            if code is None:
                continue
            self.children.pop(job_id, None)
            self.last_activity = time.time()
            status = self.store.status(job_id)
            if status.get("state") in jobs.ACTIVE_STATES:
                # The runner died before it could write its own terminal state. Say so rather than leaving a
                # "running" row that will never move.
                status.update(
                    state="error",
                    stage="error",
                    message=f"The job process exited with code {code} without reporting a result.",
                    error=f"runner exit code {code}",
                    finished=time.time(),
                    updated=time.time(),
                )
                jobs._atomic_write(self.store.status_path(job_id), status)

    def _adopt_running(self) -> None:
        """Mark running statuses whose process vanished (a manager restart is not a job restart)."""
        for status in self.store.active():
            if status.get("state") != "running":
                continue
            pid = int(status.get("pid") or 0)
            if pid and jobs.pid_alive(pid):
                continue
            if pid and pid in {child.pid for child in self.children.values()}:
                continue
            status.update(
                state="stale",
                stage="stale",
                message="The job process stopped without reporting; start the job again.",
                error="runner process is no longer alive",
                finished=time.time(),
                updated=time.time(),
            )
            jobs._atomic_write(self.store.status_path(str(status["job_id"])), status)

    def _start_queued(self) -> None:
        running = sum(1 for job in self.store.active() if job.get("state") == "running")
        if running >= self.max_jobs:
            return
        busy = self._busy_resources()
        queued = [job for job in self.store.active() if job.get("state") == "queued"]
        queued.sort(key=lambda item: float(item.get("created") or 0.0))
        for status in queued:
            if running >= self.max_jobs:
                break
            resources = {str(item) for item in status.get("resources") or []}
            if resources & busy:
                continue
            job_id = str(status["job_id"])
            if job_id in self.children:
                continue
            spec = jobs.read_json(self.store.spec_path(job_id))
            if not spec:
                continue
            if status.get("cancel_requested"):
                status.update(
                    state="cancelled",
                    stage="cancelled",
                    message="Cancelled before it started.",
                    finished=time.time(),
                    updated=time.time(),
                )
                jobs._atomic_write(self.store.status_path(job_id), status)
                continue
            try:
                child = subprocess.Popen(
                    [sys.executable, str(jobs.RUNNER_SCRIPT), "--root", str(self.store.root), "--job", job_id],
                    cwd=str(jobs.REPO_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=_lower_priority,
                )
            except OSError as exc:
                status.update(
                    state="error",
                    stage="error",
                    message=f"Could not start the job runner: {exc}",
                    error=str(exc),
                    finished=time.time(),
                    updated=time.time(),
                )
                jobs._atomic_write(self.store.status_path(job_id), status)
                continue
            self.children[job_id] = child
            running += 1
            busy.update(resources)
            self.last_activity = time.time()

    def run(self) -> int:
        self.store.root.mkdir(parents=True, exist_ok=True)
        lock = open(self.store.root / "manager.lock", "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.store.prune()
        self._write_state()
        try:
            while not self.stop:
                self._reap()
                self._adopt_running()
                self._start_queued()
                now = time.time()
                if now - self.last_heartbeat >= HEARTBEAT_S:
                    self._write_state()
                    self.last_heartbeat = now
                if not self.children and not self.store.active() and now - self.last_activity > IDLE_EXIT_S:
                    break
                time.sleep(POLL_S)
        finally:
            self._write_state(state="stopped")
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="jobs root (default: SOCCER_JOBS_ROOT or data/jobs)")
    args = parser.parse_args()
    return Manager(jobs.JobStore(args.root)).run()


if __name__ == "__main__":
    raise SystemExit(main())

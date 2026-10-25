"""Execute one queued dashboard job, in its own process, writing progress for the page.

The manager starts this script; it is deliberately small: read the immutable spec, claim the status, dispatch, and
leave a terminal state even when the work raises. Every child process writes into the job's log, and the job's
status mirrors a child's own status file so the existing scan/export bars keep working unchanged.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import game as game_lib  # noqa: E402
from soccer_analytics.analysis import identity as identity_lib  # noqa: E402
from soccer_analytics.analysis import highlights as highlights_lib  # noqa: E402
from soccer_analytics.analysis.library import MatchLibrary  # noqa: E402
from soccer_analytics.dashboard import jobs  # noqa: E402

POLL_S = 1.0


class Status:
    """The job's status file, written atomically and throttled so a polling page never sees half a document."""

    def __init__(self, path: Path, initial: dict) -> None:
        self.path = Path(path)
        self.payload = dict(initial)
        self._written = 0.0

    def update(self, *, force: bool = False, **changes) -> None:
        self.payload.update(changes)
        self.payload["updated"] = time.time()
        now = time.monotonic()
        if not force and now - self._written < 0.5:
            return
        self._written = now
        jobs._atomic_write(self.path, self.payload)


class Runner:
    def __init__(self, store: jobs.JobStore, job_id: str) -> None:
        self.store = store
        self.job_id = job_id
        self.spec = jobs.read_json(store.spec_path(job_id))
        self.status = Status(store.status_path(job_id), store.status(job_id))
        self.job_dir = store.job_dir(job_id)
        self.log = open(store.log_path(job_id), "a", buffering=1)
        self.child: subprocess.Popen | None = None
        self.cancelled = False

    # --- lifecycle ------------------------------------------------------------------------------------------
    def _signal(self, _signum, _frame) -> None:  # noqa: ANN001
        self.cancelled = True
        self._stop_child()

    def _stop_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            # The child is a process-group leader (start_new_session=True); signal the group so a scan's own
            # helper processes stop with it.
            os.killpg(child.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                child.terminate()
            except OSError:
                return
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    child.kill()
                except OSError:
                    pass

    def _write_watch(self, *, state: str, message: str, progress: float = 0.0) -> None:
        watch = self.spec.get("watch")
        if not watch:
            return
        try:
            jobs._atomic_write(
                Path(str(watch)),
                {"state": state, "stage": "starting", "progress": float(progress), "message": message,
                 "updated": time.time()},
            )
        except OSError:
            pass

    def _progress(self, stage: str, fraction: float, message: str) -> None:
        self.status.update(
            state="running",
            stage=stage,
            progress=max(0.0, min(1.0, float(fraction))),
            message=str(message),
        )

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.status.update(
            state="running",
            stage="starting",
            progress=0.0,
            message="Starting...",
            pid=os.getpid(),
            started=time.time(),
            force=True,
        )
        with redirect_stdout(self.log), redirect_stderr(self.log):
            try:
                kind = str(self.spec.get("kind") or "command")
                if kind == "command":
                    code = self._run_command()
                elif kind == "report":
                    code = self._run_report()
                elif kind == "build_and_detect":
                    code = self._run_report_then_detections()
                elif kind == "reel":
                    code = self._run_reel()
                elif kind == "player_clip":
                    code = self._run_player_clip()
                else:
                    raise ValueError(f"unknown job kind {kind!r}")
            except Exception as exc:  # noqa: BLE001 - the status file is the user-facing report
                message = f"{type(exc).__name__}: {exc}"
                print(traceback.format_exc(), file=sys.stderr)
                self.status.update(
                    state="error", stage="error", message=message, error=traceback.format_exc(limit=5),
                    finished=time.time(), force=True,
                )
                return 1
        if code != 0 and self.status.payload.get("state") == "running":
            self.status.update(
                state="error", stage="error",
                message=f"The job process exited with code {code}.",
                finished=time.time(), force=True,
            )
        return int(code)

    # --- command jobs ---------------------------------------------------------------------------------------
    def _run_command(self, *, base: float = 0.0, span: float = 1.0) -> int:
        """Run a child command, mirroring its watch file; the scans and exporters already write one."""
        argv = [str(part) for part in self.spec.get("argv") or []]
        if not argv:
            raise ValueError("command job has no argv")
        self._write_watch(state="running", message="Starting...")
        self.child = subprocess.Popen(
            argv,
            cwd=str(jobs.REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )
        self.status.update(
            state="running", stage="child", progress=base,
            message="Running...", pid=os.getpid(), child_pid=self.child.pid, force=True,
        )
        watch = self.spec.get("watch")
        while self.child.poll() is None:
            if self.cancelled:
                self._stop_child()
                self.status.update(
                    state="cancelled", stage="cancelled", message="Cancelled.",
                    finished=time.time(), force=True,
                )
                return 1
            if watch:
                inner = jobs.read_json(Path(str(watch)))
                if inner:
                    fraction = inner.get("progress")
                    if isinstance(fraction, (int, float)):
                        progress = base + span * max(0.0, min(1.0, float(fraction)))
                    else:
                        progress = float(self.status.payload.get("progress") or base)
                    self.status.update(
                        state="running",
                        stage=str(inner.get("stage") or self.status.payload.get("stage") or "child"),
                        progress=progress,
                        message=str(inner.get("message") or "Running..."),
                    )
            time.sleep(POLL_S)
        code = int(self.child.returncode or 0)
        inner = jobs.read_json(Path(str(watch))) if watch else {}
        if self.cancelled:
            self.status.update(
                state="cancelled", stage="cancelled", message="Cancelled.",
                finished=time.time(), force=True,
            )
            return 1
        if code != 0 or inner.get("state") == "error":
            message = str(inner.get("error") or inner.get("message") or f"Command exited with code {code}.")
            self.status.update(
                state="error", stage="error", message=message, error=message,
                progress=float(self.status.payload.get("progress") or base),
                finished=time.time(), force=True,
            )
            return 1
        message = str(inner.get("message") or "Finished.")
        self.status.update(
            state="done", stage="done", progress=1.0, message=message,
            result={"watch": watch, "state": inner.get("state")} if watch else {},
            finished=time.time(), force=True,
        )
        return 0

    # --- report and rebuild ---------------------------------------------------------------------------------
    def _run_report(self) -> int:
        from soccer_analytics.dashboard.rebuild import build_match_artifacts

        params = self.spec.get("params") or {}
        library = MatchLibrary(params.get("library_root") or None)
        result = build_match_artifacts(
            library,
            str(params.get("match_id") or self.spec.get("match_id")),
            segment_dir=params.get("segment_dir") or None,
            on_progress=self._progress,
        )
        self.status.update(
            state="done", stage="done", progress=1.0,
            message=(
                f"Report ready: {result.get('players', 0)} tracks, {result.get('teams', 0)} teams, "
                f"replay rebuilt."
            ),
            result=result,
            finished=time.time(), force=True,
        )
        return 0

    def _run_report_then_detections(self) -> int:
        params = self.spec.get("params") or {}
        library = MatchLibrary(params.get("library_root") or None)
        match_id = str(params.get("match_id") or self.spec.get("match_id"))
        from soccer_analytics.dashboard.rebuild import build_match_artifacts

        def report_progress(stage: str, fraction: float, message: str) -> None:
            self._progress(stage, 0.50 * float(fraction), message)

        build_match_artifacts(
            library,
            match_id,
            segment_dir=params.get("segment_dir") or None,
            on_progress=report_progress,
        )
        self._progress("detections", 0.50, "Report built; starting the detection scans...")
        command = [
            sys.executable,
            str(jobs.REPO_ROOT / "scripts" / "run_detections.py"),
            "--match", match_id,
        ]
        if params.get("video"):
            command += ["--video", str(params["video"])]
        if params.get("segment_dir"):
            command += ["--segment", str(params["segment_dir"])]
        command += ["--strictness", f"{float(params.get('strictness') or 0.0):.1f}"]
        if params.get("keep_voices"):
            command.append("--keep-voices")
        self.spec["argv"] = command
        self.spec["watch"] = str(library.path(match_id) / "detections.json")
        return self._run_command(base=0.50, span=0.50)

    # --- highlight jobs -------------------------------------------------------------------------------------
    def _run_reel(self) -> int:
        params = self.spec.get("params") or {}
        library = MatchLibrary(params.get("library_root") or None)
        match_id = str(params.get("match_id") or self.spec.get("match_id"))
        video = str(params.get("video") or "")
        tier = str(params.get("tier") or "match")
        events = library.events(match_id)
        report = library.load_report(match_id) or {}
        moments = highlights_lib.build_moments(events.events, report.get("momentum") or {})
        duration = 0.0
        try:
            from soccer_analytics.ingest.source import probe_source

            duration = float(probe_source(video).duration_s)
        except Exception:  # noqa: BLE001 - select_reel accepts a missing duration
            duration = 0.0
        reel = highlights_lib.select_reel(tier, moments, match_duration_s=duration or None)
        out_path = library.highlights_dir(match_id) / f"{tier}.mp4"
        self._progress("reel", 0.02, f"Cutting the {tier} reel ({len(reel.moments)} moments)...")
        highlights_lib.export_reel(
            video,
            reel,
            out_path,
            use_gpu=True,
            progress=lambda fraction: self._progress("reel", fraction, f"Cutting the {tier} reel..."),
            clip_offsets=_clip_offsets(video),
        )
        highlights_lib.write_manifest(reel, video, out_path.with_suffix(".json"))
        self.status.update(
            state="done", stage="done", progress=1.0,
            message=f"{tier} reel ready.",
            result={"output": str(out_path), "moments": len(reel.moments)},
            finished=time.time(), force=True,
        )
        return 0

    def _run_player_clip(self) -> int:
        params = self.spec.get("params") or {}
        library = MatchLibrary(params.get("library_root") or None)
        match_id = str(params.get("match_id") or self.spec.get("match_id"))
        replay = library.load_replay(match_id)
        if replay is None:
            raise ValueError("this match has no replay yet - build the report first")
        appearances = identity_lib.appearances_from_players(
            replay.get("players") or [],
            frames_per_second=float(replay.get("fps") or 5.0),
            start_s=float(params.get("start_s") or 0.0),
            boxes=library.load_replay_boxes(match_id),
        )
        track_id = int(params.get("track_id") or -1)
        first_t = float(params.get("first_t") or 0.0)
        chosen = next(
            (
                appearance
                for appearance in appearances.values()
                if appearance.track_id == track_id and abs(appearance.first_t - first_t) < 0.05
            ),
            None,
        )
        if chosen is None:
            raise ValueError(f"track {track_id} at {first_t:.1f}s is not in the current replay")
        source = str(params.get("source") or "")
        target = Path(str(params.get("target") or ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        self._progress("clip", 0.02, "Cutting the clip...")
        from soccer_analytics.ingest.source import probe_source

        probe = probe_source(source)
        _path, plan = highlights_lib.export_player_clip(
            source,
            *chosen.trajectory(start_s=chosen.first_t),
            target,
            start_s=chosen.first_t,
            duration_s=float(params.get("duration_s") or chosen.span_s),
            source_size=(probe.width, probe.height),
            width=int(params.get("width") or 1280),
            use_gpu=True,
            progress=lambda fraction: self._progress("clip", fraction, "Cutting the clip..."),
        )
        plan_payload = {"track": track_id, "duration": plan.duration_s, "truncated": plan.truncated}
        try:
            jobs._atomic_write(target.with_suffix(target.suffix + ".plan.json"), plan_payload)
        except OSError:
            pass
        self.status.update(
            state="done", stage="done", progress=1.0,
            message="Clip ready.",
            result={"output": str(target), **plan_payload},
            finished=time.time(), force=True,
        )
        return 0


def _clip_offsets(video: str) -> dict[str, float]:
    """Where each of the game's clips starts inside ``video`` (same rule the dashboard uses)."""
    record = game_lib.find_for_video(video)
    if record is None or len(record.clips) < 2:
        return {}
    return {
        str(Path(clip.path).resolve()): float(clip.start_s)
        for clip in record.clips
        if Path(clip.path).exists()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="jobs root")
    parser.add_argument("--job", required=True, help="job id")
    args = parser.parse_args()
    store = jobs.JobStore(args.root)
    if not store.spec_path(args.job).exists():
        return 1
    return Runner(store, args.job).run()


if __name__ == "__main__":
    raise SystemExit(main())

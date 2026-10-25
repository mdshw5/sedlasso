"""The file-based job queue: submission, terminal states, pruning, and one real worker round trip."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.dashboard import jobs
from soccer_analytics.dashboard.rebuild import build_match_artifacts


def _store(tmp_path: Path) -> jobs.JobStore:
    store = jobs.JobStore(tmp_path / "jobs")
    # Unit tests do not need a daemon; the one integration test below starts it explicitly.
    store.ensure_manager = lambda: True  # type: ignore[method-assign]
    return store


def test_submit_writes_spec_status_and_params(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status = store.submit(
        kind="report",
        label="Report + replay",
        match_id="match-a",
        params={"match_id": "match-a", "segment_dir": "/tmp/seg"},
    )
    assert status["state"] == "queued"
    assert status["params"]["segment_dir"] == "/tmp/seg"
    spec = json.loads(store.spec_path(status["job_id"]).read_text())
    assert spec["kind"] == "report"
    assert spec["label"] == "Report + replay"
    assert store.status(status["job_id"])["job_id"] == status["job_id"]


def test_active_and_has_active_filter_by_match_and_kind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.submit(kind="ball_scan", label="Ball", match_id="a")
    store.submit(kind="report", label="Report", match_id="b")
    assert {job["job_id"] for job in store.active(match_id="a")} == {first["job_id"]}
    assert store.has_active("a", kinds=("ball_scan",))
    assert not store.has_active("a", kinds=("report",))
    assert store.has_active("b")


def test_cancel_queued_job_is_terminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status = store.submit(kind="report", label="Report", match_id="a")
    cancelled = store.cancel(status["job_id"])
    assert cancelled["state"] == "cancelled"
    assert store.status(status["job_id"])["state"] == "cancelled"


def test_retry_clones_a_failed_job_spec(tmp_path: Path) -> None:
    store = _store(tmp_path)
    status = store.submit(
        kind="video_export",
        label="Video export",
        match_id="a",
        argv=["ffmpeg", "-i", "in.mp4"],
        watch=tmp_path / "watch.json",
        params={"output": "/tmp/out.mp4"},
        resources=("gpu",),
    )
    payload = store.status(status["job_id"])
    payload.update(state="error", message="boom")
    store.status_path(status["job_id"]).write_text(json.dumps(payload))

    retried = store.retry(status["job_id"])
    assert retried["job_id"] != status["job_id"]
    spec = json.loads(store.spec_path(retried["job_id"]).read_text())
    assert spec["kind"] == "video_export"
    assert spec["argv"] == ["ffmpeg", "-i", "in.mp4"]
    assert spec["params"] == {"output": "/tmp/out.mp4"}
    assert spec["resources"] == ["gpu"]


def test_clear_finished_leaves_active_jobs_alone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    done = store.submit(kind="report", label="Report", match_id="a")
    running = store.submit(kind="report", label="Report", match_id="a")
    path = store.status_path(done["job_id"])
    payload = store.status(done["job_id"])
    payload.update(state="done", message="ok")
    path.write_text(json.dumps(payload))
    running_payload = store.status(running["job_id"])
    running_payload["state"] = "running"
    store.status_path(running["job_id"]).write_text(json.dumps(running_payload))

    assert store.clear_finished(match_id="a") == 1
    assert not store.job_dir(done["job_id"]).exists()
    assert store.job_dir(running["job_id"]).exists()


def test_prune_keeps_active_and_newest_finished_jobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    active = store.submit(kind="report", label="Active", match_id="a")
    old = store.submit(kind="report", label="Old", match_id="a")
    old_status = store.status(old["job_id"])
    old_status.update(state="done", created=1.0)
    store.status_path(old["job_id"]).write_text(json.dumps(old_status))
    for index in range(jobs.KEEP_PER_MATCH + 3):
        status = store.submit(kind="report", label=f"job {index}", match_id="a")
        payload = store.status(status["job_id"])
        payload.update(state="done", created=float(index + 2))
        store.status_path(status["job_id"]).write_text(json.dumps(payload))

    store.prune()
    assert store.job_dir(active["job_id"]).exists(), "an active job must never be pruned"
    assert not store.job_dir(old["job_id"]).exists(), "the oldest finished job should go"
    assert len(store.jobs(match_id="a")) <= jobs.KEEP_PER_MATCH + 1


def test_report_pipeline_refuses_a_match_without_calibration(tmp_path: Path) -> None:
    library = MatchLibrary(tmp_path / "matches")
    record = library.create(str(tmp_path / "match.mp4"), format="9v9")
    with pytest.raises(ValueError, match="calibration"):
        build_match_artifacts(library, record.match_id)


def test_manager_runs_a_real_command_job(tmp_path: Path) -> None:
    """One end-to-end pass through the manager, runner and a child process."""
    store = jobs.JobStore(tmp_path / "jobs")
    watch = tmp_path / "watch.json"
    script = (
        "import json, time\n"
        "from pathlib import Path\n"
        f"p = Path(r'{watch}')\n"
        "p.write_text(json.dumps({'state': 'done', 'progress': 1.0, 'message': 'child ok'}))\n"
        "time.sleep(0.2)\n"
    )
    status = store.submit(
        kind="video_export",
        label="Smoke",
        match_id="demo",
        argv=[sys.executable, "-c", script],
        watch=watch,
    )
    try:
        deadline = time.time() + 25
        current = status
        while time.time() < deadline:
            current = store.status(status["job_id"])
            if current.get("state") not in jobs.ACTIVE_STATES:
                break
            time.sleep(0.4)
        assert current["state"] == "done", current
        assert json.loads(watch.read_text())["message"] == "child ok"
    finally:
        manager = store.manager_state()
        if jobs.pid_alive(int(manager.get("pid") or 0)):
            jobs._terminate(int(manager["pid"]))

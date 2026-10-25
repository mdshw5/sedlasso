"""The small Streamlit surface for the dashboard's job manager.

Two pieces: a global badge, so a job started on another match is still visible while working here, and a per-match
panel with progress, cancellation and the last few results. Both poll a handful of JSON files in a fragment; they
never run the work themselves.
"""

from __future__ import annotations

import streamlit as st

from soccer_analytics.dashboard import jobs

POLL_SECONDS = 4.0
ACTIVE = jobs.ACTIVE_STATES
TERMINAL = jobs.TERMINAL_STATES

_KIND_LABELS = {
    "report": "Report + replay",
    "build_and_detect": "Report + all detections",
    "detections": "All detections",
    "ball_scan": "Ball scan",
    "audio_scan": "Whistle scan",
    "jersey_scan": "Shirt numbers",
    "video_export": "Video export",
    "reel": "Highlight reel",
    "player_clip": "Player clip",
    "stage_a": "Stage A analysis",
}


def _label(status: dict) -> str:
    return str(status.get("label") or _KIND_LABELS.get(str(status.get("kind")), status.get("kind") or "job"))


def _percent(status: dict) -> int:
    try:
        return int(round(100.0 * max(0.0, min(1.0, float(status.get("progress") or 0.0)))))
    except (TypeError, ValueError):
        return 0


def _state_word(status: dict) -> str:
    state = str(status.get("state") or "")
    return {
        "queued": "queued",
        "running": "running",
        "done": "done",
        "error": "failed",
        "cancelled": "cancelled",
        "stale": "stopped",
    }.get(state, state or "unknown")


@st.fragment(run_every=POLL_SECONDS)
def running_jobs_badge() -> None:
    """A one-line view of work happening anywhere, so switching matches never hides a running job."""
    active = jobs.JOBS.jobs(active_only=True, limit=20)
    if not active:
        return
    running = sum(1 for job in active if job.get("state") == "running")
    queued = len(active) - running
    parts = []
    if running:
        parts.append(f"{running} running")
    if queued:
        parts.append(f"{queued} queued")
    st.info(
        f"Heavy jobs are running in the background ({', '.join(parts)}). "
        "You can keep working on another match."
    )
    with st.expander("Background jobs", expanded=False):
        for job in active:
            st.caption(
                f"**{_label(job)}** - {job.get('match_id')} - {_state_word(job)}"
                + (f" - {_percent(job)}%" if job.get("state") == "running" else "")
                + (f" - {job.get('message')}" if job.get("message") else "")
            )


@st.fragment(run_every=POLL_SECONDS)
def match_jobs_panel(match_id: str) -> None:
    """Jobs for the selected match: live rows, cancel buttons, and the last few outcomes."""
    store = jobs.JOBS
    all_jobs = store.jobs(match_id=match_id, limit=12)
    if not all_jobs:
        return
    active = [job for job in all_jobs if job.get("state") in ACTIVE]
    recent = [job for job in all_jobs if job.get("state") in TERMINAL][:4]

    # One full rerun when anything lands: a report changes the Step 3 table and replay, a reel or clip makes a new
    # file appear, and a scan changes its row. The states are kept in session memory so the rerun is edge-triggered.
    watch_key = f"job_states::{match_id}"
    previous = st.session_state.get(watch_key) or {}
    current = {str(job.get("job_id")): str(job.get("state")) for job in all_jobs}
    finished = any(
        previous.get(job_id) in ACTIVE and state in TERMINAL for job_id, state in current.items()
    )
    st.session_state[watch_key] = current

    st.subheader("Background jobs")
    st.caption(
        "Heavy work runs in its own process, so this page stays usable and you can open another match while it "
        "runs. A job started here can also be watched from the bar at the top of the page."
    )
    if active and not store.manager_alive():
        st.warning("The job manager is not reporting. Start a job and it will be launched again.")
    for job in active:
        job_id = str(job.get("job_id"))
        with st.container(border=True):
            st.markdown(
                f"**{_label(job)}** - {_state_word(job)}"
                + (f" - {_percent(job)}%" if job.get("state") == "running" else "")
            )
            if job.get("message"):
                st.caption(str(job["message"]))
            progress = float(job.get("progress") or 0.0)
            st.progress(max(0.0, min(1.0, progress)), text=str(job.get("message") or ""))
            if job.get("state") == "running":
                st.caption("Cancel stops the worker and its child process. Finished work is kept.")
            st.button(
                "Cancel",
                key=f"job_cancel::{job_id}",
                on_click=store.cancel,
                args=(job_id,),
            )
    if recent:
        with st.expander("Recent jobs", expanded=False):
            for job in recent:
                state = _state_word(job)
                st.caption(
                    f"**{_label(job)}** - {state}"
                    + (f" - {job.get('message')}" if job.get("message") else "")
                )
                if job.get("state") == "error" and job.get("error"):
                    st.code(str(job["error"]).strip(), language="text")
                log = job.get("log")
                if log:
                    st.caption(f"Log: `{log}`")
        if st.button("Clear finished jobs", key=f"job_clear::{match_id}"):
            store.clear_finished(match_id=match_id)
            st.rerun(scope="fragment")
    if finished:
        # The report payload may have been cached in the session by the old synchronous path; drop it so Step 3
        # reads the freshly saved files.
        st.session_state.pop("report", None)
        st.rerun()

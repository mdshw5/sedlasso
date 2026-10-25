"""The small Streamlit surface for the dashboard's job manager.

Three pieces: a persistent global jobs expander (so a job started somewhere is always findable), a full per-match
panel for Step 3, and a compact version that can sit beside the export, reel and clip controls where the work is
started. They poll a handful of JSON files in fragments; they never run the work themselves.
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


def _summary(status: dict, *, with_match: bool = False) -> str:
    parts = [f"**{_label(status)}**"]
    if with_match:
        parts.append(str(status.get("match_id") or ""))
    parts.append(_state_word(status))
    if status.get("state") == "running":
        parts.append(f"{_percent(status)}%")
    if status.get("message"):
        parts.append(str(status["message"]))
    return " - ".join(parts)


def _render_active(store: jobs.JobStore, active: list[dict], *, key_prefix: str, cancel: bool) -> None:
    if not active:
        st.caption("No job is running right now.")
        return
    if not store.manager_alive():
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
            if cancel:
                if job.get("state") == "running":
                    st.caption("Cancel stops the worker and its child process. Finished work is kept.")
                st.button(
                    "Cancel",
                    key=f"job_cancel::{key_prefix}::{job_id}",
                    on_click=store.cancel,
                    args=(job_id,),
                )


def _render_recent(store: jobs.JobStore, recent: list[dict], *, key_prefix: str, match_id: str | None) -> None:
    if not recent:
        return
    with st.expander("Recent jobs", expanded=False):
        for job in recent:
            st.caption(_summary(job, with_match=True))
            if job.get("state") == "error" and job.get("error"):
                st.code(str(job["error"]).strip(), language="text")
            if job.get("log"):
                st.caption(f"Log: `{job['log']}`")
            if job.get("state") in ("error", "cancelled", "stale"):
                st.button(
                    "Retry",
                    key=f"job_retry::{key_prefix}::{job.get('job_id')}",
                    on_click=store.retry,
                    args=(str(job.get("job_id")),),
                )
    if match_id is not None and st.button("Clear finished jobs", key=f"job_clear::{key_prefix}::{match_id}"):
        store.clear_finished(match_id=match_id)
        st.rerun(scope="fragment")


@st.fragment(run_every=POLL_SECONDS)
def running_jobs_badge() -> None:
    """A persistent, collapsed entry point to all jobs, so none is ever hidden behind a scroll position."""
    store = jobs.JOBS
    all_jobs = store.jobs(limit=10)
    active = [job for job in all_jobs if job.get("state") in ACTIVE]
    recent = [job for job in all_jobs if job.get("state") in TERMINAL][:4]
    label = f"Background jobs ({len(active)} running)" if active else "Background jobs"
    with st.expander(label, expanded=bool(active)):
        st.caption(
            "Report builds, scans, exports, reels and clips run in their own processes. "
            "Open a match's Step 3 for the full panel and cancellation."
        )
        if active:
            for job in active:
                st.caption(_summary(job, with_match=True))
        if recent:
            st.caption("Recent:")
            for job in recent:
                st.caption(_summary(job, with_match=True))
        if not active and not recent:
            st.caption("No background job has been started yet.")


@st.fragment(run_every=POLL_SECONDS)
def match_jobs_panel(match_id: str, *, compact: bool = False, key_prefix: str = "jobs") -> None:
    """Jobs for the selected match: live rows, cancel buttons, and the last few outcomes.

    ``compact`` drops the section heading for use beside a control (the export step, the reels), where a second
    heading would compete with the control it belongs to. ``key_prefix`` keeps widget keys unique when the full
    panel and a compact panel are both on one page.
    """
    store = jobs.JOBS
    all_jobs = store.jobs(match_id=match_id, limit=12)
    active = [job for job in all_jobs if job.get("state") in ACTIVE]
    recent = [job for job in all_jobs if job.get("state") in TERMINAL][:4]

    # One full rerun when anything lands: a report changes the Step 3 table and replay, a reel or clip makes a new
    # file appear, and a scan changes its row. The states are kept in session memory so the rerun is edge-triggered.
    watch_key = f"job_states::{key_prefix}::{match_id}"
    previous = st.session_state.get(watch_key) or {}
    current = {str(job.get("job_id")): str(job.get("state")) for job in all_jobs}
    finished = any(
        previous.get(job_id) in ACTIVE and state in TERMINAL for job_id, state in current.items()
    )
    st.session_state[watch_key] = current

    if not compact:
        st.subheader("Background jobs")
        st.caption(
            "Heavy work runs in its own process, so this page stays usable and you can open another match while it "
            "runs. The same list is available from the Background jobs bar at the top of every page."
        )
        _render_active(store, active, key_prefix=key_prefix, cancel=True)
        _render_recent(store, recent, key_prefix=key_prefix, match_id=match_id)
    else:
        label = f"Background jobs ({len(active)} running)" if active else "Background jobs"
        with st.expander(label, expanded=bool(active)):
            if not all_jobs:
                st.caption("No background job for this match yet.")
            else:
                _render_active(store, active, key_prefix=key_prefix, cancel=True)
                _render_recent(store, recent, key_prefix=key_prefix, match_id=match_id)
    if finished:
        # The report payload may have been cached in the session by the old synchronous path; drop it so Step 3
        # reads the freshly saved files.
        st.session_state.pop("report", None)
        st.rerun()

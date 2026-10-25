"""Dashboard for the gimbal-camera (single following camera) match analysis.

Workflow, in the order it appears on screen:

1. **Choose footage** - pick or browse to a video, optionally start part-way in, and run the heavy per-segment pass
   (Stage A) in the background with a progress bar and resume.
2. **Register the pitch** - click landmarks in a reference frame; the camera's motion is already known, so this
   calibrates position and orientation and reports how well it fits, per landmark.
3. **Build the report** - cheap and re-runnable: pitch-space tracking, teams, distances, momentum, and the
   separately-scanned ball track when one has run. One press of "Build report + run all detections" also starts
   the ball, whistle and shirt-number scans in the background, infers the detectable events and rebuilds the
   replay with everything the scans found.
4. **Watch, tag and cut highlights** - the playback's own tag buttons stamp an event on the current second, and
   the review queue, the detectors' settings and the highlight reels sit directly under the pitch, so tagging
   happens while the game is on screen rather than in a section of its own further down the page.

Every number on the page is either measured or explicitly labeled as a guess or a manual tag. Where a metric is
beyond what the footage (and the scans built on it) can honestly support, the page says so rather than inventing one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit import runtime as st_runtime

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis import identity as identity_lib
from soccer_analytics.analysis import stage_a
from soccer_analytics.analysis import detect_run
from soccer_analytics.analysis.events import (
    MIN_PROMINENCE,
    VERDICT_FALSE,
    VERDICT_TRUE,
    Event,
)
from soccer_analytics.analysis.framing import COMMAND_RATE_HZ, SHORTEST_CLIP_SECONDS, default_clip_length
from soccer_analytics.analysis.highlights import (
    TIER_SECONDS,
    build_moments,
    moment_for_event,
    select_reel,
)
from soccer_analytics.analysis.library import (
    MatchLibrary,
    analysis_dir_for,
    discover_videos,
    segments_root_for,
    upgrade_report_payload,
    upgrade_replay_payload,
    video_roots,
)
from soccer_analytics.analysis.kit import color_hex, color_name, suggest_team_name
from soccer_analytics.analysis.jerseys import merge_numbers
from soccer_analytics.analysis.projection import fit_segment_pan_axis, segment_poses
from soccer_analytics.analysis.rosters import RosterLibrary, team_rosters_for
from soccer_analytics.analysis.stage_a import SegmentConfig, load_segment, read_status, resolve_window, segment_dir_for
from soccer_analytics.geometry.gimbal_motion import (
    load_pan_model,
    save_pan_model,
    segment_has_log,
    segment_pose_source,
)
from soccer_analytics.geometry.pitch_lines import LineObservation, observations_to_landmarks
from soccer_analytics.geometry.pitch_markings import markings_for, with_goal_width
from soccer_analytics.dashboard.pitch_clicks import (
    actual_center,
    canvas_scale,
    clamp01,
    frame_change,
    frame_points,
    inside_crop,
    landmark_help,
    landmark_table,
    landmarks_from_points,
    marker_feature,
    merge_clicked,
    order_clicks,
    overlay_homographies,
    parse_result,
    pitch_marking_polylines,
    pitch_overlay,
    point_center,
    points_in_crop,
    projected_landmarks,
    repeated_labels_within_a_frame,
    restored_points,
    split_duplicate_clicks,
    zoom_box,
)
from soccer_analytics.dashboard import timeline
from soccer_analytics.dashboard.export import preview_stream_url
from soccer_analytics.dashboard.match_state import match_footage
from soccer_analytics.dashboard.reports import (
    DEFAULT_TEAM_NAMES,
    colors_were_recorded,
    is_default_team_name,
    momentum_on_window_clock,
    team_colors,
    team_name,
)
from soccer_analytics.dashboard import job_ui, jobs
from soccer_analytics.dashboard.replay import (
    aiming_goal,
    deleted_events,
    event_from_tag,
    event_play,
    player_table_rows,
    repeated_manual_tag,
)
from soccer_analytics.dashboard.stream import configured_base, configured_port, is_reachable
from soccer_analytics.geometry.pitch_calibration import (
    MIN_CLICKS_PER_ANCHOR,
    MIN_REFERENCE_CLICKS,
    PitchCalibration,
    calibrate,
    diagnose_fit,
    format_scale_note,
)
from soccer_analytics.ingest.ffmpeg_reader import grab_frame
from soccer_analytics.ingest.source import probe_source

# Everything the tool computes for a match now lives beside the footage, in an ``analysis/<id>`` directory
# (analysis.library builds the paths); the old data/matches root is still listed until migrated.
ANNOTATION_COMPONENT = components.declare_component(
    "field_annotation_editor", path=str(Path(__file__).parent / "field_annotation_component")
)
REPLAY_COMPONENT = components.declare_component(
    "pitch_replay", path=str(Path(__file__).parent / "pitch_replay_component")
)
GAME_TIMELINE_COMPONENT = components.declare_component(
    "game_timeline", path=str(Path(__file__).parent / "game_timeline_component")
)

MATCH_FORMATS: dict[str, tuple[float, float]] = {  # length m, width m
    "5v5": (40.0, 25.0),
    "7v7": (50.0, 35.0),
    "9v9": (60.0, 40.0),
    "11v11": (100.0, 64.0),
}

# Zoom the landmark view starts at. The whole frame is the main viewport, so starting zoomed in is cheap: there is
# always a view that shows the whole pitch, and the magnified inset is immediately accurate enough to click.
DEFAULT_ZOOM = 8.0

# The camera rig is a pole at a fixed height, so the calibration solver pins the height instead of solving it -
# one fewer degree of freedom for the clicks to pin down, which measurably tightens a four-click fit.
CAMERA_HEIGHT_M = 4.0

# How often the fragments that watch background tasks re-read their status files. They matter only while the task
# runs: the running -> finished transition triggers one full rerun, and the fragment stops being drawn.
POLL_SECONDS = 4.0

st.set_page_config(page_title="Match analysis", layout="wide")


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------
def _location_label(path: Path) -> str:
    """A short, readable name for where something lives: under a footage root when possible, else absolute.

    Analyzes live beside the footage now rather than in the repository, so "relative to the repo" is no longer
    the right display rule; the root's own name ("Xbot/2026-10-03/analysis/...") tells the user which disk and
    which match folder holds the directory, which is what they need to find it themselves.
    """
    path = Path(path)
    for root in video_roots():
        try:
            return f"{root.name}/{path.relative_to(root)}"
        except ValueError:
            continue
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _video_label(path: str) -> str:
    """A video's label in the picker: where it is (relative to its root) and whether an analysis exists."""
    relative = _location_label(Path(path))
    return relative + ("  -  analysis saved" if _analysis_for_video(path) else "")


def _analysis_for_video(path: str) -> str | None:
    """The id of the analysis saved beside this video, if there is one (cheap: one stat beside the file)."""
    directory = analysis_dir_for(path)
    return directory.name if (directory / "match.json").exists() else None


def _clock(seconds: float) -> str:
    """Match time as a coach reads it; hours appear only when the footage needs them."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _event_label(event: Event, index: int, match_clock: str | None = None) -> str:
    """One line describing an event, for the preview picker.

    The body of this string must not depend on anything that changes while the page is open - a verdict, say. A
    selectbox's labels are rendered once and handed back by the frontend as the widget's value, so a label that
    later no longer matches the option list (because a verdict changed it) leaves the page holding a string where it
    expects an index. The verdict is shown under the player and in the table instead.

    ``match_clock`` is the time on the game's own clock when the caller can work it out. A candidate's own
    ``time_s`` is a time on the recording it was found in, which is not what a coach reads out - a whistle at 5 s
    of the second camera clip is 30 minutes into the match - so the match time leads when it is known.
    """
    origin = "audio" if event.source == "audio" else (f"team {event.team}" if event.team >= 0 else "manual")
    note = f" - {event.note[:48]}" if event.note else ""
    when = match_clock if match_clock and match_clock != "-" else _clock(event.time_s)
    return f"#{index + 1}  {when}  {event.type} ({origin}){note}"


def _encode(frame: np.ndarray) -> str:
    """Frame images are sent to the component as JPEG.

    Two images go over on every rerun - the crop and the whole frame - and PNG of a photographic frame is megabytes,
    which is the difference between the view following a gesture promptly and feeling stuck. The compression cannot
    move a click: the mapping is in image coordinates either way.
    """
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("could not encode the frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def landmark_clicker(
    crop: np.ndarray,
    overview: np.ndarray | None,
    features: list[dict],
    markers: list[tuple[float, float]],
    center: tuple[float, float],
    zoom: float,
    box: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    key: str,
    *,
    proxy_url: str = "",
    frame_count: int = 0,
    initial_frame: int = 0,
    marker_kinds: list[str] | None = None,
    mount_nonce: int = 0,
    overlay_homographies: list[dict] | None = None,
    overlay_polylines: list[list[list[float]]] | None = None,
) -> None:
    """Draw the landmark clicker over one crop, with the whole frame as its navigation viewport.

    The component owns the whole view: the whole-frame viewport where clicking aims, scrolling zooms and dragging
    pans; the magnified crop where the landmarks are clicked; and - when a proxy is available - the timeline bar
    under them both, which scrubs that same viewport.

    This only draws it. What the user did arrives as the component's widget value, which Streamlit hands to the
    *next* run before it starts - so the caller reads that (``parse_result``) and settles it into session state
    before the crop is computed, and one gesture costs one run rather than two. Nothing here can act on the value:
    by the time it is returned, the frame above has already been read and cropped.
    """
    scale = canvas_scale(crop.shape[1])
    canvas = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame_width, frame_height = frame_size
    x0, y0, _w, _h = box
    placed_x, placed_y = actual_center(box, frame_size)
    result = ANNOTATION_COMPONENT(
        frame_data=_encode(canvas),
        # Only the still-frame fallback needs the whole frame as an image; with a proxy the video is the picture.
        overview_data=_encode(overview) if overview is not None else "",
        proxy_url=proxy_url,
        frame_count=int(frame_count),
        initial_frame=int(initial_frame),
        marker_kinds=list(marker_kinds or []),
        mount_nonce=int(mount_nonce),
        # The whole-frame viewport is scrubbed and played in the browser, so Python cannot draw the pitch overlay
        # onto frames it never sees. Instead it sends sampled pitch->pixel homographies plus the markings in pitch
        # meters, and the component projects the overlay itself for whatever frame is on screen - and re-derives it
        # from the same arguments whenever the calibration changes, so a refit redraws immediately.
        overlay_homographies=list(overlay_homographies or []),
        overlay_polylines=list(overlay_polylines or []),
        initial_polygon=[],
        initial_features=features,
        marker_points=[[float(u), float(v)] for u, v in markers],
        width=int(canvas.shape[1]),
        height=int(canvas.shape[0]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        zoom=float(zoom),
        actual_x=float(placed_x),
        actual_y=float(placed_y),
        box_x0=int(x0),
        box_y0=int(y0),
        scale=float(scale),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        key=key,
        default=None,
    )
    del result  # the gesture is read at the top of the next run, not from this return value


def _served_video_url(path: Path, coordinates: str) -> str | None:
    """Serve a video through Streamlit's own media endpoint so the browser can seek it.

    A component's static route ignores HTTP ``Range`` requests - it answers every request with the whole file and no
    ``Accept-Ranges`` - and a video the browser cannot seek is useless for scrubbing. The media endpoint (the one
    ``st.video`` uses) does range requests properly.

    It has to be registered on *every* run. Streamlit drops a session's media references before each run and deletes
    anything the body did not re-register, so a URL cached across runs ends up pointing at a file that has been
    swept away - the video dies with a 404. The URL is a content hash, so registering again keeps it stable and the
    iframe is not remounted. Returns ``None`` when the endpoint is not available.
    """
    if not st_runtime.exists():
        return None
    try:
        return st_runtime.get_instance().media_file_mgr.add(str(path), "video/mp4", coordinates)
    except Exception:  # a media-endpoint failure must not take the page down
        return None


def _timeline_media_url(proxy: Path, coordinates: str) -> str:
    """The segment's scrub proxy, with the component's static route as a last resort."""
    return _served_video_url(proxy, coordinates) or timeline.PROXY_FILE


def _replay_media_url(path: Path) -> str:
    """Serve the replay JSON through the media endpoint, like the timeline proxy.

    The replay is megabytes of per-frame points. Sending it through the component's *arguments* would serialize it
    onto the Streamlit websocket on every rerun, so the component is handed a URL and fetches it once. The file must
    be re-registered every run for the same reason as the proxy: Streamlit sweeps media a run did not mention.
    """
    if not st_runtime.exists():
        return ""
    try:
        return st_runtime.get_instance().media_file_mgr.add(str(path), "application/json", f"replay::{path.name}")
    except Exception:
        return ""


def replay_view(
    data_url: str,
    numbers: dict[int, dict],
    selected_track: int,
    team_names: list[str],
    key: str,
    events: list[dict] | None = None,
    momentum: dict | None = None,
    half_minute: float | None = None,
    match_id: str = "",
    stream_offset_s: float | None = None,
    stream_port: int | None = None,
    stream_base: str = "",
    seek_to_s: float | None = None,
    seek_seq: int = 0,
    ack_tag_seq: int = 0,
    ack_delete_seq: int = 0,
) -> dict | None:
    """Draw the animated pitch view; playback lives entirely in the browser.

    ``events`` and ``momentum`` are the timeline strip's data: the tagged and detected events (as JSON dicts) and
    the per-minute momentum buckets. Both arrive already on the strip's own (window-relative) clock. They travel
    as component arguments rather than in the replay payload because they are small and change without the replay
    being rebuilt - a new tag shows up on the strip on the next rerun.

    ``match_id``, ``stream_offset_s`` (the recording second of strip second 0) and ``stream_port`` point the pane
    beside the pitch at the annotated match stream, so it can play the footage of any second the animation is
    showing. ``seek_to_s`` plus ``seek_seq`` is the "jump to this moment" request from the picker above; the
    component acts when the sequence changes, so unrelated reruns never re-seek.

    Two deliberate gestures post back (the playback controls never round-trip): the tag buttons under the pitch
    send ``{"action": "tag", "tags": [...]}`` and a right-click delete on the strip sends
    ``{"action": "delete", "deletes": [...]}``. The return value is what they sent, or None. ``ack_tag_seq`` and
    ``ack_delete_seq`` are the newest gesture already stored; the component drops everything up to them, so a
    gesture is neither lost nor applied twice. Returns the value so the caller can store it and redraw the strip.
    """
    return REPLAY_COMPONENT(
        data_url=data_url,
        numbers={str(track): entry for track, entry in numbers.items()},
        selected_track=int(selected_track),
        team_names=list(team_names),
        events=list(events or []),
        momentum=[
            # Minutes travel as floats: the window's start can sit mid-minute, and the strip positions a bucket
            # at its minute's midpoint of the window's own clock.
            {"minute": float(minute), **bucket} for minute, bucket in sorted((momentum or {}).items())
        ],
        half_minute=half_minute,
        match_id=match_id,
        stream_offset_s=stream_offset_s,
        stream_port=stream_port,
        stream_base=stream_base,
        seek_to_s=seek_to_s,
        seek_seq=int(seek_seq),
        ack_tag_seq=int(ack_tag_seq),
        ack_delete_seq=int(ack_delete_seq),
        key=key,
        default=None,
    )


def _apply_quick_tags(
    library: MatchLibrary,
    match_id: str,
    video: str,
    window_start: float,
    replay_duration_s: float,
    tag_result,
) -> None:
    """Store the tags the playback's own buttons sent, then redraw with the new markers on the strip.

    The component's value is sticky: it comes back on every later run until the component sees an acknowledgment,
    so only tags newer than the newest already stored are applied. The acknowledgment is written before the
    rerun, which is what clears the component's pending list and stops a tag being stored twice. A tag is stamped
    with the playback's own second when the button was pressed - not with the moment the round trip finishes - so
    tagging while watching lands on what was on screen.

    A page that reconnects after a server restart re-sends the component's sticky value, and the new session has
    no acknowledgment to compare against - so the store is also idempotent: a press that is already on the
    timeline (same type, team, recording and second) is skipped rather than duplicated.
    """
    if not isinstance(tag_result, dict):
        return
    tags = [tag for tag in (tag_result.get("tags") or []) if isinstance(tag, dict)]
    if not tags:
        return
    ack_key = f"replay_tag_ack::{match_id}"
    newest = int(st.session_state.get(ack_key, 0))
    fresh: list[Event] = []
    for tag in tags:
        newest = max(newest, int(tag.get("seq") or 0))
        if int(tag.get("seq") or 0) <= int(st.session_state.get(ack_key, 0)):
            continue
        try:
            fresh.append(
                event_from_tag(
                    tag, video=str(Path(video).resolve()), window_start=window_start, duration_s=replay_duration_s
                )
            )
        except (TypeError, ValueError):
            continue  # a malformed tag is dropped, but its sequence still advances the acknowledgment
    if newest <= int(st.session_state.get(ack_key, 0)):
        return  # nothing new: the sticky value was already applied on an earlier run
    st.session_state[ack_key] = newest
    added: list[Event] = []
    if fresh:
        log = library.events(match_id)
        for event in fresh:
            if repeated_manual_tag(log.events, event) is not None:
                continue  # the same press delivered twice: it is already on the timeline
            log.add(event)
            added.append(event)
        if added:
            library.save_events(match_id, log)
            last = added[-1]
            st.session_state["events_flash"] = (
                "success",
                f"Tagged {last.type} at {_clock(last.time_s)} - it is on the timeline strip above.",
            )
        else:
            st.session_state["events_flash"] = (
                "success",
                "That tag is already on the timeline - nothing was added.",
            )
    else:
        st.session_state["events_flash"] = (
            "warning",
            "A tag arrived in a shape the page could not read; it was skipped.",
        )
    # Rerun so the strip, the table and the acknowledged tag bar all see the stored event in this same gesture.
    st.rerun()


def _reindex_event_pickers(match_id: str, removed_indices: set[int], remaining: int) -> None:
    """Move the index-based event pickers onto the rows they still mean after a deletion."""
    for key in (f"pitch_clip_event::{match_id}", f"event_remove::{match_id}"):
        current = st.session_state.get(key)
        if not isinstance(current, int):
            continue
        if current in removed_indices:
            st.session_state[key] = min(current, remaining - 1) if remaining else 0
        else:
            st.session_state[key] = current - sum(1 for index in removed_indices if index < current)
    st.session_state[f"event_remove_confirm::{match_id}"] = False


def _keep_event_pickers_valid(match_id: str, log, removed_indices: set[int]) -> None:
    """Repair the pickers from a widget callback, where their keys may still be written."""
    _reindex_event_pickers(match_id, removed_indices, len(log.events))


def _apply_event_picker_repair(match_id: str) -> None:
    """Apply a repair queued by a deletion made mid-run, before the pickers are created.

    A timeline delete is handled while the page is rendering, after ``pitch_clip_event`` has already been
    instantiated - Streamlit refuses a write to that key then. The delete queues the indices instead, and the
    rerun applies them here, at the top of the replay section and before either picker exists.
    """
    pending = st.session_state.pop(f"event_reindex::{match_id}", None)
    if not pending:
        return
    removed = {int(index) for index in pending.get("removed") or ()}
    _reindex_event_pickers(match_id, removed, max(0, int(pending.get("remaining") or 0)))


def _apply_timeline_deletes(library: MatchLibrary, match_id: str, result) -> None:
    """Delete the markers a right-click on the timeline strip named, then redraw without them.

    The strip's component value is sticky, so every delete carries a sequence number and this page acknowledges
    the newest one it handled - the same contract as the tag bar. The acknowledgment is written even when a row
    no longer matches the log (another tab may have changed it); otherwise the strip would keep hiding a marker
    that is still stored. The rows themselves carry the event's identity, not just its position, because a
    position can point at a different event by the time the round trip lands.
    """
    if not isinstance(result, dict):
        return
    rows = [row for row in (result.get("deletes") or []) if isinstance(row, dict)]
    if not rows:
        return
    ack_key = f"replay_delete_ack::{match_id}"
    newest = int(st.session_state.get(ack_key, 0))
    for row in rows:
        try:
            newest = max(newest, int(row.get("seq") or 0))
        except (TypeError, ValueError):
            continue
    if newest <= int(st.session_state.get(ack_key, 0)):
        return  # nothing new: the sticky value was already handled on an earlier run
    st.session_state[ack_key] = newest
    log = library.events(match_id)
    removed = deleted_events(log.events, rows)
    if not removed:
        st.session_state["events_flash"] = (
            "warning",
            "That event is not on the timeline any more - it may have been removed in another tab.",
        )
        st.rerun()
        return
    for index, _event in removed:
        log.remove(index)
    library.save_events(match_id, log)
    # The footage-moment picker already exists in this run, so its key cannot be written now. Queue the repair
    # for the rerun, which applies it before the pickers are created.
    st.session_state[f"event_reindex::{match_id}"] = {
        "removed": sorted(index for index, _event in removed),
        "remaining": len(log.events),
    }
    if len(removed) == 1:
        event = removed[0][1]
        message = f"Removed {event.type} at {_clock(event.time_s)} from the timeline."
    else:
        message = f"Removed {len(removed)} events from the timeline."
    st.session_state["events_flash"] = ("success", message)
    # Rerun so the strip, the table and any open picker all see the deletion in the same gesture.
    st.rerun()


def _team_name_editor(library: MatchLibrary, match_id: str, payload: dict) -> list[str]:
    """Show each team's kit color and let it be named; returns the names to use from here on.

    Everything the pipeline writes calls the teams 0 and 1 (the more red kit is 0), which nobody can picture. The
    report carries the color the clustering measured for each of them, so the swatch is the anchor: a person
    names "the red team", not "team 0", and from then on the table, the chart, the replay and the tags agree. The
    measured color also suggests the name itself ("Reds", "Dark blues"), which turns naming into a press and an
    edit rather than a blank field - and a team that already has a name keeps it, because the suggestion only ever
    fills a placeholder.
    """
    record = library.load(match_id)
    stored = list(record.team_names)[:2] + list(DEFAULT_TEAM_NAMES[len(record.team_names) :])
    # A report written before the colors were recorded has no such field at all; one that has the field but no
    # value is a match whose kits the clustering could not separate. The two need different advice.
    team_rows = list(payload.get("teams") or [])
    colors_recorded = colors_were_recorded(team_rows)
    colors = team_colors(team_rows)
    suggestions = [suggest_team_name(rgb) if rgb is not None else "" for rgb in colors]

    st.subheader("Teams")
    st.caption(
        "The two teams are told apart by kit color, and each keeps the same number every time this report is built. "
        "Name them here - the table below, the momentum chart, the replay and the event tags then use the names."
    )
    for index, column in enumerate(st.columns(2)):
        with column:
            rgb = colors[index]
            if rgb is not None:
                st.color_picker(
                    f"Team {index + 1} kit color",
                    value=color_hex(rgb),
                    disabled=True,
                    key=f"kit_swatch::{match_id}::{index}",
                    help="Measured from the match footage: the average color of the kit pixels of the players the clustering put in this team.",
                )
                looks = f"Looks {color_name(rgb)} in the footage."
                if suggestions[index] and is_default_team_name(stored[index]):
                    looks += f" A name for that might be **{suggestions[index]}**."
                st.caption(looks)
            elif not colors_recorded:
                st.caption(
                    f"Team {index + 1}: this report was built before kit colors were recorded, so there is no "
                    "swatch or suggested name to show yet - build the report again below and they appear."
                )
            else:
                st.caption(f"Team {index + 1}: no kit color recorded - the kits were not separable here.")

    with st.form(f"team_names::{match_id}"):
        for index in (0, 1):
            current = stored[index]
            if is_default_team_name(current):
                current = suggestions[index] or DEFAULT_TEAM_NAMES[index]
            st.text_input(
                f"Team {index + 1} name",
                value=current,
                key=f"team_name::{match_id}::{index}",
                max_chars=40,
            )

        def _save_names() -> None:
            fresh = library.load(match_id)
            fresh.team_names = [
                str(st.session_state.get(f"team_name::{match_id}::{i}", "")).strip() or DEFAULT_TEAM_NAMES[i]
                for i in (0, 1)
            ]
            library.save(fresh)
            st.session_state["step3_flash"] = (
                "success",
                f"Teams are now {fresh.team_names[0]} (team 1) and {fresh.team_names[1]} (team 2).",
            )

        st.form_submit_button("Save team names", on_click=_save_names)
    return [
        str(st.session_state.get(f"team_name::{match_id}::{i}", stored[i])).strip() or DEFAULT_TEAM_NAMES[i]
        for i in (0, 1)
    ]


@st.cache_data(show_spinner=False)
def _load_replay_cached(path: str, mtime_ns: int) -> dict:
    """Parse the replay once per file version; it is re-read on every rerun otherwise."""
    del mtime_ns
    return upgrade_replay_payload(json.loads(Path(path).read_text()))


@st.cache_data(show_spinner=False)
def _load_report_cached(path: str, mtime_ns: int) -> dict:
    """Parse the report once per file version, restoring its minute keys to integers."""
    del mtime_ns
    payload = upgrade_report_payload(json.loads(Path(path).read_text()))
    momentum = {}
    for key, value in (payload.get("momentum") or {}).items():
        try:
            momentum[int(key)] = value
        except (TypeError, ValueError):
            continue
    payload["momentum"] = momentum
    return payload


def _report_payload(library: MatchLibrary, match_id: str | None) -> dict | None:
    """The saved report, cached until the file changes; the payload is tens of megabytes."""
    if match_id is None:
        return None
    path = library.path(match_id) / "report.json"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    return _load_report_cached(str(path), mtime)


BALL_STATUS_FILE = "ball_scan.json"  # written by scripts/run_ball_scan.py beside the segment
BALL_TRACK_FILE = "ball_track.json"  # the scan's result, beside the status
JERSEY_STATUS_FILE = "jerseys_status.json"  # written by scripts/extract_jerseys.py beside the match
AUDIO_STATUS_FILE = "audio_scan.json"  # written by scripts/run_audio_scan.py beside the match
SCAN_STATUS_STALE_S = 300.0  # any scan silent for this long counts as stopped, whatever its file says


def _scan_alive(status: dict) -> bool:
    """Whether a background scan is running right now: its status says so *and* it was updated recently.

    The recentness is the point. A process killed without a chance to write (a reboot, a crash) leaves "running"
    behind, and a control disabled by a dead process cannot be pressed to start it again - so anything that has
    not reported for minutes counts as stopped. Every scan updates every few seconds while it runs, so the
    window is generous.
    """
    if status.get("state") != "running":
        return False
    return (time.time() - float(status.get("updated") or 0.0)) < SCAN_STATUS_STALE_S


def _ball_status(segment_dir: Path) -> dict:
    """The ball scan's status file beside the segment; {} until it has ever run."""
    try:
        return json.loads((Path(segment_dir) / BALL_STATUS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _ball_scan_alive(segment_dir: Path) -> bool:
    """Whether the ball scan is running right now - its button stays disabled while it is."""
    return _scan_alive(_ball_status(segment_dir))


def _queue_status(path: Path, *, message: str, **extra) -> None:
    """Mark a scan's own status file queued before the worker replaces it with "running"."""
    try:
        Path(path).write_text(
            json.dumps({"state": "queued", "message": message, "updated": time.time(), **extra})
        )
    except OSError:
        pass  # an unwritable directory fails the child too, and its error lands in its status


def _library_root(library: MatchLibrary) -> str | None:
    """The explicit matches root, when the page was given one; None means the worker discovers the usual roots."""
    return str(library.root) if library.root is not None else None


def _submit_report_job(
    library: MatchLibrary,
    match_id: str,
    segment_dir: Path,
    *,
    with_detections: bool = False,
    video: str = "",
    strictness: float = MIN_PROMINENCE,
    keep_voices: bool = True,
) -> dict:
    """Queue a report build (or the report followed by the full detection chain) and return its status."""
    kind = "build_and_detect" if with_detections else "report"
    label = "Report + all detections" if with_detections else "Report + replay"
    params = {
        "match_id": str(match_id),
        "segment_dir": str(segment_dir),
        "library_root": _library_root(library),
    }
    if with_detections:
        params.update(
            video=str(video), strictness=float(strictness), keep_voices=bool(keep_voices)
        )
    return jobs.JOBS.submit(
        kind=kind,
        label=label,
        match_id=match_id,
        params=params,
        resources=("gpu",) if with_detections else (),
    )


DETECTIONS_STATUS_FILE = "detections.json"  # written by scripts/run_detections.py
VIDEO_EXPORT_STATUS_FILE = "video_export.json"  # written by scripts/run_video_export.py


def _video_export_status(library: MatchLibrary, match_id: str) -> dict:
    """The match-video export's status file beside the match; {} until it has ever run."""
    try:
        return json.loads((library.path(match_id) / VIDEO_EXPORT_STATUS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _video_export_alive(library: MatchLibrary, match_id: str) -> bool:
    """Whether a video export is running right now, not just leaving "running" behind after a crash."""
    return _scan_alive(_video_export_status(library, match_id))


def _start_video_export(
    library: MatchLibrary,
    match_id: str,
    segment_dir: Path,
    source: str,
    output: Path,
    width: int,
    fps: float,
    codec: str,
    team: bool,
    scoreboard: bool,
    ball: bool,
    numbers: bool,
) -> None:
    """Start the export as its own process, writing a "Starting..." status before the child boots."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_video_export.py"),
        "--match", match_id,
        "--segment", str(segment_dir),
        "--source", str(source),
        "--output", str(output),
        "--width", str(int(width)),
        "--fps", f"{float(fps):g}",
        "--codec", codec,
    ]
    if team:
        command.append("--team")
    if scoreboard:
        command.append("--scoreboard")
    if ball:
        command.append("--ball")
    if numbers:
        command.append("--numbers")
    overlays = {
        "team": bool(team),
        "scoreboard": bool(scoreboard),
        "ball": bool(ball),
        "numbers": bool(numbers),
    }
    status_path = library.path(match_id) / VIDEO_EXPORT_STATUS_FILE
    _queue_status(
        status_path,
        message="Queued - the job manager will start the export.",
        stage="prepare",
        progress=0.0,
        output=str(output),
        width=int(width),
        fps=float(fps),
        codec=codec,
        overlays=overlays,
    )
    jobs.JOBS.submit(
        kind="video_export",
        label="Video export",
        match_id=match_id,
        argv=command,
        watch=status_path,
        resources=(),
    )


def _detections_status(library: MatchLibrary, match_id: str) -> dict:
    """The all-detections run's status file beside the match; {} until it has ever run."""
    try:
        return json.loads((library.path(match_id) / DETECTIONS_STATUS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _detections_alive(library: MatchLibrary, match_id: str) -> bool:
    """Whether the enrichment run is in progress, not just leaving "running" behind after a crash."""
    return _scan_alive(_detections_status(library, match_id))


def _start_ball_scan(library: MatchLibrary, match_id: str, segment, segment_dir: Path, video: str) -> None:
    """Start the ball scan as its own process; its Scan activity row reflects the run from the click on.

    The child takes seconds to boot (interpreter, torch, the segment's meta) and only then writes its first
    status. Writing the same "Starting..." the scan itself would write closes that window: the button disables
    from the moment of the click instead of briefly offering a second scan that would race the first over the
    same checkpoint. The child's own first update overwrites this one. The scan decodes at the SEGMENT's clock,
    so the segment's own video wins when this machine has it; the page's selected recording is the fallback.
    """
    segment_video = str(segment.meta.get("video") or "") if segment is not None else ""
    scan_video = segment_video if segment_video and Path(segment_video).exists() else video
    status_path = Path(segment_dir) / BALL_STATUS_FILE
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_ball_scan.py"),
        "--segment",
        str(segment_dir),
        "--video",
        scan_video,
    ]
    _queue_status(status_path, message="Queued - the job manager will start the ball scan.")
    jobs.JOBS.submit(
        kind="ball_scan",
        label="Ball scan",
        match_id=match_id,
        argv=command,
        watch=status_path,
        resources=("gpu",),
    )


def _start_jersey_scan(library: MatchLibrary, match_id: str, segment, segment_dir: Path, video: str) -> None:
    """Start the shirt-number scan as its own process; its Scan activity row reflects the run from the click on.

    The scan decodes at the SEGMENT's clock - the clock of the video the segment was built from, which is not
    necessarily whatever the picker holds: decoding a clip at game-clock offsets reads the wrong film entirely.
    So the segment's own video wins when it is still reachable; the picker is only the fallback. The child spends
    its first minutes loading the segment and rebuilding the tracks before it writes anything; the "Starting..."
    below closes that window.
    """
    segment_video = str(segment.meta.get("video") or "") if segment is not None else ""
    scan_video = segment_video if segment_video and Path(segment_video).exists() else video
    status_path = library.path(match_id) / JERSEY_STATUS_FILE
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "extract_jerseys.py"),
        "--match", match_id,
        "--video", scan_video,
        "--segment", str(segment_dir),
    ]
    _queue_status(
        status_path,
        message="Queued - the job manager will start the shirt-number scan.",
        crops_total=0,
        crops_done=0,
        readings=0,
    )
    jobs.JOBS.submit(
        kind="jersey_scan",
        label="Shirt numbers",
        match_id=match_id,
        argv=command,
        watch=status_path,
        resources=("gpu",),
    )


def _start_audio_scan(library: MatchLibrary, match_id: str, video: str) -> None:
    """Start the whistle scan as its own process; its Scan activity row reflects the run from the click on.

    The strictness and voice settings are baked into the command now, because the background process reads
    nothing from this session - the page has moved on long before the scan finishes. The child takes seconds to
    boot (interpreter, libraries) before it writes its first status; the "Starting..." below closes that window.
    """
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_audio_scan.py"),
        "--match",
        str(match_id),
        "--video",
        str(video),
        "--strictness",
        f"{float(st.session_state.get('whistle_strictness', MIN_PROMINENCE)):.1f}",
    ]
    if not st.session_state.get("whistle_reject_voices", True):
        command.append("--keep-voices")
    status_path = library.path(match_id) / AUDIO_STATUS_FILE
    _queue_status(
        status_path, message="Queued - the job manager will start the whistle scan.", stage="extract", progress=0.0
    )
    jobs.JOBS.submit(
        kind="audio_scan",
        label="Whistle scan",
        match_id=match_id,
        argv=command,
        watch=status_path,
        resources=(),
    )


def _footage_moment(
    match_id: str,
    replay_duration_s: float,
    event_log,
    game_record,
    window_start: float,
) -> tuple[float | None, int, int | None]:
    """The moment picker beside the footage pane: the strip second to jump to, and a sequence for the request.

    The pane's footage is a live MJPEG stream, and an MJPEG connection cannot be seeked - so the way to watch
    "from that moment" is to move the *animation* there and let the pane follow it. The moment is chosen from the
    same event log the table and the strip show and translated onto the strip's clock exactly as the strip's own
    markers are, so the picker, the marker and the stream all agree on when the moment is.

    Returns ``(seek_to_s, seek_seq, index)``. ``seek_to_s`` is ``None`` when the selected moment lies outside the
    analyzed window (there is no second to jump to); ``seek_seq`` changes only when the button is pressed, and
    the component acts on a changed sequence so an unrelated rerun never re-seeks. ``index`` is the selected
    event's position in the log - the review buttons under the pitch act on it - or ``None`` when there are no
    events at all.
    """
    if not event_log.events:
        return None, 0, None
    # The picker is the same list the table shows, so a moment is chosen once and read everywhere. Its label leads
    # with the match clock where the game is marked, because that is the time a coach reads out.
    def _label(index: int) -> str:
        event = event_log.events[index]
        match_clock = None
        if game_record is not None and game_record.bounds() is not None:
            on_game = game_lib.game_time(game_record, event.time_s, event.video or game_record.output)
            if on_game is not None:
                half = game_record.half_of(on_game)
                marks = game_record.bounds()
                if half == 1:
                    match_clock = _clock(on_game - marks[0])
                elif half == 2:
                    match_clock = f"{_clock(on_game - marks[1])} (2H)"
        return _event_label(event, index, match_clock)

    pick_col, button_col = st.columns([3, 1])
    with pick_col:
        index = st.selectbox(
            "Footage moment to jump to",
            range(len(event_log.events)),
            format_func=_label,
            key=f"pitch_clip_event::{match_id}",
            help=(
                "The annotated stream beside the pitch plays from this moment: the button jumps the animation "
                "there and starts both in step."
            ),
        )
    event = event_log.events[index]
    moment = moment_for_event(event)
    # The moment on the strip's clock: its own recording time translated onto the game's and then onto the
    # analyzed window's - the same translation the strip's markers use, so a jump lands where the marker is.
    on_game = (
        game_lib.game_time(game_record, moment.time_s, event.video)
        if game_record is not None and event.video
        else moment.time_s
    )
    strip_start = None if on_game is None else on_game - window_start
    strip_end = None if strip_start is None else strip_start + (moment.end_s - moment.start_s)
    # The strip's clock runs 0..duration over the analyzed window; a moment outside it has nothing to jump to.
    if strip_start is not None and (strip_end < 0 or strip_start > replay_duration_s):
        strip_start = strip_end = None
    seq_key = f"pitch_seek_seq::{match_id}"
    with button_col:
        st.write("")  # line the button up with the selectbox
        if st.button(
            "Jump to this moment",
            key=f"pitch_moment_jump::{match_id}",
            disabled=strip_start is None,
            help="Jump the animation to this moment and play it with the footage beside it.",
        ):
            st.session_state[seq_key] = int(st.session_state.get(seq_key, 0)) + 1
    if strip_start is None:
        st.caption("This moment is outside the analyzed window, so there is no second to jump to.")
    return strip_start, int(st.session_state.get(seq_key, 0)), index


@st.cache_data(show_spinner=False)
def probe_cached(video: str):  # noqa: ANN201 - VideoProbe, kept out of the import list of callers
    """ffprobe once per video: the page re-renders on every gesture and every poll, and each probe is a process.

    A ``game.json`` manifest is probed through the clip-aware source: the size and rate come from the clips and
    the duration from the manifest, which is the whole game - never the merged file that no longer exists.
    """
    return probe_source(video)


def segment_fingerprint(segment_dir: Path, chunks: int) -> str:
    """Cache key for a segment's contents: it changes when meta, status, the pan model or the chunks change."""
    meta_mtime = (segment_dir / "meta.json").stat().st_mtime_ns if (segment_dir / "meta.json").exists() else 0
    status_mtime = (segment_dir / "status.json").stat().st_mtime_ns if (segment_dir / "status.json").exists() else 0
    pan_mtime = (segment_dir / "pan_model.json").stat().st_mtime_ns if (segment_dir / "pan_model.json").exists() else 0
    return f"{meta_mtime}-{status_mtime}-{pan_mtime}-{chunks}"


@st.cache_resource(show_spinner=False, max_entries=2)
def segment_and_poses_cached(segment_dir: str, fingerprint: str):  # noqa: ANN201
    """Load a segment and rebuild its pose chain once per process, not once per rerun.

    Step 2 is a gesture loop: every aim click re-renders the page, and loading a whole-game segment (hundreds of MB)
    and integrating 21k poses took the better part of a minute of *every* rerun. The fingerprint in the key is what
    keeps this honest: a re-analysis (new meta/status/chunks) misses the cache instead of serving stale poses.
    """
    del fingerprint
    segment = load_segment(segment_dir)
    q, focal = segment_poses(segment)
    return segment, q, focal


@st.cache_resource(show_spinner=False, max_entries=6)
def frame_cached(video: str, time_s: float, width: int):  # noqa: ANN201 - a shared, read-only frame array
    """A frame read from the source, shared across reruns.

    Aiming changes the *crop*, not the frame: without this every click paid a fresh 4K ffmpeg seek (2+ s). Callers
    treat the frame as read-only - the crop is a view, and the fit check draws on a resized copy.
    """
    return grab_frame(video, time_s, width=width)


def timeline_ready(segment, segment_dir: Path) -> bool:
    """Whether the segment's scrub proxy is ready, starting the background build if it is not.

    The whole-frame viewport doubles as the timeline, which plays a small proxy of the segment. That proxy is a full
    pass over the video - minutes of work for 4K/60 footage - so it is built as its own process (like Stage A) and
    the viewport falls back to a still frame until it lands.
    """
    if timeline.proxy_is_ready(segment_dir):
        return True
    start_s = float(segment.meta.get("start_s", segment.time[0]))
    duration_s = float(segment.meta.get("end_s", segment.time[-1] + 1.0)) - start_s
    timeline.start_background_build(segment_dir, segment.meta["video"], start_s=start_s, duration_s=duration_s)
    return False


@st.fragment(run_every=POLL_SECONDS)
def _timeline_status(segment_dir: Path, watch_key: str) -> None:
    """Live progress of the scrub-timeline build, and the switch-over when it lands."""
    state = timeline.read_build_state(segment_dir)
    status = state.get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = status
    if status == "error":
        st.warning(
            f"Could not build the scrub timeline ({state.get('error')}). Using a still frame and a plain frame "
            "slider - you can retry the background build below."
        )

        def _retry_build() -> None:
            timeline.write_build_state(segment_dir, state="idle", error=None)

        st.button("Retry building the scrub timeline", on_click=_retry_build, key=f"timeline_retry::{segment_dir}")
    elif status == "running":
        fraction = float(state.get("progress") or 0.0)
        st.progress(
            min(1.0, fraction),
            text="Building the scrub timeline - a one-time pass over this segment...",
        )
        st.caption(
            "The build runs in the background and this bar updates by itself. Until it lands the view below uses a "
            "still frame with a plain slider; afterwards it becomes a video you can scrub. Long segments are built "
            "from keyframes only - about one picture per second, so minutes rather than the length of the footage - "
            "and the frame you click is still read from the source at full resolution."
        )
    else:
        st.info("Preparing the scrub timeline in the background...")
    if previous == "running" and status != "running":
        st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def _stage_a_status(segment_dir: Path, watch_key: str) -> None:
    """Live Stage A progress.

    Polls the status file in a fragment, so the bar moves without a manual refresh. When the run finishes, one full
    rerun replaces this fragment and lets the rest of the page - which was rendered before the change - pick the
    finished segment up.
    """
    status = read_status(segment_dir)
    state = (status or {}).get("state")
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if status is None:
        st.info("Not analyzed yet.")
    else:
        st.write(f"Stage A: **{state}**, {status.get('frames_done', 0)}/{status.get('total_frames', 0)} frames")
        if state == "queued":
            st.info("Queued - the job manager will start the analysis when a slot is free.")
        elif state == "running":
            fraction = status.get("frames_done", 0) / max(1, status.get("total_frames", 1))
            detail = f"{status.get('fps', 0):.1f} analyzed fps"
            eta = status.get("eta_s")
            if isinstance(eta, (int, float)) and eta > 0:
                detail += f", about {_clock(float(eta))} to go"
            st.progress(min(1.0, fraction), text=detail)
        elif state == "error":
            st.error(status.get("error", "unknown error"))
        if status.get("lost"):
            st.caption(f"{status['lost']} frame(s) had no usable camera motion; they are excluded from metrics.")
    st.caption(
        f"Output directory: `{_location_label(segment_dir)}` "
        f"({stage_a.completed_chunks(segment_dir)} chunk(s) on disk)"
    )
    if previous == "running" and state != "running":
        st.rerun()


def saved_calibration(
    library: MatchLibrary,
    match_id: str | None,
    *,
    expected_source: str | None = None,
) -> PitchCalibration | None:
    """The calibration for this match: whatever was just clicked, else whatever is on disk.

    Returns None when there is no match selected, which is the normal state on a first visit to the page.

    When ``expected_source`` is given and the saved clicks name a different recording, the saved fit is not
    returned. Its clicks are frame-indexed into that other recording, so projecting it through this match's chain
    would draw a confidently wrong overlay - the numbers look healthy because the clicks agree with each other,
    just on the wrong pictures.
    """
    pending = st.session_state.get("calibration")
    if pending:
        return PitchCalibration.from_json(pending)
    if match_id is None:
        return None
    clicks_source = library.load_clicks_source(match_id)
    if expected_source is not None and clicks_source is not None:
        if Path(clicks_source).resolve() != Path(expected_source).resolve():
            return None
    clicks_pitch = library.load_clicks_pitch(match_id)
    if clicks_pitch is not None:
        record = library.load(match_id)
        if abs(float(record.pitch_length_m) - clicks_pitch[0]) > 0.05 or abs(float(record.pitch_width_m) - clicks_pitch[1]) > 0.05:
            return None
    return library.load_calibration(match_id)


def line_landmarks_for_match(library: MatchLibrary, match_id: str) -> list:
    """Cached automatic line observations for this match, or an empty list when the scan has not run.

    The scan itself is a whole-video decode, so it is run offline (``scripts/fit_pan_axis.py --detect-lines``) and
    cached beside the match. The dashboard fit only consumes the cache; a missing cache simply means the fit uses
    the clicks alone.
    """
    path = library.path(match_id) / "line_observations.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
        observations = [
            LineObservation(
                frame=int(row["frame"]),
                u=float(row["u"]),
                v=float(row["v"]),
                pitch_x=float(row["pitch_x"]),
                pitch_y=float(row["pitch_y"]),
                direction=tuple(float(v) for v in row["direction"]),
                label=str(row.get("label", "line")),
                contrast=float(row.get("contrast", 0.0)),
            )
            for row in rows
        ]
        return observations_to_landmarks(observations)
    except (OSError, ValueError, KeyError, TypeError):
        return []


def fit_pan_axis_from_clicks(
    library: MatchLibrary,
    match_id: str,
    segment,
    labeled: list[tuple[dict, str]],
    land_table: dict[str, tuple[float, float]],
    length_m: float,
    width_m: float,
    source: str | None = None,
):
    """Fit the 3-DOF tripod axis, refit the calibration through it, and save both.

    This is the slow, explicit action: the search solves a camera for each candidate axis, so it takes minutes on
    a whole game. It uses the cached line observations when they exist and the clicks otherwise. Returns
    ``(calibration, fit)`` or ``None`` when the segment has no log / the ground control cannot place a camera.
    """
    landmarks = landmarks_from_points(labeled, land_table)
    all_landmarks = landmarks + line_landmarks_for_match(library, match_id)
    progress = st.progress(0.0, text="Fitting the tripod pan axis...")

    def report(fraction: float) -> None:
        progress.progress(
            min(0.99, max(0.0, float(fraction))),
            text=f"Fitting the tripod pan axis... {int(min(1.0, max(0.0, fraction)) * 100)}%",
        )

    try:
        result = fit_segment_pan_axis(
            segment, all_landmarks, fixed_height_m=CAMERA_HEIGHT_M, on_progress=report
        )
    except Exception as exc:  # noqa: BLE001 - reported on the page
        st.session_state["calib_fit_error"] = f"{type(exc).__name__}: {exc}"
        return None
    if result is None:
        st.session_state["calib_fit_error"] = (
            "the tripod-axis fit could not place a camera through these landmarks (or this segment has no "
            "gimbal log). The saved calibration was left unchanged."
        )
        return None
    calibration, _q_model, _focal_model, fit = result
    save_pan_model(segment, fit.model)
    clicks = [{"frame": lm.frame, "u": lm.u, "v": lm.v, "label": lm.label} for lm in landmarks]
    click_count = len(landmarks)
    excluded_clicks = {index for index in calibration.excluded if index < click_count}
    click_rms = float(
        np.sqrt(
            np.mean(
                [
                    residual
                    for index, residual in enumerate(calibration.residuals_m[:click_count])
                    if index not in excluded_clicks
                ]
                or [0.0]
            )
        )
    )
    st.session_state["calib_axis_message"] = (
        f"Tripod axis fitted: click RMS {click_rms:.2f} m (trimmed fit RMS {fit.rms_m:.2f} m). "
        "The projection now uses the fitted axis."
    )
    st.session_state["calibration"] = calibration.to_json()
    st.session_state["calib_landmarks"] = clicks
    library.save_calibration(match_id, calibration)
    library.save_clicks(match_id, clicks, (length_m, width_m), source=source)
    progress.progress(1.0, text="Tripod pan axis fitted.")
    return calibration, fit


def fit_from_clicks(
    library: MatchLibrary,
    match_id: str,
    segment,
    q: np.ndarray,
    focal: np.ndarray,
    labeled: list[tuple[dict, str]],
    land_table: dict[str, tuple[float, float]],
    length_m: float,
    width_m: float,
    source: str | None = None,
) -> PitchCalibration | None:
    """Fit the calibration from the labeled clicks and save it; None when the fit failed.

    Shared by the automatic refit and the manual button, so both do exactly the same thing: solve, store in the
    session, and write the calibration and its clicks to the match record. The caller reports the outcome - the
    message differs between the two paths.

    The rig's height is known (a fixed 4 m pole), so it is pinned rather than solved - one fewer degree of freedom
    for the clicks to pin down, which measurably tightens a four-click fit.
    """
    landmarks = landmarks_from_points(labeled, land_table)
    chain = {point["frame"]: (q[point["frame"]], float(focal[point["frame"]])) for point, _ in labeled}
    try:
        calibration = calibrate(
            landmarks, chain, segment.aspect, pose_source=segment_pose_source(segment), fixed_height_m=CAMERA_HEIGHT_M
        )
    except Exception as exc:  # noqa: BLE001 - reported on the page
        # Keep the cause for the page: one generic sentence hid real bugs behind "check the labels" - the solver
        # saying *why* it failed is the difference between a five-minute fix and a hunt.
        st.session_state["calib_fit_error"] = f"{type(exc).__name__}: {exc}"
        return None
    # A 3-DOF axis fit belongs to the segment, not to one click set: a later re-label or re-click must not throw it
    # away. The q the caller passed already comes from the sidecar (segment_and_poses_cached keys on its mtime), so
    # preserving the model here keeps the calibration and the camera path describing the same camera.
    if load_pan_model(segment) is not None:
        existing = st.session_state.get("calibration")
        previous = PitchCalibration.from_json(existing) if existing else library.load_calibration(match_id)
        if previous is not None and previous.pan_model:
            calibration = replace(calibration, pan_model=previous.pan_model)
    # A fit that cannot be reconciled must never replace a good saved calibration. Restoring a click set against
    # the wrong recording (or a mislabeled set) produces a wildly wrong but numerically "successful" solve; saving
    # it would destroy the calibration that was working. Keep the previous calibration on disk and report the
    # rejection instead.
    diagnosis = diagnose_fit(calibration)
    if diagnosis.reason or diagnosis.too_few_agreeing:
        if diagnosis.too_few_agreeing and len(landmarks) <= 4:
            cause = (
                "exactly four clicks were supplied, and one of them disagrees with the rest - there is no spare "
                "click to drop, so the fit cannot be verified. Add a fifth spread landmark, or remove the click "
                "the residual table flags, and refit"
            )
        else:
            cause = diagnosis.reason or "fewer than four clicks agree with each other"
        st.session_state["calib_fit_error"] = (
            "the refit is not usable - " + cause + ". The saved calibration was left unchanged."
        )
        return None
    st.session_state.pop("calib_fit_error", None)
    clicks = [{"frame": lm.frame, "u": lm.u, "v": lm.v, "label": lm.label} for lm in landmarks]
    st.session_state["calibration"] = calibration.to_json()
    st.session_state["calib_landmarks"] = clicks
    library.save_calibration(match_id, calibration)
    library.save_clicks(match_id, clicks, (length_m, width_m), source=source)
    return calibration


def calibration_failure() -> str:
    """What a failed fit shows: the solver's own cause when it left one, else the generic advice."""
    cause = st.session_state.get("calib_fit_error")
    if cause:
        return f"Calibration failed: {cause}"
    return "Calibration failed - the clicks do not determine a camera. Check the labels."


def fit_message(calibration: PitchCalibration, labeled: list[tuple[dict, str]]) -> str:
    """What a successful fit says about itself: the RMS, and whether the drift got re-anchored."""
    message = f"Fit RMS error {calibration.rms_error_m:.2f} m."
    if calibration.drift is not None:
        message += (
            f" The chain is re-anchored at the {len(calibration.drift.frames)} clicked frame(s) - "
            "click landmarks later in the video to pin the drift there too."
        )
    elif len({point["frame"] for point, _label in labeled}) > 1:
        message += (
            " No drift correction was fitted: it needs at least two moments with "
            f"{MIN_CLICKS_PER_ANCHOR} or more clicks each, and the moments without enough clicks "
            "still judge the fit - check their residuals below."
        )
    return message


_SCAN_STATE_WORDS = {
    "queued": "queued",
    "running": "running",
    "done": "done",
    "error": "failed",
    "partial": "stopped early",
}


def _scan_row(title: str, state: str | None):
    """The shell every scan row shares: a bordered box whose first line names the scan and its state.

    Returns the box to draw the row's body into, so every scan reads the same way: name and state first, then a
    live bar, a stall warning, an error, or the last result.
    """
    box = st.container(border=True)
    with box:
        word = _SCAN_STATE_WORDS.get(str(state), "not run yet") if state else "not run yet"
        st.markdown(f"**{title}** - {word}")
    return box


@st.fragment(run_every=POLL_SECONDS)
def scan_activity_section(
    library: MatchLibrary, match_id: str, segment, segment_dir: Path, video: str
) -> None:
    """The one spot for every background scan: start it, watch it, read its last result.

    The scans used to report - and be started - under their own controls, so watching two of them meant
    scrolling between expanders. Here each scan gets a row rendered the same way: its state, a live bar
    carrying the scan's own message while it runs, a stall warning when a "running" file has gone quiet, or
    the last result when it is idle - and the control that starts or resumes it (with the whistle's settings)
    lives in the same row. One panel is the whole switchboard, and the whole progress report.
    """
    st.subheader("Scan activity")
    st.caption(
        "One row per scan - ball, whistle, shirt numbers - and for the all-detections run that chains them. "
        "Each row starts or resumes its scan and shows the last result; bars update by themselves while a scan "
        "runs."
    )
    finished_any = False

    # --- the ball scan ----------------------------------------------------------------------------------------
    ball = _ball_status(segment_dir)
    previous = st.session_state.get(f"ball_watch::{match_id}")
    st.session_state[f"ball_watch::{match_id}"] = ball.get("state")
    with _scan_row("Ball scan", ball.get("state")):
        if not ball:
            st.caption("Not scanned yet - about an hour on a whole game.")
        elif ball.get("state") == "queued":
            st.caption("Queued - it will start when the job manager has a free slot.")
        elif ball.get("state") == "running":
            if _scan_alive(ball):
                st.progress(
                    min(1.0, float(ball.get("progress") or 0.0)),
                    text=str(ball.get("message") or "Scanning..."),
                )
                st.caption("Running in the background - it can be closed and resumed later.")
            else:
                st.warning(
                    "Says it is running but has not reported for minutes, so it has probably stopped. Start it "
                    "again - it resumes from its last checkpoint."
                )
        elif ball.get("state") == "error":
            st.error(f"The scan failed: {ball.get('error')}")
        else:
            counts = ball.get("counts") or {}
            scanned = int(ball.get("scanned") or 0)
            if counts and scanned:
                total = int(ball.get("total_frames") or 0)
                seen = 100 * counts.get("tracking", 0) / scanned
                forecast = 100 * counts.get("coasting", 0) / scanned
                off = 100 * (counts.get("out_of_view", 0) + counts.get("lost", 0)) / scanned
                summary = (
                    f"Last scan: {scanned} of {total} frame(s) - the ball was seen on {seen:.0f}% of them, "
                    f"forecast across a missed frame on {forecast:.0f}%, off the picture on {off:.0f}%."
                )
            else:
                # A status written by an older version of the scan has no counts in it; the replay itself does
                # not care - the track on disk is the same - so the summary degrades to the message.
                summary = f"Last scan: {ball.get('message') or 'finished'}."
            if ball.get("state") == "partial":
                summary += " The scan was stopped early; start it again to resume it."
            else:
                summary += " Press **Build report** to draw the track on the replay."
            st.caption(summary)
        if st.button(
            "Scan for the ball (background)",
            key=f"scan_ball::{match_id}",
            disabled=_ball_scan_alive(segment_dir) or jobs.JOBS.has_active(match_id, kinds=("ball_scan",)),
            help="Follows the ball frame by frame at full resolution. It checkpoints, so a stopped scan resumes.",
        ):
            _start_ball_scan(library, match_id, segment, segment_dir, video)
            st.rerun(scope="fragment")
    finished_any = finished_any or (previous == "running" and ball.get("state") != "running")

    # --- the whistle scan -------------------------------------------------------------------------------------
    audio = library.load_audio_scan_status(match_id)
    previous = st.session_state.get(f"whistle_watch::{match_id}")
    st.session_state[f"whistle_watch::{match_id}"] = audio.get("state")
    with _scan_row("Whistle scan", audio.get("state")):
        if not audio:
            st.caption("Not scanned yet - decodes the recording's audio once, then looks for whistle blasts.")
        elif audio.get("state") == "queued":
            st.caption("Queued - it will start when the job manager has a free slot.")
        elif audio.get("state") == "running":
            if _scan_alive(audio):
                st.progress(
                    min(1.0, float(audio.get("progress") or 0.0)),
                    text=str(audio.get("message") or "Scanning..."),
                )
                st.caption("Running in the background - this bar updates by itself.")
            else:
                st.warning(
                    "Says it is running but has not reported for minutes, so it has probably stopped. Start it "
                    "again."
                )
        elif audio.get("state") == "error":
            st.error(f"The scan failed: {audio.get('error')}")
        else:
            found = int(audio.get("found", 0))
            minutes = float(audio.get("minutes", 0.0))
            if found:
                already = found - int(audio.get("added", found))
                dropped = int(audio.get("dropped", 0))
                st.caption(
                    f"Last scan: {minutes:.0f} min of audio, {found} candidate(s)"
                    + (f", {already} already recorded" if already else "")
                    + (f", {dropped} dropped as no longer detected" if dropped else "")
                    + f", the strongest at {float(audio.get('strongest', 0.0)):.0f}x the match level."
                )
            else:
                st.caption(
                    f"Last scan: {minutes:.0f} min of audio, no whistle candidates. The detector wants a loud, "
                    "sustained blast, so a quiet recording - or a referee a long way from the camera - can leave "
                    "it with nothing to report."
                )
        st.number_input(
            "Detector strictness (x the match's own level)",
            min_value=10.0,
            max_value=2000.0,
            value=float(MIN_PROMINENCE),
            step=10.0,
            key="whistle_strictness",
            help=(
                "How far above the match's typical level in the whistle band a blast must sit to be reported. "
                "Higher means fewer, more confident candidates. Bump it up if the pitches next door dominate the "
                "list; drop it if the referee's own whistle is being missed."
            ),
        )
        st.checkbox(
            "Reject voices and calls (keep only lone tones)",
            value=True,
            key="whistle_reject_voices",
            help=(
                "A whistle puts everything into its own narrow band. A shout - at any pitch - and a bird of "
                "prey's call bring their own lower harmonics and formants with them, so a blast that lifts the "
                "region below the band is dropped as a voice or a call. The thresholds were set from this match's "
                "own confirmed and rejected candidates; turn it off if a real referee's whistle is being missed."
            ),
        )
        if st.button(
            "Scan audio for whistles (background)",
            key=f"scan_audio::{match_id}",
            disabled=_scan_alive(audio) or jobs.JOBS.has_active(match_id, kinds=("audio_scan",)),
            help="Searches the whistle band for loud, sustained blasts; the settings above travel with the run.",
        ):
            _start_audio_scan(library, match_id, video)
            st.rerun(scope="fragment")
    finished_any = finished_any or (previous == "running" and audio.get("state") != "running")

    # --- the shirt-number scan --------------------------------------------------------------------------------
    jersey = library.load_jerseys_status(match_id)
    previous = st.session_state.get(f"jersey_watch::{match_id}")
    st.session_state[f"jersey_watch::{match_id}"] = jersey.get("state")
    with _scan_row("Shirt numbers", jersey.get("state")):
        if not jersey:
            st.caption("Not scanned yet - a few minutes of OCR over torso crops.")
        elif jersey.get("state") == "queued":
            st.caption("Queued - it will start when the job manager has a free slot.")
        elif jersey.get("state") == "running":
            if _scan_alive(jersey):
                done, total = int(jersey.get("crops_done", 0)), int(jersey.get("crops_total", 0))
                detail = str(jersey.get("message") or f"{done}/{total} crops, {jersey.get('readings', 0)} readings")
                st.progress(min(1.0, done / max(1, total)), text=detail)
                st.caption("Running in the background - this bar updates by itself.")
            else:
                st.warning(
                    "Says it is running but has not reported for minutes, so it has probably stopped. Start it "
                    "again."
                )
        elif jersey.get("state") == "error":
            st.error(f"The scan failed: {jersey.get('message')}")
        else:
            meta = library.load_jerseys(match_id).get("meta", {})
            st.caption(
                f"Last scan: {meta.get('crops', 0)} crops from {meta.get('tracks_scanned', 0)} tracks, "
                f"{meta.get('readings', 0)} readings, {meta.get('suggested', 0)} suggested number(s). "
                + (jersey.get("message") or "")
            )
        if st.button(
            "Scan for shirt numbers (background)",
            key=f"scan_jerseys::{match_id}",
            disabled=_scan_alive(jersey) or jobs.JOBS.has_active(match_id, kinds=("jersey_scan",)),
            help="Reads numbers off torso crops with OCR; only a number several readings agree on is reported.",
        ):
            _start_jersey_scan(library, match_id, segment, segment_dir, video)
            st.rerun(scope="fragment")
    finished_any = finished_any or (previous == "running" and jersey.get("state") != "running")

    # --- the run that chains them all -------------------------------------------------------------------------
    run = _detections_status(library, match_id)
    previous = st.session_state.get(f"detections_watch::{match_id}")
    st.session_state[f"detections_watch::{match_id}"] = run.get("state")
    with _scan_row("All-detections run", run.get("state")):
        if not run:
            st.caption(
                "Not run yet - **Build report + run all detections** above chains the three scans, the event "
                "detectors and a final rebuild, skipping whatever already finished."
            )
        elif run.get("state") == "queued":
            st.caption("Queued - it will start when the job manager has a free slot.")
        elif run.get("state") == "running":
            if _scan_alive(run):
                st.progress(
                    min(1.0, float(run.get("progress") or 0.0)),
                    text=str(run.get("message") or "Running the detections..."),
                )
                st.caption(
                    "Running in the background - the scans check their own checkpoints, so the run can be left "
                    "overnight; the report and the replay are rebuilt when everything is in."
                )
            else:
                st.warning(
                    "The run says it is running but has not reported for minutes, so it has probably stopped. "
                    "Press the button again: every stage that already finished is skipped."
                )
        elif run.get("state") == "error":
            st.error(f"All-detections run failed: {run.get('error') or run.get('message')}")
        else:
            st.caption(f"Last all-detections run: {str(run.get('message') or 'finished').rstrip('.')}.")
    finished_any = finished_any or (previous == "running" and run.get("state") != "running")

    # When anything the page draws from lands - shirt numbers for the replay, a ball track that wants a rebuild,
    # whistle candidates for the events table - one full rerun picks it all up.
    if finished_any:
        st.rerun()


@st.cache_data(show_spinner=False)
def _load_replay_boxes_cached(path: str, mtime_ns: int) -> dict:
    """Each track's own image boxes, read once per file version (they are only needed when a clip is cut)."""
    del mtime_ns
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError):  # missing, torn, or written by an older build: no boxes is a normal answer
        return {}


def _centered_clip_name(track_id: int, start_s: float, duration_s: float) -> str:
    """A stable name for a centered clip, so re-cutting the same window reuses the file instead of re-encoding.

    The framing command rate is part of the key: it *is* the pan's update rate (ffmpeg's ``sendcmd`` cannot
    interpolate), so a cached clip cut by a choppier planner must not be served after the planner is fixed -
    re-cutting the same window gets a fresh file.
    """
    key = f"{track_id}|{start_s:.2f}|{duration_s:.2f}|{COMMAND_RATE_HZ:g}hz"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"player_{track_id}_{digest}.mp4"


def _appearance_thumbnail(
    library: MatchLibrary,
    match_id: str,
    source: Path,
    appearance,  # noqa: ANN001 - identity.Appearance
) -> Path | None:
    """One crop of an appearance, cut from the footage once and kept beside the match.

    Seeing who an appearance is *is* the fastest way to recognize it - the point of the roster is typing a number
    against the right track, and a track id means nothing. One frame is decoded and its box cropped, which costs a
    single seek rather than a scan, and the JPEG is cached so looking again is free. Returns None when the frame
    cannot be read or the box has no size.
    """
    directory = library.identities_dir(match_id)
    # The window's start is part of the name: a rebuilt pipeline can reuse a track id for a different appearance,
    # and a stale still under the same name would then show the wrong player.
    target = directory / f"track_{int(appearance.track_id)}_{int(appearance.first_t)}.jpg"
    if target.exists():
        return target
    times = np.asarray(appearance.traj_t if appearance.traj_t is not None else [], dtype=np.float64)
    boxes = np.asarray(appearance.traj_box if appearance.traj_box is not None else np.zeros((0, 4)), dtype=np.float64)
    if len(times) == 0 or len(boxes) != len(times):
        return None
    # The frame where the player is largest: a thumbnail exists to be recognized, and a big crop is recognizable.
    heights = np.abs(boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(heights))
    frame = grab_frame(source, float(times[index]), width=960)
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) * width for v in boxes[index])
    margin = 0.05 * max(1.0, y2 - y1)
    left, right = int(max(0, x1 - margin)), int(min(width, x2 + margin))
    top, bottom = int(max(0, y1 - margin)), int(min(height, y2 + margin))
    if right - left < 8 or bottom - top < 8:
        return None
    crop = frame[top:bottom, left:right]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        return None
    return target


def _link_team_roster(library: MatchLibrary, match_id: str, team: int, roster: str) -> None:
    """Point one team of the match at a saved roster ("" to unlink it).

    The record is re-read before writing: saving the copy loaded for the render could clobber an edit made in
    between, and this page is not the only writer a match can have.
    """
    fresh = library.load(match_id)
    links = (list(fresh.team_rosters) + ["", ""])[:2]
    links[int(team)] = str(roster or "").strip()
    fresh.team_rosters = links
    library.save(fresh)


def _team_roster_editors(
    library: MatchLibrary,
    roster_library: RosterLibrary,
    match_id: str,
    record,  # noqa: ANN001 - library.MatchRecord
    replay: dict,
    numbers: dict[int, dict],
    team_names: list[str],
) -> None:
    """Per-team roster editors: connect the shirt numbers seen on the footage with the players' names.

    A number is read off the footage per track - that is all the video can know. A name is knowledge about the
    squad, so it is typed once per team as a *roster* (number -> name), saved to the roster library and linked to
    the match. Every track this match - or a future one between the same teams - gives that number to is then
    labelled with the name, without touching any track. The table rows are every number known for the team: the
    scan's detections for this match's tracks, plus the linked roster's own numbers (so a squad can be prepared
    before the scan has seen the shirts).
    """
    saved = roster_library.names()
    stats = {int(player["track_id"]): player["stats"] for player in replay.get("players") or []}
    team_of = {int(player["track_id"]): int(player["team"]) for player in replay.get("players") or []}
    st.caption(
        "The number scan reads each track's shirt; the name beside it is yours to fill once per team. **Save "
        "roster** stores the associations in the roster library, shared across matches - link the same roster in "
        "a future game and the names are back without retyping. The *tracks* column lists the tracks a number "
        "was worn on, the *seen* column for how long. Clear a name and save to drop the entry."
    )

    def _rows_for(team: int, linked: dict[int, str]) -> dict[int, dict]:
        rows: dict[int, dict] = {int(number): {"name": str(name), "tracks": []} for number, name in linked.items()}
        for track, entry in numbers.items():
            if team_of.get(int(track), -1) != team:
                continue
            number = entry.get("number")
            if number is None:
                continue
            row = rows.setdefault(int(number), {"name": "", "tracks": []})
            if not row["name"]:
                row["name"] = str(entry.get("name") or "")
            seen = float((stats.get(int(track)) or {}).get("time_s") or 0.0)
            row["tracks"].append((int(track), seen))
        return rows

    columns = st.columns(2)
    for team in (0, 1):
        with columns[team]:
            name = team_name(team, team_names)
            st.markdown(f"**{name}**")
            current = record.team_roster(team)
            # A saved roster with this team's own name is the obvious candidate: offer it first, after "(no
            # roster)", rather than auto-linking on a guess - the link is the user's choice, not the page's.
            match_name = next(
                (entry for entry in saved if entry.strip().casefold() == str(name).strip().casefold()), None
            )
            ordered = ([match_name] + [entry for entry in saved if entry != match_name]) if match_name else saved
            options = ["(no roster)"] + ordered
            choice = st.selectbox(
                "Roster",
                options,
                index=options.index(current) if current in options else 0,
                key=f"team_roster_choice::{match_id}::{team}",
                help="The saved roster whose names label this team's tracks. Rosters are shared between matches.",
            )
            if current and current not in saved:
                st.warning(f"The linked roster '{current}' is no longer in the library - pick another roster above.")
            picked = "" if choice == "(no roster)" else str(choice)
            if picked != current:
                _link_team_roster(library, match_id, team, picked)
                st.session_state["replay_flash"] = (
                    "success",
                    f"'{picked}' now names the players of {name}." if picked else f"{name} is no longer linked to a roster.",
                )
                st.rerun()  # the number merge at the top of the page reads the link: rebuild with it applied
            if match_name and picked != match_name:
                st.caption(f"A saved roster named '{match_name}' matches this team's name - pick it above to reuse those names.")
            linked = roster_library.load(picked) if picked else {}
            rows = _rows_for(team, linked)
            table = pd.DataFrame(
                [
                    {
                        "number": number,
                        "name": row["name"],
                        "seen (s)": round(sum(seen for _track, seen in row["tracks"]), 1),
                        "tracks": ", ".join(
                            str(track) for track, _seen in sorted(row["tracks"], key=lambda item: -item[1])
                        ),
                    }
                    for number, row in sorted(rows.items())
                ],
                columns=["number", "name", "seen (s)", "tracks"],
            )
            edited = st.data_editor(
                table,
                column_config={
                    "number": st.column_config.NumberColumn("number", min_value=1, max_value=99, step=1),
                    "name": st.column_config.TextColumn("name", max_chars=30),
                    "seen (s)": st.column_config.NumberColumn("seen (s)", disabled=True),
                    "tracks": st.column_config.TextColumn("tracks", disabled=True),
                },
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=f"team_roster_editor::{match_id}::{team}::{picked}",
            )
            save_name = st.text_input(
                "Roster name",
                value=picked or match_name or str(name),
                max_chars=40,
                key=f"team_roster_save_name::{match_id}::{team}::{picked}",
                help="Saving under an existing name overwrites that roster - in every match linked to it.",
            )
            if st.button(
                "Save roster",
                key=f"team_roster_save::{match_id}::{team}::{picked}",
                disabled=not str(save_name).strip(),
            ):
                roster_numbers: dict[int, str] = {}
                for row in edited.to_dict("records"):
                    number = row.get("number")
                    player_name = str(row.get("name") or "").strip()
                    if number is None or pd.isna(number) or not player_name:
                        continue
                    try:
                        number = int(number)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= number <= 99:
                        roster_numbers[number] = player_name
                target = str(save_name).strip()
                if not roster_numbers:
                    st.session_state["replay_flash"] = (
                        "warning",
                        "Nothing to save yet - type at least one player's name next to a number.",
                    )
                else:
                    roster_library.save(target, roster_numbers)
                    _link_team_roster(library, match_id, team, target)
                    st.session_state["replay_flash"] = (
                        "success",
                        f"Saved {len(roster_numbers)} name(s) as '{target}' and linked it to {name}.",
                    )
                    st.rerun()


def identity_section(
    library: MatchLibrary,
    match_id: str,
    segment,
    replay: dict,
    numbers: dict[int, dict],
) -> None:
    """Unique players: group each person's appearances, and cut a clip centered on one of them.

    The grouping is exact and comes from what is *known* - the team plus the shirt number or name, from the team
    rosters and the shirt-number scan. It deliberately does not guess from appearance: measured on this
    footage, embeddings do not separate individual players (the numbers are in the README), and a wrong merge would
    put one player's clips under another player's name. Appearances nobody has named are listed separately, so they
    can still be clipped and can still be named.
    """
    appearances = identity_lib.appearances_from_players(
        replay.get("players") or [],
        frames_per_second=float(replay.get("fps") or 5.0),
        start_s=float(segment.meta.get("start_s", 0.0)),
        # Each player's own boxes live beside the payload (they are tens of megabytes, and the browser that draws
        # the pitch never needs them); without them an appearance has no trajectory and no clip can be framed.
        boxes=_load_replay_boxes_cached(
            str(library.path(match_id) / "boxes.npz"),
            (library.path(match_id) / "boxes.npz").stat().st_mtime_ns
            if (library.path(match_id) / "boxes.npz").exists()
            else 0,
        ),
    )
    if not appearances:
        return
    identities = identity_lib.identities_from_labels(appearances, numbers)
    team_names = library.load(match_id).team_names
    named = sum(1 for identity in identities if identity.grouped_by)
    with st.expander("Unique players - and a clip centered on one of them", expanded=named > 0):
        # Numbers are attached to *tracks*, so a scan read against an older pitch fit describes tracks that may no
        # longer exist under those ids. Two checks say so - ids that have vanished from the report entirely, and a
        # scan saved before the calibration on disk - because "no numbers yet" and "numbers worn by the wrong
        # people" must not look the same.
        jerseys = library.load_jerseys(match_id)
        stored = list((jerseys.get("suggestions") or {}).keys())
        scan_saved = (jerseys.get("meta") or {}).get("calibration_saved")
        calibration_path = library.path(match_id) / "calibration.json"
        current_saved = calibration_path.stat().st_mtime if calibration_path.exists() else None
        stale_reason = identity_lib.numbers_are_stale(
            stored,
            appearances.keys(),
            scan_calibration_saved=None if scan_saved is None else float(scan_saved),
            calibration_saved=current_saved,
        )
        # Shown even when no number made it into the grouping: a stale file is exactly why the list looks empty of
        # names, and saying "no numbers yet" would send the user to re-run a scan that would produce the same file.
        if stored and stale_reason:
            st.warning(
                f"The stored shirt numbers cannot be trusted for this report: {stale_reason}. Re-run the number "
                "scan - or correct the numbers in the per-track table below - and the grouping here will be right."
            )
        st.caption(
            "Appearances are grouped into a person by what is known about them - the shirt number (or name) from "
            "the team rosters below and the automatic scan - never by guessing from the picture: measured on this "
            "footage, appearance embeddings do not separate individual players well enough to merge them (see the "
            "README). So the list is exact but incomplete: an appearance nobody has named stays its own row until "
            "it is named below."
        )
        rows = identity_lib.identity_rows(
            identities,
            appearances,
            lambda team: team_name(team, team_names),
        )
        frame = pd.DataFrame(rows)
        frame["first"] = frame["first_t"].map(_clock)
        frame["last"] = frame["last_t"].map(_clock)
        frame["tracks"] = frame["tracks"].map(lambda ids: ", ".join(str(track) for track in ids))
        frame = frame[
            ["player", "team", "appearances", "first", "last", "seen_s", "grouped_by", "tracks"]
        ].rename(columns={"seen_s": "seen (s)", "grouped_by": "grouped by"})
        st.dataframe(frame, hide_index=True, width="stretch")

        by_id = {identity.identity_id: identity for identity in identities}
        choice = st.selectbox(
            "Player to cut a clip around",
            [identity.identity_id for identity in identities],
            format_func=lambda identity_id: (
                f"{by_id[identity_id].label} "
                f"({team_name(by_id[identity_id].team, team_names)}, "
                f"{len(by_id[identity_id].members)} appearance(s), "
                f"{_clock(by_id[identity_id].first_t)}-{_clock(by_id[identity_id].last_t)})"
            ),
            key=f"identity_choice::{match_id}",
        )
        identity = by_id[choice]
        members = [appearances[track_id] for track_id in identity.members if track_id in appearances]
        members.sort(key=lambda appearance: appearance.first_t)
        appearance = st.selectbox(
            "Which appearance to cut",
            list(range(len(members))),
            format_func=lambda index: (
                f"{_clock(members[index].first_t)}-{_clock(members[index].last_t)} "
                f"({members[index].span_s:.0f} s, track {members[index].track_id})"
            ),
            key=f"identity_appearance::{match_id}",
        )
        chosen = members[appearance]
        source = Path(str(segment.meta.get("video") or ""))

        # A replay built before the boxes were added (or by a process still running the older code - Streamlit
        # keeps already-imported modules across an edit) has no per-player boxes, and the crop cannot be framed
        # from a pitch position: the clip needs to know where the player is *in the picture*.
        framed = chosen.traj_box is not None and len(chosen.traj_box) > 0
        if not framed:
            st.info(
                "This report does not carry each player's position in the frame, so a centered clip cannot be cut "
                "from it. Press **Build report** above and the appearances here will be clip-able."
            )
            return

        preview_col, cut_col = st.columns([1, 3])
        with preview_col:
            if st.button("Show this appearance", key=f"identity_look::{match_id}", disabled=not source.exists()):
                thumbnail = _appearance_thumbnail(library, match_id, source, chosen)
                if thumbnail is None:
                    st.warning("Could not cut a still of this appearance - the footage may not be available.")
                else:
                    st.image(str(thumbnail), caption=f"track {chosen.track_id}", width="stretch")
        # The default length is the appearance's own span, capped, and always a float (see `default_clip_length`:
        # an int default against float bounds is rejected by Streamlit and breaks the page for short appearances).
        # The key carries the appearance: each one gets its own default, so switching from a 50 s appearance to a
        # 17 s one offers 17 s rather than the length chosen for the previous player.
        duration = st.slider(
            "Clip length (s)",
            min_value=SHORTEST_CLIP_SECONDS,
            max_value=60.0,
            value=default_clip_length(chosen.span_s),
            step=1.0,
            key=f"identity_duration::{match_id}::{chosen.track_id}",
        )
        if not source.exists():
            st.caption(f"`{source}` is not available on this machine, so the clip cannot be cut.")
            return
        with cut_col:
            st.write("")  # line the button up with the slider's caption
            cut = st.button(
                "Cut the clip, centered on this player",
                key=f"identity_cut::{match_id}",
                disabled=jobs.JOBS.has_active(match_id, kinds=("player_clip",)),
                help=(
                    "The clip is cut from the analyzed video with a moving crop that follows this appearance, sized "
                    "to the player's own height. Where the appearance has a gap of more than a couple of seconds "
                    "the clip stops rather than panning across a stretch nobody saw."
                ),
            )
        clip_name = _centered_clip_name(chosen.track_id, chosen.first_t, duration)
        centered = library.highlights_dir(match_id) / "centered" / clip_name
        legacy = library.highlights_dir(match_id) / "centred" / clip_name  # clips cut before the American spelling
        target = centered if centered.exists() or not legacy.exists() else legacy
        if cut and not target.exists():
            jobs.JOBS.submit(
                kind="player_clip",
                label=f"Clip for track {chosen.track_id}",
                match_id=match_id,
                params={
                    "match_id": match_id,
                    "library_root": _library_root(library),
                    "source": str(source),
                    "track_id": chosen.track_id,
                    "first_t": chosen.first_t,
                    "start_s": float(segment.meta.get("start_s", 0.0)),
                    "duration_s": float(duration),
                    "target": str(target),
                    "width": 1280,
                },
            )
            st.success("Clip queued - progress appears under Background jobs.")
        if target.exists():
            # A path, not the media URL the component would be given: `st.video` registers the file itself, and a
            # scheme-less ``/media/<hash>.mp4`` string is read as a *local* path, which fails with
            # MediaFileStorageError. The endpoint URL is only for callers that fetch it themselves (the replay
            # component's clip pane).
            st.video(str(target))
            plan_path = target.with_suffix(target.suffix + ".plan.json")
            try:
                note = json.loads(plan_path.read_text())
            except (OSError, json.JSONDecodeError):
                note = {}
            covers = float(note.get("duration") or duration)
            st.caption(
                f"`{target.name}` - {_clock(chosen.first_t)} to {_clock(chosen.first_t + covers)}, cropped to "
                "follow this player, cut from the analyzed video."
                + (
                    f" The clip stops there rather than following across a stretch nobody saw ({note['truncated']})."
                    if note.get("truncated")
                    else ""
                )
            )
        else:
            st.caption("Press **Cut the clip** to produce a clip cropped around this player.")


def replay_section(
    library: MatchLibrary, match_id: str, segment, segment_dir: Path, video: str, calibration
) -> None:
    """The animated pitch view plus everything known about each player.

    Playback is entirely client-side (the component fetches the replay JSON once), so scrubbing and play/pause cost
    no Streamlit round trips. What *is* Python-side: the annotated player table, the roster editor and the two
    background scans (shirt numbers, ball) - the table, the editor and the number scan share the merge rule
    "manual beats scan", and only the ball scan feeds the *map* itself, through the replay payload.
    """
    replay_path = library.path(match_id) / "replay.json"
    if not replay_path.exists():
        st.info("The animated replay is built together with the report above - press **Build report**.")
        return
    _apply_event_picker_repair(match_id)
    replay = _load_replay_cached(str(replay_path), replay_path.stat().st_mtime_ns)
    record = library.load(match_id)
    team_names = record.team_names
    # The payload on disk carries whatever the names were when the report was built; the record is where renaming
    # lands, so the loaded copy is brought up to date rather than the table disagreeing with the map beside it.
    replay["team_names"] = list(team_names)
    jerseys = library.load_jerseys(match_id)
    roster = library.load_roster(match_id)
    roster_library = RosterLibrary()
    track_ids = [int(player["track_id"]) for player in replay["players"]]
    suggestions = {int(track): entry for track, entry in (jerseys.get("suggestions") or {}).items()}
    team_of = {int(player["track_id"]): int(player["team"]) for player in replay["players"]}
    # Names come from the team rosters linked to the match - shared across matches, so a squad is typed once. A
    # link whose file has gone missing simply contributes no names (the number merge is about display, not data).
    numbers = merge_numbers(
        track_ids,
        auto=suggestions,
        manual=roster,
        team_of=team_of,
        team_rosters=team_rosters_for(roster_library, record.team_rosters),
    )
    track_stats = {int(player["track_id"]): player["stats"] for player in replay["players"]}

    def player_label(track: int) -> str:
        entry = numbers.get(track, {})
        who = f"#{entry['number']}" if entry.get("number") else f"track {track}"
        if entry.get("name"):
            who += f" {entry['name']}"
        team = team_of.get(track, -1)
        return f"{who} ({team_name(team, team_names)})"

    selected = st.selectbox(
        "Highlight a player in the replay",
        [-1] + sorted(track_ids, key=lambda track: -track_stats[track]["observations"]),
        format_func=lambda track: "no highlight" if track < 0 else player_label(track),
        key=f"replay_selected::{match_id}",
    )
    # The timeline strip above the pitch draws the momentum curve and every event. The events are the same log the
    # table below shows (manual tags and detected candidates), and the momentum is the report's own per-minute
    # buckets; both are small, so they travel as component arguments and update without rebuilding the replay.
    #
    # The three things on the strip keep three different clocks, and all of them are brought onto the strip's own
    # (0-based over the analyzed window) here, in one place:
    # * an event's ``time_s`` is a time on the recording it was found in - the combined game for a detected event,
    #   a single camera file for a whistle. The analyzed window starts at ``segment.meta["start_s"]`` of *its own*
    #   video, so a candidate from another recording is mapped through that file's own clock, or clipped off the
    #   strip when it does not lie inside the window at all.
    # * the momentum buckets are keyed by minutes of the *recording* (Stage B's own clock, the same one the
    #   highlight clips cut with), so they are shifted onto the window's clock like an event from this recording
    #   is - on the real game the window starts at 9:00, and without the shift the curve sat nine minutes to the
    #   right of the events it belongs beside.
    # * the half-time mark is a game-clock time, so it is mapped the same way as an event.
    event_log = library.events(match_id)
    report_payload = st.session_state.get("report") or _report_payload(library, match_id) or {}
    game_record_for_video = game_lib.find_for_video(video)
    half_minute = None
    if game_record_for_video is not None and game_record_for_video.bounds() is not None:
        half_minute = game_record_for_video.bounds()[1] / 60.0
    window_start = float(segment.meta.get("start_s", 0.0))
    strip_events: list[dict] = []
    for index, event in enumerate(event_log.events):
        # The strip's clock is the analyzed window's own (0 at its first frame), so an event's time is translated
        # out of its recording's clock onto the game's and then onto the window's. One helper does the first step,
        # so the strip, the table and the half labels cannot disagree.
        on_game = (
            game_lib.game_time(game_record_for_video, event.time_s, event.video)
            if game_record_for_video is not None and event.video
            else event.time_s
        )
        if on_game is None:
            continue  # found in a recording that is not part of this game
        strip_t = on_game - window_start
        if strip_t < -1.0 or strip_t > float(replay.get("duration_s", 0.0)) + 1.0:
            continue  # a moment from another recording that is not part of this analyzed window
        strip_events.append(
            {
                **event.to_json(),
                # ``time_s`` is the strip's clock; the log's own second rides along so a right-click delete can
                # prove the row it names is still the row the marker described.
                "time_s": round(max(0.0, strip_t), 3),
                "index": index,
                "log_time_s": round(float(event.time_s), 3),
            }
        )
    strip_half_minute = None
    if half_minute is not None:
        strip_half_minute = half_minute * 60.0 - window_start
        if strip_half_minute < 0 or strip_half_minute > float(replay.get("duration_s", 0.0)):
            strip_half_minute = None
    # The moment to watch with the footage: the picker translates the chosen event onto the strip's clock and the
    # button asks the component to jump the animation there. The stream pane follows the animation, so one jump
    # moves both clocks.
    seek_to_s, seek_seq, review_index = _footage_moment(
        match_id, float(replay.get("duration_s", 0.0)), event_log, game_record_for_video, window_start
    )
    # The pane needs the stream server (Streamlit cannot serve a route of its own). Say whether it is up - and
    # offer the one click that starts it - instead of letting the pane fail quietly.
    stream_port = configured_port()
    if is_reachable(stream_port):
        st.caption(f"Footage stream: connected on port {stream_port}.")
    else:
        st.caption(
            f"Footage stream: not running. Start `scripts/run_match_stream.py --port {stream_port}` (or press the "
            "button) to play the annotated footage beside the animation."
        )

        def _start_stream() -> None:
            subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_match_stream.py"),
                    "--port",
                    str(stream_port),
                    "--match",
                    match_id,
                ],
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            st.session_state["replay_flash"] = ("success", f"Footage stream starting on port {stream_port}.")

        st.button("Start the footage stream", key=f"start_stream::{match_id}", on_click=_start_stream)
    replay_action = replay_view(
        _replay_media_url(replay_path),
        numbers,
        selected,
        team_names,
        key=f"replay::{match_id}",
        events=strip_events,
        momentum=momentum_on_window_clock(report_payload.get("momentum") or {}, window_start),
        half_minute=None if strip_half_minute is None else strip_half_minute / 60.0,
        match_id=match_id,
        stream_offset_s=window_start,
        stream_port=stream_port,
        stream_base=configured_base(),
        seek_to_s=seek_to_s,
        seek_seq=seek_seq,
        ack_tag_seq=int(st.session_state.get(f"replay_tag_ack::{match_id}", 0)),
        ack_delete_seq=int(st.session_state.get(f"replay_delete_ack::{match_id}", 0)),
    )
    # Two gestures write to the event log: a tag from the bar under the pitch (stamped with the playback's own
    # second) and a right-click delete on the strip. Both redraw in the same gesture via a rerun.
    _apply_timeline_deletes(library, match_id, replay_action)
    _apply_quick_tags(library, match_id, video, window_start, float(replay.get("duration_s", 0.0)), replay_action)
    has_ball = any(entry is not None for entry in (replay.get("ball") or []))
    excluded = int(replay.get("bystanders_excluded") or 0)
    st.caption(
        "Press play or drag the timeline. Player markers wear each team's measured kit color; the camera icon "
        "points at the spot the camera was aimed at - "
        + (
            "the ball proxy used when the ball scan has not found the ball. The white football is the ball the "
            "scan tracked: drawn where a detector saw it, a dashed ring where the position is a short forecast "
            "across a missed frame."
            if has_ball
            else "the best ball proxy this footage allows (the gimbal follows the ball; the scan below can track "
            "the ball itself)."
        )
        + " Trails and a pitch-usage heat map (all players, or just the selected one) are toggled above the map. "
        "Numbers are read off the footage automatically; names come from the team rosters below. A track with "
        "neither shows its track id once it lasts 12 s, when the debug layer is on."
        + (
            f" {excluded} bystander track(s) - touchline coaches, photographers, spectators - are excluded from "
            "the field of play: they never cover ground the way a player does."
            if excluded
            else ""
        )
    )
    # Everything that used to be Step 4 - the tag review, the detectors' controls and the highlight reels - lives
    # directly under the playback now, so tagging and reviewing happen while the game is on screen.
    events_section(library, match_id, segment_dir, video, calibration, review_index)
    only_major = st.checkbox(
        "Show only main tracks (seen for 6 s or more)",
        value=True,
        key=f"replay_major::{match_id}",
        help=(
            "The camera follows the ball, so players leave and re-enter the shot constantly and each visit becomes "
            "its own track (fragments nearby are stitched back together automatically). Short tracks are still "
            "drawn in the replay - untick to see them in the table too."
        ),
    )
    table = player_table_rows(replay, numbers)
    if only_major:
        table = table[table["seen (s)"] >= 6.0]
    st.dataframe(table, hide_index=True, width="stretch")

    identity_section(library, match_id, segment, replay, numbers)

    with st.expander("Team rosters - shirt numbers and player names", expanded=True):
        _team_roster_editors(library, roster_library, match_id, record, replay, numbers, team_names)

    with st.expander("Correct one track (manual override)"):
        st.caption(
            "The team rosters above connect each number with a name; use this table to correct a single track "
            "when the *number* itself is wrong. A manual entry here wins over both the scan and the team roster, "
            "so use it sparingly - and note it belongs to this match's track ids only. Distances, speeds and "
            "touches are in the table above; a touch means the player was the nearest to the ball - the scanned "
            "ball in the picture when available, else the camera aim point - and inside the possession gate."
        )
        editors = []
        for player in replay["players"]:
            track = int(player["track_id"])
            entry = numbers.get(track, {})
            editors.append(
                {
                    "track": track,
                    "team": team_name(player["team"], team_names),
                    "number": entry.get("number"),
                    "name": entry.get("name") or "",
                    "source": entry.get("source") or "unassigned",
                }
            )
        edited = st.data_editor(
            pd.DataFrame(editors),
            column_config={
                "track": st.column_config.NumberColumn("track", disabled=True),
                "team": st.column_config.TextColumn("team", disabled=True),
                "number": st.column_config.NumberColumn("number", min_value=1, max_value=99, step=1),
                "name": st.column_config.TextColumn("name", max_chars=30),
                "source": st.column_config.TextColumn("source", disabled=True),
            },
            hide_index=True,
            width="stretch",
            key=f"roster_editor::{match_id}",
        )
        if st.button("Save shirt numbers", key=f"save_roster::{match_id}"):
            new_roster: dict[int, dict] = {}
            for row in edited.to_dict("records"):
                number = row.get("number")
                number = int(number) if number is not None and not pd.isna(number) else None
                name = str(row.get("name") or "").strip()
                if number or name:
                    new_roster[int(row["track"])] = {"number": number, "name": name}
            library.save_roster(match_id, new_roster)
            st.session_state["replay_flash"] = ("success", f"Saved {len(new_roster)} player identit(ies).")
            st.rerun()

    show_flash("replay_flash")


def events_section(
    library: MatchLibrary,
    match_id: str,
    segment_dir: Path,
    video: str,
    calibration,
    review_index: int | None,
) -> None:
    """Tag review, the event detector and the highlight reels - directly under the playback.

    This is where Step 4 used to live. Tagging itself is the playback's own tag bar (a button stamps the second
    being watched, so no round trip can move the tag); what remains Python-side is everything around it: the
    review queue of detected candidates, the event detector that fills it, and the reel exports. All of it sits
    below the pitch so a coach can watch, tag, review and cut without leaving the game.

    ``review_index`` is the moment the jump picker above the pitch is on: the verdict buttons act on it, so
    choosing a moment to watch is also choosing it to review, and marking it moves the picker to the next
    unreviewed candidate.
    """
    events = library.events(match_id)
    record = library.load(match_id)
    team_names = record.team_names
    # The game's marks turn an event's own recording time into a half and a match clock. They are re-found here
    # rather than read from a module global, because this section must stand on its own once the report exists.
    game_record_here = game_lib.find_for_video(video)
    game_marks_here = game_record_here.bounds() if game_record_here is not None else None

    def _match_clock_for(event: Event) -> str | None:
        if game_record_here is None or game_marks_here is None:
            return None
        on_game = game_lib.game_time(game_record_here, event.time_s, event.video or game_record_here.output)
        if on_game is None:
            return None
        half = game_record_here.half_of(on_game)
        if half is None:
            return None
        return _clock(on_game - game_marks_here[0]) if half == 1 else f"{_clock(on_game - game_marks_here[1])} (2H)"

    # These run before the next run's body, so the table below is rebuilt with the change already in it. Doing the
    # work inline instead would need a st.rerun() to redraw it, which is a second full run for one button.
    def _discard_detected() -> None:
        log = library.events(match_id)
        removed = log.discard_detected()
        library.save_events(match_id, log)
        st.session_state["events_flash"] = (
            "success",
            f"Discarded {removed} auto-detected event(s); {len(log.events)} manual tag(s) left.",
        )

    def _detect_events() -> None:
        """Infer goals, shots, corners, penalties, clearances and tackles from the ball and player tracks.

        The ball scan and the report are the inputs, so both have to exist first; the shared runner is what the
        background "run all detections" pass calls too, so the two cannot disagree. Its output is added as review
        candidates (``source="ball"``), exactly like the whistle scan's, so the same verdict buttons and the same
        "discard detected" button apply.
        """
        try:
            segment_here, q_here, focal_here = segment_and_poses_cached(
                str(segment_dir), segment_fingerprint(segment_dir, stage_a.completed_chunks(segment_dir))
            )
        except Exception as exc:  # a partially written segment must not take the page down
            st.session_state["events_flash"] = ("warning", f"Could not read the segment results: {exc}")
            return
        try:
            result = detect_run.run_detection(
                library,
                match_id,
                segment_dir,
                video=str(Path(video).resolve()),
                half_bounds=game_marks_here,
                segment=segment_here,
                poses=(q_here, focal_here),
            )
        except detect_run.MissingInput as exc:
            st.session_state["events_flash"] = ("warning", str(exc))
            return
        stale = f", {result['dropped']} stale removed" if result["dropped"] else ""
        st.session_state["events_flash"] = (
            "success",
            f"Detected {result['detected']} event(s); {result['added']} new{stale}. Review them below.",
        )

    def _mark_verdict(index: int, verdict: str) -> None:
        """Record the review verdict for the moment the picker showed, then move on to the next one.

        The index arrives as a button argument rather than being read back out of session state: a selectbox's
        state is only guaranteed to be the option where it has been able to match the frontend's label against the
        options, and this has to work either way. The callback runs before the next run's body, so the log is read
        from disk here - the list below is then rebuilt with the new verdict already in it, in one run.

        Moving the picker on is the point of a review queue: ninety candidates one at a time is only bearable if
        deciding does not also mean finding the next row by hand.
        """
        log = library.events(match_id)
        try:
            number = int(index)
            log.set_verdict(number, verdict)
        except (IndexError, TypeError, ValueError):
            st.session_state["events_flash"] = ("warning", "That moment is not in the list any more - pick it again.")
            return
        library.save_events(match_id, log)
        what = {VERDICT_TRUE: "a true positive", VERDICT_FALSE: "a false positive"}.get(verdict, "unreviewed")
        st.session_state["events_flash"] = ("success", f"Marked #{number + 1} as {what}.")
        if verdict in (VERDICT_TRUE, VERDICT_FALSE):
            order = list(range(number + 1, len(log.events))) + list(range(0, number + 1))
            next_unreviewed = next((i for i in order if log.events[i].verdict == ""), None)
            if next_unreviewed is not None:
                # The next unreviewed moment, over the *current* list, so the review keeps moving without the
                # reviewer hunting for the next row. The picker above the pitch is moved, so the next press of
                # its jump button plays that moment with its clip.
                st.session_state[f"pitch_clip_event::{match_id}"] = next_unreviewed

    def _discard_false() -> None:
        log = library.events(match_id)
        removed = log.discard_false()
        library.save_events(match_id, log)
        st.session_state["events_flash"] = (
            "success",
            f"Discarded {removed} rejected candidate(s); {len(log.events)} event(s) left.",
        )

    swap_key = f"swap_attack::{match_id}"

    def _set_swap_attack() -> None:
        """Persist the user's correction and flip the directions already written into the replay.

        The record is re-read first: this page is not the only writer a match can have, and saving a stale copy
        would clobber another edit. Inverting the replay's ``attack.directions`` makes the pitch arrows and the
        event table's aim change on the next run without rebuilding the report; a future rebuild and event run
        both read the saved record setting and produce the same sides.
        """
        swap = bool(st.session_state.get(swap_key, False))
        fresh = library.load(match_id)
        if bool(fresh.swap_attack) == swap:
            return
        fresh.swap_attack = swap
        library.save(fresh)
        replay = library.load_replay(match_id)
        if replay and isinstance(replay.get("attack"), dict):
            directions = replay["attack"].get("directions") or []
            replay["attack"]["directions"] = [
                [(-int(direction) if direction else direction) for direction in pair] if pair else pair
                for pair in directions
            ]
            library.save_replay(match_id, replay)
        st.session_state["events_flash"] = (
            "success",
            "Attack directions swapped: the pitch arrows, the aim column and future detections now use the other "
            "ends."
            if swap
            else "Attack directions restored to the measured sides.",
        )

    def _remove_event(index: int) -> None:
        """Delete one row from the event log and keep every index-based picker on a valid row."""
        log = library.events(match_id)
        try:
            removed = log.remove(int(index))
        except (IndexError, TypeError, ValueError):
            st.session_state["events_flash"] = (
                "warning",
                "That event is not in the list any more - pick it again.",
            )
            return
        library.save_events(match_id, log)
        _keep_event_pickers_valid(match_id, log, {int(index)})
        st.session_state["events_flash"] = (
            "success",
            f"Removed {removed.type} at {_clock(removed.time_s)} from the timeline.",
        )

    st.subheader("Events and review")
    show_flash("events_flash")
    st.caption(
        "The tag bar under the playback stamps an event on the second being watched. Whistles are found in the "
        "audio and offered as candidates; goals, shots, corners, penalties, clearances and tackles can also be "
        "inferred from the ball scan and the player tracks once both exist. Saves and blocks stay yours to tag: a "
        "keeper's save and a shot wide look the same to a ball track. The detector is deliberately strict - it "
        "would rather miss a call than invent one - so every candidate carries a note saying what it measured, "
        "and the weak ones are yours to reject. Each row also says whether the act was attacking or defensive by "
        "its type, and - for the attacks - which goal the acting team was attacking then, from the same measured "
        "directions the arrows on the pitch use."
    )
    st.checkbox(
        "Swap attacking and defending teams",
        value=bool(record.swap_attack),
        key=swap_key,
        on_change=_set_swap_attack,
        help=(
            "Inverts every half's measured attack direction. Use it when the arrows, the aim column or the "
            "detector's clearance sides have the teams the wrong way round; pressing it again restores the "
            "measured sides. The correction is saved with the match, so a rebuilt report and a re-run detector "
            "use the same sides."
        ),
    )
    # The detector that adds to the review list sits with it: press it, then review what it adds below.
    st.button(
        "Detect events from the ball and player tracks",
        on_click=_detect_events,
        key=f"detect_events::{match_id}",
        help=(
            "Reads the ball scan and the player tracks to infer goals, shots, corners, penalties, clearances "
            "and tackles. Needs the report (Step 3) and the ball scan. The results are review candidates, like "
            "the whistle scan's - check them and mark the false ones."
        ),
    )
    if events.events:
        counts = events.review_counts()
        hide_false = st.checkbox(
            "Hide the rejected candidates",
            value=False,
            key=f"hide_false::{match_id}",
            help=(
                "Only changes what the table shows - the verdicts themselves are kept, and so is everything you "
                "have not reviewed yet."
            ),
        )
        shown = [e for e in events.events if not (hide_false and e.verdict == VERDICT_FALSE)]
        # The video path is not in the table: it is long, it is the same file for every candidate, and it is
        # already named under the player when a candidate needs a different one from the selected video. The team
        # is the name the user gave it, so a row reads the way the report does.
        event_rows = [
            {
                **{key: value for key, value in event.to_json().items() if key != "video"},
                "team": "unspecified" if event.team < 0 else team_name(event.team, team_names),
                # Whether the act was an attack or a defense is the event type's own meaning - no game clock
                # needed, so the column is there for every archive.
                "play": event_play(event.type),
            }
            for event in shown
        ]
        if game_marks_here is not None and game_record_here is not None:
            # The half is a question about the game clock, so it comes from the marks, not from the analyzed
            # window: an event outside kick-off/full-time is labeled "-" rather than guessed at. Each event is
            # translated out of its own recording's clock first - a whistle candidate's seconds are seconds of a
            # single camera file, and 5 s of that clip is 30 minutes into the match.
            labels = game_lib.half_labels_for_events(game_record_here, shown)
            event_rows = [{**row, "half": label} for row, label in zip(event_rows, labels)]
            # A candidate's time is a time on *its own recording* - a camera clip's 5:00 is the game's 35:00 - so
            # the table also shows the match time a coach would read out. Events outside the marked game (a
            # whistle scanned on a clip that is not part of it) keep "-".
            event_rows = [
                {**row, "match clock": _match_clock_for(event) or "-"} for row, event in zip(event_rows, shown)
            ]
            # Whether the act was an attack or a defense, and - for the attacks - which goal mouth the acting
            # team was attacking then. The play is the event type's own meaning; the aim comes from the replay
            # payload's measured per-half directions, so the table, the arrows on the pitch and the detector
            # cannot disagree about who was going which way. A payload from before directions existed - or an
            # event whose own recording cannot be placed on the game's clock - leaves the aim blank.
            attack = (library.load_replay(match_id) or {}).get("attack") or {}
            directions_by_half = attack.get("directions") or []

            def _half_of_event(event: Event) -> int | None:
                on_game = game_lib.game_time(
                    game_record_here, event.time_s, event.video or game_record_here.output
                )
                return None if on_game is None else game_record_here.half_of(on_game)

            aims = []
            for event in shown:
                half = _half_of_event(event)
                # ``directions`` is one entry per half in order, and the game's halves are numbered from 1.
                pair = (
                    directions_by_half[half - 1]
                    if half is not None and 0 <= half - 1 < len(directions_by_half)
                    else None
                )
                direction = pair[event.team] if pair is not None and 0 <= event.team < len(pair) else None
                aims.append(aiming_goal(direction) if event_play(event.type) == "attacking" else "")
            event_rows = [{**row, "aim": aim} for row, aim in zip(event_rows, aims)]
        st.dataframe(pd.DataFrame(event_rows), hide_index=True, width="stretch")
        st.caption(
            f"Reviewed: {counts[VERDICT_TRUE]} true · {counts[VERDICT_FALSE]} false · "
            f"{counts['unreviewed']} still to look at."
        )
        with st.expander("Remove an individual event", expanded=False):
            st.caption(
                "The fastest way is the timeline strip above: right-click any marker to delete it. This form is "
                "for deleting one row from `events.json` by name - a manual tag is gone until you tag it again, "
                "and a detected candidate can come back if the detector is run again. This cannot be undone."
            )
            remove_index = st.selectbox(
                "Event to remove",
                range(len(events.events)),
                format_func=lambda index: _event_label(
                    events.events[index], index, _match_clock_for(events.events[index])
                ),
                key=f"event_remove::{match_id}",
            )
            confirmed = st.checkbox(
                "Confirm removal",
                value=False,
                key=f"event_remove_confirm::{match_id}",
            )
            st.button(
                "Remove this event",
                key=f"remove_event::{match_id}",
                disabled=not confirmed,
                on_click=_remove_event,
                args=(int(remove_index),),
                help="Delete the selected event from the match's event log. This cannot be undone.",
            )
        # Review the moment the jump picker above is on. Watching it, judging it and moving on all happen here,
        # next to the buttons that tag the same list - no separate preview panel to find and wait on a clip cut.
        if review_index is not None and 0 <= review_index < len(events.events):
            picked = events.events[review_index]
            when = _match_clock_for(picked) or _clock(picked.time_s)
            verdict_columns = st.columns([1, 1, 1, 2])
            with verdict_columns[0]:
                st.button(
                    "True positive",
                    key=f"verdict_true::{match_id}",
                    on_click=_mark_verdict,
                    args=(review_index, VERDICT_TRUE),
                    disabled=picked.verdict == VERDICT_TRUE,
                    help="A real stoppage: keep it, and let the reels use it.",
                )
            with verdict_columns[1]:
                st.button(
                    "False positive",
                    key=f"verdict_false::{match_id}",
                    on_click=_mark_verdict,
                    args=(review_index, VERDICT_FALSE),
                    disabled=picked.verdict == VERDICT_FALSE,
                    help="Not a real event: rejected candidates are left out of the reels.",
                )
            with verdict_columns[2]:
                st.button(
                    "Clear verdict",
                    key=f"verdict_clear::{match_id}",
                    on_click=_mark_verdict,
                    args=(review_index, ""),
                    disabled=not picked.verdict,
                    help="Back to unreviewed.",
                )
            with verdict_columns[3]:
                st.caption(
                    f"Reviewing #{review_index + 1}: {picked.type} at {when} - press **Jump to this moment** "
                    "above to watch it with the footage, then mark it here."
                )
        detected = events.detected()
        if detected:
            st.button(
                f"Discard {len(detected)} auto-detected event(s)",
                key="discard_detected_events",
                help=(
                    "Remove the candidates the scans added and keep the tags you made yourself. Re-running a scan "
                    "finds them again, so nothing is lost that cannot be brought back."
                ),
                on_click=_discard_detected,
            )
        if counts[VERDICT_FALSE]:
            st.button(
                f"Discard the {counts[VERDICT_FALSE]} rejected candidate(s)",
                key="discard_false_events",
                help=(
                    "Drop the candidates you marked as false positives and keep everything else, including "
                    "the ones you have not reviewed yet. A re-scan brings rejected candidates back."
                ),
                on_click=_discard_false,
            )
    else:
        st.info(
            "No events yet - tag one from the buttons under the playback, or press **Detect events from the ball "
            "and player tracks** above."
        )

    payload = st.session_state.get("report") or _report_payload(library, match_id)
    st.subheader("Highlight reels")
    if not events.events:
        st.info("Tag an event, or run the detector above, before cutting highlights.")
    else:
        moments = build_moments(events.events, (payload or {}).get("momentum"))
        st.write(f"{len(moments)} candidate moment(s).")
        st.caption("Reels keep the source's resolution, frame rate and picture quality.")
        for tier, seconds in TIER_SECONDS.items():
            reel = select_reel(tier, moments, match_duration_s=probe_cached(video).duration_s)
            reel_busy = any(
                job.get("kind") == "reel"
                and str((job.get("params") or {}).get("tier")) == tier
                for job in jobs.JOBS.active(match_id=match_id)
            )
            tier_col, button_col = st.columns([3, 1])
            with tier_col:
                st.write(f"**{tier}** - {len(reel.moments)} moment(s), {reel.duration_s:.0f}s of a {seconds:.0f}s target")
            with button_col:
                export_clicked = st.button(
                    f"Export {tier}", key=f"export_{tier}", disabled=not reel.moments or reel_busy
                )
            if export_clicked:
                jobs.JOBS.submit(
                    kind="reel",
                    label=f"{tier} reel",
                    match_id=match_id,
                    params={
                        "match_id": match_id,
                        "video": str(video),
                        "tier": tier,
                        "library_root": _library_root(library),
                    },
                )
                st.success(f"{tier} reel queued - progress appears under Background jobs.")


def show_flash(key: str) -> None:
    """Show a message that a save-then-rerun action left behind.

    ``st.rerun()`` throws away the elements of the run it was called from, so a ``st.success`` printed immediately
    before it is never seen at all. Actions that save and then reload stash ``(level, text)`` under ``key`` and call
    this as the first thing on the next run.
    """
    flash = st.session_state.pop(key, None)
    if not flash:
        return
    level, message = flash
    if level == "warning":
        st.warning(message)
    else:
        st.success(message)


# --------------------------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------------------------
st.title("Match analysis")
st.caption(
    "Single gimbal camera: the camera's own motion is recovered first, so player positions can be placed on the pitch "
    "even though the camera pans and zooms to follow the ball."
)
# Heavy work runs outside this script; this line stays visible when another match's job is running.
job_ui.running_jobs_badge()

# There is no options sidebar any more: every clip the page cuts or plays is built at full detail, because the
# cheap preview modes were there to make reviewing a scan bearable and the scan's candidates are now reviewed
# straight off the playback instead of as cut clips.

# The library spans every analysis directory beside the footage - one per match folder, whichever disk it is on -
# plus the repository's old data/matches archives until they are migrated.
library = MatchLibrary()

# Everything the page keeps in the session belongs to one match: the calibration just fitted, the report just
# built, the landmarks behind the residual table. Selecting another archive has to drop all of it - otherwise the
# previous match's numbers are quietly shown for the new one, which is what makes an archive look like it loaded
# when it did not.
MATCH_SCOPED_STATE = (
    "calibration",
    "report",
    "calib_points",
    "calib_landmarks",
    "calib_next_pid",
    "calib_center",
    "calib_zoom",
    "calib_reference",
    "calib_clicks_key",
    "calib_dropped_note",
    "calib_fit_error",
    "events_flash",
    "step3_flash",
)

st.header("Select session")
match_ids = library.list_ids()

# Two compact choices: which analysis to reopen, and which footage folder this session works on. The videos
# themselves are chosen in Step 1, right where they are used - picking a video up here and the same video again as
# a "clip" below was one selection too many. The analysis lives beside the footage either way.
with st.container(border=True):
    match_col, folder_col = st.columns([1, 1])
    with match_col:
        # A match saved in the last run asks to be selected. A widget's state can only be set before that widget
        # exists in the run, and the save button runs after this module has been drawn, so it leaves the id here for
        # the next run to pick up.
        pending = st.session_state.pop("archive_select_pending", None)
        if pending in match_ids:
            st.session_state["archive_selection"] = pending
        # A match whose analysis was cleared or moved out from under the page must not keep this selectbox on a
        # value it no longer offers (Streamlit breaks on a stored value that is not among the options): the page
        # resets to "(new analysis)", which is exactly what a cleared analysis should mean.
        if st.session_state.get("archive_selection") not in match_ids + ["(new analysis)"]:
            st.session_state.pop("archive_selection", None)
        selection = st.selectbox(
            "Match",
            match_ids + ["(new analysis)"],
            index=0,
            key="archive_selection",
            help=(
                "One entry per match that has an analysis saved beside its footage (and the repository's older "
                "archives until they are moved out). Choosing a video that already has an analysis opens it "
                "automatically, so this list is mainly for going back to a match."
            ),
        )
        match_id = None if selection == "(new analysis)" else selection
        previous_scope = st.session_state.get("archive_scope")
        just_selected = match_id is not None and previous_scope != match_id
        if previous_scope != match_id:
            st.session_state["archive_scope"] = match_id
            for scoped in MATCH_SCOPED_STATE:
                st.session_state.pop(scoped, None)

    with folder_col:
        video_options = discover_videos()
        if not video_options:
            searched = ", ".join(str(root) for root in video_roots())
            st.error(f"No video files found under {searched}.")
            st.stop()

        # Selecting an analysis has to bring back the footage it was recorded from, or it cannot be reopened: the
        # picker would stay on whichever video happened to be newest and Step 2 would find no segment belonging to
        # the match. The recorded path is resolved against the analysis directory, so a match folder that has been
        # moved still opens its own video.
        pinned_source: str | None = None
        default_video_folder = "All folders"
        match = match_footage(library, match_id) if match_id is not None else None
        if match is not None:
            if match.videos:
                pinned_source = match.videos[0]
                default_video_folder = str(Path(match.videos[0]).parent)
            if match.missing:
                st.warning(
                    "Some footage this match used is not reachable: "
                    + ", ".join(Path(path).name for path in match.missing)
                )
            # Seed the picker on selection, before its widget is created: an existing match reopens with the
            # exact recordings it was built from (a game's clips, not the virtual merged output name).
            if just_selected:
                st.session_state[f"game_clips::{match_id}"] = list(match.videos)

        # The recordings live in one folder per match (and matches for several teams sit side by side), so the
        # folder narrows Step 1's video list instead of scrolling one flat list from every disk.
        folders = sorted({str(Path(option).parent) for option in map(str, video_options)})
        folder_choices = ["All folders", *folders]
        folder_index = (
            folder_choices.index(default_video_folder) if default_video_folder in folder_choices else 0
        )
        # A widget's stored value survives a match switch, so put the match's own folder back on selection.
        if just_selected and default_video_folder in folder_choices:
            st.session_state[f"video_folder::{match_id}"] = default_video_folder
        folder = st.selectbox(
            "Folder",
            folder_choices,
            index=folder_index,
            format_func=lambda entry: entry if entry == "All folders" else _location_label(Path(entry)),
            key=f"video_folder::{match_id}",
            help="Step 1 lists the videos in the chosen folder; the analysis is saved beside them.",
        )

        with st.expander("...or open one video by its path"):
            manual = st.text_input(
                "Path to the video", placeholder="/path/to/match.MP4", key=f"manual_video::{match_id}"
            )

    st.caption(
        f"Everything for this match is saved with the footage: `{_location_label(library.path(match_id))}`"
        if match_id
        else "Pick a folder here; its videos are chosen in Step 1 below, and the analysis is saved beside them."
    )

# --------------------------------------------------------------------------------------------------------------
# Step 1 - the heavy pass
# --------------------------------------------------------------------------------------------------------------
st.header("Step 1 - Run the heavy pass")
show_flash("session_flash")
show_flash("step1_flash")

# --- the session's footage: the video, or the game's clips ----------------------------------------------------
# One chooser, right where the footage is used. One recording is analyzed as itself; several are one game - the
# camera names each clip by its start time, so their order is read from the names - and the button below records
# the manifest that ties them together. Nothing is copied or re-encoded, and no merged video is produced. A game
# that already exists is recognized from any of its clips, so picking one reopens it.
visible = [
    path
    for path in video_options
    if folder == "All folders" or str(Path(path).parent) == folder
]
# Whatever folder is shown, the open match's own footage and the last choice must stay selectable - a multiselect
# whose stored value is not among its options would not come back.
for keep in [pinned_source, *(st.session_state.get(f"game_clips::{match_id}") or [])]:
    if keep and keep not in {str(path) for path in visible}:
        visible.insert(0, Path(keep))
option_strings = [str(path) for path in visible]


def _open_game_for(picked_paths: list[str]):
    """The first picked recording that belongs to a game, and that game; ``(None, None)`` when none does.

    A game's manifest stores its clips beside the footage, so any clip of the game identifies it: that is how a
    session finds its game without a separate selection.
    """
    for path in picked_paths:
        found = game_lib.find_for_video(path)
        if found is not None:
            return path, found
    return None, None


def _open_analysis_beside_video() -> None:
    """A single picked video that already has an analysis saved beside it opens that analysis automatically.

    Only while no analysis is selected, and only for a single pick: several picks are a game being assembled,
    and switching the archive out from under that would fight the user.
    """
    if match_id is not None:
        return
    picked_now = st.session_state.get(f"game_clips::{match_id}") or []
    if len(picked_now) != 1:
        return
    found = _analysis_for_video(str(picked_now[0]))
    if found:
        st.session_state["archive_select_pending"] = found


# The widget reports its value only after it has been drawn, so before the expander the kept selection from the
# last run stands in for it - that is what lets the expander know whether a game is already open on the first
# render. A fresh session starts on the newest video of the folder, and when that video belongs to a game, on
# the whole game (all its clips) - so the picker and the marking step below agree about what is being worked on.
stored_picks = st.session_state.get(f"game_clips::{match_id}")
if stored_picks is not None:
    seed_picks = list(stored_picks)
elif match_id is not None or not option_strings:
    seed_picks = []
else:
    seed_picks = [option_strings[0]]
    found_via_newest = game_lib.find_for_video(option_strings[0])
    if found_via_newest is not None:
        in_options = set(option_strings)
        seed_picks = [str(clip.path) for clip in found_via_newest.clips if str(clip.path) in in_options] or seed_picks
if match_id is not None:
    game_record = match.game if match is not None else None
    _seed_source = (
        (match.analysis_source if match is not None else None)
        or pinned_source
        or (seed_picks[0] if seed_picks else None)
    )
    if game_record is None and _seed_source:
        game_record = game_lib.find_for_video(_seed_source)
else:
    _seed_source, game_record = _open_game_for(seed_picks)
game_window_selection: str | None = None

with st.expander("Footage (one video, or the game's clips)", expanded=game_record is None):
    st.caption(
        "Pick one recording to analyze it as it stands, or all of a game's camera files - in any order, their "
        "own timestamps put them in playing order. The game is ready in an instant: the manifest records the "
        "chosen clips, and every reader - marking, analysis, frame grabs - resolves the game clock to the right "
        "clip directly. Nothing is copied or re-encoded, and no merged video is produced."
    )
    picked_clips = st.multiselect(
        "Videos (newest first)",
        option_strings,
        default=seed_picks,
        format_func=_video_label,
        key=f"game_clips::{match_id}",
        on_change=_open_analysis_beside_video,
        help=(
            "Videos found under the footage roots, newest first; the folder above narrows the list. One that "
            "already has an analysis (marked `- analysis saved`) or belongs to a game opens it when picked."
        ),
    )
    if not picked_clips:
        st.info("Pick at least one clip.")
    else:
        ordered_clips, expected_output, expected_dir = game_lib.locations(picked_clips)
        planned_game = game_lib.plan(ordered_clips)
        total_minutes = sum(clip.duration_s for clip in planned_game.clips) / 60.0
        if len(planned_game.clips) == 1:
            only = Path(planned_game.clips[0].path)
            st.write(f"Already one game video: {only.name} - {total_minutes:.0f} min")
            st.caption(
                "Used as it stands. The game metadata and the half-time marks live in "
                f"`{_location_label(expected_dir)}` beside it, not in the video itself, so the file is never "
                "written to."
            )
        else:
            st.write(
                "Playing order: "
                + " -> ".join(Path(clip.path).name for clip in planned_game.clips)
                + f" - {total_minutes:.0f} min"
            )
            st.caption(
                f"The game metadata will live in `{_location_label(expected_dir)}`; no video file is written."
            )
        if planned_game.problem:
            st.error(
                f"These clips cannot be combined: {planned_game.problem}. A combination without a re-encode needs "
                "the same codec and frame size, and re-encoding tens of gigabytes is not something to start "
                "silently."
            )
        else:

            def _build_game() -> None:
                command = [sys.executable, str(REPO_ROOT / "scripts" / "run_build_game.py")]
                for clip in ordered_clips:
                    command += ["--clip", str(clip)]
                command += ["--out", str(expected_dir)]
                subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.session_state["step1_flash"] = (
                    "success",
                    "Preparing the game (a manifest write - ready in an instant). Refresh in a moment.",
                )

            build_state = game_lib.read_build_state(expected_dir)
            running = build_state.get("state") == "running"
            label = "Use this game video" if len(planned_game.clips) == 1 else "Use these clips as one game"
            st.button(
                label,
                disabled=running,
                on_click=_build_game,
                key=f"build_game::{match_id}",
            )
            if running:
                elapsed = int(max(0.0, time.time() - float(build_state.get("started", time.time()))))
                st.info(
                    f"Working in the background: {build_state.get('stage', 'combining')} ({elapsed}s so far). "
                    "Refresh in a few minutes."
                )
            elif build_state.get("state") == "error":
                st.error(f"The build failed: {build_state.get('error')}")
            elif (expected_dir / game_lib.MANIFEST_FILE).exists():
                st.success(
                    f"Game ready: {len(planned_game.clips)} clip(s), {total_minutes:.0f} min. The game is "
                    "recognized from any of its clips; mark the clock on it below."
                )

# What the page works on, now that the picker above has reported: the open match's footage, or the picked
# videos - the first pick that belongs to a game (any clip identifies its game), else the first pick itself. A
# pasted path wins outright, as it always did. Until the widget has drawn once, the kept selection from the last
# run stood in for it above, so the expander already knew whether a game was open.
manual_path = manual.strip()
if manual_path:
    chosen = manual_path
    game_record = game_lib.find_for_video(chosen)
elif match_id is not None:
    chosen = (
        (match.analysis_source if match is not None else None)
        or pinned_source
        or (list(picked_clips)[0] if picked_clips else option_strings[0])
    )
    game_record = game_lib.find_for_video(chosen)
else:
    chosen, game_record = _open_game_for(list(picked_clips))
    if chosen is None:
        chosen = list(picked_clips)[0] if picked_clips else option_strings[0]
if not Path(chosen).exists():
    st.error("File not found.")
    st.stop()
probe = probe_cached(chosen)

# Two states with nothing to analyze yet, each said in one line below: several videos picked but no game made
# (the button above is the missing step), or nothing picked at all.
game_pending = manual_path == "" and match_id is None and len(picked_clips) > 1 and game_record is None
no_pick = manual_path == "" and match_id is None and not picked_clips

# Starting the session: the match record beside the footage, which is where every later step stores its results.
# It is created by the button here - or automatically by the first Run analysis - and it is re-opened by picking
# a video that has one (or any clip of a game that has one).
if match_id is None and not game_pending and not no_pick:
    analysis_home = game_lib.manifest_dir(game_record) if game_record is not None else analysis_dir_for(chosen)
    st.caption(
        f"Starting an analysis saves it with the footage: `{_location_label(analysis_home)}` - everything later "
        "lands in that folder."
    )

    def _create_match() -> None:
        """Runs before the next run's body, so this module picks the new analysis up in that same run."""
        source = str(game_record.output) if game_record is not None else chosen
        existing = _analysis_for_video(source)
        record = library.create(source)
        st.session_state["archive_select_pending"] = record.match_id
        st.session_state["session_flash"] = (
            "success",
            f"`{record.match_id}` already had an analysis beside the footage - opened it rather than starting "
            "a second one."
            if existing
            else f"`{record.match_id}` started - saved with the footage in "
            f"`{_location_label(library.path(record.match_id))}`.",
        )

    st.button(
        "Start a match analysis for this footage",
        type="primary",
        on_click=_create_match,
        help=(
            "Writes the match record into an `analysis/` folder beside the video. Every later step stores its "
            "results there too, so copying the match folder carries the whole analysis with it."
        ),
    )

if game_record is not None:
    game_directory = game_lib.manifest_dir(game_record)
    if len(game_record.clips) == 1:
        source_label = f"`{Path(game_record.clips[0].path).name}`"
    else:
        source_label = f"{len(game_record.clips)} camera clips, read directly (nothing merged)"
    st.markdown(f"**Game** {source_label} - {game_record.duration_s / 60:.0f} min")
    mark_problem = game_record.mark_problem()
    if mark_problem and not mark_problem.startswith("still to mark"):
        st.warning(f"The marks cannot be used yet: {mark_problem}.")

    mark_port = configured_port()
    stream_ready = is_reachable(mark_port)
    if not stream_ready:
        st.info(
            "The marking stream comes from the footage server (`scripts/run_match_stream.py`): starting it puts "
            "the clips here - frames are read directly, so there is no proxy to build and nothing to wait for. "
            "The same server plays the annotated footage beside the replay later."
        )

        def _start_mark_stream() -> None:
            # One server serves everything - these marking frames and the annotated stream - so this is the same
            # action as the replay section's button, offered here because this is where the missing server is
            # felt first.
            subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_match_stream.py"),
                    "--port",
                    str(mark_port),
                ],
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            st.session_state["step1_flash"] = ("success", f"Footage stream starting on port {mark_port}.")

        st.button(
            "Start the footage stream",
            key=f"start_mark_stream::{game_record.game_id}",
            on_click=_start_mark_stream,
        )
    else:
        mark_result = GAME_TIMELINE_COMPONENT(
            game_id=game_record.game_id,
            stream_port=mark_port,
            stream_base=configured_base(),
            duration_s=game_record.duration_s,
            marks={"start": game_record.start_s, "half": game_record.half_s, "end": game_record.end_s},
            key=f"game_marks::{game_record.game_id}",
            default=None,
        )
        marks_seen = st.session_state.setdefault("game_marks_seen", {})
        if isinstance(mark_result, dict) and str(mark_result.get("action")) == "mark":
            # The component's value is sticky, so it comes back on every later run: only a sequence number
            # newer than the last one handled is a new press. That is why the key itself never changes - a
            # remount would throw the playback position away in the middle of marking.
            sequence = int(mark_result.get("seq") or 0)
            if sequence > int(marks_seen.get(game_record.game_id, 0)):
                which = str(mark_result.get("mark") or "")
                if which in game_lib.MARKS:
                    at = float(mark_result.get("time") or 0.0)
                    game_record.set_mark(which, at)
                    game_record.save(game_directory)
                    marks_seen[game_record.game_id] = sequence
                    # The marks define the analyzed window, so apply it now rather than waiting for the radio
                    # to be touched: marking kick-off and full-time is exactly how the window is chosen.
                    if game_record.bounds() is not None:
                        selection = str(
                            st.session_state.get(
                                f"game_window::{game_record.game_id}", game_lib.WINDOW_WHOLE
                            )
                        )
                        window_start, window_end = game_record.window(selection)
                        st.session_state[f"start_s::{chosen}"] = float(window_start)
                        st.session_state[f"length_s::{chosen}"] = float(window_end - window_start)
                    st.session_state["step1_flash"] = ("success", f"Marked {which} at {_clock(at)}.")
                    st.rerun()

    st.write(
        "  ".join(
            f"**{name}** {_clock(value) if value is not None else '-'}"
            for name, value in (
                ("kick-off", game_record.start_s),
                ("half-time", game_record.half_s),
                ("full-time", game_record.end_s),
            )
        )
    )
    if game_record.bounds() is not None:
        window_key = f"game_window::{game_record.game_id}"
        # A reopened match should show the window it was analyzed with. When that window is one of the three
        # standard choices, put the radio on it; a custom saved window is restored into the number inputs below.
        if just_selected and match is not None and match.window is not None:
            saved_start, saved_end = match.window
            for choice in game_lib.WINDOW_CHOICES:
                start, end = game_record.window(choice)
                if abs(start - saved_start) < 1e-6 and abs(end - saved_end) < 1e-6:
                    st.session_state[window_key] = choice
                    break

        def _apply_game_window() -> None:
            selection = str(st.session_state.get(window_key))
            window_start, window_end = game_record.window(selection)
            st.session_state[f"start_s::{chosen}"] = float(window_start)
            st.session_state[f"length_s::{chosen}"] = float(window_end - window_start)
            st.session_state["step1_flash"] = (
                "success",
                f"Analyzing {selection.lower()}: {_clock(window_start)} to {_clock(window_end)}.",
            )

        game_window_selection = st.radio(
            "Restrict the analysis to",
            list(game_lib.WINDOW_CHOICES),
            index=0,
            horizontal=True,
            key=window_key,
            on_change=_apply_game_window,
            help=(
                "Kick-off to full-time drops the warm-up and the walk-off; one half on its own is what a per-half "
                "report needs. Each window gets its own analysis directory, so switching back and forth is cheap."
            ),
        )
        if st.button("Clear the marks", key=f"clear_marks::{game_record.game_id}"):
            game_record.clear_marks()
            game_record.save(game_directory)
            st.rerun()

# A game is analyzed from its manifest, not from any one clip: the picked clip only identifies the game here,
# and the manifest carries the clip offsets the reader resolves. A plain video is analyzed as itself, as before.
analysis_source = (
    str(game_lib.manifest_dir(game_record) / game_lib.MANIFEST_FILE) if game_record is not None else chosen
)
# Each window of a marked game gets its own segment directory: the stored meta refuses a different window in the
# same directory (correctly), and a separate directory is what lets the two halves be analyzed without redoing
# either - and resumed independently.
segment_dir = segment_dir_for(
    analysis_source,
    segments_root_for(analysis_source),
    window_label=(
        game_record.window_label(game_window_selection)
        if game_record is not None and game_window_selection
        else None
    ),
)
config = SegmentConfig()

if game_pending:
    st.info(
        "These clips are not one game yet - press **Use these clips as one game** above: the analysis then runs "
        "on the whole game, because every later step needs one clock."
    )
elif no_pick:
    st.info("Pick the video to analyze (or the game's clips) above.")
else:
    span_col, length_col = st.columns(2)
    # A marked game starts on kick-off rather than at the top of the recording - that is the point of marking it -
    # and the window radio above overwrites these two when another window is picked.
    game_marks = game_record.bounds() if game_record is not None else None
    # Restore the saved analysis window on selection, before the inputs exist. This also covers a custom window
    # that is not one of the standard whole/first-half/second-half choices.
    if just_selected and match is not None and match.window is not None:
        st.session_state[f"start_s::{chosen}"] = float(match.window[0])
        st.session_state[f"length_s::{chosen}"] = float(match.window[1] - match.window[0])
    with span_col:
        # Keyed by video: the segment belongs to the footage, so one match's analysis window must not become the
        # next one's. A leftover offset would silently analyze the wrong part of a different match.
        start_s = st.number_input(
            "Start offset (s)",
            min_value=0.0,
            value=0.0 if game_marks is None else float(game_marks[0]),
            step=30.0,
            key=f"start_s::{chosen}",
        )
    with length_col:
        duration_s = st.number_input(
            "Length to analyze (s)",
            min_value=0.0,
            value=0.0 if game_marks is None else float(game_marks[2] - game_marks[0]),
            step=30.0,
            key=f"length_s::{chosen}",
            help="0 analyzes from the offset to the end of the video; a marked game defaults to kick-off to full-time.",
        )

    # The window is on the source's clock: for a game that is the whole game (all its clips, one clock), for a
    # plain recording its own length. A marked game overwrites both inputs with kick-off to full-time anyway.
    window_limit = float(game_record.duration_s) if game_record is not None else probe.duration_s
    window_start, window_end = resolve_window(start_s, duration_s, window_limit)
    # Reuse the exact segment directory the match was analyzed into when the requested window still matches it.
    # This is what makes a custom saved window resumable even though it has no standard window label.
    if match_id is not None and match is not None and match.saved_segment and match.window is not None:
        if abs(match.window[0] - window_start) < 1e-6 and abs(match.window[1] - window_end) < 1e-6:
            segment_dir = Path(match.saved_segment)
    status = read_status(segment_dir)
    existing_meta: dict | None = None
    meta_file = segment_dir / "meta.json"
    if meta_file.exists():
        try:
            existing_meta = json.loads(meta_file.read_text())
        except (OSError, json.JSONDecodeError):
            existing_meta = None
    st.caption(
        f"Analyzing **{_clock(window_start)} to {_clock(window_end)}** "
        f"({window_end - window_start:.0f} s"
        + (", the entire video from the offset" if duration_s <= 0 else "")
        + ")."
    )
    if game_record is not None:
        st.caption(
            f"Source: the whole game - {len(game_record.clips)} clip(s), {game_record.duration_s / 60:.0f} min "
            f"({probe.width}x{probe.height}, {probe.fps:.0f} fps)."
        )
    else:
        st.caption(
            f"Source: `{_location_label(Path(chosen))}` - {probe.width}x{probe.height}, {probe.fps:.0f} fps, "
            f"{probe.duration_s / 60:.1f} min."
        )
    if game_marks is not None:
        st.caption(
            f"Game clock: kick-off {_clock(game_marks[0])}, half-time {_clock(game_marks[1])}, full-time "
            f"{_clock(game_marks[2])}. Events and the report are labeled by half."
        )
    stored_width_now = int(existing_meta.get("detect_width", 1920)) if existing_meta is not None else 0
    detail_options = [
        f"Full resolution ({int(probe.width)} px wide - slower, best far-side recall)",
        f"Fast ({min(1920, int(probe.width))} px wide - about 3x faster)",
    ]
    detail = st.selectbox(
        "Detection detail",
        detail_options,
        index=1 if existing_meta is not None and stored_width_now < int(probe.width) else 0,
        key=f"detect_detail::{chosen}",
        help=(
            "Full resolution finds more players on the far side of the pitch. Fast detection is roughly three "
            "times faster and keeps the near-side players; it is the right choice when a first pass matters more "
            "than every distant box. The width is stored with the analysis, so a different choice restarts it."
        ),
    )
    analysis_width = int(probe.width) if detail.startswith("Full") else min(1920, int(probe.width))
    if analysis_width < int(probe.width):
        st.caption(
            f"Fast detection samples the video at {analysis_width} px wide instead of {int(probe.width)} px."
        )
    use_log_motion = st.checkbox(
        "Reuse gimbal motion for tracking",
        value=True,
        key=f"log_motion::{chosen}",
        help=(
            "Stage A already measures a camera-motion step for the pose chain. When a gimbal log is present, "
            "BoT-SORT reuses that step instead of estimating global motion again at full resolution, which is "
            "faster. Turn this off if track identities fragment during fast pans."
        ),
    )

    def _same_window(meta: dict) -> bool:
        """Whether the results already in the output directory cover exactly the requested window."""
        return (
            abs(float(meta.get("start_s", -1.0)) - window_start) < 1e-6
            and abs(float(meta.get("end_s", -1.0)) - window_end) < 1e-6
        )

    window_conflict = existing_meta is not None and not _same_window(existing_meta)
    fps_conflict = existing_meta is not None and abs(float(existing_meta.get("fps", config.fps)) - config.fps) > 1e-6
    # Detection width is a launch choice (full resolution or the 1920-wide Fast mode), and `analyze_segment` refuses
    # to resume results from another width into one directory - so a width mismatch supersedes like a window or rate
    # one. A meta without `detect_width` predates the full-resolution default: it was analyzed at 1920.
    stored_width = stored_width_now
    width_conflict = existing_meta is not None and stored_width != int(analysis_width)

    info_col, run_col = st.columns([2, 1])
    with info_col:
        if window_conflict or fps_conflict or width_conflict:
            stored = (
                f"{_clock(float(existing_meta['start_s']))} to {_clock(float(existing_meta['end_s']))} "
                f"at {float(existing_meta.get('fps', 5.0)):g} fps, detected at {stored_width} px"
            )
            st.warning(
                f"This video already has results for **{stored}**. There is one output directory per video, and a "
                "different window, analysis rate or detection width cannot be resumed into the same one - running "
                f"again (this run: {_clock(window_start)} to {_clock(window_end)} at {config.fps:g} fps, detecting "
                f"at {analysis_width} px) will move the existing results aside (renamed, not deleted)."
            )
        _stage_a_status(segment_dir, watch_key=f"stage_a_watch::{chosen}")
    with run_col:
        def _run_analysis() -> None:
            """Spawn the pass from the callback, which runs before the page is drawn again - one run, not two."""
            archived_note = ""
            if existing_meta is not None and (not _same_window(existing_meta) or fps_conflict or width_conflict):
                # `analyze_segment` refuses to mix windows, analysis rates or detection widths in one output
                # directory, so the old results move aside under a timestamped name. Renaming (rather than deleting)
                # means putting the earlier analysis back is a rename away, which matters because re-analysis of the
                # same footage is what created the conflict.
                archived = segment_dir.with_name(f"{segment_dir.name}_superseded_{time.strftime('%Y%m%d_%H%M%S')}")
                segment_dir.rename(archived)
                archived_note = f" The previous results are at `{_location_label(archived)}`."
            saved_note = ""
            record_id = match_id
            if record_id is None:
                # Starting the pass starts the analysis, and the analysis lives with the footage: there is no
                # separate save step to forget, and no segment can end up orphaned in the repository.
                # A game's directory comes from its manifest's own output name (where the merge would have gone),
                # which is exactly where the manifest already sits.
                record = library.create(str(game_record.output) if game_record is not None else chosen)
                record_id = record.match_id
                st.session_state["archive_select_pending"] = record_id
                saved_note = (
                    f" Saved with the footage in `{_location_label(library.path(record_id))}`."
                )
            # The match record keeps the segment list, which is what `summaries` and the rebuild scripts read.
            library.add_segment(record_id, segment_dir)
            # The chosen width travels explicitly: the stored meta is what the resume/conflict checks read, and a
            # Fast run must not silently resume full-resolution chunks or vice versa.
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_stage_a.py"),
                "--video", analysis_source,
                "--out", str(segment_dir),
                "--start", str(start_s),
                "--duration", str(duration_s),
                "--width", str(analysis_width),
            ]
            if not use_log_motion:
                command.append("--no-log-motion")
            _queue_status(
                segment_dir / "status.json",
                message="Queued - the job manager will start the analysis.",
                frames_done=0,
                total_frames=0,
            )
            jobs.JOBS.submit(
                kind="stage_a",
                label="Stage A analysis",
                match_id=record_id,
                argv=command,
                watch=segment_dir / "status.json",
                resources=("gpu",),
            )
            st.session_state["step1_flash"] = (
                "success",
                "Analysis queued - it runs in the background and progress updates here automatically."
                + saved_note
                + archived_note,
            )

        stage_status = read_status(segment_dir) or {}
        stage_busy = stage_status.get("state") in ("running", "queued") or (
            match_id is not None and jobs.JOBS.has_active(match_id, kinds=("stage_a",))
        )
        st.button("Run analysis (background)", type="primary", on_click=_run_analysis, disabled=stage_busy)

# --------------------------------------------------------------------------------------------------------------
# Step 2 - pitch registration
# --------------------------------------------------------------------------------------------------------------
st.header("Step 2 - Register the pitch")
st.caption(
    "Click pitch landmarks on a frame. Four are enough; the further apart they are, the better the fit. The table "
    "below reports the error for each click, because landmarks on the far side of the pitch carry meters of "
    "uncertainty."
)

segment = None
chunks_done = stage_a.completed_chunks(segment_dir)
if chunks_done == 0:
    st.info("Run Step 1 first: the camera motion from that pass is what lets landmarks from any frame be used.")
else:
    try:
        # Cached across reruns: this is the gesture loop, and an uncached load + pose integration is the difference
        # between a click responding immediately and a click taking most of a minute on a whole-game segment.
        segment, q, focal = segment_and_poses_cached(
            str(segment_dir), segment_fingerprint(segment_dir, chunks_done)
        )
    except Exception as exc:  # a partially written segment must not break the page
        st.warning(f"Could not read the segment results: {exc}")

if segment is not None:
    # Say where the camera motion comes from: the gimbal log is a hardware measurement and does not drift, so a
    # calibration built on it holds across the whole game; the estimated chain drifts and needs the clicks to
    # re-anchor it. The user should know which one they are looking at.
    if segment_has_log(segment):
        st.caption(
            "Camera motion is taken from the gimbal's own log (yaw/pitch telemetry beside the video), so it does "
            "not drift over the game - landmarks clicked late in the match stay consistent with early ones."
        )
    else:
        st.caption(
            "Camera motion is estimated from the picture; it drifts slowly, so clicks spread across the match "
            "re-anchor it. A gimbal log beside the video would remove that drift."
        )
    saved_click_source = library.load_clicks_source(match_id) if match_id is not None else None
    if (
        saved_click_source is not None
        and Path(saved_click_source).resolve() != Path(analysis_source).resolve()
    ):
        st.warning(
            f"The saved calibration was clicked on `{Path(saved_click_source).name}`, not this match's analyzed "
            f"footage (`{Path(analysis_source).name}`). It is not loaded here because its clicks are frame-indexed "
            "into a different recording - recalibrate on the analyzed footage."
        )
    # The pitch size lives on the match record, not in a widget: the calibration was clicked against those
    # dimensions, so reading them back from the record is what stops the two drifting apart across reloads.
    record = library.load(match_id)
    stored_format = record.format if record.format in MATCH_FORMATS else list(MATCH_FORMATS)[-2]
    format_col, length_col, width_col, goal_col = st.columns([2, 1, 1, 1])
    with format_col:
        format_name = st.selectbox(
            "Match format", list(MATCH_FORMATS), index=list(MATCH_FORMATS).index(stored_format)
        )
    # Youth pitches are marked to the format's convention but not to a surveyor's tolerance. The format supplies
    # the templates (markings, landmark list); these numbers are what the click coordinates are actually built
    # from, so a field measured at 64 x 42 m is calibrated as 64 x 42 m instead of being forced onto 60 x 40.
    default_length, default_width = MATCH_FORMATS[format_name]
    if record.format == format_name:
        default_length, default_width = record.pitch_length_m, record.pitch_width_m
    default_goal_width = markings_for(format_name).goal_width_m
    if record.format == format_name and record.goal_width_m is not None:
        default_goal_width = record.goal_width_m
    with length_col:
        length_m = st.number_input(
            "Pitch length (m)",
            min_value=20.0,
            max_value=120.0,
            value=float(default_length),
            step=0.5,
            key=f"pitch_length::{match_id}::{format_name}",
            help="Measure along a touchline, goal line to goal line. The format only supplies the default.",
        )
    with width_col:
        width_m = st.number_input(
            "Pitch width (m)",
            min_value=15.0,
            max_value=80.0,
            value=float(default_width),
            step=0.5,
            key=f"pitch_width::{match_id}::{format_name}",
            help="Measure across the pitch, touchline to touchline.",
        )
    with goal_col:
        goal_width_m = st.number_input(
            "Goal width (m)",
            min_value=1.0,
            max_value=20.0,
            value=float(default_goal_width),
            step=0.1,
            key=f"goal_width::{match_id}::{format_name}",
            help=(
                "Distance between the posts. The format supplies the default (9v9: 4.88 m, 11v11: 7.32 m); "
                "set the measured value when the goals are a different size. Goal detection and the goal-post "
                "landmarks use this."
            ),
        )
    markings = with_goal_width(markings_for(format_name), goal_width_m)
    markings_detail = (
        f"{markings.penalty_area_width_m:.2f} x {markings.penalty_area_depth_m:.2f} m penalty area"
        + (
            f", {markings.goal_area_width_m:.2f} x {markings.goal_area_depth_m:.2f} m goal area"
            if markings.has_goal_area
            else ""
        )
        + (
            f", {markings.center_circle_radius_m:.2f} m center circle"
            if markings.has_center_circle
            else ""
        )
        + f", {markings.goal_width_m:.2f} m goal"
    )
    st.caption(f"**{format_name} markings:** {markings_detail}. These are the markings the fit and overlay expect.")
    record_goal_width = (
        record.goal_width_m
        if record.goal_width_m is not None
        else markings_for(record.format).goal_width_m
    )
    if (record.format, record.pitch_length_m, record.pitch_width_m, record_goal_width) != (
        format_name,
        length_m,
        width_m,
        goal_width_m,
    ):
        had_calibration = library.load_calibration(match_id) is not None
        record.format, record.pitch_length_m, record.pitch_width_m = format_name, length_m, width_m
        record.goal_width_m = float(goal_width_m)
        library.save(record)
        st.session_state.pop("calibration", None)
        # The click pixels were placed against the old world coordinates: a corner at (0, 64) is no longer the
        # same physical point at (0, 45), and the goal posts move when the goal width changes. Keeping those
        # clicks in the gesture state and re-solving them against the new table is what produces a misleading
        # "clicks do not agree" error with exactly four landmarks. Drop the set; the restore block below will
        # only bring it back when the saved clicks were explicitly labelled against these dimensions.
        st.session_state["calib_points"] = []
        st.session_state["calib_landmarks"] = []
        st.session_state["calib_seeded"] = []
        st.session_state.pop("calib_fit_signature", None)
        if had_calibration:
            st.warning(
                "The pitch or goal size changed after this match was calibrated, so the saved calibration no "
                "longer matches. Click the landmarks again before relying on the report."
            )

    # q/focal come from the same cache as the segment: rebuilding the chain costs ~1 s on the whole-game segment
    # even with the fast path, and this line is on the gesture loop's hot path.
    frame_count = len(segment.time)
    land_table = landmark_table(length_m, width_m, format_name=format_name, goal_width_m=goal_width_m)
    help_table = landmark_help(format_name, goal_width_m=goal_width_m)
    names = list(land_table)

    # Clicks belong to one match, one video and one pitch size. Switching any of them must not quietly blend the
    # two, so the clicks, the view position and any in-session calibration are dropped when the combination changes.
    # The clicks saved with the last calibration are read back instead of starting empty: a page refresh starts a
    # new session, and the whole point of the saved set is to outlive the browser tab. Frames are indices into the
    # segment they were clicked on, so a set from another segment is refused (restored_points) rather than
    # renumbered into nonsense.
    # Bump the version when the restore rules change: a running page keeps its session state across reruns, so a
    # stale (empty) set would otherwise stay empty until the user switched matches. v2 re-resolves saved sources
    # against the footage root, which is what makes a moved archive's clicks visible again.
    clicks_key = f"v2|{match_id}|{chosen}|{format_name}|{length_m:.3f}|{width_m:.3f}"
    if st.session_state.get("calib_clicks_key") != clicks_key:
        saved_clicks = library.load_clicks(match_id)
        saved_source = library.load_clicks_source(match_id)
        saved_pitch = library.load_clicks_pitch(match_id)
        same_pitch = saved_pitch is None or (
            abs(saved_pitch[0] - length_m) < 0.05 and abs(saved_pitch[1] - width_m) < 0.05
        )
        # Frames are indices into the recording the clicks were made on. Restoring a set made on one clip while a
        # different video is open puts the same frame number on different pictures; the fit then looks like a bad
        # calibration and (worse) can overwrite the good one on disk. New sets name their source. Legacy sets have
        # no source, so they are only restored when the open video is one this match actually recorded - otherwise
        # another match's clicks would be applied to this footage and then re-saved into this archive.
        if saved_source:
            same_source = Path(saved_source).resolve() == Path(chosen).resolve()
        else:
            same_source = any(Path(source).resolve() == Path(chosen).resolve() for source in record.sources)
        restored = (
            restored_points(saved_clicks, frame_count) if saved_clicks and same_source and same_pitch else None
        )
        if restored:
            # A format that has no centre circle (5v5/7v7) also has no landmark by that name. A set saved before
            # the format-aware tables could still carry one; keep the click but let the user re-label it rather
            # than crashing the fit on a name that is no longer in the table.
            for point in restored:
                if point.get("label") and point["label"] not in land_table:
                    point.pop("label")
        st.session_state["calib_clicks_key"] = clicks_key
        st.session_state["calib_points"] = restored or []
        st.session_state["calib_landmarks"] = saved_clicks if restored else []
        st.session_state["calib_next_pid"] = len(restored or [])
        st.session_state["calib_restored_count"] = len(restored or [])
        st.session_state["calib_seeded"] = []
        st.session_state["calib_center"] = (0.5, 0.5)
        st.session_state["calib_zoom"] = DEFAULT_ZOOM
        st.session_state["calib_reference"] = int(restored[0]["frame"]) if restored else 0
        st.session_state.pop("calibration", None)
        if saved_clicks and not same_pitch:
            st.session_state["calib_restore_note"] = (
                "The saved clicks were labelled against a different pitch size, so they were not restored. "
                "Re-click the landmarks on this pitch size."
            )
        elif saved_clicks and saved_source and not same_source:
            st.session_state["calib_restore_note"] = (
                f"The saved clicks were made on `{Path(saved_source).name}`, not this match's analyzed footage. "
                "They are not restored against a different recording - recalibrate on the analyzed footage."
            )
    restore_note = st.session_state.pop("calib_restore_note", "")
    if restore_note:
        st.info(restore_note)

    # The clicking workspace stays on the page: it is the thing you are working in while a calibration is being
    # made, and hiding it behind a toggle would put a click between every aim and every landmark. Only the saved
    # clicks list is collapsed - it is long once a match is calibrated and is only needed to check or re-label a
    # click.
    with st.expander("What are pitch landmarks, and how do I click them?", expanded=False):
        st.markdown(
            f"""
The camera moves, so the app has to work out *where the camera was* before it can say where a player is. It does
that from **landmarks**: fixed markings on the pitch whose real position you already know, because the laws of the
game put them there. The corners, the posts, the halfway line, the penalty spots, and the center circle where the
format has one - those are the only things in the frame with a known real-world position, so the calibration is
only ever as good as these clicks. This match uses the **{format_name}** markings listed below.

**What to click**

| Landmark | Click exactly on |
| --- | --- |
"""
        + "\n".join(f"| `{name}` | {help_table[name]} |" for name in land_table)
        + """

**How to do it**

1. Pick a frame where the pitch markings are visible - any frame works, because the camera's motion is already
   known. Use the timeline under the whole-frame view to scrub to one, or press **Play** and pause where the
   markings are clearest.
2. **Aim the magnified view from the whole-frame view.** Click a point on the whole frame and the magnified view
   moves so that point is in the middle, reloading at whatever frame the timeline is showing; **scroll** to zoom in
   or out on it. The yellow box shows what the magnified view covers, so it doubles as a check that you are looking
   at the right part of the pitch.
3. **Click the landmark on the magnified view.** A click opens the landmark list right where you clicked - pick
   which landmark it is and the click is committed at once; there is no Apply step. Zoom in far enough that
   the corner is unmistakable: at zoom 8 a screen pixel is a pixel of the 4K frame, whereas a full-frame view is
   worth roughly a meter of ground error per pixel on the far side of the pitch.
4. **If a corner is not in shot, use the goalposts instead.** The near corner flags are often out of frame or lost
   against the grass, but the posts are easy to pick out, and the base of a post - where it meets the goal line -
   says as much about that end of the pitch as the flag does. Click **goal post left-near** / **goal post
   left-far** (or the `right` pair) in place of a corner you cannot see. The posts are {markings.goal_width_m:.2f} m
   apart for this format, so the two of them together also fix the goal line's direction.
5. **The penalty spots are usually visible too, and so is the center circle where the format has one.** Those
   markings sit at a range of distances from the camera - exactly the spread the fit is short of when the near
   corners are out of shot. Click the ones you can see clearly. The boxes are deliberately not in the list: the
   goal area (where the format has one) is small and lost against the netting, and a penalty-area corner is a bare
   junction of two lines with nothing to focus on - the posts carry that end of the pitch instead, and the penalty
   spots pin the box's depth.
6. **What you cannot substitute is spread.** Four clicks that are all a long way off, or all along one line, leave
   the fit with tens of meters of doubt - measured on a simulated match, four distant landmarks were out by more
   than 50 m. Aim for landmarks at a *range* of distances from the camera: the halfway line where it meets the near
   touchline is a good near one, because a long white line is easy to pick out. Click as many as you can be sure
   of. **Six to eight spread across the frame is comfortable; four is the absolute minimum.**
7. Repeat as needed. The app discards a bad click for you, and the fit check projects the corners you never
   clicked, so you can see where it thinks they are.

Once a calibration is saved, the whole-frame view carries it too: the pitch markings and every landmark, projected
through the fit onto whichever frame is loaded. To re-anchor a later moment, press **Place the calibrated landmarks
as markers** - each landmark the calibration used lands in the magnified view where the fit expects it, and you
**drag** it onto the real marking. A drag is a measurement, so it says how far the fit has slid by that moment; it
is also far quicker than finding five markings from scratch.

**The one thing that must be consistent:** label the corners in a loop around the pitch - one touchline, then the
other (this is the order in the dropdowns: `near-left`, `near-right`, then the two `far` ones). Get a consistent
loop and the app can work out the rest.

**Then check the yellow markings** the app draws back onto the frame. They are the whole pitch - touchlines, halfway
line, both boxes, the center circle, the penalty arcs and spots and the corner arcs - projected through your
calibration. If they land on the real markings, the registration is good. If they are mirrored, or clearly in the
wrong place, fix the labels and recalibrate - the per-landmark error table tells you which click to look at.

Be honest with yourself about the far corners: at 40-90 m, four pixels of click noise is worth several meters of
ground error in the depth direction. If the far corner is a blurred smudge, leave it out and use the landmarks you
are sure of.
"""
        )

    # The whole-frame viewport is also the timeline. It needs the proxy; while that builds (a one-time pass over
    # the segment) the viewport keeps a still frame and a plain slider picks the frame instead.
    proxy_ready = timeline_ready(segment, segment_dir)
    if not proxy_ready:
        _timeline_status(segment_dir, watch_key=f"timeline_watch::{segment_dir}")

    # A gesture arrives as the component's own widget value, and Streamlit hands widget values to the run *before* it
    # starts - so it can be read here and settled into session state before the crop is computed. That is what keeps
    # a gesture to a single run: acting on it after the component costs a second full run (another frame read from
    # the 4K source, and the whole page rendered again) because the crop is built above it.
    #
    # The value is sticky - it comes back on every later rerun - and clicks now commit the moment they happen, so
    # there is no Apply button to re-key the component after. Instead every reported gesture carries a sequence
    # number (plus the mount nonce, so a re-keyed component restarting from seq 1 is never mistaken for an old
    # gesture), and a gesture is acted on only when it has not been seen before.
    jump_nonce = int(st.session_state.get("calib_jump_nonce", 0))
    component_key = f"calib::{chosen}::{jump_nonce}"
    gesture = parse_result(st.session_state.get(component_key))
    gesture_key = f"calib_gesture::{chosen}"
    fresh = (gesture.mount, gesture.seq) != st.session_state.get(gesture_key, (-1, -1))
    if fresh:
        st.session_state[gesture_key] = (gesture.mount, gesture.seq)
    if fresh and gesture.action == "navigate":
        # Landing the view where the gesture asked is idempotent, so the value being sticky costs nothing: on a
        # later run it asks for the center and zoom that are already stored.
        if gesture.center is not None:
            st.session_state["calib_center"] = gesture.center
        if gesture.zoom is not None:
            st.session_state["calib_zoom"] = gesture.zoom
        moved_frame = frame_change(gesture.frame, st.session_state.get("calib_reference", 0), frame_count)
        if moved_frame is not None:
            st.session_state["calib_reference"] = moved_frame

    reference = int(np.clip(st.session_state.get("calib_reference", 0), 0, max(0, frame_count - 1)))
    if proxy_ready:
        st.write(
            "**Scrub the timeline to a frame where the markings are clearest, then click the frame to load it.** "
            "Scrubbing runs in the browser, so the magnified view is only re-read when you click a point - that is "
            "what keeps the drag smooth. Any frame works, because the camera's motion is already known."
        )
    else:
        reference = st.slider(
            "Frame to click on",
            0,
            max(0, frame_count - 1),
            reference,
            help="Any frame works, because the camera motion is known.",
        )

    # With a calibration saved, the whole-frame viewport can carry it: the pitch and every landmark projected
    # through the *corrected* chain onto the frame's own still. It is the same picture as the fit check further
    # down, in the place the user is actually working - which is what shows how far the fit has slid by this moment
    # without leaving the frame being clicked on. Drawn only into the still, so it appears exactly when the
    # viewport is at rest on the crop's frame, the same rule the markers follow.
    calibration_for_view = saved_calibration(library, match_id, expected_source=analysis_source)
    overlay_on = False
    if calibration_for_view is not None:
        overlay_on = st.checkbox(
            "Show the calibration on the whole frame",
            value=True,
            key="calib_overlay_on",
            help="Projects the pitch outline and every landmark through the saved calibration onto the whole-frame "
                 "view, at whichever frame is loaded.",
        )

    full = frame_cached(chosen, float(segment.time[reference]), int(probe.width))
    if full is None:
        st.error("Could not read that frame from the video.")
    else:
        height, width = full.shape[:2]
        center = st.session_state.get("calib_center", (0.5, 0.5))
        zoom = float(st.session_state.get("calib_zoom", DEFAULT_ZOOM))
        box = zoom_box(width, height, zoom, center[0], center[1])
        x0, y0, crop_w, crop_h = box
        crop = full[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # The whole frame is scrubbed as a video inside the component when the proxy exists, so scrubbing stays
        # local to the browser. The still is sent either way: the proxy is built from the camera's keyframes only
        # (one picture per second), so the component puts this *exact* frame over the coarse proxy picture whenever
        # the viewport is at rest on it - which is what keeps the markers on the corner they were clicked on.
        overview = cv2.resize(full, (1280, int(1280 * height / width)), interpolation=cv2.INTER_AREA)
        if overlay_on and calibration_for_view is not None:
            view_q, view_focal = calibration_for_view.corrected_frame(q[reference], float(focal[reference]), reference)
            overview = pitch_overlay(
                overview,
                calibration_for_view,
                view_q,
                view_focal,
                length_m,
                width_m,
                format_name=format_name,
                table=land_table,
            )
        # The browser-side overlay: sampled homographies along the corrected chain plus the markings in pitch
        # meters, so the component can draw the pitch on whatever frame the timeline is showing - scrubbed or
        # playing - without a round trip per frame. Recomputed from the current calibration on every run, so a
        # refit redraws the overlay immediately.
        browser_overlay = (
            overlay_homographies(
                calibration_for_view, q, focal, length_m, width_m, frame_count, format_name=format_name
            )
            if overlay_on and calibration_for_view is not None
            else []
        )
        browser_polylines = (
            [line.tolist() for line in pitch_marking_polylines(length_m, width_m, format_name=format_name)]
            if browser_overlay
            else []
        )
        proxy_url = (
            _timeline_media_url(timeline.proxy_path(segment_dir), coordinates=f"timeline::{segment_dir}")
            if proxy_ready
            else ""
        )

        stored = st.session_state.setdefault("calib_points", [])
        next_pid = st.session_state.get("calib_next_pid", 0)
        scale = canvas_scale(crop_w)

        if fresh and gesture.action == "apply":
            # A click or a finished drag commits the moment it happens, so the crop just computed is the one the
            # click was made on: this is exactly the geometry that was on screen. The frame the gesture carried is
            # ignored on purpose - the points belong to the crop that is already loaded.
            if gesture.points:
                stored, next_pid = merge_clicked(stored, gesture.points, reference, box, scale, width, next_pid)
                st.session_state["calib_points"] = stored
                st.session_state["calib_next_pid"] = next_pid
            # Whatever was on screen has become real clicks; any placed markers it came from are no longer pending.
            st.session_state["calib_seeded"] = [
                seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
            ]
        elif fresh and gesture.action == "clear":
            # Clear removes the applied points in the frame of reference - the frame the crop is on - and leaves
            # the clicks of every other frame alone. The component has already emptied its own copy; this is where
            # the stored clicks follow.
            stored = [point for point in stored if point["frame"] != reference]
            st.session_state["calib_points"] = stored
            st.session_state["calib_seeded"] = [
                seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
            ]

        # A stray double-click of one landmark on one frame is resolved the way the user's last gesture meant it:
        # the newest click wins and the older one is dropped right here, before the viewport is drawn - the marker
        # that lost never even appears, and a slip never blocks the fit. The match is on the *stored* tag, which a
        # popover pick and a dropdown edit both write; an untagged click merely displays the list's fallback name,
        # and the render body cannot tell that from a deliberate choice, so untagged clicks are left alone. Other
        # frames are untouched - re-clicking a landmark later in the video is the drift-anchoring workflow.
        _kept, dropped = split_duplicate_clicks(
            (point["frame"], point["label"], point["pid"]) for point in stored if point.get("label")
        )
        if dropped:
            drop_pids = {pid for _frame, _landmark, pid in dropped}
            stored = [point for point in stored if point["pid"] not in drop_pids]
            st.session_state["calib_points"] = stored
            # One-shot: the note is popped where the calibrate button is, so it is read once and not repeated on
            # every later rerun of the same state.
            st.session_state["calib_dropped_note"] = (
                "Dropped the older duplicate click(s) automatically: "
                + ", ".join(f"`{landmark}` on frame {frame}" for frame, landmark, _pid in dropped)
                + " - a landmark counts once per frame, so clicking it again replaces the older click."
            )

        # The re-anchoring shortcut. With a calibration saved, each landmark it used can be *placed* at where the
        # fit projects it - through the corrected chain, so it already carries the drift measured so far - and
        # dragged onto the real marking. The drag is the measurement (it says how far the fit has slid by this
        # moment) and the label arrives attached, so a later moment gets anchored without hunting five markings
        # from scratch. The run happens before the features are built, so the markers appear in this same render.
        if calibration_for_view is not None:
            used_labels = {
                click.get("label")
                for click in (st.session_state.get("calib_landmarks") or library.load_clicks(match_id))
                if click.get("label")
            }
            placeable = [name for name in land_table if name in used_labels]
            if placeable and st.button(
                "Place the calibrated landmarks as markers",
                help="Puts each landmark from the saved calibration into the magnified view at the position the fit "
                     "projects it to. Drag each marker onto the real marking - the drag commits the moment you let "
                     "go, and it is what tells the fit how far it has slid by this frame.",
            ):
                seed_q, seed_focal = calibration_for_view.corrected_frame(q[reference], float(focal[reference]), reference)
                placed = [
                    spot
                    for spot in projected_landmarks(calibration_for_view, placeable, land_table, seed_q, seed_focal)
                    if inside_crop(spot["u"], spot["v"], box, scale, width)
                ]
                if placed:
                    st.session_state["calib_seeded"] = [
                        seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] != reference
                    ] + [{"frame": reference, **spot} for spot in placed]
                else:
                    st.warning(
                        "None of the landmarks from the saved calibration fall inside the magnified view - zoom "
                        "out or aim at one of them (click it on the whole frame), then press again."
                    )

        seeds_here = [seed for seed in st.session_state.get("calib_seeded", []) if seed["frame"] == reference]
        # Show this frame's existing clicks that fall inside the current crop, so they can be added to - plus any
        # placed markers, which wait here (in frame coordinates, so a zoom keeps them on their landmark) until a
        # click or a drag commits them.
        features = points_in_crop(stored, reference, box, scale, width)
        features += [marker_feature(seed["label"], seed["u"], seed["v"], box, scale, width) for seed in seeds_here]

        st.caption(
            "The magnified view is for clicking landmarks; the whole frame beside it is for aiming **and for "
            "scrubbing**. **Scrub** to a frame where the markings are clear, then **click** a point on the whole "
            "frame to load that frame and bring the point into the middle of the magnified view; **scroll** there to "
            "zoom in or out, or use the +/- buttons. **Click a landmark on the magnified view** and pick which one it "
            "is in the list that appears at the click - the click is committed the moment you pick. **Clear** removes "
            "the clicks of the frame you are looking at, and leaves the rest alone."
        )
        landmark_clicker(
            crop,
            overview,
            features,
            frame_points(stored, reference) + [(seed["u"], seed["v"]) for seed in seeds_here],
            center,
            zoom,
            box,
            (width, height),
            key=component_key,
            proxy_url=proxy_url,
            frame_count=frame_count,
            initial_frame=reference,
            marker_kinds=names,
            mount_nonce=jump_nonce,
            overlay_homographies=browser_overlay,
            overlay_polylines=browser_polylines,
        )

    # The saved clicks and their labels: the one thing worth keeping behind a toggle, because the list is
    # long once a match is calibrated and it is only needed when a click has to be checked or re-labeled.
    # The calibrate button itself stays outside: it is the step's main action, and hiding it behind a toggle
    # meant the page read as "nothing to do here" until the expander was opened.
    stored = st.session_state.get("calib_points", [])
    ordered = order_clicks(stored)
    labeled: list[tuple[dict, str]] = []
    display_duplicates: list[str] = []
    if ordered:
        clicked_frames = {int(point["frame"]) for point in ordered}
        st.caption(
            f"{len(ordered)} saved landmark click(s) on {len(clicked_frames)} frame(s). The magnified view draws "
            "only the clicks on the frame it is showing; use a frame button in **Landmark clicks** to jump to "
            "one, or scrub to that frame."
        )

    # Callbacks run before the next run's body, so the list below is already rebuilt without the click. Doing the
    # work inline instead would need a st.rerun() to redraw it, which is a second full run for one button. They are
    # defined here, outside the expander, because the Clear button they serve sits outside it too.
    def _drop_click(pid: int) -> None:
        st.session_state["calib_points"] = [
            p for p in st.session_state.get("calib_points", []) if p["pid"] != pid
        ]

    def _goto_click(point: dict, frame_size: tuple[int, int]) -> None:
        """Show the frame a stored landmark was clicked on, centered on the click itself.

        The click list is the way back to a landmark. After a dozen clicks on scattered frames, finding one again
        by scrubbing is guesswork - and the point of going back is to check the click against the marking, which
        needs the crop on its own frame and its own spot. The frame is re-read from the source, so what comes up
        is exactly what was marked, not a nearby frame.

        The jump nonce retires the component's key: the viewport then lands on the frame the way it does on first
        mount (the same mechanism a click uses). Without it a jump to the frame already loaded would move the
        center but leave the scrub bar where the user had left it - the whole-frame view would be somewhere else.
        """
        st.session_state["calib_reference"] = int(np.clip(point["frame"], 0, max(0, frame_count - 1)))
        st.session_state["calib_center"] = point_center(point["u"], point["v"], frame_size)
        st.session_state["calib_selected_pid"] = point["pid"]
        st.session_state["calib_jump_nonce"] = int(st.session_state.get("calib_jump_nonce", 0)) + 1

    def _clear_clicks() -> None:
        st.session_state["calib_points"] = []
        st.session_state["calib_landmarks"] = []
        st.session_state["calib_seeded"] = []
        st.session_state.pop("calib_fit_signature", None)
        library.clear_calibration(match_id)
        st.session_state.pop("calibration", None)

    def _retag(pid: int, widget_key: str) -> None:
        """Keep the stored tag in step with the dropdown: an edit in the list is the click's new label.

        The selectbox value is what the fit uses, but the *stored* tag is what the click carries when it is
        re-committed (a drag re-sends the marker's label) and what the duplicate drop matches on - so a correcting
        choice has to land in the stored click too, or the next drag would quietly revert it. The callback fires
        only on a real edit, which is the signal that separates a deliberate choice from the list's fallback for
        an untagged click - the render body cannot tell those apart.
        """
        choice = st.session_state.get(widget_key, "")
        if not choice:
            return
        points = st.session_state.get("calib_points", [])
        for point in points:
            if point["pid"] == pid:
                point["label"] = choice
        st.session_state["calib_points"] = points

    with st.expander("Landmark clicks", expanded=False):
        if ordered:
            restored = int(st.session_state.get("calib_restored_count", 0))
            note = " (brought back from the saved calibration)" if restored == len(ordered) else ""
            st.write(f"**{len(ordered)} landmark click(s)**{note} - tell the app which is which:")

            labeled: list[tuple[dict, str]] = []
            for index, point in enumerate(ordered):
                point_col, label_col, drop_col = st.columns([1, 3, 1])
                with point_col:
                    # Pressing the frame brings that landmark up in the crop - its own frame, centered on the click, so
                    # the marking can be checked against what was clicked without hunting for the frame again.
                    st.button(
                        f"frame {point['frame']}",
                        key=f"goto_{point['pid']}",
                        type="primary" if point["pid"] == st.session_state.get("calib_selected_pid") else "secondary",
                        on_click=_goto_click,
                        args=(point, (width, height)),
                        help="Load this landmark's own frame into the magnified view, centered on where it was clicked.",
                    )
                with label_col:
                    # The landmark is chosen at the click, in the popover that appears where it was made - so the
                    # dropdown here is only for correcting a choice afterwards. It opens on the click's own tag;
                    # an edit is persisted to the click by `_retag`, which is what makes the corrected label the
                    # click's label everywhere (including the duplicate drop below).
                    default = point.get("label") or names[0]
                    widget_key = f"landmark_label_{point['pid']}"
                    label = st.selectbox(
                        f"Landmark {index + 1}",
                        names,
                        index=names.index(default) if default in names else 0,
                        key=widget_key,
                        label_visibility="collapsed",
                        on_change=_retag,
                        args=(point["pid"], widget_key),
                    )
                    # A saved click can carry a tag the current landmark table no longer offers (the landmark set
                    # changed between versions). The dropdown silently falls back to the click-order suggestion, so
                    # without this the click would be measured against a marking it was never clicked on - say so
                    # instead of letting it through.
                    if point.get("label") and point["label"] not in names:
                        st.warning(
                            f"Tagged `{point['label']}`, which is no longer a landmark. Re-tag it (or remove the "
                            "click) before calibrating - as it stands the click would be measured against "
                            f"`{label}`."
                        )
                with drop_col:
                    st.button("Remove", key=f"drop_{point['pid']}", on_click=_drop_click, args=(point["pid"],))
                labeled.append((point, label))
                # Always describe the landmark that is actually selected: it is the moment the user picks a
                # non-default landmark that they most need to check what it means.
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{help_table.get(label, '')}")

            # What the *fit* uses is each click's dropdown value, and two dropdowns can still show the same
            # landmark without the stored tags agreeing: an untagged click displays the list's first entry, and a
            # click tagged with exactly that name on the same frame then matches it. Only one of the pair carries
            # a tag, so the drop above cannot see the collision - warn here instead, naming what to re-tag. The
            # fit proceeds either way: the residual table is the honest judge of what the clash costs.
            display_duplicates = repeated_labels_within_a_frame((point["frame"], label) for point, label in labeled)

            # A session with clicks on more than one moment is asking for the drift correction, and that only works if
            # each moment carries enough clicks to identify its own correction. Say which frame currently registers the
            # pose and which moments are too thin to anchor anything, rather than leaving it to the fit diagnostics.
            frames_with_clicks = sorted({point["frame"] for point, _label in labeled})
            if len(frames_with_clicks) > 1:
                counts = {frame: sum(1 for point, _l in labeled if point["frame"] == frame) for frame in frames_with_clicks}
                registered = min(frames_with_clicks, key=lambda frame: (-counts[frame], frame))
                if counts[registered] >= MIN_REFERENCE_CLICKS:
                    st.caption(
                        f"Clicks on {len(frames_with_clicks)} moments. Frame {registered} carries the most "
                        f"({counts[registered]}), so the camera pose is registered on it and the other moments pin the "
                        "drift between them."
                    )
                else:
                    st.caption(
                        f"Clicks on {len(frames_with_clicks)} moments. No single moment carries "
                        f"{MIN_REFERENCE_CLICKS} landmarks to register the pose on, so it is pooled from every click "
                        "- registering on one moment is steadier."
                    )
                sparse = [frame for frame in frames_with_clicks if counts[frame] < MIN_CLICKS_PER_ANCHOR]
                collinear = []
                for frame in frames_with_clicks:
                    points = [
                        land_table[label]
                        for point, label in labeled
                        if point["frame"] == frame and label in land_table
                    ]
                    if len(points) >= MIN_CLICKS_PER_ANCHOR:
                        pitch_points = np.asarray(points, dtype=np.float64)
                        if np.linalg.matrix_rank(pitch_points - pitch_points.mean(0), tol=0.5) < 2:
                            collinear.append(frame)
                weak = sorted(set(sparse) | set(collinear))
                if weak:
                    st.info(
                        "Not enough spread to anchor the drift at frame(s) "
                        + ", ".join(str(frame) for frame in weak)
                        + f". A correction there has four unknowns: {MIN_CLICKS_PER_ANCHOR} landmarks spread "
                        "across the pitch give it a check, while fewer - or several on one goal line - let it "
                        "twist the projection away from the markings. Add a landmark off that line at those "
                        "moments."
                    )

            if len(labeled) < 4:
                st.info(f"Click at least four landmarks - {4 - len(labeled)} to go.")
        else:
            st.info(
                "No landmarks clicked yet. Aim with the whole-frame view, then click a landmark on the magnified "
                "view and pick which one it is in the list that appears at the click."
            )

    dropped_note = st.session_state.pop("calib_dropped_note", "")
    if dropped_note:
        st.info(dropped_note)
    axis_note = st.session_state.pop("calib_axis_message", "")
    if axis_note:
        st.success(axis_note)
    if display_duplicates:
        st.warning(
            "Two clicks are measured as the same landmark on one frame: " + ", ".join(display_duplicates)
            + ". Tag each click with its own landmark - an untagged click displays the list's first entry."
        )

    # The calibration refits itself whenever the clicks change - a click, a re-label or a removal is a new
    # measurement, and the fit is seconds, so there is nothing to wait for. The button stays for the two cases the
    # automatic pass cannot see: a refit against a changed camera-motion source, and "just fit it again".
    if len(labeled) >= 4:
        signature = [(point["frame"], point["pid"], label) for point, label in labeled]
        if st.session_state.get("calib_fit_signature") != signature:
            st.session_state["calib_fit_signature"] = signature
            calibration = fit_from_clicks(
                library, match_id, segment, q, focal, labeled, land_table, length_m, width_m, source=chosen
            )
            if calibration is None:
                st.session_state.pop("calib_fit_signature", None)  # let the next change retry
                st.error(calibration_failure())
            else:
                st.toast(f"Calibration updated: RMS {calibration.rms_error_m:.2f} m", icon=":material/check_circle:")

    button_col, clear_col = st.columns([2, 1])
    with button_col:
        if st.button(
            "Calibrate from these landmarks",
            type="primary",
            disabled=len(labeled) < 4,
            help="Refit the calibration from the clicks as they are labeled right now. The fit also runs "
                 "automatically whenever the clicks change; this is here for a refit against a changed camera "
                 "motion, or when you just want to run it again.",
        ):
            calibration = fit_from_clicks(
                library, match_id, segment, q, focal, labeled, land_table, length_m, width_m, source=chosen
            )
            if calibration is None:
                st.error(calibration_failure())
            else:
                st.success(fit_message(calibration, labeled))
    with clear_col:
        st.button("Clear all clicks", on_click=_clear_clicks)

    # The tripod is a real, imperfect structure: its true pan axis is a little rolled/tilted relative to the
    # gimbal's physical prior, and that error depends on heading. This is the explicit, slow action that fits it
    # from the clicks and any cached line observations; the ordinary calibrate button stays seconds-fast.
    axis_col, _axis_spacer = st.columns([2, 1])
    with axis_col:
        if st.button(
            "Fit tripod axis (slow)",
            disabled=len(labeled) < 4 or segment is None or not segment_has_log(segment),
            help="Fit the camera's true pan axis (roll/tilt/yaw scale) to these clicks and any cached white-line "
                 "observations, then refit the calibration through it. This removes a heading-dependent overlay "
                 "shift that a time-based drift correction cannot. Takes minutes on a whole game; run the "
                 "line-scan script first for the most stable fit.",
        ):
            fitted = fit_pan_axis_from_clicks(
                library, match_id, segment, labeled, land_table, length_m, width_m, source=chosen
            )
            if fitted is None:
                st.error(calibration_failure())
            else:
                st.rerun()

    calibration = saved_calibration(library, match_id, expected_source=analysis_source)
    if calibration is not None:
        used = st.session_state.get("calib_landmarks") or library.load_clicks(match_id)
        diagnosis = diagnose_fit(calibration)
        # A calibration is fitted against one camera-motion source's reference frame. If the footage now has a
        # gimbal log but the saved fit was built against the estimated chain (or vice versa), the pose it was fitted
        # to is not the pose the projection will use - measured at 24 m of error at the median - so say so and ask
        # for a refit rather than project through a stale pose.
        if segment is not None and calibration.pose_source != segment_pose_source(segment):
            st.warning(
                f"This calibration was fitted against the {calibration.pose_source} camera motion, but this "
                f"segment now uses the {segment_pose_source(segment)} motion. The two have different reference "
                "frames, so the fit is stale - press **Calibrate from these landmarks** to refit it against "
                "the current motion."
            )

        def name_of(index: int) -> str:
            """A click as the user knows it: its landmark, and the frame it came from."""
            if index < len(used):
                click = used[index]
                return f"`{click.get('label', '?')}`" + (f" (frame {click['frame']})" if "frame" in click else "")
            return f"click #{index + 1}"

        def _discard_calibration() -> None:
            library.clear_calibration(match_id)
            st.session_state.pop("calibration", None)
            st.session_state["calib_landmarks"] = []

        def _drop_outliers(indices: tuple[int, ...]) -> None:
            """Remove the clicks the fit flags, matched through the click list the fit itself used.

            The diagnosis indexes into the last fit's landmark list - the saved click list in order - so matching
            on ``(frame, label)`` rather than an index into the live session clicks keeps the removal right even
            after clicks were added on top of a stale calibration (a failed refit leaves the old residuals on
            screen). The refit then runs by itself: the click set changed, so the fit signature changed.
            """
            used_clicks = st.session_state.get("calib_landmarks") or library.load_clicks(match_id)
            wanted = {
                (int(used_clicks[i]["frame"]), str(used_clicks[i].get("label") or ""))
                for i in indices
                if i < len(used_clicks) and "frame" in used_clicks[i]
            }
            st.session_state["calib_points"] = [
                point
                for point in st.session_state.get("calib_points", [])
                if (point.get("frame"), str(point.get("label") or "")) not in wanted
            ]
            st.session_state.pop("calib_fit_signature", None)

        metric_col, discard_col = st.columns([2, 1])
        with metric_col:
            st.metric("Fit RMS error", f"{calibration.rms_error_m:.2f} m")
        with discard_col:
            st.write("")
            st.button(
                "Discard this calibration",
                help="Forget the saved fit and start the clicks again.",
                on_click=_discard_calibration,
            )

        if diagnosis.reason:
            st.error(f"**This fit is not usable.** The solver {diagnosis.reason}.")
        if diagnosis.too_few_agreeing:
            worst = ", ".join(name_of(i) for i in diagnosis.disagreeing) or "most of the clicks"
            st.error(
                f"**No pitch fits these clicks.** Only {len(diagnosis.agreeing)} of {len(calibration.residuals_m)} "
                f"agree with each other (within {diagnosis.tolerance_m:.1f} m). The ones that do not are {worst}. "
                "That is not click noise - it means at least one click is on the wrong thing, or two landmarks from "
                "different pitches have been mixed up. Check whether every click is on the *same* pitch: a venue "
                "with several goals in view makes it easy to click a corner or goal that belongs to the next pitch "
                "along, and nothing in the fit can tell you that except this."
            )
        elif diagnosis.disagreeing:
            odd = ", ".join(name_of(i) for i in diagnosis.disagreeing)
            st.warning(
                f"Fit is workable, but these clicks do not agree with the rest: {odd}. Re-click them, or drop "
                "them below and refit."
            )
        elif not diagnosis.reason:
            st.success("Good fit: every click agrees with the rest.")
        scale_note = format_scale_note(calibration)
        if scale_note:
            st.info(f"One thing to check: {scale_note}")

        x, y, z = (float(v) for v in calibration.position)
        st.caption(
            f"Camera at ({x:.0f}, {y:.0f}) m, {z:.2f} m high, lens scale {calibration.focal_scale:.2f}x."
        )
        if calibration.drift is not None and segment is not None and len(segment.time) > 1:
            anchors = calibration.drift.frames
            first, last = min(anchors), max(anchors)
            times = np.asarray(segment.time, dtype=np.float64)
            if last >= times.size:
                # The anchors are analysis-frame indices of the segment they were clicked on. A calibration saved
                # against another window (a different segment of the same video, say) cannot describe this one, and
                # indexing this segment's clock with its frames is how the page used to crash here.
                st.warning(
                    f"This match's calibration carries drift anchors up to frame {last}, but the analyzed window "
                    f"holds {times.size} frames - it was solved on a different segment. Re-click the landmarks on "
                    "this one (Step 2) to fit a correction that covers it."
                )
            else:
                analyzed_s = float(times[-1] - times[0])
                covered_s = float(times[last] - times[first])
                line = (
                    f"**Drift correction** is fitted: the chain is re-anchored at the {len(anchors)} clicked "
                    f"frame(s), from {_clock(times[first])} to {_clock(times[last])} of the analyzed window "
                    f"({_clock(covered_s)} of {_clock(analyzed_s)})."
                )
                if covered_s < 0.6 * analyzed_s:
                    line += (
                        " Between anchors it is interpolated, but outside them it is held flat - so the projection "
                        "can still slide away from the markings in the parts of the video with no clicks. Clicking "
                        "the same landmarks again later in the video (any frame works) pins it there too - **Place "
                        "the calibrated landmarks as markers** above does the clicking for you, and you nudge the "
                        "markers onto the markings instead."
                    )
                st.caption(line)
        st.dataframe(
            pd.DataFrame(
                {
                    "click": [f"#{i + 1}" for i in range(len(calibration.residuals_m))],
                    "landmark": [used[i]["label"] if i < len(used) else "?" for i in range(len(calibration.residuals_m))],
                    "frame": [used[i]["frame"] if i < len(used) else "?" for i in range(len(calibration.residuals_m))],
                    "error (m)": [round(r, 2) for r in calibration.residuals_m],
                    "used in fit": [
                        "no (outlier)" if i in calibration.excluded else "yes"
                        for i in range(len(calibration.residuals_m))
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if calibration.excluded:
            st.warning(
                "Dropped as outliers: "
                + ", ".join(f"#{i + 1}" for i in calibration.excluded)
                + " - those clicks look wrong; check them on the frame."
            )
        outliers = sorted(set(calibration.excluded) | set(diagnosis.disagreeing))
        if outliers:
            st.button(
                f"Drop {len(outliers)} outlier click(s) and refit",
                on_click=_drop_outliers,
                args=(tuple(outliers),),
                help="Remove the clicks this fit flags - the ones the solver dropped and the ones the diagnosis "
                     "says disagree with the rest - then refit from what is left. The cleared clicks leave the "
                     "map, so re-click anything that was actually right.",
            )

# --------------------------------------------------------------------------------------------------------------
# Step 3 - report
# --------------------------------------------------------------------------------------------------------------
st.header("Step 3 - Build the tactical report")
show_flash("step3_flash")

calibration = saved_calibration(library, match_id, expected_source=analysis_source)
if segment is None or calibration is None:
    st.info("Steps 1 and 2 are needed first: the report uses the segment results and the pitch calibration.")
else:
    # Two ways in: build just the report, or build it and let the background pipeline finish everything else -
    # the scans, the detectors and a final rebuild - so a rich report is one press. Both queue a job and return:
    # the page stays usable (and another match can be opened) while the work runs in its own process.
    report_busy = jobs.JOBS.has_active(match_id, kinds=("report", "build_and_detect"))
    detections_busy = _detections_alive(library, match_id) or jobs.JOBS.has_active(
        match_id, kinds=("build_and_detect", "detections")
    )
    build_col, build_all_col = st.columns(2)
    with build_col:
        build_clicked = st.button("Build report", type="primary", disabled=report_busy)
    with build_all_col:
        build_all_clicked = st.button(
            "Build report + run all detections",
            type="primary",
            disabled=report_busy or detections_busy,
            help=(
                "Queues the report and replay build, then the ball, whistle and shirt-number scans, the event "
                "detectors over what they find, and a final report + replay rebuild so the ball and the numbers "
                "land on it. Every stage skips finished work and resumes from its checkpoint, so a long ball scan "
                "can be closed and picked up again."
            ),
        )
    if build_clicked:
        _submit_report_job(library, match_id, segment_dir)
        st.session_state["step3_flash"] = (
            "success",
            "Report build queued. It runs in the background; the page stays usable and another match can be opened.",
        )
        st.rerun()
    if build_all_clicked:
        _submit_report_job(
            library,
            match_id,
            segment_dir,
            with_detections=True,
            video=str(chosen),
            strictness=float(st.session_state.get("whistle_strictness", MIN_PROMINENCE)),
            keep_voices=bool(st.session_state.get("whistle_reject_voices", True)),
        )
        st.session_state["step3_flash"] = (
            "success",
            "Report and detections queued. The job manager runs them in the background; this page stays usable.",
        )
        st.rerun()
    job_ui.match_jobs_panel(match_id)
    # One place to start and watch every scan - the ball, the whistle, the shirt numbers - and the all-detections
    # run that chains them (its button is above).
    scan_activity_section(library, match_id, segment, segment_dir, chosen)

    payload = st.session_state.get("report") or _report_payload(library, match_id)
    if payload:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Frames analyzed", payload["frames_analyzed"])
        metric_columns[1].metric("Player detections used", payload["detections_used"])
        metric_columns[2].metric("Players tracked", len(payload["players"]))
        metric_columns[3].metric("Tracks with a team", sum(1 for p in payload["players"] if p["team"] >= 0))
        team_names = _team_name_editor(library, match_id, payload)
        # The table says the name rather than the index, and leaves out the kit color: it is shown as a swatch just
        # above, where it can actually be seen. Possession is shown as the percentage a viewer expects rather than
        # the raw fraction stored in the report.
        team_rows = []
        for row in payload["teams"]:
            listed = {"team": team_name(int(row["team"]), team_names)}
            listed.update({key: value for key, value in row.items() if key not in ("kit_rgb", "possession_share")})
            listed["possession %"] = round(100.0 * float(row.get("possession_share") or 0.0), 1)
            team_rows.append(listed)
        st.dataframe(pd.DataFrame(team_rows), hide_index=True, width="stretch")
        st.caption(
            "Possession is the share of contested frames - frames with a labeled player near the ball - "
            "combined by time, so a quiet minute cannot outweigh a busy one. With a ball scan, 'near' is "
            "measured in the picture; otherwise the camera aim point is the fallback."
        )
        # The momentum curve is not drawn here any more: it is the timeline strip above the pitch, where it sits
        # on the same clock as the events and the replay. A second copy as a chart was a second thing to read.
        st.subheader("Player replay - pitch usage over time")
        replay_section(library, match_id, segment, segment_dir, chosen, calibration)
        for note in payload["notes"]:
            st.caption(f"- {note}")


# --------------------------------------------------------------------------------------------------------------
# Step 4 - export the match video
# --------------------------------------------------------------------------------------------------------------


def _stream_preview_base(port: int) -> str:
    """Where the browser should reach the footage stream: the configured base, else this app's own host.

    ``st.context.url`` is the URL the user opened the dashboard on (scheme + host + port), which is exactly what
    the browser needs to reach the stream server on its separate port. An explicit ``SOCCER_STREAM_BASE`` wins
    because a proxy may front the stream at a different path.
    """
    base = configured_base().strip()
    if base:
        return base.rstrip("/")
    url = st.context.url if hasattr(st, "context") else None
    if url:
        parsed = urlparse(url)
        if parsed.hostname:
            return f"{parsed.scheme or 'http'}://{parsed.hostname}:{int(port)}"
    return f"http://localhost:{int(port)}"


def _export_preview(
    match_id: str,
    *,
    start_s: float,
    width: int,
    fps: float,
    codec_label: str,
    team: bool,
    scoreboard: bool,
    ball: bool,
    numbers: bool,
    nonce: int,
) -> None:
    """Stream one live preview of the export's settings through the footage server.

    The preview is H.264 at up to 1080p so it plays smoothly in the browser and exercises the same overlay
    renderer as the file export. It starts at the analyzed window's beginning and runs to its end; the
    Restart button changes ``nonce``, which reloads the player from that start.
    """
    port = configured_port()
    if not is_reachable(port):
        st.warning(
            f"The footage stream is not running on port {port}. Start it with "
            f"`scripts/run_match_stream.py --port {port}` (the replay section has a Start button), then "
            "re-check this preview."
        )
        return
    preview_width = max(320, min(1920, int(width))) // 2 * 2
    preview_fps = max(1.0, min(float(fps), 60.0))
    url = preview_stream_url(
        match_id,
        base=_stream_preview_base(port),
        start_s=float(start_s),
        width=preview_width,
        fps=preview_fps,
        team=team,
        scoreboard=scoreboard,
        ball=ball,
        numbers=numbers,
        nonce=int(nonce),
    )
    st.video(url)
    st.caption(
        f"Live H.264 preview at {preview_width}px and up to {preview_fps:g} fps with the selected overlays and "
        f"match audio. The exported file uses {width}px, {fps:g} fps and {codec_label}."
    )


@st.fragment(run_every=POLL_SECONDS)
def _video_export_progress(library: MatchLibrary, match_id: str) -> None:
    """Live state of the background export; a finished/failed run is terminal until the next press."""
    status = _video_export_status(library, match_id)
    if not status:
        return
    state = str(status.get("state") or "")
    watch_key = f"video_export_watch::{match_id}"
    previous = st.session_state.get(watch_key)
    st.session_state[watch_key] = state
    if state == "queued":
        st.info("Queued - the job manager will start the export when a slot is free.")
    elif state == "running":
        progress = float(status.get("progress") or 0.0)
        st.progress(min(1.0, max(0.0, progress)), text=str(status.get("message") or "Exporting..."))
        if not _video_export_alive(library, match_id):
            st.warning(
                "The export process has not reported for several minutes, so it looks stopped. Press Export to "
                "start it again."
            )
    elif state == "done":
        output = Path(str(status.get("output") or ""))
        size = int(status.get("bytes") or (output.stat().st_size if output.exists() else 0))
        if size >= 1e9:
            size_label = f"{size / 1e9:.2f} GB"
        elif size:
            size_label = f"{size / 1e6:.1f} MB"
        else:
            size_label = ""
        st.success(f"Export ready: `{_location_label(output)}`" + (f" - {size_label}" if size_label else ""))
    elif state == "error":
        st.error(f"Export failed: {status.get('message') or 'unknown error'}")
        details = str(status.get("error") or "").strip()
        if details:
            with st.expander("Error details"):
                st.code(details)
    elif state in ("cancelled", "stale"):
        st.info(f"The export is not running: {status.get('message') or state}.")
    if previous == "running" and state != "running":
        st.rerun()


def video_export_section(
    library: MatchLibrary,
    match_id: str | None,
    segment,
    segment_dir: Path | None,
    source: str | None,
    probe,
) -> None:
    """The final workflow step: write the whole analyzed match to one MP4."""
    st.divider()
    st.header("Export the match video")
    if match_id is None or segment is None or segment_dir is None or not source:
        st.info("Select an analyzed match and finish Step 1 first.")
        return

    match_dir = library.path(match_id)
    calibration_ready = saved_calibration(library, match_id, expected_source=source) is not None
    replay_ready = (match_dir / "replay.json").exists() and (match_dir / "boxes.npz").exists()
    overlay_ready = calibration_ready and replay_ready
    ball_ready = (Path(segment_dir) / BALL_TRACK_FILE).exists()
    ball_ready = ball_ready and overlay_ready
    running = _video_export_alive(library, match_id) or jobs.JOBS.has_active(
        match_id, kinds=("video_export",)
    )

    st.caption(
        "With every overlay off this is the match as filmed: the analyzed window is re-encoded at the resolution, "
        "frame rate and codec chosen below. Overlays use the same renderer as the live annotated stream; they take "
        "longer because every frame is drawn in Python."
    )
    overlay_cols = st.columns(4)
    with overlay_cols[0]:
        team = st.checkbox(
            "Team boxes",
            value=False,
            key=f"export_team::{match_id}",
            disabled=not overlay_ready,
            help=(
                "Team-coloured boxes around each tracked player. Needs the pitch calibration (Step 2) and a "
                "built report/replay (Step 3)."
            ),
        ) and overlay_ready
    with overlay_cols[1]:
        scoreboard = st.checkbox(
            "Scoreboard",
            value=False,
            key=f"export_scoreboard::{match_id}",
            disabled=not overlay_ready,
            help=(
                "Team names, the confirmed-goal score and the match clock. Goals come from confirmed goal "
                "events; with none it reads 0-0. Needs the pitch calibration (Step 2) and a built report/replay "
                "(Step 3)."
            ),
        ) and overlay_ready
    with overlay_cols[2]:
        ball = st.checkbox(
            "Ball tracker",
            value=False,
            key=f"export_ball::{match_id}",
            disabled=not ball_ready,
            help=(
                "Highlights the positions where the ball scan actually detected the ball. Needs the pitch "
                "calibration, a built report/replay and the ball scan (Step 3)."
            ),
        ) and ball_ready
    with overlay_cols[3]:
        numbers = st.checkbox(
            "Jersey numbers",
            value=False,
            key=f"export_numbers::{match_id}",
            disabled=not overlay_ready,
            help=(
                "Draws the scanned shirt number or roster name above each tracked player. Needs the report/replay; "
                "a player with no number and no name has no chip."
            ),
        ) and overlay_ready
    if not overlay_ready:
        st.caption(
            "The overlay options need the pitch calibration and a built report. "
            + ("Register the pitch in Step 2. " if not calibration_ready else "")
            + ("Build the report in Step 3." if not replay_ready else "")
        )
    if not ball_ready:
        st.caption("The ball tracker needs the ball scan; run it in the Scan activity section of Step 3.")

    option_cols = st.columns(3)
    with option_cols[0]:
        resolution = st.selectbox(
            "Resolution",
            [
                f"Source ({probe.width} x {probe.height})",
                "1080p (1920 x 1080)",
                "720p (1280 x 720)",
            ],
            key=f"export_resolution::{match_id}",
        )
    with option_cols[1]:
        frame_rate = st.selectbox(
            "Frame rate",
            [f"Source ({probe.fps:g} fps)", "30 fps", "15 fps", "10 fps"],
            key=f"export_fps::{match_id}",
        )
    with option_cols[2]:
        codec_label = st.selectbox(
            "Format",
            ["H.264 (widely compatible)", "H.265 / HEVC (smaller files)"],
            key=f"export_codec::{match_id}",
        )
    width = probe.width if resolution.startswith("Source") else (1920 if resolution.startswith("1080") else 1280)
    fps = float(probe.fps) if frame_rate.startswith("Source") else float(frame_rate.split()[0])
    codec = "h264" if codec_label.startswith("H.264") else "hevc"
    tags = [
        name
        for name, enabled in (("teams", team), ("scoreboard", scoreboard), ("ball", ball), ("numbers", numbers))
        if enabled
    ]
    overlay_tag = "_".join(tags) or "source"
    resolution_tag = "source" if width == probe.width else f"{width}p"
    # A plain export at the source size would otherwise say "source" twice.
    parts = [overlay_tag] if resolution_tag == overlay_tag else [overlay_tag, resolution_tag]
    output = match_dir / "exports" / f"{match_id}_{'_'.join(parts)}_{fps:g}fps_{codec}.mp4"
    st.caption(f"Saved under `{_location_label(output.parent)}` as `{output.name}`.")

    preview = st.checkbox(
        "Stream a preview of this output",
        value=False,
        key=f"export_preview::{match_id}",
        help=(
            "Plays the analyzed window with the selected overlays through the footage stream before the export "
            "runs. The preview is H.264, capped at 1080p, and includes the match audio; the file uses the "
            "resolution, frame rate and format above."
        ),
    )
    if preview:
        preview_seq_key = f"export_preview_seq::{match_id}"
        if st.button("Restart preview", key=f"export_preview_restart::{match_id}"):
            st.session_state[preview_seq_key] = int(st.session_state.get(preview_seq_key, 0)) + 1
        _export_preview(
            match_id,
            start_s=float(segment.meta["start_s"]),
            width=width,
            fps=fps,
            codec_label=codec_label,
            team=team,
            scoreboard=scoreboard,
            ball=ball,
            numbers=numbers,
            nonce=int(st.session_state.get(preview_seq_key, 0)),
        )

    st.button(
        "Export the match video",
        type="primary",
        disabled=running,
        on_click=_start_video_export,
        args=(library, match_id, segment_dir, source, output, width, fps, codec, team, scoreboard, ball, numbers),
        help="Runs in the background; progress appears here and the page stays usable.",
    )
    _video_export_progress(library, match_id)


video_export_section(library, match_id, segment, segment_dir, analysis_source, probe)


# --------------------------------------------------------------------------------------------------------------
st.divider()
st.header("Archive contents")
rows = library.summaries()
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("Show artifacts for", [row["match_id"] for row in rows])
    st.dataframe(pd.DataFrame(library.artifacts(pick)), hide_index=True, width="stretch")
else:
    st.info("No matches archived yet.")

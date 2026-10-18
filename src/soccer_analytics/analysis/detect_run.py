"""Run the event detectors against everything a match has on disk.

The detectors themselves are pure (``event_detection.detect_events``); this module is the wiring that loads a
match's artifacts - the replay payload's player tracks, the ball scan's positions projected through the segment's
camera chain, the pitch calibration, the recorded whistles and the shirt numbers - and reconciles the result into
the match's event log.

Both the dashboard's own "Detect events" button and the background ``scripts/run_detections.py`` go through here,
so a candidate found from the page and one found by the background pass cannot disagree about times, teams or
which rows are replaced.

``MissingInput`` carries a human-readable reason the detectors cannot run yet; callers show it - as a warning
under the button, or as the run's error status - rather than letting a bare exception out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soccer_analytics.analysis.event_detection import detect_events, player_tracks_from_replay
from soccer_analytics.analysis.jerseys import merge_numbers
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import project_ball_track, segment_poses
from soccer_analytics.analysis.stage_a import SegmentData, load_segment

BALL_TRACK_FILE = "ball_track.json"  # written by scripts/run_ball_scan.py beside the segment


class MissingInput(Exception):
    """A prerequisite artifact is missing; the message says which one and what to do about it."""


def run_detection(
    library: MatchLibrary,
    match_id: str,
    segment_dir: str | Path,
    *,
    video: str | None = None,
    half_bounds: tuple[float, float, float] | None = None,
    segment: SegmentData | None = None,
    poses: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Detect events for one match and reconcile them into its event log.

    Returns ``{"detected": n, "added": a, "dropped": d}``: the candidates the detectors found, and how many the
    log gained and lost on the way in. The log is saved only when the detectors could run at all - every failure
    is a :class:`MissingInput` before anything is written.

    ``video`` names the recording the segment's times are on (its own ``meta`` supplies the default); it is what
    stamps detected events so readers can translate them onto the game clock. ``half_bounds`` is the game's
    ``(kick-off, half-time, full-time)`` in that recording's seconds, when the game has been marked.

    ``segment`` and ``poses`` let a caller that already has the loaded segment and its camera chain pass them in;
    the dashboard's cached loader does, because integrating a whole game's poses again per click is minutes of
    work. Without them the segment is loaded from ``segment_dir`` and the chain rebuilt here.
    """
    replay = library.load_replay(match_id)
    if replay is None:
        raise MissingInput("Build the report first - the detectors need the player tracks it saves.")
    try:
        payload = json.loads((Path(segment_dir) / BALL_TRACK_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        payload = None
    if not payload or not payload.get("frames"):
        raise MissingInput("Run the ball scan first - the detectors read the ball's track.")
    calibration = library.load_calibration(match_id)
    if calibration is None:
        raise MissingInput("Register the pitch first - the ball's track has to be projected onto it to be of use.")
    segment = segment if segment is not None else load_segment(segment_dir)
    if poses is None:
        q, focal = segment_poses(segment)
    else:
        q, focal = poses

    ball = project_ball_track(payload["frames"], calibration, q, focal)
    record = library.load(match_id)
    fps = float(segment.meta["fps"])
    # Frame indices are within the analyzed window, so the tracks' times need the window's own start offset -
    # the kick-off offset from Step 1. Without it every player-derived event lands `start_s` too early.
    players = player_tracks_from_replay(replay, fps, float(segment.meta.get("start_s", 0.0)))
    times = np.asarray(segment.time, dtype=np.float64)
    log = library.events(match_id)
    # The whistle candidates are what tell a penalty from a free kick; read before the reconcile, which is what
    # leaves them alone (they are another detector's rows).
    whistles = [event.time_s for event in log.events if event.source == "audio"]
    roster = library.load_roster(match_id)
    jerseys = library.load_jerseys(match_id)
    numbers = merge_numbers(
        [int(player["track_id"]) for player in replay.get("players", [])],
        auto={int(key): value for key, value in (jerseys.get("suggestions") or {}).items()},
        manual=roster,
    )
    detected = detect_events(
        ball[0],
        ball[1],
        times,
        players,
        (float(record.pitch_length_m), float(record.pitch_width_m)),
        whistles=whistles,
        numbers=numbers,
        half_bounds=half_bounds,
        swap_attack=bool(record.swap_attack),
        video=str(video or segment.meta.get("video", "")),
    )
    # Reconcile rather than append: the detector's own rows are replaced, so a re-run after a fix (or after a
    # better ball scan) cannot leave the previous run's rows behind. Whistles are a different source and are left
    # alone, and a candidate a human confirmed keeps its verdict.
    added, dropped = log.reconcile_detected(detected, source="ball")
    library.save_events(match_id, log)
    return {"detected": len(detected), "added": added, "dropped": dropped}

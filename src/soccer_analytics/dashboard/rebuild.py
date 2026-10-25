"""Build one match's report and replay from its analyzed segment.

This is the work the dashboard's **Build report** button used to do inside the Streamlit script thread - minutes of
projecting, tracking and replay assembly that made the whole page wait. It lives here so the job runner can call it
in a separate process, and the headless rebuild script can use the same code path.
"""

from __future__ import annotations

import json
from pathlib import Path

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.ball_lock import ball_image_from_scan
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import project_ball_track, project_segment, segment_poses
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.dashboard.replay import build_replay, roles_and_attack, track_boxes

BALL_TRACK_FILE = "ball_track.json"


def _ball_records(segment_dir: Path) -> list[dict] | None:
    """The segment's raw ball scan records, or None when it has not been scanned."""
    try:
        payload = json.loads((Path(segment_dir) / BALL_TRACK_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("frames") or None


def build_match_artifacts(
    library: MatchLibrary,
    match_id: str,
    *,
    segment_dir: str | Path | None = None,
    on_progress=None,
) -> dict:
    """Project, track and save ``report.json`` + ``replay.json``; returns a small summary for the job row.

    ``on_progress(stage, fraction, message)`` receives a fraction in ``[0, 1]`` across the whole job, so a caller
    can map it onto whatever bar it owns.
    """

    def progress(stage: str, fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(stage, max(0.0, min(1.0, float(fraction))), message)

    progress("load", 0.0, "Loading the match and its calibration...")
    record = library.load(match_id)
    calibration = library.load_calibration(match_id)
    if calibration is None:
        raise ValueError(f"no calibration saved for {match_id}: register the pitch first")
    directory = Path(segment_dir) if segment_dir else (
        Path(record.segments[0]) if record.segments else None
    )
    if directory is None or not directory.exists():
        raise ValueError("no analyzed segment for this match - run Step 1 first")

    segment = load_segment(directory)
    progress("load", 0.03, "Recovering the camera motion...")
    q, focal = segment_poses(segment, pan_model=calibration.pan_axis_model())

    detections = project_segment(
        segment,
        calibration,
        poses=(q, focal),
        on_progress=lambda fraction: progress("project", 0.05 + 0.35 * fraction, "Projecting detections..."),
    )
    ball_records = _ball_records(directory)
    ball_image = ball_image_from_scan(ball_records, len(segment.time)) if ball_records else None
    progress("report", 0.40, "Tracking players, teams and momentum...")
    report, assignment = stage_b.build_report(
        detections,
        pitch_length_m=record.pitch_length_m,
        pitch_width_m=record.pitch_width_m,
        match_frames=len(segment.time),
        ball_image=ball_image,
        on_progress=lambda fraction: progress(
            "report", 0.40 + 0.45 * fraction, "Tracking players, teams and momentum..."
        ),
    )

    progress("replay", 0.86, "Building the replay...")
    ball = project_ball_track(ball_records, calibration, q, focal) if ball_records else None
    roles, attack = roles_and_attack(
        report,
        detections,
        assignment,
        segment=segment,
        pitch_length_m=record.pitch_length_m,
        swap_attack=bool(record.swap_attack),
    )
    replay = build_replay(
        (record.pitch_length_m, record.pitch_width_m),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        record.team_names,
        ball=ball,
        ball_image=ball_image,
        team_colors=[metrics.kit_rgb for metrics in report.teams],
        camera_xy=detections.camera_xy,
        roles=roles,
        attack=attack,
    )

    progress("save", 0.96, "Saving the report and the replay...")
    report_payload = {
        "teams": [vars(team) for team in report.teams],
        "players": [
            {
                "track_id": player.track_id,
                "team": player.team,
                "observations": len(player.frame),
                "distance_m": round(player.distance_m, 1),
                "top_speed_kmh": round(float(player.speed_kmh.max()), 1),
                "xy": [[round(float(x), 1), round(float(y), 1)] for x, y in player.xy],
            }
            for player in report.players
        ],
        "momentum": report.momentum,
        "possession": report.possession,
        "notes": report.notes,
        "pitch": [record.pitch_length_m, record.pitch_width_m],
        "detections_used": report.detections_used,
        "frames_analyzed": report.frames_analyzed,
    }
    library.save_report(match_id, report_payload)
    library.save_replay(match_id, replay, boxes=track_boxes(report.players))
    progress("done", 1.0, "Report and replay ready.")
    return {
        "report": str(library.path(match_id) / "report.json"),
        "replay": str(library.path(match_id) / "replay.json"),
        "players": len(report.players),
        "teams": len(report.teams),
        "frames": report.frames_analyzed,
        "detections": report.detections_used,
        "possession": report.possession,
    }

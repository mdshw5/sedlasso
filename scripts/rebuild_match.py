"""Rebuild one match's report and replay from its segment, headlessly.

The dashboard's "Build report" button runs exactly this pipeline; this script is the same work as its own process,
for the cases the button cannot serve: a rebuild after the analysis code changed (the artifacts on disk were
produced by an older build and no longer describe what the code does), a machine that runs headless, or a repair
after a stale Streamlit process wrote artifacts from out-of-date modules.

What it does, in the order the pipeline runs:

1. loads the segment and its saved calibration,
2. rebuilds the camera chain (``segment_poses`` - the gimbal log supplies the orientation where there is one),
3. projects the detections to the pitch, runs the tracker, the stitcher and the kit clustering (``build_report``),
4. rebuilds the replay payload - player boxes included, which is what the centered clips and any annotated view
   follow - and re-projects the segment's ball scan (``ball_track.json``) into it,
5. saves both beside the match (``report.json``, ``replay.json``) atomically.

The jersey scan is *not* rerun (it is easyocr over the whole segment - the dashboard starts it in the
background); when the track numbering changes the scan's suggestions stop matching, so rerun
``scripts/extract_jerseys.py`` yourself afterwards if the numbers matter.

Usage::

    python scripts/rebuild_match.py --match 2026-10-03_game_16-28-37-784 [--segment <footage>/analysis/<id>/segments/<name>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.ball_lock import ball_image_from_scan
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.analysis.projection import project_ball_track, project_segment, segment_poses
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.dashboard.replay import build_replay, roles_and_attack, track_boxes, track_boxes

BALL_TRACK_FILE = "ball_track.json"


def _ball_records(segment_dir: Path) -> list[dict] | None:
    """The segment's raw ball scan records, or None when it has not been scanned."""
    try:
        payload = json.loads((segment_dir / BALL_TRACK_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    records = payload.get("frames") or []
    return records or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", required=True, help="match id, or the path to its analysis directory")
    parser.add_argument("--segment", default=None, help="segment directory (default: the match's first one)")
    parser.add_argument(
        "--root",
        default=None,
        help="single matches root; default: every analysis beside the footage roots (plus the legacy data/matches)",
    )
    args = parser.parse_args()

    root = args.root
    library = MatchLibrary(root)
    record = library.load(args.match)
    calibration = library.load_calibration(args.match)
    if calibration is None:
        print(f"no calibration saved for {args.match}: register the pitch first", file=sys.stderr)
        return 1
    segment_dir = Path(args.segment) if args.segment else Path(record.segments[0])
    if not segment_dir.exists():
        print(f"segment directory not found: {segment_dir}", file=sys.stderr)
        return 1

    started = time.time()
    segment = load_segment(segment_dir)
    q, focal = segment_poses(segment, pan_model=calibration.pan_axis_model())
    print(f"loaded {segment_dir.name} ({time.time() - started:.0f}s)", flush=True)

    detections = project_segment(segment, calibration, poses=(q, focal))
    print(f"projected {len(detections.frame)} detections ({time.time() - started:.0f}s)", flush=True)

    ball_records = _ball_records(segment_dir)
    ball_image = ball_image_from_scan(ball_records, len(segment.time)) if ball_records else None
    report, assignment = stage_b.build_report(
        detections,
        pitch_length_m=record.pitch_length_m,
        pitch_width_m=record.pitch_width_m,
        match_frames=len(segment.time),
        ball_image=ball_image,
    )
    print(f"tracked {len(report.players)} players ({time.time() - started:.0f}s)", flush=True)

    roles, attack = roles_and_attack(
        report, detections, assignment, segment=segment, pitch_length_m=record.pitch_length_m,
        swap_attack=bool(record.swap_attack),
    )
    replay = build_replay(
        (record.pitch_length_m, record.pitch_width_m),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        record.team_names,
        ball=project_ball_track(ball_records, calibration, q, focal) if ball_records else None,
        ball_image=ball_image,
        team_colors=[metrics.kit_rgb for metrics in report.teams],
        camera_xy=detections.camera_xy,
        roles=roles,
        attack=attack,
    )
    boxes = track_boxes(report.players)
    boxed = len(boxes)

    # The same payload shape the dashboard saves, so every reader sees one record format.
    library.save_report(
        args.match,
        {
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
        },
    )
    library.save_replay(args.match, replay, boxes=track_boxes(report.players))
    print(
        f"saved report + replay ({time.time() - started:.0f}s): {len(replay['players'])} players on the field, "
        f"{boxed} with image boxes, ball track {'yes' if replay.get('ball') else 'no'}",
        flush=True,
    )
    by_time = sorted(replay["players"], key=lambda p: -p["stats"]["time_s"])[:5]
    for player in by_time:
        stats = player["stats"]
        print(
            f"  track {player['track_id']}: {stats['time_s']:.0f}s seen, team {player['team']}, "
            f"boxes {len(boxes.get(str(player['track_id']), ()))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

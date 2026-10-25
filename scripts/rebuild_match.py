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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.dashboard.rebuild import build_match_artifacts


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
    last = {"stage": "", "pct": -1}

    def progress(stage: str, fraction: float, message: str) -> None:
        pct = int(round(100.0 * max(0.0, min(1.0, fraction))))
        if stage != last["stage"] or pct // 5 != last["pct"]:
            last.update(stage=stage, pct=pct // 5)
            print(f"[{pct:3d}%] {message} ({time.time() - started:.0f}s)", flush=True)

    result = build_match_artifacts(
        library, args.match, segment_dir=segment_dir, on_progress=progress
    )
    replay = library.load_replay(args.match) or {}
    boxes = library.load_replay_boxes(args.match)
    boxed = len(boxes)
    print(
        f"saved report + replay ({time.time() - started:.0f}s): {len(replay.get('players', []))} players on the "
        f"field, {boxed} with image boxes, ball track {'yes' if replay.get('ball') else 'no'}, "
        f"possession source {(result.get('possession') or {}).get('source', 'camera aim')}",
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

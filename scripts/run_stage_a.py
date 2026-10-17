"""Run Stage A for one segment, as its own process so the dashboard stays responsive.

Usage::

    python scripts/run_stage_a.py --video /path/match.MP4 --out <footage>/analysis/<id>/segments/<segment> \
        [--start 300] [--duration 600]

``--duration 0`` (the default) analyzes from the offset to the end of the video. Progress is written to
``<out>/status.json`` by `analyze_segment`, which is what the dashboard polls. Resuming is implicit: the run starts
from the last completed chunk, so re-running this command after an interruption continues instead of starting over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_analytics.analysis.stage_a import (
    ANALYSIS_FPS,
    DETECT_WIDTH,
    SegmentConfig,
    analyze_segment,
    resolve_weights,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the heavy per-segment analysis pass (Stage A).")
    parser.add_argument("--video", required=True, help="source video file")
    parser.add_argument("--out", required=True, help="output directory for this segment")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in the source video (seconds)")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to analyze; 0 = to the end of the video")
    parser.add_argument(
        "--fps",
        type=float,
        default=ANALYSIS_FPS,
        help="analysis frame rate (detections per second; default 15 - 60 and 30 fps sources both divide by it)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DETECT_WIDTH,
        help="analysis frame width for detection; 0 (default) = the source's own width (full resolution)",
    )
    parser.add_argument("--chunk-frames", type=int, default=300, help="frames per checkpoint file")
    parser.add_argument(
        "--no-log-motion",
        action="store_true",
        help="ignore the gimbal log for tracker motion compensation (use BoT-SORT's optical-flow GMC)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="detection weights (default: a local data/models/*.pt if there is one, else stock yolov8n.pt)",
    )
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU for detection")
    args = parser.parse_args()

    config = SegmentConfig(
        fps=args.fps,
        detect_width=args.width,
        chunk_frames=args.chunk_frames,
        weights=resolve_weights(args.weights),
        device="cpu" if args.cpu else 0,
        use_log_motion=not args.no_log_motion,
    )
    status = analyze_segment(
        args.video,
        args.out,
        config=config,
        start_s=args.start,
        duration_s=args.duration,
        on_progress=lambda s: print(
            f"chunk {s['chunk']}/{s['total_chunks']} frames {s['frames_done']}/{s['total_frames']} "
            f"{s['fps']:.1f} fps eta {s['eta_s']}s",
            flush=True,
        ),
    )
    print(f"finished: {status['state']}")
    return 0 if status["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())

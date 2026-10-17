"""Export a match's analyzed window as one video file, in the background.

The dashboard's final section starts this process, then polls ``video_export.json`` beside the match. With every
overlay off it takes the direct ffmpeg path (seek each clip, scale/fps, concat, encode), which is the only sane
way to export an 80-minute 4K game. Any overlay switches to the annotated renderer used by the live stream, so
what is exported is exactly what the stream would draw.

The file is written to a ``.part`` name and renamed on success. A killed export therefore leaves no half-written
file that looks finished.

Usage::

    python scripts/run_video_export.py --match <id> --segment <dir> --source <game.json|video> \
        --output <file.mp4> --width 1920 --fps 30 --codec h264 [--team] [--scoreboard] [--ball] [--numbers]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.library import MatchLibrary  # noqa: E402
from soccer_analytics.dashboard.export import EXPORT_CODECS, export_source_window  # noqa: E402
from soccer_analytics.dashboard.stream import AnnotatedMatch  # noqa: E402
from soccer_analytics.dashboard.video import encode_clip  # noqa: E402

STATUS_FILE = "video_export.json"


class Status:
    """The progress file the dashboard polls; written atomically and throttled like the scan statuses."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload: dict = {
            "state": "running",
            "stage": "prepare",
            "progress": 0.0,
            "message": "Starting...",
            "started": time.time(),
        }
        self._written = 0.0

    def update(self, *, force: bool = False, **changes) -> None:
        self.payload.update(changes)
        self.payload["updated"] = time.time()
        now = time.monotonic()
        if not force and now - self._written < 0.4:
            return
        self._written = now
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.payload))
        tmp.replace(self.path)

    def progress(self, fraction: float) -> None:
        value = max(0.0, min(1.0, float(fraction)))
        if value - float(self.payload.get("progress", 0.0)) >= 0.01 or value >= 1.0:
            self.update(progress=value, message=f"Exporting... {value * 100:.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True)
    parser.add_argument("--segment", required=True, help="the analyzed segment directory")
    parser.add_argument("--source", required=True, help="the game manifest or single video to read")
    parser.add_argument("--output", required=True, help="final .mp4 path")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--codec", choices=EXPORT_CODECS, default="h264")
    parser.add_argument("--team", action="store_true", help="draw team boxes")
    parser.add_argument("--scoreboard", action="store_true", help="draw the scoreboard")
    parser.add_argument("--ball", action="store_true", help="highlight detected ball positions")
    parser.add_argument("--numbers", action="store_true", help="draw jersey-number / name chips")
    args = parser.parse_args()

    match_dir = MatchLibrary().path(args.match)
    status = Status(match_dir / STATUS_FILE)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(f"{output.stem}.part{output.suffix}")
    part.unlink(missing_ok=True)

    try:
        segment_dir = Path(args.segment)
        meta = json.loads((segment_dir / "meta.json").read_text())
        start_s = float(meta["start_s"])
        end_s = float(meta["end_s"])
        duration_s = end_s - start_s
        if duration_s <= 0 or args.width <= 0 or args.fps <= 0:
            raise ValueError("the export window, width and frame rate must all be positive")

        layers = {
            "team": bool(args.team),
            "scoreboard": bool(args.scoreboard),
            "ball": bool(args.ball),
            "numbers": bool(args.numbers),
        }
        status.update(
            state="running",
            stage="render" if any(layers.values()) else "encode",
            progress=0.0,
            output=str(output),
            width=int(args.width),
            fps=float(args.fps),
            codec=args.codec,
            overlays=layers,
            start_s=start_s,
            end_s=end_s,
            message="Starting the annotated render..." if any(layers.values()) else "Starting the encode...",
            force=True,
        )

        if any(layers.values()):
            match = AnnotatedMatch.load(args.match, segment=segment_dir)
            overlays = {
                "pitch": False,
                "boxes": bool(args.team),
                "numbers": bool(args.numbers),
                "ball": bool(args.ball),
                "hud": False,
                "debug": False,
                "scoreboard": bool(args.scoreboard),
            }
            encode_clip(
                match,
                start_s=start_s,
                duration_s=duration_s,
                fps=float(args.fps),
                width=int(args.width),
                overlays=overlays,
                output=part,
                codec=args.codec,
                audio=True,
                on_progress=status.progress,
            )
        else:
            export_source_window(
                args.source,
                start_s,
                duration_s,
                part,
                width=int(args.width),
                fps=float(args.fps),
                codec=args.codec,
                on_progress=status.progress,
            )

        if not part.exists() or part.stat().st_size == 0:
            raise RuntimeError("the encoder produced no output")
        part.replace(output)
        status.update(
            state="done",
            stage="done",
            progress=1.0,
            output=str(output),
            bytes=int(output.stat().st_size),
            finished=time.time(),
            message=f"Export ready: {output.name}",
            force=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - the status file is the user-facing report
        part.unlink(missing_ok=True)
        message = f"{type(exc).__name__}: {exc}"
        status.update(
            state="error",
            stage="error",
            message=message,
            error=traceback.format_exc(limit=4),
            finished=time.time(),
            force=True,
        )
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

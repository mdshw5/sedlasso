"""Fit the camera's true pan axis from landmark clicks and automatic line observations.

The gimbal log gives the commanded pan, but a 4 m mast is not perfectly plumb: the camera's real pan axis is
rolled or tilted a little in its own frame. That error depends on the camera's *heading*, so the time-based drift
correction cannot remove it - clicks at one heading fit, clicks at another are rejected as outliers, and the
projected pitch slides against the white lines during a pan. This script fits a small roll/tilt/yaw-scale
neighborhood around the physical prior, using both the saved clicks and (optionally) white-line observations,
then refits the calibration through the fitted model and writes both the model sidecar and the calibration.

Usage::

    python scripts/fit_pan_axis.py --match <match id or analysis directory>
    python scripts/fit_pan_axis.py --match <...> --lines-file <line_observations.json>
    python scripts/fit_pan_axis.py --match <...> --detect-lines

``--detect-lines`` scans the whole segment once (a few minutes on a long game) and caches the observations beside
the match as ``line_observations.json``; later runs reuse the cache. Without line observations the fit uses the
clicks alone, which is enough to remove a static axis/scale error but is more sensitive to a few bad clicks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.library import MatchLibrary  # noqa: E402
from soccer_analytics.analysis.projection import fit_segment_pan_axis, segment_poses  # noqa: E402
from soccer_analytics.analysis.stage_a import load_segment  # noqa: E402
from soccer_analytics.dashboard.pitch_clicks import landmark_table  # noqa: E402
from soccer_analytics.geometry.gimbal_motion import aligned_log_for_segment, save_pan_model  # noqa: E402
from soccer_analytics.geometry.pitch_calibration import Landmark  # noqa: E402
from soccer_analytics.geometry.pitch_lines import (  # noqa: E402
    LineObservation,
    detect_line_observations,
    observations_to_landmarks,
)

CAMERA_HEIGHT_M = 4.0
LINE_FILE = "line_observations.json"


def _landmarks_from_clicks(clicks: list[dict], table: dict[str, tuple[float, float]]) -> list[Landmark]:
    out: list[Landmark] = []
    for click in clicks:
        label = click.get("label")
        if not label or label not in table:
            continue
        x, y = table[label]
        out.append(
            Landmark(
                frame=int(click["frame"]),
                u=float(click["u"]),
                v=float(click["v"]),
                pitch_x=float(x),
                pitch_y=float(y),
                label=str(label),
            )
        )
    return out


def _load_line_landmarks(path: Path) -> list[Landmark]:
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
        for row in json.loads(path.read_text())
    ]
    return observations_to_landmarks(observations)


def _print_click_residuals(calibration, labels: list[str], title: str) -> None:
    print(f"{title}: RMS {calibration.rms_error_m:.3f} m, excluded {len(calibration.excluded)}")
    for index, (label, residual) in enumerate(zip(labels, calibration.residuals_m)):
        flag = " EXCLUDED" if index in calibration.excluded else ""
        print(f"  {index:2d} {label:28s} {residual:6.2f} m{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", required=True, help="match id, or the path to its analysis directory")
    parser.add_argument("--segment", default=None, help="segment directory (default: the match's first one)")
    parser.add_argument("--lines-file", default=None, help=f"line observations JSON (default: <match>/{LINE_FILE})")
    parser.add_argument("--detect-lines", action="store_true", help="scan the video for white-line observations")
    parser.add_argument("--line-stride", type=int, default=200, help="frame stride for --detect-lines")
    parser.add_argument("--line-start", type=int, default=0, help="first analysis frame to scan for lines")
    parser.add_argument("--line-end", type=int, default=None, help="one past the last analysis frame to scan")
    parser.add_argument("--height", type=float, default=CAMERA_HEIGHT_M, help="rig height in meters (known)")
    parser.add_argument("--dry-run", action="store_true", help="report the fit without writing calibration/sidecar")
    args = parser.parse_args()

    started = time.time()
    library = MatchLibrary()
    record = library.load(args.match)
    match_dir = library.path(args.match)
    calibration = library.load_calibration(args.match)
    if calibration is None:
        print(f"no calibration saved for {args.match}: register the pitch first", file=sys.stderr)
        return 1
    segment_dir = Path(args.segment) if args.segment else Path(record.segments[0])
    segment = load_segment(segment_dir)
    print(f"loaded {segment_dir.name} in {time.time() - started:.0f}s", flush=True)

    aligned = aligned_log_for_segment(segment)
    if aligned is None:
        print("this segment has no gimbal log; there is no pan model to fit", file=sys.stderr)
        return 1

    table = landmark_table(
        record.pitch_length_m,
        record.pitch_width_m,
        format_name=record.format,
        goal_width_m=getattr(record, "goal_width_m", None),
    )
    click_landmarks = _landmarks_from_clicks(library.load_clicks(args.match), table)
    if len(click_landmarks) < 4:
        print(f"only {len(click_landmarks)} usable clicks; need at least 4", file=sys.stderr)
        return 1
    print(f"{len(click_landmarks)} clicks", flush=True)

    lines_file = Path(args.lines_file) if args.lines_file else match_dir / LINE_FILE
    line_landmarks: list[Landmark] = []
    if args.detect_lines:
        print("scanning for line observations (one decode pass)...", flush=True)
        observations = detect_line_observations(
            segment,
            calibration,
            *segment_poses(segment),
            length_m=record.pitch_length_m,
            width_m=record.pitch_width_m,
            stride=args.line_stride,
            start_frame=args.line_start,
            end_frame=args.line_end,
            on_progress=lambda fraction: None,
        )
        line_landmarks = observations_to_landmarks(observations)
        payload = [
            {
                "frame": o.frame,
                "u": o.u,
                "v": o.v,
                "pitch_x": o.pitch_x,
                "pitch_y": o.pitch_y,
                "direction": list(o.direction),
                "label": o.label,
                "contrast": o.contrast,
            }
            for o in observations
        ]
        lines_file.parent.mkdir(parents=True, exist_ok=True)
        lines_file.write_text(json.dumps(payload))
        print(f"wrote {len(observations)} line observations to {lines_file}", flush=True)
    elif lines_file.exists():
        line_landmarks = _load_line_landmarks(lines_file)
        print(f"{len(line_landmarks)} line observations from {lines_file}", flush=True)
    else:
        print("no line observations supplied; fitting from clicks alone", flush=True)

    all_landmarks = click_landmarks + line_landmarks
    fit_started = time.time()
    result = fit_segment_pan_axis(segment, all_landmarks, fixed_height_m=args.height)
    if result is None:
        print("the 3-DOF fit could not place a camera through these landmarks", file=sys.stderr)
        return 1
    refined, _q_model, _focal_model, fit = result
    print(
        f"3-DOF fit: trimmed RMS {fit.rms_m:.3f} m, {fit.evaluations} evaluations in "
        f"{time.time() - fit_started:.0f}s",
        flush=True,
    )
    print(f"  axis {np.round(fit.model.axis, 4)} scale {fit.model.scale:.4f}", flush=True)
    click_count = len(click_landmarks)
    click_residuals = refined.residuals_m[:click_count]
    click_excluded = tuple(i for i in refined.excluded if i < click_count)
    click_rms = float(
        np.sqrt(np.mean([r for i, r in enumerate(click_residuals) if i not in click_excluded] or [0.0]))
    )
    print(f"refit calibration: click RMS {click_rms:.3f} m, excluded clicks {len(click_excluded)}", flush=True)
    _print_click_residuals(refined, [lm.label for lm in click_landmarks], "refit clicks")

    if args.dry_run:
        print("dry run: nothing written", flush=True)
        return 0

    save_pan_model(segment, fit.model)
    library.save_calibration(args.match, refined)
    diagnostic = {
        "fitted_model": fit.model.to_json(),
        "trimmed_rms_m": fit.rms_m,
        "click_rms_m": click_rms,
        "click_residuals_m": list(click_residuals),
        "click_excluded": list(click_excluded),
        "line_observations": len(line_landmarks),
        "height_m": args.height,
    }
    (match_dir / "pan_axis_fit.json").write_text(json.dumps(diagnostic, indent=2))
    print(f"saved {match_dir / 'calibration.json'}, {segment_dir / 'pan_model.json'}, and pan_axis_fit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

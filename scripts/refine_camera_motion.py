"""Re-estimate a segment's camera-motion chain with a large-motion-aware step estimator.

The analysis pass (``stage_a``) estimates each frame's motion against the last good frame with sparse optical flow
(Lucas-Kanade) first, and only falls back to descriptor matching (SIFT) when LK *fails*. LK is a local
linearization of the image, so on a fast pan it systematically *under*-estimates the rotation - it succeeds, so the
fallback never runs, and the small bias is integrated into the chain. Over a pan the projected pitch then lags
behind the real markings: the overlay trails the white lines.

This script re-decodes the segment's video at the analysis fps and re-estimates every step, preferring descriptor
matching whenever the motion is large enough for LK's linearization to matter. It re-integrates the chain and
writes the refined ``step``/``focal``/``ok`` arrays back into the segment's chunks, so every downstream consumer
(projection, calibration, replay) picks them up without re-running Stage A. Times, detections and kit descriptors
are preserved untouched.

Only the motion is touched. The pitch calibration is a separate fit on top of the chain, so re-running this does
not invalidate a saved calibration - it makes the chain the calibration was fitted against more accurate.

Usage::

    python scripts/refine_camera_motion.py --segment <footage>/analysis/<id>/segments/<dir> --dry-run
    python scripts/refine_camera_motion.py --segment <footage>/analysis/<id>/segments/<dir>

``--dry-run`` reports what would change without writing. Pass ``--calibration`` to also report the reprojection
error of the calibration's clicked landmarks before and after, which is the number that matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from soccer_analytics.analysis.stage_a import chunk_path, completed_chunks, load_segment  # noqa: E402
from soccer_analytics.geometry.camera_motion import (  # noqa: E402
    DEFAULT_FOCAL,
    FOCAL_RANGE,
    CameraMotionTracker,
    MotionStep,
    _unit,
    decompose_step,
    estimate_step_lk,
    estimate_step_sift,
    integrate_poses,
    normalizer,
    overlay_mask,
    step_is_plausible,
)
from soccer_analytics.ingest.source import open_reader  # noqa: E402

# Above this per-step rotation (degrees) LK's linearization is the dominant error, so descriptor matching is
# preferred. Measured on the reference game: the median step is 0.08 deg and the 90th percentile is 1.7 deg, so
# this only fires on the pans that actually cause the lag.
LARGE_MOTION_DEG = 1.5
# A step is only accepted with this much support, matching the tracker's own gates.
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.30
# How far a step may be from an exact rotation+zoom before it is rejected, matching the tracker's own gate.
MAX_SPREAD = 0.12


def _rotation_deg(step: np.ndarray, focal: float, aspect: float) -> float:
    """Rotation magnitude of a step, in degrees (0 when it cannot be decomposed)."""
    try:
        rotation = decompose_step(step, focal, aspect).rotation
    except Exception:
        return 0.0
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _usable(step: MotionStep | None) -> bool:
    return step is not None and step.inliers >= MIN_INLIERS and step.inlier_ratio >= MIN_INLIER_RATIO


def choose_step(
    lk: MotionStep | None, sift: MotionStep | None, lk_deg: float, large_motion_deg: float
) -> tuple[MotionStep | None, str]:
    """Pick the step to trust, and say which estimator it came from.

    LK is accurate and cheap for small motion, so it is used as-is. When it is unusable, or the motion is large
    enough that its linearization under-estimates the rotation, the descriptor match is preferred - and LK is only
    fallen back to if the descriptor match is itself unusable. Returns ``(None, "lost")`` when neither is usable.
    """
    if _usable(lk) and lk_deg < large_motion_deg:
        return lk, "lk"
    if _usable(sift):
        return sift, "sift"
    if _usable(lk):
        return lk, "lk"
    return None, "lost"


def validate_step(normalized: np.ndarray, focal: float, aspect: float):
    """Decompose a candidate step, or return ``None`` when it is not a rotation+zoom a real lens could produce.

    This is the tracker's own gate (``CameraMotionTracker._try_step``) and it is not optional: a step that is not a
    valid rotation+zoom - a shear, or a focal outside the plausible range - is a bad fit, and accepting it lets the
    error accumulate over thousands of frames. Skipping this check is what made the first full refinement diverge
    (the chain walked 10.8 deg off and the landmark reprojection got *worse*, 545 -> 741 px).
    """
    try:
        rotation = decompose_step(normalized, focal, aspect)
    except Exception:
        return None
    if rotation.spread > MAX_SPREAD or not (FOCAL_RANGE[0] <= rotation.focal <= FOCAL_RANGE[1]):
        return None
    return rotation


def refine_steps(
    segment,
    *,
    motion_width: int,
    large_motion_deg: float = LARGE_MOTION_DEG,
    max_frames: int = 0,
    on_progress=None,
) -> tuple[list[np.ndarray | None], dict]:
    """Re-estimate every step, preferring SIFT when the motion is large.

    Returns the refined steps (``None`` where no motion could be measured) and a small report of how often each
    estimator was chosen.
    """
    meta = segment.meta
    aspect = segment.aspect
    motion_height = int(round(motion_width * meta["height"] / meta["width"])) // 2 * 2
    mask = overlay_mask((motion_height, motion_width))
    norm = normalizer(motion_width)
    norm_inv = np.linalg.inv(norm)

    reader = open_reader(
        meta["video"],
        fps=float(meta["fps"]),
        width=motion_width,
        start_s=float(meta["start_s"]),
        duration_s=float(meta["end_s"]) - float(meta["start_s"]),
    )

    steps: list[np.ndarray | None] = []
    report = {"lk": 0, "sift": 0, "lost": 0, "large_motion": 0}
    good_gray: np.ndarray | None = None
    focal = float(segment.focal[0]) if len(segment.focal) else DEFAULT_FOCAL
    total = len(segment.time)
    limit = total if max_frames <= 0 else min(total, max_frames)

    for index, (_time, frame) in enumerate(reader.frames()):
        if index >= limit:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if good_gray is None:
            good_gray = gray
            steps.append(None)  # the reference frame carries no motion
            continue

        lk = estimate_step_lk(good_gray, gray, mask)
        lk_deg = _rotation_deg(_unit(norm @ lk.homography @ norm_inv), focal, aspect) if _usable(lk) else 0.0

        # The motion magnitude decides which estimator to trust. LK is accurate and cheap for small motion, so it
        # is used as-is; only when it is unusable or the motion is large enough for its linearization to
        # under-estimate the rotation is the (much slower) descriptor match run.
        if _usable(lk) and lk_deg < large_motion_deg:
            chosen, source = lk, "lk"
        else:
            chosen, source = choose_step(lk, estimate_step_sift(good_gray, gray, mask), lk_deg, large_motion_deg)
        report[source] += 1
        if source == "sift" and lk_deg >= large_motion_deg:
            report["large_motion"] += 1

        if chosen is None:
            steps.append(None)
            report["lost"] += 1
            continue

        normalized = _unit(norm @ chosen.homography @ norm_inv)
        if not step_is_plausible(normalized):
            steps.append(None)
            report["lost"] += 1
            continue

        # The same validation the tracker applies: the step must decompose to a rotation+zoom a real lens could
        # produce. Skipping this is what let the first full run diverge - a step that is not a valid rotation+zoom
        # was accepted, and the error accumulated over thousands of frames.
        rotation = validate_step(normalized, focal, aspect)
        if rotation is None:
            steps.append(None)
            report["lost"] += 1
            continue

        steps.append(normalized)
        good_gray = gray
        focal = rotation.focal  # keep the large-motion test and the next decomposition current

        if on_progress is not None and (index % 200 == 0 or index == limit - 1):
            on_progress((index + 1) / max(1, limit))
    return steps, report


def _reprojection_error(
    calibration, clicks: list[dict], table: dict[str, tuple[float, float]], q: np.ndarray, focal: np.ndarray
) -> float:
    """RMS pixel error of the calibration's clicked landmarks, in pixels at a 1920-wide frame.

    Clicks are stored as ``{frame, u, v, label}``; the label names a landmark in ``table``, which supplies the
    pitch position the click claims to be.
    """
    from soccer_analytics.geometry.pitch_calibration import pitch_to_pixels

    errors = []
    for click in clicks:
        frame = int(click["frame"])
        label = click.get("label")
        if frame >= len(q) or label not in table:
            continue
        uv, in_front = pitch_to_pixels(
            calibration, np.array([table[label]]), q[frame], float(focal[frame])
        )
        if not in_front[0] or not np.isfinite(uv[0]).all():
            continue
        errors.append(float(np.linalg.norm(uv[0] - np.array([click["u"], click["v"]]))) * 1920.0)
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def _write_back(segment_dir: Path, steps: list[np.ndarray | None], focals: np.ndarray, ok: np.ndarray) -> None:
    """Rewrite only the step/focal/ok arrays of every chunk, atomically, preserving everything else."""
    chunks = completed_chunks(segment_dir)
    offset = 0
    for index in range(chunks):
        path = chunk_path(segment_dir, index)
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        count = len(payload["time"])
        block = slice(offset, offset + count)
        payload["step"] = np.asarray(
            [steps[i] if steps[i] is not None else np.eye(3) for i in range(block.start, block.stop)],
            dtype=np.float64,
        ).reshape(-1, 3, 3)
        payload["focal"] = np.asarray(focals[block], dtype=np.float32)
        payload["ok"] = np.asarray(ok[block], dtype=bool)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        offset += count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True, type=Path)
    parser.add_argument("--motion-width", type=int, default=960, help="width the motion is estimated at")
    parser.add_argument("--large-motion-deg", type=float, default=LARGE_MOTION_DEG)
    parser.add_argument("--max-frames", type=int, default=0, help="only refine the first N frames (0 = all)")
    parser.add_argument("--calibration", type=Path, default=None, help="report landmark reprojection before/after")
    parser.add_argument("--clicks", type=Path, default=None, help="clicks.json for the calibration report")
    parser.add_argument("--dry-run", action="store_true", help="report only; do not write the chunks")
    args = parser.parse_args()

    segment_dir = args.segment if args.segment.is_absolute() else REPO_ROOT / args.segment
    segment = load_segment(segment_dir)
    aspect = segment.aspect
    focal0 = float(segment.focal[0]) if len(segment.focal) else DEFAULT_FOCAL

    print(f"segment {segment_dir.name}: {len(segment.time)} frames, {completed_chunks(segment_dir)} chunk(s)")

    # Before: the chain as stored.
    q_before, focal_before = integrate_poses(
        [None if not ok else step for ok, step in zip(segment.ok, segment.step)],
        focal0,
        aspect,
        known_focals=segment.focal,
    )

    print("re-estimating motion...")
    steps, report = refine_steps(
        segment,
        motion_width=args.motion_width,
        large_motion_deg=args.large_motion_deg,
        max_frames=args.max_frames,
        on_progress=lambda f: print(f"  {f:.0%}", end="\r", flush=True),
    )
    print()
    print(
        f"estimator: lk {report['lk']}, sift {report['sift']} (of which {report['large_motion']} large-motion), "
        f"lost {report['lost']}"
    )

    # After: re-integrate from the refined steps. The focal is re-searched from the refined steps, so it is written
    # back too - the stored focal was derived from the old steps.
    q_after, focal_after = integrate_poses(steps, focal0, aspect)
    ok_after = np.array([step is not None for step in steps], dtype=bool)

    # How far the chain moved, as a sanity check that the refinement is not a no-op or a wild swing.
    span = min(len(q_before), len(q_after))
    moved = [
        float(np.degrees(np.arccos(np.clip((np.trace(q_before[i].T @ q_after[i]) - 1) / 2, -1, 1))))
        for i in range(span)
    ]
    print(f"chain change: median {np.median(moved):.3f} deg, p90 {np.percentile(moved, 90):.3f} deg, max {max(moved):.3f} deg")

    if args.calibration is not None:
        from soccer_analytics.dashboard.pitch_clicks import landmark_table
        from soccer_analytics.geometry.pitch_calibration import PitchCalibration

        calibration = PitchCalibration.from_json(json.loads(args.calibration.read_text()))
        record = json.loads((args.calibration.parent / "match.json").read_text())
        table = landmark_table(
            float(record["pitch_length_m"]),
            float(record["pitch_width_m"]),
            format_name=record.get("format"),
        )
        clicks_path = args.clicks or (args.calibration.parent / "clicks.json")
        clicks = json.loads(clicks_path.read_text())
        clicks = clicks.get("clicks", clicks) if isinstance(clicks, dict) else clicks
        before = _reprojection_error(calibration, clicks, table, q_before, focal_before)
        after = _reprojection_error(calibration, clicks, table, q_after, focal_after)
        print(f"landmark reprojection RMS: before {before:.1f} px -> after {after:.1f} px")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    _write_back(segment_dir, steps, focal_after, ok_after)
    print(f"wrote refined step/focal/ok into {completed_chunks(segment_dir)} chunk(s) of {segment_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
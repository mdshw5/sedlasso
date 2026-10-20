"""Turn the gimbal's own telemetry into a camera pose: the drift-free replacement for the estimated chain.

``geometry.camera_motion`` recovers the camera's orientation from the picture, step by step, and the error in each
step accumulates - measured on this footage the chain sits 3.7 px from a direct match after 15-30 s and 11.6 px
after 150 s, which is why ``geometry.drift`` exists to re-anchor it against landmark clicks. The gimbal log removes
the need for that: it records the yaw and pitch the camera actually commanded, so the orientation is a *measurement*
of the hardware, not an integration of image fits, and it does not drift.

What the log gives, and what it does not
----------------------------------------
The log records **yaw** (pan) and **pitch** (tilt) in degrees and a **zoom step**. On this footage the gimbal holds
a fixed tilt (11.6 deg) and pans +-62 deg, so the motion is a pure pan about a fixed axis. The axis is not the
image's vertical, because the camera is tilted down: it is the **world vertical** expressed in the camera's frame,
which makes an angle of ``90 - tilt`` with the optical axis (78.4 deg here). That is a physical prior, and it is
what the landmark clicks prefer - fitting the axis to the estimated chain instead lands near 49 deg, because the
gimbal's rotation center is not the lens center and each step carries a small translation that tilts the apparent
axis (the classic rotating-camera self-calibration error). The yaw *scale* (degrees of rotation per degree of
logged yaw) is fitted from the chain's large per-step rotations, which are accurate even though the chain's
accumulated orientation is not.

The zoom step is a hardware number with no published mapping to focal length, and the chain's own focal estimate is
too coarse to recover it (it is quantized to a few values). So the focal is left to the existing calibration, which
already solves a focal scale from landmark clicks; only the orientation is replaced here.

Measured on the whole 2026-10-03 game: a model fitted on the first minutes predicts landmark clicks 40 minutes
later to a median 16 m, against 48 m for the estimated chain - the drift the chain accumulates over a game is
exactly what the log removes.

Everything in this module is pure numpy: log records in, rotation matrices out. The alignment and the axis fit are
the parts where a mistake is invisible downstream, so they are tested against the chain and the clicks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from soccer_analytics.geometry.gimbal_log import GimbalFrame, GimbalLog, load_gimbal_log

# The yaw scale is fitted over windows this many analysis frames long. Longer is better: a window's ratio of chain
# rotation to logged yaw is noisy at 5 frames, and measured on the reference game the estimate converges from 0.94
# (5 frames) to 0.99 (20 frames). Shorter windows are tried as a fallback so a quiet or short segment can still
# fit - the caller keeps the estimated chain when even those have too little motion.
SCALE_WINDOW = 20
AXIS_WINDOW = 5  # the shortest window the scale fit may fall back to
MIN_AXIS_ANGLE_DEG = 1.0  # a window that barely moved says nothing about the axis
MIN_AXIS_SAMPLES = 30  # below this the fit is not trusted and the caller should fall back to the chain
# Only windows whose net rotation is essentially *about* the pan axis may vote on the scale. A window whose
# rotation leans off-axis (roll corrections during fast tracking, a couple of bad steps composed together) says
# almost nothing about the yaw channel, and projecting it onto the axis under-counts - these windows dragged the
# median from ~1.0 down to 0.70 on the reference game, which under-rotated every projection by 30%.
AXIS_ALIGNMENT_COS = 0.99

# The physical prior is a pan about world vertical with the camera tilted down. A mast that is not plumb, or a
# head that is rolled, puts the true axis somewhere else in the camera frame - and that error is a function of the
# *yaw*, so a time-based drift correction cannot remove it. The fit below searches a small neighborhood of the
# physical axis: roll (camera x), tilt (camera y/z), and the yaw scale. Eight degrees and twenty percent cover a
# 4 m mast on uneven ground with room to spare while keeping a stray click from moving the camera somewhere
# unphysical.
PAN_FIT_ROLL_DEG = 8.0
PAN_FIT_TILT_DEG = 8.0
PAN_FIT_SCALE = 0.20
# The objective trims this fraction of the worst landmarks. A weighted fit would let one mislabeled far-side click
# bend the axis; trimming keeps the fit on the clicks/lines that agree with each other and is what makes the fit
# usable when a few of the saved clicks are wrong.
PAN_FIT_TRIM = 0.15
# Clicks alone cannot separate a yaw-scale error from a focal-length error: a few percent of each trades almost
# exactly against the other, and a handful of mislabeled far-side clicks can pull the pair a long way. The first
# line scan showed the same failure from the other side - observations detected near a biased prediction pulled the
# scale to a visibly over-rotated overlay - so the scale stays regularized toward the chain-derived prior. Roll and
# tilt remain free to follow the ground control.
PAN_FIT_SCALE_PRIOR = 100.0  # meters of objective per unit of squared scale delta
# Line observations can number in the thousands; the axis is a three-parameter model, so a spread sample carries
# the same information at a fraction of the solve cost.
PAN_FIT_MAX_LINES = 40
PAN_MODEL_FILE = "pan_model.json"


@dataclass(frozen=True)
class PanModel:
    """The fitted mapping from the log's yaw to a rotation: a fixed axis and a signed scale.

    ``axis`` is the pan axis in the reference frame (the frame the chain's ``Q`` maps into), ``scale`` converts a
    degree of logged yaw into a degree of rotation about that axis (its sign is the pan direction), and ``yaw0`` is
    the yaw the reference frame corresponds to. ``rms_deg`` is how well the model reproduces the chain's per-step
    rotations - a large value means the fit did not find a consistent pan and the caller should not trust it.
    """

    axis: np.ndarray  # (3,) unit vector in the reference frame
    scale: float  # signed degrees of rotation per degree of logged yaw
    yaw0: float  # logged yaw at the reference frame
    rms_deg: float
    samples: int

    def to_json(self) -> dict:
        return {
            "axis": [float(v) for v in self.axis],
            "scale": float(self.scale),
            "yaw0": float(self.yaw0),
            "rms_deg": float(self.rms_deg),
            "samples": int(self.samples),
        }

    @classmethod
    def from_json(cls, data: dict) -> "PanModel":
        return cls(
            axis=np.asarray(data["axis"], dtype=np.float64),
            scale=float(data["scale"]),
            yaw0=float(data["yaw0"]),
            rms_deg=float(data["rms_deg"]),
            samples=int(data["samples"]),
        )

    def with_angles(self, roll_deg: float, tilt_deg: float, scale_delta: float = 0.0) -> "PanModel":
        """A nearby model with the pan axis rolled/tilted and the yaw scale shifted.

        ``roll_deg`` rotates the axis about the camera's optical axis (what a mast that is not plumb or a head that
        is not level does to a nominally vertical pan), and ``tilt_deg`` rotates it in the camera's vertical plane
        (the physical tilt prior being slightly wrong). Both are expressed in the reference frame of ``self``.
        """
        axis = self.axis / np.linalg.norm(self.axis)
        tilt = cv2.Rodrigues(np.radians(float(tilt_deg)) * np.array([1.0, 0.0, 0.0]))[0]
        roll = cv2.Rodrigues(np.radians(float(roll_deg)) * np.array([0.0, 0.0, 1.0]))[0]
        axis = roll @ (tilt @ axis)
        return PanModel(
            axis=axis / np.linalg.norm(axis),
            scale=float(self.scale + scale_delta),
            yaw0=float(self.yaw0),
            rms_deg=float(self.rms_deg),
            samples=int(self.samples),
        )


def align_log(
    logs: list[tuple[GimbalLog, float]],
    *,
    start_s: float,
    fps: float,
    frame_count: int,
) -> dict[str, np.ndarray]:
    """Sample the log at every analysis frame, using the clip start times to place each log on the game clock.

    ``logs`` pairs each parsed log with the second in the combined game video at which its clip begins (the same
    ``start_s`` the game manifest records). The log's own timestamps are seconds into its clip, so adding the clip
    start puts every log on one clock; the analysis frame ``i`` is at ``start_s + i / fps`` on that clock, and the
    nearest log sample is taken. Returns per-frame arrays (NaN where the log has no value), so a caller can see the
    coverage rather than assume it.
    """
    times: list[float] = []
    yaw: list[float] = []
    pitch: list[float] = []
    zoom: list[float] = []
    lock: list[float] = []
    ball_x: list[float] = []
    for log, clip_start in logs:
        for record in log.frames:
            if record.time_s is None:
                continue
            times.append(clip_start + record.time_s)
            yaw.append(record.yaw_deg if record.yaw_deg is not None else np.nan)
            pitch.append(record.pitch_deg if record.pitch_deg is not None else np.nan)
            zoom.append(record.zoom_sz if record.zoom_sz is not None else np.nan)
            lock.append(1.0 if record.locked else 0.0)
            ball_x.append(record.ball_x if record.ball_x is not None else np.nan)
    if not times:
        empty = np.full(frame_count, np.nan)
        return {"yaw": empty, "pitch": empty.copy(), "zoom": empty.copy(), "lock": np.zeros(frame_count), "ball_x": empty.copy()}
    order = np.argsort(times)
    times_arr = np.asarray(times)[order]
    at = start_s + np.arange(frame_count) / fps
    index = np.searchsorted(times_arr, at).clip(1, len(times_arr) - 1)
    left = np.abs(times_arr[index - 1] - at) < np.abs(times_arr[index] - at)
    index = np.where(left, index - 1, index)
    return {
        "yaw": np.asarray(yaw)[order][index],
        "pitch": np.asarray(pitch)[order][index],
        "zoom": np.asarray(zoom)[order][index],
        "lock": np.asarray(lock)[order][index],
        "ball_x": np.asarray(ball_x)[order][index],
    }


def fit_pan_model(
    chain_q: np.ndarray,
    ok: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray | None = None,
    *,
    window: int = SCALE_WINDOW,
    min_samples: int = MIN_AXIS_SAMPLES,
) -> PanModel | None:
    """Fit the pan model: the physical axis from the logged tilt, and the yaw scale from the chain's steps.

    The pan axis is not free. A gimbal pans about the **world vertical**, and the reference frame is the camera's
    own frame at frame 0, so the axis is the world-up direction expressed there: it lies in the image's vertical
    plane and makes an angle of ``90 - tilt`` with the optical axis, where ``tilt`` is the pitch the log records.
    That is a *physical* prior, and it matters: fitting the axis to the chain's per-step rotations instead lands
    near 49 deg, because the gimbal's rotation center is not the lens center and each step carries a small
    translation that tilts the apparent axis - the classic rotating-camera self-calibration error. The physical
    axis (78 deg here) is what the landmark clicks prefer, and it generalizes: a model fitted on the first minutes
    predicts clicks 40 minutes later to 16 m, against 48 m for the chain.

    The scale (degrees of rotation per degree of logged yaw) is fitted from windows of chain motion that are both
    large enough to be signal and *aligned with the pan axis*: a window that rotated off-axis (roll corrections
    during fast tracking, a noisy burst of steps composed together) has most of its angle in a channel the yaw
    scale cannot explain, and projecting it onto the axis under-counts. On the reference game those windows
dragged the median from ~1.0 down to 0.70 - every projection then under-rotated by 30%, and the overlay visibly
    trailed the pan - so only axis-aligned windows vote. The scale a fit on this footage should give is ~1.0:
    the gimbal's yaw is true degrees, confirmed independently by re-measuring the largest sweeps from the film
    with descriptor matching (agreement to 1-3%).
    """
    frames = len(chain_q)
    tilt = _median_pitch(pitch, yaw)
    if tilt is None:
        return None
    axis = np.array([0.0, np.sin(np.radians(90.0 - tilt)), np.cos(np.radians(90.0 - tilt))])
    axis /= np.linalg.norm(axis)

    def aligned_samples(window_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """(signed rotation about the axis, logged dyaw) for the windows that may vote on the scale."""
        signed_out: list[float] = []
        dyaw_out: list[float] = []
        for i in range(window_frames, frames, window_frames):
            if not (ok[i] and ok[i - window_frames]):
                continue
            if not (np.isfinite(yaw[i]) and np.isfinite(yaw[i - window_frames])):
                continue
            dyaw = float(yaw[i] - yaw[i - window_frames])
            if abs(dyaw) < MIN_AXIS_ANGLE_DEG:
                continue
            vec = cv2.Rodrigues(chain_q[i] @ chain_q[i - window_frames].T)[0].ravel()
            magnitude = float(np.linalg.norm(vec))
            signed = float(vec @ axis)
            if magnitude < 1e-9 or abs(signed) < AXIS_ALIGNMENT_COS * magnitude:
                continue
            signed_out.append(signed)
            dyaw_out.append(dyaw)
        return np.asarray(signed_out), np.asarray(dyaw_out)

    selected: tuple[np.ndarray, np.ndarray] | None = None
    for candidate in (window, max(window // 2, AXIS_WINDOW), AXIS_WINDOW):
        signed_arr, dyaw_arr = aligned_samples(candidate)
        if len(signed_arr) >= min_samples:
            selected = (signed_arr, dyaw_arr)
            break
    if selected is None:
        return None
    signed_arr, dyaw_arr = selected
    ratios = signed_arr / np.radians(dyaw_arr)
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or abs(scale) < 0.2:
        return None
    predicted = np.radians(scale * dyaw_arr)
    residual = np.degrees(np.abs(signed_arr - predicted))
    return PanModel(
        axis=axis,
        scale=scale,
        yaw0=float(yaw[np.isfinite(yaw)][0]),
        rms_deg=float(np.sqrt(np.mean(residual**2))),
        samples=int(len(signed_arr)),
    )


def _median_pitch(pitch: np.ndarray | None, yaw: np.ndarray) -> float | None:
    """The gimbal's tilt, from the log's pitch where it has one, else a level-camera default.

    The tilt sets the pan axis, so it is worth taking from the log rather than assuming: on this footage the
    gimbal holds 11.6 deg all match, which puts the axis 78.4 deg from the optical axis. A log with no pitch at all
    falls back to a level camera (axis 90 deg from the optical axis, i.e. the image vertical).
    """
    if pitch is not None:
        values = pitch[np.isfinite(pitch)]
        if len(values):
            return float(np.median(values))
    return 0.0


@dataclass(frozen=True)
class PanAxisFit:
    """The result of :func:`fit_pan_axis`: the refined model and how well it matches the ground control."""

    model: PanModel
    rms_m: float  # trimmed RMS of the landmarks the fit used, in ground meters
    evaluations: int


def fit_pan_axis(
    model: PanModel,
    yaw: np.ndarray,
    landmarks,
    focal: np.ndarray,
    aspect: float,
    *,
    fixed_height_m: float | None = None,
    max_lines: int = PAN_FIT_MAX_LINES,
    on_progress=None,
) -> PanAxisFit:
    """Fit the pan axis and yaw scale to clicks and automatic line observations.

    The physical prior (a vertical pan axis at the logged tilt) is only right if the mast is plumb and the head is
    level. A 4 m mast on grass is often neither: the camera's true pan axis is rolled or tilted in its own frame,
    and because that error depends on the camera's *heading*, the time-based drift correction cannot remove it -
    clicks at one heading fit and clicks at another are rejected as outliers. The same is true of the yaw scale:
    a five-percent error is invisible near the middle of the pan and meters at its ends.

    This searches a bounded roll/tilt/scale neighborhood around ``model``, minimizing a trimmed calibration
    residual over both the user's clicks and any line observations supplied. Only the camera's *pointing* model
    changes; the base pose and focal scale are refitted by the caller with the returned model. Trimming means a
    handful of mislabeled clicks or a misdetected line cannot drag the axis somewhere unphysical, while the
    surviving landmarks still constrain it.

    Returns the original model (with its cost) when there is too little ground control or the solver cannot place
    a camera through it. The search is deliberately small and deterministic - this runs on a saved calibration,
    not per frame - and never replaces the physical prior with a free 3-DOF camera.
    """
    if len(landmarks) < 6:
        return PanAxisFit(model, float("nan"), 0)

    sample = _pan_fit_sample(landmarks, max_lines=max_lines)
    cache: dict[tuple[float, float, float], float] = {}
    evaluations = 0
    budget = 90.0

    def report() -> None:
        if on_progress is not None:
            on_progress(min(0.99, evaluations / budget))

    def cost(roll: float, tilt: float, scale_delta: float) -> float:
        nonlocal evaluations
        key = (round(roll, 5), round(tilt, 5), round(scale_delta, 6))
        if key in cache:
            return cache[key]
        evaluations += 1
        candidate = model.with_angles(roll, tilt, scale_delta)
        q = log_orientation(yaw, candidate)
        chain = {lm.frame: (q[lm.frame], float(focal[lm.frame])) for lm in sample}
        try:
            from soccer_analytics.geometry.pitch_calibration import calibrate  # noqa: PLC0415 - avoids an import cycle

            calibration = calibrate(
                sample,
                chain,
                aspect,
                correct_drift=False,
                reject_outliers=False,
                robust=True,
                fixed_height_m=fixed_height_m,
                pose_source="log",
            )
        except Exception:  # noqa: BLE001 - a candidate that cannot place a camera is simply not an improvement
            cache[key] = float("inf")
            report()
            return float("inf")
        errors = np.asarray(calibration.residuals_m, dtype=np.float64)
        keep = max(4, int(round(len(errors) * (1.0 - PAN_FIT_TRIM))))
        value = float(np.sqrt(np.mean(np.sort(errors)[:keep] ** 2)))
        cache[key] = value
        report()
        return value

    def score(roll: float, tilt: float, scale_delta: float) -> tuple[float, float]:
        rms = cost(roll, tilt, scale_delta)
        if not np.isfinite(rms):
            return float("inf"), rms
        return rms + PAN_FIT_SCALE_PRIOR * float(scale_delta) ** 2, rms

    best_score, best_rms = score(0.0, 0.0, 0.0)
    if not np.isfinite(best_score):
        return PanAxisFit(model, float("nan"), evaluations)
    best = (0.0, 0.0, 0.0)  # roll, tilt, scale delta

    # Coordinate descent with a couple of diagonal moves: tilt and scale are correlated (both trade far-side
    # height against near-side width), so axis-aligned steps alone can crawl along the valley instead of down it.
    for step in (2.0, 1.0, 0.5, 0.25):
        step_scale = step / 100.0
        for _ in range(8):
            r0, t0, s0 = best
            neighbours = [
                (r0 + step, t0, s0),
                (r0 - step, t0, s0),
                (r0, t0 + step, s0),
                (r0, t0 - step, s0),
                (r0, t0, s0 + step_scale),
                (r0, t0, s0 - step_scale),
                (r0 + step, t0 + step, s0 + step_scale),
                (r0 - step, t0 - step, s0 - step_scale),
            ]
            improved = False
            for roll, tilt, delta in neighbours:
                if abs(roll) > PAN_FIT_ROLL_DEG or abs(tilt) > PAN_FIT_TILT_DEG or abs(delta) > PAN_FIT_SCALE:
                    continue
                value, rms = score(roll, tilt, delta)
                if value < best_score - 1e-6:
                    best_score, best_rms, best = value, rms, (roll, tilt, delta)
                    improved = True
            if not improved:
                break
    return PanAxisFit(model.with_angles(*best), best_rms, evaluations)


def _pan_fit_sample(landmarks, *, max_lines: int) -> list:
    """Clicks plus a spread sample of line observations, so a thousands-of-lines scan does not dominate."""
    clicks = [lm for lm in landmarks if getattr(lm, "direction", None) is None]
    lines = [lm for lm in landmarks if getattr(lm, "direction", None) is not None]
    if len(lines) > max_lines > 0:
        indices = np.linspace(0, len(lines) - 1, max_lines).round().astype(int)
        lines = [lines[i] for i in indices]
    return clicks + lines


def refine_pan_model(
    model: PanModel,
    yaw: np.ndarray,
    landmarks,
    focal: np.ndarray,
    aspect: float,
    *,
    fixed_height_m: float | None = None,
) -> PanModel:
    """Compatibility wrapper around :func:`fit_pan_axis`; returns the model unchanged when it cannot be fitted."""
    fit = fit_pan_axis(model, yaw, landmarks, focal, aspect, fixed_height_m=fixed_height_m)
    return fit.model if np.isfinite(fit.rms_m) else model


def _perturb(model: PanModel, d_pitch_deg: float, d_scale: float) -> PanModel:
    """A nearby pan model: the axis tilted by ``d_pitch_deg`` in the image's vertical plane, scale shifted."""
    tilt = 90.0 - np.degrees(np.arctan2(model.axis[1], model.axis[2]))
    new_tilt = tilt + d_pitch_deg
    axis = np.array([0.0, np.sin(np.radians(90.0 - new_tilt)), np.cos(np.radians(90.0 - new_tilt))])
    axis /= np.linalg.norm(axis)
    return PanModel(axis=axis, scale=model.scale + d_scale, yaw0=model.yaw0, rms_deg=model.rms_deg, samples=model.samples)


def _click_rms(model: PanModel, yaw: np.ndarray, landmarks, focal: np.ndarray, aspect: float) -> float | None:
    """Calibration RMS (meters) for a pan model, or ``None`` when the fit does not converge."""
    from soccer_analytics.geometry.pitch_calibration import calibrate

    q = log_orientation(yaw, model)
    chain = {lm.frame: (q[lm.frame], focal[lm.frame]) for lm in landmarks}
    try:
        return calibrate(landmarks, chain, aspect, correct_drift=False).rms_error_m
    except Exception:  # noqa: BLE001 - a degenerate candidate is simply not an improvement
        return None


def log_orientation(yaw: np.ndarray, model: PanModel) -> np.ndarray:
    """Per-frame orientation ``Q`` (F, 3, 3) from the logged yaw and a fitted pan model.

    ``Q`` maps frame rays to the reference frame's rays, the same convention as ``camera_motion.integrate_poses``,
    so it drops straight into the calibration and the projection. Frames with no logged yaw hold the previous
    orientation (a gap in the log is a missing measurement, not a jump to zero).
    """
    frames = len(yaw)
    out = np.empty((frames, 3, 3), dtype=np.float64)
    axis = model.axis / np.linalg.norm(model.axis)
    last = np.eye(3)
    for i in range(frames):
        value = yaw[i]
        if np.isfinite(value):
            angle = np.radians(model.scale * (float(value) - model.yaw0))
            last = cv2.Rodrigues(angle * axis)[0]
        out[i] = last
    return out


def load_logs_for_clips(
    log_paths: list[str],
    clip_starts: list[float],
) -> list[tuple[GimbalLog, float]]:
    """Pair each log file with its clip's start second, for :func:`align_log`."""
    return [(load_gimbal_log(path), float(start)) for path, start in zip(log_paths, clip_starts)]


# The camera writes its logs in a directory beside the clips, usually named for the camera model. The clip's own
# name is a timestamp ("16:28:37.784.MP4") and the log's is the same timestamp ("2026-10-03 16:28:37.json"), so a
# clip is matched to its log by the time-of-day part of the name. These names are only preferred spellings: the
# folder may be renamed (or its case changed), so every sibling folder is searched after them.
LOG_DIR_NAMES = ("Chameleon Logs", "Falcon Logs", "Logs")
_CLIP_TIME = re.compile(r"(\d{2})[:-](\d{2})[:-](\d{2})")


def _log_directories(directory: Path) -> list[Path]:
    """Folders to search for logs: the clip's own folder, then its usual siblings, then every other sibling."""
    if not directory.is_dir():
        return []
    children = [path for path in directory.iterdir() if path.is_dir()]
    preferred = [
        path
        for name in LOG_DIR_NAMES
        for path in children
        if path.name.casefold() == name.casefold()
    ]
    others = [path for path in sorted(children, key=lambda path: path.name.casefold()) if path not in preferred]
    return [directory] + preferred + others


def find_logs_for_clips(clip_paths: list[str | Path]) -> list[tuple[str, float]] | None:
    """Locate the gimbal logs for a game's clips, returning ``(log_path, clip_start_s)`` pairs in clip order.

    The clips are the files the game was combined from, each with the second it begins at in the combined video.
    The logs are named by the same wall-clock time as the clip, so each clip is matched to the log whose name
    carries its time-of-day. They may sit beside the clips, in a directory named for the camera (``Chameleon
    Logs`` and friends), or in any other sibling folder: the clip's own folder is tried first, then the known
    names, then every other sibling folder. Returns ``None`` when no folder has a matching log for every clip, so
    a caller can fall back to the estimated chain rather than fail.
    """
    if not clip_paths:
        return None
    wanted: list[str] = []
    for clip in clip_paths:
        match = _CLIP_TIME.search(Path(clip).name)
        if match is None:
            return None
        hh, mm, ss = match.groups()
        wanted.append(f"{hh}:{mm}:{ss}")
    for log_dir in _log_directories(Path(clip_paths[0]).parent):
        logs = sorted(log_dir.glob("*.json"))
        if not logs:
            continue
        pairs: list[tuple[str, float]] = []
        for key in wanted:
            found = next((log for log in logs if key in log.name), None)
            if found is None:
                break
            pairs.append((str(found), 0.0))
        else:
            return pairs
    return None


def build_log_poses(
    clip_paths: list[str | Path],
    clip_starts: list[float],
    chain_q: np.ndarray,
    ok: np.ndarray,
    focal: np.ndarray,
    *,
    start_s: float,
    fps: float,
    landmarks=None,
    aspect: float | None = None,
    pan_model: PanModel | None = None,
    fixed_height_m: float | None = None,
) -> tuple[np.ndarray, PanModel, dict[str, np.ndarray]] | None:
    """The whole log pipeline: find the logs, align them, fit the pan model, and return the orientation.

    Returns ``(Q, model, aligned)`` where ``Q`` is the per-frame orientation to use in place of the chain's,
    ``model`` is the pan model (for the record), and ``aligned`` carries the per-frame yaw/pitch/zoom/lock the log
    supplied. Returns ``None`` when there is no log for these clips, so the caller keeps the chain.

    ``pan_model`` is a model already fitted against absolute ground control (see :func:`fit_pan_axis`). When it is
    given it is used as-is; otherwise the physical prior is fitted from the chain, and ``landmarks`` (clicks and
    line observations) refine it when supplied. ``fixed_height_m`` is the rig height for that refinement.
    """
    pairs = find_logs_for_clips(clip_paths)
    if pairs is None:
        return None
    # ``find_logs_for_clips`` returns the logs in clip order; the clip start times come from the game manifest.
    logs = [(load_gimbal_log(path), float(start)) for (path, _), start in zip(pairs, clip_starts)]
    aligned = align_log(logs, start_s=start_s, fps=fps, frame_count=len(chain_q))
    if pan_model is None:
        model = fit_pan_model(chain_q, ok, aligned["yaw"], aligned["pitch"])
        if model is None:
            return None
        if landmarks is not None and aspect is not None:
            model = refine_pan_model(
                model, aligned["yaw"], landmarks, focal, aspect, fixed_height_m=fixed_height_m
            )
    else:
        model = pan_model
    return log_orientation(aligned["yaw"], model), model, aligned


# The game manifest for a combined video is found by matching its ``output`` path. Cached because the projection
# calls this once per run and the manifest is small but the directory scan is not free.
_GAME_MANIFEST_CACHE: dict[str, dict | None] = {}


def _game_manifest_for(video: str) -> dict | None:
    """The game manifest whose combined video is ``video``, or ``None`` when the footage is not a combined game.

    The manifest is what connects a combined video back to the clips it was built from - and so to the gimbal logs
    beside those clips. Candidates are read from every place a manifest may live, because the combined video may
    have been moved or re-merged since the manifest was written:

    1. a ``game.json`` beside the video (the manifest was copied with it);
    2. a ``game.json`` in the video's own analysis directory (``<footage>/analysis/<id>/game.json`` - where the
       game build writes it, next to the footage the clips came from);
    3. ``data/games/*/game.json`` in the repository (the archive root);
    and each candidate matches when its ``output`` path equals the video's path, or its basename does (moved but
    not renamed).

    Among the matches, a manifest whose clips name files other than the video itself wins. That is the
    distinction that matters to the log alignment: a manifest that lists the combined video as its own single
    clip has no clip information to place the camera logs with, while one listing the source clips connects each
    log to its start second on the game clock. The first informative candidate wins, else the first candidate,
    and ``None`` when nothing matches - so the caller keeps the estimated chain.
    """
    if video in _GAME_MANIFEST_CACHE:
        return _GAME_MANIFEST_CACHE[video]
    import json

    repo_root = Path(__file__).resolve().parents[3]
    games_root = repo_root / "data" / "games"
    video_path = Path(video)

    def load(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    if video_path.name == "game.json" and video_path.exists():
        # The never-merged workflow analyzes the manifest itself (``meta["video"]`` names it), so the manifest
        # is its own description - the search below, which looks for the manifest *of* a video, does not apply.
        payload = load(video_path)
        _GAME_MANIFEST_CACHE[video] = payload
        return payload

    def matches(payload: dict | None) -> bool:
        if payload is None:
            return False
        output = str(payload.get("output", ""))
        return output == video or Path(output).name == video_path.name

    def informative(payload: dict) -> bool:
        """Whether the manifest's clips say something about the video's provenance (are not the video itself)."""
        return any(
            Path(str(clip.get("path", ""))) != video_path and Path(str(clip.get("path", ""))).name != video_path.name
            for clip in (payload.get("clips") or [])
        )

    candidate_paths: list[Path] = [video_path.parent / "game.json"]
    analysis_root = video_path.parent / "analysis"
    if analysis_root.is_dir():
        candidate_paths += sorted(analysis_root.glob("*/game.json"))
    if games_root.is_dir():
        candidate_paths += sorted(games_root.glob("*/game.json"))

    found: dict | None = None
    fallback: dict | None = None
    for path in candidate_paths:
        payload = load(path)
        if not matches(payload):
            continue
        if fallback is None:
            fallback = payload
        if informative(payload):
            found = payload
            break
    _GAME_MANIFEST_CACHE[video] = found if found is not None else fallback
    return _GAME_MANIFEST_CACHE[video]


def _resolve_clip_path(value: str, base: Path) -> str:
    """One clip's path as a usable file: absolute when it exists, else resolved against the footage directory.

    Manifests store the source clips relative to the footage folder they sit in, while a moved footage directory
    keeps the names but changes the prefix - so a bare name is tried beside the combined video before giving up.
    An untouched value survives, so a manifest that named something unresolvable still reaches the caller as it
    was written rather than as a silent miss.
    """
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = base / path
    if candidate.exists():
        return str(candidate)
    candidate = base / path.name
    return str(candidate) if candidate.exists() else str(value)


def _clips_for_segment(segment) -> tuple[list[str], list[float]]:
    """The clips a segment's footage came from, and where each begins on the segment's own clock.

    Three shapes exist. A segment analyzed from the *game manifest* (the never-merged workflow) names the
    manifest, whose clips and offsets load through :class:`GameRecord` - paths resolved against the footage
    directory, exactly as the manifest stored them. A segment analyzed from the *combined game video* names that
    video, and the game manifest connecting it to its source clips lists, per clip, the path and the second it
    begins at in the combined video (paths are resolved against the video's own directory - see
    :func:`_resolve_clip_path`). A segment analyzed from a *single raw clip* names the clip itself, which begins
    at 0. Either way the result is a list of clip paths and their start seconds, which places the logs on the
    analysis clock.
    """
    video = str(segment.meta.get("video", ""))
    if Path(video).name == "game.json":
        from soccer_analytics.analysis.game import GameRecord  # noqa: PLC0415 - avoids an import cycle

        try:
            record = GameRecord.load(Path(video).parent)
        except (OSError, ValueError, KeyError, TypeError):
            # An unreadable manifest is no manifest: the segment's own chunks are complete without it, and the
            # documented best effort is to keep the estimated chain rather than fail the whole results read.
            record = None
        if record is not None and record.clips:
            return ([clip.path for clip in record.clips], [float(clip.start_s) for clip in record.clips])
    manifest = _game_manifest_for(video)
    if manifest is not None:
        clips = manifest.get("clips") or []
        if clips:
            base = Path(video).parent
            return (
                [_resolve_clip_path(str(clip["path"]), base) for clip in clips],
                [float(clip["start_s"]) for clip in clips],
            )
    return [video], [0.0]


def log_poses_for_segment(
    segment,
    chain_q: np.ndarray,
    focal: np.ndarray,
    *,
    landmarks=None,
    pan_model: PanModel | None = None,
    fixed_height_m: float | None = None,
) -> tuple[np.ndarray, PanModel, dict[str, np.ndarray]] | None:
    """Log-backed orientation for a segment, or ``None`` when its footage has no gimbal log.

    The segment's ``meta`` names its video; the clips behind it (from the game manifest, or the raw clip itself)
    carry the logs, and the clip start seconds place them on the analysis clock. Everything is best-effort: a
    missing manifest, a missing log directory, or a log that does not cover the segment all return ``None`` so the
    caller keeps the estimated chain.

    A saved ``pan_model`` (the per-segment sidecar written by :func:`save_pan_model`) wins over the physical prior;
    it is the one fitted to absolute ground control. An explicit ``pan_model`` argument wins over the sidecar, so a
    caller holding a calibration can be sure it projects through the model that calibration was fitted against.
    """
    clip_paths, clip_starts = _clips_for_segment(segment)
    # A segment without the clock fields (a synthetic or hand-built one) has no log to align to; keep the chain.
    if "start_s" not in segment.meta or "fps" not in segment.meta:
        return None
    if pan_model is None:
        pan_model = load_pan_model(segment)
    return build_log_poses(
        clip_paths,
        clip_starts,
        chain_q,
        segment.ok,
        focal,
        start_s=float(segment.meta["start_s"]),
        fps=float(segment.meta["fps"]),
        landmarks=landmarks,
        aspect=segment.aspect,
        pan_model=pan_model,
        fixed_height_m=fixed_height_m,
    )


def load_pan_model(segment) -> PanModel | None:
    """The saved pan model for a segment's own footage, or ``None`` when it has not been fitted.

    The model lives beside the segment's chunks rather than in the match record: it belongs to the camera path
    (the segment), and every consumer that rebuilds poses - the dashboard, the stream, the report rebuild - should
    pick it up without having to know which match archive is open. A malformed sidecar is ignored rather than
    taking the page down; the physical prior is a safe fallback.
    """
    directory = getattr(segment, "directory", None)
    if directory is None:
        return None
    path = Path(directory) / PAN_MODEL_FILE
    if not path.exists():
        return None
    try:
        return PanModel.from_json(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_pan_model(segment, model: PanModel) -> Path | None:
    """Write a fitted pan model beside the segment's chunks; returns the path (``None`` when it has no directory)."""
    directory = getattr(segment, "directory", None)
    if directory is None:
        return None
    path = Path(directory) / PAN_MODEL_FILE
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(model.to_json(), indent=2))
    temp.replace(path)
    return path


def aligned_log_for_segment(segment) -> dict[str, np.ndarray] | None:
    """The segment's gimbal log sampled on its own analysis clock, or ``None`` when it has no log.

    This is the alignment without the pose fit: the yaw/pitch/zoom the log supplied at each frame. A caller that
    already has ground control (clicks, line observations) can fit a pan model directly from it instead of first
    fitting one to the visual chain.
    """
    if "start_s" not in segment.meta or "fps" not in segment.meta:
        return None
    clip_paths, clip_starts = _clips_for_segment(segment)
    pairs = find_logs_for_clips(clip_paths)
    if pairs is None:
        return None
    logs = [(load_gimbal_log(path), float(start)) for (path, _), start in zip(pairs, clip_starts)]
    return align_log(
        logs,
        start_s=float(segment.meta["start_s"]),
        fps=float(segment.meta["fps"]),
        frame_count=len(segment.time),
    )


def segment_has_log(segment) -> bool:
    """Whether a segment's footage has a gimbal log, for the dashboard to say so without rebuilding the poses."""
    clip_paths, _ = _clips_for_segment(segment)
    return find_logs_for_clips(clip_paths) is not None


def segment_pose_source(segment) -> str:
    """Which camera-motion source ``segment_poses`` will use for this segment: ``"log"`` or ``"chain"``.

    A calibration is fitted against one source's reference frame, so this is what lets the app notice that a saved
    calibration was built against the other and refit it rather than project through a stale pose.
    """
    return "log" if segment_has_log(segment) else "chain"
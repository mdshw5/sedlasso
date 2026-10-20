"""Stage B, step 1: project raw detections into pitch meters, with an honest uncertainty for every position.

A detection's *foot point* (bottom-center of its box) is where the player meets the ground, so it is the pixel whose
ray we intersect with the pitch plane. Depth is the weak axis: on distant ground one pixel of vertical error is worth
several meters. Each projected point therefore carries ``sigma_m``, the ground distance of a one-pixel vertical
shift, so downstream metrics (distance run, speed, possession proxies) can ignore or down-weight unreliable far-side
positions instead of silently treating them as exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_analytics.analysis.stage_a import SegmentData
from soccer_analytics.geometry.camera_motion import integrate_poses, smooth_pose_chain
from soccer_analytics.geometry.gimbal_motion import log_poses_for_segment
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels, pixels_to_pitch

# Typical foot-point localization error of a YOLO box bottom edge, in pixels at 1920 wide. Players' feet are not
# always inside the box tightly (running poses, occlusion), so this is deliberately not 1.
FOOT_PIXEL_SIGMA = 4.0
PITCH_MARGIN_M = 1.5  # a player's foot may sit just outside the line; spectators stand well beyond it


@dataclass
class PitchDetections:
    """Detections of one segment mapped to the pitch. All arrays are length D (one row per detection)."""

    frame: np.ndarray  # (D,) analysis-frame index within the segment
    time: np.ndarray  # (D,) source seconds
    xy: np.ndarray  # (D, 2) pitch meters, NaN where the ray never meets the ground
    sigma_m: np.ndarray  # (D,) ground uncertainty of this position (m); inf where invalid
    valid: np.ndarray  # (D,) bool: foot ray hits the ground and the camera state was trustworthy
    height_px: np.ndarray  # (D,) box height, pixels at 1920 wide
    # (D, 4) the detection's own box, ``(x1, y1, x2, y2)`` normalized by frame width. The pitch position is where
    # the player stands; this is where they *are in the picture*, which is what a clip cropped around them has to
    # follow - so it travels with the detections instead of being looked up again from the segment by every caller.
    box: np.ndarray
    conf: np.ndarray  # (D,)
    kit: np.ndarray  # (D, K)
    # (D,) Stage A's BoT-SORT identity per detection, -1 where the tracker had none (v1 chunks, or a detection
    # below the tracker's own threshold). A hint for Stage B's association, never the authority: the pitch-space
    # gate decides whether a continuation is physically possible.
    det_track: np.ndarray
    det_index: np.ndarray  # (D,) index into the segment's detection arrays (for provenance)
    # (F, 2) pitch position the camera was aimed at each frame. The gimbal follows the ball, so this is a good
    # ball proxy where the ball's own scan (analysis.ball) has not been run - and it is a *measurement* either way,
    # not a guess about which pixel is the ball.
    aim_xy: np.ndarray
    # (2,) the camera's own ground position (its X/Y, ignoring height). Fixed for the segment; the replay draws a
    # line from here to the aim point so the direction the camera is pointing is visible on the pitch.
    camera_xy: np.ndarray


def segment_poses(segment: SegmentData, focal0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ``Q`` and focal for a segment, rebuilt from stored raw steps (lost frames carry no motion).

    When the requested ``focal0`` is the one the analysis pass used (the default), the stored per-frame focal
    lengths are reused and the per-step focal search is skipped - same chain, one SVD per frame instead of a bounded
    minimization of many. On a whole-game segment (21k frames) that is 41 s of every dashboard rerun turned into
    under a second.

    When the footage has a gimbal log (``geometry.gimbal_motion``), the *orientation* is taken from the log instead
    of the chain: the log records the yaw the camera actually commanded, so it does not accumulate the drift the
    chain does. The focal still comes from the chain, because the log's zoom step has no published mapping to focal
    length and the calibration solves a focal scale from the clicks anyway. A segment with no log keeps the chain.
    """
    default_focal = float(segment.meta["default_focal"])
    focal0 = default_focal if focal0 is None else focal0
    steps = [None if (not ok or i == 0) else step for i, (ok, step) in enumerate(zip(segment.ok, segment.step))]
    known = np.asarray(segment.focal, dtype=np.float64) if abs(float(focal0) - default_focal) < 1e-9 else None
    q, focal = integrate_poses(steps, focal0, segment.aspect, known_focals=known)
    if abs(float(focal0) - default_focal) < 1e-9:
        logged = log_poses_for_segment(segment, q, focal)
        if logged is not None:
            # The logged orientation is absolute and drift-free but quantized: it holds for a few frames and then
            # steps, and every projected player and pitch marking jumps with it. A zero-phase filter removes the
            # steps without lagging a pan, and the focal track from the optical-flow chain gets the same treatment.
            # The calibration is fitted through this same chain, and an existing one remains valid (the whole-game
            # click RMS moved by 0.2 px on the reference match). A segment without a log keeps the raw visual
            # chain, whose steps are already smooth optical-flow measurements.
            q, focal = smooth_pose_chain(logged[0], focal, float(segment.meta["fps"]))
    return q, focal


def project_segment(
    segment: SegmentData,
    calibration: PitchCalibration,
    *,
    poses: tuple[np.ndarray, np.ndarray] | None = None,
    pitch_size: tuple[float, float] | None = None,
    on_progress=None,
) -> PitchDetections:
    """Map every detection to pitch meters. If ``pitch_size=(length, width)`` is given, nothing is clipped here;
    use ``on_pitch_mask`` to decide who is a player. ``on_progress(fraction)`` is called a few times per run."""
    q, focal = poses if poses is not None else segment_poses(segment)
    # The calibration's drift correction belongs to the chain, not to the caller's copy of it: every projection in
    # the app goes through here, so applying it centrally is what keeps the report, the replay and the fit check
    # all looking at the same (corrected) camera path.
    q = calibration.corrected_chain(q)
    focal = calibration.corrected_focal(focal)
    frames = segment.det_frame
    boxes = segment.det_box
    foot_uv = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])  # bottom-center, width-normalized
    n = len(frames)
    xy = np.full((n, 2), np.nan)
    sigma = np.full(n, np.inf)
    valid = np.zeros(n, dtype=bool)
    pixel = FOOT_PIXEL_SIGMA / 1920.0

    unique_frames = np.unique(frames)
    progress_every = max(1, len(unique_frames) // 25)
    for index, frame in enumerate(unique_frames):
        rows = np.where(frames == frame)[0]
        if segment.ok[frame]:
            here, ok = pixels_to_pitch(calibration, foot_uv[rows], q[frame], focal[frame])
            shifted, ok2 = pixels_to_pitch(calibration, foot_uv[rows] + [0.0, pixel], q[frame], focal[frame])
            usable = ok & ok2
            xy[rows[usable]] = here[usable]
            sigma[rows[usable]] = np.linalg.norm(shifted[usable] - here[usable], axis=1)
            valid[rows[usable]] = True
        if on_progress is not None and (index % progress_every == 0 or index == len(unique_frames) - 1):
            on_progress((index + 1) / len(unique_frames))

    return PitchDetections(
        frame=frames.astype(np.int32),
        time=segment.time[frames],
        xy=xy,
        sigma_m=sigma,
        valid=valid,
        height_px=(boxes[:, 3] - boxes[:, 1]) * 1920.0,
        box=boxes.copy(),
        conf=segment.det_conf,
        kit=segment.det_kit,
        det_track=getattr(segment, "det_track", np.full(n, -1, dtype=np.int32)),
        det_index=np.arange(n, dtype=np.int64),
        aim_xy=_aim_points(calibration, q, focal, segment.aspect, len(segment.time)),
        camera_xy=np.asarray(calibration.position[:2], dtype=np.float64),
    )


def _aim_points(
    calibration: PitchCalibration, q: np.ndarray, focal: np.ndarray, aspect: float, frames: int
) -> np.ndarray:
    """Ground position the camera was pointing at for every frame (NaN when aimed above the horizon)."""
    aim = np.full((frames, 2), np.nan)
    center = np.array([[0.5, 0.5 * aspect]])
    for frame in range(min(frames, len(q))):
        hit, ok = pixels_to_pitch(calibration, center, q[frame], focal[frame])
        if ok[0]:
            aim[frame] = hit[0]
    return aim


def project_ball_track(
    records: list[dict],
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The ball scan's per-frame positions in pitch meters, and whether each one was a detection.

    ``records`` is the ``frames`` list of a segment's ``ball_track.json`` (see ``analysis.ball`` and
    ``scripts/run_ball_scan.py``) - one entry per analysis frame with ``status`` and image-space ``u``/``v``.
    Returns ``(xy (F, 2), measured (F,))``: ``xy`` is NaN wherever the scan has no position to show, and
    ``measured`` is 1 where a detector saw the ball that frame, 0 where the position is the tracker's forecast
    across a miss. Frames the scan has not reached, and frames where the ball was out of view or lost, stay NaN -
    a forecast that left the picture is exactly what the replay must not draw as a sighting.

    The projection intersects the ball's *center* pixel with the ground, the same way the foot points are
    intersected, so a frame reads a ball radius or so beyond the true spot; at these distances that is well
    inside the ±meter spread of a click. The camera path is the calibration's corrected chain, so ball and
    players land on the same pitch.
    """
    q = calibration.corrected_chain(q)
    focal = calibration.corrected_focal(focal)
    frames = len(q)
    xy = np.full((frames, 2), np.nan)
    measured = np.zeros(frames)
    for record in records:
        status = record.get("status")
        u, v = record.get("u"), record.get("v")
        frame = int(record.get("i", -1))
        if status not in ("tracking", "coasting") or u is None or v is None or not 0 <= frame < frames:
            continue
        hit, ok = pixels_to_pitch(calibration, np.array([[float(u), float(v)]]), q[frame], focal[frame])
        if ok[0]:
            xy[frame] = hit[0]
            measured[frame] = 1.0 if status == "tracking" else 0.0
    return xy, measured


def goal_mouth_image(
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
    pitch_length_m: float,
    pitch_width_m: float,
    goal_half_width_m: float,
) -> np.ndarray:
    """Where each goal mouth is in the picture, per frame: ``(F, 2, 3)``.

    The last axis is ``(u_min, u_max, v_line)``: the two posts' normalized image x and the mean of their image y.
    A ball is at that goal when its own ``(u, v)`` sits between the posts and around the line. Doing this test in
    the picture is deliberate: the ground-plane projection is singular near the horizon, which is exactly where a
    far goal sits, so a ball that is plainly in the net can project to ``NaN`` or to hundreds of meters out.

    Side order is 0 = the x=0 goal, 1 = the x=L goal (the same convention :func:`event_detection._goal_mouth_hit`
    uses). Frames whose goal line is behind the camera, or beyond the horizon, are NaN.
    """
    q = calibration.corrected_chain(q)
    focal = calibration.corrected_focal(focal)
    frames = min(len(q), len(focal))
    out = np.full((frames, 2, 3), np.nan)
    half_width = float(pitch_width_m) / 2.0
    for side, goal_x in enumerate((0.0, float(pitch_length_m))):
        posts = np.array([[goal_x, half_width - goal_half_width_m], [goal_x, half_width + goal_half_width_m]])
        for frame in range(frames):
            uv, in_front = pitch_to_pixels(calibration, posts, q[frame], focal[frame])
            if in_front.all() and np.isfinite(uv).all():
                out[frame, side] = (
                    float(min(uv[0, 0], uv[1, 0])),
                    float(max(uv[0, 0], uv[1, 0])),
                    float(np.mean(uv[:, 1])),
                )
    return out


def center_spot_image(
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
    pitch_length_m: float,
    pitch_width_m: float,
) -> np.ndarray:
    """The center spot in the picture, per frame: ``(F, 2)`` normalized ``(u, v)``, NaN where not in front."""
    q = calibration.corrected_chain(q)
    focal = calibration.corrected_focal(focal)
    frames = min(len(q), len(focal))
    out = np.full((frames, 2), np.nan)
    center = np.array([[float(pitch_length_m) / 2.0, float(pitch_width_m) / 2.0]])
    for frame in range(frames):
        uv, in_front = pitch_to_pixels(calibration, center, q[frame], focal[frame])
        if in_front[0] and np.isfinite(uv[0]).all():
            out[frame] = uv[0]
    return out


def on_pitch_mask(
    detections: PitchDetections, length_m: float, width_m: float, *, margin_m: float = PITCH_MARGIN_M
) -> np.ndarray:
    """True for detections standing on the pitch (within ``margin_m`` of its lines).

    The pitch rectangle spans ``x in [0, length]`` and ``y in [0, width]`` in the calibration's frame. A foot point
    that projects beyond this (spectators, coaches, subs) is excluded *geometrically*, which kit color cannot do.
    The margin widens with the point's own uncertainty so a far-side player is not dropped by a meter of noise.
    """
    x, y = detections.xy[:, 0], detections.xy[:, 1]
    slack = margin_m + np.where(np.isfinite(detections.sigma_m), np.minimum(detections.sigma_m, 4.0), 0.0)
    inside = (x >= -slack) & (x <= length_m + slack) & (y >= -slack) & (y <= width_m + slack)
    return detections.valid & inside

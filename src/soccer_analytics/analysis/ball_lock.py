"""The gimbal's own ball lock: a second ball measurement, and possession that needs no homography.

The camera tracks the ball to aim itself, and its log records that tracking loop: whether it is holding the ball
(``Lock:a/b``), where the ball sits in the picture (``Ballx``, and the box's offset from the frame center), how big
it looks (``BallSz``) and how fast it is moving in the image (``xv``/``yv``). That is a *hardware* ball measurement
beside the expensive ``analysis.ball`` scan, and it is available for the whole game at no extra cost.

Two things it gives that the pitch-space pipeline cannot:

* **Ball in play.** The lock is the camera's own answer to "is there a ball to follow". A long stretch with no lock
  is a stoppage, a substitution, or the ball out of shot; the locked stretches are the live play. This needs no
  calibration at all.
* **Possession in image space.** The ball's image position and the players' image positions are both measurements in
  the same frame, so "which player is nearest the ball" is a question about the picture, not about the pitch. It
  works even where the homography is poor or absent, which is exactly the case the pitch-space possession proxy
  struggles with.

The image coordinates are normalized by the log's own frame width (2560 px on this camera, where ``Crowdx1280`` is
the center), so they are resolution-independent and line up with the detection boxes, which the pipeline also
normalizes by frame width.

The ball scan (``analysis.ball``) records the same quantity per frame - a normalized ``u``/``v`` with a status - so
:class:`BallImage` is the shared shape for both measurements. Stage B and the replay prefer the scan's position
where it has one and fall back to the hardware lock or the camera aim point; the geometry, and therefore the
image-space measure, is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The log's frame is 2560 px wide (its center is Crowdx1280) and 1440 tall; the ball's image x is absolute in that
# frame and the box offset is from its center. Normalizing by the width matches the detection-box convention.
LOG_FRAME_WIDTH = 2560.0
LOG_FRAME_CENTER_Y = 720.0

# A lock this many analysis frames long is a real spell of play, not a one-frame flicker of the detector.
MIN_PLAY_FRAMES = 5
# The ball is "near" a player within this image distance (normalized by frame width) for the possession proxy.
# Measured against the real game: a player on the ball is within ~0.05 of the ball's image position; a player a few
# meters away is 0.1-0.2, so the gate sits between them.
POSSESSION_IMAGE_RADIUS = 0.08


@dataclass
class BallLock:
    """Per-frame hardware ball state, aligned to the analysis frames of a segment.

    ``u``/``v`` are the ball's image position (normalized by frame width), NaN where the hardware had no position;
    ``locked`` is the hardware's own "I am holding the ball" flag; ``size`` is the apparent ball size and ``speed``
    the image-space speed (normalized units per frame). All arrays are length F (one per analysis frame).
    """

    locked: np.ndarray  # (F,) bool
    u: np.ndarray  # (F,) ball image x / width, NaN where unknown
    v: np.ndarray  # (F,) ball image y / width, NaN where unknown
    size: np.ndarray  # (F,) apparent ball size (BallSz), NaN where unknown
    speed: np.ndarray  # (F,) image-space speed, normalized units per frame

    @property
    def frames(self) -> int:
        return len(self.locked)

    def in_play(self) -> np.ndarray:
        """Frames that belong to a spell of live play: a locked run at least ``MIN_PLAY_FRAMES`` long."""
        return _long_runs(self.locked, MIN_PLAY_FRAMES)

    def to_json(self) -> dict:
        return {
            "locked": [bool(x) for x in self.locked],
            "u": [None if not np.isfinite(x) else round(float(x), 5) for x in self.u],
            "v": [None if not np.isfinite(x) else round(float(x), 5) for x in self.v],
            "size": [None if not np.isfinite(x) else float(x) for x in self.size],
            "speed": [round(float(x), 5) for x in self.speed],
        }


@dataclass
class BallImage:
    """A per-frame ball position in the picture, normalized by frame width.

    This is the common shape of the two image-space ball measurements: the ball scan's ``tracking``/``coasting``
    frames and the gimbal's hardware lock. ``u``/``v`` are NaN where there is no position; ``valid`` says whether
    the row carries a usable measurement, so a caller can report coverage instead of silently reading a gap as a
    ball at pixel zero. ``source`` names the measurement in report notes and the replay payload.
    """

    u: np.ndarray  # (F,) ball image x / width, NaN where unknown
    v: np.ndarray  # (F,) ball image y / width, NaN where unknown
    valid: np.ndarray  # (F,) bool: this row has a position
    source: str = "ball scan"

    @property
    def frames(self) -> int:
        return len(self.valid)


def ball_image_from_lock(lock: BallLock) -> BallImage:
    """The hardware lock as a :class:`BallImage`; ``valid`` requires a position, not just the lock flag."""
    return BallImage(
        u=np.asarray(lock.u, dtype=np.float64),
        v=np.asarray(lock.v, dtype=np.float64),
        valid=np.asarray(lock.locked, dtype=bool) & np.isfinite(lock.u),
        source="gimbal lock",
    )


def ball_image_from_scan(records, frame_count: int, *, source: str = "ball scan") -> BallImage:
    """Per-frame ball positions from a segment's ``ball_track.json`` records.

    ``tracking`` is a detection and ``coasting`` is the tracker's forecast across a short miss; both carry a
    position that is meaningful for possession and both are what the replay already draws. ``out_of_view`` and
    ``lost`` are deliberately excluded - they do not say where the ball is, and treating a forecast that left the
    picture as a position is exactly the mistake the ball tracker was built to avoid. A later record for the same
    frame wins.
    """
    frames = max(0, int(frame_count))
    u = np.full(frames, np.nan, dtype=np.float64)
    v = np.full(frames, np.nan, dtype=np.float64)
    valid = np.zeros(frames, dtype=bool)
    for record in records or []:
        if record.get("status") not in ("tracking", "coasting"):
            continue
        frame = int(record.get("i", -1))
        uu, vv = record.get("u"), record.get("v")
        if not (0 <= frame < frames) or uu is None or vv is None:
            continue
        u[frame] = float(uu)
        v[frame] = float(vv)
        valid[frame] = True
    return BallImage(u=u, v=v, valid=valid, source=source)


def foot_distance(box: np.ndarray, u: float, v: float) -> np.ndarray:
    """Distance in the picture from each box's foot point to a ball position.

    ``box`` is one ``(4,)`` box or an ``(N, 4)`` array, ``(x1, y1, x2, y2)`` normalized by frame width. The ball's
    ``v`` may be NaN when the measurement only knows its horizontal position (the gimbal lock without a box
    offset); then horizontal distance is the best available measure, and the caller is still measuring the picture
    rather than the ground.
    """
    box = np.asarray(box, dtype=np.float64)
    foot_u = (box[..., 0] + box[..., 2]) / 2.0
    foot_v = box[..., 3]
    if not np.isfinite(v):
        return np.abs(foot_u - u)
    return np.hypot(foot_u - u, foot_v - v)


def _long_runs(flags: np.ndarray, minimum: int) -> np.ndarray:
    """A boolean mask keeping only the True runs at least ``minimum`` long."""
    out = np.zeros(len(flags), dtype=bool)
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= minimum:
                out[start:i] = True
            start = None
    if start is not None and len(flags) - start >= minimum:
        out[start:] = True
    return out


def ball_lock_from_aligned(aligned: dict[str, np.ndarray]) -> BallLock:
    """Build a :class:`BallLock` from the per-frame arrays ``gimbal_motion.align_log`` returns.

    The aligned arrays carry the lock flag and the ball's image x directly; the ball's image y comes from the box
    offset (``ball_ym``), which the alignment does not carry, so it is left NaN here and filled by
    :func:`ball_lock_from_logs` when the raw records are available.
    """
    locked = np.asarray(aligned["lock"], dtype=bool)
    u = np.asarray(aligned["ball_x"], dtype=np.float64) / LOG_FRAME_WIDTH
    frames = len(locked)
    return BallLock(
        locked=locked,
        u=u,
        v=np.full(frames, np.nan),
        size=np.full(frames, np.nan),
        speed=np.zeros(frames),
    )


def ball_lock_from_logs(logs, *, start_s: float, fps: float, frame_count: int) -> BallLock:
    """Build a :class:`BallLock` straight from parsed logs, with the ball's image y and size.

    ``logs`` is the ``(GimbalLog, clip_start_s)`` list the alignment uses. The ball's image y is the box's offset
    from the frame center plus the center, normalized by width; the speed is the magnitude of the logged image
    velocity (``xv``/``yv``), which is already in the log's pixels per frame.
    """
    from soccer_analytics.geometry.gimbal_motion import align_log

    aligned = align_log(logs, start_s=start_s, fps=fps, frame_count=frame_count)
    lock = ball_lock_from_aligned(aligned)
    # Fill v, size and speed from the raw records, sampled at the same analysis frames.
    times: list[float] = []
    v: list[float] = []
    size: list[float] = []
    speed: list[float] = []
    for log, clip_start in logs:
        for record in log.frames:
            if record.time_s is None:
                continue
            times.append(clip_start + record.time_s)
            if record.ball_ym is not None:
                v.append((LOG_FRAME_CENTER_Y + record.ball_ym) / LOG_FRAME_WIDTH)
            else:
                v.append(np.nan)
            size.append(float(record.ball_sz) if record.ball_sz is not None else np.nan)
            if record.ball_xv is not None and record.ball_yv is not None:
                speed.append(float(np.hypot(record.ball_xv, record.ball_yv)) / LOG_FRAME_WIDTH)
            else:
                speed.append(0.0)
    if not times:
        return lock
    order = np.argsort(times)
    times_arr = np.asarray(times)[order]
    at = start_s + np.arange(frame_count) / fps
    index = np.searchsorted(times_arr, at).clip(1, len(times_arr) - 1)
    left = np.abs(times_arr[index - 1] - at) < np.abs(times_arr[index] - at)
    index = np.where(left, index - 1, index)
    lock.v = np.asarray(v)[order][index]
    lock.size = np.asarray(size)[order][index]
    lock.speed = np.asarray(speed)[order][index]
    return lock


def image_space_possession(
    ball: BallLock | BallImage,
    det_frame: np.ndarray,
    det_box: np.ndarray,
    det_team: np.ndarray,
    *,
    radius: float = POSSESSION_IMAGE_RADIUS,
) -> dict:
    """Possession from the ball's image position and the players' image positions - no pitch, no homography.

    For each frame with a ball position, the nearest player *in the picture* is found (by the distance from the
    ball's image position to the player's foot point) and, if within ``radius``, that player's team is credited
    with the touch. Returns per-team touch counts, the share, and how many measured frames were contested.

    ``ball`` is either the gimbal's :class:`BallLock` or a :class:`BallImage` built from the ball scan. Both are
    picture measurements, so this survives a bad calibration; a lock with no ``v`` still works from horizontal
    distance. ``det_frame``/``det_box`` are the segment's detection arrays (box normalized by frame width,
    ``x1,y1,x2,y2``); ``det_team`` is the team label per detection (``-1`` for referee/unknown), which the caller
    gets from Stage B's track assignment. This answers "who is on the ball" without ever projecting to the ground.
    """
    image = ball_image_from_lock(ball) if isinstance(ball, BallLock) else ball
    by_frame: dict[int, list[int]] = {}
    for row, frame in enumerate(det_frame):
        by_frame.setdefault(int(frame), []).append(row)

    touches = {0: 0, 1: 0}
    contested = 0
    for frame in range(image.frames):
        if not image.valid[frame] or not np.isfinite(image.u[frame]):
            continue
        rows = by_frame.get(frame)
        if not rows:
            continue
        distance = foot_distance(det_box[rows], float(image.u[frame]), float(image.v[frame]))
        nearest = int(np.argmin(distance))
        if distance[nearest] > radius:
            continue
        team = int(det_team[rows[nearest]])
        if team in touches:
            touches[team] += 1
        contested += 1
    total = touches[0] + touches[1]
    return {
        "touches": touches,
        "contested_frames": contested,
        "share": {0: (touches[0] / total if total else 0.0), 1: (touches[1] / total if total else 0.0)},
    }
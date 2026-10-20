"""Recover a PTZ gimbal camera's motion from the footage itself.

A gimbal camera pans, tilts, rolls slightly and zooms about (nearly) its optical center, so successive frames are
related by ``H = K_c R K_g^-1``: a rotation ``R`` seen through intrinsics that differ only in focal length. We
therefore integrate camera *orientation* (a rotation matrix) and focal length, not free 8-parameter homographies.

That distinction matters. Frame-to-frame fits measured on real footage agree with a rotation+zoom model to the same
0.16 px as a free homography, but the *extra* freedom of a free homography is pure noise, and chained over a few
hundred frames it random-walks into runaway perspective terms until the homography's denominator crosses zero and
every point maps to infinity (observed: a 674x "zoom" after 70 s). A rotation matrix cannot do that.

Conventions
-----------
* Pixel coordinates are normalized by the analysis frame *width* (``u = x / W``, ``v = y / W``) so results do not
  depend on the resolution the motion was estimated at. Principal point is the frame center ``(0.5, 0.5 * H / W)``.
* ``to_ref`` maps normalized pixel coordinates of a frame to the *reference plane*: the normalized image plane of
  frame 0 of the segment. ``to_ref`` of frame 0 is the identity.
* Focal lengths are in frame widths. Real footage gives about 0.9 (a ~58 degree horizontal field of view).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Burned-in overlays (fractions x0, y0, x1, y1 of the frame): XbotGo logo bottom-right, wall clock bottom-left.
# They are static in the image, so they would vote for "no camera motion" in the motion estimate.
DEFAULT_OVERLAY_RECTS: tuple[tuple[float, float, float, float], ...] = (
    (0.845, 0.890, 0.990, 0.980),
    (0.005, 0.930, 0.105, 0.995),
)

DEFAULT_FOCAL = 0.82  # frame widths. Self-calibrated on 189 real large-rotation steps (median spread minimum at
# 0.81, valley 0.70-0.88); only weakly constrained by motion alone, so pitch calibration refines it from landmarks.
FOCAL_RANGE = (0.3, 6.0)  # a lens outside this range means the chain has gone wrong, not that the camera zoomed

_MAX_STEP_SCALE = 2.2
_MAX_STEP_PERSPECTIVE = 0.5
_ZOOM_SEARCH = 0.9  # |log(f_c / f_g)| searched per step
_ZOOM_DEADBAND = 0.0006  # ~7 sigma of the per-step zoom noise measured on real static footage (std 0.00008)


def overlay_mask(
    shape: tuple[int, int],
    rects: tuple[tuple[float, float, float, float], ...] = DEFAULT_OVERLAY_RECTS,
) -> np.ndarray:
    """uint8 mask for ``shape = (height, width)``: 255 where pixels may be used, 0 on overlays."""
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        mask[int(y0 * height) : int(np.ceil(y1 * height)), int(x0 * width) : int(np.ceil(x1 * width))] = 0
    return mask


def normalizer(width: int) -> np.ndarray:
    return np.array([[1.0 / width, 0.0, 0.0], [0.0, 1.0 / width, 0.0], [0.0, 0.0, 1.0]])


def intrinsics(focal: float, aspect: float) -> np.ndarray:
    """Normalized intrinsics; ``aspect = height / width``."""
    return np.array([[focal, 0.0, 0.5], [0.0, focal, 0.5 * aspect], [0.0, 0.0, 1.0]])


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Maps an (N, 2) array of points through a 3x3 homography."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(matrix, dtype=np.float64).T
    w = homogeneous[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homogeneous[:, :2] / w


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix[2, 2]


def step_is_plausible(step: np.ndarray) -> bool:
    """Cheap pre-gate on a free homography: rejects flips, extreme zoom jumps and strong perspective."""
    if not np.all(np.isfinite(step)):
        return False
    determinant = np.linalg.det(step[:2, :2])
    # A negative determinant is a mirror image: a classic degenerate RANSAC solution a real lens cannot produce.
    if determinant <= 0 or not (1.0 / _MAX_STEP_SCALE**2 <= determinant <= _MAX_STEP_SCALE**2):
        return False
    return abs(step[2, 0]) < _MAX_STEP_PERSPECTIVE and abs(step[2, 1]) < _MAX_STEP_PERSPECTIVE


# --------------------------------------------------------------------------------------------------------------
# Rotation + zoom model
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRotation:
    """A frame-to-frame homography projected onto the rotation+zoom manifold."""

    rotation: np.ndarray  # R_cg: rays of the good frame -> rays of the current frame
    focal: float  # focal length of the current frame
    spread: float  # 0 for an exact rotation+zoom; how far the measured homography is from one


def step_rotation(step: np.ndarray, focal_current: float, focal_good: float, aspect: float) -> np.ndarray:
    """Nearest rotation for a measured step when *both* frames' focal lengths are already known.

    This is `decompose_step` without its one-dimensional search: the search exists to discover ``f_c``, and when the
    focal lengths come from the analysis pass (``SegmentData.focal``) skipping it costs one SVD per step instead of
    a bounded minimization of many - the difference between a 20-minute segment taking a minute to load and taking a
    second. The rotation is identical: the same matrix is decomposed and projected onto SO(3) the same way.
    """
    homography = _unit(np.asarray(step, dtype=np.float64))
    m = np.linalg.inv(intrinsics(focal_current, aspect)) @ homography @ intrinsics(focal_good, aspect)
    u, _s, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def decompose_step(
    step: np.ndarray, focal_good: float, aspect: float, *, zoom_deadband: float = _ZOOM_DEADBAND
) -> StepRotation:
    """Fits ``step ~ K_c R K_g^-1`` for ``R`` and ``f_c``, given the good frame's focal length.

    ``K_c^-1 H K_g`` is a scaled rotation exactly when ``f_c`` is right, so we search the one unknown focal length
    for the value that makes its three singular values equal, then take the nearest rotation.
    """
    homography = _unit(np.asarray(step, dtype=np.float64))
    k_good = intrinsics(focal_good, aspect)

    def evaluate(log_ratio: float):
        focal = focal_good * float(np.exp(log_ratio))
        m = np.linalg.inv(intrinsics(focal, aspect)) @ homography @ k_good
        u, s, vt = np.linalg.svd(m)
        return float((s[0] - s[2]) / max(s.mean(), 1e-12)), u, vt, focal

    grid = np.linspace(-_ZOOM_SEARCH, _ZOOM_SEARCH, 37)
    costs = [evaluate(x)[0] for x in grid]
    best = int(np.argmin(costs))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    log_ratio = float(minimize_scalar(lambda x: evaluate(x)[0], bounds=(low, high), method="bounded").x)
    if abs(log_ratio) < zoom_deadband:
        log_ratio = 0.0
    spread, u, vt, focal = evaluate(log_ratio)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return StepRotation(rotation, focal, spread)


def to_reference(q: np.ndarray, focal: float, focal0: float, aspect: float) -> np.ndarray:
    """Frame -> reference-plane homography for orientation ``q`` (frame rays -> frame-0 rays) and focal ``focal``."""
    return _unit(intrinsics(focal0, aspect) @ q @ np.linalg.inv(intrinsics(focal, aspect)))


class RotationChain:
    """Integrates steps (always measured against the last good frame) into orientation and focal length."""

    def __init__(
        self,
        focal0: float = DEFAULT_FOCAL,
        aspect: float = 9 / 16,
        *,
        q: np.ndarray | None = None,
        focal: float | None = None,
    ):
        self.focal0 = focal0
        self.aspect = aspect
        self.q = np.eye(3) if q is None else np.asarray(q, dtype=np.float64)
        self.focal = focal0 if focal is None else focal

    def preview(self, step: np.ndarray) -> tuple[np.ndarray, float, float]:
        rot = decompose_step(step, self.focal, self.aspect)
        return self.q @ rot.rotation.T, rot.focal, rot.spread

    def commit(self, q: np.ndarray, focal: float) -> None:
        self.q, self.focal = q, focal

    @property
    def to_ref(self) -> np.ndarray:
        return to_reference(self.q, self.focal, self.focal0, self.aspect)


def integrate_poses(
    steps: list[np.ndarray | None],
    focal0: float = DEFAULT_FOCAL,
    aspect: float = 9 / 16,
    *,
    known_focals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame orientation ``Q`` (F, 3, 3) and focal length (F,) from stored steps (``None`` = no motion measured).

    ``Q`` maps frame rays to the segment's reference-frame rays; this is what pitch calibration consumes. Unlike
    ``integrate_steps`` it never builds a planar reference image, so it is valid at any pan angle.

    ``known_focals`` (the analysis pass's own per-frame focal lengths, ``SegmentData.focal``) skips the per-step
    focal search - see :func:`step_rotation`. Use it whenever the chain is being rebuilt for the footage it was
    measured from; leave it out when a different ``focal0`` is being explored.
    """
    chain = RotationChain(focal0, aspect)
    qs, focals = [], []
    for index, step in enumerate(steps):
        if step is not None:
            if known_focals is not None and index < len(known_focals) and float(known_focals[index]) > 0:
                focal = float(known_focals[index])
                chain.commit(chain.q @ step_rotation(step, focal, chain.focal, aspect).T, focal)
            else:
                q, focal, _spread = chain.preview(step)
                chain.commit(q, focal)
        qs.append(chain.q.copy())
        focals.append(chain.focal)
    return np.asarray(qs).reshape(-1, 3, 3), np.asarray(focals, dtype=np.float64)


def smooth_pose_chain(
    q: np.ndarray,
    focal: np.ndarray,
    fps: float,
    *,
    window_s: float = 0.75,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase smoothing of an integrated pose chain, for drawing and projecting through it.

    A gimbal log is a hardware measurement, but its yaw is quantized: the orientation derived from it holds for
    a few frames and then steps. Everything projected through that pose - the pitch markings, a player's feet,
    the animation's camera marker - jumps with it. A second-order Savitzky-Golay filter keeps the pan and zoom
    acceleration while removing the steps; it is zero-phase, so the smoothed chain does not lag the footage.
    A calibration fitted before the smoothing keeps working: on the reference match, smoothing moved the clicked
    landmark RMS from 87.3 to 87.1 px (the drift correction included).

    The reference frame is preserved in the mean, and a chain too short to filter is returned unchanged. Focal
    smoothing is skipped when the focal track is not all finite and positive, so a malformed track can never
    turn into a NaN pose.
    """
    q = np.asarray(q, dtype=np.float64)
    focal = np.asarray(focal, dtype=np.float64).reshape(-1)
    if q.ndim != 3 or q.shape[1:] != (3, 3) or len(q) != len(focal):
        raise ValueError("a pose chain needs (F, 3, 3) orientations and F focal lengths")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("the frame rate must be positive to size a smoothing window")
    frames = len(q)
    if frames < 5:
        return q.copy(), focal.copy()
    window = int(round(float(window_s) * float(fps))) | 1
    window = min(window, frames if frames % 2 else frames - 1)
    if window < 5:
        return q.copy(), focal.copy()
    polyorder = max(1, min(int(polyorder), window - 2))

    quat = Rotation.from_matrix(q).as_quat()
    for index in range(1, len(quat)):
        if float(np.dot(quat[index], quat[index - 1])) < 0.0:
            quat[index] = -quat[index]  # q and -q are the same rotation; a sign flip would look like a jump
    smoothed = savgol_filter(quat, window, polyorder, axis=0, mode="interp")
    lengths = np.linalg.norm(smoothed, axis=1, keepdims=True)
    if not np.all(lengths > 0.0):
        return q.copy(), focal.copy()
    q_smooth = Rotation.from_quat(smoothed / lengths).as_matrix()

    if np.all(np.isfinite(focal)) and np.all(focal > 0.0):
        focal_smooth = np.exp(savgol_filter(np.log(focal), window, polyorder, mode="interp"))
    else:
        focal_smooth = focal.copy()
    return q_smooth, focal_smooth


def integrate_steps(
    steps: list[np.ndarray | None], focal0: float = DEFAULT_FOCAL, aspect: float = 9 / 16
) -> list[np.ndarray]:
    """Rebuilds the ``to_ref`` chain from stored accepted steps (``None`` = frame was lost), for any ``focal0``.

    The first entry is the reference frame itself. Stored steps are raw measurements, so recalibrating the focal
    length later (e.g. from pitch landmarks) never requires re-reading the video.
    """
    chain = RotationChain(focal0, aspect)
    out: list[np.ndarray] = []
    for step in steps:
        if step is not None:
            q, focal, _spread = chain.preview(step)
            chain.commit(q, focal)
        out.append(chain.to_ref)
    return out


# --------------------------------------------------------------------------------------------------------------
# Per-pair estimation
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionStep:
    """Homography mapping frame A pixels to frame B pixels, with how trustworthy the fit is."""

    homography: np.ndarray
    inlier_ratio: float
    inliers: int
    method: str


def _fit(points_a: np.ndarray, points_b: np.ndarray, method: str) -> MotionStep | None:
    if len(points_a) < 12:
        return None
    homography, inlier_mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return None
    return MotionStep(homography, float(inlier_mask.mean()), int(inlier_mask.sum()), method)


def estimate_step_lk(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray | None) -> MotionStep | None:
    """Frame-to-frame motion from sparse optical flow on corner features (fast, small-to-medium motion)."""
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=800, qualityLevel=0.01, minDistance=10, mask=mask)
    if points is None or len(points) < 20:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    return _fit(points[ok].reshape(-1, 2), moved[ok].reshape(-1, 2), "lk")


def estimate_step_sift(
    prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray | None, *, features: int = 2000
) -> MotionStep | None:
    """Descriptor matching: slower, but copes with fast whip-pans and large zoom changes."""
    sift = cv2.SIFT_create(nfeatures=features)
    keypoints_a, descriptors_a = sift.detectAndCompute(prev_gray, mask)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray, mask)
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 12 or len(keypoints_b) < 12:
        return None
    pairs = cv2.BFMatcher().knnMatch(descriptors_a, descriptors_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])
    return _fit(points_a, points_b, "sift")


@dataclass(frozen=True)
class TrackerState:
    to_ref: np.ndarray
    ok: bool
    inlier_ratio: float
    method: str
    # Raw normalized step (last good frame -> this frame) that produced this state; None for init/lost frames.
    step: np.ndarray | None = None
    focal: float = DEFAULT_FOCAL
    spread: float = 0.0


class CameraMotionTracker:
    """Streams grayscale frames in and keeps the camera's orientation and zoom up to date.

    Every frame is measured against the last frame we could place (not simply the previous one), so a lost frame
    cannot leak a bad reference into the chain. Frames that cannot be placed are flagged ``ok=False`` with the
    state held at its last value.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        *,
        focal0: float = DEFAULT_FOCAL,
        overlay_rects: tuple[tuple[float, float, float, float], ...] = DEFAULT_OVERLAY_RECTS,
        min_inlier_ratio: float = 0.30,
        min_inliers: int = 25,
        max_spread: float = 0.12,
        chain: RotationChain | None = None,
    ):
        self.height, self.width = frame_shape
        self.mask = overlay_mask(frame_shape, overlay_rects)
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inliers = min_inliers
        self.max_spread = max_spread
        self._norm = normalizer(self.width)
        self._norm_inv = np.linalg.inv(self._norm)
        self.chain = chain or RotationChain(focal0, self.height / self.width)
        self._good_gray: np.ndarray | None = None

    def seed_good_frame(self, gray: np.ndarray) -> None:
        """Resume mid-segment: ``gray`` is the frame the restored ``chain`` state belongs to."""
        self._good_gray = gray

    def _try_step(self, step: MotionStep | None) -> TrackerState | None:
        if step is None or step.inliers < self.min_inliers or step.inlier_ratio < self.min_inlier_ratio:
            return None
        normalized = _unit(self._norm @ step.homography @ self._norm_inv)
        if not step_is_plausible(normalized):
            return None
        q, focal, spread = self.chain.preview(normalized)
        if spread > self.max_spread or not (FOCAL_RANGE[0] <= focal <= FOCAL_RANGE[1]):
            return None  # not something a pan/tilt/zoom camera can do: the fit latched onto the wrong thing
        self.chain.commit(q, focal)
        return TrackerState(self.chain.to_ref, True, step.inlier_ratio, step.method, normalized, focal, spread)

    def update(self, gray: np.ndarray) -> TrackerState:
        if gray.shape != (self.height, self.width):
            raise ValueError(f"expected frame shape {(self.height, self.width)}, got {gray.shape}")
        if self._good_gray is None:
            self._good_gray = gray
            return TrackerState(self.chain.to_ref, True, 1.0, "init", None, self.chain.focal)

        state = self._try_step(estimate_step_lk(self._good_gray, gray, self.mask))
        if state is None:
            # Optical flow gave up (blur, whip-pan): fall back to descriptor matching against the same frame.
            state = self._try_step(estimate_step_sift(self._good_gray, gray, self.mask))
        if state is None:
            return TrackerState(self.chain.to_ref, False, 0.0, "lost", None, self.chain.focal)
        self._good_gray = gray
        return state

"""Camera-motion recovery against a physically exact synthetic camera.

The scene is a distant textured plane on a wide image; frames are rendered by warping it through
``K(f) R K(f0)^-1`` which is precisely what a pinhole camera rotating (and zooming) about its optical center sees.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from soccer_analytics.analysis import projection
from soccer_analytics.analysis.projection import segment_poses
from soccer_analytics.geometry.camera_motion import (
    DEFAULT_FOCAL,
    CameraMotionTracker,
    RotationChain,
    apply_homography,
    decompose_step,
    integrate_steps,
    intrinsics,
    normalizer,
    overlay_mask,
    smooth_pose_chain,
    step_is_plausible,
    to_reference,
)

W, H = 640, 360
ASPECT = H / W
F0 = DEFAULT_FOCAL


def _scene(seed: int = 3) -> np.ndarray:
    """Textured image large enough to be seen at +-25 degrees; represents what frame 0 sees at f=F0, widened."""
    rng = np.random.default_rng(seed)
    height, width = 1800, 2800
    layers = [cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), s) for s in (1.5, 4.0, 12.0)]
    img = sum(layer / layer.std() for layer in layers)
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


SCENE = _scene()
# Scene pixel (u, v) <-> unit-plane ray: the scene is centered and viewed at focal length S_F (in scene widths).
_SW, _SH = SCENE.shape[1], SCENE.shape[0]


def _rot(pan: float, tilt: float, roll: float) -> np.ndarray:
    ry, _ = cv2.Rodrigues(np.array([0.0, pan, 0.0]))
    rx, _ = cv2.Rodrigues(np.array([tilt, 0.0, 0.0]))
    rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, roll]))
    return ry @ rx @ rz


def _render(pan: float, tilt: float, roll: float, focal: float, logo: bool = True) -> np.ndarray:
    """Frame (W x H) of the scene seen by a camera with the given orientation and focal length (frame widths)."""
    k_frame = np.array([[focal * W, 0, W / 2], [0, focal * W, H / 2], [0, 0, 1.0]])
    k_scene = np.array([[F0 * _SW * 0.55, 0, _SW / 2], [0, F0 * _SW * 0.55, _SH / 2], [0, 0, 1.0]])
    # scene pixel = K_scene R K_frame^-1 frame pixel  (frame -> scene), warpPerspective wants this direction
    frame_to_scene = k_scene @ _rot(pan, tilt, roll) @ np.linalg.inv(k_frame)
    frame = cv2.warpPerspective(SCENE, frame_to_scene, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    if logo:  # static overlay, identical in every frame, exactly where the real logo sits
        rng = np.random.default_rng(99)
        x0, y0, x1, y1 = int(0.85 * W), int(0.90 * H), int(0.985 * W), int(0.975 * H)
        frame[y0:y1, x0:x1] = (rng.random((y1 - y0, x1 - x0)) * 255).astype(np.uint8)
    return frame


def _true_to_ref(pan: float, tilt: float, roll: float, focal: float, pose0: tuple, focal0: float) -> np.ndarray:
    """Exact frame -> reference-plane homography for the synthetic camera (reference = frame 0)."""
    r0, r = _rot(*pose0), _rot(pan, tilt, roll)
    # frame ray -> scene ray is R; reference frame ray = R0^-1 R frame ray
    return intrinsics(focal0, ASPECT) @ r0.T @ r @ np.linalg.inv(intrinsics(focal, ASPECT))


def _error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean disagreement, in frame widths, over a grid of normalized image points."""
    grid = np.array([[u, v] for u in np.linspace(0.05, 0.95, 5) for v in np.linspace(0.05, 0.5, 4)])
    return float(np.linalg.norm(apply_homography(a, grid) - apply_homography(b, grid), axis=1).mean())


def _run(poses: list[tuple], tracker: CameraMotionTracker | None = None) -> tuple[list, CameraMotionTracker]:
    tracker = tracker or CameraMotionTracker((H, W))
    states = [tracker.update(_render(*pose)) for pose in poses]
    return states, tracker


# --------------------------------------------------------------------------------------------------------------


def test_overlay_mask_blocks_logo_and_clock_only() -> None:
    mask = overlay_mask((H, W))
    assert mask[int(0.93 * H), int(0.92 * W)] == 0  # logo
    assert mask[int(0.96 * H), int(0.05 * W)] == 0  # clock
    assert mask[H // 2, W // 2] == 255
    assert 0.9 < (mask > 0).mean() < 1.0


def test_step_plausibility_rejects_impossible_fits() -> None:
    assert step_is_plausible(np.eye(3))
    assert not step_is_plausible(np.diag([5.0, 5.0, 1.0]))  # 5x zoom between consecutive frames
    assert not step_is_plausible(np.diag([-1.0, 1.0, 1.0]))  # mirror
    assert not step_is_plausible(np.array([[1, 0, 0], [0, 1, 0], [0.9, 0, 1.0]]))  # extreme perspective
    assert not step_is_plausible(np.full((3, 3), np.nan))


@pytest.mark.parametrize("pan,tilt,roll,zoom", [(0.05, 0.0, 0.0, 1.0), (0.0, -0.04, 0.0, 1.0), (0.03, 0.02, 0.01, 1.15)])
def test_decompose_recovers_known_rotation_and_zoom(pan: float, tilt: float, roll: float, zoom: float) -> None:
    r = _rot(pan, tilt, roll)
    f_good, f_cur = F0, F0 * zoom
    step = intrinsics(f_cur, ASPECT) @ r.T @ np.linalg.inv(intrinsics(f_good, ASPECT))  # good frame -> current
    out = decompose_step(step / step[2, 2], f_good, ASPECT)
    assert out.focal == pytest.approx(f_cur, rel=2e-3)
    assert np.allclose(out.rotation, r.T, atol=2e-3)
    assert out.spread < 1e-3


def test_decompose_flags_homography_that_is_not_a_rotation() -> None:
    shear = np.array([[1.0, 0.35, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert decompose_step(shear, F0, ASPECT).spread > 0.05


def test_recovers_pan_tilt_roll_zoom_trajectory() -> None:
    poses = [(0.012 * i, -0.004 * i, 0.0006 * i, F0 * (1 + 0.003 * i)) for i in range(50)]
    states, _ = _run(poses)
    assert all(s.ok for s in states)
    errors = [_error(s.to_ref, _true_to_ref(*p, poses[0][:3], poses[0][3])) for s, p in zip(states, poses)]
    assert errors[0] == pytest.approx(0.0, abs=1e-9)
    assert max(errors) < 0.02  # < 2% of frame width after 50 frames of simultaneous pan, tilt, roll and zoom
    assert states[-1].focal == pytest.approx(poses[-1][3], rel=0.05)


def test_static_logo_does_not_pull_estimate_toward_zero_motion() -> None:
    poses = [(0.015 * i, 0.0, 0.0, F0) for i in range(30)]
    states, _ = _run(poses)
    true = _true_to_ref(*poses[-1], poses[0][:3], poses[0][3])
    assert _error(states[-1].to_ref, true) < 0.01
    # The recovered pan must be the real one, not damped by the static overlay.
    assert apply_homography(states[-1].to_ref, np.array([[0.5, 0.28]]))[0, 0] == pytest.approx(
        apply_homography(true, np.array([[0.5, 0.28]]))[0, 0], abs=0.02
    )


def test_long_pan_back_and_forth_does_not_drift_into_degenerate_homography() -> None:
    """The failure that motivated the rotation model: chained free homographies gain runaway perspective."""
    sweep = [0.02 * np.sin(2 * np.pi * i / 60) * 6 for i in range(180)]  # three full left-right sweeps
    poses = [(a, 0.0, 0.0, F0) for a in sweep]
    states, _ = _run(poses)
    assert all(s.ok for s in states)
    for s in states:
        w = s.to_ref[2, 0] * 0.5 + s.to_ref[2, 1] * 0.28 + s.to_ref[2, 2]
        assert 0.5 < w < 2.0  # the homography denominator never approaches zero
        assert 0.7 < s.focal / F0 < 1.4  # and the lens never "zooms" without us asking
    # Returning to the starting pose returns the chain to (nearly) the identity.
    assert _error(states[-1].to_ref, _true_to_ref(*poses[-1], poses[0][:3], poses[0][3])) < 0.03


def test_lost_frame_is_flagged_and_chain_recovers_without_offset() -> None:
    poses = [(0.012 * i, -0.003 * i, 0.0, F0) for i in range(20)]
    tracker = CameraMotionTracker((H, W))
    states = []
    for i, pose in enumerate(poses):
        frame = np.full((H, W), 127, dtype=np.uint8) if i in (8, 9) else _render(*pose)
        states.append(tracker.update(frame))
    assert [s.ok for s in states[8:10]] == [False, False]
    assert all(s.method == "lost" for s in states[8:10])
    assert np.allclose(states[8].to_ref, states[7].to_ref)  # held at the last good value while lost
    for i in (10, 15, 19):
        assert states[i].ok
        assert _error(states[i].to_ref, _true_to_ref(*poses[i], poses[0][:3], poses[0][3])) < 0.02


def test_camera_leaving_all_texture_is_reported_lost() -> None:
    tracker = CameraMotionTracker((H, W))
    assert tracker.update(_render(0.0, 0.0, 0.0, F0)).ok
    state = tracker.update(np.full((H, W), 90, dtype=np.uint8))
    assert not state.ok and state.method == "lost"


def test_rejects_wrong_frame_shape() -> None:
    tracker = CameraMotionTracker((H, W))
    with pytest.raises(ValueError):
        tracker.update(np.zeros((H + 1, W), dtype=np.uint8))


def test_featureless_frames_never_invent_motion() -> None:
    tracker = CameraMotionTracker((H, W))
    flat = np.full((H, W), 90, dtype=np.uint8)
    first, second = tracker.update(flat), tracker.update(flat)
    assert first.ok and not second.ok
    assert np.allclose(second.to_ref, np.eye(3))


def test_integrate_steps_reproduces_the_live_chain_and_supports_recalibration() -> None:
    poses = [(0.014 * i, -0.002 * i, 0.0, F0) for i in range(25)]
    states, _ = _run(poses)
    steps = [s.step for s in states]  # first is None (init)
    rebuilt = integrate_steps(steps, focal0=F0, aspect=ASPECT)
    for live, again in zip(states, rebuilt):
        assert np.allclose(live.to_ref, again, atol=1e-9)
    # A different assumed focal length rescales the pan but keeps the chain well-formed (no re-read of video).
    other = integrate_steps(steps, focal0=F0 * 1.3, aspect=ASPECT)
    assert all(np.all(np.isfinite(m)) for m in other)
    assert not np.allclose(other[-1], rebuilt[-1])


def test_rotation_chain_to_reference_is_identity_at_start() -> None:
    chain = RotationChain(F0, ASPECT)
    assert np.allclose(chain.to_ref, np.eye(3))
    assert np.allclose(to_reference(np.eye(3), F0, F0, ASPECT), np.eye(3))
    assert np.allclose(normalizer(W) @ np.linalg.inv(normalizer(W)), np.eye(3))


def _pan_series(angles_deg: np.ndarray) -> np.ndarray:
    """Rotations about the image vertical, the axis a pan moves about in this model."""
    vectors = np.column_stack([np.zeros(len(angles_deg)), np.zeros(len(angles_deg)), np.radians(angles_deg)])
    return Rotation.from_rotvec(vectors).as_matrix()


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = (np.einsum("fij,fij->f", first, second) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def test_pose_smoothing_removes_quantized_yaw_steps_without_lagging() -> None:
    """A logged yaw that holds and then jumps is the jitter seen in the projected markings and players.

    The filter must remove the steps while still following the ideal pan - a smoother that lags would trade
    a jump for a trail, which is the problem this is meant to fix.
    """
    fps = 15.0
    times = np.arange(0.0, 12.0, 1.0 / fps)
    ideal = 8.0 * times  # a constant 8 deg/s pan
    quantized = np.round(ideal / 2.0) * 2.0  # the log's 2-degree yaw steps
    q = _pan_series(quantized)
    focal = np.full(len(times), 0.82)
    smooth, smooth_focal = smooth_pose_chain(q, focal, fps)
    assert np.allclose(smooth_focal, focal)

    ideal_q = _pan_series(ideal)
    raw_error = _rotation_error_deg(q, ideal_q)
    smooth_error = _rotation_error_deg(smooth, ideal_q)
    assert np.max(smooth_error) < np.max(raw_error) * 0.75
    # Zero phase: the smoothed angle follows the ideal pan instead of trailing it.
    angles = np.degrees(Rotation.from_matrix(smooth).as_euler("xyz")[:, 2])
    lag = angles - ideal
    assert abs(float(np.mean(lag))) < 0.15
    assert abs(float(lag[-1])) < 0.2
    # The high-frequency part is what the eye sees as jitter.
    raw_jitter = np.abs(np.diff(raw_error, n=2))
    smooth_jitter = np.abs(np.diff(smooth_error, n=2))
    assert np.percentile(smooth_jitter, 95) < np.percentile(raw_jitter, 95) * 0.5


def test_pose_smoothing_preserves_a_constant_pan_and_zoom() -> None:
    """Second-order filtering is exact for constant velocity: no reshape of real camera motion."""
    fps = 15.0
    times = np.arange(0.0, 8.0, 1.0 / fps)
    q = _pan_series(3.0 * times)
    focal = 0.82 * np.exp(0.002 * times)
    smooth, smooth_focal = smooth_pose_chain(q, focal, fps)
    assert np.max(_rotation_error_deg(smooth, q)) < 1e-3
    assert np.allclose(smooth_focal, focal, rtol=1e-6)


def test_pose_smoothing_reduces_focal_jitter() -> None:
    """The focal comes from the optical-flow chain; alternating estimator noise must not move the markings."""
    fps = 15.0
    times = np.arange(0.0, 10.0, 1.0 / fps)
    rng = np.random.default_rng(7)
    ideal = 0.82 * np.exp(0.001 * times)
    jittery = ideal * np.exp(rng.normal(0.0, 0.004, len(times)))
    q = _pan_series(2.0 * times)
    _smooth_q, smooth_focal = smooth_pose_chain(q, jittery, fps)
    assert np.std(np.diff(np.log(smooth_focal))) < np.std(np.diff(np.log(jittery))) * 0.5
    assert abs(float(np.mean(np.log(smooth_focal) - np.log(ideal)))) < 0.001


def test_a_short_or_malformed_chain_is_left_alone() -> None:
    q = _pan_series(np.array([0.0, 1.0, 2.0, 3.0]))
    focal = np.full(4, 0.82)
    smooth, smooth_focal = smooth_pose_chain(q, focal, 15.0)
    assert np.array_equal(smooth, q) and np.array_equal(smooth_focal, focal)
    with pytest.raises(ValueError):
        smooth_pose_chain(q, focal[:2], 15.0)
    with pytest.raises(ValueError):
        smooth_pose_chain(q, focal, 0.0)


def test_segment_poses_projects_through_a_smoothed_chain(monkeypatch) -> None:
    """The projection and the stream both read segment_poses; the smoothing has to happen there, not in one caller."""
    fps, frames = 15.0, 120
    times = np.arange(frames) / fps
    raw_q = _pan_series(4.0 * times)
    stepped_q = _pan_series(np.round(4.0 * times))  # what a quantized log contributes when it holds and steps
    focal = np.full(frames, 0.82)
    segment = SimpleNamespace(
        ok=np.ones(frames, dtype=bool),
        step=[None] + [np.eye(3)] * (frames - 1),
        focal=focal,
        aspect=9 / 16,
        meta={"default_focal": 0.82, "fps": fps},
    )
    monkeypatch.setattr(projection, "integrate_poses", lambda *args, **kwargs: (raw_q.copy(), focal.copy()))
    monkeypatch.setattr(projection, "log_poses_for_segment", lambda *args, **kwargs: (stepped_q, None, {}))

    q, _focal = segment_poses(segment)

    def jitter(series: np.ndarray) -> float:
        angles = np.degrees(Rotation.from_matrix(series).as_euler("xyz")[:, 2])
        return float(np.mean(np.abs(np.diff(angles, n=2))))

    assert jitter(q) < jitter(stepped_q) * 0.5
    assert not np.allclose(q, stepped_q), "the stepped log must actually be filtered"
    assert not np.allclose(q, raw_q), "the log orientation is the one projected, not the raw chain"

"""The gimbal log as BoT-SORT's camera-motion compensation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from soccer_analytics.analysis.log_motion import LogGMC, LogMotion, install_log_gmc
from soccer_analytics.geometry.camera_motion import to_reference
from soccer_analytics.geometry.gimbal_motion import PanModel, log_orientation


def _synthetic_pan(frames: int = 400) -> tuple[LogMotion, np.ndarray]:
    tilt = 11.6
    axis = np.array([0.0, np.sin(np.radians(90 - tilt)), np.cos(np.radians(90 - tilt))])
    yaw = np.linspace(-40.0, 40.0, frames)
    pitch = np.full(frames, tilt)
    model = PanModel(axis=axis, scale=1.0, yaw0=yaw[0], rms_deg=0.0, samples=frames)
    orientation = log_orientation(yaw, model)
    motion = LogMotion(yaw, pitch, focal0=0.82, aspect=9 / 16, width=3840, height=2160)
    for index in range(frames):
        motion.observe(index, 0.82)
        motion.add_pose(orientation[index], True)
    return motion, orientation


def test_the_log_warp_matches_the_camera_rotation() -> None:
    motion, orientation = _synthetic_pan()
    assert motion.ready, "300+ posed frames should fit the pan model"

    warp = motion.warp(1)
    assert warp is not None and warp.shape == (2, 3)

    previous = to_reference(orientation[0], 0.82, 0.82, 9 / 16)
    current = to_reference(orientation[1], 0.82, 0.82, 9 / 16)
    step = np.linalg.inv(current) @ previous
    scale = np.diag([3840.0, 3840.0, 1.0])
    step_pixels = scale @ step @ np.diag([1 / 3840, 1 / 3840, 1.0])

    points = np.array([[0, 0], [3840, 0], [0, 2160], [3840, 2160], [1920, 1080]], dtype=np.float32)
    exact_h = np.array(
        [[step_pixels[0, 0] * x + step_pixels[0, 1] * y + step_pixels[0, 2],
          step_pixels[1, 0] * x + step_pixels[1, 1] * y + step_pixels[1, 2]] for x, y in points]
    )
    exact = exact_h / np.array(
        [[step_pixels[2, 0] * x + step_pixels[2, 1] * y + step_pixels[2, 2]] for x, y in points]
    )
    approximated = np.array([[warp[0, 0] * x + warp[0, 1] * y + warp[0, 2],
                              warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]] for x, y in points])

    # A rotation+zoom is not exactly a similarity; over one small pan step the similarity fit should still land
    # within a couple of pixels at 4K.
    assert np.abs(exact - approximated).max() < 8.0


def test_a_measured_chain_step_is_preferred_over_the_logged_orientation() -> None:
    """The chain's step is what the optical GMC would have measured; a logged yaw discontinuity must not win."""
    motion, _ = _synthetic_pan()
    motion.observe_chain(1, np.eye(3), 0.82, True, np.eye(3))
    warp = motion.warp(1)
    assert warp is not None
    # The identity chain step wins over the synthetic pan's logged rotation.
    assert np.allclose(warp, np.eye(2, 3), atol=1e-6)


def test_the_log_gmc_falls_back_until_the_pan_model_is_ready() -> None:
    calls = {"n": 0}

    class _Fallback:
        method = "sparseOptFlow"

        def apply(self, frame, detections=None):
            calls["n"] += 1
            return np.eye(2, 3, dtype=np.float32)

    motion = LogMotion(np.zeros(10), np.zeros(10), focal0=0.82, aspect=9 / 16, width=1280, height=720)
    gmc = LogGMC(motion, _Fallback())
    gmc.frame_index = 1
    gmc.apply(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert calls["n"] == 1

    # Setting a fitted orientation is enough for the log path; the fallback must not be touched again.
    motion.q = np.tile(np.eye(3), (10, 1, 1))
    motion.focals[0] = motion.focals[1] = 0.82
    warp = gmc.apply(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert calls["n"] == 1
    assert warp is not None and warp.shape == (2, 3)


def test_install_log_gmc_wraps_an_existing_tracker_gmc() -> None:
    motion = LogMotion(np.zeros(10), np.zeros(10), focal0=0.82, aspect=9 / 16, width=1280, height=720)

    class _Fallback:
        method = "sparseOptFlow"

        def apply(self, frame, detections=None):
            return np.eye(2, 3, dtype=np.float32)

    tracker = SimpleNamespace(gmc=_Fallback())
    model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))
    installed = install_log_gmc(model, motion)
    assert installed is not None
    assert tracker.gmc is installed
    assert install_log_gmc(model, motion) is installed  # re-installing the same motion is a no-op

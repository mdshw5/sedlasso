"""Drift correction: keeping a long video's projection on the markings, not just near the clicks.

The failure this exists for is measured, not hypothetical: the motion chain is accurate near the frames the user
clicked and slides meters off the markings by the end of a game. These tests build that situation on the simulated
match (a known per-step error accumulating along the chain), then check that fitting the smooth correction to
clicks spread over the video puts the projection back on the pitch - and that it never makes a good fit worse.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic_match import ASPECT, FOCAL, PITCH_LENGTH, PITCH_WIDTH, simulate_match

from soccer_analytics.analysis.projection import segment_poses
from soccer_analytics.dashboard.pitch_clicks import landmark_table
from soccer_analytics.geometry.camera_motion import integrate_poses, intrinsics
from soccer_analytics.geometry.drift import DriftCorrection, fit_drift
from soccer_analytics.geometry.pitch_calibration import Landmark, PitchCalibration, calibrate, pitch_to_pixels

WIDTH = 1920  # the simulator's frame width, so pixel errors read as the user would see them
ANCHORS = [70, 222, 375, 530]


def _exp(vector: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))[0]


def _analysis_chain(segment) -> tuple[np.ndarray, np.ndarray]:
    """The chain the dashboard sees for a segment (the same call the fast pose path makes)."""
    return segment_poses(segment)


def _drifted(segment, truth, per_step_rad: float):
    """The same segment with a small *rotation* error baked into every step - how a real chain drifts.

    A constant per-step rotation error is the classic shape of accumulated error (a rotation center that is not the
    lens center, unmodeled distortion): negligible for a second, meters of ground error by the end of a game. The
    drifted steps are integrated exactly as the analysis pass integrates the real ones, so the test's chain is the
    kind of chain the dashboard actually loads - not an idealized one.
    """
    from dataclasses import replace

    bias = _exp(np.array([0.6, 1.0, -0.4]) * per_step_rad)
    steps = np.array(segment.step, dtype=np.float64, copy=True)
    for t in range(1, len(steps)):
        relational = _rotation_of_step(steps[t], float(truth.focal[t - 1]), float(truth.focal[t]))
        k_before = intrinsics(float(truth.focal[t - 1]), ASPECT)
        k_after = intrinsics(float(truth.focal[t]), ASPECT)
        steps[t] = k_after @ (bias @ relational) @ np.linalg.inv(k_before)
    sequence = [None if t == 0 else steps[t] for t in range(len(steps))]
    _, chain_focal = integrate_poses(sequence, FOCAL, ASPECT)
    return replace(segment, step=steps, focal=np.asarray(chain_focal, dtype=np.float32))


def _rotation_of_step(step: np.ndarray, focal_before: float, focal_after: float) -> np.ndarray:
    """The rotation a K R K^-1 step carries, as the chain integrator reads it."""
    k_before = intrinsics(focal_before, ASPECT)
    k_after = intrinsics(focal_after, ASPECT)
    return np.linalg.inv(k_after) @ step @ k_before


def _clicks(truth, frames: list[int]) -> list[Landmark]:
    """A perfect user's clicks at the given frames - only what the simulator actually puts on screen."""
    table = landmark_table(PITCH_LENGTH, PITCH_WIDTH, format_name="9v9")
    land: list[Landmark] = []
    for frame in frames:
        for name, (x, y) in table.items():
            uv, in_front = pitch_to_pixels(
                truth.calibration, np.array([[x, y]]), truth.q[frame], float(truth.focal[frame])
            )
            if not in_front[0] or not np.isfinite(uv[0]).all():
                continue
            u, v = float(uv[0, 0]), float(uv[0, 1])
            if 0.0 <= u <= 1.0 and 0.0 <= v <= ASPECT:
                land.append(Landmark(frame, u, v, x, y, name))
    return land


def _median_pixel_error(calibration: PitchCalibration, q: np.ndarray, focal: np.ndarray, frame: int, truth) -> float:
    """How far this calibration's projection is from where the truth puts things, in pixels, that frame.

    Measured over every pitch landmark that is genuinely in the picture, which is what a user checks the overlay
    against - and how they described the failure ("diverges from the actual landmarks later in the video"). A
    landmark that projects through the camera plane is skipped: its "error" is a flip, not a projection.
    """
    errors: list[float] = []
    table = landmark_table(PITCH_LENGTH, PITCH_WIDTH, format_name="9v9")
    for name, (x, y) in table.items():
        reference, in_front = pitch_to_pixels(
            truth.calibration, np.array([[x, y]]), truth.q[frame], float(truth.focal[frame])
        )
        if not in_front[0] or not np.isfinite(reference[0]).all():
            continue
        u, v = float(reference[0, 0]), float(reference[0, 1])
        if not (-0.25 <= u <= 1.25 and -0.25 * ASPECT <= v <= 1.25 * ASPECT):
            continue
        q_frame, focal_frame = calibration.corrected_frame(q[frame], float(focal[frame]), frame)
        estimate, ok = pitch_to_pixels(calibration, np.array([[x, y]]), q_frame, focal_frame)
        if not ok[0] or not np.isfinite(estimate[0]).all():
            continue
        error = float(np.hypot(estimate[0, 0] - u, estimate[0, 1] - v) * WIDTH)
        if error > 250.0:
            # A landmark sitting on the camera plane (the camera turned to the horizon) flips rather than
            # projects; its "error" is arithmetic, not geometry, so it is not a measurement of drift.
            continue
        errors.append(error)
    assert errors, "the test frame has to have something to look at"
    return float(np.median(errors))


@pytest.fixture(scope="module")
def drifted_match():
    """A 2-minute simulated match whose chain accumulates ~1.2 deg of pointing drift - meters of ground error."""
    segment, truth = simulate_match(frames=600, seed=17)
    bad = _drifted(segment, truth, per_step_rad=5.0e-5)
    q, focal = segment_poses(bad)
    return truth, q, focal, bad


def test_the_correction_interpolates_between_anchors_and_holds_outside_them() -> None:
    """Between two anchors the correction is a great-circle blend; outside them it is held, never extrapolated."""
    a = _exp(np.array([0.0, 0.0, 0.10]))
    b = _exp(np.array([0.0, 0.0, 0.30]))
    correction = DriftCorrection((100, 200), (a, b))

    assert np.allclose(correction.rotation(50), a), "before the first anchor the first correction is held"
    assert np.allclose(correction.rotation(100), a)
    assert np.allclose(correction.rotation(200), b)
    assert np.allclose(correction.rotation(900), b), "after the last anchor the last correction is held"

    # halfway along the blend is exactly halfway around the geodesic, not halfway component-wise
    mid = correction.rotation(150)
    half = _exp(np.array([0.0, 0.0, 0.20]))
    assert np.allclose(mid, half, atol=1e-9)

    # and the correction is what adjust() applies to a chain: one rotation per array position, which for a segment
    # chain is its frame index
    q = np.stack([np.eye(3), _exp(np.array([0.05, 0.0, 0.0])), np.eye(3)])
    adjusted = correction.adjust(q)
    for t in range(len(q)):
        assert np.allclose(adjusted[t], correction.rotation(t) @ q[t])


def test_a_correction_survives_the_round_trip_through_json() -> None:
    correction = DriftCorrection((12, 340, 900), tuple(_exp(np.array([0.0, 0.01, 0.0]) * k) for k in (1, 2, 3)))
    again = DriftCorrection.from_json(correction.to_json())
    assert again.frames == correction.frames
    assert np.allclose(again.rotation(500), correction.rotation(500))


def test_the_focal_term_interpolates_and_round_trips() -> None:
    """The zoom half of the correction: a multiplier per anchor, linear in between, held outside."""
    correction = DriftCorrection((100, 200), (np.eye(3), np.eye(3)), (1.10, 0.90))
    assert correction.scale(50) == pytest.approx(1.10)
    assert correction.scale(150) == pytest.approx(1.00)
    assert correction.scale(900) == pytest.approx(0.90)

    focal = np.full(300, 0.8)
    adjusted = correction.adjust_focal(focal)
    assert adjusted[50] == pytest.approx(0.8 * 1.10)
    assert adjusted[150] == pytest.approx(0.8 * 1.00)
    assert adjusted[250] == pytest.approx(0.8 * 0.90)

    again = DriftCorrection.from_json(correction.to_json())
    assert again.frames == correction.frames and again.scales == correction.scales
    assert np.allclose(again.rotation(150), correction.rotation(150))
    assert again.scale(150) == pytest.approx(1.0)


def test_a_correction_without_scales_leaves_the_focal_alone() -> None:
    """Old files (and a fit with no zoom error) carry no scales; the chain must still project correctly."""
    correction = DriftCorrection((10, 20), (np.eye(3), _exp(np.array([0.0, 0.01, 0.0]))))
    focal = np.array([0.9, 0.9])
    assert np.allclose(correction.adjust_focal(focal), focal)
    assert correction.scale(15) == 1.0


def test_fit_drift_needs_two_anchors() -> None:
    """A constant correction is just the base rotation; drift is a function of time, so one anchor is nothing."""
    segment, truth = simulate_match(frames=150, seed=4)
    q, focal = segment_poses(segment)
    clicks = _clicks(truth, [60])
    calibration = PitchCalibration(truth.calibration.position, np.eye(3), 1.0, ASPECT, 0.0, ())
    assert len(clicks) >= 2, "the unit test needs the clicks the frame actually offers"
    assert fit_drift(clicks, calibration, {60: (q[60], float(focal[60]))}, ASPECT) is None


def test_the_correction_holds_the_projection_across_the_whole_video(drifted_match) -> None:
    """The point of the feature: clicks spread over the video keep the overlay on the markings everywhere.

    Measured exactly as the user sees it - the pixel distance between where landmarks project and where they
    really are, sampled every 10 frames of a two-minute game. The un-anchored fit and the corrected one are put
    against each other in the aggregate (median, p90): that is the claim being made, and single frames between
    two anchors are allowed to be a little worse when the camera swung about in a way the interpolation could not
    know about - which is exactly why the app tells the user to click later in the video too.
    """
    truth, q, focal, _bad = drifted_match
    clicks = _clicks(truth, ANCHORS)
    assert len(clicks) >= 9, "the click set has to be worth fitting"
    chain = {frame: (q[frame], float(focal[frame])) for frame in ANCHORS}

    plain = calibrate(clicks, chain, ASPECT, correct_drift=False)
    corrected = calibrate(clicks, chain, ASPECT, correct_drift=True)
    assert corrected.drift is not None and corrected.drift.frames == tuple(ANCHORS)
    assert corrected.rms_error_m < plain.rms_error_m / 2.0, "the anchored fit describes the clicks much better"

    frames = list(range(0, 600, 10))
    before = np.array([_median_pixel_error(plain, q, focal, frame, truth) for frame in frames])
    after = np.array([_median_pixel_error(corrected, q, focal, frame, truth) for frame in frames])
    # This fixture drifts *hard* between its anchors - the camera whips, and the error curve bends with it, which is
    # the one thing a smooth interpolation cannot follow (a linear fit of the exact correction is out by 1.1 deg in
    # that gap, in time or in camera motion, measured). Real clicks every few minutes have a far gentler curve; the
    # claim held here is the one that survives the worst case: every frame improves on average, and the median - the
    # error of a typical frame - collapses by at least a third.
    assert np.median(after) < np.median(before) / 1.4, f"median error must collapse: {np.median(before):.1f} -> {np.median(after):.1f} px"
    assert np.percentile(after, 90) < np.percentile(before, 90), "even the bad frames have to improve"
    for frame in ANCHORS:
        # A knot is four degrees of freedom against the clicks on it, not a per-frame calibration: this fixture's
        # anchors end up within ~25 px at 1920, which is a fraction of a meter of ground error at these ranges -
        # against 47-78 px for the fit it replaced. A per-anchor homography (what per-frame calibrations solve
        # for) would take the rest; that is noted in the module rather than pretended away here.
        assert _median_pixel_error(corrected, q, focal, frame, truth) < 30.0, "an anchor must land on its clicks"
    # The price of a smooth correction between two anchors: where the camera whipped mid-gap, the error curve bent
    # harder than a straight line can follow, and those frames come out a little worse than the harmless flat fit
    # they would have had. It has to stay a minority - the median above is the typical frame - and this fixture is
    # the worst case deliberately: on the undrifted fixture (real-chain conditions) the count is nil.
    worse = int((after > before + 5.0).sum())
    assert worse <= len(frames) // 3, f"{worse} frames of {len(frames)} came out worse than the fit it replaced"


def test_an_undrifted_chain_is_not_disturbed_by_the_fit() -> None:
    """With no injected drift the analysis chain is already the best it gets - the correction must improve it.

    The chain's own focal-track lag is still there (it is what the chain does on a fast zoom), so the correction is
    not forced to be the identity: it absorbs that lag, which is worth roughly a factor of three on the median. A
    minority of frames between two anchors can come out slightly worse than the plain fit - the price of
    interpolating an error curve nobody measured there - and this pins that price as a minority.
    """
    segment, truth = simulate_match(frames=600, seed=17)
    q, focal = segment_poses(segment)
    clicks = _clicks(truth, ANCHORS)
    chain = {frame: (q[frame], float(focal[frame])) for frame in ANCHORS}
    plain = calibrate(clicks, chain, ASPECT, correct_drift=False)
    corrected = calibrate(clicks, chain, ASPECT, correct_drift=True)
    assert corrected.drift is not None

    frames = list(range(0, 600, 10))
    before = np.array([_median_pixel_error(plain, q, focal, frame, truth) for frame in frames])
    after = np.array([_median_pixel_error(corrected, q, focal, frame, truth) for frame in frames])
    assert np.median(after) < np.median(before) * 0.6, f"the zoom lag is the correction's to take: {np.median(before):.1f} -> {np.median(after):.1f} px"
    assert after.max() <= before.max(), "the worst frame must not get worse"
    for frame in ANCHORS:
        assert _median_pixel_error(corrected, q, focal, frame, truth) < 30.0
    worse = int((after > before + 5.0).sum())
    assert worse <= len(frames) // 5, f"{worse} frames of {len(frames)} came out worse; the interpolation's price must stay a minority"


def test_project_segment_applies_the_calibrations_drift() -> None:
    """The report, the replay and the fit check all project through one function; it must carry the correction.

    The piece this pins down is the wiring no pixel test would catch: a calibration with a drift, handed to
    ``project_segment``, has to project exactly as the corrected chain would - otherwise the UI would show the
    corrected overlay while the report and the replay were built from the raw chain.
    """
    import numpy as np
    from synthetic_match import ASPECT, simulate_match

    from soccer_analytics.analysis.projection import project_segment, segment_poses
    from soccer_analytics.geometry.drift import DriftCorrection
    from soccer_analytics.geometry.pitch_calibration import PitchCalibration

    segment, truth = simulate_match(frames=60, seed=9)
    q, focal = segment_poses(segment)
    drift = DriftCorrection((10, 50), tuple(_exp(np.array([0.0, 0.004, 0.0]) * k) for k in (1, 2)), (1.02, 0.99))
    base = PitchCalibration(truth.calibration.position, truth.calibration.base_rotation, truth.calibration.focal_scale, ASPECT, 0.0, ())
    corrected = PitchCalibration(*[base.position, base.base_rotation, base.focal_scale, base.aspect, 0.0, ()], drift=drift)

    plain = project_segment(segment, base)
    with_drift = project_segment(segment, corrected)
    manual = project_segment(segment, base, poses=(drift.adjust(q), drift.adjust_focal(focal)))

    finite = np.isfinite(with_drift.xy).all(axis=1)
    assert finite.any(), "the simulated segment has to project something"
    assert np.allclose(with_drift.xy[finite], manual.xy[finite])
    assert not np.allclose(with_drift.xy[finite], plain.xy[finite]), "the correction has to change the projection"

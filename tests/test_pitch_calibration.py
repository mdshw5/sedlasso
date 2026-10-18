"""Pitch calibration against a known synthetic camera (exact projection + pixel noise)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from soccer_analytics.geometry.pitch_calibration import (
    FOCAL_SCALE_RANGE,
    CalibrationError,
    Landmark,
    PitchCalibration,
    calibrate,
    pitch_to_pixels,
    pixels_to_pitch,
)

ASPECT = 9 / 16
F_CHAIN = 0.82  # focal length reported by the camera-motion chain for every frame (no zoom here)

# A tripod 4 m up, 6 m beyond the touchline, roughly mid-pitch, looking across the pitch (+Y direction in the world).
TRUE_POSITION = np.array([40.0, -6.0, 4.0])
TRUE_FOCAL_SCALE = 1.07


def _look_rotation(heading: float, tilt_down: float) -> np.ndarray:
    """camera -> world rotation for a camera looking along ``heading`` (radians in the ground plane)."""
    z = np.array([np.cos(heading) * np.cos(tilt_down), np.sin(heading) * np.cos(tilt_down), -np.sin(tilt_down)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


R_BASE = _look_rotation(heading=np.radians(100), tilt_down=np.radians(12))  # frame 0 looks mostly +Y, slightly +X


def _q_for_pan(pan_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Chain orientation Q (frame rays -> reference rays) for a camera panned/tilted from the reference frame."""
    ry = cv2.Rodrigues(np.array([0.0, np.radians(pan_deg), 0.0]))[0]
    rx = cv2.Rodrigues(np.array([np.radians(tilt_deg), 0.0, 0.0]))[0]
    return ry @ rx


def _project(pitch_xy: np.ndarray, q: np.ndarray) -> np.ndarray:
    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    uv, front = pitch_to_pixels(cal, pitch_xy, q, F_CHAIN)
    return uv, front


# Standard 11-a-side landmarks (meters), origin at the near-left corner flag; pitch 100 x 64.
LANDMARKS_XY = {
    "corner_a": (0.0, 0.0),
    "corner_b": (100.0, 0.0),
    "corner_c": (100.0, 64.0),
    "corner_d": (0.0, 64.0),
    "halfway_near": (50.0, 0.0),
    "halfway_far": (50.0, 64.0),
    "center": (50.0, 32.0),
    "box_a": (16.5, 13.8),
    "box_b": (16.5, 50.2),
    "box_c": (83.5, 13.8),
    "box_d": (83.5, 50.2),
    "goal_post_a": (0.0, 32.0 - 3.66),
    "goal_post_b": (0.0, 32.0 + 3.66),
    "goal_post_c": (100.0, 32.0 - 3.66),
    "goal_post_d": (100.0, 32.0 + 3.66),
}


def _clicks(
    frames: dict[int, float], noise_px: float = 0.0, seed: int = 0, names: list[str] | None = None
) -> tuple[list[Landmark], dict]:
    """Click every landmark that falls inside each frame's field of view; returns landmarks and the chain dict.

    ``names`` restricts the clicks to a subset, which is how the "I cannot see the near corners" case is built.
    """
    rng = np.random.default_rng(seed)
    landmarks, chain = [], {}
    wanted = list(LANDMARKS_XY) if names is None else names
    for frame, pan in frames.items():
        q = _q_for_pan(pan)
        chain[frame] = (q, F_CHAIN)
        for name in wanted:
            x, y = LANDMARKS_XY[name]
            uv, front = _project(np.array([[x, y]]), q)
            if not front[0]:
                continue
            u, v = uv[0]
            if 0.0 <= u <= 1.0 and 0.0 <= v <= ASPECT:
                jitter = rng.normal(0.0, noise_px / 1920.0, 2)  # px noise at a 1920-wide frame
                landmarks.append(Landmark(frame, u + jitter[0], v + jitter[1], x, y, name))
    return landmarks, chain


def test_ground_intersection_round_trips_for_every_pan_angle() -> None:
    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    rng = np.random.default_rng(1)
    for pan in (-60, -30, 0, 30, 60):
        q = _q_for_pan(pan)
        pts = rng.uniform([5, 3], [95, 60], size=(40, 2))
        uv, front = pitch_to_pixels(cal, pts, q, F_CHAIN)
        inside = front & (uv[:, 0] > -0.5) & (uv[:, 0] < 1.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < 1.0)
        if inside.sum() == 0:
            continue
        back, valid = pixels_to_pitch(cal, uv[inside], q, F_CHAIN)
        assert valid.all()
        assert np.allclose(back, pts[inside], atol=1e-6)


def test_calibration_recovers_known_camera_from_two_pan_positions() -> None:
    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=0.0)
    assert len(landmarks) >= 8
    cal = calibrate(landmarks, chain, ASPECT)
    assert cal.rms_error_m < 0.05
    assert np.allclose(cal.position, TRUE_POSITION, atol=0.3)
    assert cal.focal_scale == pytest.approx(TRUE_FOCAL_SCALE, rel=0.02)
    assert np.allclose(cal.base_rotation, R_BASE, atol=0.02)


def test_a_known_camera_height_is_pinned_exactly_and_keeps_the_fit_healthy() -> None:
    """The dashboard pins the rig's known height (a fixed pole): the solve works in the remaining six parameters.

    Regression test for the degenerate-bounds failure: expressing the pin as equal lower/upper bounds made scipy
    refuse every start ("each lower bound must be strictly less than each upper bound"), so *every* pinned fit
    failed with "solver failed to start" and the dashboard reported "the clicks do not determine a camera".
    """
    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=0.0)
    cal = calibrate(landmarks, chain, ASPECT, fixed_height_m=4.0)
    assert cal.position[2] == pytest.approx(4.0)  # exact by construction, not merely recovered
    assert cal.rms_error_m < 0.05
    assert np.allclose(cal.position[:2], TRUE_POSITION[:2], atol=0.3)
    # The pinned coordinate must not read as a singular direction: the conditioning check runs on the free ones.
    assert not cal.ill_conditioned


def test_calibration_tolerates_realistic_click_noise() -> None:
    landmarks, chain = _clicks({0: 0.0, 1: 35.0, 2: -35.0}, noise_px=4.0, seed=5)
    cal = calibrate(landmarks, chain, ASPECT)
    # 4 px of click noise legitimately costs 1-5 m on landmarks 40-90 m away (see the sensitivity test), so the
    # residual floor is geometric. What must hold: the camera is recovered and the fit is no worse than that floor.
    assert cal.rms_error_m < 4.0
    assert np.linalg.norm(cal.position - TRUE_POSITION) < 3.0
    assert cal.focal_scale == pytest.approx(TRUE_FOCAL_SCALE, rel=0.12)
    # An independent point near the camera (where one pixel is worth little ground) is mapped to within ~1.5 m.
    probe = np.array([[45.0, 8.0], [60.0, 6.0]])
    for pan in (0.0, 35.0, -35.0):
        q = _q_for_pan(pan)
        uv, front = _project(probe, q)
        ok = front & (uv[:, 0] > 0) & (uv[:, 0] < 1) & (uv[:, 1] > 0) & (uv[:, 1] < ASPECT)
        if ok.any():
            xy, valid = pixels_to_pitch(cal, uv[ok], q, F_CHAIN)
            assert valid.all()
            assert np.linalg.norm(xy - probe[ok], axis=1).max() < 2.0


def test_far_ground_is_far_more_uncertain_than_near_ground() -> None:
    """One pixel of click error costs several times more ground at 80 m than at 35 m: downstream code must know."""
    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    q = _q_for_pan(0.0)
    e = 4.0 / 1920

    def ground_cost(xy: tuple[float, float]) -> float:
        uv, front = _project(np.array([xy]), q)
        assert front[0]
        a, _ = pixels_to_pitch(cal, uv, q, F_CHAIN)
        b, _ = pixels_to_pitch(cal, uv + [0.0, e], q, F_CHAIN)
        return float(np.linalg.norm(a - b))

    assert ground_cost((0.0, 64.0)) > 3 * ground_cost((50.0, 32.0))


def test_a_wide_pan_still_calibrates_where_a_planar_reference_would_fail() -> None:
    """Landmarks seen 85 degrees apart: a planar reference image cannot hold both, rays can."""
    landmarks, chain = _clicks({0: -42.0, 1: 43.0, 2: 0.0}, noise_px=2.0, seed=2)
    assert {lm.frame for lm in landmarks} >= {0, 1}
    cal = calibrate(landmarks, chain, ASPECT)
    # Position error here is geometric (distant landmarks, 2 px noise), not a solver weakness; see the sensitivity test.
    assert cal.rms_error_m < 2.0
    assert np.linalg.norm(cal.position - TRUE_POSITION) < 2.5


def test_a_gross_outlier_click_does_not_wreck_the_fit_and_is_flagged() -> None:
    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=1.0, seed=3)
    bad_index = 2
    victim = landmarks[bad_index]
    landmarks[bad_index] = Landmark(victim.frame, victim.u + 0.12, victim.v - 0.08, victim.pitch_x, victim.pitch_y, "typo")
    cal = calibrate(landmarks, chain, ASPECT)
    # The mistake is identified, excluded from the fit, and still reported with its (huge) residual for the UI.
    assert bad_index in cal.excluded
    assert int(np.argmax(cal.residuals_m)) == bad_index
    assert cal.residuals_m[bad_index] > 5 * np.median([r for i, r in enumerate(cal.residuals_m) if i != bad_index])
    assert cal.rms_error_m < 2.0  # the reported RMS describes the landmarks that were actually usedck stands out so the UI can point at it
    assert cal.residuals_m[bad_index] > 5 * np.median(cal.residuals_m)


def test_rejects_too_few_and_collinear_landmarks() -> None:
    landmarks, chain = _clicks({0: 0.0})
    with pytest.raises(CalibrationError, match="at least 4"):
        calibrate(landmarks[:3], chain, ASPECT)
    line = [Landmark(0, 0.2 + 0.1 * i, 0.3, 10.0 * i, 0.0, f"p{i}") for i in range(6)]
    with pytest.raises(CalibrationError, match="collinear"):
        calibrate(line, {0: chain[0]}, ASPECT)


def test_goal_posts_stand_in_for_corners_that_are_never_in_frame() -> None:
    """The realistic case: the near corners are not visible, so the goalposts carry that end of the pitch.

    Measured over the seeds below (4 px of click noise), this set is right to within ~2.5 m every time. It is the
    set the landmark instructions point at, so it is worth holding to that.
    """
    names = ["corner_c", "corner_d", "goal_post_a", "goal_post_b", "halfway_far", "center"]
    errors = []
    for seed in range(1, 5):
        landmarks, chain = _clicks({0: 0.0, 1: 35.0, 2: -35.0}, noise_px=4.0, seed=seed, names=names)
        assert len(landmarks) >= 6
        calibration = calibrate(landmarks, chain, ASPECT)
        errors.append(float(np.linalg.norm(calibration.position - TRUE_POSITION)))
    assert max(errors) < 5.0, f"goal-post set was off by {max(errors):.1f} m: {errors}"


def test_a_wrong_camera_is_never_reported_with_a_small_rms() -> None:
    """A wrong camera must never come back looking confident.

    Four landmarks that are all far away are genuinely not enough - with 4 px of click noise the answer can be tens
    of meters out. What must never happen is that such an answer comes back with neither a large rms nor a
    diagnostic: the deceptive solutions here are caught by the bound-pinned parameters or the near-singular residual
    Jacobian (the clicks barely constrain the camera), which ``suspect_fit_reason`` names for the UI.
    """
    from soccer_analytics.geometry.pitch_calibration import suspect_fit_reason

    names = ["corner_c", "corner_d", "goal_post_a", "goal_post_b"]
    saw_a_bad_fit = False
    for seed in range(1, 9):
        landmarks, chain = _clicks({0: 0.0, 1: 35.0, 2: -35.0}, noise_px=4.0, seed=seed, names=names)
        calibration = calibrate(landmarks, chain, ASPECT)
        error = float(np.linalg.norm(calibration.position - TRUE_POSITION))
        if error > 10.0:
            saw_a_bad_fit = True
            assert calibration.rms_error_m > 4.0 or suspect_fit_reason(calibration) is not None, (
                f"a {error:.0f} m error came back with an rms of {calibration.rms_error_m:.1f} m "
                "and no diagnostic to warn about it"
            )
    assert saw_a_bad_fit, "this set is expected to be unreliable; if it is not, the guidance can be relaxed"


def test_diagnosis_names_a_fit_that_is_not_a_solution() -> None:
    """A parameter parked on its bound means the answer is the corner of the search box, not a camera.

    This is the shape of a real failure: six clicks where four could not be reconciled, and the lens scaling ran to
    the top of its allowed range. Reporting "rms 35.86 m" alone leaves the user with nothing to act on.
    """
    from soccer_analytics.geometry.pitch_calibration import FitDiagnosis, diagnose_fit

    runaway = PitchCalibration(
        position=np.array([94.84, 37.03, 1.09]),
        base_rotation=R_BASE,
        focal_scale=FOCAL_SCALE_RANGE[1],  # exactly on the bound
        aspect=ASPECT,
        rms_error_m=35.86,
        residuals_m=(54.33, 1.92, 0.98, 56.96, 36.90, 12.36),
    )
    diagnosis = diagnose_fit(runaway)
    assert diagnosis.reason is not None and "lens scaling" in diagnosis.reason
    # Two clicks agree with each other; four do not, so no pitch fits the set.
    assert diagnosis.agreeing == (1, 2)
    assert diagnosis.disagreeing == (0, 3, 4, 5)
    assert diagnosis.too_few_agreeing

    # A healthy fit: every click within noise of the best, so nothing is flagged.
    sound = PitchCalibration(
        position=np.array([40.0, -6.0, 4.0]),
        base_rotation=R_BASE,
        focal_scale=1.07,
        aspect=ASPECT,
        rms_error_m=1.1,
        residuals_m=(1.4, 0.9, 1.2, 0.8, 1.5, 1.0),
    )
    clean = diagnose_fit(sound)
    assert clean.reason is None
    assert not clean.too_few_agreeing and clean.disagreeing == ()

    # One bad click among good ones: named, but the fit stands.
    one_bad = PitchCalibration(
        position=np.array([40.0, -6.0, 4.0]),
        base_rotation=R_BASE,
        focal_scale=1.07,
        aspect=ASPECT,
        rms_error_m=2.2,
        residuals_m=(0.9, 1.1, 18.0, 1.0, 0.8, 1.2),
    )
    mixed = diagnose_fit(one_bad)
    assert mixed.disagreeing == (2,)
    assert not mixed.too_few_agreeing
    assert mixed.reason is None
    assert isinstance(FitDiagnosis(), FitDiagnosis)


def test_six_clicks_can_now_shed_a_bad_one() -> None:
    """Rejection used to need more than six landmarks, so the common six-click set could never drop an outlier."""
    from soccer_analytics.geometry.pitch_calibration import MIN_LANDMARKS_AFTER_REJECTION

    assert MIN_LANDMARKS_AFTER_REJECTION == 5, "dropping one must leave enough landmarks to solve with"

    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=1.0, seed=2)
    six = landmarks[:6]
    victim = six[0]
    six[0] = Landmark(victim.frame, victim.u + 0.05, victim.v - 0.03, victim.pitch_x, victim.pitch_y, "misclick")
    calibration = calibrate(six, chain, ASPECT)
    assert len(calibration.excluded) == 1, "a six-click set should be able to flag its worst click"


def test_goal_line_clicks_cannot_anchor_a_correction() -> None:
    """Three landmarks can still be collinear, and then they say nothing about rotation about that line.

    This is the 9v9 case: the corner flag and both posts on one goal line were enough clicks to pass a count rule,
    but the goal box still projected 40-60 px wrong at 1920. The knot has to span two pitch dimensions.
    """
    from soccer_analytics.geometry.pitch_calibration import _drift_anchors

    collinear = [
        Landmark(10, 0.1, 0.1, 60.0, 0.0, "corner near-right"),
        Landmark(10, 0.2, 0.1, 60.0, 17.56, "goal post right-near"),
        Landmark(10, 0.3, 0.1, 60.0, 22.44, "goal post right-far"),
    ]
    assert _drift_anchors(collinear) == []

    spread = collinear + [Landmark(10, 0.4, 0.2, 36.4, 20.0, "center circle right")]
    assert len(_drift_anchors(spread)) == 4


def test_two_clicks_cannot_anchor_a_correction() -> None:
    """Two close landmarks (a pair of goalposts) fit four unknowns exactly, with no redundancy left to check it.

    On the 9v9 match those two-click goal anchors let the correction twist the goal box 40-60 px while still
    landing on the posts. A moment needs a third landmark before it may move the drift.
    """
    from soccer_analytics.geometry.pitch_calibration import MIN_CLICKS_PER_ANCHOR

    assert MIN_CLICKS_PER_ANCHOR == 3

    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=1.0, seed=6)
    first = [lm for lm in landmarks if lm.frame == 0]
    two = [lm for lm in landmarks if lm.frame == 1][:2]
    calibration = calibrate(first + two, chain, ASPECT)
    assert calibration.drift is None or 1 not in calibration.drift.frames


def test_a_lone_click_cannot_anchor_a_correction() -> None:
    """One click at its own moment gets no knot: two equations against four unknowns would be a private opinion.

    The whole-game archive is what taught this - single-click anchors were free in the two directions the click
    cannot see, and one walked 21 deg off its neighbors with the residual still small enough to look fine. The
    rule now: a moment clicked once is *judged* against the registration (its residual is reported, so the user
    knows to add a landmark there) but cannot move the correction.
    """
    landmarks, chain = _clicks({0: 0.0, 1: 35.0}, noise_px=1.0, seed=5)
    first = [lm for lm in landmarks if lm.frame == 0]
    assert len(first) == 4, "the fixture should give the first frame a registering set"
    victim = [lm for lm in landmarks if lm.frame == 1][0]
    lone = Landmark(victim.frame, victim.u + 0.04, victim.v - 0.03, victim.pitch_x, victim.pitch_y, "off")
    calibration = calibrate(first + [lone], chain, ASPECT)

    assert calibration.drift is None, "a correction needs two moments; only one of them carries two clicks"
    others = np.asarray(calibration.residuals_m[:-1])
    assert calibration.residuals_m[-1] > 5.0 * max(float(np.median(others)), 0.05), (
        "the lone click has to come back as the disagreement it is, not be rotated into place"
    )
    assert 1.0 < calibration.position[2] < 8.0, "and the pose stays a real camera"


def test_wrong_pitch_size_shows_up_in_the_camera_height_not_the_residual() -> None:
    """A wrong match format is absorbed by the camera position, which is not where anyone would look for it.

    Measured here and worth holding to: telling the solver the pitch is smaller or larger than it really is leaves
    the residuals small and moves the recovered height with it. So a good-looking residual is no evidence that the
    format is right, and the height is the only clue - which is what `format_scale_note` reports.
    """
    from soccer_analytics.geometry.pitch_calibration import format_scale_note

    landmarks, chain = _clicks({0: 0.0, 1: 35.0, 2: -35.0}, noise_px=2.0, seed=1)
    heights, rms = [], []
    for scale in (0.6, 1.0, 1.67):
        rescaled = [
            Landmark(lm.frame, lm.u, lm.v, lm.pitch_x * scale, lm.pitch_y * scale, lm.label) for lm in landmarks
        ]
        calibration = calibrate(rescaled, chain, ASPECT)
        heights.append(float(calibration.position[2]))
        rms.append(calibration.rms_error_m)
        assert abs(calibration.focal_scale - TRUE_FOCAL_SCALE) < 0.05, "the lens is not what absorbs the size"

    assert max(rms) < 2.5, f"the residual stays small on a wrong format: {rms}"
    assert heights[0] < heights[1] < heights[2], f"the height follows the assumed size: {heights}"
    assert format_scale_note(calibrate(landmarks, chain, ASPECT)) is None, "a correct format must not warn"
    # The 167% case lands outside the tripod range and is the one the note is there to catch.
    big = calibrate(
        [Landmark(lm.frame, lm.u, lm.v, lm.pitch_x * 1.67, lm.pitch_y * 1.67, lm.label) for lm in landmarks],
        chain,
        ASPECT,
    )
    note = format_scale_note(big)
    assert note is not None and "match format" in note


def test_landmarks_project_back_onto_the_pixel_they_were_clicked_at() -> None:
    """The fit check draws landmarks the user never clicked, so the two ends of the pipeline must agree.

    A click is normalized by frame width (what the solver consumes); the projection hands back pixels. Round-tripping
    a landmark through both is what catches those two conventions drifting apart.
    """
    from soccer_analytics.dashboard.pitch_clicks import landmark_table, project_landmarks

    calibration = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    table = landmark_table(100.0, 64.0)  # matches LANDMARKS_XY, the pitch this scene uses
    q = _q_for_pan(0.0)
    frame_width = 1920
    projected = project_landmarks(calibration, table, q, F_CHAIN, frame_width)
    assert projected, "nothing projected into the frame at all"

    for name, (px, py) in projected.items():
        clicked = Landmark(0, px / frame_width, py / frame_width, *table[name], name)
        xy, valid = pixels_to_pitch(calibration, np.array([[clicked.u, clicked.v]]), q, F_CHAIN)
        assert valid[0], f"{name} mapped to a ray that misses the ground"
        assert xy[0] == pytest.approx(table[name], abs=0.05), f"{name} came back as {xy[0]}"


def test_rejects_landmark_in_a_frame_without_camera_state() -> None:
    landmarks, chain = _clicks({0: 0.0})
    with pytest.raises(CalibrationError, match="no camera state"):
        calibrate(landmarks, {}, ASPECT)


def test_sky_ray_is_not_mapped_to_the_ground() -> None:
    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    # q maps frame rays to reference rays. The reference camera looks 12 deg down; R_x(-30) raises the optical
    # axis to +18 deg elevation (sky), while R_x(+30) lowers it to -42 deg (ground). The sign follows from the
    # handedness of ``_look_rotation`` above.
    q_up = cv2.Rodrigues(np.array([np.radians(-30.0), 0.0, 0.0]))[0]
    xy, valid = pixels_to_pitch(cal, np.array([[0.5, 0.5 * ASPECT]]), q_up, F_CHAIN)
    assert not valid[0] and np.isnan(xy[0]).all()


def test_horizon_is_the_exact_boundary_between_ground_and_sky() -> None:
    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    # Tilt so the optical axis sits at +-0.5 degrees of elevation: just below the horizon hits ground, just above doesn't.
    for rotation_deg, expect_ground in ((-11.5, True), (-12.5, False)):
        q = cv2.Rodrigues(np.array([np.radians(rotation_deg), 0.0, 0.0]))[0]
        _, valid = pixels_to_pitch(cal, np.array([[0.5, 0.5 * ASPECT]]), q, F_CHAIN)
        assert bool(valid[0]) is expect_ground


def test_calibration_round_trips_through_json() -> None:
    landmarks, chain = _clicks({0: 0.0, 1: 35.0})
    cal = calibrate(landmarks, chain, ASPECT)
    again = PitchCalibration.from_json(cal.to_json())
    assert np.allclose(again.position, cal.position)
    assert np.allclose(again.base_rotation, cal.base_rotation)
    assert again.focal_scale == cal.focal_scale
    assert again.residuals_m == cal.residuals_m


def test_calibration_records_its_pose_source() -> None:
    landmarks, chain = _clicks({0: 0.0, 1: 35.0})
    assert calibrate(landmarks, chain, ASPECT).pose_source == "chain"
    logged = calibrate(landmarks, chain, ASPECT, pose_source="log")
    assert logged.pose_source == "log"
    # The source survives a round trip, so a saved calibration can be checked against the segment's current motion.
    assert PitchCalibration.from_json(logged.to_json()).pose_source == "log"
    # An older calibration with no field reads as the chain, which is what it was fitted against.
    payload = logged.to_json()
    del payload["pose_source"]
    assert PitchCalibration.from_json(payload).pose_source == "chain"


# --------------------------------------------------------------------------------------------------------------
# Re-orienting a known camera for a later segment of the same match
# --------------------------------------------------------------------------------------------------------------

from soccer_analytics.geometry.pitch_calibration import recalibrate_orientation  # noqa: E402


def _second_segment_truth(heading_deg: float, tilt_deg: float = 12.0) -> np.ndarray:
    """A later segment starts the gimbal pointing somewhere else entirely (same tripod, same lens)."""
    return _look_rotation(np.radians(heading_deg), np.radians(tilt_deg))


def _clicks_for(r_base: np.ndarray, count: int, pan: float = 0.0, noise_px: float = 0.0, seed: int = 0):
    """Click ``count`` landmarks that are actually in view for this orientation (spread apart, nearest-first)."""
    rng = np.random.default_rng(seed)
    cal = PitchCalibration(TRUE_POSITION, r_base, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    q = _q_for_pan(pan)
    visible = []
    for name, (x, y) in LANDMARKS_XY.items():
        uv, front = pitch_to_pixels(cal, np.array([[x, y]]), q, F_CHAIN)
        if front[0] and 0.05 <= uv[0, 0] <= 0.95 and 0.05 <= uv[0, 1] <= ASPECT - 0.03:
            visible.append((name, x, y, uv[0]))
    assert len(visible) >= count, f"only {len(visible)} landmarks visible"
    visible.sort(key=lambda item: item[3][0])  # spread them across the image: take evenly spaced by u
    picks = [visible[round(i * (len(visible) - 1) / max(count - 1, 1))] for i in range(count)]
    landmarks = []
    for name, x, y, uv in picks:
        jitter = rng.normal(0.0, noise_px / 1920.0, 2)
        landmarks.append(Landmark(0, uv[0] + jitter[0], uv[1] + jitter[1], x, y, name))
    return landmarks, {0: (q, F_CHAIN)}


def test_two_clicks_reorient_a_known_camera_that_now_points_elsewhere() -> None:
    base = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    for heading in (60.0, 100.0, 130.0):
        truth = _second_segment_truth(heading)
        landmarks, chain = _clicks_for(truth, 2, seed=1)
        recal = recalibrate_orientation(base, landmarks, chain)
        assert np.allclose(recal.base_rotation, truth, atol=2e-3), f"heading {heading}"
        assert recal.rms_error_m < 0.05
        assert np.allclose(recal.position, TRUE_POSITION) and recal.focal_scale == TRUE_FOCAL_SCALE


def test_reorientation_is_usable_for_mapping_independent_points() -> None:
    base = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    truth = _second_segment_truth(75.0)
    landmarks, chain = _clicks_for(truth, 3, noise_px=2.0, seed=4)
    recal = recalibrate_orientation(base, landmarks, chain)
    probe = np.array([[45.0, 12.0]])
    cal_true = PitchCalibration(TRUE_POSITION, truth, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    uv, front = pitch_to_pixels(cal_true, probe, chain[0][0], F_CHAIN)
    assert front[0]
    mapped, valid = pixels_to_pitch(recal, uv, chain[0][0], F_CHAIN)
    assert valid[0] and np.linalg.norm(mapped[0] - probe[0]) < 1.5


def test_reorientation_input_checks() -> None:
    base = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    landmarks, chain = _clicks_for(R_BASE, 1)
    with pytest.raises(CalibrationError, match="at least 2"):
        recalibrate_orientation(base, landmarks, chain)
    two, _ = _clicks_for(R_BASE, 2)
    with pytest.raises(CalibrationError, match="no camera state"):
        recalibrate_orientation(base, two, {})

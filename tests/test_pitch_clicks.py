"""Pitch-landmark clicking: the crop math and the click -> frame mapping.

This is the part of registration that is easy to get subtly wrong and hard to notice: a coordinate that is off by
the canvas scale still produces a plausible-looking calibration, just a worse one.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.dashboard.pitch_clicks import (
    LANDMARK_HELP,
    ClickResult,
    actual_center,
    canvas_scale,
    canvas_to_frame,
    clamp01,
    frame_change,
    frame_points,
    frame_to_canvas,
    inside_crop,
    landmark_help,
    landmark_table,
    landmarks_from_points,
    marker_feature,
    merge_clicked,
    order_clicks,
    overlay_homographies,
    parse_result,
    pitch_landmark_order,
    pitch_marking_polylines,
    pitch_overlay,
    point_center,
    points_in_crop,
    projected_landmarks,
    repeated_labels_within_a_frame,
    restored_points,
    split_duplicate_clicks,
    zoom_box,
)
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels
from soccer_analytics.geometry.pitch_markings import markings_for

FRAME_W, FRAME_H = 3840, 2160


def click_in_crop(box: tuple[int, int, int, int], scale: float, fx: float, fy: float) -> tuple[float, float]:
    """Canvas coords for a click at a fraction ``(fx, fy)`` of the crop.

    Crop fractions are used in this file on purpose. There are two y conventions in play - the zoom sliders take a
    fraction of the frame *height*, while a click's ``v`` is normalized by the frame *width* (that is what the camera
    model uses) - and mixing them up silently puts the click outside the crop.
    """
    return fx * box[2] * scale, fy * box[3] * scale


def test_zoom_box_stays_inside_the_frame() -> None:
    # Centered crop.
    assert zoom_box(FRAME_W, FRAME_H, 4.0, 0.5, 0.5) == (1440, 810, 960, 540)
    # Pushed into each corner: the box must be clamped, never partly outside the frame.
    for center_x, center_y in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
        x0, y0, w, h = zoom_box(FRAME_W, FRAME_H, 8.0, center_x, center_y)
        assert 0 <= x0 and x0 + w <= FRAME_W, (x0, w)
        assert 0 <= y0 and y0 + h <= FRAME_H, (y0, h)


def test_zoom_box_keeps_the_frame_aspect() -> None:
    _x0, _y0, w, h = zoom_box(FRAME_W, FRAME_H, 3.0, 0.5, 0.5)
    assert abs(w / h - FRAME_W / FRAME_H) < 0.02


def test_zoom_box_never_collapses_to_nothing() -> None:
    _x0, _y0, w, h = zoom_box(320, 180, 1000.0, 0.5, 0.5)
    assert w >= 64 and h >= 64


def test_canvas_scale_is_one_for_small_crops() -> None:
    # A crop at or below the transfer width is sent at native size: that is what makes a click accurate.
    assert canvas_scale(480) == 1.0
    assert canvas_scale(1600) == 1.0
    assert canvas_scale(3840) == pytest.approx(1600 / 3840)


def test_canvas_and_frame_coordinates_round_trip() -> None:
    """A click must map to the frame position it was made on, in both directions and at every zoom."""
    for zoom in (1, 2, 4, 8, 16):
        box = zoom_box(FRAME_W, FRAME_H, float(zoom), 0.42, 0.61)
        scale = canvas_scale(box[2])
        # A point in the middle of the crop.
        u_true = (box[0] + box[2] * 0.4) / FRAME_W
        v_true = (box[1] + box[3] * 0.7) / FRAME_W
        x, y = frame_to_canvas(u_true, v_true, scale, box, FRAME_W)
        u, v = canvas_to_frame(x, y, scale, box, FRAME_W)
        assert u == pytest.approx(u_true, abs=1e-9)
        assert v == pytest.approx(v_true, abs=1e-9)


def test_one_canvas_pixel_is_a_native_pixel_when_zoomed_in() -> None:
    """The whole point of zooming: at zoom 8 a screen pixel is a frame pixel, not 2.4 of them."""
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    assert scale == 1.0
    x, y = frame_to_canvas(0.5, 0.5, scale, box, FRAME_W)
    u2, _v2 = canvas_to_frame(x + 1.0, y, scale, box, FRAME_W)
    assert (u2 - 0.5) * FRAME_W == pytest.approx(1.0)

    # At full-frame zoom the same single pixel is worth several frame pixels.
    wide = zoom_box(FRAME_W, FRAME_H, 1.0, 0.5, 0.5)
    wide_scale = canvas_scale(wide[2])
    xw, yw = frame_to_canvas(0.5, 0.5, wide_scale, wide, FRAME_W)
    uw, _vw = canvas_to_frame(xw + 1.0, yw, wide_scale, wide, FRAME_W)
    assert (uw - 0.5) * FRAME_W == pytest.approx(1 / wide_scale, rel=1e-6)
    assert (uw - 0.5) * FRAME_W > 2.0


def test_clicks_survive_a_change_of_zoom() -> None:
    """Clicking one corner at zoom 8, then another elsewhere at zoom 2, must keep both."""
    stored: list[dict] = []
    next_pid = 0

    corner = zoom_box(FRAME_W, FRAME_H, 8.0, 0.2, 0.2)
    scale = canvas_scale(corner[2])
    click = click_in_crop(corner, scale, 0.5, 0.5)
    stored, next_pid = merge_clicked(stored, [click], 0, corner, scale, FRAME_W, next_pid)
    stored, next_pid = merge_clicked(stored, [click], 0, corner, scale, FRAME_W, next_pid)
    assert len(stored) == 1, "pressing Apply twice must not duplicate the click"

    away = zoom_box(FRAME_W, FRAME_H, 2.0, 0.8, 0.8)
    away_scale = canvas_scale(away[2])
    stored, next_pid = merge_clicked(stored, [click_in_crop(away, away_scale, 0.5, 0.5)], 0, away, away_scale, FRAME_W, next_pid)
    assert len(stored) == 2, "the first click was wiped by working elsewhere"
    assert len({p["pid"] for p in stored}) == 2

    # Apply with nothing on screen clears that crop, and only that crop.
    kept, next_pid = merge_clicked(stored, [], 0, away, away_scale, FRAME_W, next_pid)
    assert len(kept) == 1
    native_x = kept[0]["u"] * FRAME_W
    assert corner[0] <= native_x <= corner[0] + corner[2], "the surviving click should be the zoomed-in one"


def test_clicks_on_other_frames_are_untouched() -> None:
    """Corners are often not all visible in one frame, so clicks from different frames have to coexist."""
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    click = click_in_crop(box, scale, 0.5, 0.5)
    stored, next_pid = merge_clicked([], [click], 0, box, scale, FRAME_W, 0)
    stored, next_pid = merge_clicked(stored, [click], 120, box, scale, FRAME_W, next_pid)
    assert [p["frame"] for p in order_clicks(stored)] == [0, 120]

    stored, next_pid = merge_clicked(stored, [], 120, box, scale, FRAME_W, next_pid)
    assert [p["frame"] for p in stored] == [0], "clearing frame 120 removed a click from frame 0"


def test_points_in_crop_only_returns_visible_ones() -> None:
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    inside = {"pid": 0, "frame": 0, "u": 0.5, "v": 0.5 * FRAME_H / FRAME_W}
    outside = {"pid": 1, "frame": 0, "u": 0.1, "v": 0.5 * FRAME_H / FRAME_W}
    other_frame = {"pid": 2, "frame": 7, "u": 0.5, "v": 0.5 * FRAME_H / FRAME_W}
    features = points_in_crop([inside, outside, other_frame], 0, box, scale, FRAME_W)
    assert len(features) == 1
    x, y = features[0]["point"]
    assert x == pytest.approx(box[2] / 2, abs=1.0) and y == pytest.approx(box[3] / 2, abs=1.0)
    # The id is what lets the component keep a *dragged* marker's position through an unrelated rerun.
    assert features[0]["id"] == "click:0"


def test_a_click_can_carry_the_landmark_it_was_tagged_with() -> None:
    """The popover names a landmark at the click; that tag becomes the point's label."""
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    x, y = click_in_crop(box, scale, 0.5, 0.5)

    stored, next_pid = merge_clicked([], [(x, y, "corner far-left")], 0, box, scale, FRAME_W, 0)
    assert stored[0]["label"] == "corner far-left"

    # Re-committing the same crop must not strip the tag off the point.
    stored, next_pid = merge_clicked(stored, [(x, y, "")], 0, box, scale, FRAME_W, next_pid)
    assert stored[0]["label"] == "corner far-left"
    assert stored[0]["pid"] == 0, "the identity the label hangs off has to survive too"

    # A fresh pick on the same click replaces it.
    stored, next_pid = merge_clicked(stored, [(x, y, "center spot")], 0, box, scale, FRAME_W, next_pid)
    assert stored[0]["label"] == "center spot"

    # An untagged click carries no label at all, so the dropdown opens on the first landmark.
    stored, next_pid = merge_clicked([], [(x, y)], 0, box, scale, FRAME_W, 0)
    assert "label" not in stored[0]


def test_points_in_crop_carries_the_landmark_tag_back_into_the_component() -> None:
    """The tag is handed back with the click, so re-committing a crop cannot lose it."""
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    tagged = {"pid": 0, "frame": 0, "u": 0.5, "v": 0.5 * FRAME_H / FRAME_W, "label": "goal post left-near"}
    untagged = {"pid": 1, "frame": 0, "u": 0.5, "v": 0.5 * FRAME_H / FRAME_W}
    features = points_in_crop([tagged, untagged], 0, box, scale, FRAME_W)
    assert features[0].get("label") == "goal post left-near"
    assert "label" not in features[1]


def test_landmarks_carry_the_frame_they_were_clicked_on() -> None:
    """The solver takes each landmark with its own camera pose, which is what allows multi-frame clicking."""
    table = landmark_table(60.0, 40.0)
    labeled = [
        ({"pid": 0, "frame": 3, "u": 0.10, "v": 0.20}, "corner near-left"),
        ({"pid": 1, "frame": 90, "u": 0.30, "v": 0.25}, "corner far-right"),
    ]
    landmarks = landmarks_from_points(labeled, table)
    assert [lm.frame for lm in landmarks] == [3, 90]
    assert (landmarks[0].pitch_x, landmarks[0].pitch_y) == (0.0, 0.0)
    assert (landmarks[1].pitch_x, landmarks[1].pitch_y) == (60.0, 40.0)
    assert landmarks[1].label == "corner far-right"
    # Pixels are normalized by frame width on both axes, which is the convention the solver expects.
    assert landmarks[0].u == 0.10 and landmarks[0].v == 0.20


def test_repeated_labels_within_a_frame_flags_only_same_frame_duplicates() -> None:
    """Re-clicking a landmark on a later frame is the drift-anchoring workflow; twice on one frame is the mistake."""
    pairs = [
        (299, "corner near-left"),
        (11184, "corner near-left"),
        (11184, "center spot"),
        (11184, "corner near-left"),
    ]
    assert repeated_labels_within_a_frame(pairs) == ["corner near-left"]
    assert repeated_labels_within_a_frame([(299, "corner near-left"), (1040, "corner near-left")]) == []
    assert repeated_labels_within_a_frame([]) == []


def test_split_duplicate_clicks_keeps_the_newest_of_each_same_frame_pair() -> None:
    """The newest click is the user's latest word: a re-click, or a re-label that collides, replaces the old one."""
    entries = [
        (5, "corner near-left", 3),
        (5, "corner near-left", 7),  # the re-click that supersedes pid 3
        (5, "center spot", 4),  # a different landmark on the same frame is not a duplicate
        (90, "corner near-left", 5),  # a later frame is the drift-anchoring workflow, not a duplicate
    ]
    kept, dropped = split_duplicate_clicks(entries)
    assert kept == [(5, "corner near-left", 7), (5, "center spot", 4), (90, "corner near-left", 5)]
    assert dropped == [(5, "corner near-left", 3)]
    # Nothing to do on an empty or duplicate-free set.
    assert split_duplicate_clicks([]) == ([], [])
    assert split_duplicate_clicks([(1, "center spot", 0)]) == ([(1, "center spot", 0)], [])


def test_projected_landmarks_put_each_landmark_where_the_fit_projects_it() -> None:
    """Placed markers come from the calibration itself, so dragging one onto the marking measures the error."""
    from test_pitch_calibration import (
        ASPECT,
        F_CHAIN,
        LANDMARKS_XY,
        R_BASE,
        TRUE_FOCAL_SCALE,
        TRUE_POSITION,
        _q_for_pan,
    )

    cal = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    q = _q_for_pan(0.0)
    table = {"mark": LANDMARKS_XY["center"], "behind": (40.0, -20.0)}
    spots = projected_landmarks(cal, ["mark", "behind"], table, q, F_CHAIN)
    assert [spot["label"] for spot in spots] == ["mark"], "a landmark behind the camera has no image position"
    uv, front = pitch_to_pixels(cal, np.array([table["mark"]]), q, F_CHAIN)
    assert front[0]
    assert spots[0]["u"] == pytest.approx(float(uv[0, 0]), abs=1e-9)
    assert spots[0]["v"] == pytest.approx(float(uv[0, 1]), abs=1e-9)


def test_marker_features_stay_inside_the_crop_and_carry_their_label() -> None:
    """A placed marker must be a draggable point of the crop canvas, named, and refused when out of view."""
    u, v = 0.5, 0.25
    box = zoom_box(FRAME_W, FRAME_H, 8.0, *point_center(u, v, (FRAME_W, FRAME_H)))
    scale = canvas_scale(box[2])
    assert inside_crop(u, v, box, scale, FRAME_W)
    feature = marker_feature("center spot", u, v, box, scale, FRAME_W)
    assert feature["type"] == "point" and feature["label"] == "center spot"
    assert feature["id"] == "seed:center spot"
    assert feature["point"] == list(frame_to_canvas(u, v, scale, box, FRAME_W))
    x, y = feature["point"]
    assert 0 <= x <= box[2] * scale and 0 <= y <= box[3] * scale

    away = zoom_box(FRAME_W, FRAME_H, 8.0, 0.05, 0.95)
    assert not inside_crop(u, v, away, canvas_scale(away[2]), FRAME_W)


def test_saved_clicks_come_back_after_a_refresh_only_on_their_own_segment() -> None:
    """The clicks saved with a calibration are what a new browser session restores its editor from.

    Frames are indices into the segment that was clicked on, so a set from another window must be refused - putting
    a whole-game click at frame 11,184 of a five-minute clip would be a measurement of nothing.
    """
    saved = [
        {"frame": 299, "u": 0.2, "v": 0.1, "label": "goal post right-far"},
        {"frame": 11184, "u": 0.6, "v": 0.3, "label": "corner far-left"},
    ]
    points = restored_points(saved, 21552)
    assert points is not None
    assert [p["frame"] for p in points] == [299, 11184]
    assert [p["pid"] for p in points] == [0, 1]
    assert points[0]["label"] == "goal post right-far"
    assert restored_points(saved, 1500) is None, "clicks from another segment are not this segment's clicks"
    assert restored_points([], 10) is None
    assert restored_points([{"frame": 1, "u": 0.5}], 10) is None, "a malformed set is refused whole"


def test_every_landmark_has_help_text_and_a_pitch_position() -> None:
    table = landmark_table(100.0, 64.0)
    assert set(table) == set(LANDMARK_HELP), "a landmark without instructions is not clickable in practice"
    assert pitch_landmark_order(100.0, 64.0) == list(table)
    assert all(isinstance(help_text, str) and help_text for help_text in LANDMARK_HELP.values())
    # The corners must actually be the corners of the given pitch.
    assert table["corner near-left"] == (0.0, 0.0)
    assert table["corner far-right"] == (100.0, 64.0)
    assert table["center spot"] == (50.0, 32.0)
    assert np.isfinite(list(table.values())).all()


def test_goal_posts_sit_on_their_own_goal_line_either_side_of_the_goal_center() -> None:
    """The posts are the stand-ins for corners that are out of frame, so they must be where the laws put them.

    They are only useful because a post is a hard edge that can be clicked to a pixel and its real position is fixed
    by the laws of the game - if a post drifted off the goal line, or the pair were not the format's goal width
    apart, every fit built on them would be skewed. A 9v9 goal is 16 ft wide, not the 11-a-side 8 yards.
    """
    table = landmark_table(60.0, 40.0, format_name="9v9")
    half_goal = 16 * 0.3048 / 2
    assert table["goal post left-near"] == pytest.approx((0.0, 20.0 - half_goal))
    assert table["goal post left-far"] == pytest.approx((0.0, 20.0 + half_goal))
    assert table["goal post right-near"] == pytest.approx((60.0, 20.0 - half_goal))
    assert table["goal post right-far"] == pytest.approx((60.0, 20.0 + half_goal))
    for near, far, corner_a, corner_b in (
        ("goal post left-near", "goal post left-far", "corner near-left", "corner far-left"),
        ("goal post right-near", "goal post right-far", "corner near-right", "corner far-right"),
    ):
        # On the goal line itself, not in front of it, and straddling the goal center.
        assert table[near][0] == table[far][0] == table[corner_a][0] == table[corner_b][0]
        assert table[near][1] < table[far][1]
        assert table[far][1] - table[near][1] == pytest.approx(16 * 0.3048), "the posts are 16 ft apart"
        assert (table[near][1] + table[far][1]) / 2 == pytest.approx((table[corner_a][1] + table[corner_b][1]) / 2)

    # And the whole set must not be collinear, since a straight line cannot fix a camera.
    points = np.array(list(table.values()))
    assert np.linalg.matrix_rank(points - points.mean(0), tol=0.5) == 2
    assert len({label for label in table}) == len(table), "duplicate landmark names would break the dropdowns"


def test_the_boxes_are_not_clickable() -> None:
    """The boxes are the markings the clicks cannot place accurately, so they must not be offered.

    The goal area is small and lost against the netting and the goal frame; a penalty-area corner is a bare
    junction of two lines with nothing to focus on. A click a meter out is a meter of error in the fit, so the
    posts carry that end of the pitch instead and the penalty spots pin the box's depth.
    """
    table = landmark_table(100.0, 64.0)
    assert not [name for name in table if "box" in name]
    assert not [name for name in LANDMARK_HELP if "box" in name]


def test_spot_and_circle_landmarks_sit_on_the_standard_markings() -> None:
    """The extra landmarks are only useful if they are where the laws of the game put them.

    The penalty spots and center-circle cardinals are the markings that are usually visible when the corners are
    not, so a click on any of them is a measurement of the pitch - but only if the pitch position is right. These
    are the 11-a-side dimensions; the 9v9 template is checked separately below.
    """
    length_m, width_m = 100.0, 64.0
    table = landmark_table(length_m, width_m, format_name="11v11")
    half_width = width_m / 2

    # Penalty spots: 11 m out from the goal line, on the goal center line.
    assert table["penalty spot left"] == (11.0, half_width)
    assert table["penalty spot right"] == (length_m - 11.0, half_width)

    # Center-circle cardinals: 9.15 m from the center spot, on the halfway and center lines.
    assert table["center circle near"] == (length_m / 2, half_width - 9.15)
    assert table["center circle far"] == (length_m / 2, half_width + 9.15)
    assert table["center circle left"] == (length_m / 2 - 9.15, half_width)
    assert table["center circle right"] == (length_m / 2 + 9.15, half_width)


def test_nine_a_side_markings_match_the_fa_guide_and_clear_the_center_circle() -> None:
    """The 9v9 template must use the FA's 7-yard circle and 13-yard penalty area, not the 11-a-side values.

    Regression test for the reported failure: with the 11-a-side markings, a 60 x 40 m 9v9 pitch put the penalty
    area's front edge 16.5 m from the goal line and the centre circle's near edge 20.85 m out - only 4.35 m apart,
    so the box appeared to swallow the circle. The FA 9v9 box is 13 yards deep and the circle is 7 yards in radius,
    leaving more than 11 m between them.
    """
    markings = markings_for("9v9")
    yard = 0.9144
    assert markings.penalty_area_width_m == pytest.approx(32 * yard)
    assert markings.penalty_area_depth_m == pytest.approx(13 * yard)
    assert markings.goal_area_width_m == pytest.approx(14 * yard)
    assert markings.goal_area_depth_m == pytest.approx(4 * yard)
    assert markings.penalty_spot_distance_m == pytest.approx(9 * yard)
    assert markings.center_circle_radius_m == pytest.approx(7 * yard)
    assert markings.goal_width_m == pytest.approx(16 * 0.3048)

    length_m, width_m = 60.0, 40.0
    table = landmark_table(length_m, width_m, format_name="9v9")
    half_width = width_m / 2
    assert table["penalty spot left"] == pytest.approx((9 * yard, half_width))
    assert table["center circle left"] == pytest.approx((length_m / 2 - 7 * yard, half_width))
    assert table["center circle near"] == pytest.approx((length_m / 2, half_width - 7 * yard))

    # The failure this guards: the penalty box front edge and the centre circle near edge must not meet.
    box_front = markings.penalty_area_depth_m
    circle_back = length_m / 2 - markings.center_circle_radius_m
    assert circle_back - box_front > 10.0, "the 9v9 penalty area is too close to the centre circle"

    # The drawn markings must use the same dimensions, not just the landmark table.
    lines = pitch_marking_polylines(length_m, width_m, format_name="9v9")
    penalty_boxes = [line for line in lines if len(line) == 4 and np.isclose(np.ptp(line[:, 1]), 32 * yard)]
    assert len(penalty_boxes) == 2, "one penalty area at each end"
    assert all(np.isclose(np.ptp(line[:, 0]), 13 * yard) for line in penalty_boxes)
    circles = [
        line
        for line in lines
        if np.allclose(np.linalg.norm(line - np.array([length_m / 2, half_width]), axis=1), 7 * yard)
    ]
    assert len(circles) == 1, "exactly one 7-yard centre circle"


def test_mini_soccer_markings_omit_the_goal_area_and_center_circle() -> None:
    """A format that does not have a marking must not offer a click for it or draw a degenerate one."""
    for format_name, length_m, width_m in (("5v5", 40.0, 25.0), ("7v7", 50.0, 35.0)):
        markings = markings_for(format_name)
        assert not markings.has_goal_area
        assert not markings.has_center_circle
        assert not markings.has_penalty_arc

        table = landmark_table(length_m, width_m, format_name=format_name)
        assert "center spot" not in table
        assert not [name for name in table if name.startswith("center circle")]

        # Outline + halfway + one penalty area per end + four corner arcs. No goal area, circle or penalty arc.
        lines = pitch_marking_polylines(length_m, width_m, format_name=format_name)
        assert len(lines) == 8


def test_landmark_help_quotes_the_selected_formats_distances() -> None:
    """The click instructions must not tell a 9v9 user to find an 11 m spot or a 9.15 m circle."""
    help_9v9 = landmark_help("9v9")
    assert "8.23 m" in help_9v9["penalty spot left"]
    assert "6.40 m center circle" in help_9v9["center circle near"]
    assert "4.88 m" in help_9v9["goal post left-near"]
    assert "9.15 m" not in help_9v9["center circle near"]


def test_overlay_homographies_reproduce_the_projection() -> None:
    """The browser draws the overlay from these samples, so each must be the calibration's own projection.

    The check is against a point the fit never saw (the far corner, behind the camera in this scene): the ground
    plane's projection is a homography, so H fitted on the visible markings must map the hidden ones too - that is
    the whole reason the browser can draw the overlay on frames Python never renders.
    """
    from test_pitch_calibration import (
        ASPECT,
        F_CHAIN,
        R_BASE,
        TRUE_FOCAL_SCALE,
        TRUE_POSITION,
        _q_for_pan,
    )

    calibration = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    frame_count = 301
    q = np.stack([_q_for_pan(pan) for pan in np.linspace(-40, 40, frame_count)])
    focal = np.full(frame_count, F_CHAIN)
    samples = overlay_homographies(calibration, q, focal, 100.0, 64.0, frame_count, samples=7)
    assert len(samples) == 7
    assert [s["frame"] for s in samples] == sorted(s["frame"] for s in samples)
    assert all(len(s["h"]) == 9 for s in samples)

    for sample in samples:
        matrix = np.array(sample["h"]).reshape(3, 3)
        for probe_xy in ([30.0, 20.0], [0.0, 64.0], [100.0, 0.0]):
            projected = matrix @ np.array([*probe_xy, 1.0])
            uv_h = projected[:2] / projected[2]
            uv_p, front = pitch_to_pixels(
                calibration,
                np.array([probe_xy]),
                q[sample["frame"]],
                F_CHAIN,
            )
            if not front[0]:
                # Behind the camera the direct projection is undefined; the homography still maps the ground
                # plane (that is what lets the browser draw the far side), but there is nothing to compare.
                continue
            assert np.allclose(uv_h, uv_p[0], atol=1e-9), f"{probe_xy} at frame {sample['frame']}"


def test_overlay_homographies_skip_frames_without_camera_state() -> None:
    """A chain that does not cover every frame must not be indexed past its end."""
    from test_pitch_calibration import ASPECT, F_CHAIN, R_BASE, TRUE_FOCAL_SCALE, TRUE_POSITION, _q_for_pan

    calibration = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    q = np.stack([_q_for_pan(0.0)] * 3)
    focal = np.full(3, F_CHAIN)
    assert overlay_homographies(calibration, q, focal, 100.0, 64.0, 1000) != []
    assert overlay_homographies(calibration, q, focal, 100.0, 64.0, 0) == []


def test_pitch_marking_polylines_cover_the_standard_markings() -> None:
    """The overlay draws the whole pitch, not just the outline, so a fit can be checked against the visible markings.

    The penalty arc is the part of the 9.15 m circle around the penalty spot that lies *outside* the penalty box -
    the detail that makes it worth drawing rather than a full circle.
    """
    length_m, width_m = 100.0, 64.0
    lines = pitch_marking_polylines(length_m, width_m)
    assert len(lines) >= 10, "touchlines, halfway, two boxes each end, center circle, two arcs, four corners"
    for line in lines:
        assert line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2
        assert np.isfinite(line).all()

    # The penalty arc must stay outside the penalty box: its x never comes closer to the goal line than 16.5 m.
    arc_half_height = 9.15 * np.sin(np.arccos(5.5 / 9.15))
    arcs = [line for line in lines if np.isclose(np.ptp(line[:, 1]), 2 * arc_half_height, atol=0.5)]
    assert arcs, "the penalty arcs should be present"
    for arc in arcs:
        assert arc[:, 0].min() >= 16.5 - 1e-6 or arc[:, 0].max() <= length_m - 16.5 + 1e-6

    # The center circle is a closed loop of radius 9.15 m around the center spot.
    center = np.array([length_m / 2, width_m / 2])
    circles = [line for line in lines if np.allclose(np.linalg.norm(line - center, axis=1), 9.15, atol=1e-6)]
    assert len(circles) == 1, "exactly one center circle"
    assert len(circles[0]) >= 16, "a circle needs enough points to look round under perspective"


def test_pitch_overlay_draws_the_markings_onto_the_frame() -> None:
    """The overlay is the only visual check of a registration, so it must actually paint the markings."""
    from test_pitch_calibration import ASPECT, F_CHAIN, R_BASE, TRUE_FOCAL_SCALE, TRUE_POSITION, _q_for_pan

    calibration = PitchCalibration(TRUE_POSITION, R_BASE, TRUE_FOCAL_SCALE, ASPECT, 0.0, ())
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    drawn = pitch_overlay(frame, calibration, _q_for_pan(0.0), F_CHAIN, 100.0, 64.0)
    assert drawn.shape == frame.shape
    assert not np.array_equal(drawn, frame), "the overlay drew nothing"
    # The markings are yellow (BGR 0, 255, 255); the untouched frame is black.
    assert ((drawn[:, :, 0] == 0) & (drawn[:, :, 1] == 255) & (drawn[:, :, 2] == 255)).any()


def test_actual_center_reports_where_the_crop_really_is() -> None:
    """A centered request is honored exactly; a request near an edge is held back, and the component says so."""
    middle = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    actual_x, actual_y = actual_center(middle, (FRAME_W, FRAME_H))
    assert actual_x == pytest.approx(0.5, abs=1e-3) and actual_y == pytest.approx(0.5, abs=1e-3)

    edge = zoom_box(FRAME_W, FRAME_H, 8.0, 1.0, 1.0)
    edge_x, edge_y = actual_center(edge, (FRAME_W, FRAME_H))
    assert edge_x < 1.0 and edge_y < 1.0, "the crop cannot be centered on the very corner of the frame"
    # Held back by exactly half a crop, which is what the indicator's "held at ..." note reports.
    assert (1.0 - edge_x) * FRAME_W == pytest.approx(240.0, abs=2.0)
    assert (1.0 - edge_y) * FRAME_H == pytest.approx(135.0, abs=2.0)


def test_clamp01_keeps_browser_values_in_range() -> None:
    assert clamp01(0.4) == 0.4
    assert clamp01(-3.0) == 0.0
    assert clamp01(7.5) == 1.0
    assert clamp01(1.0) == 1.0


def test_a_click_becomes_the_center_of_the_crop_around_it() -> None:
    """Jumping back to a stored landmark means centering the view on it - across two y conventions.

    A click's ``v`` is normalized by the frame *width* because that is the ray the solver wants; the view's center
    is a fraction of the frame you see across and down. Converting once, here, is what lands the jump on the
    landmark instead of an aspect ratio away from it.
    """
    u, v = 0.09427083333333333, 0.18020833333333333  # a real corner-flag click, in the solver's convention
    x, y = u * FRAME_W, v * FRAME_W  # where it is in native pixels
    center = point_center(u, v, (FRAME_W, FRAME_H))
    assert center == pytest.approx((x / FRAME_W, y / FRAME_H))

    box = zoom_box(FRAME_W, FRAME_H, 8.0, *center)
    assert actual_center(box, (FRAME_W, FRAME_H)) == pytest.approx(center), "an interior point is honored exactly"
    x0, y0, w, h = box
    assert x0 <= x <= x0 + w and y0 <= y <= y0 + h, "the click itself is inside the crop it centers"


def test_navigation_does_not_touch_stored_clicks() -> None:
    """Moving the view is not a commit: it must leave the clicks exactly as they were.

    The component reports `action`, and the page only merges points when that action is "apply" - a mismatch here
    would silently duplicate or drop clicks every time the user panned.
    """
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    click = click_in_crop(box, scale, 0.25, 0.75)
    stored, next_pid = merge_clicked([], [click], 0, box, scale, FRAME_W, 0)
    before = [dict(point) for point in stored]

    # A navigation value carries the same points back, but with action "navigate": they are ignored.
    result = ClickResult(points=[(1.0, 2.0, "")], center=(0.7, 0.3), zoom=4.0, action="navigate")
    assert result.action != "apply"
    assert stored == before

    # The center is a separate piece of state from the clicks, so moving the view keeps them both.
    assert [point["pid"] for point in order_clicks(stored)] == [0]
    assert next_pid == 1
    assert result.zoom == 4.0, "zoom travels with navigation, since the component now owns it"


def test_frame_points_marks_only_the_frame_being_edited() -> None:
    """The whole-frame view shows dots for the clicks on the frame you are looking at, and nothing else."""
    stored = [
        {"pid": 0, "frame": 0, "u": 0.1, "v": 0.2},
        {"pid": 1, "frame": 5, "u": 0.3, "v": 0.4},
        {"pid": 2, "frame": 0, "u": 0.5, "v": 0.6},
    ]
    assert frame_points(stored, 0) == [(0.1, 0.2), (0.5, 0.6)]
    assert frame_points(stored, 5) == [(0.3, 0.4)]
    assert frame_points(stored, 9) == []
    assert frame_points([], 0) == []


def test_frame_change_only_reports_a_real_move() -> None:
    """A scrub is only acted on while it still means a *change*.

    The component's value is sticky - it comes back on every later rerun - so a scrub that were re-applied each time
    would loop forever. Reporting ``None`` for "same frame" is what breaks the loop, and it also has to cover the
    still-frame fallback, which sends no frame at all.
    """
    assert frame_change(None, 12, 100) is None, "no timeline yet, so nothing to move"
    assert frame_change(12, 12, 100) is None, "the sticky value replaying itself"
    assert frame_change(13, 12, 100) == 13
    assert frame_change(9999, 12, 100) == 99, "clamped into the segment, never past its end"
    assert frame_change(-5, 12, 100) == 0
    # A one-frame segment cannot move anywhere, so it must not report a change either.
    assert frame_change(0, 0, 1) is None


def test_parse_result_reads_the_component_wire_format() -> None:
    """One place knows what the component sends; everything else works with the dataclass."""
    raw = {
        "polygon": [],
        "features": [
            {"type": "point", "point": [12.5, 40.0], "label": "corner far-left"},
            {"type": "point", "point": [3.0, 4.0]},
            {"type": "line", "points": [[0, 0], [1, 1]]},
            {"type": "point", "point": [1.0]},
        ],
        "center_x": 0.35,
        "center_y": 0.62,
        "zoom": 4.5,
        "action": "apply",
        "frame": 120,
        "seq": 7,
        "mount": 2,
    }
    result = parse_result(raw)
    assert result.action == "apply" and result.frame == 120
    assert result.center == pytest.approx((0.35, 0.62)) and result.zoom == pytest.approx(4.5)
    # Only well-formed point features become clicks, and the popover's tag rides on the click.
    assert result.points == [(12.5, 40.0, "corner far-left"), (3.0, 4.0, "")]
    # The gesture identity rides along too: it is what lets the page ignore the sticky value on later reruns.
    assert (result.seq, result.mount) == (7, 2)
    bare = parse_result({"features": [], "action": "navigate"})
    assert (bare.seq, bare.mount) == (0, 0), "an older component value without identity reads as gesture zero"

    # Anything that is not a component value reads as "nothing happened" rather than raising.
    for nothing in (None, "nonsense", 7):
        empty = parse_result(nothing)
        assert empty.points == [] and empty.center is None and empty.action == ""

    # A still-frame fallback sends a null frame and no view; that must not become a click or a move.
    still = parse_result({"features": [], "center_x": None, "zoom": None, "action": "navigate", "frame": None})
    assert still.frame is None and still.center is None and still.zoom is None


def test_re_applying_keeps_each_click_its_identity() -> None:
    """A click that is re-committed unchanged must keep its pid.

    The label dropdowns are keyed by pid, so handing back fresh pids on every Apply resets the landmarks to their
    defaults - which is exactly what made the tags appear to revert on their own.
    """
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    first = click_in_crop(box, scale, 0.2, 0.3)
    second = click_in_crop(box, scale, 0.8, 0.6)

    stored: list[dict] = []
    stored, next_pid = merge_clicked(stored, [first, second], 0, box, scale, FRAME_W, 0)
    assert [point["pid"] for point in stored] == [0, 1]
    before = {point["pid"]: (point["u"], point["v"]) for point in stored}

    # Apply again with the same two points on screen: same pids, so any label already chosen survives.
    stored, next_pid = merge_clicked(stored, [first, second], 0, box, scale, FRAME_W, next_pid)
    assert [point["pid"] for point in stored] == [0, 1]
    assert next_pid == 2, "no new identity should have been handed out"
    assert {point["pid"]: (point["u"], point["v"]) for point in stored} == before

    # Adding a third click gives it a new pid and leaves the first two alone.
    third = click_in_crop(box, scale, 0.5, 0.9)
    stored, next_pid = merge_clicked(stored, [first, second, third], 0, box, scale, FRAME_W, next_pid)
    assert [point["pid"] for point in stored] == [0, 1, 2]
    assert next_pid == 3


def test_identity_is_not_stolen_by_a_neighboring_click() -> None:
    """Two different landmarks close together must stay two landmarks."""
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    first = click_in_crop(box, scale, 0.4, 0.4)
    second = click_in_crop(box, scale, 0.4, 0.4 + 40.0 / box[3])  # far enough apart to be its own landmark

    stored, next_pid = merge_clicked([], [first], 0, box, scale, FRAME_W, 0)
    stored, next_pid = merge_clicked(stored, [first, second], 0, box, scale, FRAME_W, next_pid)
    assert len(stored) == 2
    assert len({point["pid"] for point in stored}) == 2, "the second click took over the first one's identity"


def test_clicks_outside_the_crop_keep_their_identity_across_an_apply() -> None:
    """Applying at one zoom must not disturb the identity of the clicks already there.

    The component always echoes back every point inside the crop (they are handed to it as the initial features and
    returned with the Apply), so this passes both the old and the new point, which is what really happens.
    """
    zoomed = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    zoomed_scale = canvas_scale(zoomed[2])
    stored, next_pid = merge_clicked(
        [], [click_in_crop(zoomed, zoomed_scale, 0.5, 0.5)], 0, zoomed, zoomed_scale, FRAME_W, 0
    )
    original = stored[0]
    assert original["pid"] == 0

    wide = zoom_box(FRAME_W, FRAME_H, 2.0, 0.8, 0.8)
    wide_scale = canvas_scale(wide[2])
    # The wide crop is clamped at the frame edge, and the first click happens to fall inside it.
    echoed = frame_to_canvas(original["u"], original["v"], wide_scale, wide, FRAME_W)
    added = click_in_crop(wide, wide_scale, 0.5, 0.5)
    stored, next_pid = merge_clicked(stored, [echoed, added], 0, wide, wide_scale, FRAME_W, next_pid)

    assert len(stored) == 2
    assert [point["pid"] for point in stored] == [0, 1], "the echoed click lost its identity"
    assert stored[0]["u"] == pytest.approx(original["u"], abs=1e-9)


def test_a_click_that_is_not_echoed_back_is_dropped() -> None:
    """Apply commits what is on screen, so the page and the component must agree on what that is.

    This is the sharp edge of the design: `merge_clicked` trusts the component to report every point inside the
    crop. If it ever reported fewer, the missing one would be silently deleted - so it is worth having the behavior
    written down rather than assumed.
    """
    box = zoom_box(FRAME_W, FRAME_H, 8.0, 0.5, 0.5)
    scale = canvas_scale(box[2])
    first = click_in_crop(box, scale, 0.2, 0.2)
    second = click_in_crop(box, scale, 0.8, 0.8)
    stored, next_pid = merge_clicked([], [first, second], 0, box, scale, FRAME_W, 0)
    assert len(stored) == 2

    # Only the first is echoed back: the second is gone from the view, so it is gone from the state.
    stored, next_pid = merge_clicked(stored, [first], 0, box, scale, FRAME_W, next_pid)
    assert len(stored) == 1
    assert stored[0]["pid"] == 0

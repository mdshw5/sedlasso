"""Pitch-landmark clicking: where to click, and how a click on a zoomed crop maps back to the frame.

This is deliberately kept out of the Streamlit page. The coordinate math is the part of landmark registration that
is easy to get subtly wrong (and impossible to unit-test inside a Streamlit script), so it lives here as pure
functions and the page only wires it to the component.

Three coordinate systems meet here, and mixing them up is silent rather than loud:

* **native pixels** - the frame as decoded, e.g. 3840x2160. `zoom_box` and the click crops work in these.
* **frame-normalized** - ``u = x / frame_width`` and ``v = y / frame_width``. Note ``v`` is divided by the *width*,
  not the height: that is the convention the camera model and the solver use, and it is not the same as the fraction
  of the frame you see on screen.
* **canvas pixels** - what the browser component reports, after the crop has been shrunk to at most `CLICK_MAX_WIDTH`.
  `canvas_scale` converts between these and native pixels, in both directions, so a click lands where it was made.

The zoom sliders take a fraction of the frame *width* and *height* respectively, because that is what a user reads
off the picture in front of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import cv2
import numpy as np

from soccer_analytics.geometry.pitch_calibration import Landmark, PitchCalibration, pitch_to_pixels
from soccer_analytics.geometry.pitch_markings import markings_for

CLICK_MAX_WIDTH = 1600  # canvas pixels sent to the browser; small crops go at native size, which gives the accuracy

# The fixed markings are per format (see geometry.pitch_markings): a 9v9 pitch has a 13-yard penalty area and a
# 7-yard centre circle, and the mini-soccer formats have neither a goal area nor a centre circle. The old code used
# the 11-a-side values for every format, which put the 9v9 penalty area within ~4 m of the centre circle.


def landmark_table(
    length_m: float, width_m: float, *, format_name: str | None = None
) -> dict[str, tuple[float, float]]:
    """Pitch landmarks a user can click, in the order they should be clicked.

    Corners first because they spread the fit across the pitch, then the goalposts - which matter because the near
    corners are often out of frame, and a post is a far easier thing to pick out than a corner flag. The halfway
    line and center spot then pin the middle down.

    The rest are the markings that are *usually* visible and add spread where the corners cannot: the penalty
    spots, and the four cardinals of the center circle. Their positions come from the format's own marking spec
    (see geometry.pitch_markings), so a 9v9 click on the 7-yard circle is measured against 7 yards, not against the
    11-a-side 9.15 m circle. The mini-soccer formats have no centre circle, so those landmarks are omitted rather
    than offered against a marking that is not on the pitch.

    The boxes are deliberately *not* clickable. The goal area is the hardest marking on the pitch to place
    accurately - the box is small, its lines are lost against the netting and the goal frame - and the penalty
    area corners are little better: the corner is a bare junction of two lines with nothing to focus on, so a click
    a meter out is a meter of error in the fit. The posts carry that end of the pitch instead, and the penalty
    spots pin the box's depth.
    """
    markings = markings_for(format_name)
    half_length = length_m / 2
    half_width = width_m / 2
    half_goal = markings.goal_half_width_m
    table: dict[str, tuple[float, float]] = {
        "corner near-left": (0.0, 0.0),
        "corner near-right": (length_m, 0.0),
        "corner far-right": (length_m, width_m),
        "corner far-left": (0.0, width_m),
        # The posts, not the goal center: the base of a post is a hard, high-contrast edge that can be clicked to a
        # pixel, whereas the middle of the goal mouth is a judgment call between two posts - and the posts are what
        # the goal area is measured from, so they are the more useful pair.
        "goal post left-near": (0.0, half_width - half_goal),
        "goal post left-far": (0.0, half_width + half_goal),
        "goal post right-near": (length_m, half_width - half_goal),
        "goal post right-far": (length_m, half_width + half_goal),
        "halfway near": (half_length, 0.0),
        "halfway far": (half_length, width_m),
    }
    if markings.has_center_circle:
        table["center spot"] = (half_length, half_width)
    # Penalty spots: out from the goal line on the goal center line, by the format's own distance.
    table["penalty spot left"] = (markings.penalty_spot_distance_m, half_width)
    table["penalty spot right"] = (length_m - markings.penalty_spot_distance_m, half_width)
    if markings.has_center_circle:
        # Center circle cardinals: the format's radius from the center spot, on the halfway and center lines.
        table["center circle near"] = (half_length, half_width - markings.center_circle_radius_m)
        table["center circle far"] = (half_length, half_width + markings.center_circle_radius_m)
        table["center circle left"] = (half_length - markings.center_circle_radius_m, half_width)
        table["center circle right"] = (half_length + markings.center_circle_radius_m, half_width)
    return table


def landmark_help(format_name: str | None = None) -> dict[str, str]:
    """What each clickable landmark is, using the selected format's own distances.

    The distances matter: a 9v9 penalty spot is 9 yards out, not 11 m, and the posts are 16 feet apart, not 8
    yards. Telling a user to click "11 m out" on a pitch marked at 9 yards is a measurement instruction that is
    wrong by two metres.
    """
    markings = markings_for(format_name)
    posts = (
        "The base of the goalpost at the left-hand end, on the near side of the goal - where the post meets the goal "
        f"line. The posts are {markings.goal_width_m:.2f} m apart for this format, so the pair also fixes the goal "
        "line's direction. Use them when the corner flag is out of frame: a post is a hard vertical edge, far "
        "easier to click than the middle of the goal mouth."
    )
    help_text: dict[str, str] = {
        "corner near-left": "Corner flag at the left-hand end, on the touchline nearest the camera.",
        "corner near-right": "Corner flag at the right-hand end, on the touchline nearest the camera.",
        "corner far-right": "Corner flag at the right-hand end, on the far touchline - usually the hardest one to click.",
        "corner far-left": "Corner flag at the left-hand end, on the far touchline.",
        "goal post left-near": posts,
        "goal post left-far": (
            "The base of the goalpost at the left-hand end, on the far side of the goal - where the post meets the "
            "goal line."
        ),
        "goal post right-near": (
            "The base of the goalpost at the right-hand end, on the near side of the goal - where the post meets the "
            "goal line."
        ),
        "goal post right-far": (
            "The base of the goalpost at the right-hand end, on the far side of the goal - where the post meets the "
            "goal line."
        ),
        "halfway near": "Where the halfway line meets the near touchline.",
        "halfway far": "Where the halfway line meets the far touchline.",
    }
    if markings.has_center_circle:
        help_text["center spot"] = "The center spot (or the center of the center circle)."
    help_text["penalty spot left"] = (
        f"The penalty spot at the left-hand end - {markings.penalty_spot_distance_m:.2f} m out from the goal line, "
        "on the goal center line."
    )
    help_text["penalty spot right"] = (
        f"The penalty spot at the right-hand end - {markings.penalty_spot_distance_m:.2f} m out from the goal line, "
        "on the goal center line."
    )
    if markings.has_center_circle:
        help_text["center circle near"] = (
            f"Where the {markings.center_circle_radius_m:.2f} m center circle crosses the center line, on the near "
            "side of the halfway line."
        )
        help_text["center circle far"] = (
            f"Where the {markings.center_circle_radius_m:.2f} m center circle crosses the center line, on the far "
            "side of the halfway line."
        )
        help_text["center circle left"] = (
            f"Where the {markings.center_circle_radius_m:.2f} m center circle crosses the halfway line, on the "
            "left-hand side of the center spot."
        )
        help_text["center circle right"] = (
            f"Where the {markings.center_circle_radius_m:.2f} m center circle crosses the halfway line, on the "
            "right-hand side of the center spot."
        )
    return help_text


# The 11-a-side wording, kept as a module constant for callers that predate the format-aware helper.
LANDMARK_HELP: dict[str, str] = landmark_help("11v11")


def pitch_landmark_order(length_m: float, width_m: float, *, format_name: str | None = None) -> list[str]:
    """The click order, which is also the order the landmark dropdowns default to."""
    return list(landmark_table(length_m, width_m, format_name=format_name))


def zoom_box(width: int, height: int, zoom: float, center_x: float, center_y: float) -> tuple[int, int, int, int]:
    """The native-pixel crop the clicker shows: ``(x0, y0, w, h)``, always inside the frame.

    Zooming matters more than it sounds. A full 4K frame shown at 1600 px means one screen pixel is ~2.4 native
    pixels, and on the far side of the pitch that is roughly a meter of ground error per pixel. At zoom 8 a screen
    pixel is a native pixel, so the click is as good as the user's eye.
    """
    w = max(64, int(round(width / zoom)))
    h = max(64, int(round(w * height / width)))
    x0 = int(round(center_x * width - w / 2))
    y0 = int(round(center_y * height - h / 2))
    return max(0, min(width - w, x0)), max(0, min(height - h, y0)), w, h


def actual_center(box: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[float, float]:
    """Where the crop really ended up, as fractions of frame width and height.

    This is the same as the requested center unless the crop was clamped at a frame edge, and the difference is what
    the indicator in the component reports back to the user.
    """
    x0, y0, w, h = box
    frame_width, frame_height = frame_size
    return (x0 + w / 2) / frame_width, (y0 + h / 2) / frame_height


def clamp01(value: float) -> float:
    """Keep a 0-1 fraction inside its range, whatever the browser sends back."""
    return float(min(1.0, max(0.0, value)))


def point_center(u: float, v: float, frame_size: tuple[int, int]) -> tuple[float, float]:
    """Where a stored landmark click sits for the view's center: x by width, y by height.

    A click's ``v`` is normalized by the frame *width* (the solver's convention), while the view's center is the
    fraction of the frame you see across and down. The vertical term is therefore converted exactly once, here -
    which is what makes "jump to this landmark" land on it instead of a fraction of an aspect ratio away.
    """
    frame_width, frame_height = frame_size
    return (float(u), float(v) * frame_width / frame_height)


def canvas_scale(crop_width: int) -> float:
    """How much a crop is shrunk before being sent to the browser.

    Used for both directions of the mapping, so a stored click lands exactly where it was clicked when it is drawn
    back onto the crop.
    """
    return min(1.0, CLICK_MAX_WIDTH / crop_width)


def frame_to_canvas(
    u: float, v: float, scale: float, box: tuple[int, int, int, int], frame_width: int
) -> tuple[float, float]:
    """Frame-normalized position -> canvas pixels of the current crop."""
    x0, y0, _w, _h = box
    return (u * frame_width - x0) * scale, (v * frame_width - y0) * scale


def canvas_to_frame(
    x: float, y: float, scale: float, box: tuple[int, int, int, int], frame_width: int
) -> tuple[float, float]:
    """Canvas pixels of the current crop -> frame-normalized position (both axes by width, as the solver expects)."""
    x0, y0, _w, _h = box
    return (x0 + x / scale) / frame_width, (y0 + y / scale) / frame_width


def inside_crop(u: float, v: float, box: tuple[int, int, int, int], scale: float, frame_width: int) -> bool:
    """Whether a frame-normalized position falls inside the current crop (so a marker there is draggable)."""
    x, y = frame_to_canvas(u, v, scale, box, frame_width)
    _x0, _y0, w, h = box
    return 0 <= x <= w * scale and 0 <= y <= h * scale


def marker_feature(label: str, u: float, v: float, box: tuple[int, int, int, int], scale: float, frame_width: int) -> dict:
    """One placed marker as a component feature: canvas position, the landmark it stands for, and an id.

    The id is what lets the component tell a *moved* marker from a stale copy of it: the positions Python sends
    only change when an Apply commits them, so anything the user has dragged since is a measurement in progress
    and must survive the reruns that other gestures (aiming, zooming) cause.
    """
    x, y = frame_to_canvas(u, v, scale, box, frame_width)
    return {"type": "point", "point": [x, y], "label": label, "id": f"seed:{label}"}


def merge_clicked(
    stored: list[dict],
    clicked: list[tuple[float, float] | tuple[float, float, str]],
    frame_index: int,
    box: tuple[int, int, int, int],
    scale: float,
    frame_width: int,
    next_pid: int,
    tolerance_px: float = 3.0,
) -> tuple[list[dict], int]:
    """Fold one Apply into the stored clicks.

    Apply commits what is on screen, so it replaces the clicks of this frame that lie inside the current crop and
    leaves the ones elsewhere alone. That is what lets a corner be clicked at zoom 8 and a different corner of the
    same frame be clicked at another zoom, without either wiping the other.

    A click that comes back where it was keeps its ``pid``. That matters well beyond tidiness: the landmark
    dropdowns are keyed by ``pid``, so re-creating the points on every Apply would silently throw away the landmark
    each click had been tagged as, and the tags would appear to revert on their own.

    A click may carry a third element: the landmark the marker picker had selected when it was made. That becomes
    the point's ``label``, the default the dropdown below the canvas opens on. A click without one keeps whatever
    its ``pid`` already had, so re-committing the crop cannot strip a tag off a point.
    """
    x0, y0, w, h = box

    def inside(point: dict) -> bool:
        x, y = point["u"] * frame_width, point["v"] * frame_width
        return x0 - 1 <= x <= x0 + w + 1 and y0 - 1 <= y <= y0 + h + 1

    kept = [point for point in stored if not (point["frame"] == frame_index and inside(point))]
    # The clicks this Apply is re-reporting, paired with where they sat on the canvas.
    reusable = [
        (frame_to_canvas(point["u"], point["v"], scale, box, frame_width), point)
        for point in stored
        if point["frame"] == frame_index and inside(point)
    ]
    merged = list(kept)
    for click in clicked:
        x, y = click[0], click[1]
        picked = click[2] if len(click) > 2 else ""
        same = next(
            (
                index
                for index, ((px, py), _point) in enumerate(reusable)
                if math.hypot(px - x, py - y) <= tolerance_px
            ),
            None,
        )
        if same is None:
            pid, previous = next_pid, ""
            next_pid += 1
        else:
            point = reusable.pop(same)[1]
            pid, previous = point["pid"], point.get("label", "")
        u, v = canvas_to_frame(x, y, scale, box, frame_width)
        entry = {"pid": pid, "frame": frame_index, "u": u, "v": v}
        if picked or previous:
            entry["label"] = picked or previous
        merged.append(entry)
    return merged, next_pid


def points_in_crop(
    stored: list[dict], frame_index: int, box: tuple[int, int, int, int], scale: float, frame_width: int
) -> list[dict]:
    """The stored clicks of one frame that are visible in the current crop, in the component's canvas coordinates."""
    x0, y0, w, h = box
    features = []
    for point in stored:
        if point["frame"] != frame_index:
            continue
        px, py = frame_to_canvas(point["u"], point["v"], scale, box, frame_width)
        if 0 <= px <= w * scale and 0 <= py <= h * scale:
            feature = {"type": "point", "point": [px, py], "id": f"click:{point['pid']}"}
            if point.get("label"):
                # The tag rides with the click, so an Apply cannot strip a landmark off a point.
                feature["label"] = point["label"]
            features.append(feature)
    return features


def landmarks_from_points(
    labeled: list[tuple[dict, str]], table: dict[str, tuple[float, float]]
) -> list[Landmark]:
    """Turn labeled clicks into solver landmarks, each carrying the frame it was clicked on."""
    return [
        Landmark(point["frame"], point["u"], point["v"], table[label][0], table[label][1], label)
        for point, label in labeled
    ]


def order_clicks(stored: list[dict]) -> list[dict]:
    """Clicks in a stable display order: by frame, then by the order they were clicked."""
    return sorted(stored, key=lambda point: (point["frame"], point["pid"]))


def frame_points(stored: list[dict], frame_index: int) -> list[tuple[float, float]]:
    """Every click on one frame as ``(u, v)``, so the whole-frame view can show the work in progress."""
    return [(point["u"], point["v"]) for point in order_clicks(stored) if point["frame"] == frame_index]


def projected_landmarks(
    calibration: PitchCalibration,
    labels: list[str],
    table: dict[str, tuple[float, float]],
    q: np.ndarray,
    focal: float,
) -> list[dict]:
    """Where the calibration puts the named landmarks on this frame, as frame-normalized ``(u, v)``.

    These are the *predicted* positions - where the current fit believes each marking is. Their whole use is the
    re-anchoring workflow: place them as markers, drag each onto the real marking, and the drags measure exactly
    how far the fit has slid by this moment in the video, which is what the drift correction is fitted from.
    """
    names = [name for name in labels if name in table]
    if not names:
        return []
    xy = np.array([table[name] for name in names], dtype=np.float64)
    uv, in_front = pitch_to_pixels(calibration, xy, q, focal)
    out: list[dict] = []
    for name, point, front in zip(names, uv, in_front):
        if not front or not np.isfinite(point).all():
            continue
        out.append({"label": name, "u": float(point[0]), "v": float(point[1])})
    return out


def restored_points(saved: list[dict], frame_count: int) -> list[dict] | None:
    """Session clicks rebuilt from the clicks saved with a calibration, or None when they cannot be used.

    A page refresh starts a new Streamlit session, so without this the committed clicks only exist in the match
    record while the editor shows an empty crop. Frames are indices into the segment that was clicked on, though -
    a set saved against the whole-game window means nothing on a five-minute one - so a set with any frame outside
    the current segment is refused rather than renumbered into nonsense. Pids are handed out in order, exactly as
    the editor would have.
    """
    points: list[dict] = []
    for index, click in enumerate(saved):
        try:
            frame, u, v = int(click["frame"]), float(click["u"]), float(click["v"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= frame < max(1, frame_count):
            return None
        point = {"pid": index, "frame": frame, "u": u, "v": v}
        if click.get("label"):
            point["label"] = str(click["label"])
        points.append(point)
    return points or None


@dataclass
class ClickResult:
    """What the component last reported.

    ``action`` separates the three things a gesture in the component can mean: ``apply`` commits the points on
    screen (a click or a finished drag - the component commits the moment it happens, there is no Apply button),
    ``clear`` asks for the applied points of the frame being viewed to be removed, and ``navigate`` moved, zoomed
    or re-framed the view (a click, scroll or pan on the whole frame, which is also what loads a scrubbed-to frame
    into the crop). Scrubbing the timeline itself reports nothing at all: it is handled entirely in the browser.

    Each entry of ``points`` is ``(x, y, label)``: canvas pixels, plus the landmark chosen in the popover that
    appeared at the click.

    ``seq``/``mount`` identify the gesture. The component's value is sticky - it comes back on every later rerun -
    and with clicks committing immediately there is no Apply button to re-key the component after, so the page
    dedupes on these instead: a gesture is acted on only when it has not been seen before (same mount, higher seq).
    """

    points: list[tuple[float, float, str]] = field(default_factory=list)
    center: tuple[float, float] | None = None
    zoom: float | None = None
    action: str = ""
    frame: int | None = None
    seq: int = 0
    mount: int = 0


def frame_change(result_frame: int | None, reference: int, frame_count: int) -> int | None:
    """The clamped frame a gesture moved the timeline to, or ``None`` when it did not move it.

    ``None`` is also the answer for "no timeline at all", since the component sends ``null`` while the viewport is
    running on a still frame. Returning ``None`` for "no change" is what stops a rerun loop: a component's value is
    sticky and comes back on every later rerun, so a gesture may only be acted on while it still means a change.
    """
    if result_frame is None:
        return None
    clamped = int(np.clip(result_frame, 0, max(0, frame_count - 1)))
    current = int(np.clip(reference, 0, max(0, frame_count - 1)))
    return None if clamped == current else clamped


def parse_result(value: object) -> ClickResult:
    """Read the value a component gesture reports, as a :class:`ClickResult`.

    The component always reports the whole of its state - the points, the view and the frame - so this is the one
    place that knows the wire format. It is kept here, and pure, because the parsing is easy to get silently wrong:
    a missing ``features`` key reads as "no points", which is indistinguishable from the user having cleared the
    crop, and in Point mode the clicks live in ``features`` rather than ``polygon``.
    """
    if not isinstance(value, dict):
        return ClickResult()
    points: list[tuple[float, float, str]] = []
    for feature in value.get("features") or []:
        if not isinstance(feature, dict) or feature.get("type") != "point":
            continue
        point = feature.get("point") or []
        if len(point) == 2:
            label = feature.get("label")
            points.append((float(point[0]), float(point[1]), str(label) if label else ""))
    center = None
    if isinstance(value.get("center_x"), (int, float)) and isinstance(value.get("center_y"), (int, float)):
        center = (clamp01(value["center_x"]), clamp01(value["center_y"]))
    zoom = float(value["zoom"]) if isinstance(value.get("zoom"), (int, float)) else None
    frame = int(value["frame"]) if isinstance(value.get("frame"), (int, float)) else None
    seq = int(value["seq"]) if isinstance(value.get("seq"), (int, float)) else 0
    mount = int(value["mount"]) if isinstance(value.get("mount"), (int, float)) else 0
    return ClickResult(
        points=points, center=center, zoom=zoom, action=str(value.get("action") or ""), frame=frame,
        seq=seq, mount=mount,
    )


def repeated_labels_within_a_frame(pairs: Iterable[tuple[int, str]]) -> list[str]:
    """Landmarks clicked more than once *on one frame* - the clicks that contradict each other.

    Clicking one landmark again on a *different* frame is how the chain gets re-anchored across a long video, so it
    is not a duplicate and must not be blocked. Two clicks of one landmark on one picture, though, claim two
    different pitch positions for a single image point: whichever is wrong would drag the fit toward itself.
    """
    counts: dict[tuple[int, str], int] = {}
    for frame, label in pairs:
        counts[(frame, label)] = counts.get((frame, label), 0) + 1
    return sorted({label for (frame, label), count in counts.items() if count > 1})


def split_duplicate_clicks(
    entries: Iterable[tuple[int, str, int]],
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, int]]]:
    """Split ``(frame, landmark, pid)`` entries into ``(kept, dropped)``, keeping the newest of each duplicate.

    Two clicks of one landmark on *one* picture claim two pitch positions for a single image point, so at most one
    can be right - and the newest click is the user's latest word on where the marking is. A re-click to fix a slip
    and a re-label that collides with an older click both mean exactly that: replace the old one. Resolving the
    contradiction here, automatically, is what keeps a stray click from blocking the fit - refusing to fit until
    the user finds and removes it reads as the app rejecting perfectly good work.

    Clicks on *different* frames are not duplicates (that is the drift-anchoring workflow), and an untagged click
    cannot contradict anything - callers simply do not list those, and this function only groups exact
    ``(frame, landmark)`` matches. The newest is the highest ``pid``: pids are handed out in click order.
    """
    items = list(entries)
    counts: dict[tuple[int, str], int] = {}
    newest: dict[tuple[int, str], int] = {}
    for frame, label, pid in items:
        key = (frame, label)
        counts[key] = counts.get(key, 0) + 1
        newest[key] = max(newest.get(key, -1), pid)
    kept = [entry for entry in items if counts[(entry[0], entry[1])] == 1 or newest[(entry[0], entry[1])] == entry[2]]
    dropped = [entry for entry in items if counts[(entry[0], entry[1])] > 1 and newest[(entry[0], entry[1])] != entry[2]]
    return kept, dropped


def project_landmarks(
    calibration: PitchCalibration, table: dict[str, tuple[float, float]], q: np.ndarray, focal: float, frame_width: int
) -> dict[str, tuple[float, float]]:
    """Where each named landmark falls in one frame, in pixels.

    Landmarks behind the camera, or off the image, are left out. Drawing these is how the user sees where the corners
    they could not click on have ended up: the fit determines them, so they can be checked against the markings.
    """
    names = list(table)
    xy = np.array([table[name] for name in names], dtype=np.float64)
    uv, in_front = pitch_to_pixels(calibration, xy, q, focal)
    projected: dict[str, tuple[float, float]] = {}
    for name, point, front in zip(names, uv, in_front):
        if not front or not np.isfinite(point).all():
            continue
        x, y = point * frame_width
        if 0 <= x <= frame_width and 0 <= y <= frame_width:
            projected[name] = (float(x), float(y))
    return projected


def pitch_marking_polylines(
    length_m: float, width_m: float, *, format_name: str | None = None, arc_points: int = 48
) -> list[np.ndarray]:
    """The pitch markings as polylines in pitch meters, for drawing the outline back onto a frame.

    Everything is a polyline - circles and arcs are sampled - so the caller only has to project points and draw
    segments. That is what keeps the overlay honest under the perspective projection, where a circle on the ground is
    not a circle in the image. The shapes come from the format's own marking spec, not from a single 11-a-side
    template: a 9v9 pitch has a 13-yard penalty area and a 7-yard centre circle, and the mini-soccer formats have
    no goal area or centre circle at all.
    """
    markings = markings_for(format_name)
    half_length, half_width = length_m / 2, width_m / 2
    lines = [
        # Touchlines and goal lines, as one closed loop.
        np.array([[0.0, 0.0], [length_m, 0.0], [length_m, width_m], [0.0, width_m], [0.0, 0.0]]),
        # Halfway line.
        np.array([[half_length, 0.0], [half_length, width_m]]),
    ]
    # Goal area and penalty area at both ends, using this format's own dimensions.
    for goal_x, inward in ((0.0, 1.0), (length_m, -1.0)):
        if markings.has_goal_area:
            lines.append(
                np.array(
                    [
                        [goal_x, half_width - markings.goal_area_half_width_m],
                        [goal_x + inward * markings.goal_area_depth_m, half_width - markings.goal_area_half_width_m],
                        [goal_x + inward * markings.goal_area_depth_m, half_width + markings.goal_area_half_width_m],
                        [goal_x, half_width + markings.goal_area_half_width_m],
                    ]
                )
            )
        lines.append(
            np.array(
                [
                    [goal_x, half_width - markings.penalty_area_half_width_m],
                    [goal_x + inward * markings.penalty_area_depth_m, half_width - markings.penalty_area_half_width_m],
                    [goal_x + inward * markings.penalty_area_depth_m, half_width + markings.penalty_area_half_width_m],
                    [goal_x, half_width + markings.penalty_area_half_width_m],
                ]
            )
        )
    if markings.has_center_circle:
        angles = np.linspace(0.0, 2.0 * np.pi, arc_points)
        lines.append(
            np.column_stack(
                [
                    half_length + markings.center_circle_radius_m * np.cos(angles),
                    half_width + markings.center_circle_radius_m * np.sin(angles),
                ]
            )
        )
    if markings.has_penalty_arc:
        # The part of the arc around each penalty spot that lies outside the penalty box.
        theta = np.arccos(
            (markings.penalty_area_depth_m - markings.penalty_spot_distance_m) / markings.penalty_arc_radius_m
        )
        for spot_x, facing in (
            (markings.penalty_spot_distance_m, 0.0),
            (length_m - markings.penalty_spot_distance_m, np.pi),
        ):
            arc = np.linspace(facing - theta, facing + theta, arc_points)
            lines.append(
                np.column_stack(
                    [
                        spot_x + markings.penalty_arc_radius_m * np.cos(arc),
                        half_width + markings.penalty_arc_radius_m * np.sin(arc),
                    ]
                )
            )
    # Corner arcs (1 m radius).
    for corner_x, corner_y, start in (
        (0.0, 0.0, 0.0),
        (length_m, 0.0, np.pi / 2),
        (length_m, width_m, np.pi),
        (0.0, width_m, 3 * np.pi / 2),
    ):
        arc = np.linspace(start, start + np.pi / 2, 16)
        lines.append(np.column_stack([corner_x + np.cos(arc), corner_y + np.sin(arc)]))
    return lines


def overlay_homographies(
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
    length_m: float,
    width_m: float,
    frame_count: int,
    *,
    format_name: str | None = None,
    samples: int = 240,
) -> list[dict]:
    """Pitch->pixel homographies sampled along the chain, for the browser to project the overlay itself.

    The whole-frame viewport is scrubbed and played *in the browser*, so Python never sees most of the frames the
    overlay has to be drawn on. Instead Python samples the corrected chain at ``samples`` frames and, for each,
    fits the homography that maps pitch meters to frame-normalized pixels through the calibration at that moment.
    The browser picks the two samples bracketing the frame it is showing and interpolates the coefficients - the
    chain is smooth on this spacing, so the interpolation error is far below a pixel.

    Each entry is ``{"frame": int, "h": [9 floats]}`` with ``h`` row-major, mapping ``(x_m, y_m, 1)`` to
    ``(u, v, 1)`` in frame-normalized pixels (u by width, v by width - the convention everywhere here).
    """
    if frame_count <= 0 or len(q) == 0:
        return []
    # The chain is the authority on how far the overlay can reach: a segment shorter than the timeline (or a
    # still-frame fallback) must not be indexed past its end.
    limit = min(frame_count, len(q))
    frames = np.unique(np.linspace(0, limit - 1, min(samples, limit)).round().astype(int))
    # The markings are the same for every sample; only the projection changes.
    points = np.concatenate(pitch_marking_polylines(length_m, width_m, format_name=format_name), axis=0)
    out: list[dict] = []
    for frame in frames:
        q_frame, focal_frame = calibration.corrected_frame(q[frame], float(focal[frame]), int(frame))
        uv, in_front = pitch_to_pixels(calibration, points, q_frame, focal_frame)
        visible = in_front & np.isfinite(uv).all(axis=1)
        # A camera at midfield never has the whole pitch in front of it, so the fit uses the visible markings -
        # the ground-plane projection is a homography, so fitting on any four-plus of them recovers the same H,
        # which then maps the rest of the plane too.
        if visible.sum() < 4:
            continue
        seen, projected = points[visible], uv[visible]
        # Solve seen -> projected exactly: 2N equations for the 8 unknowns of H (h33 fixed at 1).
        ones = np.ones((len(seen), 1))
        xy1 = np.hstack([seen, ones])
        a = np.zeros((2 * len(seen), 8))
        b = np.empty(2 * len(seen))
        a[0::2, 0:3] = xy1
        a[0::2, 6:8] = -seen * projected[:, 0, None]
        a[1::2, 3:6] = xy1
        a[1::2, 6:8] = -seen * projected[:, 1, None]
        b[0::2] = projected[:, 0]
        b[1::2] = projected[:, 1]
        h, *_ = np.linalg.lstsq(a, b, rcond=None)
        # Row-major [h11..h32, h33=1], so reshaping to 3x3 puts the fixed scale last.
        out.append({"frame": int(frame), "h": [*map(float, h), 1.0]})
    return out


def pitch_overlay(
    image: np.ndarray,
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: float,
    length_m: float,
    width_m: float,
    *,
    format_name: str | None = None,
    table: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw the pitch markings back into the frame - the check that the landmarks were labeled the right way round.

    Without this the user has no way to tell a good registration from a mirrored one: the numbers alone look
    plausible either way. The whole set of standard markings is drawn - touchlines, halfway line, the format's
    boxes, the center circle, the penalty arcs and spots and the corner arcs - so a fit can be checked against the
    markings that are actually visible rather than only the outline. Pass ``table`` to mark each named landmark as
    well, which shows the corners that were never clicked where the fit puts them.
    """
    markings = markings_for(format_name)
    drawn = image.copy()
    width = drawn.shape[1]
    for points in pitch_marking_polylines(length_m, width_m, format_name=format_name):
        uv, in_front = pitch_to_pixels(calibration, points, q, focal)
        pixels = uv * width
        for index in range(len(points) - 1):
            if not (in_front[index] and in_front[index + 1]):
                continue
            start, end = pixels[index], pixels[index + 1]
            if not (np.isfinite(start).all() and np.isfinite(end).all()):
                continue
            cv2.line(drawn, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), (0, 255, 255), 3)

    # The penalty spots are points, not lines, so they are drawn as small filled dots.
    spots = np.array(
        [
            [markings.penalty_spot_distance_m, width_m / 2],
            [length_m - markings.penalty_spot_distance_m, width_m / 2],
        ]
    )
    uv, in_front = pitch_to_pixels(calibration, spots, q, focal)
    for point, front in zip(uv * width, in_front):
        if front and np.isfinite(point).all():
            cv2.circle(drawn, tuple(np.round(point).astype(int)), 5, (0, 255, 255), -1)

    if table:
        for name, (x, y) in project_landmarks(calibration, table, q, focal, width).items():
            spot = (int(round(x)), int(round(y)))
            cv2.drawMarker(drawn, spot, (255, 0, 255), cv2.MARKER_CROSS, 14, 3)
            # Outlined text so the labels stay readable over grass as well as over the pitch markings.
            for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(drawn, name, (spot[0] + 10, spot[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return drawn

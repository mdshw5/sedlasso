"""Standard pitch markings for each match format, in metres.

The markings do **not** scale with the pitch: a 9v9 pitch has its own penalty area, goal area, penalty spot and
centre circle, and a 7v7 or 5v5 pitch does not have a centre circle or a goal area at all. Drawing the 11-a-side
markings on a 9v9 pitch is what put the penalty area 16.5 m out from the goal line while the centre circle was
still 9.15 m in radius - on a 60 m pitch the box and the circle are then only ~4 m apart, and the box appears to
swallow the circle.

The values are taken from *The FA Guide to Pitch and Goalpost Dimensions* (the U11/U12 9v9 diagram on pages 18-19):
32 x 13 yard penalty area, 14 x 4 yard goal area, 9 yard penalty spot, 7 yard centre circle and penalty arc, and
16 x 7 ft goals. The 5v5 and 7v7 layouts use the FA mini-soccer diagrams (no centre circle, no goal area).
Widths are full widths across the pitch; the drawing code takes halves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

YARD_M = 0.9144
FOOT_M = 0.3048


@dataclass(frozen=True)
class PitchMarkings:
    """The fixed markings of one playing format.

    ``has_*`` flags exist because the smaller formats genuinely do not have those markings, rather than having them
    with a zero size: the click list and the overlay must omit them, not draw a degenerate circle at the centre spot.
    """

    goal_width_m: float
    goal_area_width_m: float
    goal_area_depth_m: float
    penalty_area_width_m: float
    penalty_area_depth_m: float
    penalty_spot_distance_m: float
    center_circle_radius_m: float
    penalty_arc_radius_m: float
    has_goal_area: bool = True
    has_center_circle: bool = True
    has_penalty_arc: bool = True

    @property
    def goal_half_width_m(self) -> float:
        return self.goal_width_m / 2.0

    @property
    def goal_area_half_width_m(self) -> float:
        return self.goal_area_width_m / 2.0

    @property
    def penalty_area_half_width_m(self) -> float:
        return self.penalty_area_width_m / 2.0


# The 11-a-side laws, in the approximation the project already used (7.32 m goal, 18.32 m x 5.5 m goal area,
# 40.32 m x 16.5 m penalty area, 11 m spot, 9.15 m circles). Keeping these values unchanged means an existing
# 11v11 calibration keeps exactly the markings it was fitted against.
ELEVEN_A_SIDE = PitchMarkings(
    goal_width_m=7.32,
    goal_area_width_m=18.32,
    goal_area_depth_m=5.5,
    penalty_area_width_m=40.32,
    penalty_area_depth_m=16.5,
    penalty_spot_distance_m=11.0,
    center_circle_radius_m=9.15,
    penalty_arc_radius_m=9.15,
)

# FA U11/U12 9v9: 80 x 50 yard recommended pitch (the match record may be smaller), 16 x 7 ft goals,
# 32 x 13 yard penalty area, 14 x 4 yard goal area, 9 yard spot, 7 yard centre circle and penalty arc.
NINE_A_SIDE = PitchMarkings(
    goal_width_m=16 * FOOT_M,
    goal_area_width_m=14 * YARD_M,
    goal_area_depth_m=4 * YARD_M,
    penalty_area_width_m=32 * YARD_M,
    penalty_area_depth_m=13 * YARD_M,
    penalty_spot_distance_m=9 * YARD_M,
    center_circle_radius_m=7 * YARD_M,
    penalty_arc_radius_m=7 * YARD_M,
)

# FA U9/U10 7v7: 18 x 10 yard penalty area, 8 yard spot, 12 x 6 ft goals. The mini-soccer layout has no goal area
# and no centre circle - the halfway line doubles as the retreat line.
SEVEN_A_SIDE = PitchMarkings(
    goal_width_m=12 * FOOT_M,
    goal_area_width_m=0.0,
    goal_area_depth_m=0.0,
    penalty_area_width_m=18 * YARD_M,
    penalty_area_depth_m=10 * YARD_M,
    penalty_spot_distance_m=8 * YARD_M,
    center_circle_radius_m=0.0,
    penalty_arc_radius_m=0.0,
    has_goal_area=False,
    has_center_circle=False,
    has_penalty_arc=False,
)

# FA U7/U8 5v5: 16 x 9 yard penalty area, 7 yard spot, 12 x 6 ft goals. No goal area or centre circle.
FIVE_A_SIDE = PitchMarkings(
    goal_width_m=12 * FOOT_M,
    goal_area_width_m=0.0,
    goal_area_depth_m=0.0,
    penalty_area_width_m=16 * YARD_M,
    penalty_area_depth_m=9 * YARD_M,
    penalty_spot_distance_m=7 * YARD_M,
    center_circle_radius_m=0.0,
    penalty_arc_radius_m=0.0,
    has_goal_area=False,
    has_center_circle=False,
    has_penalty_arc=False,
)

MARKINGS_BY_FORMAT: dict[str, PitchMarkings] = {
    "5v5": FIVE_A_SIDE,
    "7v7": SEVEN_A_SIDE,
    "9v9": NINE_A_SIDE,
    "11v11": ELEVEN_A_SIDE,
}


def with_goal_width(markings: PitchMarkings, goal_width_m: float | None) -> PitchMarkings:
    """The format's markings with a measured goal width substituted, when one is given.

    A pitch can be marked for one format and fitted with goals of another size (a 9v9 pitch using full-size
    frames is common on shared grounds). The format still supplies every other marking; only the goal mouth and
    the goal-post landmarks move.
    """
    if goal_width_m is None:
        return markings
    return replace(markings, goal_width_m=float(goal_width_m))


def markings_for(format_name: str | None) -> PitchMarkings:
    """The markings for a match format; unknown or missing formats fall back to 11-a-side.

    The fallback is deliberate: older archives predate the format field, and the 11-a-side template is what every
    calibration made before the format-aware markings was fitted against. Guessing a smaller format from the pitch
    size would silently move the landmarks under those clicks.
    """
    if not format_name:
        return ELEVEN_A_SIDE
    key = str(format_name).strip().lower().replace(" ", "").replace("-", "")
    aliases = {
        "5v5": "5v5",
        "5aside": "5v5",
        "7v7": "7v7",
        "7aside": "7v7",
        "9v9": "9v9",
        "9aside": "9v9",
        "11v11": "11v11",
        "11aside": "11v11",
    }
    return MARKINGS_BY_FORMAT.get(aliases.get(key, ""), ELEVEN_A_SIDE)

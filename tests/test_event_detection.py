"""Event detection from the ball scan and the player tracks.

Built on hand-made ball tracks with known geometry rather than the simulated match, because the point of each test
is one detector's rule: a goal needs a center-spot reset, a shot does not, a clearance needs to know which goal the
team defends. The simulated match is used only for the orientation, where the ground truth is the teams' own shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.event_detection import (
    BallMotion,
    _dedupe_goals,
    ball_motion,
    detect_events,
    team_orientations,
)
from soccer_analytics.analysis.events import Event
from soccer_analytics.analysis.stage_b import PlayerTrack

FPS = 5.0
PITCH = (60.0, 40.0)


def _times(frames: int) -> np.ndarray:
    return np.arange(frames) / FPS


def _track(
    track_id: int,
    team: int,
    frames: list[int],
    xy: list[tuple[float, float]],
    speed_kmh: list[float] | None = None,
) -> PlayerTrack:
    frame = np.asarray(frames, dtype=np.int64)
    positions = np.asarray(xy, dtype=np.float64)
    time = frame / FPS
    speed = np.asarray(speed_kmh if speed_kmh is not None else [0.0] * len(frames), dtype=np.float64)
    return PlayerTrack(
        track_id=track_id,
        team=team,
        frame=frame,
        time=time,
        xy=positions,
        sigma_m=np.full(len(frames), 0.5),
        speed_kmh=speed,
        distance_m=0.0,
    )


def _ball(positions: list[tuple[float, float] | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ball track from a list of positions (``None`` for a frame the scan did not see)."""
    frames = len(positions)
    xy = np.full((frames, 2), np.nan)
    measured = np.zeros(frames)
    for i, position in enumerate(positions):
        if position is not None:
            xy[i] = position
            measured[i] = 1.0
    return xy, measured, _times(frames)


def test_ball_motion_is_nan_across_a_gap() -> None:
    """A gap is not a measurement of motion: the velocity across it must stay unknown."""
    xy, measured, times = _ball([(0.0, 0.0), (1.0, 0.0), None, (5.0, 0.0), (6.0, 0.0)])
    motion = ball_motion(xy, measured, times)
    assert not np.isfinite(motion.speed[2]), "a frame with no position has no speed"
    assert not np.isfinite(motion.speed[3]), "a window spanning the gap is not a velocity"


def test_ball_motion_rejects_positions_off_the_pitch() -> None:
    """The projection of a ball near the horizon lands kilometers away; that is not a position."""
    xy, measured, times = _ball([(20.0, 20.0)] * 5 + [(5000.0, 9000.0)] + [(20.0, 20.0)] * 5)
    motion = ball_motion(xy, measured, times, pitch=PITCH)
    assert not np.isfinite(motion.xy[5, 0]), "a position kilometers off the pitch must be dropped"
    assert not np.isfinite(motion.speed[5])


def test_ball_motion_caps_physically_impossible_speeds() -> None:
    """No ball travels at 200 m/s: a reading that fast is a measurement error, not motion."""
    # 50 m in one frame at 5 fps is 250 m/s.
    xy, measured, times = _ball([(10.0, 20.0)] * 3 + [(60.0, 20.0)] * 3)
    motion = ball_motion(xy, measured, times, pitch=PITCH)
    assert np.all(np.isnan(motion.speed) | (motion.speed <= 45.0))


def test_ball_motion_measures_straightness() -> None:
    """A ball traveling in a line is straight; jitter that happens to be fast is not."""
    straight_xy, measured, times = _ball([(10.0 + 2.0 * i, 20.0) for i in range(12)])
    motion = ball_motion(straight_xy, measured, times, pitch=PITCH)
    assert motion.straight[6] > 0.95, "a straight run must read as straight"

    # A zig-zag: fast frame to frame, but going nowhere. The window is two frames, so the test uses a zig-zag whose
    # period is longer than that - which is what real projection jitter looks like (it wanders over several frames).
    zig = [(20.0 + (3.0 if (i // 2) % 2 else 0.0), 20.0) for i in range(12)]
    zig_xy, measured, times = _ball(zig)
    zigzag = ball_motion(zig_xy, measured, times, pitch=PITCH)
    assert zigzag.straight[6] < 0.3, "a zig-zag must not read as travel"


def test_a_goal_needs_the_center_spot_reset() -> None:
    """The ball reaching the goal line is not enough - it has to be put back on the center spot."""
    # Ball still, then driven into the left goal, then reset to the center spot and held there.
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    positions += [(15.0, 20.0), (10.0, 20.0), (5.0, 20.0), (0.5, 20.0), (0.5, 20.0)]
    positions += [None, None, None]
    positions += [(30.0, 20.0)] * 20
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH)
    goals = [e for e in events if e.type == "goal"]
    assert len(goals) == 1, [e.type for e in events]
    assert "center spot" in goals[0].note
    assert goals[0].source == "ball"


def test_a_ball_rolling_through_the_middle_is_not_a_reset() -> None:
    """A goal needs the ball *put back* on the center spot, not merely passing through the middle."""
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    positions += [(15.0, 20.0), (10.0, 20.0), (5.0, 20.0), (0.5, 20.0), (0.5, 20.0)]
    positions += [None, None, None]
    # The ball crosses the center spot at speed and keeps going: a pass, not a restart.
    positions += [(30.0, 20.0), (35.0, 20.0), (40.0, 20.0), (45.0, 20.0), (50.0, 20.0), (55.0, 20.0)]
    positions += [(58.0, 20.0)] * 10
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH)
    assert not any(e.type == "goal" for e in events), [e.type for e in events]


def test_a_forecast_reset_is_not_a_goal() -> None:
    """The scan's forecast across a missed frame is not a sighting: the *reset* has to be measured.

    The ball crossing the line may be a forecast - the scan loses the ball against the net - but the center-spot
    restart that confirms the goal has to be something a detector actually saw.
    """
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    positions += [(15.0, 20.0), (10.0, 20.0), (5.0, 20.0), (0.5, 20.0), (0.5, 20.0)]
    positions += [None, None, None]
    positions += [(30.0, 20.0)] * 20
    xy, measured, times = _ball(positions)
    # Mark the frames at the center spot as forecasts rather than detections.
    measured[17:] = 0.0
    events = detect_events(xy, measured, times, [], PITCH)
    assert not any(e.type == "goal" for e in events), "a forecast reset is not a restart"


def _image_goal_scene(
    *,
    frames: int = 120,
    goal_frames: range = range(20, 36),
    reset_frames: range | None = range(100, 116),
    goal_side: int = 1,
    moving_ball: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A ball seen in the picture at the goal while its pitch projection is blind (NaN).

    Returns ``(xy, measured, times, ball_uv, goal_mouth_uv, center_uv)``. The goal mouth is the right one by
    default (side 1), at normalized u 0.4-0.6 and goal-line v 0.2; the ball sits at u=0.5, v=0.1 (behind the
    line). The optional reset puts a measured ball at the center spot, in both pitch and picture space.
    """
    times = _times(frames)
    xy = np.full((frames, 2), np.nan)
    measured = np.zeros(frames)
    ball_uv = np.full((frames, 2), np.nan)
    mouth = np.full((frames, 2, 3), np.nan)
    mouth[:, 0] = (-1.0, -1.0, -1.0)  # left goal behind the camera
    mouth[:, 1] = (0.4, 0.6, 0.2)
    for index, frame in enumerate(goal_frames):
        measured[frame] = 1.0
        # The pitch projection is deliberately NaN: this is the far-goal/horizon failure the image path exists for.
        ball_uv[frame] = (0.5 + (0.002 * index if moving_ball else 0.0), 0.1)
    if reset_frames is not None:
        for frame in reset_frames:
            measured[frame] = 1.0
            xy[frame] = (30.0, 20.0)
            ball_uv[frame] = (0.5, 0.25)
    center_uv = np.tile(np.array([0.5, 0.25]), (frames, 1))
    return xy, measured, times, ball_uv, mouth, center_uv


def test_a_goal_is_confirmed_in_the_picture_when_the_pitch_projection_is_blind() -> None:
    """At the far goal the ball can be at the horizon; the picture still shows it in the net."""
    xy, measured, times, ball_uv, mouth, center_uv = _image_goal_scene()
    events = detect_events(
        xy,
        measured,
        times,
        [],
        PITCH,
        goal_half_width_m=3.66,
        ball_uv=ball_uv,
        goal_mouth_uv=mouth,
        center_uv=center_uv,
    )
    goals = [event for event in events if event.type == "goal"]
    assert len(goals) == 1, [event.type for event in events]
    assert "in the picture" in goals[0].note
    assert goals[0].confidence == 0.65


def test_a_picture_goal_without_a_reset_is_not_confirmed() -> None:
    """A ball at the goal mouth is a crossing; only a restart (or a half boundary) confirms it."""
    xy, measured, times, ball_uv, mouth, center_uv = _image_goal_scene(reset_frames=None)
    events = detect_events(
        xy,
        measured,
        times,
        [],
        PITCH,
        goal_half_width_m=3.66,
        ball_uv=ball_uv,
        goal_mouth_uv=mouth,
        center_uv=center_uv,
    )
    assert not any(event.type == "goal" for event in events), [event.type for event in events]


def test_a_goal_on_the_stroke_of_half_time_uses_the_second_half_restart() -> None:
    """A first-half goal's restart is the second-half kickoff, which is outside the half but inside the window."""
    frames = 350
    times = _times(frames)
    xy = np.full((frames, 2), np.nan)
    measured = np.zeros(frames)
    ball_uv = np.full((frames, 2), np.nan)
    mouth = np.full((frames, 2, 3), np.nan)
    mouth[:, 0] = (0.4, 0.6, 0.2)
    mouth[:, 1] = (-1.0, -1.0, -1.0)
    for frame in range(0, 31):
        measured[frame] = 1.0
    for frame, x in zip(range(11, 16), (15.0, 10.0, 5.0, 0.5, 0.5)):
        xy[frame] = (x, 20.0)
    xy[16:31] = (0.5, 20.0)
    for frame in range(14, 31):
        ball_uv[frame] = (0.5, 0.1)
    center_uv = np.tile(np.array([0.5, 0.25]), (frames, 1))
    events = detect_events(
        xy,
        measured,
        times,
        [],
        PITCH,
        half_bounds=(0.0, 60.0, 80.0),
        goal_half_width_m=3.66,
        ball_uv=ball_uv,
        goal_mouth_uv=mouth,
        center_uv=center_uv,
    )
    goals = [event for event in events if event.type == "goal"]
    assert len(goals) == 1, [event.type for event in events]
    assert "second half" in goals[0].note


def test_the_goal_width_setting_changes_the_mouth() -> None:
    """A shot 4 m from the centre is outside a 9v9 goal and inside an 11-a-side one."""
    positions: list[tuple[float, float] | None] = [(20.0, 24.0)] * 11
    positions += [(15.0, 24.0), (10.0, 24.0), (5.0, 24.0), (0.5, 24.0), (0.5, 24.0)]
    positions += [None, None, None]
    positions += [(30.0, 20.0)] * 20
    xy, measured, times = _ball(positions)
    narrow = detect_events(xy, measured, times, [], PITCH, goal_half_width_m=2.44)
    wide = detect_events(xy, measured, times, [], PITCH, goal_half_width_m=3.66)
    assert not any(event.type == "goal" for event in narrow), [event.type for event in narrow]
    assert any(event.type == "goal" for event in wide), [event.type for event in wide]


def test_a_manual_offside_suppresses_a_goal_candidate() -> None:
    """The ball went in, but the human said offside: the detector must not overrule that with a goal."""
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    positions += [(15.0, 20.0), (10.0, 20.0), (5.0, 20.0), (0.5, 20.0), (0.5, 20.0)]
    positions += [None, None, None]
    positions += [(30.0, 20.0)] * 20
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH, offside_times=[times[14]])
    assert not any(event.type == "goal" for event in events), [event.type for event in events]


def test_goal_dedupe_keeps_one_event_per_episode() -> None:
    """A goal and its celebration/rebound within half a minute are one goal, not two."""
    first = Event(time_s=100.0, type="goal", confidence=0.65, video="v")
    second = Event(time_s=110.0, type="goal", confidence=0.8, video="v")
    third = Event(time_s=160.0, type="goal", confidence=0.65, video="v")
    kept = _dedupe_goals([first, second, third])
    assert [event.time_s for event in kept] == [110.0, 160.0]


def test_a_shot_without_a_reset_is_not_a_goal() -> None:
    """A hard kick at the goal that is not followed by a center-spot reset is a shot, not a goal."""
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    # A hard, straight shot at the left goal that travels and then stops short of the line.
    positions += [(15.0, 20.0), (10.0, 20.0), (5.0, 20.0), (2.0, 20.0), (2.0, 20.0)]
    positions += [(2.0, 20.0)] * 6
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH)
    assert not any(e.type == "goal" for e in events)
    shots = [e for e in events if e.type == "shot"]
    assert len(shots) == 1, [e.type for e in events]
    assert "goal" in shots[0].note


def test_a_hard_kick_that_goes_nowhere_is_not_a_shot() -> None:
    """A shot travels: a hard kick that stops after a few meters is not one."""
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 11
    positions += [(18.0, 20.0), (17.0, 20.0), (17.0, 20.0), (17.0, 20.0)]
    positions += [(17.0, 20.0)] * 6
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH)
    assert not any(e.type == "shot" for e in events), [e.type for e in events]


def test_a_corner_is_a_still_ball_at_the_flag_then_a_kick() -> None:
    positions: list[tuple[float, float] | None] = [(0.5, 0.5)] * 9
    positions += [(5.0, 5.0), (10.0, 10.0), (15.0, 15.0)]
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [], PITCH)
    corners = [e for e in events if e.type == "corner"]
    assert len(corners) == 1, [e.type for e in events]
    assert "corner" in corners[0].note


def test_a_penalty_needs_a_whistle() -> None:
    """The same geometry without a whistle is not a penalty - the whistle is what makes it one."""
    positions: list[tuple[float, float] | None] = [(11.0, 20.0)] * 9
    positions += [(16.0, 20.0), (21.0, 20.0), (26.0, 20.0)]
    xy, measured, times = _ball(positions)

    with_whistle = detect_events(xy, measured, times, [], PITCH, whistles=[1.0])
    assert any(e.type == "penalty" for e in with_whistle), [e.type for e in with_whistle]

    without = detect_events(xy, measured, times, [], PITCH, whistles=[])
    assert not any(e.type == "penalty" for e in without)


def test_a_clearance_needs_the_defending_side() -> None:
    """A hard, long kick away from the goal a team defends, from its own third, is a clearance."""
    # Team 0 defends the left goal (its players sit at small x); team 1 defends the right.
    team0 = _track(1, 0, list(range(30)), [(15.0, 20.0)] * 30)
    team1 = _track(2, 1, list(range(30)), [(45.0, 20.0)] * 30)
    # A team-0 player beside the ball just before it is kicked up the pitch and toward the sideline.
    defender = _track(3, 0, list(range(30)), [(18.0, 20.0)] * 30)
    positions: list[tuple[float, float] | None] = [(10.0, 20.0)] * 9
    # A long, fast, straight kick out of the defensive third: 5 m per frame is 25 m/s.
    positions += [(15.0, 22.0), (20.0, 24.0), (25.0, 26.0), (30.0, 28.0), (35.0, 30.0), (40.0, 32.0)]
    positions += [(45.0, 34.0)] * 10
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [team0, team1, defender], PITCH)
    clearances = [e for e in events if e.type == "clearance"]
    assert len(clearances) == 1, [e.type for e in events]
    assert clearances[0].team == 0
    assert "left goal" in clearances[0].note
    assert "traveling" in clearances[0].note


def test_a_short_kick_out_of_defense_is_not_a_clearance() -> None:
    """A pass out of the third is not a clearance: the kick has to travel."""
    team0 = _track(1, 0, list(range(30)), [(15.0, 20.0)] * 30)
    team1 = _track(2, 1, list(range(30)), [(45.0, 20.0)] * 30)
    defender = _track(3, 0, list(range(30)), [(18.0, 20.0)] * 30)
    positions: list[tuple[float, float] | None] = [(10.0, 20.0)] * 9
    # Fast, but the ball stops after 8 m: a pass, not a clearance.
    positions += [(15.0, 20.0), (18.0, 20.0), (18.0, 20.0), (18.0, 20.0), (18.0, 20.0), (18.0, 20.0)]
    positions += [(18.0, 20.0)] * 10
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [team0, team1, defender], PITCH)
    assert not any(e.type == "clearance" for e in events), [e.type for e in events]


def test_a_tackle_is_a_player_stopping_beside_the_ball_as_it_changes() -> None:
    player = _track(
        7,
        0,
        list(range(11)),
        [(24.0, 20.0)] * 11,
        speed_kmh=[15.0, 15.0, 15.0, 15.0, 15.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 5
    positions += [(25.0, 20.0), (30.0, 20.0), (35.0, 20.0), (40.0, 20.0), (45.0, 20.0), (50.0, 20.0)]
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [player], PITCH)
    tackles = [e for e in events if e.type == "tackle"]
    assert len(tackles) == 1, [e.type for e in events]
    assert tackles[0].player_track == 7
    assert "not pose" in tackles[0].note


def test_events_carry_the_shirt_number_when_one_is_known() -> None:
    player = _track(7, 0, list(range(11)), [(24.0, 20.0)] * 11, speed_kmh=[15.0] * 5 + [1.0] * 6)
    positions: list[tuple[float, float] | None] = [(20.0, 20.0)] * 5
    positions += [(25.0, 20.0), (30.0, 20.0), (35.0, 20.0), (40.0, 20.0), (45.0, 20.0), (50.0, 20.0)]
    xy, measured, times = _ball(positions)
    events = detect_events(xy, measured, times, [player], PITCH, numbers={7: {"number": 9, "name": "Ada"}})
    tackle = next(e for e in events if e.type == "tackle")
    assert tackle.player_number == 9


def test_orientation_reads_the_defending_side_from_where_players_stand() -> None:
    team0 = _track(1, 0, list(range(10)), [(15.0, 20.0)] * 10)
    team1 = _track(2, 1, list(range(10)), [(45.0, 20.0)] * 10)
    orientations = team_orientations([team0, team1], PITCH[0])
    assert len(orientations) == 1
    assert orientations[0].defending_goal == {0: "left", 1: "right"}
    assert orientations[0].attack_direction == {0: 1, 1: -1}


def test_orientation_swap_flips_every_end_and_direction() -> None:
    """The user's side correction inverts both the defended goal and the attack vector for each team."""
    team0 = _track(1, 0, list(range(10)), [(15.0, 20.0)] * 10)
    team1 = _track(2, 1, list(range(10)), [(45.0, 20.0)] * 10)
    swapped = team_orientations([team0, team1], PITCH[0], swap=True)
    assert swapped[0].defending_goal == {0: "right", 1: "left"}
    assert swapped[0].attack_direction == {0: -1, 1: 1}


def test_orientation_splits_by_half_when_the_clock_is_marked() -> None:
    """Teams change ends at half-time, so the sides are read per half from the game clock."""
    # Team 0 is at small x in the first half and large x in the second: it changed ends.
    first = _track(1, 0, list(range(0, 10)), [(15.0, 20.0)] * 10)
    second = _track(1, 0, list(range(10, 20)), [(45.0, 20.0)] * 10)
    other_first = _track(2, 1, list(range(0, 10)), [(45.0, 20.0)] * 10)
    other_second = _track(2, 1, list(range(10, 20)), [(15.0, 20.0)] * 10)
    orientations = team_orientations(
        [first, second, other_first, other_second], PITCH[0], half_bounds=(0.0, 2.0, 4.0)
    )
    by_half = {o.half: o for o in orientations}
    assert by_half[1].defending_goal[0] == "left"
    assert by_half[2].defending_goal[0] == "right", "team 0 changed ends at half-time"


def test_no_events_from_an_empty_ball_track() -> None:
    xy, measured, times = _ball([None] * 10)
    assert detect_events(xy, measured, times, [], PITCH) == []


def test_detection_runs_on_a_simulated_match_without_inventing_events() -> None:
    """The detectors must survive real track shapes - and a match with no goals must produce no goals.

    The simulated match has no goal, corner or penalty in it: the ball wanders around the middle of the pitch. So
    the strong claim here is the negative one - the detectors must not invent a goal from ordinary play. Shots and
    clearances may or may not appear (the simulated ball does move fast at times), which is why only the goal,
    corner and penalty counts are asserted.
    """
    from soccer_analytics.analysis.projection import project_segment, segment_poses
    from soccer_analytics.analysis.stage_b import build_report
    from soccer_analytics.geometry.pitch_calibration import PitchCalibration
    from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match

    segment, truth = simulate_match(frames=300, seed=5)
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    calibration = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    detections = project_segment(segment, calibration, poses=(q, focal))
    report, _ = build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    # The simulated ball is the truth's own path, projected as if the scan had seen every frame.
    ball_xy = truth.ball.copy()
    ball_measured = np.ones(len(ball_xy))
    events = detect_events(
        ball_xy, ball_measured, np.asarray(segment.time, dtype=np.float64), report.players, (PITCH_LENGTH, PITCH_WIDTH)
    )
    assert not any(e.type == "goal" for e in events), "a goal was invented from ordinary play"
    assert not any(e.type == "corner" for e in events), "a corner was invented from ordinary play"
    assert not any(e.type == "penalty" for e in events), "a penalty was invented from ordinary play"
    # Every event that was reported must be attributable and carry a note saying what was measured.
    for event in events:
        assert event.source == "ball"
        assert event.note
        assert event.type in ("shot", "clearance", "tackle")


# --------------------------------------------------------------------------------------------------------------
# Rebuilding tracks from the replay payload: the frame indices are within the analyzed window.
# --------------------------------------------------------------------------------------------------------------
def test_player_tracks_from_replay_applies_the_window_offset() -> None:
    """A track's time is ``start_s + frame / fps``, not ``frame / fps``.

    Frame indices are within the analyzed window, and the window starts at the kick-off offset chosen in Step 1.
    Getting this wrong shifted every player-derived event by the offset: on the real game (kick-off at 9:00) a
    tackle at 11:11 was reported at 2:11, and the clip cut for it showed the wrong part of the match entirely.
    """
    from soccer_analytics.analysis.event_detection import player_tracks_from_replay

    replay = {
        "players": [
            {"track_id": 7, "team": 0, "frames": [0, 5, 10], "xy": [[10.0, 20.0]] * 3, "speed": [0.0] * 3,
             "stats": {"distance_m": 1.0}},
        ]
    }
    tracks = player_tracks_from_replay(replay, fps=5.0, start_s=540.804)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.time[0] == pytest.approx(540.804), "frame 0 is the window's first frame, not the video's"
    assert track.time[1] == pytest.approx(540.804 + 1.0)
    assert track.time[2] == pytest.approx(540.804 + 2.0)
    assert track.distance_m == 1.0, "the stats ride along"


def test_player_tracks_from_replay_without_an_offset_is_zero_based() -> None:
    """A segment that starts at the video's own frame 0 keeps the old behavior."""
    from soccer_analytics.analysis.event_detection import player_tracks_from_replay

    replay = {"players": [{"track_id": 1, "team": 0, "frames": [10], "xy": [[5.0, 5.0]], "speed": [0.0]}]}
    tracks = player_tracks_from_replay(replay, fps=5.0)
    assert tracks[0].time[0] == pytest.approx(2.0)

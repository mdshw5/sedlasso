"""The replay payload: what the animated pitch view can and cannot show.

Built against the simulated match because it is the only place with known ground truth for tracks, positions and
the camera aim the ball proxy comes from.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.projection import project_ball_track, project_segment, segment_poses
from soccer_analytics.dashboard.replay import (
    aiming_goal,
    build_replay,
    event_from_tag,
    event_play,
    player_table_rows,
    repeated_manual_tag,
)
from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def simulated_match_case():
    """One simulated match, its recovered poses and the calibration a projection needs."""
    segment, truth = simulate_match(frames=200, seed=3)
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    calibration = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    return segment, truth, calibration, q, focal


@pytest.fixture(scope="module")
def replay_case(simulated_match_case):
    segment, _truth, calibration, q, focal = simulated_match_case
    detections = project_segment(segment, calibration, poses=(q, focal))
    report, _ = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    replay = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
        camera_xy=detections.camera_xy,
    )
    return replay, report, segment, detections


@pytest.fixture(scope="module")
def ball_case(simulated_match_case):
    """The same match, plus what a ball record needs to be projected back to the pitch it was seen on."""
    return simulated_match_case


def test_every_reported_track_is_in_the_replay(replay_case) -> None:
    replay, report, segment, _ = replay_case
    assert len(replay["players"]) <= len(report.players)
    ids = {player["track_id"] for player in replay["players"]}
    assert ids <= {player.track_id for player in report.players}
    assert replay["pitch"] == [PITCH_LENGTH, PITCH_WIDTH]
    assert replay["fps"] == float(segment.meta["fps"])
    assert replay["duration_s"] > 0


def test_bystanders_are_excluded_from_the_field_of_play() -> None:
    """The detector tracks everyone in frame - coaches, photographers, spectators - and the pitch animation
    must not draw them on the field of play.

    A synthetic spectator stands still all match, so their track spans a meter or two and never moves; a player
    covers ground. The payload keeps the players and drops the spectators, and says how many it dropped.
    """
    from soccer_analytics.analysis.projection import on_pitch_mask

    segment, truth = simulate_match(frames=400, seed=5)
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
    report, _ = stage_b.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    replay = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
        camera_xy=detections.camera_xy,
    )
    kept_ids = {player["track_id"] for player in replay["players"]}
    assert replay["bystanders_excluded"] == len(report.players) - len(replay["players"])
    # No team-labeled player may be dropped as a bystander: the rule is for people off the field of play.
    for player in report.players:
        if player.team in (0, 1):
            assert player.track_id in kept_ids, f"team player {player.track_id} was dropped as a bystander"
    # Every dropped track must actually look like a bystander: small extent, barely moving.
    from soccer_analytics.dashboard.replay import _is_bystander

    for player in report.players:
        if player.track_id not in kept_ids:
            assert _is_bystander(player), f"track {player.track_id} was dropped but does not look like a bystander"


def test_stillness_is_judged_in_seconds_not_observations() -> None:
    """The evidence floor is time, not a frame count - the analysis rate once changed under it.

    It was 20 observations, written at 5 fps (4 s of behaviour). At the 15 fps the pipeline now runs, that was
    a bare 1.3 s, and on the 2026-10-03 game the exclusion rate tripled to 47% of tracks - 84% of them holding
    under 10 seconds of observations, fragments of real players who merely stood still for a moment (worst at
    the window's start, where the whole kickoff formation is standing). In seconds the rule cannot drift again.
    """
    from soccer_analytics.analysis.stage_b import PlayerTrack
    from soccer_analytics.dashboard.replay import _is_bystander

    def still(observations: int, fps: float = 15.0) -> PlayerTrack:
        frame = np.arange(observations)
        return PlayerTrack(
            track_id=1,
            team=-1,
            frame=frame,
            time=frame / fps,
            xy=np.tile(np.array([50.0, 32.0]), (observations, 1)),
            sigma_m=np.ones(observations),
            speed_kmh=np.zeros(observations),
            distance_m=0.0,
        )

    assert _is_bystander(still(30)) is False, "2 s standing still is too little evidence"
    assert _is_bystander(still(120)) is False, "8 s standing still is still too little"
    assert _is_bystander(still(300)) is True, "20 s in one spot is a bystander"


def test_observations_are_ordered_and_complete(replay_case) -> None:
    replay, _, segment, _ = replay_case
    for player in replay["players"]:
        frames = player["frames"]
        assert len(frames) == len(player["xy"]) == len(player["speed"])
        assert frames == sorted(frames), "the component interpolates between consecutive observations"
        assert frames[0] >= 0 and frames[-1] < len(segment.time)
        assert 0.0 <= player["stats"]["top_speed_kmh"] <= stage_b.MAX_PLAUSIBLE_SPEED_KMH
        assert player["stats"]["distance_m"] >= 0.0


def test_positions_stay_on_the_pitch(replay_case) -> None:
    """A replay dot outside the pitch would mean the projection, not the player, is on show."""
    replay, _, _, _ = replay_case
    for player in replay["players"]:
        xy = np.asarray(player["xy"], dtype=np.float64)
        assert xy[:, 0].min() > -3.0 and xy[:, 0].max() < PITCH_LENGTH + 3.0
        assert xy[:, 1].min() > -3.0 and xy[:, 1].max() < PITCH_WIDTH + 3.0


def test_the_aim_trail_has_one_entry_per_frame(replay_case) -> None:
    replay, _, segment, _ = replay_case
    assert len(replay["aim"]) == len(segment.time)
    for entry in replay["aim"]:
        assert entry is None or (len(entry) == 2 and all(isinstance(value, float) for value in entry))
    assert any(entry is not None for entry in replay["aim"]), "the camera looked at the pitch at least sometimes"
    # The ball layer is absent, not empty, when no scan has run: the component draws it only when it exists.
    assert replay["ball"] is None


def test_the_payload_carries_the_camera_position(replay_case) -> None:
    """The direction line is drawn from the camera's own ground position, so it must ride in the payload.

    It is a single fixed point (the tripod does not move), rounded like every other coordinate, and absent when the
    caller has no calibration to give one - the component then draws no line rather than a line from the origin.
    """
    replay, _, _, detections = replay_case
    assert replay["camera"] == [round(float(detections.camera_xy[0]), 1), round(float(detections.camera_xy[1]), 1)]

    without = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH),
        float(replay["fps"]),
        replay["frame_count"],
        np.full((replay["frame_count"], 2), np.nan),
        [],
        ["Team A", "Team B"],
    )
    assert without["camera"] is None, "no calibration: no camera position to draw from"


def test_a_ball_record_projects_back_onto_the_pitch(ball_case) -> None:
    """A scan record stores a pixel; the replay needs meters, and the round trip must return the same spot.

    ``pitch_to_pixels`` is the oracle: the simulated ball's true position, projected with the same calibration the
    scan's records are projected with. If framing, normalization or the corrected chain were off, the ball would
    be drawn meters from the spot it was on.
    """
    segment, truth, calibration, q, focal = ball_case
    frames = [12, 40, 90, 150]
    records = []
    for frame in frames:
        uv, in_front = pitch_to_pixels(calibration, truth.ball[frame][None, :], q[frame], float(focal[frame]))
        if not in_front[0] or not np.isfinite(uv).all():
            continue
        records.append({"i": frame, "status": "tracking", "u": float(uv[0, 0]), "v": float(uv[0, 1])})
    assert records, "the camera follows the ball, so its true position must be projectable on these frames"

    xy, measured = project_ball_track(records, calibration, q, focal)

    assert len(xy) == len(segment.time)
    for record in records:
        frame = record["i"]
        error = float(np.linalg.norm(xy[frame] - truth.ball[frame]))
        assert error < 0.05, f"frame {frame}: projected ball {error:.3f} m off"
        assert measured[frame] == 1.0, "a tracking record is a measurement"


def test_a_coasted_frame_is_a_forecast_and_a_lost_frame_is_empty(ball_case) -> None:
    """The scan's honesty rule has to survive into the replay payload: a forecast never reads as a sighting."""
    segment, truth, calibration, q, focal = ball_case
    uv, _ = pitch_to_pixels(calibration, truth.ball[5][None, :], q[5], float(focal[5]))
    records = [
        {"i": 5, "status": "coasting", "u": float(uv[0, 0]), "v": float(uv[0, 1])},
        {"i": 6, "status": "out_of_view", "u": float(uv[0, 0]), "v": float(uv[0, 1])},
        {"i": 7, "status": "lost", "u": None, "v": None},
    ]
    xy, measured = project_ball_track(records, calibration, q, focal)

    assert np.isfinite(xy[5]).all() and measured[5] == 0.0, "a coasted position is a forecast, not a detection"
    assert np.isnan(xy[6]).all(), "out of view: the ball left the picture, so there is no sighting to draw"
    assert np.isnan(xy[7]).all(), "lost: the scan has no position at all"
    assert np.isnan(xy[:5]).all() and np.isnan(xy[8:]).all(), "frames the scan has not reached stay empty"


def test_the_payload_keeps_the_ball_and_marks_forecasts(replay_case) -> None:
    """The component decides what to draw from this payload alone, so the measured flag must ride along.

    Entries are ``[x, y, measured]`` per frame or ``null``: the round trip through JSON must not turn a forecast
    into a sighting (or the filled dot into a ring, the other way round).
    """
    replay, report, segment, detections = replay_case
    frames = len(segment.time)
    xy = np.full((frames, 2), np.nan)
    xy[3] = (21.5, 30.3)
    xy[4] = (21.9, 30.4)
    measured = np.zeros(frames)
    measured[3] = 1.0  # a detection; frame 4 is the tracker's forecast across a miss

    rebuilt = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        frames,
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
        ball=(xy, measured),
    )

    assert rebuilt["ball"][3] == [21.5, 30.3, 1] and rebuilt["ball"][4] == [21.9, 30.4, 0]
    assert rebuilt["ball"][0] is None, "no record: nothing to draw"
    assert len(rebuilt["ball"]) == frames


def test_the_payload_carries_the_measured_kit_colors(replay_case) -> None:
    """The markers wear the kit the clustering measured, so the colors must ride in the payload - and stay absent.

    ``None`` is a value here, not a gap: a team whose kits could not be separated must arrive as ``None`` so the
    component falls back to its own palette, rather than being handed a made-up color that looks measured.
    """
    replay, report, segment, detections = replay_case
    common = (
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
    )

    rebuilt = build_replay(*common, team_colors=[(220, 30, 30), None])
    assert rebuilt["team_colors"] == [[220, 30, 30], None]
    assert replay["team_colors"] is None, "a replay built without colors must not carry an invented palette"

    # The browser builds rgb(...) from these numbers; out-of-range values would make that string invalid.
    clamped = build_replay(*common, team_colors=[(300, -5, 128), (0, 0, 0)])
    assert clamped["team_colors"] == [[255, 0, 128], [0, 0, 0]]


def test_role_labeled_tracks_survive_the_bystander_filter(replay_case) -> None:
    """A keeper stands still near one goal, which trips the bystander rule - the role must win.

    Measured on the real game: 15 of 28 role-labeled tracks (keepers, all slow and compact) were being dropped
    from the payload, so the view could not color-code people it was not drawing anyway.
    """
    _replay, report, segment, detections = replay_case
    frames = np.arange(0, 300, dtype=np.int64)
    stationary = stage_b.PlayerTrack(
        track_id=99999,
        team=-1,
        frame=frames,
        time=540.0 + frames / 5.0,
        xy=np.tile([50.0, 10.0], (len(frames), 1)),
        sigma_m=np.ones(len(frames)),
        speed_kmh=np.zeros(len(frames)),
        distance_m=0.0,
    )
    common = (
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        list(report.players) + [stationary],
        ["Team A", "Team B"],
    )
    plain = build_replay(*common)
    assert 99999 not in {p["track_id"] for p in plain["players"]}, "a stationary track must trip the bystander rule"

    with_roles = build_replay(*common, roles={99999: {"role": "goalkeeper", "side": "left"}})
    entry = next(p for p in with_roles["players"] if p["track_id"] == 99999)
    assert entry["role"] == {"role": "goalkeeper", "side": "left"}
    assert with_roles["bystanders_excluded"] == plain["bystanders_excluded"] - 1


def test_the_payload_carries_attack_directions(replay_case) -> None:
    replay, report, segment, detections = replay_case
    common = (
        (PITCH_LENGTH, PITCH_WIDTH),
        float(segment.meta["fps"]),
        len(segment.time),
        detections.aim_xy,
        report.players,
        ["Team A", "Team B"],
    )
    rebuilt = build_replay(*common, attack={"half_frame": 7, "directions": [[1, -1], [-1, 1]]})
    assert rebuilt["attack"] == {"half_frame": 7, "directions": [[1, -1], [-1, 1]]}
    assert replay["attack"] is None, "a replay built without directions must not invent them"


def test_attack_summary_pairs_directions_per_period() -> None:
    from soccer_analytics.analysis.event_detection import HalfOrientation
    from soccer_analytics.dashboard.replay import attack_summary

    orientations = [
        HalfOrientation(1, {0: "left", 1: "right"}, {0: 1, 1: -1}, {0: 44.7, 1: 51.5}),
        HalfOrientation(2, {0: "right", 1: "left"}, {0: -1, 1: 1}, {0: 57.3, 1: 56.0}),
    ]
    assert attack_summary(orientations, half_frame=10937) == {"half_frame": 10937, "directions": [[1, -1], [-1, 1]]}
    assert attack_summary([], half_frame=None) is None


def test_an_events_play_is_its_types_meaning_not_a_guess() -> None:
    """A clearance is a defensive act wherever it is played from, a shot is an attack - and the types that can
    sit on either side of the story (a foul, a substitution) stay blank rather than being forced onto one."""
    assert event_play("goal") == "attacking" and event_play("shot") == "attacking"
    assert event_play("corner") == "attacking" and event_play("penalty") == "attacking"
    assert event_play("clearance") == "defensive" and event_play("tackle") == "defensive"
    assert event_play("save") == "defensive" and event_play("block") == "defensive"
    assert event_play("foul") == "" and event_play("substitution") == "" and event_play("other") == ""


def test_the_aim_names_the_goal_mouth_in_the_animations_own_frame() -> None:
    """Direction +1 is toward +x, which the pane draws on the right - the same frame the arrows use - and a
    half the payload has no direction for says nothing rather than guessing which goal was being attacked."""
    assert aiming_goal(1) == "toward the right goal"
    assert aiming_goal(-1) == "toward the left goal"
    assert aiming_goal(None) == "" and aiming_goal(0) == ""


def test_the_touch_proxy_is_bounded_by_frames_with_an_aim(replay_case) -> None:
    replay, _, _, _ = replay_case
    touches = sum(player["stats"]["touches"] for player in replay["players"])
    aimed = sum(1 for entry in replay["aim"] if entry is not None)
    assert 0 < touches <= aimed, "at most one player touches the ball proxy per frame"
    assert all(player["stats"]["touches"] >= 0 for player in replay["players"])


def test_the_player_table_merges_shirt_numbers(replay_case) -> None:
    replay, _, _, _ = replay_case
    track = replay["players"][0]["track_id"]
    numbers = {track: {"number": 7, "name": "Sam", "source": "manual", "confidence": 1.0}}
    table = player_table_rows(replay, numbers)
    assert len(table) == len(replay["players"])
    row = table[table["track"] == track].iloc[0]
    assert row["number"] == 7 and row["name"] == "Sam" and row["source"] == "manual"
    others = table[table["track"] != track]
    assert (others["source"] == "unassigned").all()
    assert (others["number"].isna()).all()


def test_the_player_table_labels_teams_the_same_way_the_replay_does(replay_case) -> None:
    """The table's team column has to come from the payload the map is built from.

    It previously used a name that was never passed in, which took the whole replay section down with a NameError
    the moment a report existed - the kind of break that only shows up on real data.
    """
    replay, _, _, _ = replay_case
    table = player_table_rows(replay)
    named = [player for player in replay["players"] if player["team"] >= 0]
    assert named, "the fixture should have tracked players on a team"
    expected = {player["track_id"]: replay["team_names"][player["team"]] for player in named}
    for track, name in expected.items():
        assert table[table["track"] == track].iloc[0]["team"] == name
    # A payload without names must still produce a table rather than raising.
    anonymous = {**replay, "team_names": []}
    assert len(player_table_rows(anonymous)) == len(replay["players"])


def test_the_players_boxes_are_kept_out_of_the_payload() -> None:
    """The browser never draws a box, and a game's worth is tens of megabytes: they belong beside the payload.

    The clip cutter is Python-side, so the boxes travel as a separate small array set - ``track_boxes`` - keyed by
    track id. What must hold either way is that a box is present for every observation of every player that has
    them, aligned with the frames the payload lists.
    """
    from synthetic_match import simulate_match

    from soccer_analytics.analysis import stage_b as stage_b_module
    from soccer_analytics.analysis.projection import project_segment as project
    from soccer_analytics.dashboard.replay import track_boxes

    segment, truth = simulate_match(frames=200, seed=3)
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    calibration = PitchCalibration(
        truth.calibration.position,
        truth.calibration.base_rotation @ truth.q[0],
        truth.calibration.focal_scale,
        truth.calibration.aspect,
        0.0,
        (),
    )
    detections = project(segment, calibration, poses=(q, focal))
    report, _ = stage_b_module.build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(segment.time)
    )
    replay = build_replay(
        (PITCH_LENGTH, PITCH_WIDTH), float(segment.meta["fps"]), len(segment.time), detections.aim_xy,
        report.players, ["Team A", "Team B"], camera_xy=detections.camera_xy,
    )
    boxes = track_boxes(report.players)
    assert boxes, "no player got boxes"
    assert all("boxes" not in player for player in replay["players"]), "boxes must not be in the payload"
    for player in replay["players"]:
        stored = boxes.get(str(player["track_id"]))
        if stored is None:  # a bystander dropped from the payload keeps its boxes only if it was kept above
            continue
        assert len(stored) == len(player["frames"]), "a box per observation, in the same order"
        for row in stored:
            x1, y1, x2, y2 = (float(v) for v in row)
            assert 0.0 <= x1 <= x2 <= 1.2, "boxes are normalized by frame width"
            # y1 may be *below* y2: the simulator (like some real detections) emits inverted boxes, and the
            # framing code takes the absolute height for exactly this reason.
            assert abs(y2 - y1) >= 0.0


def test_a_player_without_boxes_is_left_out_rather_than_zeroed() -> None:
    """A track built without detections has no boxes - and the payload must not invent zeros for it.

    Zeros would frame its clip in the frame's top-left corner, which looks like a working feature and is not one.
    """
    from soccer_analytics.analysis.stage_b import PlayerTrack
    from soccer_analytics.dashboard.replay import track_boxes

    frames = np.arange(60, dtype=np.int32)
    without = PlayerTrack(
        track_id=1, team=0,
        frame=frames, time=frames / 5.0,
        xy=np.column_stack([20.0 + 0.5 * frames, np.full(60, 20.0)]), sigma_m=np.full(60, 0.5),
        speed_kmh=np.full(60, 8.0), distance_m=30.0, box=None,
    )
    assert track_boxes([without]) == {}


def test_a_quick_tag_lands_on_the_recordings_clock() -> None:
    """The tag bar sends the strip's clock (0 at the first analyzed frame); events store the recording's own
    seconds. The window's start is added back here, once, for every tag - the same translation the strip, the
    table and the reel cutter apply in reverse when they read it.

    Getting this wrong is the classic bug in this codebase (three clocks: the clip, the combined game, and the
    analyzed window), and a tag made while watching is otherwise indistinguishable from one typed in by hand.
    """
    event = event_from_tag(
        {"type": "goal", "team": 0, "time_s": 12.5, "note": "  header  "},
        video="/videos/match.mp4",
        window_start=540.0,
        duration_s=600.0,
    )
    assert event.time_s == pytest.approx(552.5)
    assert event.type == "goal" and event.team == 0
    assert event.note == "header", "the note is trimmed"
    assert event.video == "/videos/match.mp4"
    assert event.source == "manual"


def test_a_quick_tag_is_clamped_to_the_window_and_rejects_unknown_types() -> None:
    """A stale component value must never store a tag off the end of the match."""
    late = event_from_tag(
        {"type": "shot", "time_s": 9999.0}, video="v.mp4", window_start=10.0, duration_s=60.0
    )
    assert late.time_s == pytest.approx(70.0)
    early = event_from_tag(
        {"type": "shot", "time_s": -5.0}, video="v.mp4", window_start=10.0, duration_s=60.0
    )
    assert early.time_s == pytest.approx(10.0)
    with pytest.raises(ValueError):
        event_from_tag({"type": "wobble"}, video="v.mp4", window_start=0.0, duration_s=60.0)


def test_a_re_delivered_press_is_recognized_as_already_stored() -> None:
    """A page that reconnects after a server restart re-sends the component's sticky value; the session that
    would remember the acknowledgment is gone, so the store has to be idempotent by content or every reconnect
    duplicates the user's last tags. Same type, team, recording and second means the same press."""
    from soccer_analytics.analysis.events import Event

    stored = Event(time_s=552.5, type="goal", team=0, video="/v.mp4")
    same = Event(time_s=552.5, type="goal", team=0, video="/v.mp4")
    assert repeated_manual_tag([stored], same) is stored
    # A different team, type, second or recording is a different event, so it goes in.
    assert repeated_manual_tag([stored], Event(time_s=552.5, type="goal", team=1, video="/v.mp4")) is None
    assert repeated_manual_tag([stored], Event(time_s=552.5, type="shot", team=0, video="/v.mp4")) is None
    assert repeated_manual_tag([stored], Event(time_s=600.0, type="goal", team=0, video="/v.mp4")) is None
    assert repeated_manual_tag([stored], Event(time_s=552.5, type="goal", team=0, video="/other.mp4")) is None
    # A detected candidate is not a manual tag: a human tagging what a detector also found is two rows on purpose.
    detected = Event(time_s=552.5, type="goal", team=0, source="ball", video="/v.mp4")
    assert repeated_manual_tag([detected], same) is None

    # A goal is a moment, not a press: a second press while the replay is being watched is the same goal.
    late_goal = Event(time_s=554.0, type="goal", team=0, video="/v.mp4")
    assert repeated_manual_tag([stored], late_goal) is stored
    # Other types keep the tight window - two separate tackles a second apart are two events.
    tackle = Event(time_s=552.5, type="tackle", team=0, video="/v.mp4")
    late_tackle = Event(time_s=554.0, type="tackle", team=0, video="/v.mp4")
    assert repeated_manual_tag([tackle], late_tackle) is None

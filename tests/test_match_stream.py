"""The annotated match stream: what each frame draws, what each endpoint serves, and what is refused.

Built on the simulated match, which is the only place where tracks, boxes and a calibration all arrive with known
ground truth - the same fixture family the replay tests use. The HTTP surface is exercised against the real
server with a fake frame reader, so the multipart wire format and the routes are tested without an ffmpeg decode.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis import stage_b
from soccer_analytics.analysis.jerseys import merge_numbers
from soccer_analytics.analysis.library import MatchLibrary, MatchRecord
from soccer_analytics.analysis.projection import project_segment, segment_poses
from soccer_analytics.dashboard.replay import build_replay, track_boxes
from soccer_analytics.dashboard.stream import (
    _segment_for,
    AnnotatedMatch,
    MatchStreamServer,
    OVERLAY_NAMES,
    ROLE_BGR,
    SCOREBOARD_BGR,
    StreamError,
    TeamStyle,
    ball_stamps,
    chip_text,
    configured_base,
    configured_port,
    is_reachable,
    iter_annotated_frames,
    load_numbers,
    parse_overlays,
    streamable_matches,
)
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


@pytest.fixture(scope="module")
def stream_case():
    """One simulated match, tracked, with its replay and calibration - everything the stream draws from."""
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
    return segment, calibration, q, focal, replay, track_boxes(report.players)


def _annotated(
    stream_case,
    *,
    numbers=None,
    numbers_note=None,
    ball_records=(),
    notes=(),
    players=None,
    boxes=None,
    start_s=None,
    goals=None,
    attack=None,
    attack_path=None,
    whistle_times=(),
    half_source_s=None,
) -> AnnotatedMatch:
    segment, calibration, q, focal, replay, track_boxes_all = stream_case
    return AnnotatedMatch(
        match_id="synthetic",
        video=Path("/nonexistent/synthetic.mp4"),
        fps=float(replay["fps"]),
        start_s=float(segment.meta.get("start_s", 0.0)) if start_s is None else float(start_s),
        frame_count=int(replay["frame_count"]),
        pitch=(PITCH_LENGTH, PITCH_WIDTH),
        calibration=calibration,
        q=q,
        focal=focal,
        players=replay["players"] if players is None else players,
        boxes=track_boxes_all if boxes is None else boxes,
        numbers=numbers or {},
        numbers_note=numbers_note,
        ball_records=list(ball_records),
        teams=[TeamStyle(name="Team A", bgr=(12, 34, 56)), TeamStyle(name="Team B", bgr=(200, 30, 30))],
        notes=list(notes),
        attack=attack,
        attack_path=attack_path,
        whistle_times=list(whistle_times),
        half_source_s=half_source_s,
        goals=goals,
    )


# --------------------------------------------------------------------------------------------------------------
# The scoreboard
# --------------------------------------------------------------------------------------------------------------
def test_the_scoreboard_counts_confirmed_goals_on_the_match_clock(stream_case) -> None:
    match = _annotated(stream_case, goals=[(0.0, 0), (5.0, 1)])
    assert match.score_at(0) in ((1, 0), (0, 0))  # a goal at the very start may sit before/at frame 0
    just_before = match.index_for(match.start_s + 4.9)
    at_second = match.index_for(match.start_s + 5.1)
    assert match.score_at(just_before) == (1, 0)
    assert match.score_at(at_second) == (1, 1)


def test_the_scoreboard_draws_a_panel_without_any_other_layer(stream_case) -> None:
    match = _annotated(stream_case, goals=[(5.0, 0)])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    match.render(
        frame,
        0,
        pitch=False,
        boxes=False,
        numbers=False,
        ball=False,
        hud=False,
        debug=False,
        scoreboard=True,
    )
    assert frame.any(), "the scoreboard drew nothing"


def test_the_scoreboard_moves_to_the_bottom_left_and_hides_the_burned_in_timer(stream_case) -> None:
    """The camera's own wall clock lives in the lower-left; the scoreboard now covers it, not the top centre."""
    match = _annotated(stream_case, goals=[(5.0, 0)])
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    timer = (8, 680, 96, 708)  # inside BURNED_TIMER_BOX at this frame size
    frame[timer[1] : timer[3], timer[0] : timer[2]] = (0, 0, 255)
    match.render(
        frame,
        0,
        pitch=False,
        boxes=False,
        numbers=False,
        ball=False,
        hud=False,
        debug=False,
        scoreboard=True,
    )
    assert not np.any(np.all(frame[timer[1] : timer[3], timer[0] : timer[2]] == (0, 0, 255), axis=2)), (
        "the burned-in timer must be behind the opaque patch"
    )
    assert np.all(frame[:80, :] == 255), "the scoreboard must no longer occupy the top centre"


def test_the_scoreboard_stacks_the_teams_into_a_narrow_two_row_panel(stream_case) -> None:
    """Stacking the teams is what buys the space: the panel must stay narrow and grow taller, not wider."""
    match = _annotated(stream_case, goals=[(5.0, 0)])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    match.render(
        frame,
        0,
        pitch=False,
        boxes=False,
        numbers=False,
        ball=False,
        hud=False,
        debug=False,
        scoreboard=True,
    )
    dark = np.all(frame == SCOREBOARD_BGR, axis=2)
    ys, xs = dark.nonzero()
    assert xs.size and ys.size
    assert ys.min() > 600, "the panel must stay in the lower-left corner"
    assert xs.max() < 700, "the stacked panel should use well under half the frame width"
    assert ys.max() - ys.min() > 40, "two stacked rows need more than a single line of height"


def test_the_scoreboard_switches_attack_arrows_and_half_label_at_the_half_frame(stream_case) -> None:
    """The arrows follow the replay's per-period directions, and the clock says which half it is."""
    match = _annotated(
        stream_case, attack={"half_frame": 5, "directions": [[1, -1], [-1, 1]]}
    )
    assert match.attack_directions(0) == [1, -1]
    assert match.attack_directions(5) == [-1, 1]
    assert match.half_label(0) == "1H"
    assert match.half_label(5) == "2H"
    plain = _annotated(stream_case)
    assert plain.attack_directions(0) == [] and plain.half_label(0) == ""


def test_the_half_label_flips_at_the_first_frame_at_or_after_the_mark(stream_case) -> None:
    """The mark rarely lands exactly on a frame; the label must not show 1H for that one frame."""
    probe = _annotated(stream_case, start_s=100.0)
    match = _annotated(stream_case, start_s=100.0, half_source_s=100.0 + 5.2 / probe.fps)
    assert match.half_label(4) == "1H"
    assert match.half_label(5) == "2H"


def test_the_scoreboard_whistle_indicator_lights_near_a_whistle(stream_case) -> None:
    match = _annotated(stream_case)
    match.whistles = [match.start_s + 2.0]
    assert match.whistle_active(match.index_for(match.start_s + 2.1))
    assert not match.whistle_active(match.index_for(match.start_s + 10.0))


def test_the_scoreboard_rereads_swapped_attack_directions_from_the_sidecar(stream_case, tmp_path) -> None:
    """Swapping in the dashboard rewrites the small sidecar; a running session must pick up the new ends."""
    sidecar = tmp_path / "attack.json"
    sidecar.write_text(json.dumps({"attack": {"half_frame": 5, "directions": [[1, -1], [-1, 1]]}}))
    os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
    match = _annotated(
        stream_case,
        attack={"half_frame": 5, "directions": [[1, -1], [-1, 1]]},
        attack_path=sidecar,
    )
    assert match.attack_directions(0) == [1, -1]

    sidecar.write_text(json.dumps({"attack": {"half_frame": 5, "directions": [[-1, 1], [1, -1]]}}))
    os.utime(sidecar, ns=(2_000_000_000, 2_000_000_000))
    match.refresh_attack_if_changed(force=True)
    assert match.attack_directions(0) == [-1, 1]
    assert match.attack_directions(6) == [1, -1]


def test_the_scoreboard_draws_the_new_layers_without_error(stream_case) -> None:
    match = _annotated(
        stream_case,
        attack={"half_frame": 5, "directions": [[1, -1], [-1, 1]]},
        whistle_times=[2.0],
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    match.render(
        frame,
        2,
        pitch=False,
        boxes=False,
        numbers=False,
        ball=False,
        hud=False,
        debug=False,
        scoreboard=True,
    )
    assert frame.any()


# --------------------------------------------------------------------------------------------------------------
# The label chip
# --------------------------------------------------------------------------------------------------------------
def test_a_known_shirt_number_is_the_headline_and_the_track_stays_visible() -> None:
    """A person watching calls the player by their number; the tables and clips key them by track id - the
    number is the headline either way, and the track id joins it only in the debug layer."""
    assert chip_text(73, "Smith", 12) == ("73 Smith", ""), "a viewer does not need the tracker's key"
    assert chip_text(73, "Smith", 12, debug=True) == ("73 Smith", "track 12")
    assert chip_text(9, "", 12, debug=True) == ("9", "track 12")


def test_without_a_number_the_track_id_can_never_look_like_a_worn_number() -> None:
    """``#12`` is a tracking identity, not a shirt - the hash keeps the two apart, and the whole thing is debug
    information: without the debug layer an unidentified, unnamed player gets no chip at all."""
    assert chip_text(None, "", 12, debug=True) == ("#12", "")
    assert chip_text(None, "", 12) == ("", ""), "debug off: the box alone is the honest label"
    assert chip_text(None, "Smith", 12) == ("Smith", ""), "a name is not read off a shirt: it needs no debug"
    assert chip_text(None, "Smith", 12, debug=True) == ("Smith", "track 12")


# --------------------------------------------------------------------------------------------------------------
# The ball stamps
# --------------------------------------------------------------------------------------------------------------
def test_only_a_sighting_carries_a_position_and_a_forecast_never_does() -> None:
    """``tracking`` is a sighting; ``coasting`` is the tracker's forecast across a miss, and the stream draws
    the ball only where a detector saw it - so a coast is not carried at all, and neither are the statuses that
    have no position."""
    records = [
        {"i": 0, "status": "tracking", "u": 0.5, "v": 0.3},
        {"i": 1, "status": "coasting", "u": 0.55, "v": 0.31},
        {"i": 2, "status": "lost", "u": None, "v": None},
        {"i": 3, "status": "out_of_view", "u": 0.9, "v": 0.4},
        {"i": 99, "status": "tracking", "u": 0.1, "v": 0.1},  # outside the window
    ]
    stamps = ball_stamps(records, 5)
    assert stamps[0, 2] == 1.0 and np.allclose(stamps[0, :2], (0.5, 0.3))
    assert np.isnan(stamps[1]).all(), "a forecast is not a sighting: nothing to draw"
    assert np.isnan(stamps[2]).all() and np.isnan(stamps[3]).all()
    assert np.isnan(stamps[4]).all()


def test_the_overlays_parameter_selects_layers_with_absence_meaning_all() -> None:
    """Absent must keep meaning "everything on": every existing URL (the index page's stills, old links) relies
    on it, and ``parse_qs`` cannot distinguish an empty value from an absent one, so "none" is the explicit
    spelling for a clean frame."""
    assert parse_overlays(None) == dict.fromkeys(OVERLAY_NAMES, True)
    assert parse_overlays("none") == dict.fromkeys(OVERLAY_NAMES, False)
    assert parse_overlays("boxes,ball") == {
        "pitch": False,
        "boxes": True,
        "numbers": False,
        "ball": True,
        "hud": False,
        "debug": False,
    }
    assert parse_overlays("debug") == {**dict.fromkeys(OVERLAY_NAMES, False), "debug": True}
    assert parse_overlays(" boxes , BALL ")["boxes"] and parse_overlays(" boxes , BALL ")["ball"]
    assert parse_overlays("boxes,what")["boxes"], "unknown names are ignored, not fatal"


# --------------------------------------------------------------------------------------------------------------
# Building the by-frame index
# --------------------------------------------------------------------------------------------------------------
def test_every_box_of_every_player_is_indexed_under_its_own_frame(stream_case) -> None:
    """The stream walks frames, not players: each frame must find exactly the observations recorded on it, in
    the sidecar's own order (the box arrays align with the payload's ``frames``)."""
    _segment, _calibration, _q, _focal, replay, boxes = stream_case
    match = _annotated(stream_case)
    checked = 0
    for player in replay["players"]:
        stored = boxes.get(str(player["track_id"]))
        if stored is None or len(stored) == 0:
            continue
        frame = int(player["frames"][0])
        stamp = next(s for s in match.players_by_frame[frame] if s[4] == player["track_id"])
        assert np.allclose(stamp[:4], stored[0])
        checked += 1
        if checked >= 5:
            break
    assert checked == 5


def test_the_identities_map_follows_the_number_assignment(stream_case) -> None:
    _segment, *_rest, replay, _boxes = stream_case
    track = int(replay["players"][0]["track_id"])
    numbers = merge_numbers([track], manual={track: {"number": 7, "name": "Smith"}})
    match = _annotated(stream_case, numbers=numbers)
    assert match.identities[track] == ("7 Smith", ""), "the watching view: the number and the name"
    assert match.debug_identities[track] == ("7 Smith", f"track {track}"), "the debug view adds the track id"


def test_the_displayed_number_comes_from_the_jersey_scan_not_the_roster() -> None:
    """A number drawn on the footage is a claim about what the camera saw: only a jersey detection supplies one.
    A manual roster entry still names the player - names are not read off a shirt - but its number is not
    evidence, so it must not end up on a chip the scan never backed."""

    class _Library:
        def load_jerseys(self, _match_id):
            return {"suggestions": {"7": {"number": 9}}}

        def load_roster(self, _match_id):
            return {7: {"number": 99, "name": "Smith"}, 8: {"name": "Jones"}}

    numbers, _jerseys = load_numbers(_Library(), "m", [7, 8, 9])
    assert numbers[7] == {"number": 9, "name": "Smith", "source": "scan"}, "the detection wins over the roster"
    assert numbers[8] == {"number": None, "name": "Jones", "source": "roster"}, "a name without a number is fine"
    assert 9 not in numbers, "neither scan nor roster said anything about this track"


def test_a_team_roster_names_the_player_behind_a_scanned_number() -> None:
    """The number is a detection; the name is squad knowledge from the linked team roster (number -> name),
    which is what lets one roster typed for a team name the tracks of every match that links it."""

    class _Library:
        def load_jerseys(self, _match_id):
            return {"suggestions": {"7": {"number": 9}}}

        def load_roster(self, _match_id):
            return {}

    numbers, _jerseys = load_numbers(_Library(), "m", [7], teams={7: 0}, team_rosters={0: {9: "Smith"}})
    assert numbers[7] == {"number": 9, "name": "Smith", "source": "scan"}
    # The roster belongs to team 0; the same number on team 1 stays unnamed.
    other, _jerseys = load_numbers(_Library(), "m", [7], teams={7: 1}, team_rosters={0: {9: "Smith"}})
    assert other[7] == {"number": 9, "name": "", "source": "scan"}


def test_a_manual_name_still_beats_the_team_roster() -> None:
    class _Library:
        def load_jerseys(self, _match_id):
            return {"suggestions": {"7": {"number": 9}}}

        def load_roster(self, _match_id):
            return {7: {"name": "Jones"}}

    numbers, _jerseys = load_numbers(_Library(), "m", [7], teams={7: 0}, team_rosters={0: {9: "Smith"}})
    assert numbers[7] == {"number": 9, "name": "Jones", "source": "scan"}


def test_a_live_refresh_moves_the_chips_to_the_new_numbers(stream_case) -> None:
    """The dashboard edits the roster *beside* the stream; a chip showing a number the user just removed would
    read as the stream ignoring them, so the labels - and the staleness note about them - are rebuilt in place."""
    _segment, *_rest, replay, _boxes = stream_case
    track = int(replay["players"][0]["track_id"])
    match = _annotated(stream_case, notes=["static note"])
    assert match.identities[track] == ("", ""), "unknown and no debug: no chip"
    assert match.debug_identities[track] == (f"#{track}", "")
    assert match.notes == ["static note"]
    match.refresh_numbers({track: {"number": 7, "name": "Smith"}}, None)
    assert match.identities[track] == ("7 Smith", "")
    assert match.debug_identities[track] == ("7 Smith", f"track {track}")
    assert match.notes == ["static note"], "trusted numbers add no note"
    match.refresh_numbers({}, "shirt numbers look stale: made up for the test")
    assert match.identities[track] == ("", "")
    assert match.notes == ["static note", "shirt numbers look stale: made up for the test"]


# --------------------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------------------
def test_the_numbers_note_reads_the_live_files(tmp_path) -> None:
    """Shirt numbers attach to track ids, which a rebuild or a re-fit can move: the note the stream shows must
    come from the files actually on disk (the scan's saved fit against the current calibration), not a snapshot."""
    from soccer_analytics.dashboard.stream import numbers_note

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    mtime = calibration.stat().st_mtime
    jerseys = {"suggestions": {"7": {"number": 7}}, "meta": {"calibration_saved": mtime}}
    numbers = {7: {"number": 7}}
    assert numbers_note(tmp_path, jerseys, numbers, [7, 9]) is None, "scan and calibration agree"
    assert "no shirt numbers" in (numbers_note(tmp_path, jerseys, {}, [7]) or "")
    jerseys["meta"]["calibration_saved"] = mtime - 1000.0
    note = numbers_note(tmp_path, jerseys, numbers, [7])
    assert note and "stale" in note, "a scan read before the last calibration is flagged"
    jerseys["meta"]["calibration_saved"] = mtime
    note = numbers_note(tmp_path, jerseys, numbers, [9])
    assert note and "not in this report" in note, "a suggestion id that no longer exists is stale too"


def test_the_pitch_model_is_drawn_through_the_calibration(stream_case) -> None:
    """Rendering only the pitch markings must still change the picture when the camera is looking at the field -
    the overlay is the projection, not a fixed decal."""
    segment, calibration, q, focal, replay, _boxes = stream_case
    match = _annotated(stream_case)
    index = len(segment.time) // 2
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    before = frame.copy()
    match.render(frame, index, pitch=True, boxes=False, numbers=False, ball=False, hud=False, debug=False)
    assert np.count_nonzero(np.any(frame != before, axis=2)) > 0, "no marking landed in the frame"


def test_each_player_box_lands_on_its_recorded_pixels_in_the_team_color(stream_case) -> None:
    """Rendered on its own so nothing can overlap the assertion: the rectangle must sit on the box's own pixels
    (in stream pixels - normalized by width, both axes) and be painted in the player's team color."""
    _segment, _calibration, _q, _focal, replay, boxes = stream_case
    player = next(p for p in replay["players"] if boxes.get(str(p["track_id"])) is not None)
    stored = boxes[str(player["track_id"])]
    index = int(player["frames"][0])
    match = _annotated(stream_case, players=[player])
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(frame, index, pitch=False, boxes=True, numbers=False, ball=False, hud=False, debug=False)
    x1, y1, x2, y2 = (float(v) for v in stored[0])
    left, right = sorted((round(x1 * 640), round(x2 * 640)))
    top, bottom = sorted((round(y1 * 640), round(y2 * 640)))
    expected = np.array(match.teams[int(player["team"])].bgr)
    painted = np.all(frame[top : bottom + 1, left : right + 1] == expected, axis=-1)
    # The rectangle's own perimeter at this size is ~4*(w+h) pixels; the chip (same color) can only add more.
    assert painted.sum() >= 3 * ((right - left) + (bottom - top))


def test_officials_are_color_coded_by_role_not_by_side(stream_case) -> None:
    """A goalkeeper draws in the role color wherever the team color would have put it, and the HUD names
    that color only when such a player is on screen - an unnamed color key is decoration.

    The role rides the replay payload, where the pipeline classified it from goal-pocket behavior (kit color
    was too noisy to trust); a payload from before roles existed draws exactly the team colors it always did.
    """
    _segment, _calibration, _q, _focal, replay, boxes = stream_case
    player = next(p for p in replay["players"] if boxes.get(str(p["track_id"])) is not None)
    x1, y1, x2, y2 = (float(v) for v in boxes[str(player["track_id"])][0])
    index = int(player["frames"][0])
    left, right = sorted((round(x1 * 640), round(x2 * 640)))
    top, bottom = sorted((round(y1 * 640), round(y2 * 640)))
    keeper = dict(player, role="goalkeeper")
    match = _annotated(stream_case, players=[keeper])
    assert match.roles == {int(player["track_id"]): "goalkeeper"}, "the payload's role is what is read"
    role_bgr = tuple(int(v) for v in ROLE_BGR["goalkeeper"])

    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(frame, index, pitch=False, boxes=True, numbers=False, ball=False, hud=False, debug=False)
    assert tuple(int(v) for v in frame[(top + bottom) // 2, left]) == role_bgr, "the box wears the role color"

    keyed = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(keyed, index, pitch=False, boxes=False, numbers=False, ball=False, hud=True, debug=False)
    assert np.count_nonzero(np.all(keyed == np.array(role_bgr), axis=-1)) > 0, "the HUD keys the color"

    plain = _annotated(stream_case, players=[player])
    bare = np.full((360, 640, 3), 60, dtype=np.uint8)
    plain.render(bare, index, pitch=False, boxes=True, numbers=False, ball=False, hud=True, debug=False)
    assert np.count_nonzero(np.all(bare == np.array(role_bgr), axis=-1)) == 0, "no role in the payload: none drawn"


def test_the_box_and_number_layers_switch_independently(stream_case) -> None:
    """Rendered on a single player so nothing overlaps: the rectangle is one layer and the chip is another -
    boxes without numbers, numbers without boxes, and both must all be honest pictures of the same data."""
    _segment, _calibration, _q, _focal, replay, boxes = stream_case
    player = next(p for p in replay["players"] if boxes.get(str(p["track_id"])) is not None)
    x1, y1, x2, y2 = (float(v) for v in boxes[str(player["track_id"])][0])
    index = int(player["frames"][0])
    match = _annotated(
        stream_case, players=[player], numbers={int(player["track_id"]): {"number": 9}}
    )
    left, right = sorted((round(x1 * 640), round(x2 * 640)))
    top, bottom = sorted((round(y1 * 640), round(y2 * 640)))
    expected = match.teams[int(player["team"])].bgr
    blank = np.full((360, 640, 3), 60, dtype=np.uint8)

    def render(**layers):
        frame = blank.copy()
        match.render(frame, index, pitch=False, ball=False, hud=False, debug=False, **layers)
        return np.count_nonzero(np.all(frame == expected, axis=-1)), frame

    box_only, box_frame = render(boxes=True, numbers=False)
    both, _ = render(boxes=True, numbers=True)
    chips_only, chip_frame = render(boxes=False, numbers=True)
    assert tuple(int(v) for v in box_frame[(top + bottom) // 2, left]) == expected, "the box edge is drawn"
    assert tuple(int(v) for v in chip_frame[(top + bottom) // 2, left]) == (60, 60, 60), "without the box, no edge"
    assert both > box_only > 0, "the chip adds a filled block on top of the rectangle"
    assert chips_only > 40, "the chip survives with the box switched off"


def test_turning_every_layer_off_leaves_the_raw_frame(stream_case) -> None:
    """All toggles off means untouched footage - the one combination a viewer can use to check what the camera
    actually recorded, with nothing of ours drawn on it."""
    segment, *_rest = stream_case
    match = _annotated(stream_case)
    frame = np.full((360, 640, 3), 77, dtype=np.uint8)
    original = frame.copy()
    match.render(frame, len(segment.time) // 2, **{name: False for name in OVERLAY_NAMES})
    assert np.array_equal(frame, original)


def test_the_debug_layer_carries_the_diagnostics_and_switches_off_alone(stream_case) -> None:
    """The session name, the frame counter and the notes are the diagnosing half; the clock and the team legend
    are the watching half. Turning debug off must remove exactly the three diagnostics - a viewer who wants an
    uncluttered picture keeps their clock - and the two halves must not draw over each other (the legend steps
    down a line when the session name is above it).
    """
    segment, *_rest = stream_case
    match = _annotated(stream_case, notes=["999 off-field tracks hidden (coaches/spectators)"])
    index = len(segment.time) // 2
    blank = np.full((1080, 1920, 3), 60, dtype=np.uint8)

    def render(**layers):
        frame = blank.copy()
        match.render(frame, index, pitch=False, boxes=False, numbers=False, ball=False, **layers)
        return frame

    def changed(frame, y0, y1, x0, x1):
        return int(np.count_nonzero(np.any(frame[y0:y1, x0:x1] != blank[y0:y1, x0:x1], axis=2)))

    top_left = (0, 60, 0, 500)  # session name (debug) and the legend (hud)
    below_clock = (35, 58, 1200, 1920)  # the frame counter sits under the clock
    bottom_right = (1030, 1080, 900, 1920)  # the notes, right-aligned clear of the scoreboard

    hud_only = render(hud=True, debug=False)
    assert changed(hud_only, *top_left) > 0, "the team legend is the watching half's"
    assert changed(hud_only, *below_clock) == 0, "no frame counter without the debug layer"
    assert changed(hud_only, *bottom_right) == 0, "no notes without the debug layer"

    debug_only = render(hud=False, debug=True)
    assert changed(debug_only, *top_left) > 0, "the session name is the debug layer's"
    assert changed(debug_only, *below_clock) > 0, "the frame counter is the debug layer's"
    assert changed(debug_only, *bottom_right) > 0, "the notes are the debug layer's"


def test_the_hud_clock_reads_the_animation_clock_not_the_recording_clock(stream_case, monkeypatch) -> None:
    """The pane's animation counts from the analyzed window's start - kick-off on a marked build - and the HUD
    must show that same clock. The recording's own clock starts earlier, and a stamp in recording seconds reads
    minutes away from the animation the viewer compares it with."""
    drawn: list[str] = []
    monkeypatch.setattr(
        "soccer_analytics.dashboard.stream._outline_text",
        lambda frame, text, *args, **kwargs: drawn.append(text),
    )
    match = _annotated(stream_case, start_s=3.0)
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    layers = dict(pitch=False, boxes=False, numbers=False, ball=False, hud=True, debug=False)
    match.render(frame, int(10 * match.fps), **layers)
    assert "0:10" in drawn, "ten seconds into the window is 0:10 - the number the animation shows"
    assert "0:00:13" not in drawn, "13 s on the recording (window start + 10 s) is not the animation's clock"
    match.render(frame, 0, **layers)
    assert "0:00" in drawn, "the window's zero is the clock's zero"


def test_a_track_id_chip_is_debug_information(stream_case) -> None:
    """The tracker's id is how the tables key a player, not what a viewer calls them: with debug off, a player
    the scan has no number for draws no chip - the box alone says "a person" - and the id appears only when the
    debug layer asks for it."""
    _segment, _calibration, _q, _focal, replay, boxes = stream_case
    player = next(p for p in replay["players"] if boxes.get(str(p["track_id"])) is not None)
    index = int(player["frames"][0])
    match = _annotated(stream_case, players=[player])
    blank = np.full((360, 640, 3), 60, dtype=np.uint8)
    quiet = blank.copy()
    match.render(quiet, index, pitch=False, boxes=False, numbers=True, ball=False, hud=False, debug=False)
    assert np.array_equal(quiet, blank), "debug off and no number read: there is no chip to draw"
    corners = blank.copy()
    match.render(corners, index, pitch=False, boxes=False, numbers=False, ball=False, hud=False, debug=True)
    chipped = blank.copy()
    match.render(chipped, index, pitch=False, boxes=False, numbers=True, ball=False, hud=False, debug=True)
    assert np.count_nonzero(np.any(chipped != corners, axis=2)) > 0, "the debug layer shows the track id"


def test_a_seen_ball_is_marked_and_a_forecast_is_never_drawn(stream_case) -> None:
    """The mark on the picture is a sighting, or it is absent: a frame the scan only forecast the ball on must
    carry no mark at all - not even the hollow ring a forecast used to draw."""
    segment, _calibration, _q, _focal, _replay, _boxes = stream_case
    index = len(segment.time) // 2
    match = _annotated(stream_case)
    match.ball[index] = (0.5, 0.3, 1.0)
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(frame, index, pitch=False, boxes=False, numbers=False, ball=True, hud=False)
    spot = frame[int(0.3 * 640), 320]
    assert tuple(int(v) for v in spot) == (255, 255, 255), "a detected ball is a filled white dot"

    match.ball[index] = (0.5, 0.3, 0.0)
    forecast = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(forecast, index, pitch=False, boxes=False, numbers=False, ball=True, hud=False)
    around = forecast[int(0.3 * 640) - 7 : int(0.3 * 640) + 8, 320 - 7 : 320 + 8]
    assert np.all(around == 60), "a forecast is not a detection: not even a ring may be drawn for it"

    match.ball[index] = (np.nan, np.nan, np.nan)
    empty = np.full((360, 640, 3), 60, dtype=np.uint8)
    match.render(empty, index, pitch=False, boxes=False, numbers=False, ball=True, hud=False)
    center = empty[100:260, 200:440]
    assert np.all(center == 60), "nothing to draw means nothing drawn"


# --------------------------------------------------------------------------------------------------------------
# Frame iteration (the decode seam)
# --------------------------------------------------------------------------------------------------------------
class _FakeReader:
    """Yields solid frames on the analysis grid, standing in for the ffmpeg decode."""

    def __init__(self, path, *, fps, width, start_s, duration_s=None):
        self.start_s = float(start_s)
        self.fps = float(fps)
        self.count = min(6, max(1, int(round(float(duration_s or 1.0) * self.fps))))

    def frames(self):
        for k in range(self.count):
            yield self.start_s + k / self.fps, np.full((180, 320, 3), 90, dtype=np.uint8)


class _SlowFakeReader:
    """Many frames, slowly, so a test can stop the stream while it is still mid-flight."""

    def __init__(self, path, *, fps, width, start_s, duration_s=None):
        self.start_s = float(start_s)
        self.fps = float(fps)
        self.count = 200

    def frames(self):
        for k in range(self.count):
            time.sleep(0.02)
            yield self.start_s + k / self.fps, np.full((180, 320, 3), 90, dtype=np.uint8)


def test_iteration_decodes_from_the_requested_frame_and_yields_jpegs(stream_case) -> None:
    match = _annotated(stream_case)
    start = match.source_time(3)
    frames = list(iter_annotated_frames(match, start, rate=2, width=320, reader_factory=_FakeReader, pace=False))
    assert [index for index, _jpeg in frames][:3] == [3, 4, 5]
    assert all(jpeg.startswith(b"\xff\xd8") for _index, jpeg in frames), "each push is a JPEG"


def test_a_start_time_past_the_window_is_refused_with_the_end_in_the_message(stream_case) -> None:
    match = _annotated(stream_case)
    with pytest.raises(StreamError, match="past the end"):
        match.index_for(match.source_time(match.frame_count) + 5)


# --------------------------------------------------------------------------------------------------------------
# The archive listing and the load refusals
# --------------------------------------------------------------------------------------------------------------
def _write_match(root: Path, match_id: str, *, segment_dir: Path, with_replay: bool = True, boxes: bool = True) -> Path:
    library = MatchLibrary(root)
    library.save(
        MatchRecord(
            match_id=match_id, sources=["clip.mp4"], segments=[str(segment_dir)], team_names=["Reds", "Blues"]
        )
    )
    directory = library.path(match_id)
    (directory / "calibration.json").write_text(
        json.dumps(
            {
                "position": [30.0, -7.0, 4.5],
                "base_rotation": np.eye(3).tolist(),
                "focal_scale": 1.0,
                "aspect": 0.5625,
                "rms_error_m": 0.0,
                "residuals_m": [],
            }
        )
    )
    if with_replay:
        (directory / "replay.json").write_text(
            json.dumps(
                {
                    "pitch": [60.0, 40.0],
                    "fps": 5.0,
                    "frame_count": 10,
                    "team_names": ["Reds", "Blues"],
                    "team_colors": None,
                    "players": [{"track_id": 1, "team": 0, "frames": []}],
                }
            )
        )
        if boxes:  # the sidecar the clip cutter and the stream read, exactly as save_replay writes it
            np.savez_compressed(directory / "boxes.npz", **{"1": np.asarray([[0.1, 0.1, 0.2, 0.3]], dtype=np.float16)})
    (directory / "report.json").write_text(
        json.dumps({"teams": [{"team": 0, "kit_rgb": [200, 50, 50]}, {"team": 1, "kit_rgb": [50, 50, 200]}]})
    )
    return directory


def test_the_listing_needs_a_calibration_and_a_replay_and_carries_team_colors(tmp_path) -> None:
    segment_dir = tmp_path / "segments" / "s1"
    segment_dir.mkdir(parents=True)
    (segment_dir / "meta.json").write_text(
        json.dumps({"start_s": 10.0, "end_s": 100.0, "total_frames": 450, "fps": 5.0, "video": "v.mp4"})
    )
    root = tmp_path / "matches"
    _write_match(root, "2026-01-01_ok", segment_dir=segment_dir)
    _write_match(root, "2026-01-02_no_replay", segment_dir=segment_dir, with_replay=False)
    rows = {row["match_id"]: row for row in streamable_matches(root)}
    assert rows["2026-01-01_ok"]["streamable"] is True
    assert rows["2026-01-01_ok"]["team_names"] == ["Reds", "Blues"]
    assert rows["2026-01-01_ok"]["team_colors"][0] == [200, 50, 50]
    assert rows["2026-01-02_no_replay"]["streamable"] is False


def test_the_listing_serves_the_newest_usable_segment_not_a_stale_first_entry(tmp_path) -> None:
    """A moved directory can leave the record's first entry dangling; the run that is actually on disk must
    still make the match streamable - the same choice the stream itself makes."""
    root = tmp_path / "matches"
    fresh = tmp_path / "segments" / "fresh"
    fresh.mkdir(parents=True)
    (fresh / "meta.json").write_text(
        json.dumps({"start_s": 0.0, "end_s": 2.0, "total_frames": 10, "fps": 5.0, "video": "v.mp4"})
    )
    _write_match(root, "2026-01-03_moved", segment_dir=fresh)
    library = MatchLibrary(root)
    record = library.load("2026-01-03_moved")
    record.segments = [str(tmp_path / "segments" / "gone"), *record.segments]
    library.save(record)
    rows = {row["match_id"]: row for row in streamable_matches(root)}
    assert rows["2026-01-03_moved"]["streamable"] is True


def test_the_segment_served_is_the_one_the_replay_was_built_on(tmp_path) -> None:
    """The record lists every run, oldest first - and the ball scan lives beside one of them. Blindly taking the
    first entry is how a segment WITHOUT its ball scan silently got served; the choice is the newest run on the
    replay's own frame grid, and a replay built on an older window keeps its own segment."""
    old = tmp_path / "segments" / "old"
    new = tmp_path / "segments" / "new"
    for directory, frames in ((old, 450), (new, 10)):
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(
            json.dumps({"start_s": 0.0, "end_s": 2.0, "total_frames": frames, "fps": 5.0, "video": "v.mp4"})
        )
    record = MatchRecord(match_id="m", segments=[str(old), str(old), str(new)])  # the duplicate a re-add leaves

    replay = {"fps": 5.0, "frame_count": 10}
    assert _segment_for(record, replay) == new, "among the runs on the replay's grid, the newest wins"
    assert _segment_for(record) == new, "without a replay, the newest usable run wins"
    assert _segment_for(record, {"fps": 5.0, "frame_count": 450}) == old, (
        "a replay built on the old window keeps the segment it was built on"
    )
    gone = tmp_path / "segments" / "gone"
    assert _segment_for(MatchRecord(match_id="m", segments=[str(gone)])) == gone, (
        "a record with nothing usable still names its first entry - the load's error reports it"
    )


def test_a_replay_without_player_boxes_is_refused_with_the_rebuild_command(tmp_path) -> None:
    """Streaming boxes that are silently absent would be a lie of omission; the error names the fix instead."""
    segment_dir = tmp_path / "segments" / "s1"
    segment_dir.mkdir(parents=True)
    (segment_dir / "meta.json").write_text(json.dumps({"start_s": 0.0, "end_s": 1.0, "video": "v.mp4"}))
    root = tmp_path / "matches"
    _write_match(root, "2026-01-01_old", segment_dir=segment_dir, boxes=False)
    with pytest.raises(StreamError, match="rebuild_match"):
        AnnotatedMatch.load("2026-01-01_old", root=root)


# --------------------------------------------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------------------------------------------
def test_the_stream_port_and_base_can_be_configured(monkeypatch) -> None:
    """The dashboard and the components talk to this server on a configurable port and base URL, so a machine
    with 8510 taken (or a proxy in front) is a setting, not a code change."""
    monkeypatch.setenv("SOCCER_STREAM_PORT", "9001")
    monkeypatch.setenv("SOCCER_STREAM_BASE", "https://example.test/footage")
    assert configured_port() == 9001
    assert configured_base() == "https://example.test/footage"
    monkeypatch.setenv("SOCCER_STREAM_PORT", "not-a-port")
    monkeypatch.delenv("SOCCER_STREAM_BASE")
    assert configured_port() == 8510, "a bad value falls back rather than taking the page down"
    assert configured_base() == ""


def test_reachability_tracks_a_listening_socket(tmp_path) -> None:
    """What the dashboard's caption and start button hang on: something listening means connected, nothing
    listening means offer the start button - and neither answer may block the rerun."""
    assert is_reachable(9, timeout_s=0.05) is False, "nothing listens on the discard port"
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        assert is_reachable(port)
    finally:
        server.shutdown()
        server.server_close()
    assert is_reachable(port, timeout_s=0.05) is False


def test_a_stream_is_refused_with_503_while_every_slot_is_held(stream_case, tmp_path, monkeypatch) -> None:
    """Every viewer gets a decoder, so concurrency is capped; a request that cannot get a slot within the brief
    wait is refused with 503 (the pane retries it with backoff), and a freed slot serves the next request."""
    from soccer_analytics.dashboard import stream as stream_module

    match = _annotated(stream_case)
    monkeypatch.setattr(stream_module, "STREAM_SLOT_WAIT_S", 0.2)
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path, max_streams=1, reader_factory=_FakeReader, pace=False)
    server._sessions["synthetic"] = match
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        assert server.stream_slots.acquire(blocking=False), "take the only slot"
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", f"/stream/synthetic.mjpg?start={match.start_s:.1f}&width=320")
        assert connection.getresponse().status == 503
        server.stream_slots.release()
        connection.request("GET", f"/stream/synthetic.mjpg?start={match.start_s:.1f}&width=320")
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_a_client_can_stop_its_own_stream(stream_case, tmp_path) -> None:
    """The pane abandons its stream on every seek, toggle and pause - and browsers will not abort an MJPEG fetch
    on command - so a token lets the server end the stream itself, deterministically, instead of a decoder
    draining until the write watchdog happens to notice. Stopping is idempotent."""
    match = _annotated(stream_case)
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path, reader_factory=_SlowFakeReader, pace=False)
    server._sessions["synthetic"] = match
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", f"/stream/synthetic.mjpg?start={match.start_s:.1f}&width=320&token=abc123")
        response = connection.getresponse()
        assert response.status == 200
        first = response.read(20000)
        assert b"--frame" in first, "the stream must be flowing before it is stopped"
        stopper = http.client.HTTPConnection(host, port, timeout=10)
        stopper.request("GET", "/stop?token=abc123")
        stop_response = stopper.getresponse()
        assert stop_response.status == 200
        assert stop_response.getheader("Access-Control-Allow-Origin") == "*", (
            "the pane fetches the stop from the dashboard's origin: without the header the browser blocks it"
        )
        stopper.close()
        rest = response.read()  # ends because the server closed the connection, not because frames ran out
        frames = (first + rest).count(b"--frame")
        assert frames < 200, f"the stream ran past the stop: {frames} frames"
        connection.close()
        # A stop that races the stream's own end (or names a token that never existed) is not an error.
        stopper = http.client.HTTPConnection(host, port, timeout=10)
        stopper.request("GET", "/stop?token=abc123")
        assert stopper.getresponse().status == 200
        stopper.close()
    finally:
        server.shutdown()
        server.server_close()


def test_a_still_request_rereads_swapped_directions_before_drawing(stream_case, tmp_path, monkeypatch) -> None:
    """The scoreboard is drawn server-side; a swap saved since the session loaded must be in the very next still."""
    from soccer_analytics.dashboard import stream as stream_module

    sidecar = tmp_path / "attack.json"
    sidecar.write_text(json.dumps({"attack": {"half_frame": 5, "directions": [[1, -1], [-1, 1]]}}))
    os.utime(sidecar, ns=(1_000_000_000, 1_000_000_000))
    match = _annotated(
        stream_case,
        attack={"half_frame": 5, "directions": [[1, -1], [-1, 1]]},
        attack_path=sidecar,
    )
    monkeypatch.setattr(
        stream_module, "grab_frame", lambda _path, _at, *, width=0: np.zeros((180, 320, 3), dtype=np.uint8)
    )
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path, reader_factory=_FakeReader, pace=False)
    server._sessions["synthetic"] = match
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        url = f"/frame/synthetic.jpg?t={match.start_s + 1.0:.1f}&width=320&scoreboard=1"
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", url)
        first = connection.getresponse().read()
        assert first.startswith(b"\xff\xd8"), "the still is a JPEG"

        sidecar.write_text(json.dumps({"attack": {"half_frame": 5, "directions": [[-1, 1], [1, -1]]}}))
        os.utime(sidecar, ns=(2_000_000_000, 2_000_000_000))
        connection.request("GET", url)
        second = connection.getresponse().read()
        assert second != first, "the scoreboard still showed the old ends after the swap"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_the_stream_endpoint_serves_multipart_mjpeg_and_the_listing_serves_json(stream_case, tmp_path) -> None:
    """The wire format is the point of the endpoint: browsers only play ``multipart/x-mixed-replace`` streams, and
    the parts must be real JPEGs. The fake reader keeps this independent of any footage on the machine."""
    match = _annotated(stream_case)
    server = MatchStreamServer(("127.0.0.1", 0), root=tmp_path, reader_factory=_FakeReader, pace=False)
    server._sessions["synthetic"] = match  # the overlay, injected: the decode is the only faked piece
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", f"/stream/synthetic.mjpg?start={match.start_s:.1f}&rate=2&width=320")
        response = connection.getresponse()
        assert response.status == 200
        assert "multipart/x-mixed-replace" in response.getheader("Content-Type")
        body = response.read()
        assert body.count(b"--frame") >= 3, "the multipart boundary frames the pushes"
        assert body.count(b"\xff\xd8") >= 3, "each part is a JPEG"

        connection.request("GET", "/matches")
        listing = connection.getresponse()
        assert listing.status == 200
        assert "matches" in json.loads(listing.read())

        # The play page embeds the stream in an <img> (browsers download a bare multipart URL instead of playing
        # it), and passes the requested speed through to the stream it points at.
        connection.request("GET", "/play/synthetic?start=1.0&rate=4")
        page = connection.getresponse()
        assert page.status == 200
        text = page.read().decode()
        assert "/stream/synthetic.mjpg?start=1&rate=4" in text
        connection.close()
    finally:
        server.shutdown()
        server.server_close()

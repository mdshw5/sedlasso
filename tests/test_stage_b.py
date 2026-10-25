"""Stage B tested against a simulated match with known truth: does it recover what it claims to?"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_analytics.analysis.ball_lock import BallImage
from soccer_analytics.analysis.projection import PitchDetections, project_segment, segment_poses
from soccer_analytics.analysis.stage_b import (
    _momentum,
    _possession_from_momentum,
    _team_assignment,
    _track_people,
    build_report,
)
from soccer_analytics.geometry.pitch_calibration import PitchCalibration

from synthetic_match import PITCH_LENGTH, PITCH_WIDTH, simulate_match


def _prepared(frames: int = 400, seed: int = 1, attack_bias: float = 0.0):
    segment, truth = simulate_match(frames=frames, seed=seed, attack_bias=attack_bias)
    q, focal = segment_poses(segment, focal0=float(truth.focal[0]))
    base = truth.calibration.base_rotation @ truth.q[0]
    cal = PitchCalibration(
        truth.calibration.position, base, truth.calibration.focal_scale, truth.calibration.aspect, 0.0, ()
    )
    detections = project_segment(segment, cal, poses=(q, focal))
    return segment, truth, detections


@pytest.fixture(scope="module")
def prepared():
    return _prepared()


def _track_purity(truth, detections, assignment) -> tuple[float, int]:
    """Mean fraction of each track's detections that belong to its single most common true person."""
    from soccer_analytics.analysis.projection import on_pitch_mask

    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    pures = []
    for tid, rows in assignment.tracks.items():
        rows = rows[keep[rows]]
        if len(rows) < 8:
            continue
        persons = truth.det_person[rows]
        counts = np.bincount(persons)
        pures.append(counts.max() / len(rows))
    return (float(np.mean(pures)) if pures else 0.0), len(pures)


def test_tracking_keeps_one_person_per_track(prepared) -> None:
    _, truth, detections = _prepared()
    from soccer_analytics.analysis.projection import on_pitch_mask

    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    assignment = _track_people(detections, keep)
    purity, n_tracks = _track_purity(truth, detections, assignment)
    assert n_tracks > 8, "tracking produced almost no tracks"
    assert purity > 0.90, f"tracks are mixing people (purity {purity:.2f})"


def test_team_clustering_matches_the_simulated_kits(prepared) -> None:
    from soccer_analytics.analysis.projection import on_pitch_mask

    _, truth, detections = _prepared()
    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    assignment = _track_people(detections, keep)
    teams, quality, _colors = _team_assignment(detections, assignment)

    correct = total = 0
    for tid, team in teams.items():
        persons = truth.det_person[assignment.tracks[tid]]
        true_role = int(np.bincount(persons).argmax())
        expected = truth.team_of_player[true_role]
        if expected < 0:
            continue  # referee/spectator: not a team assignment
        correct += int(team == expected)
        total += 1
    assert total >= 6, "too few tracks had a team label"
    assert correct / total > 0.9, f"team assignment accuracy {correct}/{total}"


def test_each_clustered_team_carries_the_color_it_wears() -> None:
    """The clustering already knows which kit is which; the report has to keep that so a team can be named.

    The simulated kits are deliberately invented numbers rather than colors (the simulator only promises that the
    two are *separable*), so this checks the plumbing - a color per team, read back as RGB - while the color
    math itself is held by ``test_kit_color``.
    """
    from soccer_analytics.analysis.kit import kit_rgb
    from soccer_analytics.analysis.projection import on_pitch_mask

    _, _truth, detections = _prepared()
    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    assignment = _track_people(detections, keep)
    teams, _quality, colors = _team_assignment(detections, assignment)

    assert teams, "no team labels were produced"
    assert set(colors) == {0, 1}, f"expected a color for each team, got {sorted(colors)}"
    red, blue = kit_rgb(colors[0]), kit_rgb(colors[1])
    assert red is not None and blue is not None, "a team's color did not read back as RGB"
    assert red != blue, "the two kits are separable, so they cannot be reported as the same color"
    # The color vector is [L, a, b, sat, val] padded into the 12-float descriptor layout. Measured on the real
    # game, passing the bare 5-float vector read a/b/sat as L/a/b and turned a red kit into "light gray" - the
    # swatch must decode the L/a/b the clustering actually clustered on.
    for color in colors.values():
        assert len(color) == 12, f"expected a full 12-float descriptor, got {len(color)}"
        assert color[0] > 0, "kit_fraction must be set or kit_rgb refuses to decode the color"


def test_the_report_records_the_color_of_each_team_s_kit() -> None:
    """What the page shows as a swatch and turns into a name has to survive into the saved report.

    The report is what the dashboard reads back, so the color has to be in it - not only in the moment the
    clustering ran.
    """
    from soccer_analytics.analysis.stage_b import build_report

    _, _truth, detections = _prepared(frames=200, seed=4)
    report, _ = build_report(
        detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH, match_frames=len(detections.time)
    )
    assert report.teams, "the report has no teams"
    for team in report.teams:
        assert team.kit_rgb is not None, f"team {team.team} has no kit color"
        assert len(team.kit_rgb) == 3 and all(0 <= channel <= 255 for channel in team.kit_rgb)


def test_kit_quality_is_low_when_two_teams_wear_similar_colors(monkeypatch) -> None:
    """The report must be able to say 'this is a guess' rather than assert a team.

    Everything on the pitch wears the same color here (teams *and* the referee), so any split k-means produces is
    arbitrary and the separation must collapse.
    """
    import synthetic_match as sim

    same = dict(L=0.55, a=0.62, b=0.58, sat=0.8, val=0.85, hue=[0.0, 0.0, 0.0, 0.1, 0.9, 0.0])
    monkeypatch.setitem(sim.TEAM_KITS, 1, same)
    monkeypatch.setitem(sim.TEAM_KITS, 2, same)
    _, _, detections = _prepared(frames=200, seed=2)
    from soccer_analytics.analysis.projection import on_pitch_mask

    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    assignment = _track_people(detections, keep)
    _, quality, _colors = _team_assignment(detections, assignment)
    assert quality and max(quality.values()) < 0.5


def _visible_truth_distance(truth, players: int = 18) -> float:
    """Distance the camera could actually have measured: only frames where that person was detected, in runs."""
    total = 0.0
    for person in range(players):
        frames = np.where(truth.detected[:, person])[0]
        if len(frames) < 2:
            continue
        runs = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
        for run in runs:
            if len(run) < 2:
                continue
            steps = np.linalg.norm(np.diff(truth.positions[run, person], axis=0), axis=1)
            total += float(steps.sum())
    return total


def test_distances_and_speeds_are_physically_plausible(prepared) -> None:
    _, truth, detections = _prepared()
    report, _ = build_report(detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH)
    assert report.players
    for player in report.players:
        assert player.distance_m >= 0
        assert 0 <= player.speed_kmh.max() <= 36.0
        # no player can cover more than ~5 m/s over the simulated passage
        assert player.distance_m < 5.0 * (player.time[-1] - player.time[0]) + 5

    # Compare with the distance that was actually observable: a following camera sees only part of the pitch, so
    # summing the whole squad's motion over every frame would compare our estimate against motion nobody could see.
    observable = _visible_truth_distance(truth)
    estimated = float(sum(p.distance_m for p in report.players))
    assert 0.5 < estimated / observable < 1.4, f"distance {estimated:.0f} m vs observable truth {observable:.0f} m"


def test_territory_and_momentum_follow_the_simulated_play() -> None:
    """With team 0 attacking (attack_bias), it must show up further up the pitch and hold more of the action."""
    _, _, detections = _prepared(frames=600, seed=3, attack_bias=1.0)
    report, _ = build_report(detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH)
    team0, team1 = report.teams
    assert team0.players_observed > 0 and team1.players_observed > 0
    assert team0.mean_x_fraction > team1.mean_x_fraction, (team0.mean_x_fraction, team1.mean_x_fraction)
    assert report.momentum, "no momentum buckets were produced"
    for bucket in report.momentum.values():
        assert bucket["team_0"] + bucket["team_1"] == pytest.approx(1.0, abs=1e-3)
    mean_share = _possession_from_momentum(report.momentum)
    assert mean_share > 0.5, f"team 0 was attacking but only held {mean_share:.0%} of the action"
    assert team0.possession_share == pytest.approx(mean_share, abs=0.01)
    assert team0.possession_share + team1.possession_share == pytest.approx(1.0, abs=0.02)


def test_match_possession_weights_minutes_by_contested_frames() -> None:
    """A minute with one contested frame must not count as much as a minute with hundreds."""
    momentum = {
        0: {"team_0": 1.0, "team_1": 0.0, "contested_frames": 1},
        1: {"team_0": 0.0, "team_1": 1.0, "contested_frames": 99},
    }
    assert _possession_from_momentum(momentum) == pytest.approx(0.01)
    assert _possession_from_momentum({}) == 0.0


def _two_team_detections() -> PitchDetections:
    """Two players and two frames; the camera aim is on team 1 both times."""
    box = np.array(
        [
            [0.48, 0.2, 0.52, 0.4],
            [0.88, 0.2, 0.92, 0.4],
            [0.48, 0.2, 0.52, 0.4],
            [0.88, 0.2, 0.92, 0.4],
        ]
    )
    return PitchDetections(
        frame=np.array([0, 0, 1, 1], dtype=np.int64),
        time=np.array([0.0, 0.0, 1.0, 1.0]),
        xy=np.array([[10.0, 10.0], [90.0, 10.0], [10.0, 10.0], [90.0, 10.0]]),
        sigma_m=np.ones(4),
        valid=np.ones(4, dtype=bool),
        height_px=np.full(4, 100.0),
        box=box,
        conf=np.ones(4),
        kit=np.zeros((4, 12)),
        det_track=np.array([1, 2, 1, 2]),
        det_index=np.arange(4),
        aim_xy=np.array([[90.0, 10.0], [90.0, 10.0]]),
        camera_xy=np.array([0.0, 0.0]),
    )


def test_image_space_ball_overrides_the_camera_aim_proxy() -> None:
    """Frame 0's measured ball sits on team 0, while the camera aim is on team 1; frame 1 has no ball image."""
    detections = _two_team_detections()
    ball_image = BallImage(
        u=np.array([0.52, np.nan]),
        v=np.array([0.40, np.nan]),
        valid=np.array([True, False]),
    )
    momentum, coverage = _momentum(
        detections,
        np.ones(4, dtype=bool),
        {1: 0, 2: 1},
        np.array([1, 2, 1, 2]),
        ball_image=ball_image,
    )
    assert momentum[0]["team_0"] == pytest.approx(0.5)
    assert coverage == {
        "image_available_frames": 1,
        "image_contested_frames": 1,
        "aim_contested_frames": 1,
    }


def test_without_a_ball_image_possession_stays_on_the_aim_proxy() -> None:
    momentum, coverage = _momentum(
        _two_team_detections(),
        np.ones(4, dtype=bool),
        {1: 0, 2: 1},
        np.array([1, 2, 1, 2]),
    )
    assert momentum[0]["team_1"] == pytest.approx(1.0)
    assert coverage["image_contested_frames"] == 0 and coverage["aim_contested_frames"] == 2


def test_report_states_its_own_limitations(prepared) -> None:
    _, _, detections = _prepared()
    report, _ = build_report(detections, pitch_length_m=PITCH_LENGTH, pitch_width_m=PITCH_WIDTH)
    joined = " ".join(report.notes).lower()
    assert "manual" in joined and "ball" in joined
    assert report.detections_used > 0
    assert report.frames_analyzed == 400


def test_tracking_is_deterministic(prepared) -> None:
    from soccer_analytics.analysis.projection import on_pitch_mask

    _, _, detections = _prepared()
    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    first = _track_people(detections, keep).track_id
    second = _track_people(detections, keep).track_id
    assert np.array_equal(first, second)


def test_the_weak_label_note_counts_only_labeled_tracks() -> None:
    """A track whose label is -1 has no label, so it cannot be a "best guess" of one.

    The old rule counted every track with quality below the bar, and unlabeled fragments carry quality 0 - on the
    real game the note fired with 13,257 entries out of 9,164 tracks, which drowned the signal it exists to carry.
    """
    from soccer_analytics.analysis.stage_b import _weak_labels

    teams = {1: 0, 2: 1, 3: -1}
    quality = {0: 0.0, 1: 0.9, 2: 0.1, 3: 0.5}  # track 0 is unlabeled; 1 solid; 2 marginal; 3 "other"

    assert _weak_labels(teams, quality) == [2]


def test_a_clean_match_does_not_report_weak_labels() -> None:
    """The synthetic kits are plainly separable, so a note about weakly separated colors would be crying wolf.

    With the old rule this note fired for essentially every match, because every non-voter track (most of them
    unlabeled fragments) counted as weak - it said nothing about whether any displayed label was doubtful.
    """
    from soccer_analytics.analysis.projection import on_pitch_mask
    from soccer_analytics.analysis.stage_b import _weak_labels

    _, _truth, detections = _prepared(frames=200, seed=4)
    keep = on_pitch_mask(detections, PITCH_LENGTH, PITCH_WIDTH)
    assignment = _track_people(detections, keep)
    teams, quality, _colors = _team_assignment(detections, assignment)

    assert sum(1 for team in teams.values() if team >= 0) > 0, "no labels to judge"
    assert _weak_labels(teams, quality) == []

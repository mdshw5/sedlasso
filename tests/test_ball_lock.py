"""The gimbal's hardware ball lock: in-play detection and image-space possession."""

from __future__ import annotations

import numpy as np

from soccer_analytics.analysis.ball_lock import (
    BallImage,
    BallLock,
    _long_runs,
    ball_image_from_scan,
    ball_lock_from_aligned,
    foot_distance,
    image_space_possession,
)


def test_long_runs_keeps_only_sustained_spells() -> None:
    flags = np.array([True, True, False, True, True, True, True, False, True])
    kept = _long_runs(flags, 3)
    assert kept.tolist() == [False, False, False, True, True, True, True, False, False]


def test_ball_lock_from_aligned_normalizes_image_x() -> None:
    aligned = {
        "lock": np.array([1.0, 0.0, 1.0]),
        "ball_x": np.array([1280.0, np.nan, 2560.0]),
    }
    lock = ball_lock_from_aligned(aligned)
    assert lock.locked.tolist() == [True, False, True]
    assert lock.u[0] == 0.5  # center of a 2560-wide frame
    assert lock.u[2] == 1.0
    assert not np.isfinite(lock.u[1])


def test_in_play_requires_a_sustained_lock() -> None:
    locked = np.zeros(20, dtype=bool)
    locked[2:4] = True  # too short
    locked[8:15] = True  # a real spell
    lock = BallLock(locked=locked, u=np.full(20, 0.5), v=np.full(20, 0.3), size=np.zeros(20), speed=np.zeros(20))
    play = lock.in_play()
    assert not play[2] and not play[3]
    assert play[8] and play[14]
    assert not play[15]


def test_image_space_possession_credits_the_nearest_player() -> None:
    # Two players in the picture: team 0 at u=0.5, team 1 at u=0.9. The ball sits on team 0.
    det_frame = np.array([0, 0])
    det_box = np.array([[0.48, 0.2, 0.52, 0.4], [0.88, 0.2, 0.92, 0.4]])  # foot points at u=0.5 and 0.9
    det_team = np.array([0, 1])
    lock = BallLock(
        locked=np.array([True]),
        u=np.array([0.5]),
        v=np.array([0.4]),
        size=np.array([2.0]),
        speed=np.array([0.0]),
    )
    result = image_space_possession(lock, det_frame, det_box, det_team)
    assert result["touches"][0] == 1
    assert result["touches"][1] == 0
    assert result["share"][0] == 1.0


def test_image_space_possession_ignores_a_ball_far_from_everyone() -> None:
    det_frame = np.array([0])
    det_box = np.array([[0.48, 0.2, 0.52, 0.4]])
    det_team = np.array([0])
    lock = BallLock(
        locked=np.array([True]),
        u=np.array([0.95]),  # far from the only player
        v=np.array([0.4]),
        size=np.array([2.0]),
        speed=np.array([0.0]),
    )
    result = image_space_possession(lock, det_frame, det_box, det_team)
    assert result["contested_frames"] == 0
    assert result["touches"] == {0: 0, 1: 0}


def test_image_space_possession_skips_unlocked_frames() -> None:
    det_frame = np.array([0])
    det_box = np.array([[0.48, 0.2, 0.52, 0.4]])
    det_team = np.array([0])
    lock = BallLock(
        locked=np.array([False]),
        u=np.array([0.5]),
        v=np.array([0.4]),
        size=np.array([2.0]),
        speed=np.array([0.0]),
    )
    result = image_space_possession(lock, det_frame, det_box, det_team)
    assert result["contested_frames"] == 0


def test_ball_image_from_scan_keeps_only_measured_or_forecast_frames() -> None:
    records = [
        {"i": 1, "status": "tracking", "u": 0.25, "v": 0.30},
        {"i": 2, "status": "coasting", "u": 0.30, "v": 0.31},
        {"i": 3, "status": "out_of_view", "u": 0.90, "v": 0.90},
        {"i": 4, "status": "lost"},
        {"i": 99, "status": "tracking", "u": 0.10, "v": 0.10},
        {"i": 0, "status": "tracking", "u": None, "v": 0.20},
    ]
    image = ball_image_from_scan(records, 5)
    assert image.valid.tolist() == [False, True, True, False, False]
    assert image.u[1] == 0.25 and image.v[2] == 0.31
    assert image.source == "ball scan"


def test_image_space_possession_accepts_a_scan_image() -> None:
    det_frame = np.array([0, 0])
    det_box = np.array([[0.48, 0.2, 0.52, 0.4], [0.88, 0.2, 0.92, 0.4]])
    det_team = np.array([0, 1])
    image = BallImage(u=np.array([0.5]), v=np.array([0.4]), valid=np.array([True]), source="ball scan")
    result = image_space_possession(image, det_frame, det_box, det_team)
    assert result["touches"] == {0: 1, 1: 0}


def test_foot_distance_falls_back_to_horizontal_when_v_is_unknown() -> None:
    box = np.array([[0.48, 0.2, 0.52, 0.4]])
    assert np.isclose(foot_distance(box, 0.6, np.nan)[0], 0.1)
    assert np.isclose(foot_distance(box, 0.5, 0.4)[0], 0.0)
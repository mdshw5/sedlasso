"""The gimbal log parser and the pan model built from it.

The parser is tested on synthetic log text that mirrors the real format (including the camera's habit of splitting
a line mid-token), and the pan model on a synthetic pan whose axis and scale are known - so a regression in either
is caught without the real footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.geometry.gimbal_log import GimbalFrame, load_gimbal_logs, parse_gimbal_log
from soccer_analytics.geometry.gimbal_motion import (
    PanModel,
    _GAME_MANIFEST_CACHE,
    _clips_for_segment,
    _game_manifest_for,
    align_log,
    find_logs_for_clips,
    fit_pan_model,
    log_orientation,
    segment_has_log,
)

HEADER = """2026-10-03 16:28:37.951
SDKV3.3.1-1230-type11
Soccer yawLim120 speedTH50 areaTH0 pitchInit12 Zoom1.4
SDKStart
"""


def _frame(number: int, *, yaw: float, pitch: float = 11.6, zoom: int = 20, lock: tuple[int, int] = (0, 0)) -> str:
    return (
        f"\n2026-10-03 16:28:{38 + number // 10:02d}.{number % 10:03d}\n"
        f"Frame{number} fps10 SZ28 Yaw{yaw} Pit{pitch}\n"
        f"AFilterP26 mot21 Deque25(V125,x6)(V115,x0)(V112,x0)\n"
        f"Fast16 Crowdx1586\n"
        f"BallSz2 TSz7 BallFail\n"
        f"\n2026-10-03 16:28:{38 + number // 10:02d}.{number % 10:03d}\n"
        f"Frame{number} FIN:x1280 y720 V0 YawErr0 out0 Ctrl:X0 Y0 zoomSz{zoom} In:0 Lock:{lock[0]}/{lock[1]}\n"
    )


def test_parse_merges_motion_and_fin_records() -> None:
    log = parse_gimbal_log(HEADER + _frame(1, yaw=-48.2, zoom=18))
    assert log.sdk == "V3.3.1-1230-type11"
    assert log.config.startswith("yawLim120")
    assert len(log.frames) == 1
    record = log.frames[0]
    assert record.frame == 1
    assert record.yaw_deg == pytest.approx(-48.2)
    assert record.pitch_deg == pytest.approx(11.6)
    assert record.zoom_sz == 28  # from the motion line's SZ
    assert record.fin_zoom_sz == 18  # from the FIN line's zoomSz
    assert record.lock == (0, 0)
    assert record.ball_sz == 2
    assert record.ball_fail is True
    assert record.motion == 21
    assert record.crowd_x == 1586


def test_parse_ball_track_and_box() -> None:
    text = HEADER + (
        "\n2026-10-03 16:28:42.392\n"
        "Frame36 fps10 SZ29 Yaw-1.9 Pit11.6\n"
        "AFilterP27 mot21 Deque33(V52,x-15)(V45,x12)(V43,x15)\n"
        "Fast1 Crowdx2482\n"
        "BallSz2 TSz5 \n"
        "T0 RB12 xv1729 yv53 xm-137 ym21\n"
        "BT0mayStaRB(-1371,149,18,18)V-137toL Ballx32 OutX32\n"
        "\n2026-10-03 16:28:42.393\n"
        "Frame36 FIN:x49 y720 V200 YawErr-1230 out-540 Ctrl:X-50 Y0 zoomSz12 In:0 Lock:1/9\n"
    )
    log = parse_gimbal_log(text)
    record = log.frames[0]
    assert record.locked is True
    assert record.ball_track_id == 0
    assert record.ball_xv == pytest.approx(1729)
    assert record.ball_xm == pytest.approx(-137)
    assert record.ball_box == pytest.approx((-1371, 149, 18, 18))
    assert record.ball_dir == "L"
    assert record.ball_x == pytest.approx(32)
    assert record.out_x == pytest.approx(32)


def test_parse_reassembles_a_line_split_mid_token() -> None:
    # The camera occasionally splits a content line across two lines, with the frame's timestamp between them.
    text = HEADER + (
        "\n2026-10-03 16:29:09.710\n"
        "Frame100 fps10 SZ28 Yaw-1.9 Pit11.6\n"
        "AFilterP26 mot21 Deque25(V125,x6)(V115,x0)(V112,x0)\n"
        "Fast16 Crowdx1586\n"
        "BallSz2 TSz7 \n"
        "BT0mayStaRB(-1435,278,20,21)V17\n"
        "\n2026-10-03 16:29:09.711\n"
        "8toR Ballx1194 OutX1280\n"
        "\n2026-10-03 16:29:09.712\n"
        "Frame100 FIN:x1280 y720 V0 YawErr0 out0 Ctrl:X0 Y0 zoomSz12 In:0 Lock:1/9\n"
    )
    log = parse_gimbal_log(text)
    assert log.unknown_lines == 0
    record = log.frames[0]
    assert record.ball_box == pytest.approx((-1435, 278, 20, 21))
    assert record.ball_dir == "R"
    assert record.ball_x == pytest.approx(1194)


def test_parse_counts_unknown_lines_without_raising() -> None:
    log = parse_gimbal_log(HEADER + "\n2026-10-03 16:28:39.000\nSomethingNew: 42\n")
    assert log.unknown_lines == 1


def test_load_merges_logs_by_frame_number(tmp_path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(HEADER + _frame(1, yaw=0.0) + _frame(2, yaw=1.0))
    second.write_text(HEADER + _frame(3, yaw=2.0))
    merged = load_gimbal_logs([first, second])
    assert [r.frame for r in merged.frames] == [1, 2, 3]


def test_align_log_places_frames_on_the_game_clock() -> None:
    # Two clips: the first starts at 0 s, the second at 100 s. A frame at 2 s into the second clip is at 102 s.
    log_a = parse_gimbal_log(HEADER + _frame(1, yaw=0.0))
    log_b = parse_gimbal_log(HEADER + _frame(2, yaw=10.0))
    # Rewrite the timestamps so the second log's frame is 2 s after its clip start.
    log_b.frames[0].time_s = 2.0
    log_a.frames[0].time_s = 0.0
    aligned = align_log([(log_a, 0.0), (log_b, 100.0)], start_s=0.0, fps=1.0, frame_count=103)
    assert aligned["yaw"][0] == pytest.approx(0.0)
    assert aligned["yaw"][102] == pytest.approx(10.0)


def _synthetic_pan(frames: int, axis: np.ndarray, scale: float, yaw: np.ndarray) -> np.ndarray:
    """Orientation for a pure pan about ``axis`` with the given yaw track (reference frame = frame 0).

    The reference frame is frame 0, so the rotation is relative to the first yaw - the same convention
    ``log_orientation`` uses (its ``yaw0`` is the first logged yaw).
    """
    import cv2

    out = np.empty((frames, 3, 3))
    for i in range(frames):
        out[i] = cv2.Rodrigues(np.radians(scale * (yaw[i] - yaw[0])) * axis)[0]
    return out


def test_fit_pan_model_recovers_the_physical_axis_and_scale() -> None:
    import cv2

    frames = 400
    tilt = 11.6
    axis = np.array([0.0, np.sin(np.radians(90 - tilt)), np.cos(np.radians(90 - tilt))])
    yaw = np.linspace(-40.0, 40.0, frames)
    q = _synthetic_pan(frames, axis, 0.8, yaw)
    ok = np.ones(frames, dtype=bool)
    pitch = np.full(frames, tilt)
    model = fit_pan_model(q, ok, yaw, pitch)
    assert model is not None
    assert np.allclose(model.axis, axis, atol=1e-6)
    assert model.scale == pytest.approx(0.8, abs=0.02)
    # The orientation rebuilt from the model matches the synthetic one.
    rebuilt = log_orientation(yaw, model)
    for i in range(0, frames, 40):
        rel = rebuilt[i].T @ q[i]
        angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        assert angle < 0.5


def test_fit_pan_model_returns_none_without_motion() -> None:
    frames = 100
    q = np.tile(np.eye(3), (frames, 1, 1))
    yaw = np.zeros(frames)
    assert fit_pan_model(q, np.ones(frames, dtype=bool), yaw, np.full(frames, 11.6)) is None


def test_the_yaw_scale_is_not_dragged_down_by_windows_that_rotated_off_axis() -> None:
    """Motion that is not about the pan axis must not vote on the scale.

    On the reference game, bursts of fast-tracking motion that the yaw channel cannot explain covered most windows
    and pulled the median ratio of chain rotation to logged yaw from ~1.0 down to 0.70 - every projection then
    under-rotated by 30%, and the overlay visibly trailed the pan. This synthetic reproduces that mixture: four
    fifths of the timeline rotates about an axis 45 deg off the pan axis (so its projection under-counts by
    cos(45)), a fifth is a clean pan, and the scale must come out ~1.0 because the off-axis windows are screened
    out of the vote - not because they are absent.
    """
    import cv2

    frames = 1200
    tilt = 11.6
    axis = np.array([0.0, np.sin(np.radians(90 - tilt)), np.cos(np.radians(90 - tilt))])
    # 45 deg off the pan axis, so a window about this axis projects only cos(45) ~ 0.71 of its angle onto the pan.
    off_axis = cv2.Rodrigues(np.radians(45.0) * np.array([1.0, 0.0, 0.0]))[0] @ axis
    # 0.2 deg of rotation per frame, so even the 5-frame fallback window clears the minimum-angle gate.
    angles = np.radians(np.linspace(-120.0, 120.0, frames))

    q = np.empty((frames, 3, 3))
    for i in range(frames):
        # Per 30-frame group: 6 clean frames (24..29 of the previous group are the off-axis burst; the grid below
        # makes the first 6 frames of each group the clean stretch, so the 5-frame windows that end at multiples
        # of 5 land exactly inside it).
        use = axis if i % 30 < 6 else off_axis
        q[i] = cv2.Rodrigues(angles[i] * use)[0]

    yaw = np.linspace(-120.0, 120.0, frames)
    ok = np.ones(frames, dtype=bool)
    pitch = np.full(frames, tilt)
    model = fit_pan_model(q, ok, yaw, pitch)
    assert model is not None
    assert model.scale == pytest.approx(1.0, abs=0.03), (
        "off-axis windows must be screened out of the scale fit"
    )


def test_log_orientation_holds_through_a_gap() -> None:
    model = PanModel(axis=np.array([0.0, 0.98, 0.2]), scale=0.8, yaw0=0.0, rms_deg=0.0, samples=1)
    yaw = np.array([0.0, 10.0, np.nan, 20.0])
    q = log_orientation(yaw, model)
    # The gap frame holds the previous orientation rather than jumping to identity.
    assert np.allclose(q[2], q[1])
    assert not np.allclose(q[3], q[1])


def test_find_logs_matches_a_clip_to_its_log_by_time_of_day(tmp_path) -> None:
    # The camera names a clip "16:28:37.784.MP4" and its log "2026-10-03 16:28:37.json"; the time-of-day links them.
    log_dir = tmp_path / "Chameleon Logs"
    log_dir.mkdir()
    (log_dir / "2026-10-03 16:28:37.json").write_text(HEADER)
    (log_dir / "2026-10-03 16:58:38.json").write_text(HEADER)
    clips = [str(tmp_path / "16:28:37.784.MP4"), str(tmp_path / "16:58:38.391.MP4")]
    pairs = find_logs_for_clips(clips)
    assert pairs is not None
    assert [Path(p).name for p, _ in pairs] == ["2026-10-03 16:28:37.json", "2026-10-03 16:58:38.json"]


def test_find_logs_accepts_logs_beside_the_clips(tmp_path) -> None:
    """Keeping the logs in the footage folder is the simplest layout, so it is the first place searched."""
    log = tmp_path / "2026-10-03 16:28:37.json"
    log.write_text(HEADER)
    pairs = find_logs_for_clips([str(tmp_path / "16:28:37.784.MP4")])
    assert pairs is not None
    assert Path(pairs[0][0]) == log
    assert pairs[0][1] == 0.0


def test_find_logs_accepts_a_renamed_or_differently_cased_log_directory(tmp_path) -> None:
    """The camera's folder name is a convention, not a requirement: a renamed folder still holds the logs."""
    for index, folder in enumerate(("chameleon logs", "camera telemetry")):
        base = tmp_path / f"case{index}"
        base.mkdir()
        log_dir = base / folder
        log_dir.mkdir()
        (log_dir / "2026-10-03 16:28:37.json").write_text(HEADER)
        pairs = find_logs_for_clips([str(base / "16:28:37.784.MP4")])
        assert pairs is not None, folder
        assert Path(pairs[0][0]) == log_dir / "2026-10-03 16:28:37.json"
        assert pairs[0][1] == 0.0


def test_find_logs_returns_none_without_a_log_directory(tmp_path) -> None:
    assert find_logs_for_clips([str(tmp_path / "16:28:37.784.MP4")]) is None


def test_manifest_with_real_clips_wins_over_one_listing_the_combined_video(tmp_path) -> None:
    """A manifest that names the source clips places the logs; one that names only the combined video cannot.

    The game build writes its manifest (with the source clips) into the footage's analysis directory, but an
    older/other archive manifest may list the combined video as its own single clip - and that one says nothing
    about where the camera logs go. When both match the video, the one carrying real clips must win; when only
    the collapsed one exists, it is still returned (the caller falls back to the video itself).
    """
    import json

    footage = tmp_path / "footage"
    footage.mkdir()
    combined = footage / "game_x.mp4"
    combined.write_bytes(b"")
    (footage / "16:28:37.784.MP4").write_bytes(b"")
    (footage / "16:58:38.391.MP4").write_bytes(b"")
    # The non-informative manifest sits beside the video (checked first)...
    (footage / "game.json").write_text(
        json.dumps({"output": str(combined), "clips": [{"path": str(combined), "start_s": 0.0}]})
    )
    # ...and the real one in the analysis directory.
    analysis = footage / "analysis" / "2026-10-03_game_x"
    analysis.mkdir(parents=True)
    (analysis / "game.json").write_text(
        json.dumps(
            {
                "output": str(combined),
                "clips": [
                    {"path": "16:28:37.784.MP4", "start_s": 0.0},
                    {"path": "16:58:38.391.MP4", "start_s": 1800.2},
                ],
            }
        )
    )
    _GAME_MANIFEST_CACHE.clear()
    try:
        found = _game_manifest_for(str(combined))
        assert found is not None
        assert len(found["clips"]) == 2, "the manifest with the source clips must win"
    finally:
        _GAME_MANIFEST_CACHE.clear()

    # A video whose only manifest is the collapsed one still gets that manifest (the fallback).
    solo_dir = tmp_path / "solo"
    solo_dir.mkdir()
    solo = solo_dir / "game_solo.mp4"
    solo.write_bytes(b"")
    (solo_dir / "game.json").write_text(
        json.dumps({"output": str(solo), "clips": [{"path": str(solo), "start_s": 0.0}]})
    )
    _GAME_MANIFEST_CACHE.clear()
    try:
        found = _game_manifest_for(str(solo))
        assert found is not None and len(found["clips"]) == 1
    finally:
        _GAME_MANIFEST_CACHE.clear()


def test_clips_for_segment_resolves_relative_clip_paths(tmp_path) -> None:
    """Clip paths stored relative to the footage folder (how the game build writes them) must come out usable.

    The combined video lives beside the raw clips; the manifest names them bare ("16:28:37.784.MP4"). Unresolved,
    the log lookup would look for "Chameleon Logs" below the process's own directory and find nothing - so the
    paths must come back joined to the footage folder, with the start seconds intact.
    """
    import json
    from types import SimpleNamespace

    footage = tmp_path / "footage"
    footage.mkdir()
    combined = footage / "game_x.mp4"
    combined.write_bytes(b"")
    clips = [("16:28:37.784.MP4", 0.0), ("16:58:38.391.MP4", 1800.2), ("17:28:37.430.MP4", 3599.1)]
    for name, _start in clips:
        (footage / name).write_bytes(b"")
    analysis = footage / "analysis" / "2026-10-03_game_x"
    analysis.mkdir(parents=True)
    (analysis / "game.json").write_text(
        json.dumps(
            {
                "output": str(combined),
                "clips": [{"path": name, "start_s": start} for name, start in clips],
            }
        )
    )
    segment = SimpleNamespace(meta={"video": str(combined)})
    _GAME_MANIFEST_CACHE.clear()
    try:
        paths, starts = _clips_for_segment(segment)
        assert [Path(p) for p in paths] == [footage / name for name, _ in clips]
        assert starts == [0.0, 1800.2, 3599.1]
    finally:
        _GAME_MANIFEST_CACHE.clear()


def test_clips_for_segment_survives_a_missing_or_unreadable_game_manifest(tmp_path) -> None:
    """Reading a segment's results must not depend on the clip manifest; only the gimbal-log refinement does.

    A production copy of the archive may have lost - or never received - the small ``game.json`` beside its
    analysis directory, and the documented best effort is then to report no clips, so the caller keeps the
    estimated chain. Letting the read fail is what the dashboard turns into "Could not read the segment
    results", although the segment's own chunks are complete without the manifest.
    """
    from types import SimpleNamespace

    missing = tmp_path / "analysis" / "2026-10-03_game_x" / "game.json"
    segment = SimpleNamespace(meta={"video": str(missing)})
    assert _clips_for_segment(segment) == ([str(missing)], [0.0])
    assert segment_has_log(segment) is False

    # A manifest that is there but unreadable degrades the same way.
    missing.parent.mkdir(parents=True)
    missing.write_text("this is not json")
    _GAME_MANIFEST_CACHE.clear()
    assert _clips_for_segment(segment) == ([str(missing)], [0.0])


def test_game_manifest_is_found_by_path_and_by_basename() -> None:
    # A combined video is connected back to its clips (and so its logs) through the game manifest. The lookup must
    # survive the video being moved, because the manifest records the path it was written at. This uses the real
    # repo manifest when one is present, and is skipped otherwise.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    manifests = list((repo_root / "data" / "games").glob("*/game.json"))
    if not manifests:
        pytest.skip("no game manifest in this checkout")
    import json

    output = json.loads(manifests[0].read_text())["output"]
    _GAME_MANIFEST_CACHE.clear()
    try:
        assert _game_manifest_for(output) is not None  # exact path
        _GAME_MANIFEST_CACHE.clear()
        moved = str(Path("/moved/elsewhere") / Path(output).name)
        assert _game_manifest_for(moved) is not None  # moved but not renamed
        _GAME_MANIFEST_CACHE.clear()
        assert _game_manifest_for("/moved/elsewhere/definitely_not_a_game.mp4") is None
    finally:
        _GAME_MANIFEST_CACHE.clear()
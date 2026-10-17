"""Reopening a saved match: the footage it used and the segment to continue.

The failure these guard is the never-merged game: the match record names the *merged* output, which does not
exist, while the real recordings are the clips in the match's ``game.json``. A match picker that only stats the
recorded source therefore opens with no videos selected and no analysis window.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.dashboard.match_state import match_footage


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _match(root: Path, match_id: str, sources: list[str], *, segments: list[str] | None = None) -> Path:
    directory = root / match_id
    _write_json(
        directory / "match.json",
        {
            "match_id": match_id,
            "sources": sources,
            "segments": segments or [],
            "pitch_length_m": 60.0,
            "pitch_width_m": 40.0,
            "format": "9v9",
            "created": 0.0,
            "team_names": ["Team A", "Team B"],
            "team_rosters": [],
            "notes": [],
        },
    )
    return directory


def _clip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 32)
    return path


def _game(directory: Path, clips: list[Path], *, output: Path | None = None) -> Path:
    manifest = directory / "game.json"
    _write_json(
        manifest,
        {
            "game_id": directory.name,
            "output": str(output or (directory / "game_merged.mp4")),
            "duration_s": 100.0,
            "clips": [
                {
                    "path": str(clip),
                    "start_s": index * 50.0,
                    "duration_s": 50.0,
                    "bytes": clip.stat().st_size,
                }
                for index, clip in enumerate(clips)
            ],
            "start_s": 10.0,
            "half_s": 50.0,
            "end_s": 90.0,
        },
    )
    return manifest


def _segment(directory: Path, name: str, start: float, end: float, video: Path, *, mtime: float | None = None) -> Path:
    path = directory / "segments" / name
    _write_json(path / "meta.json", {"start_s": start, "end_s": end, "video": str(video)})
    if mtime is not None:
        os.utime(path / "meta.json", (mtime, mtime))
    return path


def test_a_game_match_reopens_its_clips_source_and_saved_segment(tmp_path: Path) -> None:
    footage = tmp_path / "footage"
    clip_a = _clip(footage / "10:23:11.800.mp4")
    clip_b = _clip(footage / "10:53:11.427.mp4")
    analysis_root = tmp_path / "analysis"
    directory = _match(analysis_root, "game_id", ["game_10-23-11.800.mp4"])
    manifest = _game(directory, [clip_a, clip_b])
    segment = _segment(directory, "whole_game_10_90", 10.0, 90.0, manifest)

    found = match_footage(MatchLibrary(root=analysis_root), "game_id")

    assert found.videos == (str(clip_a), str(clip_b))
    assert found.analysis_source == str(manifest)
    assert found.game is not None and [clip.path for clip in found.game.clips] == [str(clip_a), str(clip_b)]
    assert found.saved_segment == str(segment)
    assert found.window == (10.0, 90.0)
    assert found.missing == ()


def test_a_single_video_match_reopens_that_video_and_window(tmp_path: Path) -> None:
    video = _clip(tmp_path / "footage" / "match.mp4")
    analysis_root = tmp_path / "analysis"
    directory = _match(analysis_root, "single", [str(video)])
    segment = _segment(directory, "whole_5_55", 5.0, 55.0, video)

    found = match_footage(MatchLibrary(root=analysis_root), "single")

    assert found.videos == (str(video),)
    assert found.analysis_source == str(video)
    assert found.game is None
    assert found.saved_segment == str(segment)
    assert found.window == (5.0, 55.0)


def test_missing_footage_is_reported_without_pretending_there_is_a_source(tmp_path: Path) -> None:
    missing = str(tmp_path / "gone.mp4")
    analysis_root = tmp_path / "analysis"
    _match(analysis_root, "gone", [missing])

    found = match_footage(MatchLibrary(root=analysis_root), "gone")

    assert found.videos == ()
    assert found.analysis_source is None
    assert found.missing == (missing,)
    assert found.saved_segment is None and found.window is None


def test_the_last_recorded_segment_wins_over_a_newer_directory(tmp_path: Path) -> None:
    """The record's append order is the user's analysis order; a newer stray directory must not steal the resume."""
    video = _clip(tmp_path / "footage" / "match.mp4")
    analysis_root = tmp_path / "analysis"
    directory = _match(analysis_root, "ordered", [str(video)])
    first = _segment(directory, "first", 0.0, 40.0, video, mtime=100.0)
    second = _segment(directory, "second", 40.0, 80.0, video, mtime=200.0)
    stashed = _segment(directory, "stashed", 80.0, 120.0, video, mtime=999.0)
    _write_json(
        directory / "match.json",
        json.loads((directory / "match.json").read_text()) | {"segments": [str(first), str(second)]},
    )

    found = match_footage(MatchLibrary(root=analysis_root), "ordered")

    assert found.saved_segment == str(second), "the last recorded segment is the one to continue"
    assert found.saved_segment != str(stashed)
    assert found.window == (40.0, 80.0)


def test_the_segment_video_is_the_last_resort_when_the_record_lost_its_path(tmp_path: Path) -> None:
    video = _clip(tmp_path / "footage" / "match.mp4")
    analysis_root = tmp_path / "analysis"
    directory = _match(analysis_root, "moved", [str(tmp_path / "old_disk" / "match.mp4")])
    _segment(directory, "whole_0_60", 0.0, 60.0, video)

    found = match_footage(MatchLibrary(root=analysis_root), "moved")

    assert found.videos == (str(video),)
    assert found.analysis_source == str(video)
    assert found.window == (0.0, 60.0)

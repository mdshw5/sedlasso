"""The analysis-beside-the-footage layout: derivation, discovery, portability and migration.

The point of the layout is that an analysis is *self-contained*: everything a match needs lives in one directory
next to the video, so the footage directory can be copied or moved as a unit. These tests pin the three
properties that make that true - where the directory is derived from, that recorded paths survive being moved,
and that the old repository-era data migrates into the same shape.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis.library import (
    MatchLibrary,
    MatchRecord,
    analysis_dir_for,
    analysis_id_for,
    discover_videos,
    segments_root_for,
)
from soccer_analytics.analysis.migration import migrate_all


def _fake_video(path: Path, size: int = 4096, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def footage_root(tmp_path: Path, monkeypatch) -> Path:
    """An isolated footage root, so the library scan sees only what a test puts there."""
    root = tmp_path / "Xbot"
    root.mkdir()
    monkeypatch.setenv("SOCCER_VIDEO_ROOTS", str(root))
    return root


def test_the_analysis_directory_is_derived_from_the_video(footage_root: Path) -> None:
    """Recording date (from the dated folder) + file name; segments and everything else live inside it."""
    video = _fake_video(footage_root / "2026-10-03" / "16:58:38.391.MP4")
    assert analysis_id_for(video) == "2026-10-03_16-58-38-391"
    assert analysis_dir_for(video) == footage_root / "2026-10-03" / "analysis" / "2026-10-03_16-58-38-391"
    assert segments_root_for(video) == analysis_dir_for(video) / "segments"


def test_saving_a_replay_writes_the_small_attack_sidecar(tmp_path: Path) -> None:
    """The stream server polls this file to notice a swap; it must always exist and always hold the same
    directions as the replay, including an explicit null when the build had none."""
    library = MatchLibrary(tmp_path / "matches")
    record = library.create(str(tmp_path / "x.mp4"))
    attack = {"half_frame": 5, "directions": [[1, -1], [-1, 1]]}
    library.save_replay(record.match_id, {"players": [], "attack": attack})
    sidecar = json.loads((library.path(record.match_id) / "attack.json").read_text())
    assert sidecar == {"attack": attack}

    library.save_replay(record.match_id, {"players": []})
    sidecar = json.loads((library.path(record.match_id) / "attack.json").read_text())
    assert sidecar == {"attack": None}


def test_the_id_uses_the_recording_date_not_today(tmp_path: Path) -> None:
    """A folder that is not a date falls back to the file's own timestamp, never "when it was analyzed"."""
    video = _fake_video(tmp_path / "misc" / "game.mp4", mtime=1_700_000_000.0)  # 2023-11-14
    assert analysis_id_for(video).startswith("2023-11-14_")


def test_creating_a_match_writes_beside_the_video_and_discovery_finds_it(footage_root: Path) -> None:
    video = _fake_video(footage_root / "2026-10-03" / "game_16-28-37.784.mp4")
    library = MatchLibrary()
    record = library.create(video)

    directory = analysis_dir_for(video)
    assert (directory / "match.json").exists()
    assert record.match_id == directory.name
    assert library.match_for_video(video) == record.match_id
    assert record.match_id in library.list_ids()
    assert library.path(record.match_id) == directory
    # Creating again opens the same record rather than starting a second one.
    assert library.create(video).match_id == record.match_id


def test_discovery_never_lists_an_analysis_output_as_footage(footage_root: Path) -> None:
    """The reels and clips the tool writes are MP4s under the footage root now; they are outputs, not sources.

    A backup of the analysis (moved aside as ``analysis.old``) is the same kind of output, so the exclusion
    follows that name too - the picker must not offer an archived match's previews back as footage."""
    video = _fake_video(footage_root / "2026-10-03" / "game.mp4")
    reel = footage_root / "2026-10-03" / "analysis" / "2026-10-03_game" / "highlights" / "goals.mp4"
    reel.parent.mkdir(parents=True)
    reel.write_bytes(b"reel")
    archived = footage_root / "2026-10-03" / "analysis.old" / "2026-10-03_game" / "highlights" / "previews" / "p.mp4"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"preview")
    assert discover_videos() == [video]


def test_a_moved_aside_analysis_resets_the_match(footage_root: Path) -> None:
    """Moving ``analysis`` aside is how a match is reset: it leaves the library, its footage reads as never
    analyzed, and none of its media leaks into the picker; moving it back restores everything."""
    video = _fake_video(footage_root / "2026-10-03" / "game.mp4")
    library = MatchLibrary()
    record = library.create(video)
    analysis = analysis_dir_for(video)
    (analysis / "highlights" / "previews").mkdir(parents=True)
    (analysis / "highlights" / "previews" / "p.mp4").write_bytes(b"preview")

    moved = analysis.parent.with_name("analysis.old")
    shutil.move(str(analysis.parent), str(moved))

    fresh = MatchLibrary()
    assert record.match_id not in fresh.list_ids(), "a moved-aside analysis is not a live match"
    assert fresh.match_for_video(video) is None
    assert discover_videos() == [video], "an archived analysis's media is not footage"

    shutil.move(str(moved), str(analysis.parent))
    again = MatchLibrary()
    assert record.match_id in again.list_ids()
    assert again.match_for_video(video) == record.match_id


def test_a_moved_footage_directory_still_opens_its_own_files(
    footage_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """Paths are stored relative to the match directory, so moving the folder is all "portability" has to mean.

    The video sits beside the analysis, the segments sit inside it; after a move both are re-pointed at wherever
    the directory is now, and the raw match.json on disk carries no absolute path that would go stale.
    """
    video = _fake_video(footage_root / "2026-10-03" / "game_16-28-37.784.mp4")
    library = MatchLibrary()
    record = library.create(video)
    segment = analysis_dir_for(video) / "segments" / "game_16-28-37.784_123"
    segment.mkdir(parents=True)
    (segment / "meta.json").write_text(json.dumps({"video": str(video)}))
    library.add_segment(record.match_id, segment)

    stored = json.loads((analysis_dir_for(video) / "match.json").read_text())
    assert stored["sources"] == ["game_16-28-37.784.mp4"], "the source is stored relative to the match folder"
    assert stored["segments"] == ["segments/game_16-28-37.784_123"], "the segment is stored inside the match"

    moved_day = tmp_path / "moved" / "2026-10-03"
    moved_day.parent.mkdir()
    shutil.move(str(footage_root / "2026-10-03"), str(moved_day))
    moved_video = moved_day / "game_16-28-37.784.mp4"

    # The footage has moved to another root; tell the tool where the archive lives now (the point is that the
    # *analysis* needs no fixing up - it re-resolves its own files from wherever the folder is).
    monkeypatch.setenv("SOCCER_VIDEO_ROOTS", str(tmp_path / "moved"))
    fresh = MatchLibrary()
    reloaded = fresh.load(record.match_id)
    assert reloaded.sources == [str(moved_video)]
    assert Path(reloaded.segments[0]).exists()
    assert Path(reloaded.segments[0]).is_relative_to(moved_day)
    assert fresh.match_for_video(moved_video) == record.match_id


def test_the_attack_swap_setting_round_trips(tmp_path: Path) -> None:
    """The user's side correction is part of the match record, so a reload keeps it."""
    library = MatchLibrary(tmp_path)
    library.save(MatchRecord(match_id="m", swap_attack=True))
    assert library.load("m").swap_attack is True
    assert MatchRecord(match_id="old").swap_attack is False, "archives without the field load as unswapped"


def test_saved_clicks_follow_a_moved_game_manifest(footage_root: Path, tmp_path: Path) -> None:
    """A calibration clicked on the manifest before a footage move still names the same recording.

    Click sources are stored absolute (the recording's path when the calibration was made). When the footage
    directory moves to another mount, that path is gone even though the manifest is the same file beside the
    analysis. The library re-points it by the analysis-directory name - the game id - so the dashboard does not
    refuse a valid calibration as if it had been clicked on different footage. A manifest from a different game
    is left alone.
    """
    video = _fake_video(footage_root / "2026-10-03" / "16:28:37.784.MP4", size=100)
    library = MatchLibrary()
    record = library.create(video)
    directory = library.path(record.match_id)
    local_manifest = directory / "game.json"
    local_manifest.write_text(
        json.dumps({"game_id": record.match_id, "output": "game.mp4", "duration_s": 1.0, "clips": []})
    )
    clicks_path = directory / "clicks.json"
    old_manifest = tmp_path / "old" / "2026-10-03" / "analysis" / record.match_id / "game.json"
    clicks_path.write_text(json.dumps({"clicks": [], "pitch": [60.0, 40.0], "source": str(old_manifest)}))
    assert library.load_clicks_source(record.match_id) == str(local_manifest)

    other_manifest = tmp_path / "old" / "2026-10-03" / "analysis" / "2026-10-03_another_game" / "game.json"
    clicks_path.write_text(json.dumps({"clicks": [], "pitch": [60.0, 40.0], "source": str(other_manifest)}))
    assert library.load_clicks_source(record.match_id) == str(other_manifest), (
        "a manifest from another game must not be mistaken for this match's"
    )


def test_a_game_manifest_sits_in_the_combined_videos_analysis_directory(footage_root: Path) -> None:
    first = _fake_video(footage_root / "2026-10-03" / "16:28:37.784.MP4", size=100)
    second = _fake_video(footage_root / "2026-10-03" / "16:58:38.391.MP4", size=200)
    _ordered, output, directory = game_lib.locations([first, second])
    assert output == footage_root / "2026-10-03" / "game_16-28-37.784.mp4"
    assert directory == analysis_dir_for(output)
    _fake_video(output)  # the combined video a real manifest describes exists next to its clips

    record = game_lib.GameRecord(game_id="game_x", output=str(output), duration_s=100.0)
    record.save(directory)
    # The manifest sits with the analysis while the video it names sits in the footage folder: the raw file
    # stores the name relative to that folder, so a copy of the folder needs no fixing up.
    stored = json.loads((directory / "game.json").read_text())
    assert stored["output"] == output.name
    found = game_lib.find_for_video(output)
    assert found is not None and found.game_id == "game_x"
    assert Path(found.output).resolve() == output.resolve()
    assert game_lib.find_dir_by_id("game_x") == directory
    # An explicit legacy root still finds manifests written the old way.
    legacy = footage_root / "legacy-games" / "game_y"
    record2 = game_lib.GameRecord(game_id="game_y", output=str(output), duration_s=100.0)
    record2.save(legacy)
    assert game_lib.find_for_video(output, root=footage_root / "legacy-games") is not None


def test_a_moved_footage_directory_keeps_its_game_clock_and_clips(
    footage_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """The game manifest is part of the portable archive: moved footage still resolves its video and clips.

    The raw manifest stores names relative to the footage folder, and a manifest written by an older version
    (absolute paths at the old location) is re-pointed by name at load - both end at the same working record.
    """
    clip = _fake_video(footage_root / "2026-10-03" / "16:28:37.784.MP4", size=100)
    video = _fake_video(footage_root / "2026-10-03" / "game_16-28-37.784.mp4")
    directory = analysis_dir_for(video)
    record = game_lib.GameRecord(
        game_id="game_z",
        output=str(video),
        duration_s=100.0,
        clips=[game_lib.Clip(path=str(clip), start_s=0.0, duration_s=100.0, bytes=100)],
    )
    record.save(directory)
    stored = json.loads((directory / "game.json").read_text())
    assert stored["output"] == video.name
    assert stored["clips"][0]["path"] == clip.name

    moved_day = tmp_path / "moved" / "2026-10-03"
    moved_day.parent.mkdir()
    shutil.move(str(footage_root / "2026-10-03"), str(moved_day))
    monkeypatch.setenv("SOCCER_VIDEO_ROOTS", str(tmp_path / "moved"))

    moved_video = moved_day / "game_16-28-37.784.mp4"
    found = game_lib.find_for_video(moved_video)
    assert found is not None and found.game_id == "game_z"
    assert Path(found.output).resolve() == moved_video.resolve()
    assert Path(found.clips[0].path).resolve() == (moved_day / clip.name).resolve()

    # A manifest written before this change (absolute paths into the old location) resolves the same way.
    stale = dict(stored)
    stale["output"] = str(tmp_path / "old-root" / video.name)
    stale["clips"] = [{**stored["clips"][0], "path": str(tmp_path / "old-root" / clip.name)}]
    fresh_analysis = analysis_dir_for(moved_video)
    (fresh_analysis / "game.json").write_text(json.dumps(stale))
    reloaded = game_lib.find_for_video(moved_video)
    assert reloaded is not None and Path(reloaded.output).resolve() == moved_video.resolve()


def test_migration_moves_matches_segments_and_games_beside_the_footage(
    tmp_path: Path, monkeypatch
) -> None:
    """The repository-era layout (three flat roots) becomes one analysis directory per match, in place."""
    repo = tmp_path / "repo"
    video = _fake_video(tmp_path / "footage" / "2026-10-03" / "game_16-28-37.784.mp4")

    # Legacy match: absolute source and segment paths, the id dated by the save rather than the recording.
    legacy_match = repo / "data" / "matches" / "2026-10-04_16-58-38-391"
    legacy_match.mkdir(parents=True)
    legacy_match.joinpath("report.json").write_text(json.dumps({"frames_analyzed": 3}))
    legacy_segment = repo / "data" / "segments" / "game_16-28-37.784_123__whole_game_1_2"
    legacy_segment.mkdir(parents=True)
    (legacy_segment / "meta.json").write_text(json.dumps({"video": str(video), "fps": 5.0}))
    (legacy_segment / "chunk_00000.npz").write_bytes(b"npz")
    legacy_match.joinpath("match.json").write_text(
        json.dumps(
            {
                "match_id": "2026-10-04_16-58-38-391",
                "sources": [str(video)],
                "segments": [str(legacy_segment)],
                "team_names": ["Team A", "Team B"],
            }
        )
    )
    legacy_game = repo / "data" / "games" / "game_16-28-37.784_99"
    legacy_game.mkdir(parents=True)
    legacy_game.joinpath("game.json").write_text(
        json.dumps(
            {
                "game_id": "game_16-28-37.784_99",
                "output": str(video),
                "duration_s": 300.0,
                "clips": [],
                "start_s": 10.0,
                "half_s": 100.0,
                "end_s": 290.0,
            }
        )
    )

    # A dry run must not touch anything.
    migrate_all(repo, dry_run=True, log=lambda *args: None)
    assert legacy_match.exists() and legacy_segment.exists() and legacy_game.exists()

    summary = migrate_all(repo, log=lambda *args: None)
    assert summary["matches"] == 1 and summary["games"] == 1

    target = analysis_dir_for(video)
    assert target.name == "2026-10-03_game_16-28-37-784"
    assert (target / "match.json").exists()
    assert (target / "report.json").exists(), "every artifact moved with the record"
    assert (target / "segments" / legacy_segment.name / "chunk_00000.npz").exists()
    assert (target / "game.json").exists(), "the game manifest moved in beside the record"
    assert not legacy_match.exists() and not legacy_segment.exists() and not legacy_game.exists()

    monkeypatch.setenv("SOCCER_VIDEO_ROOTS", str(tmp_path / "footage"))
    fresh = MatchLibrary()
    record = fresh.load(target.name)
    assert record.match_id == target.name
    assert record.sources == [str(video)]
    assert Path(record.segments[0]).exists()
    stored = json.loads((target / "match.json").read_text())
    assert stored["sources"] == [video.name], "rewritten with portable, footage-relative paths"
    assert stored["segments"] == [f"segments/{legacy_segment.name}"]
    stored_game = json.loads((target / "game.json").read_text())
    assert stored_game["output"] == video.name, "the game manifest is rewritten portable with the record"

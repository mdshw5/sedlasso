"""Reopening a saved match: the footage it was built from and the segment to continue.

A match record names its source as the *game's* output file (``game_10-23-11.800.mp4``), which a never-merged
game never writes. The actual recordings are the clips in the match's ``game.json``. Selecting a match must
therefore resolve the manifest - not just stat ``sources[0]`` - or the video picker stays empty and the dashboard
cannot resume the work.

The saved segment is the other half of the state: its ``meta.json`` carries the exact analyzed window, which the
dashboard uses to restore the start/length inputs (and, when the window is not one of the standard whole/half
choices, to keep using that segment directory instead of computing a new one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from soccer_analytics.analysis.game import MANIFEST_FILE, GameRecord, find_for_video, manifest_dir
from soccer_analytics.analysis.library import MatchLibrary


@dataclass(frozen=True)
class MatchFootage:
    """What a saved match was built from, ready to seed the dashboard.

    ``videos`` are the recordings the picker should show selected. ``analysis_source`` is what Stage A reads: the
    ``game.json`` manifest for a game (so frame grabs and analysis use the game clock), or the single video for an
    ordinary match. ``saved_segment``/``window`` describe the most relevant existing analysis segment.
    """

    videos: tuple[str, ...] = ()
    analysis_source: str | None = None
    game: GameRecord | None = None
    saved_segment: str | None = None
    window: tuple[float, float] | None = None
    missing: tuple[str, ...] = ()


def _segment_meta(directory: Path) -> dict | None:
    """The segment's ``meta.json``, or None when it is absent/unreadable (an unfinished segment is not a target)."""
    try:
        payload = json.loads((directory / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _segment_window(meta: dict) -> tuple[float, float] | None:
    try:
        start, end = float(meta["start_s"]), float(meta["end_s"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _saved_segment(directory: Path, record) -> tuple[str | None, tuple[float, float] | None]:
    """The segment to continue: the last one the record listed, else the newest one on disk.

    A record's ``segments`` list is append-ordered, so its last entry is the most recently produced analysis. Older
    archives (and the currently running game) can have an empty list while the segment directory exists beside the
    match, so the directory is scanned as a fallback. A segment without a readable meta is ignored: it cannot say
    what window it covers, which is exactly what the dashboard needs to restore.
    """
    recorded = [Path(entry) for entry in record.segments]
    scanned: list[Path] = []
    segments_root = directory / "segments"
    if segments_root.exists():
        scanned = [path for path in segments_root.iterdir() if path.is_dir()]

    candidates: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    for path in [*recorded, *scanned]:
        resolved = path.resolve()
        if resolved in seen or not path.is_dir():
            continue
        meta = _segment_meta(path)
        if meta is None:
            continue
        seen.add(resolved)
        candidates.append((path, meta))
    if not candidates:
        return None, None

    recorded_set = {path.resolve() for path in recorded}
    recorded_candidates = [candidate for candidate in candidates if candidate[0].resolve() in recorded_set]
    if recorded_candidates:
        path, meta = recorded_candidates[-1]
    else:
        path, meta = max(
            candidates,
            key=lambda candidate: (candidate[0] / "meta.json").stat().st_mtime_ns,
        )
    return str(path), _segment_window(meta)


def _existing_videos(clips) -> tuple[list[str], list[str]]:
    """Split a game's clips into the files that exist and the names that do not."""
    videos: list[str] = []
    missing: list[str] = []
    for clip in clips:
        path = str(Path(clip.path))
        (videos if Path(path).exists() else missing).append(path)
    return videos, missing


def match_footage(library: MatchLibrary, match_id: str) -> MatchFootage:
    """Resolve a saved match's recordings and the segment to resume.

    The match's own ``game.json`` wins: it is the archive's source of truth and its clips may be relative to the
    match directory. If there is no manifest, a recorded clip (or manifest path) is asked which game it belongs to,
    which is how a match saved from one clip still finds its siblings. Finally the segment's own ``meta.video`` is
    used, because a moved archive may have lost the record's original path but kept the analysis.
    """
    directory = library.path(match_id)
    record = library.load(match_id)

    game: GameRecord | None = None
    manifest = directory / MANIFEST_FILE
    if manifest.exists():
        try:
            game = GameRecord.load(directory)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            game = None

    videos: list[str] = []
    missing: list[str] = []
    source: str | None = None

    if game is not None and game.clips:
        videos, missing = _existing_videos(game.clips)
        source = str(manifest) if videos else None
    else:
        # A recorded clip (or the manifest itself) identifies its game, even when the record names the merged
        # output that a never-merged game never wrote.
        for recorded in record.sources:
            found = find_for_video(recorded)
            if found is None or not found.clips:
                continue
            game = found
            found_manifest = manifest_dir(found) / MANIFEST_FILE
            videos, missing = _existing_videos(found.clips)
            if videos:
                source = str(found_manifest) if found_manifest.exists() else str(Path(recorded))
            break
        else:
            for recorded in record.sources:
                path = str(Path(recorded))
                (videos if Path(path).exists() else missing).append(path)
            source = videos[0] if videos else None

    if not videos:
        # Last resort: the segment remembers the source it was analyzed from. This covers archives whose match
        # record lost its footage path but whose analysis (and video) are still on disk.
        segments_root = directory / "segments"
        if segments_root.exists():
            for meta_path in sorted(segments_root.glob("*/meta.json"), reverse=True):
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                video = meta.get("video")
                if video and Path(video).exists():
                    videos = [str(Path(video))]
                    source = str(Path(video))
                    break

    saved_segment, window = _saved_segment(directory, record)
    return MatchFootage(
        videos=tuple(videos),
        analysis_source=source,
        game=game,
        saved_segment=saved_segment,
        window=window,
        missing=tuple(missing),
    )

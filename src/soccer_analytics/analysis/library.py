"""Match archive: everything computed for a match lives beside the footage it was computed from.

``data/matches`` in the repository was the first home for this, and it made an analysis a thing that only existed
on one machine. The archive now sits with the recording instead::

    <footage directory>/
        16-28-37.784.MP4          the original clips, untouched
        game_16-28-37.784.mp4     the combined game, also untouched (a stream copy of the clips)
        analysis/
            2026-10-03_game_16-28-37.784/        one directory per analyzed video, named by recording date + file
                match.json        this index (sources, segments, format, team names)
                events.json, report.json, replay.json, calibration.json, highlights/, identities/ ...
                segments/         the Stage A results (and the ball scan's) for this match's windows
                audio/            the recording's audio as wav (the whistle scan's cache; re-extracted if removed)
                game.json         the game manifest and its marking proxy, when this video is a combined game

Copying that directory (or the whole footage directory) carries the analysis with it: the recorded paths are
stored relative to the analysis directory when they sit inside it, and re-pointed at load time, so a moved match
still opens its own footage and segments. The old ``data/matches`` root stays readable - archives that predate
this layout keep working until ``scripts/migrate_analysis.py`` moves them next to their videos.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from soccer_analytics.analysis.events import EventLog
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from soccer_analytics.ingest.source import as_source

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIRNAME = "analysis"  # the folder each analyzed video's match directory lives in
SEGMENTS_DIRNAME = "segments"  # Stage A results, inside the match directory
AUDIO_DIRNAME = "audio"  # the recording's audio as wav (the whistle scan's cache), inside the match directory
GAME_MANIFEST_FILENAME = "game.json"  # a game's clip manifest; the manifest itself is the analyzed source
LEGACY_MATCHES_ROOT = REPO_ROOT / "data" / "matches"  # where archives lived before the move beside the footage
MATCHES_ROOT = LEGACY_MATCHES_ROOT  # kept for callers that still name the old root
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Archives written before the project switched to American spellings keep their old keys and landmark labels.
# They are mapped on load so an existing match directory keeps working unchanged.
_LEGACY_REPLAY_KEYS = {"team_colours": "team_colors"}
_LEGACY_REPORT_KEYS = {"frames_analysed": "frames_analyzed"}
_LEGACY_LANDMARK_LABELS = {
    "centre spot": "center spot",
    "centre circle near": "center circle near",
    "centre circle far": "center circle far",
    "centre circle left": "center circle left",
    "centre circle right": "center circle right",
}


def upgrade_legacy_keys(payload: dict, legacy: dict[str, str]) -> dict:
    """Rename keys spelled the British way; a current key wins if both are present."""
    for old, new in legacy.items():
        if old in payload and new not in payload:
            payload[new] = payload.pop(old)
    return payload


def upgrade_replay_payload(payload: dict) -> dict:
    """Map a replay payload from before the American spelling switch onto the current keys."""
    return upgrade_legacy_keys(payload, _LEGACY_REPLAY_KEYS)


def upgrade_report_payload(payload: dict) -> dict:
    """Map a report payload from before the American spelling switch onto the current keys."""
    return upgrade_legacy_keys(payload, _LEGACY_REPORT_KEYS)


def upgrade_landmark_label(label: str | None) -> str | None:
    """Map a landmark label from before the American spelling switch onto its current name."""
    return None if label is None else _LEGACY_LANDMARK_LABELS.get(label, label)


def video_roots() -> list[Path]:
    """Where match footage is looked for: the repo's own ``data/videos``, then the machine's own archives.

    ``SOCCER_VIDEO_ROOTS`` (colon-separated) names the recording archive directories; the default is the author's
    camera share. Kept in one function so the dashboard, the discovery scan and the stream server agree about
    which disks hold footage.
    """
    extra = os.environ.get("SOCCER_VIDEO_ROOTS", "/srv/storage/home_video/Xbot")
    roots = [REPO_ROOT / "data" / "videos"]
    roots += [Path(part).expanduser() for part in extra.split(":") if part.strip()]
    return roots


def _under_analysis(path: Path) -> bool:
    """Whether a path lies under an analysis directory - including one moved aside (``analysis.old``).

    Moving the analysis directory is how a match is reset; what the move leaves behind is still app-produced
    media (preview clips, montages), never footage, so the exclusion follows the name's prefix rather than the
    literal ``analysis`` alone. The live archive itself is still discovered by its exact name
    (:func:`discover_match_manifests`), so a moved-aside copy is a backup, not a second match.
    """
    return any(part == ANALYSIS_DIRNAME or part.startswith(ANALYSIS_DIRNAME + ".") for part in path.parts[:-1])


def discover_videos(roots: list[Path] | None = None) -> list[Path]:
    """Video files under the footage roots, newest first, so the most recent match is the default.

    Anything under an ``analysis`` directory is excluded: the reels, preview clips and centered clips the tool
    itself writes are MP4s too, and listing an analysis's own output back as footage to analyze is nonsense.
    A directory moved aside as a backup (``analysis.old``) is excluded the same way - its media is still output.
    """
    found: list[Path] = []
    for root in roots if roots is not None else video_roots():
        if not root.exists():
            continue
        found += [p for p in root.rglob("*.MP4") if not _under_analysis(p)]
        found += [p for p in root.rglob("*.mp4") if not _under_analysis(p)]
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_match_manifests(roots: list[Path] | None = None) -> list[Path]:
    """Every ``analysis/<id>/match.json`` under the footage roots (the self-contained archives)."""
    found: list[Path] = []
    for root in roots if roots is not None else video_roots():
        if root.exists():
            found += [p for p in root.glob(f"**/{ANALYSIS_DIRNAME}/*/match.json")]
    return found


def analysis_id_for(video: str | Path) -> str:
    """The analysis directory's name for one video: the recording's date, then the file's own name.

    The date comes from the footage directory's name when it is a ``YYYY-MM-DD`` folder (how the camera share is
    organized), else from the file's timestamp - so ids sort chronologically by when the match was *played*, not
    by when someone got round to analyzing it, and two teams' matches never share an id by accident.
    """
    video = Path(video)
    if DATE_DIR_RE.match(video.parent.name):
        date = video.parent.name
    else:
        try:
            date = time.strftime("%Y-%m-%d", time.localtime(video.stat().st_mtime))
        except OSError:
            date = time.strftime("%Y-%m-%d")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in video.stem)[:60]
    return f"{date}_{safe}"


def new_match_id(source: str | Path) -> str:
    """Historical name for :func:`analysis_id_for`; the id doubles as the analysis directory's name."""
    return analysis_id_for(source)


def analysis_dir_for(video: str | Path) -> Path:
    """Where the analysis of ``video`` lives: an ``analysis/<id>`` directory beside the footage itself.

    Derived from the video's path alone, so it can be computed before anything exists (the combined game's path,
    or a match directory about to be created) and it never depends on which machine or repository root is in use.
    """
    video = Path(video).expanduser()
    parent = video.parent.absolute()
    return parent / ANALYSIS_DIRNAME / analysis_id_for(video)


def segments_root_for(video: str | Path) -> Path:
    """Where Stage A output for this footage goes: with the rest of its analysis, not in the repository.

    A ``game.json`` manifest is itself the analyzed source (the clips are never merged), and it already lives in
    its analysis directory - the segments belong in a ``segments/`` folder right beside it.
    """
    path = Path(video)
    if path.name == GAME_MANIFEST_FILENAME:
        return path.parent / SEGMENTS_DIRNAME
    return analysis_dir_for(path) / SEGMENTS_DIRNAME


def audio_cache_path(match_dir: str | Path, video: str | Path) -> Path:
    """Where a match's extracted audio (the whistle scan's wav cache) lives: inside the match directory.

    Everything derived from a match lives beside the match, this cache included, so clearing or moving the
    analysis directory resets or carries the audio with the rest of the state. The file name is keyed by what
    the video *is* - a clip set's game id, a file's stem - not by its basename alone: every never-merge game's
    manifest is called ``game.json``, and ``game.wav`` would hand the first game's audio to every later one.
    """
    return Path(match_dir) / AUDIO_DIRNAME / f"{as_source(video).key_name}.wav"


def match_id_from_path(path: str | Path) -> str:
    """A match id given as a path (a script argument, an archive someone moved): the last path component."""
    return Path(path).name


def store_path(value: str | Path, base: Path) -> str:
    """Store a path relative to ``base`` when it lies inside it, else absolute.

    Relative-when-inside is what makes a match directory portable: move the footage directory and
    ``game_16-28-37.784.mp4`` still means the file next door, while an absolute path would point at the old disk.
    """
    path = Path(value)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def resolve_path(value: str, base: Path) -> str:
    """The usable form of a stored path: absolute if it exists, else resolved against the match directory.

    A stored relative path is looked up next to the analysis first; an absolute path that no longer exists is
    looked up by name beside the analysis too (moving a whole footage directory keeps the names but changes the
    prefix, and that is exactly the case this is for). A value nothing matches stays as it is, so a manifest
    that named something relative to the process's own directory (older archives did) keeps working the way it
    always did.
    """
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return str(path)
        candidate = base / path.name
        return str(candidate) if candidate.exists() else str(path)
    candidate = base / path
    if candidate.exists():
        return str(candidate)
    candidate = base / path.name
    return str(candidate) if candidate.exists() else str(value)


@dataclass
class MatchRecord:
    match_id: str
    sources: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    pitch_length_m: float = 100.0
    pitch_width_m: float = 64.0
    format: str = "11v11"
    created: float = field(default_factory=time.time)
    team_names: list[str] = field(default_factory=lambda: ["Team A", "Team B"])
    # The saved team roster (analysis.rosters) each team's player names come from, by team index; "" = none
    # linked. Only the *link* lives in the match - the names stay in the roster library, so fixing a roster
    # updates every match that links it, including already-analyzed ones.
    team_rosters: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def team_roster(self, team: int) -> str:
        """The roster name linked to ``team``, or "" when none is linked (or the team index is out of range)."""
        return self.team_rosters[team] if 0 <= team < len(self.team_rosters) else ""

    def directory(self, root: str | Path = MATCHES_ROOT) -> Path:
        return Path(root) / self.match_id


def _atomic_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


class MatchLibrary:
    """The match archives, wherever they live.

    ``MatchLibrary()`` finds every analysis: the self-contained ``analysis/<id>`` directories beside the footage,
    and - until migrated - the old ``data/matches`` archives. ``MatchLibrary(root)`` keeps the old single-root
    behavior for tests and scripts that point at one directory explicitly.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        self._index: dict[str, Path] | None = None

    # --- discovery -------------------------------------------------------------------------------------------
    def _scan(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        if self.root is not None:
            if self.root.exists():
                for path in sorted(self.root.iterdir()):
                    if (path / "match.json").exists():
                        found[path.name] = path
            return found
        if LEGACY_MATCHES_ROOT.exists():
            for path in sorted(LEGACY_MATCHES_ROOT.iterdir()):
                if (path / "match.json").exists():
                    found[path.name] = path
        for manifest in discover_match_manifests():
            found.setdefault(manifest.parent.name, manifest.parent)
        return found

    def _index_map(self) -> dict[str, Path]:
        if self._index is None:
            self._index = self._scan()
        return self._index

    def path(self, match_id: str) -> Path:
        """The directory of a match, whether the id is an id or a path to a match directory."""
        candidate = Path(match_id)
        if (candidate / "match.json").exists():
            return candidate
        if self.root is not None:
            return self.root / match_id
        return self._index_map().get(match_id, LEGACY_MATCHES_ROOT / match_id)

    def list_ids(self) -> list[str]:
        if self.root is not None:
            if not self.root.exists():
                return []
            return sorted(p.name for p in self.root.iterdir() if (p / "match.json").exists())
        return sorted(self._index_map())

    def match_for_video(self, video: str | Path) -> str | None:
        """The id of the analysis saved beside this video, or ``None`` when it has not been analyzed yet."""
        directory = analysis_dir_for(video)
        return directory.name if (directory / "match.json").exists() else None

    # --- records ---------------------------------------------------------------------------------------------
    def create(self, source: str | Path, **kwargs) -> MatchRecord:
        """The match for this footage, creating its analysis directory *beside the video* if it does not exist.

        The directory is derived from the video, so creating twice from the same footage lands on the same place,
        and re-creating never rewrites the record (format, team names and the segment list survive).
        """
        if self.root is not None:
            match_id = analysis_id_for(source)
            directory = self.root / match_id
        else:
            directory = analysis_dir_for(source)
            match_id = directory.name
        if (directory / "match.json").exists():
            return self.load(match_id)
        directory.mkdir(parents=True, exist_ok=True)
        record = MatchRecord(match_id=match_id, sources=[str(source)], **kwargs)
        if self._index is not None:
            self._index[match_id] = directory
        self.save(record, directory=directory)
        return record

    def save(self, record: MatchRecord, directory: str | Path | None = None) -> Path:
        """Write the record, storing paths relative to the match directory when they lie inside it."""
        directory = Path(directory) if directory is not None else self.path(record.match_id)
        payload = asdict(record)
        payload["sources"] = [store_path(source, directory.parent.parent) for source in record.sources]
        payload["segments"] = [store_path(segment, directory) for segment in record.segments]
        path = directory / "match.json"
        _atomic_write(path, payload)
        return path

    def load(self, match_id: str) -> MatchRecord:
        directory = self.path(match_id)
        data = json.loads((directory / "match.json").read_text())
        record = MatchRecord(**data)
        # Repoint the recorded paths at wherever this match directory is now: the footage usually sits beside the
        # analysis (one level up), the segments usually inside it. A moved directory still opens its own files.
        record.sources = [resolve_path(source, directory.parent.parent) for source in record.sources]
        record.segments = [resolve_path(segment, directory) for segment in record.segments]
        return record

    def add_segment(self, match_id: str, segment_dir: str | Path) -> MatchRecord:
        """Remember that a segment directory belongs to this match, so the archive says what has been produced."""
        record = self.load(match_id)
        directory = self.path(match_id)
        entry = resolve_path(str(segment_dir), directory)
        if entry not in record.segments:
            record.segments.append(entry)
            self.save(record)
        return record

    def summaries(self) -> list[dict]:
        """One row per match for the dashboard's library list, with what has been produced so far."""
        rows = []
        for match_id in self.list_ids():
            directory = self.path(match_id)
            record = self.load(match_id)
            rows.append(
                {
                    "match_id": match_id,
                    "directory": str(directory),
                    "sources": record.sources,
                    "format": record.format,
                    "segments": len(record.segments),
                    "has_calibration": (directory / "calibration.json").exists(),
                    "has_report": (directory / "report.json").exists(),
                    "events": len(EventLog.load(directory / "events.json").events),
                    "highlights": len(list((directory / "highlights").glob("*.mp4"))) if (directory / "highlights").exists() else 0,
                    "created": record.created,
                }
            )
        return rows

    # --- artifacts -------------------------------------------------------------------------------------------
    def save_calibration(self, match_id: str, calibration: PitchCalibration) -> Path:
        path = self.path(match_id) / "calibration.json"
        _atomic_write(path, calibration.to_json())
        return path

    def load_calibration(self, match_id: str) -> PitchCalibration | None:
        path = self.path(match_id) / "calibration.json"
        if not path.exists():
            return None
        return PitchCalibration.from_json(json.loads(path.read_text()))

    def clear_calibration(self, match_id: str) -> bool:
        """Forget the saved calibration, for when the fit was no good. Returns whether there was one to remove."""
        path = self.path(match_id) / "calibration.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def save_clicks(
        self,
        match_id: str,
        clicks: list[dict],
        pitch: tuple[float, float] | None = None,
        source: str | None = None,
    ) -> Path:
        """Keep the landmark clicks next to the calibration they produced.

        Without this a bad fit cannot be looked at afterwards: the residual table in the page can only name a click
        by its landmark while the browser session that made it is still alive. ``source`` records the recording the
        clicks were made on: frames are indices into that recording's timeline, and restoring them against a
        different video puts the same frame number on different pictures.
        """
        path = self.path(match_id) / "clicks.json"
        _atomic_write(
            path,
            {"clicks": clicks, "pitch": list(pitch) if pitch else None, "source": source},
        )
        return path

    def load_clicks(self, match_id: str) -> list[dict]:
        path = self.path(match_id) / "clicks.json"
        if not path.exists():
            return []
        clicks = json.loads(path.read_text()).get("clicks", [])
        for click in clicks:  # labels from before the American spelling switch
            if click.get("label"):
                click["label"] = upgrade_landmark_label(click["label"])
        return clicks

    def load_clicks_source(self, match_id: str) -> str | None:
        """The recording the saved clicks were made on, when the archive recorded it.

        Legacy archives have no source and return ``None``. The dashboard then falls back to the old behavior
        (restore against whichever video is open), because refusing an unlabelled set would silently discard the
        only copy of a pre-upgrade calibration.
        """
        path = self.path(match_id) / "clicks.json"
        if not path.exists():
            return None
        source = json.loads(path.read_text()).get("source")
        return str(source) if source else None

    def save_report(self, match_id: str, payload: dict) -> Path:
        path = self.path(match_id) / "report.json"
        _atomic_write(path, payload)
        return path

    def load_report(self, match_id: str) -> dict | None:
        path = self.path(match_id) / "report.json"
        return upgrade_report_payload(json.loads(path.read_text())) if path.exists() else None

    def save_replay(self, match_id: str, payload: dict, boxes: dict | None = None) -> Path:
        """Per-frame track data for the animated pitch view (fetched by the browser as a media file).

        ``boxes`` - each track's own image boxes, ``{track_id: (N, 4)}`` - are written beside it as an ``npz``
        instead of inside the payload: the browser never draws them, and a whole game's worth is tens of megabytes
        of JSON that every view of the match would download. The centered-clip cutter reads them from here.
        """
        path = self.path(match_id) / "replay.json"
        _atomic_write(path, payload)
        if boxes is not None:
            np.savez_compressed(self.path(match_id) / "boxes.npz", **boxes)
        return path

    def load_replay_boxes(self, match_id: str) -> dict:
        """``{track_id: (N, 4)}`` boxes for the replay's tracks; {} when the report predates them."""
        path = self.path(match_id) / "boxes.npz"
        if not path.exists():
            return {}
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def load_replay(self, match_id: str) -> dict | None:
        path = self.path(match_id) / "replay.json"
        return upgrade_replay_payload(json.loads(path.read_text())) if path.exists() else None

    def save_jerseys(self, match_id: str, payload: dict) -> Path:
        """The background scanner's raw readings and per-track suggestions."""
        path = self.path(match_id) / "jerseys.json"
        _atomic_write(path, payload)
        return path

    def load_jerseys(self, match_id: str) -> dict:
        path = self.path(match_id) / "jerseys.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_roster(self, match_id: str, roster: dict[int, dict]) -> Path:
        """Manual identity per track: the user's own reads of shirt numbers and names, which win over the scanner."""
        path = self.path(match_id) / "roster.json"
        _atomic_write(path, {"players": {str(track): entry for track, entry in roster.items()}})
        return path

    def load_roster(self, match_id: str) -> dict[int, dict]:
        path = self.path(match_id) / "roster.json"
        if not path.exists():
            return {}
        players = json.loads(path.read_text()).get("players", {})
        return {int(track): entry for track, entry in players.items()}

    def load_jerseys_status(self, match_id: str) -> dict:
        """Progress of the background shirt-number scan: {} until it has ever run."""
        path = self.path(match_id) / "jerseys_status.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def load_audio_scan_status(self, match_id: str) -> dict:
        """Progress of the background whistle scan: {} until it has ever run."""
        path = self.path(match_id) / "audio_scan.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def identities_dir(self, match_id: str) -> Path:
        """Where a still of each appearance is kept, so recognizing who a track is costs one seek and no scan."""
        path = self.path(match_id) / "identities"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def events(self, match_id: str) -> EventLog:
        return EventLog.load(self.path(match_id) / "events.json")

    def save_events(self, match_id: str, log: EventLog) -> Path:
        path = self.path(match_id) / "events.json"
        log.save(path)
        return path

    def highlights_dir(self, match_id: str) -> Path:
        path = self.path(match_id) / "highlights"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts(self, match_id: str) -> list[dict]:
        """Everything on disk for a match, with sizes, for a 'what did we produce?' view."""
        directory = self.path(match_id)
        out = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix != ".tmp":
                out.append({"path": str(path.relative_to(directory)), "size_mb": round(path.stat().st_size / 1e6, 2)})
        return out

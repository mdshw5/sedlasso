"""The annotated match stream: every frame of the analyzed window, drawn with what the pipeline knows.

This is the moving-picture sibling of the pitch replay. Where that view draws measured pitch positions on a
diagram, this one serves the footage itself as MJPEG - each frame carries:

* the **pitch model**, projected back into the picture through the calibration's corrected camera chain (the
  touchlines, the halfway line, both boxes, the circles and the spots) - the same overlay the calibration view
  uses to show a fit against the real markings, per frame;
* the **detection boxes** of every field player the tracker follows, in that player's measured team color;
* the **ball**, from the segment's ball scan, drawn only where a detector saw it - a filled dot in a small box,
  never a position the tracker only forecast, so the mark always means "seen right here";
* a **label chip** on each player - the shirt number the roster or the OCR scan assigned, or the track id when
  nobody has named them - so an appearance in the footage can be matched to their row in the dashboard tables.

What it deliberately does not do is invent: a player without a number shows their track id, not a guess; a ball
position the scan did not actually detect is left unmarked, forecast or not; and a replay that predates the
per-player boxes is refused with the one command that rebuilds it, rather than streaming boxes that would be
silently absent.

The HTTP surface lives in :class:`MatchStreamServer` (index page, ``/matches``, ``/stream/<id>.mjpg`` and a
single-frame ``/frame/<id>.jpg``); ``scripts/run_match_stream.py`` is the process that runs it. Each viewer gets
their own ffmpeg decode, so streams can start at different times and speeds.
"""

from __future__ import annotations

import errno
import html
import json
import os
import select
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis.identity import numbers_are_stale
from soccer_analytics.analysis.library import MatchLibrary, MatchRecord, resolve_path
from soccer_analytics.analysis.projection import segment_poses
from soccer_analytics.analysis.rosters import RosterLibrary, team_rosters_for
from soccer_analytics.analysis.stage_a import load_segment
from soccer_analytics.dashboard.pitch_clicks import pitch_marking_polylines
from soccer_analytics.dashboard.video import (
    ClipCache,
    CODECS,
    DEFAULT_CLIP_SECONDS,
    DEFAULT_LIVE_WIDTH,
    MAX_CLIP_SECONDS,
    MAX_VIDEO_WIDTH,
    VideoError,
    clip_key,
    encode_clip,
    iter_live_chunks,
)
from soccer_analytics.geometry.pitch_calibration import pitch_to_pixels
from soccer_analytics.ingest.ffmpeg_reader import grab_frame
from soccer_analytics.ingest.source import open_reader, source_from_record

REPO_ROOT = Path(__file__).resolve().parents[3]
MATCHES_ROOT = REPO_ROOT / "data" / "matches"
BALL_TRACK_FILE = "ball_track.json"

# Decode/render defaults. 1600 px wide keeps every label readable and every player box several pixels across,
# while staying cheap enough that a 4x stream renders well inside one core.
DEFAULT_WIDTH = 1600
MIN_WIDTH = 320
MAX_WIDTH = 2560
MARK_FPS = 8.0  # the Step 1 marking stream: watchable enough to spot kick-off, cheap enough to decode live
DEFAULT_RATE = 1.0
MAX_RATE = 8.0
MIN_RATE = 0.25
MAX_CONCURRENT_STREAMS = 3
# A frame write blocking this long means the client is not reading (see MatchStreamHandler._stream).
STREAM_WRITE_TIMEOUT_S = 15.0
# How long a new stream waits for a decode slot before being refused. A restart (a seek, an overlay toggle)
# overlaps the previous connection for a moment, and that connection's slot is only freed when the server
# notices the client is gone - the write watchdog's job, which is quicker than any seek. Waiting absorbs the
# overlap; the component retries with backoff if even that is not enough.
STREAM_SLOT_WAIT_S = 4.0

# The overlay layers a frame can carry, in draw order, and the ``overlays`` query parameter that selects them:
# absent means all on, ``none`` means a clean picture, else a comma-separated list of these names (unknown names
# are ignored). Each layer is independent so the pane's toggles can ask for any combination - boxes without
# numbers, the pitch model alone, or no annotation at all. ``hud`` is what a viewer wants while watching (the
# clock and who is playing); ``debug`` is what a debugger wants (session name, frame counter, notes).
OVERLAY_NAMES = ("pitch", "boxes", "numbers", "ball", "hud", "debug")
ALL_OVERLAYS = {name: True for name in OVERLAY_NAMES}


def parse_overlays(value: str | None) -> dict[str, bool]:
    """Which overlay layers a request asked for; the parameter's absence asks for all of them.

    An *empty* value cannot be used to ask for none: ``parse_qs`` drops blank query values, so ``overlays=``
    arrives as "the parameter is absent", which must keep meaning "everything on" - the pre-toggle behavior
    every existing link and the index page's stills rely on. "none" is the explicit spelling for a clean frame.
    """
    if value is None:
        return dict(ALL_OVERLAYS)
    wanted = {part.strip().lower() for part in value.split(",") if part.strip()}
    if "none" in wanted:
        return {name: False for name in OVERLAY_NAMES}
    return {name: name in wanted for name in OVERLAY_NAMES}

JPEG_QUALITY = 85
FONT = cv2.FONT_HERSHEY_SIMPLEX

# The port and base URL the dashboard and its components talk to this server on. ``SOCCER_STREAM_PORT`` moves the
# port (the dashboard checks that port before offering the pane); ``SOCCER_STREAM_BASE`` overrides the base URL the
# browser builds, for when the server sits behind a proxy that is not the dashboard's own host and port.
DEFAULT_PORT = 8510
PORT_ENV = "SOCCER_STREAM_PORT"
BASE_ENV = "SOCCER_STREAM_BASE"


def configured_port() -> int:
    """The port the stream server listens on: ``SOCCER_STREAM_PORT`` when set and readable, else the default."""
    try:
        return int(os.environ.get(PORT_ENV, DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT


def configured_base() -> str:
    """The base URL override for the browser's stream requests, or "" (derive it from the dashboard's host)."""
    return os.environ.get(BASE_ENV, "")


def is_reachable(port: int, timeout_s: float = 0.2) -> bool:
    """Whether something is listening on a local port - a non-blocking connect, not a process scan.

    Used by the dashboard to say whether the footage pane has a server to talk to, on every rerun, so it must not
    block: a refused connection (nothing listening) returns immediately, and only a connection that is genuinely
    in progress the rare time it is allowed to wait.
    """
    with socket.socket() as sock:
        sock.setblocking(False)
        code = sock.connect_ex(("127.0.0.1", int(port)))
        if code == 0:
            return True
        if code not in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY):
            return False
        _, writable, _ = select.select([], [sock], [], timeout_s)
        return bool(writable) and sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0


class StreamError(RuntimeError):
    """The match cannot be streamed as asked (missing or outdated artifacts, bad start time)."""


class StreamControl:
    """The knobs a client can turn on its own stream while the MJPEG connection stays open.

    Right now that is the stop flag, and it exists because a browser will not abort an MJPEG fetch on command:
    when the pane replaces its ``<img>`` (a seek, an overlay change, a pause) Chromium keeps *draining* the old
    connection - bytes flowing, nobody watching - so the decoder would run until the write watchdog or the tab
    noticed. The component sends ``/stop?token=...`` before switching away; the handler checks the flag between
    frames and ends the stream itself, which closes the connection deterministically for both ends.
    """

    def __init__(self) -> None:
        self._stopped = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stopped = True

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped


@dataclass(frozen=True)
class TeamStyle:
    """One team's display color and name, resolved once so every frame paints them the same."""

    name: str
    bgr: tuple[int, int, int]


# Fallbacks when the kit clustering could not separate a team's color (the same situation the pitch view falls
# back on its own palette for). BGR, because that is what OpenCV draws with.
FALLBACK_BGR = ((60, 60, 230), (230, 130, 60))
OTHER_BGR = (150, 150, 150)
# Roles are color-coded rather than spelled into the chips (the user's call), matching the pane's ring
# colors so the two views name the same person the same way. BGR order, unlike the pane's hex.
ROLE_BGR = {"referee": (247, 85, 168), "goalkeeper": (11, 158, 245)}  # #a855f7, #f59e0b
PITCH_BGR = (0, 200, 255)
INK = (30, 30, 30)
PAPER = (245, 245, 245)


def _bgr(rgb) -> tuple[int, int, int]:
    r, g, b = (int(np.clip(int(channel), 0, 255)) for channel in rgb)
    return (b, g, r)


def _text_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever stays readable on this chip color."""
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    return INK if luminance > 150 else PAPER


def chip_text(number, name: str, track_id: int, *, debug: bool = False) -> tuple[str, str]:
    """The label chip's two lines for one player: what they are called, and how to find them in the tables.

    The number is a jersey *detection* (only the scan supplies one - see :func:`load_numbers`): a number drawn on
    the footage is a claim about what the camera saw. The track id is the tracker's own key - it is how every
    table and clip names the player, but a viewer calls them by their number - so it is debug information: with
    ``debug`` it heads an unidentified player's chip (``#1234``, which can never be mistaken for a worn number)
    and sits under an identified one (``track 1234``); without it, a player with neither a number nor a name gets
    no chip at all, and the box alone says the one thing that is known.
    """
    name = (name or "").strip()
    detail = f"track {track_id}" if debug else ""
    if number:
        return f"{int(number)} {name}".strip(), detail
    if name:
        return name, detail
    return (f"#{track_id}", "") if debug else ("", "")


def ball_stamps(records: list[dict], frame_count: int) -> np.ndarray:
    """The ball scan's records as ``(F, 3)`` arrays of ``u, v, measured``, NaN where there is nothing to draw.

    Only ``tracking`` - a detection, ``measured=1`` - carries a position. ``coasting`` is the tracker's own
    forecast across a miss: the position was predicted, not seen, and the stream draws the ball only where a
    detector saw it, so a coast is not carried here at all. ``lost`` and ``out_of_view`` have no position
    either.
    """
    out = np.full((frame_count, 3), np.nan)
    for record in records:
        frame = int(record.get("i", -1))
        status = record.get("status")
        u, v = record.get("u"), record.get("v")
        if not 0 <= frame < frame_count or status != "tracking" or u is None or v is None:
            continue
        out[frame] = (float(u), float(v), 1.0)
    return out


def load_numbers(
    library: MatchLibrary,
    match_id: str,
    track_ids,
    teams: dict[int, int] | None = None,
    team_rosters: dict[int, dict[int, str]] | None = None,
) -> tuple[dict[int, dict], dict]:
    """Shirt numbers per track for the footage, plus the raw scan payload.

    The number a chip draws comes from the jersey scan only: a number on the footage is a claim about what the
    camera saw, and only a detection saw it. A name is not read off a shirt, so it can come from the manual
    roster (a per-track correction) or from the *team roster* linked to the track's team (``analysis.rosters``:
    number -> name, typed once per squad and reused across matches) - the manual name wins when both exist.
    The staleness note is recomputed from the same scan payload the numbers came from, so a running stream can
    never disagree with itself.

    One reader for the two callers that need the same answer: the initial load, and the per-request refresh that
    keeps a running stream in step with roster edits made in the dashboard beside it.
    """
    teams = teams or {}
    team_rosters = team_rosters or {}
    jerseys = library.load_jerseys(match_id)
    suggestions = {int(track): entry for track, entry in (jerseys.get("suggestions") or {}).items()}
    roster = {int(track): entry for track, entry in (library.load_roster(match_id) or {}).items()}
    numbers: dict[int, dict] = {}
    for track_id in track_ids:
        track = int(track_id)
        suggestion = suggestions.get(track) or {}
        manual = roster.get(track) or {}
        detected = suggestion.get("number")
        name = str(manual.get("name") or "").strip()
        if not name and detected:
            name = str((team_rosters.get(int(teams.get(track, -1))) or {}).get(int(detected)) or "")
        if not detected and not name:
            continue
        numbers[track] = {"number": detected, "name": name, "source": "scan" if detected else "roster"}
    return numbers, jerseys


def numbers_note(match_dir: Path, jerseys: dict, numbers: dict[int, dict], player_ids) -> str | None:
    """Why the assigned shirt numbers cannot be trusted for this build, or None when they can.

    Shirt numbers are attached to track ids, and a rebuild or a re-fit can move players between tracks - the same
    staleness the dashboard warns about (``identity.numbers_are_stale``). Computed from live files rather than
    stored, so the note a running stream shows is about the numbers it is actually drawing.
    """
    if not numbers:
        return "no shirt numbers assigned yet (roster empty and no scan)"
    scan_saved = (jerseys.get("meta") or {}).get("calibration_saved")
    calibration_path = match_dir / "calibration.json"
    stale = numbers_are_stale(
        list((jerseys.get("suggestions") or {}).keys()),
        list(player_ids),
        scan_calibration_saved=None if scan_saved is None else float(scan_saved),
        calibration_saved=calibration_path.stat().st_mtime if calibration_path.exists() else None,
    )
    return f"shirt numbers look stale: {stale}" if stale else None


def _outline_text(frame: np.ndarray, text: str, org, color: tuple[int, int, int], font: float) -> None:
    """A text line with a dark outline, so it reads over whatever the camera saw."""
    cv2.putText(frame, text, org, FONT, font, INK, 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, font, color, 1, cv2.LINE_AA)


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """The unit quaternion ``(w, x, y, z)`` of a 3x3 rotation, by the stable branch of the standard conversion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([s / 4.0, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))  # one diagonal dominates: its branch cancels least
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2.0
    quat = np.zeros(4)
    quat[0] = (m[k, j] - m[j, k]) / s
    quat[i + 1] = s / 4.0
    quat[j + 1] = (m[j, i] + m[i, j]) / s
    quat[k + 1] = (m[k, i] + m[i, k]) / s
    return quat


def _matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """A rotation between two 3x3 rotations along the shortest arc - the camera does not go the long way round."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    a = _quaternion_from_matrix(first)
    b = _quaternion_from_matrix(second)
    if float(a @ b) < 0.0:
        b = -b  # q and -q are the same rotation; the sign must be fixed or the arc is the far side
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 0.9995:  # so close that sin() carries no useful fraction: normalized linear is exact enough
        return _matrix_from_quaternion(a + alpha * (b - a))
    theta = float(np.arccos(dot))
    return _matrix_from_quaternion((np.sin((1 - alpha) * theta) * a + np.sin(alpha * theta) * b) / np.sin(theta))


def _fits_replay(meta_path: Path, replay: dict) -> bool:
    """Whether a segment's frame grid is the one the replay payload was built on.

    The stream draws the replay's per-frame boxes and the segment's per-frame ball onto the segment's own frame
    grid; a segment of another window or rate would put them at the wrong times. The two numbers that pin a
    segment's grid - its analysis rate and frame count - are both stored in its meta and baked into the replay
    at build time, so they are what a candidate is judged by.
    """
    try:
        meta = json.loads(meta_path.read_text())
        return (
            abs(float(meta["fps"]) - float(replay["fps"])) < 1e-6
            and abs(int(meta["total_frames"]) - int(replay["frame_count"])) <= 2
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _segment_for(record: MatchRecord, replay: dict | None = None) -> Path:
    """The record's segment a full-fidelity reader should open - for the stream, the one the replay belongs to.

    ``MatchLibrary.add_segment`` appends: the segments list is an archive, oldest first, so blindly taking the
    first entry (as this stream used to) serves the OLDEST run of the footage. That is how the ball silently
    disappeared: the oldest run predated the ball scan, and the scan written beside a later run was never
    opened. The right segment is the one the replay was built from - the most recently added run on the same
    frame grid (:func:`_fits_replay`), the last match winning because runs are appended as they start. A
    record whose segments all disagree with the replay still gets its newest usable one, and a record with a
    single segment behaves exactly as it always did.
    """
    seen: set[str] = set()
    usable: list[Path] = []
    fitting: Path | None = None
    for entry in record.segments:
        directory = Path(entry)
        key = os.path.normpath(str(directory))
        if key in seen:
            continue
        seen.add(key)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        usable.append(directory)
        if replay is not None and _fits_replay(meta_path, replay):
            fitting = directory
    if fitting is not None:
        return fitting
    return usable[-1] if usable else Path(record.segments[0])


class AnnotatedMatch:
    """One match, ready to draw: boxes by frame, ball by frame, the camera chain and the calibration.

    Everything heavy (segment load, pose chain, replay parse) happens once here; :meth:`render` then only reads
    arrays and is safe to call from several stream threads at once.
    """

    def __init__(
        self,
        *,
        match_id: str,
        video: str | Path,
        fps: float,
        start_s: float,
        frame_count: int,
        native: tuple[int, int, float] = (0, 0, 0.0),
        pitch: tuple[float, float],
        format_name: str | None = None,
        calibration,
        q: np.ndarray,
        focal: np.ndarray,
        players: list[dict],
        boxes: dict,
        numbers: dict[int, dict],
        numbers_note: str | None,
        ball_records: list[dict],
        teams: list[TeamStyle],
        notes: list[str],
    ):
        self.match_id = match_id
        self.video = Path(video)
        self.fps = float(fps)
        self.start_s = float(start_s)
        self.frame_count = int(frame_count)
        # What the source video actually is - 3840x2160 at 60 fps on the real matches. The encoded video routes
        # default to it ("full framerate and resolution" is the point of them); 0 means the segment never said.
        self.native_width = int(native[0]) or 0
        self.native_height = int(native[1]) or 0
        self.native_fps = float(native[2]) or self.fps
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.format_name = format_name
        self.calibration = calibration
        self.q = q
        self.focal = focal
        self.teams = teams
        self.notes: list[str] = []
        # The notes that never change (bystanders hidden, no ball scan); the numbers note is recomputed by
        # ``refresh_numbers`` because the roster and the scan are edited beside a running stream.
        self._static_notes = list(notes)
        self._player_ids = [int(player["track_id"]) for player in players]
        # Each player's team index, so a team roster can name the player behind a scanned number.
        self._teams = {int(player["track_id"]): int(player.get("team", -1)) for player in players}
        self.pitch_lines = pitch_marking_polylines(*self.pitch, format_name=self.format_name)

        # Boxes and labels, indexed by analysis frame: exactly what one rendered frame has to walk. The boxes
        # arrive separately from the payload (the ``boxes.npz`` beside it); each player's array is aligned with
        # the ``frames`` the payload lists, which is the one invariant the pairing relies on.
        boxes = {int(track): np.asarray(array, dtype=np.float64) for track, array in boxes.items()}
        self._players = players
        self.boxed_tracks: set[int] = set()
        # The payload's role labels (referee/goalkeeper), if the build had them; boxes for these tracks draw in
        # their role color instead of the team color, so a viewer can tell the officials from the sides.
        self.roles: dict[int, str] = {
            int(player["track_id"]): str(player["role"]) for player in players if player.get("role")
        }
        players_by_frame: list[list[tuple]] = [[] for _ in range(self.frame_count)]
        for player in players:
            track_id, team = int(player["track_id"]), int(player["team"])
            positions = boxes.get(track_id)
            if positions is None or len(positions) == 0:
                continue  # a track built without detections (or a payload from an older build): nothing to draw
            self.boxed_tracks.add(track_id)
            for frame, box in zip(player["frames"], positions):
                frame = int(frame)
                if 0 <= frame < self.frame_count:
                    x1, y1, x2, y2 = (float(v) for v in box)
                    players_by_frame[frame].append((x1, y1, x2, y2, track_id, team))
        self.refresh_numbers(numbers, numbers_note)
        self.players_by_frame = players_by_frame
        self.ball = ball_stamps(ball_records, self.frame_count)

    def refresh_numbers(self, numbers: dict[int, dict], numbers_note: str | None) -> None:
        """Rebuild the label chips and the numbers note from a fresh assignment (both are edited live).

        Called on every stream request by the server, because the dashboard edits the roster and runs the scan
        beside the match: a chip still showing a number the user just removed would read as the stream ignoring
        them, and a stale-numbers warning about numbers the stream no longer draws would be worse. The new
        mapping is built before the swap, so a frame rendering concurrently keeps a consistent dict either way.
        """
        identities: dict[int, tuple[str, str]] = {}
        debug_identities: dict[int, tuple[str, str]] = {}
        for track_id in self.boxed_tracks:
            entry = numbers.get(track_id) or {}
            identities[track_id] = chip_text(entry.get("number"), entry.get("name"), track_id)
            debug_identities[track_id] = chip_text(entry.get("number"), entry.get("name"), track_id, debug=True)
        self.identities = identities
        self.debug_identities = debug_identities
        self.notes = self._static_notes + ([numbers_note] if numbers_note else [])

    # ----------------------------------------------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, match_id: str, *, root: str | Path | None = None, segment: str | Path | None = None) -> "AnnotatedMatch":
        """Load a match from the archive; raises :class:`StreamError` when it cannot be streamed as asked.

        The per-player image boxes do not travel inside the replay payload (they would be tens of megabytes of
        JSON the browser never draws); they live beside it in ``boxes.npz``, and a replay without them is refused
        with the one command that rebuilds both - a stream of boxes labeled "missing" would be worse than one
        that says why it cannot start. A payload that still carries its boxes inline (a build from before the
        sidecar) is tolerated so the stream works across the transition. The segment served is the one the replay
        was built from, not blindly the record's oldest entry (:func:`_segment_for`): the per-frame ball scan
        lives beside the segment, and the oldest run's missing scan is exactly how the ball silently disappeared
        from this stream once. The jersey numbers come from the OCR scan and the names from the manual roster or
        the *team rosters* linked to the match (the same merge rule the dashboard applies), so both views name a
        player the same way; a scan built against an older fit is called out in the notes rather than silently
        mismapping numbers onto the wrong people.
        """
        library = MatchLibrary(root)
        match_dir = library.path(match_id)
        if not match_dir.exists():
            raise StreamError(f"no such match: {match_id}")
        record = library.load(match_id)
        calibration = library.load_calibration(match_id)
        if calibration is None:
            raise StreamError(f"{match_id} has no calibration: register the pitch in the dashboard first")
        replay = library.load_replay(match_id)
        if replay is None or not replay.get("players"):
            raise StreamError(f"{match_id} has no replay built - run scripts/rebuild_match.py --match {match_id}")
        boxes = library.load_replay_boxes(match_id)
        if not boxes:
            boxes = {
                int(player["track_id"]): player["boxes"]
                for player in replay["players"]
                if player.get("boxes")
            }
        if not boxes:
            raise StreamError(
                f"{match_id}'s replay has no per-player image boxes (no boxes.npz, or one that predates them) - "
                f"run scripts/rebuild_match.py --match {match_id}"
            )
        segment_dir = Path(segment) if segment else _segment_for(record, replay)
        meta_path = segment_dir / "meta.json"
        if not meta_path.exists():
            raise StreamError(f"segment not found for {match_id}: {segment_dir}")
        segment = load_segment(segment_dir)
        q, focal = segment_poses(segment)

        teams = {int(p["track_id"]): int(p["team"]) for p in replay["players"]}
        numbers, jerseys = load_numbers(
            library,
            match_id,
            [int(p["track_id"]) for p in replay["players"]],
            teams=teams,
            team_rosters=team_rosters_for(RosterLibrary(), record.team_rosters),
        )

        ball_records: list[dict] = []
        try:
            ball_records = json.loads((segment_dir / BALL_TRACK_FILE).read_text()).get("frames") or []
        except (OSError, json.JSONDecodeError):
            pass

        notes: list[str] = []
        excluded = int(replay.get("bystanders_excluded") or 0)
        if excluded:
            notes.append(f"{excluded} off-field tracks hidden (coaches/spectators)")
        if not ball_records:
            # Judged from the segment just opened, not from the replay payload: the two disagreed silently once
            # (the segment fix above), and this note must describe what the stream is actually drawing.
            notes.append("no ball scan for this segment")

        team_names = list(replay.get("team_names") or record.team_names)
        team_colors = list(replay.get("team_colors") or [])
        teams: list[TeamStyle] = []
        for team in (0, 1):
            color = team_colors[team] if team < len(team_colors) else None
            if color is None:
                bgr = FALLBACK_BGR[team]
            else:
                bgr = _bgr(color)
            name = str(team_names[team]) if team < len(team_names) else f"Team {team + 1}"
            teams.append(TeamStyle(name=name, bgr=bgr))

        # The segment remembers the filming machine's own path for the video it was built from, and a legacy
        # absolute path outlives the disk it was written on when the archive moves beside its footage. Re-point
        # it the way the record's sources are: look the stored name up beside the analysis and beside the
        # footage, and only then accept the record's own resolved source as the fallback.
        video = str(segment.meta["video"])
        if not Path(video).exists():
            for base in (match_dir, match_dir.parent.parent):
                candidate = Path(resolve_path(video, base))
                if candidate.exists():
                    video = str(candidate)
                    break
            else:
                reachable = next((source for source in record.sources if Path(source).exists()), None)
                if reachable is not None:
                    video = str(reachable)

        return cls(
            match_id=match_id,
            video=video,
            fps=float(replay.get("fps") or segment.meta["fps"]),
            start_s=float(segment.meta["start_s"]),
            frame_count=int(replay.get("frame_count") or len(segment.time)),
            native=(
                int(segment.meta.get("width") or 0),
                int(segment.meta.get("height") or 0),
                float(segment.meta.get("source_fps") or 0.0),
            ),
            pitch=(replay["pitch"][0], replay["pitch"][1]),
            format_name=record.format,
            calibration=calibration,
            q=q,
            focal=focal,
            players=replay["players"],
            boxes=boxes,
            numbers=numbers,
            numbers_note=numbers_note(match_dir, jerseys, numbers, [int(p["track_id"]) for p in replay["players"]]),
            ball_records=ball_records,
            teams=teams,
            notes=notes,
        )

    # ----------------------------------------------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------------------------------------------
    def source_time(self, index: int) -> float:
        """Wall time in the source video for an analysis frame (the segment clock in seconds)."""
        return self.start_s + index / self.fps

    def index_for(self, start_s: float) -> int:
        """The first analysis frame at or after a source-clock time, clamped to the analyzed window."""
        index = int(round((float(start_s) - self.start_s) * self.fps))
        if index >= self.frame_count:
            raise StreamError(
                f"start time {start_s:.1f}s is past the end of the analyzed window "
                f"(ends at {self.source_time(self.frame_count - 1):.0f}s)"
            )
        return max(0, index)

    def _neighbors(self, position: float) -> tuple[int, int, float]:
        """The two analysis samples around a fractional position and where between them it sits."""
        p = float(np.clip(position, 0.0, max(0.0, self.frame_count - 1)))
        earlier = int(p)
        return earlier, min(earlier + 1, self.frame_count - 1), p - earlier

    def players_at(self, position: float) -> list[tuple]:
        """The boxes at a fractional analysis position, interpolated between the neighboring samples.

        A track seen in both samples moves linearly between them; a track seen only in the earlier one holds its
        last box (a player briefly unobserved is not teleported); a track seen only in the later one is not drawn
        yet - the same appearance it has at the analysis rate, when the sample it appears on is reached.
        """
        earlier, later, alpha = self._neighbors(position)
        before = {item[4]: item for item in self.players_by_frame[earlier]}
        if later == earlier or alpha <= 0.0:
            return list(before.values())
        after = {item[4]: item for item in self.players_by_frame[later]}
        out = []
        for track_id, (x1, y1, x2, y2, _track, team) in before.items():
            following = after.get(track_id)
            if following is None:
                out.append((x1, y1, x2, y2, track_id, team))
            else:
                out.append(
                    (
                        x1 + (following[0] - x1) * alpha,
                        y1 + (following[1] - y1) * alpha,
                        x2 + (following[2] - x2) * alpha,
                        y2 + (following[3] - y2) * alpha,
                        track_id,
                        team,
                    )
                )
        return out

    def ball_at(self, position: float) -> tuple[float, float, float]:
        """The ball at a fractional analysis position, only where a detector actually saw it.

        Between two *neighboring sightings* the position interpolates (a 30 or 60 fps render moves the dot the
        way it moves the boxes); anywhere else there is nothing to draw - before the first sighting and after
        the last, and across any sample the scan did not detect the ball on. A forecast is never drawn: the mark
        means "a detector saw the ball right here". The *measured* flag rides along for the caller.
        """
        earlier, later, alpha = self._neighbors(position)
        u0, v0, m0 = self.ball[earlier]
        if not (np.isfinite(u0) and np.isfinite(v0)):
            return float("nan"), float("nan"), float("nan")
        if alpha > 0.0:
            u1, v1, m1 = self.ball[later]
            if not (np.isfinite(u1) and np.isfinite(v1)):
                return float("nan"), float("nan"), float("nan")
            return (u0 + (u1 - u0) * alpha, v0 + (v1 - v0) * alpha, m0 if alpha < 0.5 else m1)
        return float(u0), float(v0), float(m0)

    def pose_at(self, position: float) -> tuple[np.ndarray, float]:
        """The camera pose at a fractional analysis position: the chain slerps, the focal interpolates."""
        earlier, later, alpha = self._neighbors(position)
        if later == earlier or alpha <= 0.0:
            return self.q[earlier], float(self.focal[earlier])
        return (
            slerp_rotation(self.q[earlier], self.q[later], alpha),
            float(self.focal[earlier]) + (float(self.focal[later]) - float(self.focal[earlier])) * alpha,
        )

    def render(
        self,
        frame: np.ndarray,
        index: int,
        *,
        pitch: bool = True,
        boxes: bool = True,
        numbers: bool = True,
        ball: bool = True,
        hud: bool = True,
        debug: bool = True,
        note: str = "",
    ) -> np.ndarray:
        """Draw one analysis frame's overlay onto ``frame`` (in place) and return it.

        Every layer is optional and independent - the pane's toggles pass one :func:`parse_overlays` mapping
        straight through - so "no boxes but keep the numbers", "the pitch model alone" and "the picture with
        no diagnostics" are all expressible. ``hud`` is the watching half (who is playing, the clock, and
        ``note``, the playback speed); ``debug`` is the diagnosing half (session name, frame counter, notes).
        """
        if pitch:
            self._draw_pitch(frame, index)
        if boxes or numbers:
            self._draw_players(frame, index, boxes=boxes, numbers=numbers, debug=debug)
        if ball:
            self._draw_ball(frame, index)
        if hud:
            self._draw_hud(frame, index, note=note, session_line=debug)
        if debug:
            self._draw_debug(frame, index)
        return frame

    def render_at(
        self,
        frame: np.ndarray,
        at_s: float,
        *,
        pitch: bool = True,
        boxes: bool = True,
        numbers: bool = True,
        ball: bool = True,
        hud: bool = True,
        debug: bool = True,
        note: str = "",
    ) -> np.ndarray:
        """Draw the overlay for any source-clock time, not only the analysis grid - the full-rate entry point.

        The analysis samples the match at ``fps`` (5 a second on the real matches); a frame between two samples
        has no boxes of its own. Everything that moves - the boxes, the camera pose, and the ball between two
        sightings - interpolates between the neighboring samples (see :meth:`players_at` and :meth:`ball_at`),
        which is what makes a 60 fps render a real 60 fps rather than each sample held for twelve frames. The
        text layers stay on the nearer sample: they name where the analysis is, and that stays true between its
        frames.
        """
        position = (float(at_s) - self.start_s) * self.fps
        earlier, _later, _alpha = self._neighbors(position)
        if pitch:
            q, focal = self.pose_at(position)
            self._draw_pitch(frame, earlier, q=q, focal=focal)
        if boxes or numbers:
            self._draw_players(
                frame, earlier, boxes=boxes, numbers=numbers, players=self.players_at(position), debug=debug
            )
        if ball:
            self._draw_ball(frame, earlier, stamp=self.ball_at(position))
        if hud:
            self._draw_hud(frame, earlier, note=note, session_line=debug)
        if debug:
            self._draw_debug(frame, earlier)
        return frame

    def _draw_pitch(self, frame: np.ndarray, index: int, *, q=None, focal=None) -> None:
        """The pitch markings, projected through the corrected chain at this frame's pose.

        Drawn on a copy and blended, so the overlay reads as a measurement layered on the picture rather than
        paint over it. A point near the horizon projects to absurd coordinates; instead of drawing a line across
        the whole frame from it, vertices outside a generous picture margin are skipped - the markings that are
        actually visible are still drawn, segment by segment. ``q``/``focal`` override the stored pose (the
        full-rate path passes an interpolated one); ``index`` stays the integer sample the correction rides on.
        """
        height, width = frame.shape[:2]
        index = int(np.clip(index, 0, len(self.q) - 1))
        q_frame, focal_frame = self.calibration.corrected_frame(
            self.q[index] if q is None else np.asarray(q, dtype=np.float64),
            float(self.focal[index]) if focal is None else float(focal),
            index,
        )
        overlay = frame.copy()
        thickness = max(1, int(round(height / 420.0)))
        for points in self.pitch_lines:
            uv, in_front = pitch_to_pixels(self.calibration, points, q_frame, focal_frame)
            pixels = uv * width
            for k in range(len(points) - 1):
                if not (in_front[k] and in_front[k + 1]):
                    continue
                a, b = pixels[k], pixels[k + 1]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                sane = lambda p: -width <= p[0] <= 2 * width and -height <= p[1] <= 2 * height  # noqa: E731
                if not (sane(a) and sane(b)):
                    continue
                cv2.line(overlay, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), PITCH_BGR, thickness)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def _draw_players(
        self,
        frame: np.ndarray,
        index: int,
        *,
        boxes: bool = True,
        numbers: bool = True,
        players=None,
        debug: bool = False,
    ) -> None:
        """The players' detection boxes and their label chips - two layers, because a viewer may want either.

        With boxes off but numbers on, the chips still sit where their box would be: the label without the
        rectangle is exactly the "who is that" view, and it needs no different anchoring. ``players`` overrides
        the frame's own observations (the full-rate path passes boxes interpolated between samples), and
        ``debug`` decides whether a track id may appear: a player with no detected number and no name draws no
        chip at all without it - the box alone is the honest label for "a person, unknown".
        """
        height, width = frame.shape[:2]
        scale = height / 1080.0
        for x1, y1, x2, y2, track_id, team in (self.players_by_frame[index] if players is None else players):
            left, top = int(x1 * width), int(y1 * width)
            right, bottom = int(x2 * width), int(y2 * width)
            role = self.roles.get(track_id)
            if role in ROLE_BGR:
                color = ROLE_BGR[role]
            elif 0 <= team < len(self.teams):
                color = self.teams[team].bgr
            else:
                color = OTHER_BGR
            if boxes:
                thickness = max(2, int(round(3 * scale)))
                cv2.rectangle(frame, (left, top), (right, bottom), INK, thickness + 2)
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            box_height = abs(bottom - top)
            if not numbers or box_height < 22 * scale:
                continue  # too small on screen for a readable chip; the box alone carries the detection
            main, detail = (self.debug_identities if debug else self.identities).get(track_id, ("", ""))
            if not main:
                continue  # debug off and nothing was read or named: the box is the label
            font = max(0.4, 0.62 * scale)
            small = max(0.35, 0.42 * scale)
            show_detail = bool(detail) and box_height >= 60 * scale
            main_height = _chip_height(main, font)
            detail_height = _chip_height(detail, small) if show_detail else 0
            x = _clamp_chip_x(frame, left, max(_chip_width(main, font), _chip_width(detail, small) if show_detail else 0))
            # min(): the simulator (and a rare real detection) writes the foot where the top should be; the chip
            # belongs above the box *as drawn*, so it anchors to whichever edge is higher, not to y1.
            y = min(top, bottom) - main_height - detail_height - 3
            if y < 2:  # no room above the box: tuck the chip just inside the top of the frame
                y = 2
            if show_detail:
                _chip(frame, x, y, detail, color, small)
            _chip(frame, x, y + detail_height, main, color, font)

    def _draw_ball(self, frame: np.ndarray, index: int, *, stamp=None) -> None:
        """The ball, only where a detector saw it: a small box (the "detection box" a viewer expects) with the
        ball itself as a filled dot inside it.

        A position the scan only forecast - the tracker's coasted guess across a miss - is not drawn: the mark
        on the picture is a sighting, or it is absent. ``stamp`` is the full-rate path's interpolated position,
        which only ever sits between two neighboring sightings (see :meth:`ball_at`).
        """
        u, v, measured = self.ball[index] if stamp is None else stamp
        if measured != 1.0 or not (np.isfinite(u) and np.isfinite(v)):
            return
        height, width = frame.shape[:2]
        center = (int(round(u * width)), int(round(v * width)))
        if not (-0.05 * width <= center[0] <= 1.05 * width and -0.05 * width <= center[1] <= 1.05 * width):
            return
        radius = max(5, int(round(height / 150.0)))
        half = radius + max(3, radius // 2)
        cv2.rectangle(frame, (center[0] - half, center[1] - half), (center[0] + half, center[1] + half),
                      (255, 255, 255), max(1, radius // 4))
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        cv2.circle(frame, center, radius, (40, 40, 220), max(2, radius // 3))

    def _draw_hud(self, frame: np.ndarray, index: int, *, note: str = "", session_line: bool = False) -> None:
        """The watching corners: who is playing, which color they are, and where in the match this is.

        The clock is the animation's clock: elapsed on the analyzed window's own clock, whose zero is where the
        analysis starts (kick-off on a kick-off-to-full-time build), drawn ``M:SS`` like the pane's. The
        recording's clock is deliberately not drawn - it starts before kick-off, and a stamp in recording
        seconds reads minutes away from the animation the viewer compares it with.

        ``session_line`` holds the very first line for the debug layer's session name when both layers are on,
        so the two never draw over each other; the legend only starts a line lower when it has to.
        """
        height, width = frame.shape[:2]
        scale = height / 1080.0
        font = max(0.4, 0.55 * scale)
        pad = max(6, int(12 * scale))
        y = pad + int(18 * scale) + (int(22 * scale) if session_line else 0)
        for team, style in enumerate(self.teams):
            cv2.rectangle(frame, (pad, y - int(13 * scale)), (pad + int(18 * scale), y + int(2 * scale)), style.bgr, -1)
            _outline_text(frame, style.name, (pad + int(26 * scale), y), PAPER, font)
            y += int(21 * scale)
        # A role color nobody can name is decoration: the key is drawn only when such a box is on screen.
        for role in ("goalkeeper", "referee"):
            if role not in self.roles.values():
                continue
            cv2.rectangle(frame, (pad, y - int(13 * scale)), (pad + int(18 * scale), y + int(2 * scale)),
                          ROLE_BGR[role], -1)
            _outline_text(frame, role, (pad + int(26 * scale), y), PAPER, font)
            y += int(21 * scale)

        # ``source_time`` is the recording's clock, whose zero is the start of the recording; subtracting
        # ``start_s`` lands on the animation's zero (kick-off for a marked build), and ``M:SS`` is the shape the
        # pane's clock text has, so the two read as one clock.
        clock = max(0.0, self.source_time(index) - self.start_s)
        stamp = f"{int(clock) // 60}:{int(clock) % 60:02d}"
        # The speed a viewer is watching at is playback state, so it rides with the clock - not with the frame
        # counter - and stays on screen when the debug layer is turned off.
        line = stamp + (f"  {note}" if note else "")
        (tw, _th), _ = cv2.getTextSize(line, FONT, font, 1)
        _outline_text(frame, line, (width - pad - tw, pad + int(18 * scale)), PAPER, font)

    def _draw_debug(self, frame: np.ndarray, index: int) -> None:
        """The diagnosing corners: which session this is (top-left), where in the analysis this frame sits
        (top-right, under the clock) and the honest notes about what was left out (bottom-left)."""
        height, width = frame.shape[:2]
        scale = height / 1080.0
        font = max(0.4, 0.55 * scale)
        pad = max(6, int(12 * scale))
        _outline_text(frame, self.match_id, (pad, pad + int(18 * scale)), PAPER, font)
        counter = f"frame {index}/{self.frame_count}"
        (tw, _th), _ = cv2.getTextSize(counter, FONT, font, 1)
        _outline_text(frame, counter, (width - pad - tw, pad + int(18 * scale) + int(21 * scale)), PAPER, font)
        y = height - pad
        for line in reversed(self.notes[-3:]):
            _outline_text(frame, line, (pad, y), PAPER, max(0.35, 0.42 * scale))
            y -= int(16 * scale)


CHIP_PAD = 4


def _chip_size(text: str, font: float) -> tuple[int, int]:
    if not text:
        return (0, 0)
    (tw, th), _ = cv2.getTextSize(text, FONT, font, 1)
    return (tw + 2 * CHIP_PAD, th + 2 * CHIP_PAD)


def _chip_width(text: str, font: float) -> int:
    return _chip_size(text, font)[0]


def _chip_height(text: str, font: float) -> int:
    return _chip_size(text, font)[1]


def _clamp_chip_x(frame: np.ndarray, left: int, chip_width: int) -> int:
    """Keep a chip inside the frame: never past either edge, but still near its box."""
    return int(np.clip(left, 2, max(2, frame.shape[1] - chip_width - 2)))


def _chip(frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int], font: float) -> None:
    """A filled label chip whose top-left corner is ``(x, y)``, sized to its text."""
    if not text:
        return
    chip_w, chip_h = _chip_size(text, font)
    cv2.rectangle(frame, (x, y), (x + chip_w, y + chip_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + chip_w, y + chip_h), INK, 1)
    cv2.putText(frame, text, (x + CHIP_PAD, y + chip_h - CHIP_PAD), FONT, font, _text_bgr(color), 1, cv2.LINE_AA)


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise StreamError("JPEG encode failed")
    return buffer.tobytes()


def iter_annotated_frames(
    match: AnnotatedMatch,
    start_s: float,
    *,
    rate: float = DEFAULT_RATE,
    width: int = DEFAULT_WIDTH,
    overlays: dict[str, bool] | None = None,
    control: StreamControl | None = None,
    reader_factory=open_reader,
    pace: bool = True,
):
    """Decode the match from ``start_s`` and yield ``(index, jpeg)`` for every analysis-aligned frame.

    The decode runs at the analysis rate on the same time grid the detections were stored on (``-ss`` lands on a
    frame boundary of that grid), so frame ``index`` is drawn with exactly the boxes and ball the pipeline
    recorded for it. ``rate`` only changes how fast the frames are *pushed*; it never skips an analysis frame,
    which is what keeps a 4x stream honest.

    ``overlays`` selects the drawn layers (:func:`parse_overlays`); ``None`` keeps every layer on, which is what
    every caller that does not think about overlays expects. ``control``, when given, is checked between frames
    so a client can end its own stream (see :class:`StreamControl`). ``pace`` is the real-time throttle (a
    testing seam turns it off); when the decode falls behind, pacing simply yields to it.
    """
    layers = ALL_OVERLAYS if overlays is None else overlays
    index = match.index_for(start_s)
    decode_start = match.source_time(index)
    remaining_s = (match.frame_count - index) / match.fps
    reader = reader_factory(
        match.video, fps=match.fps, width=width, start_s=decode_start, duration_s=remaining_s
    )
    period = 1.0 / max(0.01, match.fps * rate)
    next_push = time.monotonic()
    for timestamp, frame in reader.frames():
        if control is not None and control.stopped:
            return
        frame_index = int(round((timestamp - match.start_s) * match.fps))
        if frame_index >= match.frame_count:
            return
        match.render(frame, frame_index, note=f"{rate:g}x" if rate != 1.0 else "", **layers)
        yield frame_index, encode_jpeg(frame)
        if pace:
            next_push += period
            delay = next_push - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_push = time.monotonic()


def streamable_matches(root: str | Path | None = None) -> list[dict]:
    """One row per match that could be streamed, for the index page and ``/matches``.

    Only metadata is read (never the tens-of-megabyte replay), so the index stays cheap: what it needs is which
    matches exist, what the teams are called and wear, and where the analyzed window starts and ends.
    """
    library = MatchLibrary(root)
    rows: list[dict] = []
    for match_id in library.list_ids():
        directory = library.path(match_id)
        record = library.load(match_id)
        if not record.segments or not (directory / "calibration.json").exists() or not (directory / "replay.json").exists():
            rows.append({"match_id": match_id, "streamable": False, "reason": "no calibration or replay yet"})
            continue
        segment_dir = _segment_for(record)
        meta_path = segment_dir / "meta.json"
        if not meta_path.exists():
            rows.append({"match_id": match_id, "streamable": False, "reason": "segment not found"})
            continue
        meta = json.loads(meta_path.read_text())
        colors: list[list[int] | None] = [None, None]
        try:
            report = json.loads((directory / "report.json").read_text())
            for team in report.get("teams") or []:
                team_index = int(team.get("team", -1))
                if 0 <= team_index < 2:
                    colors[team_index] = team.get("kit_rgb")
        except (OSError, json.JSONDecodeError):
            pass
        rows.append(
            {
                "match_id": match_id,
                "streamable": True,
                "team_names": list(record.team_names),
                "team_colors": colors,
                "start_s": meta.get("start_s"),
                "end_s": meta.get("end_s"),
                "frame_count": meta.get("total_frames"),
                "fps": meta.get("fps"),
            }
        )
    return rows


# --------------------------------------------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------------------------------------------
class MatchStreamServer(ThreadingHTTPServer):
    """The stream server: caches each match's overlay once loaded and caps how many streams decode at once.

    Every viewer of a stream spawns their own ffmpeg process (they may start at different times and speeds), so
    concurrency is bounded by a semaphore rather than left open for one client to multiply by opening tabs.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        root: str | Path | None = None,
        width: int = DEFAULT_WIDTH,
        max_streams: int = MAX_CONCURRENT_STREAMS,
        reader_factory=open_reader,
        pace: bool = True,
        encoder: str | None = None,
        clip_cache: ClipCache | None = None,
        clip_encoder=encode_clip,
        live_chunks=iter_live_chunks,
        games_root: str | Path | None = None,
    ):
        super().__init__(server_address, MatchStreamHandler)
        self.root = Path(root) if root is not None else None
        self.width = int(width)
        self.reader_factory = reader_factory
        self.pace = pace
        # The encoded routes share the decode slots with the MJPEG streams - an encode is as heavy as a
        # stream - and share the stop-token machinery too. ``video_encoder`` is the ffmpeg encoder's name
        # (None: probe for NVENC/x264 on first use); the encoders themselves are injectable for tests.
        self.video_encoder = encoder
        self.clip_cache = clip_cache if clip_cache is not None else ClipCache()
        self.clip_encoder = clip_encoder
        self.live_chunks = live_chunks
        # Game marking (Step 1): the /game/ route streams frames straight from the clips - no combined file, no
        # encode, no proxy build. An explicit ``games_root`` keeps the old single-root behavior; otherwise the
        # manifest is found where this version writes it (an analysis directory beside the footage) or in the
        # legacy root.
        self.games_root = Path(games_root) if games_root is not None else None
        self.stream_slots = threading.BoundedSemaphore(max_streams)
        self._sessions: dict[str, AnnotatedMatch] = {}
        self._sessions_lock = threading.Lock()
        self._controls: dict[str, StreamControl] = {}
        self._controls_lock = threading.Lock()

    def register_control(self, token: str) -> StreamControl:
        """The control object for a stream token (a fresh one per stream; tokens are not reused)."""
        control = StreamControl()
        with self._controls_lock:
            self._controls[token] = control
        return control

    def request_stop(self, token: str) -> bool:
        """Ask the stream behind a token to end; False when it is not (or no longer) running - both normal."""
        with self._controls_lock:
            control = self._controls.get(token)
        if control is None:
            return False
        control.request_stop()
        return True

    def drop_control(self, token: str) -> None:
        with self._controls_lock:
            self._controls.pop(token, None)

    def game_directory(self, game_id: str) -> Path:
        """Where a game's manifest and proxy live: an explicit legacy root when the server was told one, else
        the manifest found beside the footage (and in the old ``data/games`` archive)."""
        if self.games_root is not None:
            return game_lib.game_dir(self.games_root, game_id)
        directory = game_lib.find_dir_by_id(game_id)
        if directory is None:
            raise StreamError(f"no game {game_id} beside any footage root")
        return directory

    def session(self, match_id: str) -> AnnotatedMatch:
        """The cached overlay for a match, loading it on first demand (a few seconds of segment and chain work).

        The shirt numbers are re-read even for a cached session: the dashboard edits the roster beside the match,
        and a chip still showing a number the user just removed would read as the stream ignoring them.
        """
        with self._sessions_lock:
            session = self._sessions.get(match_id)
        if session is None:
            session = AnnotatedMatch.load(match_id, root=self.root)
            with self._sessions_lock:
                session = self._sessions.setdefault(match_id, session)
        library = MatchLibrary(self.root)
        # The record carries which team rosters the match links. A match directory without one (or with one too
        # old to parse) must still stream: it simply draws numbers without roster names.
        try:
            record = library.load(match_id)
        except (OSError, ValueError, TypeError):  # missing, torn, or a record from a different layout
            record = None
        numbers, jerseys = load_numbers(
            library,
            match_id,
            session._player_ids,
            teams=session._teams,
            team_rosters=team_rosters_for(RosterLibrary(), record.team_rosters if record else None),
        )
        session.refresh_numbers(numbers, numbers_note(library.path(match_id), jerseys, numbers, session._player_ids))
        return session


class MatchStreamHandler(BaseHTTPRequestHandler):
    """Routes: ``/`` index, ``/matches`` JSON, ``/stream/<id>.mjpg``, ``/frame/<id>.jpg``,
    ``/live/<id>.mp4``, ``/video/<id>.mp4`` and ``/game/<id>.mjpg``."""

    server: MatchStreamServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - the base class's signature
        pass  # streams are long-lived; one line per connect is noise, errors still surface on the socket

    # -- helpers ----------------------------------------------------------------------------------------------
    def _send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8", extra_headers: dict | None = None) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _query(self, parsed) -> dict:
        return {key: values[-1] for key, values in parse_qs(parsed.query).items()}

    @staticmethod
    def _float(query: dict, key: str, default: float, low: float, high: float) -> float:
        try:
            value = float(query.get(key, default))
        except (TypeError, ValueError):
            return default
        return float(np.clip(value, low, high))

    @staticmethod
    def _flag(query: dict, key: str, *, default: bool) -> bool:
        """A boolean query parameter: ``0``/``no``/``off``/``false`` turn it off, anything else leaves it on."""
        if key not in query:
            return default
        return str(query[key]).strip().lower() not in ("0", "no", "off", "false")

    @staticmethod
    def _codec(query: dict) -> str:
        """``codec=hevc`` asks for H.265 (smaller files at equal quality, pickier players); else H.264."""
        value = str(query.get("codec", "h264")).strip().lower()
        return value if value in CODECS else "h264"

    # -- routes -----------------------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - the base class's name
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self._index(parsed)
            elif parsed.path == "/matches":
                self._matches_json()
            elif parsed.path.startswith("/play/"):
                self._play(parsed)
            elif parsed.path == "/stop":
                self._stop(parsed)
            elif parsed.path.startswith("/stream/"):
                self._stream(parsed)
            elif parsed.path.startswith("/live/"):
                self._live(parsed)
            elif parsed.path.startswith("/video/"):
                self._video(parsed)
            elif parsed.path.startswith("/game/"):
                self._game_mjpeg(parsed)
            elif parsed.path.startswith("/frame/"):
                self._frame(parsed)
            else:
                self._send_text(404, "not found\n")
        except StreamError as error:
            self._send_text(409, f"{error}\n")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # the viewer closed the tab (or stopped reading); nothing to report

    def do_POST(self) -> None:  # noqa: N802 - the base class's name
        # ``sendBeacon`` can only POST; the stop endpoint accepts both so a page can say goodbye on unload.
        parsed = urlparse(self.path)
        if parsed.path != "/stop":
            self._send_text(404, "not found\n")
            return
        self._stop(parsed)

    def _stop(self, parsed) -> None:
        """End the stream behind a token (see :class:`StreamControl`). Idempotent: a stream that already ended
        (or never existed) is not an error - a stop often races the stream's own end.

        Cross-origin allowed: the pane runs on the dashboard's origin and this server is on its own port, so the
        component's ``fetch`` is a cross-origin request that would be blocked without the header. Stopping a
        stream is harmless (the token is random and the stream is this machine's own decode), so ``*`` is fine.
        """
        token = self._query(parsed).get("token", "")
        stopped = bool(token) and self.server.request_stop(token)
        self._send_text(
            200, "stopping\n" if stopped else "not running\n", extra_headers={"Access-Control-Allow-Origin": "*"}
        )

    def _match_id(self, parsed, suffix: str) -> str:
        name = parsed.path.rsplit("/", 1)[-1]
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        return name

    def _index(self, parsed) -> None:
        rows = streamable_matches(self.server.root)
        cards = []
        for row in rows:
            name = html.escape(str(row["match_id"]))
            if not row.get("streamable"):
                cards.append(f"<div class='card dead'><h2>{name}</h2><p>not streamable: {html.escape(str(row.get('reason')))}</p></div>")
                continue
            teams = row.get("team_names") or []
            swatches = []
            for index, team_name in enumerate(teams[:2]):
                color = (row.get("team_colors") or [None, None])[index]
                style = f"background:rgb({color[0]},{color[1]},{color[2]})" if color else "background:#888"
                swatches.append(f"<span class='sw' style='{style}'></span>{html.escape(str(team_name))}")
            start = row.get("start_s") or 0.0
            end = row.get("end_s") or 0.0
            frames = row.get("frame_count") or 0
            fps = row.get("fps") or 5.0
            base = f"/play/{name}?start={start:.1f}"
            links = " ".join(f"<a href='{base}&rate={rate}'>{rate:g}x</a>" for rate in (1, 2, 4))
            # A still, not a live <img> of the stream: an index that opened a decode per card would run one
            # ffmpeg for as long as the tab stayed open, for a picture nobody asked to play.
            preview_t = start + 30.0 if end and end - start > 60 else start
            cards.append(
                "<div class='card'>"
                f"<h2>{name}</h2>"
                f"<p>{' &middot; '.join(swatches)}</p>"
                f"<p class='dim'>window {start:.0f}s - {end:.0f}s, {frames} analyzed frames at {fps:g} fps</p>"
                f"<p><a href='{base}&rate=1'><img src='/frame/{name}.jpg?t={preview_t:.1f}' alt='annotated frame'></a></p>"
                f"<p>{links} &middot; <a href='/frame/{name}.jpg?t={start:.1f}'>single frame</a> &middot; "
                f"<a href='/play/{name}?format=live&start={start:.1f}'>mp4 live</a> &middot; "
                f"<a href='/play/{name}?format=clip&start={start:.1f}&duration=10'>mp4 clip (10s)</a> &middot; "
                f"<a href='/matches'>json</a></p>"
                "</div>"
            )
        body = (
            "<!doctype html><html><head><meta charset='utf-8'><title>annotated match stream</title>"
            "<style>"
            "body{font-family:system-ui,sans-serif;margin:24px;background:#141414;color:#eee}"
            ".card{max-width:960px;margin:18px auto;background:#1e1e1e;border-radius:10px;padding:14px 18px}"
            ".card img{max-width:100%;border-radius:6px}"
            ".dim{color:#999;font-size:0.9em}"
            ".sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 4px 0 10px;vertical-align:-1px}"
            "a{color:#6cf}"
            "</style></head><body>"
            "<h1>Annotated match stream</h1>"
            "<p class='dim'>The analyzed window drawn through the calibration: pitch model, detections, shirt "
            "numbers and team colors. Endpoints: <code>/stream/&lt;match&gt;.mjpg</code> "
            "(MJPEG, params <code>start=&lt;source seconds&gt;, rate=&lt;speed&gt;, width=&lt;px&gt;, overlays=</code>), "
            "<code>/live/&lt;match&gt;.mp4</code> (the same overlay encoded H.264 and streamed as a fragmented MP4), "
            "<code>/video/&lt;match&gt;.mp4</code> (a bounded, seekable clip: <code>duration=, fps=, width=, "
            "start=, overlays=, rate=, audio=, codec=h264|hevc</code>; defaults to the source's own frame rate "
            "and resolution), "
            "<code>/game/&lt;game&gt;.mjpg</code> (the game's clips read directly, un-annotated, for Step 1 marking: "
            "<code>t=&lt;game seconds&gt;, width=, fps=</code>; no encode, no cache), "
            "<code>/frame/&lt;match&gt;.jpg?t=</code>, <code>/matches</code>.</p>"
            + "".join(cards)
            + "</body></html>"
        )
        self._send_text(200, body, "text/html; charset=utf-8")

    def _matches_json(self) -> None:
        rows = streamable_matches(self.server.root)
        body = json.dumps({"matches": rows}, indent=2)
        self._send_text(200, body, "application/json")

    def _play(self, parsed) -> None:
        """A page that embeds the media in the one element each format plays in.

        MJPEG only plays in an ``<img>``; the encoded variants only in a ``<video>``. Opening either kind of URL
        as a top-level document makes the browser download it instead of showing it, so every link that means
        "watch this" lands here. ``?format=live`` asks for the endless fragmented-MP4 stream, ``?format=clip``
        for a bounded, seekable MP4; the default stays MJPEG. The page itself never touches the archive - the
        media's own request loads the match (and reports any error), keeping this route instant.
        """
        match_id = html.escape(self._match_id(parsed, ""))
        query = self._query(parsed)
        params = []
        rate = 1.0
        for key in ("start", "rate", "width", "fps", "duration"):
            if key not in query:
                continue
            try:
                value = float(query[key])
            except (TypeError, ValueError):
                continue  # a mistyped parameter falls back to the defaults rather than a 500
            if key == "rate":
                rate = value
            params.append(f"{key}={value:g}")
        if "codec" in query:
            params.append(f"codec={self._codec(query)}")

        def link(target_format: str | None, extra: tuple[tuple[str, str], ...] = ()) -> str:
            overrides = dict(extra)
            pieces = [part for part in params if part.split("=")[0] not in overrides] + [
                f"{name}={value}" for name, value in extra
            ]
            if target_format is not None:
                pieces.append(f"format={target_format}")
            query_string = "&".join(pieces)
            return f"/play/{match_id}" + (f"?{query_string}" if query_string else "")

        chosen = query.get("format", "mjpeg")
        if chosen == "clip":
            src = f"/video/{match_id}.mp4" + ("?" + "&".join(params) if params else "")
            media = f"<video src='{src}' controls autoplay muted></video>"
            note = "A clip is encoded when the player asks for it, so the first picture can take a while - after that it seeks freely. Unmute in the player for the match audio."
            beacon = ""
        elif chosen == "live":
            # The live stream gets its own stop token and a goodbye beacon: a browser that navigates away keeps
            # *draining* a media fetch (the MJPEG pane taught this), so the page must end its own encoder.
            token = os.urandom(6).hex()
            src = f"/live/{match_id}.mp4" + ("?" + "&".join(params + [f"token={token}"]))
            media = f"<video src='{src}' controls autoplay muted></video>"
            note = (
                "The live encode runs at what this machine can render: 1080p at the source's frame rate keeps "
                "up better than 4K. Unmute in the player for the match audio. Safari and other WebKit browsers "
                "will not play this endless fragmented MP4 (their media stack wants byte ranges or HLS) - use "
                "the MJPEG link there; the dashboard pane falls back to it automatically."
            )
            beacon = (
                "<script>window.addEventListener('pagehide',()=>{const u=new URL("
                "document.querySelector('video').src,location);const t=u.searchParams.get('token');"
                "if(t)navigator.sendBeacon('/stop?token='+encodeURIComponent(t));});</script>"
            )
        else:
            src = f"/stream/{match_id}.mjpg" + ("?" + "&".join(params) if params else "")
            media = f"<img src='{src}' alt='annotated match stream'>"
            note = ""
            beacon = ""
        speed = f" at {rate:g}x" if rate != 1.0 else ""
        navigation = (
            f"<a href='{link(None)}'>mjpeg</a>"
            f"<a href='{link('live')}'>mp4 live</a>"
            f"<a href='{link('clip', (('duration', '10'),))}'>mp4 clip (10s)</a>"
            "<a href='..'>index</a><a href='/matches'>json</a>"
        )
        body = (
            "<!doctype html><html><head><meta charset='utf-8'><title>"
            f"{match_id}{html.escape(speed)}</title>"
            "<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}"
            "img,video{max-width:100%;height:auto;display:block;margin:0 auto}"
            ".bar{padding:10px 16px;font-size:0.95em}.bar a{color:#6cf;margin-right:14px}"
            ".note{padding:8px 16px;font-size:0.85em;color:#9aa4b2}</style></head><body>"
            f"<div class='bar'><b>{match_id}</b>{html.escape(speed)} &middot; {navigation}</div>"
            f"{media}"
            + (f"<div class='note'>{note}</div>" if note else "")
            + beacon
            + "</body></html>"
        )
        self._send_text(200, body, "text/html; charset=utf-8")

    def _frame(self, parsed) -> None:
        match_id = self._match_id(parsed, ".jpg")
        query = self._query(parsed)
        session = self.server.session(match_id)
        width = int(self._float(query, "width", self.server.width, MIN_WIDTH, MAX_WIDTH))
        t = float(query.get("t", session.start_s))
        overlay_layers = parse_overlays(query.get("overlays"))
        frame = grab_frame(session.video, max(0.0, t), width=width)
        if frame is None:
            raise StreamError(f"could not decode a frame at {t:.1f}s")
        index = int(np.clip(round((t - session.start_s) * session.fps), 0, session.frame_count - 1))
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encode_jpeg(session.render(frame, index, **overlay_layers)))

    def _stream(self, parsed) -> None:
        match_id = self._match_id(parsed, ".mjpg")
        query = self._query(parsed)
        if not self.server.stream_slots.acquire(timeout=STREAM_SLOT_WAIT_S):
            self._send_text(503, "too many concurrent streams; retry shortly\n")
            return
        token = ""
        try:
            session = self.server.session(match_id)
            width = int(self._float(query, "width", self.server.width, MIN_WIDTH, MAX_WIDTH))
            rate = self._float(query, "rate", DEFAULT_RATE, MIN_RATE, MAX_RATE)
            start_s = float(query.get("start", session.start_s))
            overlay_layers = parse_overlays(query.get("overlays"))
            token = query.get("token", "")[:64]
            control = self.server.register_control(token) if token else None
            # A write timeout is the watchdog against a viewer that stops reading without closing - a browser
            # tab can leave an MJPEG fetch half-open when its <img> is removed (a paused pane does exactly that),
            # and without this the handler would block on a full socket buffer and keep the ffmpeg decode alive
            # indefinitely. A frame write that blocks this long means nobody is consuming; let the stream die.
            self.connection.settimeout(STREAM_WRITE_TIMEOUT_S)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for _index, jpeg in iter_annotated_frames(
                session,
                start_s,
                rate=rate,
                width=width,
                overlays=overlay_layers,
                control=control,
                reader_factory=self.server.reader_factory,
                pace=self.server.pace,
            ):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        finally:
            if token:
                self.server.drop_control(token)
            self.server.stream_slots.release()

    def _video(self, parsed) -> None:
        """Encode a bounded window into a seekable MP4 and serve it - the full-rate, full-resolution sibling of
        the MJPEG stream.

        The response *is* the encode, so it waits for it; the clip cache is what makes that wait worth reusing,
        because a player comes straight back for byte ranges as the viewer scrubs. A decode slot is held for the
        whole encode, so asking for clips while streams are running cannot stack unbounded work. ``width`` and
        ``fps`` default to what the source actually is - full resolution, full frame rate - and ``duration``
        bounds the work.
        """
        match_id = self._match_id(parsed, ".mp4")
        query = self._query(parsed)
        session = self.server.session(match_id)
        native_width = session.native_width or MAX_VIDEO_WIDTH
        window_end = session.source_time(session.frame_count - 1)
        width = int(round(self._float(query, "width", native_width, MIN_WIDTH, native_width))) // 2 * 2
        fps = self._float(query, "fps", session.native_fps, 1.0, session.native_fps)
        duration = self._float(query, "duration", DEFAULT_CLIP_SECONDS, 0.5, MAX_CLIP_SECONDS)
        start_s = self._float(query, "start", session.start_s, 0.0, window_end)
        duration = min(duration, max(0.5, window_end - start_s))
        rate = self._float(query, "rate", 1.0, MIN_RATE, MAX_RATE)
        audio = self._flag(query, "audio", default=True)
        codec = self._codec(query)
        overlay_layers = parse_overlays(query.get("overlays"))
        key = clip_key(match_id, start_s, duration, fps, width, overlay_layers, rate, audio, codec)
        path = self.server.clip_cache.lookup(key)
        if path is None:
            if not self.server.stream_slots.acquire(timeout=STREAM_SLOT_WAIT_S):
                self._send_text(503, "too many concurrent encodes; retry shortly\n")
                return
            try:
                with self.server.clip_cache.claim(key):
                    path = self.server.clip_cache.lookup(key)  # another request may have just finished it
                    if path is None:
                        target = self.server.clip_cache.target(key)
                        try:
                            self.server.clip_encoder(
                                session,
                                start_s=start_s,
                                duration_s=duration,
                                fps=fps,
                                width=width,
                                overlays=overlay_layers,
                                output=target,
                                encoder=self.server.video_encoder,
                                reader_factory=self.server.reader_factory,
                                rate=rate,
                                audio=audio,
                                codec=codec,
                            )
                        except VideoError as error:
                            target.unlink(missing_ok=True)
                            raise StreamError(f"the encode failed: {error}") from error
                        self.server.clip_cache.keep(key, target)
                        path = target
            finally:
                self.server.stream_slots.release()
        self._send_file(path, "video/mp4")

    def _game_mjpeg(self, parsed) -> None:
        """The Step 1 marking stream: raw frames straight from the clips - no combined file, no encode, no cache.

        This replaced the on-demand encode of a whole game (640 px, 4 fps, cached at the old proxy path, and
        impossible without the merged file). Frames come from the clip set through `ingest.source`, so a seek
        costs the same as seeking the old combined video (~1 s on the reference 4K footage, with pixel-identical
        frames) and marking works the moment the clips are named - before Stage A has run at all. Frames are
        paced to ``fps``; the viewer reconnects with a new ``t`` to scrub.
        """
        game_id = self._match_id(parsed, ".mjpg")
        query = self._query(parsed)
        # `_match_id` keeps only the last path segment, so the raw path is what proves the id was one segment:
        # "nested/name" must be refused before any directory is joined, not silently read as "name".
        if parsed.path != f"/game/{game_id}.mjpg" or not game_id or game_id.startswith(".") or "\\" in game_id:
            self._send_text(404, "bad game id\n")
            return
        directory = self.server.game_directory(game_id)  # StreamError -> do_GET's handler
        if not (directory / game_lib.MANIFEST_FILE).exists():
            raise StreamError(f"no game {game_id}")
        record = game_lib.GameRecord.load(directory)
        if not record.clips:
            raise StreamError(f"game {game_id} has no clips to read")
        missing = next((clip for clip in record.clips if not Path(clip.path).exists()), None)
        if missing is not None:
            raise StreamError(f"game {game_id} is missing a clip: {Path(missing.path).name}")
        source = source_from_record(record)
        width = int(round(self._float(query, "width", game_lib.PROXY_WIDTH, MIN_WIDTH, MAX_VIDEO_WIDTH))) // 2 * 2
        fps = self._float(query, "fps", MARK_FPS, 1.0, 30.0)
        start_s = self._float(query, "t", 0.0, 0.0, float(record.duration_s))
        # ``frames=1`` serves a single still (a scrub preview) and closes: browsers keep the last MJPEG frame.
        frames = int(round(self._float(query, "frames", 0.0, 0.0, 100000.0)))
        if not self.server.stream_slots.acquire(timeout=STREAM_SLOT_WAIT_S):
            self._send_text(503, "too many concurrent streams; retry shortly\n")
            return
        token = query.get("token", "")[:64]
        control = self.server.register_control(token) if token else None
        try:
            # Same write-timeout watchdog as the annotated stream: a browser that removes the <img> can leave
            # the fetch half-open, and the handler must die rather than block forever on a full socket buffer.
            self.connection.settimeout(STREAM_WRITE_TIMEOUT_S)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            reader = open_reader(
                source, fps=fps, width=width, start_s=start_s, duration_s=float(record.duration_s) - start_s
            )
            period = 1.0 / max(0.1, fps)
            next_push = time.monotonic()
            sent = 0
            for _game_time, frame in reader.frames():
                if control is not None and control.stopped:
                    return
                jpeg = encode_jpeg(frame)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                sent += 1
                if frames and sent >= frames:
                    return
                if self.server.pace:
                    next_push += period
                    delay = next_push - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_push = time.monotonic()
        finally:
            if token:
                self.server.drop_control(token)
            self.server.stream_slots.release()

    def _live(self, parsed) -> None:
        """The endless variant of the encoded stream: a fragmented MP4 produced as the frames are rendered.

        Same overlays, same stop token, same decode slots as the MJPEG stream - the difference is H.264 at the
        rate that was asked for. The writer paces the encode to real time, so a request this box cannot sustain
        (4K at 60 fps here) plays behind rather than claiming a rate it is not making; that is also why ``width``
        defaults to 1920 for the live variant while clips default to the source's own size.
        """
        match_id = self._match_id(parsed, ".mp4")
        query = self._query(parsed)
        if not self.server.stream_slots.acquire(timeout=STREAM_SLOT_WAIT_S):
            self._send_text(503, "too many concurrent streams; retry shortly\n")
            return
        token = ""
        try:
            session = self.server.session(match_id)
            native_width = session.native_width or MAX_VIDEO_WIDTH
            window_end = session.source_time(session.frame_count - 1)
            width = (
                int(round(self._float(query, "width", min(DEFAULT_LIVE_WIDTH, native_width), MIN_WIDTH, native_width)))
                // 2
                * 2
            )
            fps = self._float(query, "fps", session.native_fps, 1.0, session.native_fps)
            start_s = self._float(query, "start", session.start_s, 0.0, window_end)
            rate = self._float(query, "rate", 1.0, MIN_RATE, MAX_RATE)
            audio = self._flag(query, "audio", default=True)
            codec = self._codec(query)
            overlay_layers = parse_overlays(query.get("overlays"))
            token = query.get("token", "")[:64]
            control = self.server.register_control(token) if token else None
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            # The same watchdog as the MJPEG stream: a viewer that stops reading without closing (a browser
            # navigating away keeps the fetch half-open) blocks the next frame write, and the timeout is what
            # ends the encode then - the stop beacon usually gets there first.
            self.connection.settimeout(STREAM_WRITE_TIMEOUT_S)
            chunks = self.server.live_chunks(
                session,
                start_s=start_s,
                fps=fps,
                width=width,
                overlays=overlay_layers,
                control=control,
                encoder=self.server.video_encoder,
                reader_factory=self.server.reader_factory,
                pace=self.server.pace,
                rate=rate,
                audio=audio,
                codec=codec,
            )
            try:
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except VideoError:
                pass  # headers are out; closing the connection is the only signal left to send
            finally:
                chunks.close()  # runs the generator's cleanup: the encoder dies with the connection
        finally:
            if token:
                self.server.drop_control(token)
            self.server.stream_slots.release()

    def _send_file(self, path: Path, content_type: str) -> None:
        """A file, honoring ``Range`` - which is what makes a clip scrubbable: the browser asks for the parts
        it needs while the viewer seeks. A range the file cannot satisfy gets 416, per the spec, so a player
        falls back to a fresh request instead of guessing."""
        size = path.stat().st_size
        start, end = 0, size - 1
        partial = False
        header = self.headers.get("Range") if self.headers is not None else None
        if header and header.startswith("bytes="):
            first, _, last = header[len("bytes="):].split(",")[0].strip().partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                else:
                    start = max(0, size - int(last))
                partial = 0 <= start <= end < size
            except ValueError:
                partial = False
            if not partial:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(
    host: str = "0.0.0.0",
    port: int = 8510,
    *,
    root: str | Path = MATCHES_ROOT,
    width: int = DEFAULT_WIDTH,
    preload: tuple[str, ...] = (),
) -> None:
    """Run the stream server until interrupted; ``preload`` warms the overlay cache for named matches."""
    server = MatchStreamServer((host, port), root=root, width=width)
    for match_id in preload:
        started = time.monotonic()
        try:
            server.session(match_id)
        except StreamError as error:
            print(f"cannot preload {match_id}: {error}", flush=True)
            continue
        print(f"preloaded {match_id} in {time.monotonic() - started:.1f}s", flush=True)
    print(f"serving annotated streams on http://{host}:{port}/ (control-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

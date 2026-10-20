"""Build the animated pitch replay: the payload the pitch-view component fetches and draws.

Pure math, no Streamlit. The dashboard saves the result beside the match (``replay.json``) and hands the component
a URL, so the tens of thousands of points are fetched once by the browser instead of being serialized onto the
Streamlit websocket on every rerun.

One entry per player holds their frame indices, pitch positions (meters) and speeds; upstream positions stay in
``PlayerTrack`` order, which is sorted by time - the component interpolates between consecutive observations.

Beside the players the payload carries the two *measurements* the view draws as they are: each team's measured kit
color (what the clustering actually saw on the pitch) and the ball scan's track with its measured/forecast flag.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from soccer_analytics.analysis import event_detection, stage_b
from soccer_analytics.analysis import game as game_lib
from soccer_analytics.analysis import roles as roles_lib
from soccer_analytics.analysis.events import EVENT_TYPES, Event, same_recording
from soccer_analytics.analysis.stage_b import PlayerTrack
from soccer_analytics.dashboard.reports import team_name

# A player is "on the ball" when they are the nearest player to the camera's aim point and within this distance.
# Same gate as stage_b's momentum: the aim point is a ball proxy, not the scanned ball - a touch here means
# "nearest to where the camera was pointed", and it is labeled that way in the UI.
POSSESSION_RADIUS_M = 12.0
ROUND_M = 0.1

# Bystander exclusion. The detector sees everyone in frame - coaches on the touchline, photographers,
# spectators beyond the far touchline - and the tracker tracks them like anyone else. Drawing them on the
# pitch animation puts a crowd of stationary dots on the field of play. Two measurements separate them
# from players (calibrated on the real 2026-10-03 game, where 64 team-labeled tracks are ground truth):
# * a player track covers ground - the 5th percentile of team-labeled tracks spans a 17.9 m diagonal,
#   while a spectator's track spans a meter or two;
# * a player moves - the 5th percentile of team-labeled tracks is moving (over 1 km/h) 6% of observations,
#   while a spectator essentially never moves.
# The rule flags a track as a bystander when it is small, or small AND still. Measured on the same game:
# 1,095 of 4,378 long tracks flagged, zero of them team-labeled, and the per-frame count of surviving
# tracks has a median of 14 - what a camera following the ball actually sees.
BYSTANDER_MIN_EXTENT_M = 8.0  # below this diagonal the track never left its spot
BYSTANDER_EXTENT_M = 15.0  # the "small and still" band's upper edge
BYSTANDER_MOVING_FRACTION = 0.05  # below this share of moving observations a small track is "still"
# Evidence floor: how long a track must have lived before stillness may convict it. This was 20 observations -
# written when Stage A analyzed at 5 fps, when that was 4 seconds of behaviour. The analysis now runs at 15 fps,
# where the same count was a bare 1.3 seconds, and the exclusions on the 2026-10-03 game grew to 47% of all
# tracks (5,930 of 12,627) against the ~25% the rule was calibrated to - 84% of the dropped tracks holding
# under 10 seconds of observations: fragments of real players who merely stood still for a moment, worst at the
# window's start where the whole kickoff formation is standing. In seconds the meaning cannot silently drift
# with the analysis rate again.
BYSTANDER_MIN_SECONDS = 10.0


def _is_bystander(track: PlayerTrack) -> bool:
    """True for a track that behaves like a touchline bystander rather than a player.

    Extent is the diagonal of the track's bounding box in pitch meters; movement is the share of observations
    whose speed exceeds 1 km/h (the same threshold the report uses for "moving"). A track must have lived
    ``BYSTANDER_MIN_SECONDS`` before any of that may convict it: a fragment shows too little of a person to
    judge - dropping one would hide a real player who was briefly visible, and a fragment cannot clutter the
    animation much either way.
    """
    if len(track.time) < 2 or float(track.time[-1]) - float(track.time[0]) < BYSTANDER_MIN_SECONDS:
        return False
    finite = track.xy[np.isfinite(track.xy).all(axis=1)]
    if len(finite) < 2:
        return False
    extent = float(np.hypot(np.ptp(finite[:, 0]), np.ptp(finite[:, 1])))
    if extent < BYSTANDER_MIN_EXTENT_M:
        return True
    if extent >= BYSTANDER_EXTENT_M:
        return False
    moving = float(np.mean(np.asarray(track.speed_kmh, dtype=np.float64) > 1.0))
    return moving < BYSTANDER_MOVING_FRACTION


def _round(value: float) -> float:
    return float(round(float(value), 1))


def event_from_tag(tag: dict, *, video: str, window_start: float, duration_s: float) -> Event:
    """A manual tag from the playback's tag bar, as an ``Event`` on the recording's own clock.

    The tag bar sends the strip's own second (0 at the first analyzed frame), because that is the clock the
    playback shows. Events are stored with seconds of the *recording* they were made against - the clock every
    reader (the table, the strip, the reel cutter) translates back through the game manifest - so the analyzed
    window's start is added back here, once, for every tag from the page.

    Raises ``ValueError`` for an event type outside :data:`EVENT_TYPES`; the time is clamped into the analyzed
    window, so a tag arriving from a stale component value can never be stored off the end of the match.
    """
    tag_type = str(tag.get("type") or "")
    if tag_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {tag_type!r}; expected one of {EVENT_TYPES}")
    strip_s = min(max(float(tag.get("time_s") or 0.0), 0.0), max(0.0, float(duration_s)))
    return Event(
        time_s=float(window_start) + strip_s,
        type=tag_type,
        team=int(tag.get("team", -1)),
        note=str(tag.get("note") or "").strip()[:200],
        video=str(video),
    )


def repeated_manual_tag(
    events: Sequence[Event],
    event: Event,
    *,
    within_s: float = 0.01,
    goal_within_s: float = 3.0,
) -> Event | None:
    """The already-stored manual tag that ``event`` repeats, if there is one.

    A press is delivered *at least* once, not exactly once: a page that reconnects after a server restart re-sends
    the component's sticky value, and the new session has no acknowledgment to compare it against - so without
    this check the same press would be stored again on every reconnect (a duplicate tag on the timeline, and in
    every reel that uses it). Two manual tags of the same type for the same team on the same recording within a
    centisecond are one press; a different type, team, second or recording is a different event and is left alone.
    Detected candidates never match - a human tagging an event a detector also found is two rows on purpose.

    Goals get a wider window: a goal is a moment, not a button press, and a second press while the replay is
    being watched is the same goal. Other types keep the tight window, where two separate tags can be genuinely
    different events.
    """
    window = goal_within_s if event.type == "goal" else within_s
    for other in events:
        if (
            getattr(other, "source", "manual") == "manual"
            and other.type == event.type
            and int(other.team) == int(event.team)
            and same_recording(other.video, event.video)
            and abs(float(other.time_s) - float(event.time_s)) < window
        ):
            return other
    return None


def deleted_events(events: Sequence[Event], rows: Sequence[dict]) -> list[tuple[int, Event]]:
    """The stored events a timeline delete gesture names, highest index first.

    A right-click on the strip sends the marker's position in the event log plus the event's own identity. The
    position alone is not enough: a sticky component value can arrive after another tab (or a scan) changed the
    log, and deleting whatever now sits at that index would remove an unrelated event. Every row is therefore
    checked against the stored event's type, team, source, recording and second; a mismatch is skipped. The
    result is ordered highest-index first so removing several rows in one gesture cannot shift the others.
    """
    matched: dict[int, Event] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if index in matched or not 0 <= index < len(events):
            continue
        event = events[index]
        if str(row.get("type") or "") != event.type:
            continue
        if str(row.get("source") or "") != event.source:
            continue
        try:
            if int(row.get("team", event.team)) != int(event.team):
                continue
        except (TypeError, ValueError):
            continue
        sent_time = row.get("log_time_s")
        if sent_time is not None:
            try:
                if abs(float(sent_time) - float(event.time_s)) > 0.01:
                    continue
            except (TypeError, ValueError):
                continue
        sent_video = row.get("video")
        if sent_video is not None and str(sent_video) != str(event.video):
            continue
        matched[index] = event
    return sorted(matched.items(), key=lambda item: item[0], reverse=True)


def _kit_color(entry: Sequence[int] | None) -> list[int] | None:
    """One team's kit color as a plain JSON list of three ints, or None when there is none to show.

    Clamped rather than trusted: the browser paints markers with ``rgb(...)`` built from these numbers, and a
    value that can make that string invalid must not leave here. ``None`` is kept as ``None`` - it means the kits
    were not separable, which is exactly when the component should fall back to its own palette.
    """
    if entry is None or len(entry) != 3:
        return None
    return [int(min(255, max(0, channel))) for channel in entry]


def attack_summary(orientations, *, half_frame: int | None) -> dict | None:
    """The component's attack directions: one ``[team0, team1]`` pair per period, +1 toward +x, -1 toward -x.

    ``half_frame`` is the first frame of the second period in analysis frames, or None for a match without
    marked halves (one pair then covers the whole window). The pairs come from
    ``event_detection.team_orientations``, the same measurement the event detector uses to tell a clearance from
    a shot - so the arrows on the animation and the event labels cannot disagree about who attacks which way.
    """
    if not orientations:
        return None
    directions = []
    for orientation in sorted(orientations, key=lambda item: item.half):
        pair = orientation.attack_direction
        directions.append([int(pair[0]), int(pair[1])] if 0 in pair and 1 in pair else None)
    return {"half_frame": None if half_frame is None else int(half_frame), "directions": directions}


# What kind of act an event is, by its type: an attack on the opponent's goal or a defense of its own. The
# classification is the type's own meaning, not a guess from where it happened - a clearance is a defensive act
# wherever it is played from. The types that straddle both stay out of both sets: a fouled attacker is neither
# an attack nor a defense, and a blank is more honest than a forced answer.
ATTACKING_EVENT_TYPES = frozenset(("goal", "shot", "corner", "penalty"))
DEFENSIVE_EVENT_TYPES = frozenset(("clearance", "tackle", "save", "block"))


def event_play(event_type: str) -> str:
    """``"attacking"``, ``"defensive"`` or ``""`` for the types that are neither (foul, substitution, other)."""
    if event_type in ATTACKING_EVENT_TYPES:
        return "attacking"
    if event_type in DEFENSIVE_EVENT_TYPES:
        return "defensive"
    return ""


def aiming_goal(direction: int | None) -> str:
    """The goal mouth a team attacks when its measured direction at the time was ``direction``.

    The pane draws the pitch with the ``x=0`` goal on the left and measures ``attack`` directions in the same
    frame, so the mapping is one sign - and the label says which mouth to watch in the animation's own terms.
    ``None`` (no measurement for that half) stays blank; zero is not a direction this pipeline emits.
    """
    if direction is None or direction == 0:
        return ""
    return "toward the left goal" if direction < 0 else "toward the right goal"


def roles_and_attack(
    report, detections, assignment, *, segment, pitch_length_m: float, swap_attack: bool = False
):
    """Role labels and attack directions for the replay payload, from the tracks the report was built on.

    One place so the dashboard build and ``rebuild_match.py`` cannot ship different versions of the same
    analysis: roles come from ``analysis.roles`` (referee by range, keepers by the goal pockets) and the
    directions from ``analysis.event_detection.team_orientations``, measured per half against the game's own
    kick-off/half-time marks when the segment belongs to a marked game. ``swap_attack`` applies the match's
    user correction before the directions are written into the payload.
    """
    fps = float(segment.meta.get("fps", 5.0))
    start_s = float(segment.meta.get("start_s", 0.0))
    game_record = game_lib.find_for_video(str(segment.meta.get("video") or ""))
    bounds = game_record.bounds() if game_record is not None else None

    def kit_evidence(track_id: int):
        rows = assignment.tracks.get(int(track_id))
        return stage_b._track_kit_evidence(detections, rows) if rows is not None else None

    roles = roles_lib.classify_roles(report.players, pitch_length_m=pitch_length_m, kit_evidence=kit_evidence)
    orientations = event_detection.team_orientations(
        report.players, pitch_length_m, half_bounds=bounds, swap=swap_attack
    )
    half_frame = int(round((bounds[1] - start_s) * fps)) if bounds is not None else None
    return roles, attack_summary(orientations, half_frame=half_frame)


def build_replay(
    pitch: tuple[float, float],
    fps: float,
    frame_count: int,
    aim_xy: np.ndarray,
    players: list[PlayerTrack],
    team_names: list[str],
    ball: tuple[np.ndarray, np.ndarray] | None = None,
    team_colors: Sequence[Sequence[int] | None] | None = None,
    camera_xy: Sequence[float] | None = None,
    roles: dict[int, dict] | None = None,
    attack: dict | None = None,
) -> dict:
    """Assemble the replay payload from tracks, the per-frame camera aim (ball proxy) and the ball track.

    Shirt numbers are deliberately *not* baked in here: they come from the roster and the scanner and are passed to
    the component as a small argument, so editing a roster updates the view without rebuilding the replay. Players
    keep every observation the tracker produced - the client decides what to draw.

    ``ball`` is the optional ``(xy (F, 2), measured (F,))`` pair from ``projection.project_ball_track`` (the ball
    scan, when one has run): NaN rows become ``null``, and the ``measured`` flag rides along as a third element so
    the view can draw a detection differently from a forecast across a missed frame - the scan's own honesty rule,
    kept through to the last consumer.

    ``team_colors`` is each team's measured kit color as ``(r, g, b)``, ``None`` where the clustering could not
    separate the kits: the component paints its markers with the color that was actually on the pitch, and keeps
    its own palette only as the fallback.

    ``camera_xy`` is the camera's own ground position (its X/Y, ignoring height). The replay draws a line from it
    to the aim point each frame, so the direction the camera is pointing is visible on the pitch. It is ``None``
    when the caller has no calibration to give one.

    ``roles`` labels the few non-team people the build could identify (``{"role": "referee"}``, or a keeper with
    its goal ``side``), attached per track so the view can color-code them; ``attack`` is the compact per-half
    direction pair from :func:`attack_summary`, so the view can point each team at the goal it attacks. Both are
    optional and absent from payloads built before this existed.
    """
    aim = [
        [_round(x), _round(y)] if np.isfinite(x) and np.isfinite(y) else None
        for x, y in np.asarray(aim_xy, dtype=np.float64)
    ]

    camera = None
    if camera_xy is not None:
        camera = [_round(camera_xy[0]), _round(camera_xy[1])]

    ball_out = None
    if ball is not None:
        ball_xy, ball_measured = (np.asarray(part, dtype=np.float64) for part in ball)
        ball_out = [
            [_round(x), _round(y), int(m)] if np.isfinite(x) and np.isfinite(y) else None
            for (x, y), m in zip(ball_xy, ball_measured)
        ]

    # Who was on the ball each frame, by proximity to the camera's aim point (the same proxy stage_b uses).
    # Bystanders are excluded first: they are not players, and counting their proximity to the aim point
    # would credit touches to people who cannot touch a ball. Role-labeled tracks are never bystanders,
    # however still they stand: keepers hang around one goal and drift little, which trips the "small AND still"
    # rule measured on sideline crowds - but a keeper is exactly who must be drawn (15 of 28 role tracks were
    # being dropped from the payload before this).
    field_players = [
        track
        for track in players
        if (roles is not None and int(track.track_id) in roles) or not _is_bystander(track)
    ]
    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for track in field_players:
        for frame, position in zip(track.frame, track.xy):
            by_frame.setdefault(int(frame), []).append((int(track.track_id), float(position[0]), float(position[1])))
    touches: dict[int, int] = {}
    for frame, entries in by_frame.items():
        if frame >= len(aim) or aim[frame] is None:
            continue
        ax, ay = aim[frame]
        best_track, best_distance = None, POSSESSION_RADIUS_M
        for track_id, x, y in entries:
            distance = float(np.hypot(x - ax, y - ay))
            if distance <= best_distance:
                best_track, best_distance = track_id, distance
        if best_track is not None:
            touches[best_track] = touches.get(best_track, 0) + 1

    out_players = []
    for track in sorted(field_players, key=lambda item: int(item.track_id)):
        speed = np.asarray(track.speed_kmh, dtype=np.float64)
        moving = speed[speed > 0.5]
        entry = {
            "track_id": int(track.track_id),
            "team": int(track.team),
            "frames": [int(f) for f in track.frame],
            "xy": [[_round(x), _round(y)] for x, y in track.xy],
            "speed": [_round(s) for s in speed],
            "stats": {
                "observations": int(len(track.frame)),
                "time_s": _round(len(track.frame) / max(fps, 1e-6)),
                "first_t": _round(track.time[0]) if len(track.time) else 0.0,
                "last_t": _round(track.time[-1]) if len(track.time) else 0.0,
                "distance_m": _round(track.distance_m),
                "top_speed_kmh": _round(float(speed.max())) if len(speed) else 0.0,
                "mean_speed_kmh": _round(float(moving.mean())) if len(moving) else 0.0,
                "mean_x": _round(float(np.mean(track.xy[:, 0]))),
                "mean_y": _round(float(np.mean(track.xy[:, 1]))),
                "touches": int(touches.get(int(track.track_id), 0)),
            },
        }
        if roles is not None and int(track.track_id) in roles:
            entry["role"] = dict(roles[int(track.track_id)])
        out_players.append(entry)

    return {
        "pitch": [_round(pitch[0]), _round(pitch[1])],
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_s": _round(frame_count / max(fps, 1e-6)),
        "team_names": list(team_names),
        "team_colors": None if team_colors is None else [_kit_color(entry) for entry in team_colors],
        "aim": aim,
        "camera": camera,
        "ball": ball_out,
        "players": out_players,
        "attack": attack,
        # How many tracked people were judged bystanders (touchline coaches, photographers, spectators) and
        # left out of ``players``. The view draws the field of play only; the count keeps the exclusion honest
        # - the page can say how many tracked people are not shown rather than silently shrinking the world.
        "bystanders_excluded": len(players) - len(field_players),
    }


def player_table_rows(replay: dict, numbers: dict[int, dict] | None = None):  # noqa: ANN201
    """One row per player for the dashboard table; ``numbers`` merges shirt numbers/names in.

    Sorted by time on screen descending: with fragmented real-footage tracks there are hundreds of rows, and the
    players who were actually followed belong at the top. The team names come out of the replay payload itself -
    it already carries them for the frontend, so the table cannot end up labeling the teams differently from the
    map beside it.
    """
    import pandas as pd

    numbers = numbers or {}
    team_names = list(replay.get("team_names") or [])
    rows = []
    for player in replay["players"]:
        stats = player["stats"]
        identity = numbers.get(int(player["track_id"]), {})
        rows.append(
            {
                "track": player["track_id"],
                "team": "referee/other" if player["team"] < 0 else team_name(player["team"], team_names),
                "number": identity.get("number"),
                "name": identity.get("name") or "",
                "source": identity.get("source") or "unassigned",
                "seen (s)": stats["time_s"],
                "distance (m)": stats["distance_m"],
                "top speed (km/h)": stats["top_speed_kmh"],
                "mean speed (km/h)": stats["mean_speed_kmh"],
                "touches (proxy)": stats["touches"],
                "mean x (m)": stats["mean_x"],
                "mean y (m)": stats["mean_y"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("seen (s)", ascending=False).reset_index(drop=True)


def track_boxes(players) -> dict[str, np.ndarray]:
    """Each tracked player's own image boxes, keyed by track id as a string, for the centered-clip cutter.

    These deliberately do *not* travel inside the replay payload. The component draws pitch positions and never
    looks at a box, while a whole game's boxes are ~35 MB of JSON - which the browser would download and parse on
    every view of a match just so that a clip cut later, from Python, could follow one player. They are written
    beside the payload as a small ``.npz`` instead (float16: the crop math cannot see the difference, and it
    halves the file), and only the page's own code reads them.

    A player whose track has no boxes (built without detections, or a payload from an older build) is left out
    rather than given zeros, which would frame its clips in the frame's corner.
    """
    out: dict[str, np.ndarray] = {}
    for player in players:
        boxes = getattr(player, "box", None)
        if boxes is None or len(boxes) == 0:
            continue
        out[str(int(player.track_id))] = np.asarray(boxes, dtype=np.float16)
    return out

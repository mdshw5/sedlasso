"""Stage B: turn raw Stage A detections into pitch-space tracks, teams, metrics, events and momentum.

Everything here is cheap and re-runnable (seconds for a match), because it consumes Stage A's stored output. That is
the point of the split: re-drawing pitch landmarks, re-tuning the tracker, or changing the match format never
requires decoding the video again.

Honesty about what is and is not measurable with one following camera:
* positions/teams/distances/territory/momentum are derived from players we can see on the pitch;
* the ball has its own dedicated scan (``analysis.ball``, run from the dashboard); ``analysis.event_detection``
  reads that scan together with these tracks to infer goals, shots, corners, penalties, clearances and tackles, and
  the dashboard offers them as review candidates - saves and blocks stay manual tags, because a keeper's save and a
  shot wide look the same to a ball track;
* this stage's ball *proxy* is the aim point - the pitch position the camera was pointed at each frame;
* the camera sees part of the pitch at a time, so no full-pitch formation is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from soccer_analytics.analysis.ball_lock import BallImage, POSSESSION_IMAGE_RADIUS, foot_distance
from soccer_analytics.analysis.kit import DESCRIPTOR_SIZE, kit_rgb
from soccer_analytics.analysis.projection import PitchDetections, on_pitch_mask

# Physics-limited gating: a player covers a few meters per second at most, so the per-frame gate is a *duration's*
# worth of sprint - 7 m/s - divided by the analysis rate, not a fixed distance. The gate cannot be generous: in
# packed play two players can be a meter apart, and the simulation showed that a 6 m gate mixes 33% of tracks (at
# 5 fps). Position alone is not enough there, so appearance (kit color) is part of the cost.
DEFAULT_RATE = 5.0  # the rate every second-based value below was tuned at
MAX_STEP_PER_S = 7.0  # meters a sprinting player may cover between consecutive samples (1.4 m at 5 fps)
SIGMA_STEP_FACTOR = 1.5  # extra slack where the position itself is uncertain (far side of the pitch)
MAX_GATE_M = 3.0
MAX_AGE_GATE_FACTOR = 1.5  # a track lost for a while may be re-acquired a little further away, but not far
AGE_GATE_PER_S = 0.5  # the re-acquisition slack added per second of absence (0.1 per frame at 5 fps)
KIT_COST_WEIGHT = 3.0  # meters of equivalent position error per unit of kit-color distance
MIN_KIT_FOR_COST = 0.2  # kit fraction below which the color descriptor is not trusted
TRACK_BUFFER_S = 3.0  # seconds a track survives without a detection (15 frames at 5 fps)
VELOCITY_GAIN = 0.5  # per-frame EMA gain for track velocity at the reference rate; rate-scaled like the ball's
MIN_TRACK_OBSERVATIONS = 8
MIN_KIT_OBSERVATIONS = 6
SPEED_PERCENTILE = 95
MAX_PLAUSIBLE_SPEED_KMH = 36.0  # faster than any human sprint: a meter of far-side error, not a real speed
MAX_GAP_FOR_DISTANCE_S = 1.0  # movement across a longer unobserved gap is unknown; do not invent it as distance
_UNREACHABLE = 1e6  # cost above the gate: used to forbid an assignment rather than to rank it
_UNMATCHED_COST = 1e6  # cost of leaving a fragment unjoined; above any real join's cost, so joins are preferred


def detection_rate(detections: PitchDetections) -> float:
    """The sampling rate of a detections array, read back from its own timestamps.

    Stage A stores each analyzed frame's source time, so the rate is measured rather than assumed. Every
    frame-count window in this module is seconds-based and scaled with this value, which keeps the tracking
    behavior identical at 5 fps (the historical rate) and correct at 15 fps (the default).
    """
    times = np.asarray(detections.time, dtype=np.float64)
    if len(times) >= 2:
        steps = np.diff(np.unique(times))
        dt = float(np.median(steps)) if len(steps) else 0.0
        if dt > 1e-6:
            return 1.0 / dt
    return DEFAULT_RATE


@dataclass
class TrackAssignment:
    track_id: np.ndarray  # (D,) -1 where unassigned
    team: dict[int, int]  # track id -> team index (0/1), absent if unknown
    kit_quality: dict[int, float]  # track id -> how much its label can be trusted (0..1); low means "guess"
    tracks: dict[int, np.ndarray]  # track id -> detection row indices


@dataclass
class PlayerTrack:
    """One tracked person, summarized. Positions are pitch meters with a per-point uncertainty."""

    track_id: int
    team: int
    frame: np.ndarray
    time: np.ndarray
    xy: np.ndarray
    sigma_m: np.ndarray
    speed_kmh: np.ndarray
    distance_m: float
    # (N, 4) the player's own detection boxes over the same observations, ``(x1, y1, x2, y2)`` normalized by frame
    # width. Where the track is on the pitch is one question; where it is *in the picture* is another, and cutting
    # a clip that follows this player needs the second. None when the caller built tracks without detections.
    box: np.ndarray | None = None


@dataclass
class TeamMetrics:
    team: int
    players_observed: int
    distance_m: float
    top_speed_kmh: float
    mean_speed_kmh: float
    mean_x_fraction: float  # mean pitch x of the team's players, 0 = own goal line, 1 = opponent's
    # Share of *contested frames* the team had the nearest player to the ball, combined across the match by frames
    # rather than by minutes. A contested frame is one where a labelled player was within the possession gate - in
    # the picture when a ball scan (or lock) supplied a position, else in pitch space around the camera aim.
    # Stoppages and frames with no labelled player nearby are excluded.
    possession_share: float
    # The team's mean kit color as RGB, or None when the kits could not be separated. It is what makes "team 0"
    # recognizable - it anchors the name the user gives the team to the color they can see on the pitch.
    kit_rgb: tuple[int, int, int] | None = None


@dataclass
class MatchReport:
    teams: list[TeamMetrics]
    players: list[PlayerTrack]
    # Minute of the recording -> {"team_0": float, "team_1": float, "action_x": float}. The minute is the source
    # video's own clock - the clock the events and the highlight clips use; the dashboard shifts the buckets onto
    # the analyzed window's clock when it draws the timeline strip.
    momentum: dict
    pitch_length_m: float
    pitch_width_m: float
    frames_analyzed: int
    detections_used: int
    # Which measurement decided possession and how much of the match it covered: ``space`` is "image", "mixed"
    # or "pitch"; the frame counts make the claim auditable instead of a boolean. Absent from older reports.
    possession: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _kit_cost(
    detections: PitchDetections, here: np.ndarray, ids: list[int], states: dict[int, dict]
) -> np.ndarray:
    """Color distance between each track's running kit estimate and each candidate detection, in kit units.

    Break-even: a track keeps its own color (distance ~0) unless another detection is more than
    ``KIT_COST_WEIGHT``-times closer in color, which is what happens when two players cross.
    """
    color = detections.kit[here, 1:6]
    valid = detections.kit[here, 0] > MIN_KIT_FOR_COST
    cost = np.zeros((len(ids), len(here)))
    for i, tid in enumerate(ids):
        track_color = states[tid].get("kit")
        if track_color is None:
            continue
        distance = np.linalg.norm(color - track_color[None, :], axis=1)
        cost[i] = np.where(valid, distance, 0.0)
    return cost


def _track_people(detections: PitchDetections, keep: np.ndarray, on_progress=None) -> TrackAssignment:
    """Greedy-by-cost assignment of detections to tracks with constant-velocity prediction, in pitch meters.

    Association is gated by each detection's own uncertainty, so a far-side player (sigma of meters) is allowed a
    larger jump than a near-side one, and the gate tightens rather than loosens when the geometry is good.

    Stage A's BoT-SORT identities (``detections.det_track``) are the *hints*, not the authority: they are
    image-space, so a camera whip can hand two different players one identity, and the pitch-space gate below is
    what says whether a continuation is physically possible. A hint is honored only when it agrees with the
    geometry - the detection claims a track whose predicted position is within the gate - and a hint that fails
    the gate is ignored, exactly like a wrong kit color would be. Where there is no hint (v1 chunks, or a
    detection the tracker dropped), the assignment runs on position and kit alone, as it always did.
    """
    rows = np.where(keep)[0]
    track_id = np.full(len(detections.xy), -1, dtype=np.int32)
    tracks: dict[int, np.ndarray] = {}
    if len(rows) == 0:
        return TrackAssignment(track_id, {}, {}, tracks)

    # Every window here is seconds-based and scaled to the rate the detections were sampled at: 7 m/s of sprint
    # per second, 3 s of memory, and an EMA that keeps its time constant instead of adapting all the faster for
    # having more frames.
    rate = detection_rate(detections)
    buffer_frames = max(1, int(round(TRACK_BUFFER_S * rate)))
    max_step_m = MAX_STEP_PER_S / rate
    age_slack = AGE_GATE_PER_S / rate
    velocity_gain = 1.0 - (1.0 - VELOCITY_GAIN) ** (DEFAULT_RATE / rate)

    frame_of = detections.frame
    states: dict[int, dict] = {}  # id -> last xy, velocity, last frame
    # Stage A's identities are offset into their own space: they arrive as small non-negative ints that mean
    # nothing to this pass's own numbering, so each is mapped to a local id the first time it is honored.
    bot_ids = getattr(detections, "det_track", None)
    bot_of_row = {int(row): int(bot_ids[row]) for row in rows if bot_ids is not None and bot_ids[row] >= 0}
    bot_to_local: dict[int, int] = {}
    next_id = 0
    unique_frames = np.unique(frame_of[rows])
    progress_every = max(1, len(unique_frames) // 25)
    for index, frame in enumerate(unique_frames):
        if on_progress is not None and (index % progress_every == 0 or index == len(unique_frames) - 1):
            on_progress((index + 1) / len(unique_frames))
        here = rows[frame_of[rows] == frame]
        age_out = [tid for tid, st in states.items() if frame - st["frame"] > buffer_frames]
        for tid in age_out:
            del states[tid]
        if not states or len(here) == 0:
            for row in here:
                states[next_id] = {
                    "xy": detections.xy[row],
                    "vel": np.zeros(2),
                    "frame": int(frame),
                    "kit": detections.kit[row, 1:6].copy() if detections.kit[row, 0] > MIN_KIT_FOR_COST else None,
                }
                track_id[row] = next_id
                if row in bot_of_row:
                    bot_to_local[bot_of_row[row]] = next_id
                next_id += 1
            continue

        ids = sorted(states)
        predicted = np.array([states[tid]["xy"] + states[tid]["vel"] * (frame - states[tid]["frame"]) for tid in ids])
        measured = detections.xy[here]
        sigma = np.maximum(detections.sigma_m[here], 0.35)
        age = np.array([frame - states[tid]["frame"] for tid in ids])
        position_cost = np.linalg.norm(predicted[:, None, :] - measured[None, :, :], axis=2)
        gate = np.minimum(max_step_m + SIGMA_STEP_FACTOR * sigma[None, :], MAX_GATE_M) * np.minimum(
            1.0 + age_slack * age, MAX_AGE_GATE_FACTOR
        )[:, None]
        cost = position_cost + KIT_COST_WEIGHT * _kit_cost(detections, here, ids, states)
        cost = np.where(position_cost <= gate, cost, _UNREACHABLE)
        assigned_rows, assigned_cols = linear_sum_assignment(cost)
        matched: set[int] = set()
        for i, j in zip(assigned_rows, assigned_cols):
            if cost[i, j] >= _UNREACHABLE:
                continue
            tid, row = ids[i], here[j]
            step = (detections.xy[row] - states[tid]["xy"]) / max(1, frame - states[tid]["frame"])
            states[tid]["vel"] = (1.0 - velocity_gain) * states[tid]["vel"] + velocity_gain * step
            states[tid]["xy"] = detections.xy[row]
            states[tid]["frame"] = int(frame)
            if detections.kit[row, 0] > MIN_KIT_FOR_COST:
                # Running kit estimate: slow enough that one bad crop cannot change a track's identity.
                previous = states[tid].get("kit")
                current = detections.kit[row, 1:6]
                states[tid]["kit"] = current.copy() if previous is None else 0.85 * previous + 0.15 * current
            track_id[row] = tid
            if row in bot_of_row:
                bot_to_local[bot_of_row[row]] = tid
            matched.add(int(row))
        for row in here:  # unmatched detections start new tracks (a player entering the camera's view)
            if int(row) in matched:
                continue
            # A BoT-SORT hint can rescue a detection the geometry alone would have made a new track of: the
            # tracker has seen this identity before, so if that identity is a live state whose prediction is
            # within the gate, continue it instead of forking. This is the case the ReID embedding exists for -
            # two players crossing, where position alone is ambiguous but appearance is not.
            hint = bot_to_local.get(bot_of_row.get(row, -1), None)
            if hint is not None and hint in states:
                st = states[hint]
                gap = frame - st["frame"]
                jump = np.linalg.norm(detections.xy[row] - (st["xy"] + st["vel"] * gap))
                allowed = np.minimum(max_step_m + SIGMA_STEP_FACTOR * max(detections.sigma_m[row], 0.35), MAX_GATE_M)
                allowed *= min(1.0 + age_slack * gap, MAX_AGE_GATE_FACTOR)
                if jump <= allowed:
                    st["vel"] = (1.0 - velocity_gain) * st["vel"] + velocity_gain * ((detections.xy[row] - st["xy"]) / max(1, gap))
                    st["xy"] = detections.xy[row]
                    st["frame"] = int(frame)
                    if detections.kit[row, 0] > MIN_KIT_FOR_COST:
                        previous = st.get("kit")
                        current = detections.kit[row, 1:6]
                        st["kit"] = current.copy() if previous is None else 0.85 * previous + 0.15 * current
                    track_id[row] = hint
                    continue
            states[next_id] = {
                "xy": detections.xy[row],
                "vel": np.zeros(2),
                "frame": int(frame),
                "kit": detections.kit[row, 1:6].copy() if detections.kit[row, 0] > MIN_KIT_FOR_COST else None,
            }
            track_id[row] = next_id
            if row in bot_of_row:
                bot_to_local[bot_of_row[row]] = next_id
            next_id += 1

    for row in rows:
        if track_id[row] >= 0:
            tracks.setdefault(int(track_id[row]), np.array([], dtype=int))
    for row in rows:
        if track_id[row] >= 0:
            tracks[int(track_id[row])] = np.append(tracks[int(track_id[row])], row)
    return TrackAssignment(track_id, {}, {}, tracks)


# Offline stitching. The online tracker cannot follow a player through a big camera move (the gimbal whips between
# the ends of the pitch), and a player who leaves a tight shot and returns gets a new track. Measured on the real
# sample: 847 raw tracks, 302 of which start within 6 frames and 4 m of where an earlier track ended - the same
# players, redetected. Those fragments are reconnected after the fact.
STITCH_MAX_GAP_S = 6.0  # fragments this far apart in time are not stitched (30 frames at 5 fps)
STITCH_BASE_M = 2.0  # slack at the join, on top of the distance a sprint could cover
STITCH_SPRINT_M_S = 7.0  # a full sprint between two observations (1.4 m per frame at 5 fps)
STITCH_MAX_KIT_DISTANCE = 0.45  # kit colors (L, a, b, saturation, value) further apart than this never stitch
STITCH_NO_KIT_FACTOR = 0.6  # with no color evidence the spatial window tightens

KIT_SIZE_MIN_HEIGHT = 60.0  # px at 1920 width: below this a torso crop is a few pixels and reads gray
KIT_SIZE_MAX_HEIGHT = 200.0  # px at 1920 width: above this the box is a near-sideline bystander, not a player
KIT_SIZE_QUANTILE = 0.75  # per track, only its largest quartile of observations is trusted for color
TEAM_MIN_KIT_FRACTION = 0.55  # a track whose crops are mostly grass is not a kit reading at all
TEAM_MIN_EVIDENCE = 6.0  # sqrt of player-sized observations: below this a track's color is a guess
TEAM_LABEL_MAX_DISTANCE = 2.5  # standardized descriptor units: further than this from every center, no team color
TEAM_LABEL_MIN_MARGIN = 0.35  # how far the winning center must beat the runner-up before the label is trusted
TEAM_LABEL_MIN_HEIGHT = 32.0  # px at 1920: the propagation tier may read these crops; the voting tier may not


def _team_min_evidence(frames: int, rate: float = DEFAULT_RATE) -> float:
    """The evidence floor for team clustering, scaled to how long the segment is.

    The floor is tuned on a whole-game segment (21.5k frames at 5 fps), where a well-seen player collects hundreds
    of player-sized observations and 36 of them (sqrt = 6) is a trustworthy color. A short synthetic match (400
    frames) yields ~11 observations per track, so the same absolute floor would leave nothing to cluster. The
    frame count is first normalized to the 5 fps equivalent, so a 15 fps run of the same match weighs the same;
    the floor then scales with the square root of length: a 400-frame match needs ~9 observations, the whole game
    needs 36 - and anything longer keeps the whole-game floor.
    """
    equivalent = float(frames) * DEFAULT_RATE / max(float(rate), 1e-6)
    scaled = 6.0 * np.sqrt(max(equivalent, 1.0) / 21552.0)
    return float(np.clip(scaled, 2.6, 6.0))


def _track_kit_descriptor(detections: PitchDetections, rows: np.ndarray) -> np.ndarray | None:
    """One kit descriptor per track, taken from its clearest PLAYER-SIZED observations.

    Two failure modes are measured on the real game, and both poison the team colors:
    * most detections are tiny far-side figures whose torso crop is a handful of gray pixels - a median over all
      of a track's observations is washed out even when the same player was seen up close reading pure kit color;
    * the very largest boxes (200 px+ at 1920) are not players at all but near-sideline bystanders - coaches,
      photographers, spectators - whose random clothing lands in whichever cluster is nearest. So "largest" is
      capped: the trusted band is player-sized boxes, and anything bigger is ignored for color.
    A track that was never seen in that band has no trustworthy color at all, so it returns None rather than a
    gray median that would drag the team color toward gray. Returns None when there is not enough evidence.
    """
    chosen = _track_kit_rows(detections, rows)
    if chosen is None:
        return None
    return np.median(detections.kit[chosen, 1:6], axis=0)


def _track_kit_rows(
    detections: PitchDetections, rows: np.ndarray, *, min_height: float = KIT_SIZE_MIN_HEIGHT
) -> np.ndarray | None:
    """The observations a track's kit color is read from (player-sized band, largest quartile), or None.

    ``min_height`` is the floor of that band. The voting tier keeps the measured 60 px: below it a crop is a few
    pixels and reads gray, and a gray median inside the cluster centers poisons who is team 0. The propagation
    tier - which only decides which already-fixed center a track belongs to, and where a wrong guess costs one
    fragment's frames while no guess costs whole minutes of possession - may lower it to ``TEAM_LABEL_MIN_HEIGHT``.
    """
    usable = rows[detections.kit[rows, 0] > MIN_KIT_FOR_COST]
    if len(usable) < MIN_KIT_OBSERVATIONS:
        return None
    # abs(): the synthetic simulator emits inverted boxes (top below foot in image coords), and a box's size
    # is still meaningful there. Real detections are never inverted; this only keeps the band honest.
    heights = np.abs(detections.height_px[usable])
    in_band = (heights >= min_height) & (heights <= KIT_SIZE_MAX_HEIGHT)
    band = usable[in_band]
    if len(band) < MIN_KIT_OBSERVATIONS:
        return None
    band_heights = heights[in_band]
    big = band[band_heights >= np.quantile(band_heights, KIT_SIZE_QUANTILE)]
    return big if len(big) >= MIN_KIT_OBSERVATIONS else band


def _track_kit_evidence(
    detections: PitchDetections, rows: np.ndarray, *, min_height: float = KIT_SIZE_MIN_HEIGHT
) -> tuple[np.ndarray, float] | None:
    """Full color evidence for a track: the 11-float descriptor (L a b sat val + 6 hue bins) and how much of it
    there is - the number of player-sized observations the color was medianed over, square-rooted so a handful of
    long-lived tracks cannot shout down everyone else. Team clustering weights tracks by this; stitching keeps the
    plain 5-float descriptor because hue bins would double-count what L/a/b already say. ``min_height`` is passed
    through to :func:`_track_kit_rows` - see there for why the propagation tier reads smaller crops."""
    chosen = _track_kit_rows(detections, rows, min_height=min_height)
    if chosen is None:
        return None
    return np.median(detections.kit[chosen, 1:12], axis=0), float(np.sqrt(len(chosen)))


def _track_kit_colors(detections: PitchDetections, tracks: dict[int, np.ndarray]) -> dict[int, np.ndarray | None]:
    """Kit descriptor per track (None when too few usable crops), used only to gate stitching."""
    return {track_id: _track_kit_descriptor(detections, rows) for track_id, rows in tracks.items()}


def _join_fragments(cost: dict[tuple[int, int], float], fragment_ids: list[int]) -> dict[int, int]:
    """Which fragments continue which: the globally cheapest set of gate-passing joins, one per fragment.

    The rule this replaced - join only when each side is the other's *best* candidate - starved exactly the
    crowded moments it was meant for: on the 2026-10-03 game only 7,412 of 235,618 gate-passing pairs were
    joined and 14,617 fragments lost their preferred continuation to a contest, so players in a crowd stayed
    split and the leftover pieces then read as bystanders. Every pair here already passed the gates (sprint
    distance, kit compatibility); matching only decides *which* of those valid joins to take, consistently - a
    fragment gets at most one predecessor and one successor - by minimising the total cost. Leaving a fragment
    unjoined is allowed at a cost above any real join, so more joins always beat fewer.
    """
    position = {track: index for index, track in enumerate(fragment_ids)}
    count = len(fragment_ids)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for (before, after), value in cost.items():
        rows.append(position[before])
        cols.append(position[after])
        values.append(value + 1.0)  # keep every real edge strictly above the sparse zero and below _UNMATCHED_COST
    for index in range(count):
        rows.append(index)
        cols.append(count + index)
        values.append(_UNMATCHED_COST)
    matrix = coo_matrix((values, (rows, cols)), shape=(count, 2 * count)).tocsr()
    matched_rows, matched_cols = min_weight_full_bipartite_matching(matrix)
    return {
        fragment_ids[int(before)]: fragment_ids[int(after)]
        for before, after in zip(matched_rows, matched_cols)
        if after < count
    }


def _stitch_tracks(detections: PitchDetections, assignment: TrackAssignment) -> TrackAssignment:
    """Reconnect track fragments that the online pass had to break.

    A fragment is a stitch candidate for a later fragment when it starts where the earlier one plausibly could have
    moved to - within a sprint plus slack - and the two wear compatible kit colors (when both have color
    evidence). Which of the valid joins are taken is decided globally: every fragment gets at most one
    continuation and one predecessor, and the set of joins minimises the total cost meanwhile preferring to join
    at all (see :func:`_join_fragments`). Chains are resolved afterwards: a player who left and returned twice
    becomes one track.
    """
    tracks = {track_id: np.sort(rows) for track_id, rows in assignment.tracks.items()}
    if len(tracks) < 2:
        return assignment
    rate = detection_rate(detections)
    max_gap = max(1, int(round(STITCH_MAX_GAP_S * rate)))
    sprint_per_frame = STITCH_SPRINT_M_S / rate
    last = {track_id: (int(detections.frame[rows[-1]]), detections.xy[rows[-1]]) for track_id, rows in tracks.items()}
    first = {track_id: (int(detections.frame[rows[0]]), detections.xy[rows[0]]) for track_id, rows in tracks.items()}
    kits = _track_kit_colors(detections, tracks)

    cost: dict[tuple[int, int], float] = {}
    for before, (end_frame, end_xy) in last.items():
        for after, (start_frame, start_xy) in first.items():
            if before == after:
                continue
            gap = start_frame - end_frame
            if gap <= 0 or gap > max_gap:
                continue
            distance = float(np.linalg.norm(start_xy - end_xy))
            kit_before, kit_after = kits[before], kits[after]
            if kit_before is not None and kit_after is not None:
                kit_distance = float(np.linalg.norm(kit_before - kit_after))
                if kit_distance > STITCH_MAX_KIT_DISTANCE:
                    continue
                if distance > STITCH_BASE_M + sprint_per_frame * gap:
                    continue
                pair_cost = distance / gap + 2.0 * kit_distance
            else:
                # No color evidence on one side: only short, small jumps are believable.
                if distance > STITCH_BASE_M + STITCH_NO_KIT_FACTOR * sprint_per_frame * gap:
                    continue
                pair_cost = distance / gap + 0.5
            if not np.isfinite(pair_cost):
                continue  # a degenerate position cannot rank a join
            cost[(before, after)] = pair_cost

    successor = _join_fragments(cost, sorted(tracks))

    # Walk the chains from their heads (members with no predecessor) so a player who left and returned twice
    # becomes a single track under its first id.
    has_predecessor = set(successor.values())
    chains: dict[int, list[int]] = {}
    for head in tracks:
        if head in has_predecessor:
            continue
        members, cursor = [head], head
        while cursor in successor:
            cursor = successor[cursor]
            if cursor in members:  # cannot happen with strictly increasing frames; cheap insurance
                break
            members.append(cursor)
        chains[head] = members

    track_id = assignment.track_id.copy()
    merged_tracks: dict[int, np.ndarray] = {}
    for head, members in chains.items():
        merged_tracks[head] = np.concatenate([tracks[member] for member in members]) if len(members) > 1 else tracks[head]
        for member in members[1:]:
            track_id[assignment.track_id == member] = head
    return TrackAssignment(track_id, assignment.team, assignment.kit_quality, merged_tracks)


def _team_assignment(
    detections: PitchDetections, assignment: TrackAssignment, *, num_teams: int = 2
) -> tuple[dict[int, int], dict[int, float], dict[int, np.ndarray]]:
    """Cluster per-track kit descriptors into teams, plus a possible "other" cluster for the referee.

    Returns the team label per track (``-1`` for a cluster identified as neither team), how much each label can be
    trusted (the voters by the clustering separation, the propagated tracks by how close their color landed to a
    center), and each team's mean kit descriptor - the color that tells the two teams apart, read back out for
    display.
    Three things make this honest rather than confident:
    * only tracks with real color evidence may vote: enough player-sized observations (the sqrt-weight also
      down-weights short tracks) and crops that were mostly kit rather than grass. Measured on the real game this
      cuts ~2800 tracks to ~150 - the rest are fragments whose gray medians would outvote the real kits;
    * the cluster centers are SEEDED from the a-axis extremes (reddest vs least-red strong track) - kits are
      usually told apart by redness vs not-redness, and seeding from saturation lands on whatever is most garish
      (measured: a dark-orange bystander) instead of on a kit;
    * a third cluster is only believed when there are enough tracks, and it is only treated as "other" when it is
      clearly smaller than the two team clusters - that is the referee or a stray track, who would otherwise be
      counted as a team member and would inflate that team's numbers (a referee follows the ball everywhere).
    The bullets above decide the cluster centers; every other track that has color evidence afterwards joins the
    nearest center, if it is close enough and clearly nearest. That is what gives MOST tracks a label rather than
    the voting few - possession and the momentum curve read the label of whichever player is nearest the ball, and
    they need the label on the player in frame, not only on the player who once stood still enough to vote. This
    second tier reads color from smaller crops than the voters do (``TEAM_LABEL_MIN_HEIGHT``): measured on the
    real game the voting bar alone left ~90% of on-pitch detections unlabeled, while the kits stayed plainly
    separable by eye down to ~30 px.
    Separation is a between/within ratio, so identical kits cannot score as a confident split.
    """
    from sklearn.cluster import KMeans

    rate = detection_rate(detections)
    per_track: dict[int, tuple[np.ndarray, float]] = {}
    for tid, rows in assignment.tracks.items():
        evidence = _track_kit_evidence(detections, rows)
        if evidence is not None:
            per_track[tid] = evidence
    if len(per_track) < num_teams * 2:
        return {}, {tid: 0.0 for tid in assignment.tracks}, {}

    # Only tracks with REAL color evidence may vote: enough player-sized observations, and crops that were
    # mostly kit rather than grass (a crop that is half grass is not a kit reading at all). Measured on the real
    # game this cuts ~2800 tracks to ~150 - the rest are fragments whose gray medians would outvote the kits.
    strong: dict[int, tuple[np.ndarray, float]] = {}
    for tid, (descriptor, weight) in per_track.items():
        chosen = _track_kit_rows(detections, assignment.tracks[tid])
        if chosen is None:
            continue
        if float(np.mean(detections.kit[chosen, 0])) < TEAM_MIN_KIT_FRACTION:
            continue
        if weight < _team_min_evidence(len(detections.time), rate):
            continue
        strong[tid] = (descriptor, weight)
    if len(strong) < num_teams * 2:
        return {}, {tid: 0.0 for tid in assignment.tracks}, {}

    ids = sorted(strong)
    features = np.stack([strong[tid][0] for tid in ids])
    weights = np.array([strong[tid][1] for tid in ids])
    feature_mean = features.mean(0)
    feature_std = features.std(0) + 1e-6
    features = (features - feature_mean) / feature_std
    clusters = 3 if len(ids) >= 3 * num_teams else num_teams

    # Seed from the a-axis extremes: the reddest and the least-red strong track (plus the least saturated as the
    # "other" seed when a third cluster is wanted). Kits are usually told apart by redness vs not-redness, and
    # seeding from the extremes beats seeding from saturation, which lands on whatever is most garish (measured:
    # a dark-orange bystander) instead of on a kit.
    a_column = features[:, 1]
    seeds = [int(np.argmax(a_column)), int(np.argmin(a_column))]
    if clusters == 3:
        seeds.append(int(np.argmin(features[:, 3])))
    kmeans = KMeans(n_clusters=clusters, n_init=1, init=features[seeds], random_state=0).fit(
        features, sample_weight=weights
    )
    centers, labels = kmeans.cluster_centers_, kmeans.labels_

    sizes = np.bincount(labels, minlength=clusters)
    # The two TEAM clusters are the two most colorful (highest mean saturation in standardized space), not the
    # two largest: measured on the real game the red team was the SMALLEST of three clusters (36 of 150 strong
    # tracks) and the size rule dropped it in favor of a gray/teal junk cluster. A gray "other" cluster is
    # unsaturated by definition, so saturation picks the kits and size never overrides them.
    cluster_sat = [float(features[labels == c, 3].mean()) if np.any(labels == c) else -1.0 for c in range(clusters)]
    other_cluster = None
    if clusters == 3:
        # The "other" cluster is whichever cluster looks like non-team people: clearly smaller than the teams
        # (a referee is one person) or clearly unsaturated (gray junk). Measured on the synthetic game, the
        # neon referee out-saturated both teams, so "least saturated" alone mislabels the referee a team and
        # "most saturated" alone mislabels a team the referee - each candidate is checked against both tests.
        reference = float(np.median(sizes)) or 1.0
        for candidate in range(clusters):
            small = sizes[candidate] < 0.4 * reference
            unsaturated = cluster_sat[candidate] <= 0.0
            if small or unsaturated:
                others = [c for c in range(clusters) if c != candidate]
                if float(sum(int(sizes[c]) for c in others)) >= 0.5 * float(sizes.sum()):
                    other_cluster = candidate
                    break
    team_clusters = (
        [c for c in range(clusters) if c != other_cluster]
        if other_cluster is not None
        else list(np.argsort(-np.asarray(cluster_sat))[:num_teams])
    )

    # Deterministic team ids across runs: the more red kit (standardized 'a' axis) is team 0.
    team_clusters = sorted(team_clusters[:num_teams], key=lambda c: -centers[c, 1])
    remap = {int(cluster): team for team, cluster in enumerate(team_clusters)}
    teams = {tid: remap.get(int(label), -1) for tid, label in zip(ids, labels)}

    # The voters fix the cluster centers; every OTHER track with color evidence is then labeled by where its own
    # descriptor falls. Without this the rest of the pipeline can only see the few dozen tracks that cleared the
    # voting bar - measured on the real game, that left possession and the momentum curve with no labeled
    # teammates visible in most minutes, and most one-minute buckets were dropped entirely. The propagation tier
    # reads color from smaller crops than the voters (``TEAM_LABEL_MIN_HEIGHT``): the goal is coverage, and a
    # wrong guess costs a fragment's frames while no guess costs whole minutes of possession. The nearest center
    # is only trusted within a radius and with a margin over the runner-up, so a gray blur - or the referee, once
    # there is no third cluster to say so - stays unlabeled instead of being forced onto a team.
    evidence_by_tid: dict[int, np.ndarray] = {tid: evidence[0] for tid, evidence in per_track.items()}
    weak_ids: list[int] = []
    weak_features: list[np.ndarray] = []
    weak_quality: dict[int, float] = {}
    for tid, rows in assignment.tracks.items():
        if tid in strong:
            continue
        evidence = per_track.get(tid)
        if evidence is None:
            evidence = _track_kit_evidence(detections, rows, min_height=TEAM_LABEL_MIN_HEIGHT)
        if evidence is not None:
            evidence_by_tid[tid] = evidence[0]
            weak_ids.append(tid)
            weak_features.append(evidence[0])
    if weak_ids:
        weak = (np.stack(weak_features) - feature_mean) / feature_std
        distances = np.linalg.norm(weak[:, None, :] - centers[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        best = distances[np.arange(len(weak_ids)), nearest]
        runner_up = np.partition(distances, 1, axis=1)[:, 1]
        for tid, cluster, close, clear, distance in zip(
            weak_ids, nearest, best <= TEAM_LABEL_MAX_DISTANCE,
            runner_up - best >= TEAM_LABEL_MIN_MARGIN, best,
        ):
            if close and clear:
                label = remap.get(int(cluster), -1)
                teams[tid] = label
                if label >= 0:
                    # How much of the trusted radius the match used: a track matched just inside the edge is a
                    # guess, one matched near a center is not. On the same 0..1 "how much to trust it" scale the
                    # voters get from the clustering separation, so the report's weak-label note can count real
                    # guesses instead of every track that was not a voter.
                    weak_quality[tid] = float(np.clip(1.0 - distance / TEAM_LABEL_MAX_DISTANCE, 0.0, 1.0))

    # Each team's color, medianed over the kit descriptors of the tracks that wear it (not the cluster center,
    # which lives in standardized space and no longer means a color). The median, not a saturation-weighted
    # mean: weighting by saturation amplifies whatever is most garish, and the most garish members of a cluster
    # are its grass leaks (measured: a red/blue split came out orange/green under the weighting).
    # The vector here is [L, a, b, sat, val] - descriptor columns 1..5 - so it is padded back into the full
    # 12-float layout (leading kit_fraction) that kit_rgb expects. Measured on the real game, passing the bare
    # 5-float vector read a/b/sat as L/a/b and reported a red team as "light gray".
    members: dict[int, list[np.ndarray]] = {}
    for tid, team in teams.items():
        if team >= 0 and tid in evidence_by_tid:
            members.setdefault(team, []).append(evidence_by_tid[tid][:5])
    colors: dict[int, np.ndarray] = {}
    for team, rows in members.items():
        padded = np.zeros(DESCRIPTOR_SIZE, dtype=np.float64)
        padded[0] = 1.0
        padded[1:6] = np.median(np.stack(rows), axis=0)
        colors[team] = padded

    within = float(np.mean(np.linalg.norm(features - centers[labels], axis=1)))
    between = float(np.linalg.norm(centers[team_clusters[0]] - centers[team_clusters[1]])) if len(team_clusters) >= 2 else 0.0
    separation = float(np.clip((between - within) / (between + within + 1e-9), 0.0, 1.0))
    quality: dict[int, float] = {tid: separation for tid in ids}
    quality.update(weak_quality)
    for tid in assignment.tracks:
        quality.setdefault(tid, 0.0)
    return teams, quality, colors


def _speeds(xy: np.ndarray, time: np.ndarray, sigma_m: np.ndarray) -> np.ndarray:
    """Per-observation speed in km/h, using each step's own displacement uncertainty and a physical cap.

    A step across a gap longer than ``MAX_GAP_FOR_DISTANCE_S`` is not reported as movement at all: the player was
    not seen in between, so whatever they did is unknown - counting the straight-line jump as distance would invent
    a sprint. The same rule governs the distance sums in ``build_tracks``.
    """
    if len(time) < 2:
        return np.zeros(len(time))
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    gap = np.diff(time)
    dt = np.clip(gap, 1e-3, None)
    observed = gap <= MAX_GAP_FOR_DISTANCE_S
    noise = np.hypot(sigma_m[:-1], sigma_m[1:])
    speed = (step / dt) * 3.6
    # Only report a speed when the displacement is meaningfully larger than the measurement noise of the two points.
    speed = np.where(observed & (step > np.maximum(0.35, noise)), speed, 0.0)
    return np.concatenate([[0.0], np.clip(speed, 0.0, MAX_PLAUSIBLE_SPEED_KMH)])


def build_tracks(
    detections: PitchDetections,
    keep: np.ndarray,
    assignment: TrackAssignment | None = None,
    teams: dict[int, int] | None = None,
) -> list[PlayerTrack]:
    """Summarize assigned detections into per-player tracks (distance, speed profile)."""
    if assignment is None:
        assignment = _stitch_tracks(detections, _track_people(detections, keep))
    teams = teams if teams is not None else _team_assignment(detections, assignment)[0]
    out: list[PlayerTrack] = []
    for tid, rows in assignment.tracks.items():
        rows = np.sort(rows)
        if len(rows) < MIN_TRACK_OBSERVATIONS:
            continue
        order = np.argsort(detections.time[rows])
        rows = rows[order]
        xy, time, sigma = detections.xy[rows], detections.time[rows], detections.sigma_m[rows]
        speed = _speeds(xy, time, sigma)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        observed = np.diff(time) <= MAX_GAP_FOR_DISTANCE_S
        distance = float(np.sum(steps * observed * (speed[1:] > 0)))
        box = getattr(detections, "box", None)
        out.append(
            PlayerTrack(
                tid, teams.get(tid, -1), detections.frame[rows], time, xy, sigma, speed, distance,
                box=np.asarray(box[rows], dtype=np.float64) if box is not None else None,
            )
        )
    return out


def _momentum(
    detections: PitchDetections,
    keep: np.ndarray,
    teams_by_track: dict[int, int],
    track_id: np.ndarray,
    bucket_s: float = 60.0,
    possession_radius_m: float = 12.0,
    ball_image: BallImage | None = None,
    on_progress=None,
) -> tuple[dict, dict]:
    """Per-minute action share per team, the mean x of the action (territory), and possession coverage.

    Possession is decided by the nearest *labelled player*, not by counting players. When a ball scan (or the
    gimbal lock) supplies an image position for the frame, the nearest foot in the picture is used; otherwise the
    camera's aim point is the pitch-space proxy. The image measure deliberately avoids the homography: a ball in
    the air has no ground point, and a poor calibration distorts pitch distances, while "which player is nearest
    the ball in the picture" is a direct measurement in both cases.

    Only players with a team label take part: a track the clustering could not place - the referee above all - is
    not allowed to win frames, but unlike before, *every* labeled player can, not only the few tracks that were
    strong enough to vote on the colors.

    Buckets are keyed by the minute of the *recording* (``detections.time`` is the source video's own clock - the
    same clock the event clips cut with). The dashboard shifts them onto the analyzed window's clock for the strip.

    Counting detections instead would let the referee decide the result: a referee follows the ball all match long, so
    they are the single nearest person very often, and exposure to them depends on how the kit clustering fell - this
    was measurable in the simulator and produced a 10-point swing.

    Returns ``(buckets, coverage)``. ``coverage`` counts the frames each measurement decided, so the report can say
    how much of possession is image-space rather than merely asserting that the code supports it.
    """
    frames = np.unique(detections.frame[keep])
    if len(frames) == 0:
        return {}, {"image_available_frames": 0, "image_contested_frames": 0, "aim_contested_frames": 0}
    buckets: dict[int, dict] = {}
    coverage = {"image_available_frames": 0, "image_contested_frames": 0, "aim_contested_frames": 0}
    progress_every = max(1, len(frames) // 25)
    for index, frame in enumerate(frames):
        if on_progress is not None and (index % progress_every == 0 or index == len(frames) - 1):
            on_progress((index + 1) / len(frames))
        aim = detections.aim_xy[frame] if frame < len(detections.aim_xy) else np.full(2, np.nan)
        if not np.all(np.isfinite(aim)):
            continue
        rows = np.where(keep & (detections.frame == frame))[0]
        teams = np.array([teams_by_track.get(int(track_id[row]), -1) for row in rows])
        players = teams >= 0
        minute = int(detections.time[rows[0]] // bucket_s)
        entry = buckets.setdefault(minute, {"won": [0, 0], "x": [], "uncontested": 0})
        entry["x"].append(float(aim[0]))
        image_here = (
            ball_image is not None
            and frame < ball_image.frames
            and bool(ball_image.valid[frame])
            and np.isfinite(ball_image.u[frame])
        )
        if image_here:
            coverage["image_available_frames"] += 1
        if not players.any():
            entry["uncontested"] += 1
            continue
        if image_here:
            distance = foot_distance(
                detections.box[rows], float(ball_image.u[frame]), float(ball_image.v[frame])
            )
            gate = POSSESSION_IMAGE_RADIUS
        else:
            distance = np.linalg.norm(detections.xy[rows] - aim[None, :], axis=1)
            gate = possession_radius_m
        distance = np.where(players & np.isfinite(distance), distance, np.inf)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= gate:
            entry["won"][int(teams[nearest])] += 1
            coverage["image_contested_frames" if image_here else "aim_contested_frames"] += 1
        else:
            entry["uncontested"] += 1
    out = {}
    for minute, entry in buckets.items():
        total = entry["won"][0] + entry["won"][1]
        if total == 0:
            continue
        share = entry["won"][0] / total
        out[minute] = {
            "team_0": round(share, 3),
            "team_1": round(1.0 - share, 3),
            "action_x": round(float(np.mean(entry["x"])), 1),
            "contested_frames": int(total),
            "uncontested_frames": int(entry["uncontested"]),
        }
    return dict(sorted(out.items())), coverage


def _possession_from_momentum(momentum: dict) -> float:
    """Team 0's share of contested frames across the whole match.

    The per-minute buckets are the right resolution for the momentum curve, but averaging their shares would
    give a minute with one contested frame the same weight as a minute with hundreds. Match possession is time
    in control, so the buckets are combined by their contested-frame counts - the convention a broadcast
    possession split uses. Returns 0.0 when nothing was contested.
    """
    contested = sum(int(bucket.get("contested_frames", 0)) for bucket in momentum.values())
    if contested <= 0:
        return 0.0
    won = sum(
        float(bucket.get("team_0", 0.0)) * int(bucket.get("contested_frames", 0))
        for bucket in momentum.values()
    )
    return won / contested


def _weak_labels(teams: dict[int, int], quality: dict[int, float]) -> list[int]:
    """Tracks whose team label may be wrong: labeled, but with little confidence behind the kit reading.

    Only labeled tracks count. The note this feeds exists to flag guesses, and counting every short fragment
    whose label is ``-1`` (which the old rule did - measured on the real game it fired for 13,257 tracks in a
    match of 9,164) told the reader nothing about the labels that are actually shown.
    """
    return [tid for tid, label in teams.items() if label >= 0 and quality.get(tid, 0.0) < 0.35]


def build_report(
    detections: PitchDetections,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    match_frames: int | None = None,
    ball_image: BallImage | None = None,
    on_progress=None,
) -> tuple[MatchReport, TrackAssignment]:
    """Full Stage B pipeline: mask -> track -> stitch fragments -> team -> metrics -> momentum.

    ``ball_image`` is the optional per-frame ball position from the ball scan (``ball_image_from_scan``) or the
    gimbal lock. Where it has a position, possession is decided in the picture; elsewhere it falls back to the
    camera aim point in pitch space. ``on_progress(fraction)`` covers the whole pipeline (tracking is the long
    part; teams and metrics are quick).
    """
    def report(fraction: float) -> None:
        if on_progress is not None:
            on_progress(min(1.0, max(0.0, fraction)))

    keep = on_pitch_mask(detections, pitch_length_m, pitch_width_m)
    assignment = _stitch_tracks(detections, _track_people(detections, keep, on_progress=lambda f: report(0.55 * f)))
    report(0.6)
    teams, quality, kit_colors = _team_assignment(detections, assignment)
    assignment.team = teams
    assignment.kit_quality = quality
    players = build_tracks(detections, keep, assignment, teams)
    report(0.7)

    momentum, possession_coverage = _momentum(
        detections, keep, teams, assignment.track_id, ball_image=ball_image,
        on_progress=lambda f: report(0.7 + 0.25 * f),
    )
    summary: list[TeamMetrics] = []
    mean_share = _possession_from_momentum(momentum)
    for team in (0, 1):
        members = [p for p in players if p.team == team]
        if members:
            distance = float(sum(p.distance_m for p in members))
            speeds = np.concatenate([p.speed_kmh for p in members])
            moving = speeds[speeds > 0.5]
            # Mean pitch x per team, normalized by pitch length: a direct "where does this team play" measure.
            mean_x = float(np.mean(np.concatenate([p.xy[:, 0] for p in members])) / pitch_length_m)
        else:
            distance = mean_x = 0.0
            moving = np.zeros(1)
        possession = mean_share if team == 0 else (1.0 - mean_share if momentum else 0.0)
        summary.append(
            TeamMetrics(
                team=team,
                players_observed=len(members),
                distance_m=round(distance, 1),
                top_speed_kmh=round(float(moving.max()) if len(moving) else 0.0, 1),
                mean_speed_kmh=round(float(moving.mean()) if len(moving) else 0.0, 1),
                mean_x_fraction=round(mean_x, 3),
                possession_share=round(possession, 3),
                kit_rgb=kit_rgb(kit_colors.get(team)),
            )
        )
    image_contested = int(possession_coverage["image_contested_frames"])
    aim_contested = int(possession_coverage["aim_contested_frames"])
    contested = image_contested + aim_contested
    image_share = image_contested / contested if contested else 0.0
    source = ball_image.source if ball_image is not None else "camera aim"
    possession_meta = {
        "space": "mixed" if image_contested and aim_contested else ("image" if image_contested else "pitch"),
        "source": source,
        "image_available_frames": int(possession_coverage["image_available_frames"]),
        "image_contested_frames": image_contested,
        "aim_contested_frames": aim_contested,
        "contested_frames": contested,
        "image_share": round(image_share, 3),
    }
    notes = []
    if not any(p.team >= 0 for p in players):
        notes.append("Kit colors were not separable: team metrics are unavailable for this match.")
    weak = _weak_labels(teams, quality)
    if weak:
        notes.append(f"{len(weak)} track(s) had weakly separated kit colors; their team label is a best guess.")
    if image_contested and aim_contested:
        notes.append(
            f"Possession is measured in image space from the {source} on {image_share:.0%} of contested frames; "
            "the rest use the camera aim point as a pitch-space ball proxy."
        )
    elif image_contested:
        notes.append(
            "Possession is measured in image space - the nearest player's foot to the ball in the picture - so it "
            "does not depend on the pitch calibration."
        )
    elif ball_image is not None:
        notes.append(
            "A ball scan was provided but did not produce a contested image-space frame; possession fell back to "
            "the camera aim point."
        )
    else:
        notes.append(
            "Possession uses the camera aim point as a ball proxy; run the ball scan to measure it in image space."
        )
    notes.append(
        "Goals, shots, corners, penalties, clearances and tackles can be inferred from the ball scan and these "
        "tracks (the playback's own Detect events button, or the one-press build in Step 3); saves and blocks stay "
        "manual tags, because a keeper's save and a shot wide look the same to a ball track."
    )
    notes.append("Only part of the pitch is in view at once, so no full-pitch formation is reported.")
    report(1.0)
    return (
        MatchReport(
            teams=summary,
            players=players,
            momentum=momentum,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            frames_analyzed=int(match_frames or (int(detections.frame.max()) + 1 if len(detections.frame) else 0)),
            detections_used=int(keep.sum()),
            possession=possession_meta,
            notes=notes,
        ),
        assignment,
    )

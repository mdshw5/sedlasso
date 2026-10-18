"""Infer match events from the ball scan and the player tracks.

The whistle scan (``analysis.events``) finds stoppages; the ball scan (``analysis.ball``) follows the ball. This
module is where the two, plus the player tracks from Stage B, are read together to say *what happened*: a penalty
awarded, a goal scored, a corner won, a shot that did not go in, a clearance, a tackle.

Everything here is pure - arrays and tracks in, events out - so the logic is unit-tested against the simulated
match, where the ball's true path is known, rather than against a video file. The detectors are deliberately
conservative and every one of them says in its note what it saw, because a wrong event on the timeline is worse
than a missing one: the whole point of the last stretch of work was that a measurement of nothing poisons
everything downstream.

What each detector keys on, and why:

* **goal** - the ball reaches a goal mouth moving toward the line, and within a short window afterwards the ball is
  static near the center spot. The reset is what separates a goal from a shot that hit the side netting: the ball
  is put back on the center spot only after a goal.
* **shot** - the ball is kicked hard toward a goal from within range and does *not* produce a goal reset. A shot on
  target that the keeper saves and a shot wide look the same to a ball track, so both are reported as a shot and
  the note says where it was aimed.
* **corner** - the ball is static near a corner flag and is then kicked, or the ball appears (a fresh detection)
  entering from a corner. The static-then-kick form is the common one; the appearance form catches a corner the
  scan only picks up once it is in flight.
* **penalty** - a whistle, then the ball static at the penalty spot for a moment, then a hard kick. The whistle is
  what makes it a penalty rather than a free kick from a similar spot; without a whistle the same geometry is
  reported as a shot.
* **clearance** - the ball is in a team's own defensive third and is kicked hard *away* from the goal that team is
  defending. Which goal that is comes from the per-half orientation below.
* **tackle** - a player who was moving comes to a near stop right beside the ball while the ball's own velocity
  changes. This is a *motion* proxy, not pose: the footage has no skeleton, so "went to ground" cannot be seen
  directly and the note says the event is a challenge inferred from movement, not a confirmed tackle.

The per-half orientation (which end each team defends, and so which way it attacks) is derived from where each
team's players spend the half: a team defends the goal its players are nearer to. It is what lets a clearance be
told from a shot - the same fast kick is one or the other depending on which goal it is heading away from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soccer_analytics.analysis.events import Event
from soccer_analytics.analysis.stage_b import PlayerTrack

# --- rebuilding tracks from the replay payload ----------------------------------------------------------------
def player_tracks_from_replay(replay: dict, fps: float, start_s: float = 0.0) -> list[PlayerTrack]:
    """Rebuild Stage B's player tracks from a saved replay payload.

    The replay already carries, per player, the frame indices, pitch positions and speeds the event detectors need;
    rebuilding the tracks from it avoids re-running the whole Stage B pipeline just to detect events. ``sigma_m`` is
    not stored (the detectors do not use it) and is filled with a nominal value.

    ``start_s`` is the analyzed window's own start offset in the source video - the kick-off offset chosen in Step 1.
    Frame indices are *within the analyzed window*, so a track's time is ``start_s + frame / fps``, not
    ``frame / fps``. Getting this wrong shifts every player-derived event by the offset: on the real game (kick-off
    at 9:00) a tackle at 11:11 was reported at 2:11, and the clip cut for it showed the wrong part of the match
    entirely.
    """
    tracks: list[PlayerTrack] = []
    for player in replay.get("players", []):
        frames = np.asarray(player["frames"], dtype=np.int64)
        xy = np.asarray(player["xy"], dtype=np.float64)
        speed = np.asarray(player["speed"], dtype=np.float64)
        tracks.append(
            PlayerTrack(
                track_id=int(player["track_id"]),
                team=int(player["team"]),
                frame=frames,
                time=float(start_s) + frames / max(fps, 1e-6),
                xy=xy,
                sigma_m=np.full(len(frames), 0.5),
                speed_kmh=speed,
                distance_m=float((player.get("stats") or {}).get("distance_m", 0.0)),
            )
        )
    return tracks

# --- ball motion ----------------------------------------------------------------------------------------------
# A ball at rest on this footage reads a little above zero because the scan's position wobbles by a pixel or two;
# a ball being passed reads 1-3 m/s and a shot 15-30 m/s. The gates sit in the gaps between those.
#
# The measurement is deliberately *not* a per-frame difference. Measured on the real whole-game scan, a per-frame
# difference of the projected positions has a p90 of 57 m/s and a maximum of 96,000 m/s: the projection amplifies a
# pixel of jitter into meters when the ball is far away or near the horizon, and a single bad frame then reads as a
# 200 m/s kick. Three things fix it, and each was measured rather than guessed:
#
# * positions off the pitch are dropped - the projection of a ball near the horizon lands kilometers away, and 18%
#   of the real scan's finite positions are outside the pitch by more than 3 m;
# * a position far from its neighbors' median is dropped as a spike (589 more frames on the real scan);
# * the speed is the *net displacement over a short window*, not a per-frame step, and it is paired with a
#   straightness ratio (net displacement / path length). Jitter is fast but not straight; a kicked ball is both.
#
# The window is causal - it looks back only - so a kick is not smeared across the frames before it. With those, the
# real scan's speeds sit at a p50 of 2.4 m/s and a p90 of 15 m/s, and the physically impossible readings are gone.
# The remaining cap is physics: no ball on this pitch travels faster than ~45 m/s.
STATIC_SPEED_MS = 4.0
KICK_SPEED_MS = 12.0  # a hard kick: a pass is slower, a shot faster
SHOT_SPEED_MS = 18.0
SHOT_TRAVEL_M = 12.0  # a shot travels: a hard kick that goes nowhere is a clearance off a shin, not a shot
CLEARANCE_SPEED_MS = 20.0  # a clearance is struck harder than a pass out of defense
CLEARANCE_TRAVEL_M = 20.0  # ... and it goes a long way: a short kick out of the third is a pass
CLEARANCE_HORIZON_S = 3.0  # the window the travel is measured over
MAX_BALL_SPEED_MS = 45.0  # faster than any struck ball: a measurement error, not motion
STATIC_MIN_S = 1.0  # a ball has to be still for this long to count as "at rest", not just slow for a frame
MAX_STEP_S = 1.0  # a gap longer than this is not a velocity: the ball was not seen in between
SPEED_WINDOW_S = 0.4  # the look-back window the net displacement is measured over (a couple of frames either rate)
MIN_STRAIGHTNESS = 0.7  # net displacement / path length below which the motion is jitter, not travel
SPIKE_FACTOR = 4.0  # how far from the local median a position may sit before it is a spike
PITCH_MARGIN_M = 3.0  # a ball a little outside the lines is real; kilometers away is a projection failure

# --- geometry ------------------------------------------------------------------------------------------------
GOAL_HALF_WIDTH_M = 3.66  # half a goal mouth
GOAL_MOUTH_MARGIN_M = 1.5  # a ball a little wide of the post still reads as "at the goal"
GOAL_DEPTH_M = 2.5  # how close to the goal line the ball has to be to count as reaching it
SHOT_RANGE_M = 35.0  # beyond this a fast ball is a long clearance, not a shot
CORNER_RADIUS_M = 7.0  # a ball this close to a corner flag is at the corner
PENALTY_SPOT_DIST_M = 11.0  # the penalty spot is 11 m from the goal line, on the center line
PENALTY_SPOT_RADIUS_M = 3.5
CENTER_RADIUS_M = 12.0  # the ball is "back at the center" within this of the center spot
RESET_WINDOW_S = 60.0  # a goal's center-spot reset has to follow within this
RESET_MIN_S = 2.0  # ... and the ball has to *stay* there: a restart, not a ball rolling through the middle
DEFENSIVE_THIRD_FRACTION = 0.35  # the third of the pitch nearest a team's own goal
ATTRIBUTION_RADIUS_M = 9.0  # a player this close to the ball is the one who played it
TACKLE_RADIUS_M = 2.0  # a challenge happens right beside the ball
TACKLE_STOP_SPEED_KMH = 2.0  # a player who was running and is now this slow has stopped
TACKLE_MOVE_SPEED_KMH = 12.0  # ... having been at least this fast a moment before (a run, not a jog)
TACKLE_WINDOW_S = 1.5
TACKLE_BALL_CHANGE_MS = 10.0  # the ball's own speed has to change by this much at the challenge
ATTRIBUTION_LAG_S = 0.4  # "a moment before" look-back/search tolerance (2 frames at the 5 fps tuning rate)
MIN_EVENT_GAP_S = 3.0  # two events of the same type closer than this are one event


@dataclass(frozen=True)
class BallMotion:
    """The ball scan reduced to what the detectors need: position, speed and direction per frame.

    ``speed``/``vx``/``vy`` are NaN wherever the ball was not seen on two consecutive frames close enough in time
    for a velocity to mean anything - a gap is not a measurement of motion, and inventing one is exactly the
    mistake this codebase keeps guarding against. ``straight`` is the ratio of net displacement to path length over
    the speed window: 1 is a ball traveling in a straight line, and a low value is jitter that happens to be fast.
    """

    xy: np.ndarray  # (F, 2) pitch meters, NaN where unknown
    measured: np.ndarray  # (F,) 1 where a detector saw the ball, 0 where the position is a forecast
    times: np.ndarray  # (F,) source seconds
    speed: np.ndarray  # (F,) m/s, NaN where unknown
    vx: np.ndarray  # (F,) m/s
    vy: np.ndarray  # (F,) m/s
    straight: np.ndarray  # (F,) net displacement / path length over the speed window, NaN where unknown

    @property
    def frames(self) -> int:
        return len(self.xy)


def _reject_spikes(xy: np.ndarray, width: int = 5, factor: float = SPIKE_FACTOR) -> np.ndarray:
    """Drop positions that sit far from their neighbors' median - a projection failure, not a ball.

    The threshold is relative to the track's own typical step, so it adapts to how noisy this scan is rather than
    assuming a scale. A ball genuinely moving fast is *consistent* with its neighbors (it is on a trajectory), so
    it survives; a single frame that lands somewhere else does not.
    """
    out = xy.copy()
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if finite.sum() < 3:
        return out
    steps = np.hypot(*np.diff(xy, axis=0).T)
    typical = float(np.nanmedian(steps[np.isfinite(steps)])) if np.isfinite(steps).any() else 1.0
    limit = factor * max(typical, 0.5) * 3.0
    half = width // 2
    for i in range(len(xy)):
        if not finite[i]:
            continue
        window = xy[max(0, i - half) : min(len(xy), i + half + 1)]
        window = window[np.isfinite(window[:, 0]) & np.isfinite(window[:, 1])]
        if len(window) < 3:
            continue
        median = np.median(window, axis=0)
        if float(np.hypot(*(xy[i] - median))) > limit:
            out[i] = np.nan
    return out


def ball_motion(
    xy: np.ndarray,
    measured: np.ndarray,
    times: np.ndarray,
    *,
    pitch: tuple[float, float] | None = None,
    window_s: float = SPEED_WINDOW_S,
    max_step_s: float = MAX_STEP_S,
) -> BallMotion:
    """Per-frame ball speed and velocity from the scan's positions, robust to projection jitter.

    The speed is the net displacement over a *look-back* window divided by the window's length, paired with the
    straightness of that displacement. A per-frame difference is not used: on the real scan it reads 57 m/s at p90
    and 96,000 m/s at worst, because the projection turns a pixel of jitter into meters when the ball is far away.
    Positions more than :data:`PITCH_MARGIN_M` outside the pitch are dropped first (``pitch`` is ``(length, width)``
    in meters; without it nothing is dropped), then positions far from their neighbors' median are dropped as
    spikes.

    The window looks *backwards* only, so a kick is not smeared across the frames before it: the frame the ball is
    struck reads fast immediately, which is what the detectors key on. A window that spans a gap in the track is
    skipped - the ball was not seen in between, so its motion there is unknown - and speeds above
    :data:`MAX_BALL_SPEED_MS` are NaN for the same reason: no ball travels that fast, so the reading is a
    measurement error rather than motion.
    """
    xy = np.asarray(xy, dtype=np.float64).copy()
    measured = np.asarray(measured, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    frames = len(xy)

    if pitch is not None:
        length_m, width_m = float(pitch[0]), float(pitch[1])
        outside = (
            (xy[:, 0] < -PITCH_MARGIN_M)
            | (xy[:, 0] > length_m + PITCH_MARGIN_M)
            | (xy[:, 1] < -PITCH_MARGIN_M)
            | (xy[:, 1] > width_m + PITCH_MARGIN_M)
        )
        xy[outside] = np.nan
    xy = _reject_spikes(xy)

    speed = np.full(frames, np.nan)
    vx = np.full(frames, np.nan)
    vy = np.full(frames, np.nan)
    straight = np.full(frames, np.nan)
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    for i in range(frames):
        if not finite[i]:
            continue
        back = i
        while back > 0 and times[i] - times[back] < window_s:
            back -= 1
        span = times[i] - times[back]
        if span < window_s * 0.6 or span > window_s + max_step_s:
            continue
        if not finite[back : i + 1].all():
            continue  # a gap inside the window: the ball was not seen, so its motion is unknown
        net = xy[i] - xy[back]
        distance = float(np.hypot(net[0], net[1]))
        path = 0.0
        for m in range(back, i):
            path += float(np.hypot(*(xy[m + 1] - xy[m])))
        value = distance / span
        if value > MAX_BALL_SPEED_MS:
            continue
        speed[i] = value
        vx[i] = net[0] / span
        vy[i] = net[1] / span
        straight[i] = distance / path if path > 1e-6 else 0.0
    return BallMotion(xy=xy, measured=measured, times=times, speed=speed, vx=vx, vy=vy, straight=straight)


@dataclass(frozen=True)
class HalfOrientation:
    """Which end each team defends in one half, and so which way it attacks.

    ``defending_goal`` is ``"left"`` (the x=0 goal) or ``"right"`` (the x=L goal) per team; ``attack_direction`` is
    +1 for a team attacking toward +x and -1 toward -x. ``mean_x`` is the mean pitch x of the team's players in the
    half, which is the measurement the sides are read from.
    """

    half: int
    defending_goal: dict[int, str]
    attack_direction: dict[int, int]
    mean_x: dict[int, float]

    def to_json(self) -> dict:
        return {
            "half": int(self.half),
            "defending_goal": {str(k): v for k, v in self.defending_goal.items()},
            "attack_direction": {str(k): int(v) for k, v in self.attack_direction.items()},
            "mean_x": {str(k): round(float(v), 1) for k, v in self.mean_x.items()},
        }


def team_orientations(
    players: list[PlayerTrack],
    pitch_length_m: float,
    *,
    half_bounds: tuple[float, float, float] | None = None,
    swap: bool = False,
) -> list[HalfOrientation]:
    """Which goal each team defends, per half, from where its players spend the half.

    A team defends the goal its players are nearer to: the mean pitch x of a team's observations in a half is the
    measurement, and the team with the smaller mean x defends the x=0 goal. This is the "player start positions and
    team attack vector" the timeline needs, and it is what tells a clearance from a shot.

    ``half_bounds`` is the game clock's ``(kick-off, half-time, full-time)`` in source seconds. Without it the whole
    match is treated as one period (half 0), which is right for a single-half segment and honest for a match whose
    halves were not marked - the sides are then the match's own, not a half's. ``swap`` inverts every measured
    side and direction: some footage makes the ends ambiguous, and the user can correct that without recalibrating.
    """
    if half_bounds is None:
        periods = [(0, -np.inf, np.inf)]
    else:
        start, half, end = half_bounds
        periods = [(1, start, half), (2, half, end)]

    out: list[HalfOrientation] = []
    for half_number, lo, hi in periods:
        sums: dict[int, list[float]] = {0: [], 1: []}
        for track in players:
            if track.team not in (0, 1):
                continue
            times = np.asarray(track.time, dtype=np.float64)
            mask = (times >= lo) & (times < hi)
            if not mask.any():
                continue
            sums[track.team].extend(np.asarray(track.xy, dtype=np.float64)[mask, 0].tolist())
        mean_x = {team: float(np.mean(values)) if values else float("nan") for team, values in sums.items()}
        defending: dict[int, str] = {}
        attack: dict[int, int] = {}
        if np.isfinite(mean_x[0]) and np.isfinite(mean_x[1]):
            # The team with the smaller mean x is nearer the x=0 goal, so it defends that end.
            left_team = 0 if mean_x[0] <= mean_x[1] else 1
            for team in (0, 1):
                defends_left = team == left_team
                defending[team] = "left" if defends_left else "right"
                attack[team] = 1 if defends_left else -1  # attacks the far end from the one it defends
        if swap and defending:
            defending = {team: ("right" if side == "left" else "left") for team, side in defending.items()}
            attack = {team: -direction for team, direction in attack.items()}
        out.append(HalfOrientation(half_number, defending, attack, mean_x))
    return out


def _orientation_for(orientations: list[HalfOrientation], time_s: float, half_bounds) -> HalfOrientation | None:
    """The orientation covering ``time_s``, or the single one when the match was not split into halves."""
    if not orientations:
        return None
    if half_bounds is None:
        return orientations[0]
    start, half, end = half_bounds
    wanted = 1 if time_s < half else 2
    for orientation in orientations:
        if orientation.half == wanted:
            return orientation
    return orientations[0]


# --- player attribution --------------------------------------------------------------------------------------
def _players_by_frame(players: list[PlayerTrack]) -> dict[int, list[tuple[int, int, float, float]]]:
    """Frame index -> ``(track_id, team, x, y)`` for every player observed on that frame."""
    by_frame: dict[int, list[tuple[int, int, float, float]]] = {}
    for track in players:
        for frame, (x, y) in zip(track.frame, track.xy):
            by_frame.setdefault(int(frame), []).append((int(track.track_id), int(track.team), float(x), float(y)))
    return by_frame


def _lag_frames(motion: BallMotion) -> int:
    """An "a moment before/around" tolerance in frames, scaled to the analysis rate.

    ``ATTRIBUTION_LAG_S`` was a couple of frames at the 5 fps the detectors were tuned at; at 15 fps the same
    wall-clock tolerance is six frames, which is the point - the ball and the player are still detected a few
    tenths of a second apart, not a couple of frames apart.
    """
    if motion.frames < 2:
        return 2
    dt = float(np.median(np.diff(motion.times)))
    rate = 1.0 / dt if dt > 1e-6 else 5.0
    return max(1, int(round(ATTRIBUTION_LAG_S * rate)))


def _nearest_player(
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    frame: int,
    xy: tuple[float, float],
    radius_m: float,
    *,
    search: int = 2,
) -> tuple[int, int, float] | None:
    """The player nearest ``xy`` around ``frame`` within ``radius_m``, or ``None``.

    A couple of frames either side are searched because the ball and the player are not always detected on the same
    frame; the nearest over that small window is the one who played the ball.
    """
    best: tuple[int, int, float] | None = None
    for offset in range(-search, search + 1):
        for track_id, team, x, y in by_frame.get(frame + offset, ()):
            distance = float(np.hypot(x - xy[0], y - xy[1]))
            if distance <= radius_m and (best is None or distance < best[2]):
                best = (track_id, team, distance)
    return best


def _number_for(numbers: dict[int, dict] | None, track_id: int | None) -> int | None:
    if numbers is None or track_id is None:
        return None
    entry = numbers.get(int(track_id)) or {}
    number = entry.get("number")
    return None if number is None else int(number)


# --- event detectors -----------------------------------------------------------------------------------------
def _is_kick(motion: BallMotion, frame: int, threshold: float) -> bool:
    """Whether the ball at ``frame`` is traveling fast *and* straight - a kick, not projection jitter.

    The straightness gate is what makes this usable on real footage: a jittering position is fast but wanders, so
    its net displacement over the speed window is a fraction of the path it took, while a struck ball goes where it
    was sent. Measured on the real scan, real kicks sit at 0.7-1.0 and the jitter that survives the speed gate sits
    below 0.6.
    """
    if not np.isfinite(motion.speed[frame]) or motion.speed[frame] < threshold:
        return False
    return bool(np.isfinite(motion.straight[frame]) and motion.straight[frame] >= MIN_STRAIGHTNESS)


def _static_runs(motion: BallMotion, min_s: float) -> list[tuple[int, int]]:
    """Runs of frames where the ball is at rest, as ``(start, end)`` frame indices (end exclusive).

    A frame counts as at rest when its speed is below :data:`STATIC_SPEED_MS`; a frame with no speed (a gap) breaks
    the run, because "not seen" is not "still". A run shorter than ``min_s`` is dropped.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(motion.frames):
        still = np.isfinite(motion.speed[i]) and motion.speed[i] < STATIC_SPEED_MS
        if still and start is None:
            start = i
        elif not still and start is not None:
            if motion.times[i - 1] - motion.times[start] >= min_s:
                runs.append((start, i))
            start = None
    if start is not None and motion.times[-1] - motion.times[start] >= min_s:
        runs.append((start, motion.frames))
    return runs


def _goal_mouth_hit(motion: BallMotion, pitch_length_m: float, pitch_width_m: float) -> list[tuple[int, int]]:
    """Frames where the ball is at a goal mouth moving toward the line, as ``(frame, side)``.

    ``side`` is 0 for the x=0 goal and 1 for the x=L goal. The ball has to be within :data:`GOAL_DEPTH_M` of the
    line, inside the mouth (with a margin), and moving toward that line - a ball rolling *away* from the goal at the
    same spot is a goal kick, not a goal.

    A *forecast* position is allowed here, and that is deliberate: the scan loses the ball as it crosses the line
    (the camera is still catching up, and the ball is against the net), so demanding a detection at the line would
    miss every goal. What is not allowed to be a forecast is the center-spot reset that confirms it - see
    :func:`_center_reset_after`. The pair is what makes this honest: a forecast can suggest a goal, only a
    measurement can confirm one.
    """
    hits: list[tuple[int, int]] = []
    center_y = pitch_width_m / 2.0
    for i in range(motion.frames):
        if not np.isfinite(motion.xy[i, 0]) or not np.isfinite(motion.vx[i]):
            continue
        if abs(motion.xy[i, 1] - center_y) > GOAL_HALF_WIDTH_M + GOAL_MOUTH_MARGIN_M:
            continue
        x = motion.xy[i, 0]
        if x <= GOAL_DEPTH_M and motion.vx[i] < -KICK_SPEED_MS:
            hits.append((i, 0))
        elif x >= pitch_length_m - GOAL_DEPTH_M and motion.vx[i] > KICK_SPEED_MS:
            hits.append((i, 1))
    return hits


def _center_reset_after(motion: BallMotion, pitch_length_m: float, pitch_width_m: float, after_frame: int) -> int | None:
    """The first frame after ``after_frame`` where the ball is *seen* static near the center spot, or ``None``.

    This is the goal's signature: the ball is put back on the center spot only after a goal. The window is bounded
    by :data:`RESET_WINDOW_S` so a later stoppage is not mistaken for this goal's reset.

    The position has to be a *measurement* (``measured`` is 1), not the tracker's forecast across a missed frame.
    That distinction is the whole point of the ball scan's own honesty rule, and it matters here: after a goal the
    scan loses the ball during the celebration and its forecast can land near the center spot while the ball is
    actually still in the net. A reset nobody saw is not a reset.

    The ball also has to *stay* there for :data:`RESET_MIN_S`: a ball rolling through the middle of the pitch on its
    way somewhere else is not a restart. Measured on the real game, the celebration costs the scan the ball for a
    few seconds and the restart is only picked up ~50 s after the goal, which is why the window is a minute wide.
    """
    center = np.array([pitch_length_m / 2.0, pitch_width_m / 2.0])
    deadline = motion.times[after_frame] + RESET_WINDOW_S
    for i in range(after_frame, motion.frames):
        if motion.times[i] > deadline:
            break
        if not motion.measured[i] or not np.isfinite(motion.xy[i, 0]):
            continue
        if np.linalg.norm(motion.xy[i] - center) > CENTER_RADIUS_M:
            continue
        if not (np.isfinite(motion.speed[i]) and motion.speed[i] < STATIC_SPEED_MS):
            continue
        # It has to stay still there: walk forward while the ball remains measured, near the center and at rest.
        end = i
        while end + 1 < motion.frames and motion.times[end + 1] - motion.times[i] <= RESET_MIN_S + 1.0:
            end += 1
            if not motion.measured[end] or not np.isfinite(motion.xy[end, 0]):
                break
            if np.linalg.norm(motion.xy[end] - center) > CENTER_RADIUS_M:
                break
            if not (np.isfinite(motion.speed[end]) and motion.speed[end] < STATIC_SPEED_MS):
                break
        if motion.times[end] - motion.times[i] >= RESET_MIN_S:
            return i
    return None


def _dedupe(events: list[Event], gap_s: float = MIN_EVENT_GAP_S) -> list[Event]:
    """Keep the most confident event of each type within ``gap_s`` of another of the same type."""
    kept: list[Event] = []
    for event in sorted(events, key=lambda e: (e.type, e.time_s)):
        clash = next(
            (other for other in kept if other.type == event.type and abs(other.time_s - event.time_s) < gap_s),
            None,
        )
        if clash is None:
            kept.append(event)
        elif event.confidence > clash.confidence:
            kept[kept.index(clash)] = event
    return sorted(kept, key=lambda e: e.time_s)


def detect_events(
    ball_xy: np.ndarray,
    ball_measured: np.ndarray,
    times: np.ndarray,
    players: list[PlayerTrack],
    pitch: tuple[float, float],
    *,
    whistles: list[float] | None = None,
    numbers: dict[int, dict] | None = None,
    half_bounds: tuple[float, float, float] | None = None,
    swap_attack: bool = False,
    video: str = "",
) -> list[Event]:
    """Infer goals, shots, corners, penalties, clearances and tackles from the ball and player tracks.

    ``ball_xy``/``ball_measured`` are the ``(F, 2)`` and ``(F,)`` arrays ``projection.project_ball_track`` returns;
    ``times`` is the source seconds of each frame. ``players`` are Stage B's tracks, ``pitch`` is ``(length, width)``
    in meters, ``whistles`` are the audio scan's candidate times (used only to tell a penalty from a free kick),
    ``numbers`` maps a track id to its shirt number/name, ``half_bounds`` is the game clock's
    ``(kick-off, half-time, full-time)``, and ``swap_attack`` inverts the measured ends (the match setting).

    Returns events with ``source="ball"``, sorted by time. Every event carries a note saying what was measured, and
    the player it is attributed to when one is near enough to the ball to be named.
    """
    length_m, width_m = float(pitch[0]), float(pitch[1])
    motion = ball_motion(ball_xy, ball_measured, times, pitch=(length_m, width_m))
    by_frame = _players_by_frame(players)
    orientations = team_orientations(players, length_m, half_bounds=half_bounds, swap=swap_attack)
    whistle_times = sorted(float(w) for w in (whistles or []))

    events: list[Event] = []
    events += _goals(motion, length_m, width_m, by_frame, numbers, video)
    # Penalties are found before shots so a penalty's own kick is not also reported as a shot: the two are the same
    # fast ball toward the goal, and the penalty is the more specific reading.
    events += _penalties(motion, length_m, width_m, by_frame, numbers, video, whistle_times)
    events += _shots(motion, length_m, width_m, by_frame, numbers, video, events)
    events += _corners(motion, length_m, width_m, by_frame, numbers, video)
    events += _clearances(motion, length_m, width_m, by_frame, numbers, video, orientations, half_bounds)
    events += _tackles(motion, players, by_frame, numbers, video)
    return _dedupe(events)


def _goals(
    motion: BallMotion,
    length_m: float,
    width_m: float,
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
) -> list[Event]:
    """A goal: the ball reaches a goal mouth moving in, and the ball is reset to the center spot soon after."""
    out: list[Event] = []
    for frame, side in _goal_mouth_hit(motion, length_m, width_m):
        reset = _center_reset_after(motion, length_m, width_m, frame)
        if reset is None:
            continue
        # The scorer is the player nearest the ball a moment *before* it crossed the line (0.4 s back, scaled to
        # the analysis rate), searched either side of that moment.
        lag = _lag_frames(motion)
        scorer = _nearest_player(
            by_frame, max(0, frame - lag), tuple(motion.xy[frame]), ATTRIBUTION_RADIUS_M, search=lag
        )
        track_id = scorer[0] if scorer else None
        team = scorer[1] if scorer else -1
        out.append(
            Event(
                time_s=float(motion.times[frame]),
                type="goal",
                team=team,
                note=(
                    f"ball crossed the {'left' if side == 0 else 'right'} goal line at "
                    f"{motion.speed[frame]:.0f} m/s and was reset to the center spot "
                    f"{motion.times[reset] - motion.times[frame]:.0f}s later"
                ),
                source="ball",
                confidence=0.8,
                video=video,
                player_track=track_id,
                player_number=_number_for(numbers, track_id),
            )
        )
    return out


def _shots(
    motion: BallMotion,
    length_m: float,
    width_m: float,
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
    already: list[Event],
) -> list[Event]:
    """A shot: a hard kick toward a goal from within range that did not become a goal.

    A shot on target that the keeper saves and a shot wide look the same to a ball track, so both are reported and
    the note says which goal it was aimed at. Frames that produced a goal or a penalty are skipped - those are the
    more specific events.
    """
    goal_frames = {round(e.time_s, 3) for e in already if e.type in ("goal", "penalty")}
    out: list[Event] = []
    center_y = width_m / 2.0
    for i in range(motion.frames):
        if not _is_kick(motion, i, SHOT_SPEED_MS):
            continue
        # Where the ball was struck, not where its speed peaked: the shot has already left the spot by then.
        origin = _kick_origin(motion, i)
        x, y = motion.xy[origin]
        # Which goal is it heading for? The sign of vx picks the end; the ball has to be within range of it.
        side = 0 if motion.vx[i] < 0 else 1
        goal_x = 0.0 if side == 0 else length_m
        distance = abs(goal_x - x)
        if distance > SHOT_RANGE_M:
            continue
        # The velocity has to point at the goal mouth, not just down the pitch: a ball crossed sideways at speed is
        # not a shot. The aim point is where the ball's velocity line meets the goal line.
        if abs(motion.vx[i]) < 1e-6:
            continue
        t_to_line = (goal_x - x) / motion.vx[i]
        if t_to_line <= 0:
            continue
        aim_y = y + motion.vy[i] * t_to_line
        if abs(aim_y - center_y) > GOAL_HALF_WIDTH_M + GOAL_MOUTH_MARGIN_M:
            continue
        if any(abs(motion.times[i] - t) < MIN_EVENT_GAP_S for t in goal_frames):
            continue
        # The ball has to actually go toward the goal, not merely be pointed at it: a shot travels a long way in
        # the goal's direction. Without this, every hard kick from anywhere on the pitch that happened to be angled
        # at a goal was reported (117 of them in one match, against a handful of real shots). The travel is measured
        # from where the ball was struck, not from the frame the speed peaked - by then it has already moved.
        travel = _travel(motion, origin)
        if not np.isfinite(travel) or travel < SHOT_TRAVEL_M:
            continue
        end = origin
        while end < motion.frames - 1 and motion.times[end] - motion.times[origin] < CLEARANCE_HORIZON_S:
            end += 1
        if np.isfinite(motion.xy[end, 0]):
            # Closer to the goal than it started, by distance - the sign of the difference is not the question.
            if abs(goal_x - motion.xy[end, 0]) >= abs(goal_x - x):
                continue
        lag = _lag_frames(motion)
        shooter = _nearest_player(by_frame, max(0, origin - lag), (x, y), ATTRIBUTION_RADIUS_M, search=lag)
        track_id = shooter[0] if shooter else None
        out.append(
            Event(
                time_s=float(motion.times[i]),
                type="shot",
                team=shooter[1] if shooter else -1,
                note=(
                    f"ball kicked at {motion.speed[i]:.0f} m/s toward the "
                    f"{'left' if side == 0 else 'right'} goal from {distance:.0f} m"
                ),
                source="ball",
                confidence=0.6,
                video=video,
                player_track=track_id,
                player_number=_number_for(numbers, track_id),
            )
        )
    return out


def _corners(
    motion: BallMotion,
    length_m: float,
    width_m: float,
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
) -> list[Event]:
    """A corner: the ball is at rest near a corner flag and is then kicked, or appears entering from a corner.

    The static-then-kick form is the common one. The appearance form catches a corner the scan only picks up once
    the ball is already in flight: a fresh detection (a gap before it) near a corner moving away from it.
    """
    corners = np.array([[0.0, 0.0], [0.0, width_m], [length_m, 0.0], [length_m, width_m]])
    out: list[Event] = []
    for start, end in _static_runs(motion, STATIC_MIN_S):
        position = motion.xy[start]
        if not np.isfinite(position[0]):
            continue
        nearest = int(np.argmin(np.linalg.norm(corners - position[None, :], axis=1)))
        if np.linalg.norm(corners[nearest] - position) > CORNER_RADIUS_M:
            continue
        # A kick out of the corner: a fast frame within a couple of seconds after the ball was still.
        kick = _first_fast_after(motion, end, KICK_SPEED_MS, window_s=3.0)
        if kick is None:
            continue
        lag = _lag_frames(motion)
        taker = _nearest_player(by_frame, max(0, kick - lag), tuple(motion.xy[kick]), ATTRIBUTION_RADIUS_M, search=lag)
        track_id = taker[0] if taker else None
        out.append(
            Event(
                time_s=float(motion.times[kick]),
                type="corner",
                team=taker[1] if taker else -1,
                note=(
                    f"ball was still at the corner ({position[0]:.0f}, {position[1]:.0f}) then kicked at "
                    f"{motion.speed[kick]:.0f} m/s"
                ),
                source="ball",
                confidence=0.6,
                video=video,
                player_track=track_id,
                player_number=_number_for(numbers, track_id),
            )
        )
    return out


def _penalties(
    motion: BallMotion,
    length_m: float,
    width_m: float,
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
    whistle_times: list[float],
) -> list[Event]:
    """A penalty: a whistle, then the ball still at the penalty spot, then a hard kick.

    The whistle is what makes it a penalty rather than a free kick from a similar spot; without one the same
    geometry is left to the shot detector. The spot is 11 m from each goal line on the center line.
    """
    spots = np.array([[PENALTY_SPOT_DIST_M, width_m / 2.0], [length_m - PENALTY_SPOT_DIST_M, width_m / 2.0]])
    out: list[Event] = []
    for start, end in _static_runs(motion, STATIC_MIN_S):
        position = motion.xy[start]
        if not np.isfinite(position[0]):
            continue
        nearest = int(np.argmin(np.linalg.norm(spots - position[None, :], axis=1)))
        if np.linalg.norm(spots[nearest] - position) > PENALTY_SPOT_RADIUS_M:
            continue
        kick = _first_fast_after(motion, end, KICK_SPEED_MS, window_s=5.0)
        if kick is None:
            continue
        # A whistle in the seconds before the ball settled is what says this is a penalty, not open play.
        whistle = next(
            (w for w in whistle_times if motion.times[start] - 20.0 <= w <= motion.times[start] + 2.0), None
        )
        if whistle is None:
            continue
        lag = _lag_frames(motion)
        taker = _nearest_player(by_frame, max(0, kick - lag), tuple(motion.xy[kick]), ATTRIBUTION_RADIUS_M, search=lag)
        track_id = taker[0] if taker else None
        out.append(
            Event(
                time_s=float(motion.times[kick]),
                type="penalty",
                team=taker[1] if taker else -1,
                note=(
                    f"whistle at {whistle:.1f}s, ball still on the penalty spot, then kicked at "
                    f"{motion.speed[kick]:.0f} m/s"
                ),
                source="ball",
                confidence=0.7,
                video=video,
                player_track=track_id,
                player_number=_number_for(numbers, track_id),
            )
        )
    return out


def _travel(motion: BallMotion, frame: int, horizon_s: float = CLEARANCE_HORIZON_S) -> float:
    """How far the ball moves from ``frame`` over the next ``horizon_s``, or NaN when either end is unknown.

    This is what separates a clearance from a pass out of defense: both are kicks away from the own goal, but a
    clearance is struck to go a long way. Measured on the real scan, kicks travel a median of 14 m over three
    seconds and a clearance sits in the top quarter of that.
    """
    end = frame
    while end < motion.frames - 1 and motion.times[end] - motion.times[frame] < horizon_s:
        end += 1
    if not (np.isfinite(motion.xy[frame, 0]) and np.isfinite(motion.xy[end, 0])):
        return float("nan")
    return float(np.hypot(*(motion.xy[end] - motion.xy[frame])))


def _kick_origin(motion: BallMotion, frame: int, *, look_back_s: float = 1.5) -> int:
    """The frame the kick at ``frame`` started from: the last frame before it where the ball was still.

    A kick's *speed* peaks a frame or two after the ball is struck, by which time it has already left the spot it
    was played from. Anything that asks "where was this played from" - which third, which goal it is leaving - has
    to ask about the origin, not the peak. Falls back to the earliest frame in the look-back window when the ball
    was never still (a kick out of a scramble).
    """
    earliest = frame
    while earliest > 0 and motion.times[frame] - motion.times[earliest] < look_back_s:
        earliest -= 1
    for i in range(frame, earliest, -1):
        if np.isfinite(motion.speed[i]) and motion.speed[i] < STATIC_SPEED_MS:
            return i
    return earliest


def _clearances(
    motion: BallMotion,
    length_m: float,
    width_m: float,
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
    orientations: list[HalfOrientation],
    half_bounds,
) -> list[Event]:
    """A clearance: a hard, long kick away from the goal the team is defending, from its own defensive third.

    Which goal is "own" comes from the per-half orientation. Without an orientation (no team could be placed) no
    clearance is reported - the same fast kick is a clearance or a shot depending on which goal it leaves, and
    guessing would be worse than staying quiet. The kick also has to *travel*: a short kick out of the third is a
    pass, and reporting every one of those buried the real clearances (60 of them in one match, against a handful
    of real ones).
    """
    out: list[Event] = []
    for i in range(motion.frames):
        if not _is_kick(motion, i, CLEARANCE_SPEED_MS):
            continue
        # Where the ball was played from, not where its speed peaked: the kick has already left the spot by then.
        origin = _kick_origin(motion, i)
        x = motion.xy[origin, 0]
        orientation = _orientation_for(orientations, float(motion.times[origin]), half_bounds)
        if orientation is None or not orientation.defending_goal:
            continue
        # The player nearest the ball just before the kick decides whose clearance it is (0.4 s back, scaled to
        # the analysis rate).
        lag = _lag_frames(motion)
        player = _nearest_player(
            by_frame, max(0, origin - lag), tuple(motion.xy[origin]), ATTRIBUTION_RADIUS_M, search=lag
        )
        if player is None:
            continue
        team = player[1]
        if team not in orientation.defending_goal:
            continue
        own_goal_x = 0.0 if orientation.defending_goal[team] == "left" else length_m
        # In the team's own defensive third?
        if abs(x - own_goal_x) > DEFENSIVE_THIRD_FRACTION * length_m:
            continue
        # Moving away from the own goal, and fast.
        away = (x - own_goal_x) * motion.vx[i] > 0
        if not away:
            continue
        travel = _travel(motion, i)
        if not np.isfinite(travel) or travel < CLEARANCE_TRAVEL_M:
            continue
        out.append(
            Event(
                time_s=float(motion.times[i]),
                type="clearance",
                team=team,
                note=(
                    f"ball kicked at {motion.speed[i]:.0f} m/s away from the "
                    f"{orientation.defending_goal[team]} goal from {abs(x - own_goal_x):.0f} m out, "
                    f"traveling {travel:.0f} m"
                ),
                source="ball",
                confidence=0.5,
                video=video,
                player_track=player[0],
                player_number=_number_for(numbers, player[0]),
            )
        )
    return out


def _tackles(
    motion: BallMotion,
    players: list[PlayerTrack],
    by_frame: dict[int, list[tuple[int, int, float, float]]],
    numbers: dict[int, dict] | None,
    video: str,
) -> list[Event]:
    """A tackle: a player who was moving comes to a near stop right beside the ball as the ball's motion changes.

    This is a *motion* proxy, not pose. The footage has no skeleton, so "went to ground" cannot be seen directly;
    what can be seen is a player arriving at speed, stopping within a couple of meters of the ball, and the ball's
    own velocity changing at that moment. The note says so, so the event is read as a challenge to review rather
    than a confirmed tackle.
    """
    out: list[Event] = []
    for track in players:
        if track.team not in (0, 1):
            continue
        speed = np.asarray(track.speed_kmh, dtype=np.float64)
        frames = np.asarray(track.frame, dtype=np.int64)
        times = np.asarray(track.time, dtype=np.float64)
        for j in range(1, len(frames)):
            if speed[j] > TACKLE_STOP_SPEED_KMH or speed[j - 1] < TACKLE_MOVE_SPEED_KMH:
                continue
            if times[j] - times[j - 1] > TACKLE_WINDOW_S:
                continue
            frame = int(frames[j])
            if not np.isfinite(motion.xy[frame, 0]):
                continue
            distance = float(np.hypot(track.xy[j, 0] - motion.xy[frame, 0], track.xy[j, 1] - motion.xy[frame, 1]))
            if distance > TACKLE_RADIUS_M:
                continue
            # The ball's own motion has to change around the challenge: a player stopping beside a still ball is
            # just standing, not tackling. The change is measured over about 0.4 s either side (scaled to the
            # analysis rate), because the ball and the player are not always detected on the same frame.
            lag = _lag_frames(motion)
            before = motion.speed[max(0, frame - lag)]
            after = motion.speed[min(motion.frames - 1, frame + lag)]
            changed = (
                np.isfinite(before)
                and np.isfinite(after)
                and abs(after - before) > TACKLE_BALL_CHANGE_MS
            )
            if not changed:
                continue
            out.append(
                Event(
                    time_s=float(times[j]),
                    type="tackle",
                    team=int(track.team),
                    note=(
                        f"player stopped from {speed[j - 1]:.0f} km/h within {distance:.1f} m of the ball as its "
                        "speed changed - a challenge inferred from movement, not pose"
                    ),
                    source="ball",
                    confidence=0.4,
                    video=video,
                    player_track=int(track.track_id),
                    player_number=_number_for(numbers, int(track.track_id)),
                )
            )
    return out


def _first_fast_after(motion: BallMotion, frame: int, threshold: float, *, window_s: float) -> int | None:
    """The first frame after ``frame`` where the ball is kicked (fast and straight), within ``window_s``."""
    deadline = motion.times[min(frame, motion.frames - 1)] + window_s
    for i in range(frame, motion.frames):
        if motion.times[i] > deadline:
            return None
        if _is_kick(motion, i, threshold):
            return i
    return None


def events_to_json(events: list[Event]) -> list[dict]:
    """Serialize detected events for the replay payload (the timeline draws them)."""
    return [event.to_json() for event in events]

"""Automatic pitch-line observations: the re-anchoring the clicks cannot provide.

The landmark clicks are exact ground control, but there are only eight of them, at two moments nine minutes
apart - so the drift correction between and after them is pure interpolation, and the chain's focal (searched
per step, pinned near the default for every zoom step the operator made) is only ever pinned at two zoom levels.
The pitch markings themselves are the answer: they are visible in almost every frame, they are the same ground
truth the clicks encode, and a line crossing can be found automatically.

What a line gives, and what it does not
---------------------------------------
A detected line crossing says "this marking passes through this pixel, running this way" - one number, the
offset *across* the line. It cannot say which point *along* the line the pixel is, so the observation is
direction-constrained: the residual is the perpendicular offset only. That is strictly weaker than a click
(two numbers) but it is available at dozens of frames instead of two, which is exactly what the drift fit
needs: more knots, spread over the game, each pinning the pointing and the focal where the pitch is actually
seen rather than interpolated.

The detection is deliberately conservative, because a wrong observation is worse than a missing one - a knot
fitted to a misdetected line can absorb almost any error. A candidate is kept only when a white stroke
(brightness well above the local pitch) sits within a few pixels of the predicted line, running the predicted
way, with enough contrast to be a marking and not a shadow edge or a player's sock. Frames where the camera
is panning fast are skipped: motion blur smears the strokes and the prediction itself is least trustworthy
exactly when the pan is fastest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_analytics.geometry.pitch_calibration import PitchCalibration, pitch_to_pixels

# Sampling: one frame every LINE_STRIDE frames, skipping spans where the camera pans faster than this many
# degrees per frame (motion blur + least-trustworthy prediction). At the analysis rate this is a sample every
# few seconds - dozens of anchors over a game, which is what the drift fit wants. The stride is also a fit-cost
# budget: the drift solve is superlinear in the anchor count (400 anchors ~100 s, 100 anchors ~6 s) and measured
# recovery is the same at 100 knots as at 400 (0.88 vs 0.88 deg against a 0.99 deg truth), so 100-ish anchors
# is the sweet spot.
LINE_STRIDE = 50
MAX_PAN_DEG_PER_FRAME = 0.5

# Search geometry, in normalized pixels (the convention of the clicks and the projection; v is normalized by
# the frame *width*, like every u/v in this codebase).
SEARCH_HALF_WIDTH = 0.012  # scan this far either side of the predicted line
MAX_PERP_ERROR = 0.006  # a detection further than this from the prediction is not the same line
OUTLIER_PERP = 0.05  # a fit-time residual beyond this is a misdetected line, not drift (see refine_with_lines)
MIN_STATIONS = 5  # perpendicular scans that must find the stroke
MIN_WHITE_FRACTION = 0.5  # fraction of stations that must find it

# White-stroke test, in 8-bit values: the line must be brighter than the local pitch by this much. The local
# pitch is the median of the outer band of the search window - the grass right next to the stroke.
WHITE_MARGIN = 28.0
CONTEXT_FRACTION = 0.35  # the outer this-fraction of the half-window on each side is "context"

# A marking seen end-on or nearly out of frame says nothing.
MIN_LINE_LENGTH_PX = 40.0


@dataclass(frozen=True)
class LineObservation:
    """One detected marking crossing: the pixel it was found at, the pitch point it corresponds to, and the
    line's image normal (the direction the residual is taken along)."""

    frame: int
    u: float
    v: float
    pitch_x: float
    pitch_y: float
    direction: tuple[float, float]
    label: str
    contrast: float


def pitch_marking_lines(length_m: float, width_m: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The straight markings as pitch-space segments: touchlines, halfway line, goal lines, penalty and goal boxes.

    The center circle and the arcs are left out - they curve, and the detector's straight-line scan does not
    apply. The straight markings are also the ones a partial view most often shows as long, unambiguous strokes.
    """
    L, W = length_m, width_m
    box_d, box_w = 16.5, 20.16
    goal_d, goal_w = 5.5, 9.16
    segments = [
        ((0.0, 0.0), (L, 0.0)),  # near touchline
        ((0.0, W), (L, W)),  # far touchline
        ((L / 2, 0.0), (L / 2, W)),  # halfway line
    ]
    for x0, s in ((0.0, 1.0), (L, -1.0)):
        segments.append(((x0, 0.0), (x0, W)))  # goal line
        segments.append(((x0, 32.0 - box_w), (x0 + s * box_d, 32.0 - box_w)))  # penalty box
        segments.append(((x0 + s * box_d, 32.0 - box_w), (x0 + s * box_d, 32.0 + box_w)))
        segments.append(((x0, 32.0 + box_w), (x0 + s * box_d, 32.0 + box_w)))
        segments.append(((x0, 32.0 - goal_w), (x0 + s * goal_d, 32.0 - goal_w)))  # goal box
        segments.append(((x0 + s * goal_d, 32.0 - goal_w), (x0 + s * goal_d, 32.0 + goal_w)))
        segments.append(((x0, 32.0 + goal_w), (x0 + s * goal_d, 32.0 + goal_w)))
    return segments


def _pan_rates(segment, frame_count: int) -> np.ndarray:
    """|dyaw| per frame from the gimbal log, or zeros when there is no log.

    The clips behind the segment (and their start seconds) come from the same discovery the pose alignment uses,
    so a combined game's second and third clips get their logs placed on the game clock here too - reading the
    combined video's own name as the clip would find only the first log.
    """
    from soccer_analytics.geometry.gimbal_motion import _clips_for_segment, align_log, find_logs_for_clips
    from soccer_analytics.geometry.gimbal_log import load_gimbal_log

    pan = np.zeros(frame_count)
    clip_paths, clip_starts = _clips_for_segment(segment)
    pairs = find_logs_for_clips(clip_paths)
    if not pairs:
        return pan
    logs = [(load_gimbal_log(path), float(start)) for (path, _), start in zip(pairs, clip_starts)]
    aligned = align_log(
        logs, start_s=float(segment.meta["start_s"]), fps=float(segment.meta["fps"]), frame_count=frame_count
    )
    return np.abs(np.nan_to_num(np.diff(aligned["yaw"], prepend=aligned["yaw"][0])))


def _grab_sampled_frames(segment, frames: list[int], width: int = 960) -> dict[int, np.ndarray]:
    """One streaming decode pass, keeping only the sampled frames.

    A per-frame ffmpeg spawn costs ~2 s of startup each; a single pass at the analysis rate costs a few minutes
    for a whole game and serves every sampled frame. The reader seeks to the first wanted frame and steps at
    ``1/fps``, which is exactly the analysis clock, so reader frame ``i`` is analysis frame ``first + i``. Seeking
    matters for a windowed scan: a ten-minute window over a whole game then decodes ten minutes, not the whole
    game.
    """
    from soccer_analytics.ingest.source import open_reader

    if not frames:
        return {}
    wanted = set(frames)
    first = min(wanted)
    last = max(wanted)
    fps = float(segment.meta["fps"])
    out: dict[int, np.ndarray] = {}
    reader = open_reader(segment.meta["video"], fps=fps, width=width, start_s=first / fps)
    for index, (_t, frame) in enumerate(reader.frames()):
        actual = first + index
        if actual in wanted:
            out[actual] = frame
        if actual >= last:
            break
    return out


def detect_line_observations(
    segment,
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
    *,
    length_m: float,
    width_m: float,
    stride: int = LINE_STRIDE,
    max_pan_deg_per_frame: float = MAX_PAN_DEG_PER_FRAME,
    start_frame: int = 0,
    end_frame: int | None = None,
    on_progress=None,
) -> list[LineObservation]:
    """Scan sampled frames for white-line crossings near the predicted markings.

    For each sampled frame, every straight marking is projected through the *corrected* chain; where both
    endpoints land in frame, a perpendicular scan looks for the white stroke. A hit becomes a
    direction-constrained observation: the pitch point is the *predicted* midpoint of the marking (the detection
    only measures across the line, so the along-line coordinate stays the prediction's), and ``direction`` is
    the predicted line's image normal.

    ``start_frame``/``end_frame`` restrict the scan to a window, which is how a whole-game decode is avoided when
    fitting the pan axis from a few representative passages; the default is the whole segment.
    """
    pan = _pan_rates(segment, len(q))
    stop = len(q) if end_frame is None else min(len(q), max(0, end_frame))
    start = max(0, min(len(q), start_frame))
    sampled = [f for f in range(start, stop, stride) if pan[f] <= max_pan_deg_per_frame]
    images = _grab_sampled_frames(segment, sampled)
    segments = pitch_marking_lines(length_m, width_m)

    focal_arr = np.asarray(focal, dtype=np.float64)
    observations: list[LineObservation] = []
    for done, frame in enumerate(sampled):
        if on_progress is not None and (done % 10 == 0 or done == len(sampled) - 1):
            on_progress((done + 1) / len(sampled))
        image = images.get(frame)
        if image is None:
            continue
        q_f = calibration.corrected_chain(q[frame : frame + 1])[0]
        f_f = float(calibration.corrected_focal(focal_arr[frame : frame + 1])[0])
        uv_all, ok_all = pitch_to_pixels(
            calibration, np.array([p for seg in segments for p in seg], dtype=np.float64), q_f, f_f
        )
        uv_all = uv_all.reshape(len(segments), 2, 2)
        ok_all = ok_all.reshape(len(segments), 2)
        for seg_index, ((a, b), uv, ok) in enumerate(zip(segments, uv_all, ok_all)):
            if not ok.all():
                continue
            obs = _scan_for_line(image, frame, a, b, uv[0], uv[1], seg_index)
            if obs is not None:
                observations.append(obs)
    return observations


def _scan_for_line(
    image: np.ndarray,
    frame: int,
    a: tuple[float, float],
    b: tuple[float, float],
    pa: np.ndarray,
    pb: np.ndarray,
    seg_index: int,
) -> LineObservation | None:
    """Scan one projected marking for its white stroke; returns an observation or None.

    ``pa``/``pb`` are the predicted endpoints in normalized pixels; the scan works in image pixels of ``image``
    (which may be a different width than 1920, so everything is scaled by the image's own width).
    """
    h, w = image.shape[:2]

    pa_px = np.array([pa[0] * w, pa[1] * w])
    pb_px = np.array([pb[0] * w, pb[1] * w])
    d = pb_px - pa_px
    length = float(np.hypot(*d))
    if length < MIN_LINE_LENGTH_PX:
        return None
    tangent = d / length
    normal = np.array([-tangent[1], tangent[0]])

    half = SEARCH_HALF_WIDTH * w
    band = CONTEXT_FRACTION * half
    n_stations = max(4, int(length / 12))
    ts = np.linspace(0.15, 0.85, n_stations)
    offsets: list[float] = []
    contrast_sum = 0.0
    for t in ts:
        base = pa_px + t * d
        if not (band < base[0] < w - band and band < base[1] < h - band):
            continue
        ss = np.linspace(-half, half, 25)
        pts = base[None, :] + ss[:, None] * normal[None, :]
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        gray = image[ys, xs].astype(np.float64).mean(axis=1)
        context = gray[np.abs(ss) >= half - band]
        if len(context) < 4:
            continue
        level = float(np.median(context))
        white = gray > level + WHITE_MARGIN
        if not white.any():
            continue
        offsets.append(float(ss[white].mean()))
        contrast_sum += float(gray[white].mean() - level)
    if len(offsets) < MIN_STATIONS or len(offsets) / len(ts) < MIN_WHITE_FRACTION:
        return None
    offsets_arr = np.asarray(offsets)
    spread = float(np.percentile(offsets_arr, 75) - np.percentile(offsets_arr, 25))
    if spread > MAX_PERP_ERROR * w:
        return None  # the white pixels do not form one line
    offset = float(np.median(offsets_arr))
    if abs(offset) > MAX_PERP_ERROR * w:
        return None  # the detected line is not the predicted one

    # The observation's pixel: the predicted midpoint, shifted across the line by the measured offset. The
    # pitch point stays the predicted midpoint - the along-line coordinate is unmeasured by a line crossing.
    mid = pa_px + 0.5 * d + offset * normal
    pitch_mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return LineObservation(
        frame=frame,
        u=float(mid[0] / w),
        v=float(mid[1] / w),
        pitch_x=float(pitch_mid[0]),
        pitch_y=float(pitch_mid[1]),
        direction=(float(-normal[1]), float(normal[0])),
        label=f"line[{seg_index}]",
        contrast=contrast_sum / max(len(offsets), 1),
    )


def observations_to_landmarks(observations: list[LineObservation]) -> list:
    """Line observations as direction-constrained ``Landmark`` s, ready for :func:`fit_drift`."""
    from soccer_analytics.geometry.pitch_calibration import Landmark

    return [Landmark(o.frame, o.u, o.v, o.pitch_x, o.pitch_y, o.label, o.direction) for o in observations]


def refine_with_lines(
    calibration: PitchCalibration,
    q: np.ndarray,
    focal: np.ndarray,
    landmarks: list,
    aspect: float,
    *,
    clicks: list | None = None,
    max_rotation: float = 0.02,
    max_scale: float = 1.08,
    smoothness: float | None = None,
    on_progress=None,
) -> PitchCalibration:
    """Re-anchor a saved calibration against automatic line observations, without moving the base pose.

    The clicks registered the base pose; the line observations only say how the *chain* had drifted from it at
    each sampled frame. So the base pose is kept exactly and :func:`fit_drift` is re-run with the observations
    *and the clicks* - many knots instead of two, each pinning the pointing and the focal where the pitch was
    actually seen. The clicks stay in the fit because they are the only exact ground control: without them the
    line observations alone can walk the correction away from the registration the user made (measured: the
    frame-2739 clicks went from 7 px to 280 px of error when the fit ran on lines only). The bounds are much
    tighter than the click fit's: a line scan that wants more than a degree or two of pointing per anchor has
    misdetected, not corrected.

    Observations are screened twice before the fit: gross outliers (a misdetected line - a player's sock, a
    shadow edge) by a hard cap at ``OUTLIER_PERP``, and the tail by a per-anchor median so a frame whose
    observations disagree with each other contributes nothing. Returns a new calibration (the input is
    immutable) whose drift correction carries the line anchors. With no usable observations the calibration is
    returned unchanged.
    """
    from soccer_analytics.geometry.drift import DriftCorrection, fit_drift
    from soccer_analytics.geometry.pitch_calibration import pitch_to_pixels

    if not landmarks:
        return calibration
    chain = {int(i): (q[i], float(focal[i])) for i in range(len(q))}
    # Screen the observations against the base pose. The scan's own MAX_PERP_ERROR is the tolerance *at
    # detection time*; by fit time the base pose has drifted, so the honest screen is a generous multiple of
    # it (measured: the real game's observations sit at a median of 0.009 and a p90 of 0.13 normalized px
    # against the base pose - the drift this fit exists to correct). A hard cap kills the gross misdetections;
    # a per-anchor median screen then drops frames whose observations disagree with each other, which a
    # single-click-style screen cannot see.
    capped: list = []
    for lm in landmarks:
        Q, f = chain[int(lm.frame)]
        uv, ok = pitch_to_pixels(calibration, np.array([[lm.pitch_x, lm.pitch_y]]), Q, f)
        if not ok[0]:
            continue
        perp = abs((float(uv[0, 0]) - lm.u) * lm.direction[0] + (float(uv[0, 1]) - lm.v) * lm.direction[1])
        if perp <= OUTLIER_PERP:
            capped.append(lm)
    by_frame: dict[int, list] = {}
    for lm in capped:
        by_frame.setdefault(int(lm.frame), []).append(lm)
    kept: list = []
    for frame, group in by_frame.items():
        perps = []
        for lm in group:
            Q, f = chain[frame]
            uv, ok = pitch_to_pixels(calibration, np.array([[lm.pitch_x, lm.pitch_y]]), Q, f)
            perp = abs((float(uv[0, 0]) - lm.u) * lm.direction[0] + (float(uv[0, 1]) - lm.v) * lm.direction[1])
            perps.append(perp)
        median = float(np.median(perps))
        kept.extend(lm for lm, p in zip(group, perps) if p <= max(2.0 * median, MAX_PERP_ERROR))
    if on_progress is not None:
        on_progress(0.5)
    if not kept:
        return calibration
    # The clicks are part of the fit: they are the exact ground control the base pose was registered on, and
    # keeping them pins the correction to that registration instead of letting the lines redefine it.
    if clicks:
        kept = list(clicks) + kept
    # Warm start from the existing correction so the refinement stays a refinement.
    initial = calibration.drift
    kwargs = {} if smoothness is None else {"smoothness": smoothness}
    drift = fit_drift(
        kept,
        calibration,
        chain,
        aspect,
        initial=initial,
        max_rotation=max_rotation,
        max_scale=max_scale,
        **kwargs,
    )
    if on_progress is not None:
        on_progress(1.0)
    if drift is None:
        return calibration
    return PitchCalibration(
        position=calibration.position,
        base_rotation=calibration.base_rotation,
        focal_scale=calibration.focal_scale,
        aspect=calibration.aspect,
        rms_error_m=calibration.rms_error_m,
        residuals_m=calibration.residuals_m,
        excluded=calibration.excluded,
        ambiguous=calibration.ambiguous,
        ill_conditioned=calibration.ill_conditioned,
        drift=drift,
        pose_source=calibration.pose_source,
    )

"""Highlight reels: pick the moments, then cut them out of the source with ffmpeg.

Four tiers, mirroring what the product describes:
* ``clip``   - 15-30 s single moments, for sharing;
* ``goals``  - 1-2 min of scoring plays (from manual tags, since no goal event is inferred);
* ``saves``  - a short reel of the saves a human tagged;
* ``match``  - a ~5 min summary spread across the whole match.

Moment ranking is explicit and inspectable: manual tags outrank audio candidates, and momentum swings add weight, so
a coach can see why a clip was chosen instead of being handed a black box.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from soccer_analytics.analysis.events import VERDICT_FALSE, Event
from soccer_analytics.ingest.ffmpeg_reader import run_ffmpeg_with_progress
from soccer_analytics.ingest.source import ClipSource, as_source, concat_copies, probe_source, window_segments

TIER_SECONDS = {"clip": 24.0, "goals": 90.0, "saves": 60.0, "match": 300.0}
MIN_CLIP_S = 15.0
MAX_CLIP_S = 30.0
PREVIEW_DIR = "previews"  # previews sit under the match's highlights/ directory, one file per moment
PREVIEW_WIDTH = 1280  # a preview only has to be clear enough to check a tag, not broadcast quality
# Audio codec for the inline previews, tried in order. AAC is the natural choice for an MP4, but the browser this
# dashboard is viewed in has no AAC decoder - a canonical AAC file fails to demux with
# DEMUXER_ERROR_NO_SUPPORTED_STREAMS - while MP3 plays everywhere. Exported reels keep AAC, which is what phones
# and social platforms expect. The fallback matters for an ffmpeg built without libmp3lame.
PREVIEW_AUDIO_CODECS = ("mp3", "aac")
_AUDIO_ENCODERS = {
    "aac": ["-c:a", "aac", "-b:a", "128k"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
}
MANUAL_WEIGHT = 10.0  # a human tagged it, so it matters more than any audio hint
AUDIO_WEIGHT = 1.0
MOMENTUM_WEIGHT = 4.0
LEAD_S = 4.0  # start a little before the moment so the build-up is included
TAIL_S = 6.0
# A goal or save is worth the move that produced it, not the walk back from the net: place the event nine
# seconds into the clip and leave six seconds of aftermath. The 15 s window matches MIN_CLIP_S, so the exported
# clip is exactly this window rather than the old 4 s before / 11 s after that padding produced.
BUILD_UP_EVENT_TYPES = {"goal", "penalty", "save"}
BUILD_UP_LEAD_S = 9.0
BUILD_UP_TAIL_S = 6.0
# Exported reels are not previews: they keep the source's own resolution and frame rate and use a visually
# near-lossless encode. Previews stay on the cheap/fast settings above. CRF 18 (x264) and CQ 18 (NVENC) are
# close enough to the camera original that the reel is not the thing that softens a 4K60 recording.
REEL_CRF = 18
REEL_PRESET = "medium"
REEL_NVENC_CQ = 18

# What a preview costs, in the order of the trade. Reviewing a scan means watching ninety of these one after the
# other, so the window is deliberately short (ten seconds, the blast four seconds in) and the question is only how
# much picture to pay for. Measured on the real 4K60 game, per clip: the sound alone is 0.2 s, keyframe pictures
# ~3 s, every frame ~11 s. Note that *lowering the resolution* buys nothing - a 640 px clip costs the same 10 s as
# a 1280 px one, because decoding 10 s of 4K dominates and the encoder was never the cost. Dropping *frames* is the
# lever: the camera writes a keyframe a second, so `skip_frame=nokey` turns 600 decodes into 10.
PREVIEW_SETTINGS: dict[str, dict[str, object]] = {
    "audio": {"audio_only": True, "audio_codecs": ("mp3",), "width": 0, "fps": 0, "skip_frame": None},
    "quick": {"audio_only": False, "audio_codecs": ("mp3", "aac"), "width": 640, "fps": 4, "skip_frame": "nokey"},
    "full": {"audio_only": False, "audio_codecs": ("mp3", "aac"), "width": PREVIEW_WIDTH, "fps": 30, "skip_frame": None},
}
PREVIEW_MODE_EXTENSIONS = {"audio": ".mp3", "quick": ".mp4", "full": ".mp4"}


@dataclass(frozen=True)
class Moment:
    """A candidate highlight with its start/end in source seconds and why it was chosen."""

    time_s: float
    start_s: float
    end_s: float
    weight: float
    reason: str
    event_type: str = "other"
    team: int = -1
    video: str = ""  # the recording the seconds refer to; empty means "whatever source the caller passes"
    # Seconds on *that* recording's clock. When the moment is cut from a different file than its own ``video``, the
    # window is mapped through the game manifest: ``start_s``/``end_s`` stay on the moment's own recording and are
    # translated at cut time.
    clip_start_s: float | None = None
    clip_end_s: float | None = None


@dataclass(frozen=True)
class Reel:
    tier: str
    moments: tuple[Moment, ...]
    duration_s: float


def moment_for_event(event: Event) -> Moment:
    """The highlight window a single event describes, before any ranking or de-duplication.

    Shared by :func:`build_moments` and the dashboard's inline preview, so the clip a coach plays in place is cut
    from exactly the same seconds the exported reel would use. The window carries the event's own video with it,
    because its seconds are seconds of *that* recording.
    """
    weight = MANUAL_WEIGHT if event.source == "manual" else AUDIO_WEIGHT
    if event.type == "goal":
        weight *= 3.0
    elif event.type == "penalty":
        weight *= 2.5
    elif event.type in ("shot", "save", "block"):
        weight *= 1.6
    elif event.type == "corner":
        weight *= 1.3
    elif event.type in ("clearance", "tackle"):
        weight *= 1.2
    lead, tail = (
        (BUILD_UP_LEAD_S, BUILD_UP_TAIL_S)
        if event.type in BUILD_UP_EVENT_TYPES
        else (LEAD_S, TAIL_S)
    )
    return Moment(
        time_s=event.time_s,
        start_s=max(0.0, event.time_s - lead),
        end_s=event.time_s + tail,
        weight=weight,
        reason=f"{event.type} ({event.source})" + (f": {event.note}" if event.note else ""),
        event_type=event.type,
        team=event.team,
        video=event.video,
    )


def moment_on_source(
    moment: Moment,
    source: str | Path,
    *,
    clip_offsets: dict[str, float] | None = None,
) -> Moment:
    """The moment's window mapped onto the file it will actually be cut from.

    A moment's seconds are seconds of its own recording (``video``), but a reel is cut from one file - usually the
    selected video. When the two differ, the window is translated through ``clip_offsets`` (a clip path -> its start
    in the combined game, from the game manifest) so the clip shows the moment and not the same *number* of seconds
    of a different part of the match. A moment whose own recording is the source, or that has no video (a manual tag
    or a momentum swing, which belong to whatever the caller passes), comes back as it stands.

    Returns the moment with ``clip_start_s``/``clip_end_s`` set; the original window is untouched so the preview and
    the manifest still describe the moment in its own recording's terms.
    """
    own = Path(moment.video) if moment.video else None
    if own is None or own.resolve() == Path(source).resolve():
        return moment
    offsets = clip_offsets or {}
    offset = offsets.get(str(own.resolve()))
    if offset is None:
        return moment
    shift = offset  # the moment's recording starts this far into the combined game
    return replace(
        moment,
        clip_start_s=max(0.0, moment.start_s + shift),
        clip_end_s=moment.end_s + shift,
    )


def clamp_moment(moment: Moment, duration_s: float) -> Moment | None:
    """Trim a moment's window to a recording of ``duration_s``, or return ``None`` if it does not lie inside it.

    A window that runs off the end of the file used to be discovered by ffmpeg, which simply stops encoding: the
    clip came back shorter than the moment described, with nothing on the page saying why. Trimming it here means
    the page can say so instead - and a moment that lies *past* the recording entirely (a candidate scanned on the
    combined game, previewed against a single camera file) is refused rather than cut into a clip that cannot
    contain it.
    """
    if duration_s <= 0 or moment.start_s >= duration_s or moment.end_s <= 0:
        return None
    start_s = max(0.0, moment.start_s)
    end_s = min(float(duration_s), moment.end_s)
    if end_s - start_s < 0.05:
        return None
    if start_s == moment.start_s and end_s == moment.end_s:
        return moment
    return replace(moment, start_s=start_s, end_s=end_s)


def build_moments(
    events: list[Event],
    momentum: dict | None = None,
    *,
    match_duration_s: float | None = None,
) -> list[Moment]:
    """Score every event and momentum swing into a ranked list of candidate moments.

    Candidates a human rejected are left out: the reels are what somebody watches, and a review that does not
    change the reels is paperwork. Everything not yet reviewed stays in.
    """
    events = [event for event in events if event.verdict != VERDICT_FALSE]
    moments: list[Moment] = [moment_for_event(event) for event in events]

    # A big shift in who is on top is worth watching even with nothing tagged (the momentum chart's peaks).
    for minute, bucket in (momentum or {}).items():
        swing = abs(float(bucket["team_0"]) - 0.5)
        if swing < 0.25:
            continue
        center = float(minute) * 60.0 + 30.0
        moments.append(
            Moment(
                time_s=center,
                start_s=max(0.0, center - LEAD_S),
                end_s=center + TAIL_S,
                weight=MOMENTUM_WEIGHT * swing * 2.0,
                reason=f"momentum swing ({(bucket['team_0']):.0%} action share)",
            )
        )

    if match_duration_s:
        moments = [m for m in moments if m.start_s < match_duration_s]
    moments.sort(key=lambda m: -m.weight)
    return moments


def _clip_length(moment: Moment, tier: str) -> float:
    if tier == "clip":
        return min(MAX_CLIP_S, max(MIN_CLIP_S, moment.end_s - moment.start_s))
    return max(MIN_CLIP_S, moment.end_s - moment.start_s)


def select_reel(
    tier: str,
    moments: list[Moment],
    *,
    match_duration_s: float | None = None,
    min_gap_s: float = 20.0,
) -> Reel:
    """Greedily take the best non-overlapping moments until the tier's length budget is filled.

    For the long tiers, moments are spread across the match (not all from one hot spell) by preferring the best
    moment in each pass over the timeline and enforcing a minimum gap between picks.
    """
    if tier not in TIER_SECONDS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {tuple(TIER_SECONDS)}")
    if tier == "goals":
        candidates = [m for m in moments if m.event_type in ("goal", "penalty")]
        if not candidates:
            candidates = [m for m in moments if m.event_type in ("shot", "save", "block")]
    elif tier == "saves":
        candidates = [m for m in moments if m.event_type == "save"]
    else:
        candidates = list(moments)

    budget = TIER_SECONDS[tier]
    chosen: list[Moment] = []
    used = 0.0
    for moment in candidates:
        length = _clip_length(moment, tier)
        if used + length > budget and chosen:
            continue
        if any(abs(moment.time_s - other.time_s) < min_gap_s for other in chosen):
            continue
        chosen.append(moment)
        used += length
        if used >= budget:
            break
    chosen.sort(key=lambda m: m.time_s)
    return Reel(tier=tier, moments=tuple(chosen), duration_s=round(used, 1))


def _moment_source(moment: Moment, default: str | Path) -> Path:
    """Where this moment's seconds are to be found: its own recording when it has one, else the caller's source.

    A missing file falls back rather than failing the whole export - the footage may simply not be on this machine
    right now - but a moment that *knows* where it came from is never cut from somewhere else while that file is
    there, which is the case that silently produced clips of the wrong part of the match.
    """
    if moment.video:
        own = Path(moment.video)
        if own.exists():
            return own
    return Path(default)


def _encode_clip(
    source: Path,
    moment: Moment,
    output_path: Path,
    *,
    duration_s: float,
    width: int,
    use_gpu: bool,
    audio_codecs: tuple[str, ...] = ("aac",),
    fps: float = 30.0,
    audio_only: bool = False,
    skip_frame: str | None = None,
    video_filter: str | None = None,
    high_quality: bool = False,
    progress=None,
) -> Path:
    """Re-encode one moment window to a normalized H.264 MP4, trying each encoder until one works.

    Cutting with ``-ss``/``-t`` and re-encoding (rather than stream-copying) is what makes the cut land exactly on
    the moment instead of the nearest keyframe. The frames are scaled and the rate fixed so parts can be joined and
    so a preview seeks instantly. ``audio_codecs`` are tried in order (each with a GPU attempt, then CPU), so a
    codec this build of ffmpeg lacks falls through to the next. ``progress(fraction)`` follows the encode's own
    ``out_time_us``.

    ``audio_only`` writes the sound alone - no video decode, no video encode, nothing to scale - which is what
    makes a preview cost a fraction of a second instead of seconds. It is the right trade when the question is
    "is that a whistle?", which is a question about the sound. ``skip_frame`` (an ffmpeg input option, e.g.
    ``"nokey"``) is the other lever: it decodes only the frames the camera marked, which on this footage is one a
    second, so a whole window costs ten decodes rather than six hundred.

    ``video_filter`` replaces the default scale-and-rate filter outright. The one caller that needs it is the
    player-centered cut, whose filter is a 'sendcmd'-driven moving crop rather than a fixed scale - everything else
    about the encode (encoder fallbacks, audio codec, progress) stays the same. ``high_quality`` switches the
    encoder to the near-lossless export preset; previews keep the cheap/fast one.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    attempts = [(codec, False) for codec in audio_codecs] if audio_only else [
        (codec, gpu) for codec in audio_codecs for gpu in ([True, False] if use_gpu else [False])
    ]
    # The moment's window is on its own recording's clock; when it is being cut from a *different* file the mapped
    # ``clip_start_s``/``clip_end_s`` (set by :func:`moment_on_source`) are the seconds to cut.
    cut_start = moment.clip_start_s if moment.clip_start_s is not None else moment.start_s
    cut_end = moment.clip_end_s if moment.clip_end_s is not None else moment.end_s
    # A never-merged game's source is its manifest: the seconds are game seconds, and a window that crosses a
    # clip join cannot be cut by one ffmpeg seek. Each clip's slice is encoded by the ordinary routine - a seek
    # inside its own clip - and the pieces are joined by stream copy, all with the same size and rate. The
    # player-centered cut (`video_filter`, a sendcmd schedule against one continuous run of frames) cannot be
    # replayed piecewise and says so instead of cutting something mis-framed; its windows are seconds long
    # against half-hour clips, so meeting a join is the rare case.
    resolved = as_source(source)
    if isinstance(resolved, ClipSource):
        pieces = window_segments(resolved, cut_start, duration_s)
        if not pieces:
            raise RuntimeError(f"the moment at {cut_start:.1f}s lies outside the recording")
        if len(pieces) > 1:
            if video_filter is not None:
                raise ValueError("this clip crosses a clip join; the player-centered cut needs one recording")
            with tempfile.TemporaryDirectory(prefix="highlight_pieces_") as tmp:
                parts: list[Path] = []
                total = sum(piece.duration_s for piece in pieces) or duration_s
                done = 0.0
                for index, piece in enumerate(pieces):
                    # Same container as the finished file: an audio-only preview's piece is a real mp3, not an
                    # mp3 stream wearing an .mp4 name, so the final copy-join picks the right muxer first time.
                    part = Path(tmp) / f"piece_{index}{output_path.suffix}"

                    def piece_progress(fraction, base=done, span=piece.duration_s) -> None:
                        if progress is not None:
                            progress(min(1.0, (base + span * float(fraction)) / total))

                    _encode_clip(
                        Path(piece.path),
                        replace(moment, clip_start_s=piece.start_s, clip_end_s=piece.end_s),
                        part,
                        duration_s=piece.duration_s,
                        width=width,
                        use_gpu=use_gpu,
                        audio_codecs=audio_codecs,
                        fps=fps,
                        audio_only=audio_only,
                        skip_frame=skip_frame,
                        high_quality=high_quality,
                        progress=piece_progress,
                    )
                    done += piece.duration_s
                    parts.append(part)
                concat_copies(parts, output_path)
                if progress is not None:
                    progress(1.0)
                return output_path
        source = Path(pieces[0].path)
        cut_start = pieces[0].start_s
    for codec, gpu in attempts:
        if output_path.exists():
            output_path.unlink()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        command += ["-ss", f"{cut_start:.3f}"]
        if skip_frame:
            command += ["-skip_frame", skip_frame]  # an input option: it has to precede -i
        command += ["-i", str(source), "-t", f"{duration_s:.3f}"]
        if audio_only:
            command += ["-vn", *_AUDIO_ENCODERS[codec]]
        else:
            command += [
                "-vf", video_filter if video_filter else f"scale={width}:-2:flags=lanczos,fps={fps:g}",
                "-c:v", "h264_nvenc" if gpu else "libx264",
            ]
            if gpu:
                command += (
                    ["-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", str(REEL_NVENC_CQ), "-b:v", "0"]
                    if high_quality
                    else ["-preset", "p4"]
                )
            else:
                command += (
                    ["-preset", REEL_PRESET, "-crf", str(REEL_CRF)] if high_quality else ["-preset", "veryfast"]
                )
            command += [*_AUDIO_ENCODERS[codec], "-movflags", "+faststart"]
        command += [
            "-progress", "pipe:1", "-nostats",
            str(output_path),
        ]
        returncode, output = run_ffmpeg_with_progress(command, duration_s, progress)
        if returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return output_path
    last_error = output or f"ffmpeg exited with {returncode}"
    if output_path.exists():
        output_path.unlink()
    raise RuntimeError(f"ffmpeg failed cutting the clip at {cut_start:.1f}s: {last_error}")


def export_player_clip(
    source: str | Path,
    times,
    boxes,
    output_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    source_size: tuple[int, int] | None = None,
    width: int = 1280,
    use_gpu: bool = True,
    progress=None,
):  # noqa: ANN201 - (Path, FramingPlan)
    """Cut a window out of ``source`` that stays centered on one player, cropping rather than following the ball.

    ``times`` and ``boxes`` are the player's own observations: source seconds and width-normalized
    ``(x1, y1, x2, y2)`` rectangles (exactly what the replay payload's ``boxes`` are). The crop is sized from the
    player's median height, moved by ffmpeg's ``sendcmd`` once per source frame, and clamped to the frame - see
    ``analysis.framing`` for why the trajectory is smoothed, why the command rate is the pan's update rate, and
    what happens at a gap.

    ``source_size`` is the video's ``(width, height)``; it is probed when not given, but callers that already know
    it (the dashboard probes every video once) should pass it rather than pay for another ffprobe.

    Returns ``(path, plan)``: the plan carries the window the clip *actually* covers, which is shorter than the
    requested one when the appearance ends or has a gap in it - the caller can then say so instead of labeling a
    short clip with the length that was asked for. Raises ``ValueError`` when the track cannot be framed (no
    observations) and the ``RuntimeError`` from the encoder when ffmpeg fails - the page reports both rather than
    showing a broken clip.
    """
    from soccer_analytics.analysis.framing import command_schedule, plan_framing, sendcmd_filter

    source = Path(source)
    if source_size is None:
        from soccer_analytics.ingest.source import probe_source

        probe = probe_source(source)
        source_size = (int(probe.width), int(probe.height))
    plan = plan_framing(
        times, boxes,
        start_s=float(start_s),
        duration_s=float(duration_s),
        source_width=int(source_size[0]),
        source_height=int(source_size[1]),
    )
    if plan is None:
        raise ValueError("this appearance has no observations to frame")
    moment = Moment(
        time_s=float(start_s), start_s=float(start_s), end_s=float(start_s) + plan.duration_s,
        weight=0.0, reason="player-centered clip",
    )
    # The schedule goes to ffmpeg by *file*, not inline: at frame rate a minute-long clip's command list is
    # larger than the 128 KiB Linux allows in a single argument, and the whole encode would die on it. The file
    # is written where the temp directory lives and removed when the encode ends, either way.
    schedule = tempfile.NamedTemporaryFile("w", prefix="framing_schedule_", suffix=".txt", delete=False)
    schedule_path = Path(schedule.name)
    try:
        with schedule:
            schedule.write(command_schedule(plan))
        output = _encode_clip(
            source,
            moment,
            Path(output_path),
            duration_s=plan.duration_s,
            width=width,
            use_gpu=use_gpu,
            fps=30,
            video_filter=sendcmd_filter(plan, schedule_file=str(schedule_path), scale_width=width),
            progress=progress,
        )
    finally:
        schedule_path.unlink(missing_ok=True)
    return output, plan


def _copy_window(
    source: Path,
    start_s: float,
    duration_s: float,
    output_path: Path,
    *,
    progress=None,
) -> Path:
    """Stream-copy one already-seekable window, keeping the source's exact picture and sound.

    A copy cannot land mid-GOP: ``-ss`` starts at the keyframe at or before the window, which is why the reel's
    moments carry lead-in and tail. That is the right trade for an export - the alternative is re-encoding and
    losing some of a 4K camera original - and the copy is also far faster than an encode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{duration_s:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output_path),
    ]
    returncode, output = run_ffmpeg_with_progress(command, duration_s, progress)
    if returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed copying the clip at {start_s:.1f}s: {output or returncode}")
    return output_path


def _copy_clip(
    source: str | Path,
    moment: Moment,
    output_path: Path,
    *,
    duration_s: float,
    progress=None,
) -> Path:
    """Copy one moment window, resolving a game manifest into its clips as needed."""
    source = Path(source)
    cut_start = moment.clip_start_s if moment.clip_start_s is not None else moment.start_s
    resolved = as_source(source)
    if isinstance(resolved, ClipSource):
        pieces = window_segments(resolved, cut_start, duration_s)
        if not pieces:
            raise RuntimeError(f"the moment at {cut_start:.1f}s lies outside the recording")
        if len(pieces) > 1:
            with tempfile.TemporaryDirectory(prefix="highlight_copy_pieces_") as tmp:
                tmp_dir = Path(tmp)
                parts: list[Path] = []
                total = sum(piece.duration_s for piece in pieces) or duration_s
                done = 0.0
                for index, piece in enumerate(pieces):
                    part = tmp_dir / f"piece_{index}.mp4"

                    def piece_progress(fraction, base=done, span=piece.duration_s) -> None:
                        if progress is not None:
                            progress(min(1.0, (base + span * float(fraction)) / total))

                    _copy_window(
                        Path(piece.path), piece.start_s, piece.duration_s, part, progress=piece_progress
                    )
                    parts.append(part)
                    done += piece.duration_s
                concat_copies(parts, output_path)
                return output_path
        source = Path(pieces[0].path)
        cut_start = pieces[0].start_s
        duration_s = pieces[0].duration_s
    return _copy_window(source, cut_start, duration_s, output_path, progress=progress)


def _copy_reel(
    source: str | Path,
    reel: Reel,
    output_path: Path,
    *,
    clip_offsets: dict[str, float] | None = None,
    progress=None,
) -> Path:
    """Cut every moment by stream copy and join the parts; raises if the parts cannot be joined."""
    with tempfile.TemporaryDirectory(prefix="highlight_copy_") as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        clip_share = 0.94 / len(reel.moments)
        for index, moment in enumerate(reel.moments):
            part = tmp_dir / f"part_{index:03d}.mp4"
            on_source = moment_on_source(moment, source, clip_offsets=clip_offsets)
            _copy_clip(
                _moment_source(moment, source),
                on_source,
                part,
                duration_s=max(MIN_CLIP_S, on_source.end_s - on_source.start_s),
                progress=(
                    None
                    if progress is None
                    else (lambda fraction, index=index: progress(clip_share * (index + fraction)))
                ),
            )
            parts.append(part)
        if progress is not None:
            progress(0.96)
        concat_copies(parts, output_path)
    if progress is not None:
        progress(1.0)
    return output_path


def _encode_reel(
    source: str | Path,
    reel: Reel,
    output_path: Path,
    *,
    width: int,
    fps: float,
    use_gpu: bool,
    clip_offsets: dict[str, float] | None = None,
    progress=None,
) -> Path:
    """The fallback: re-encode every moment at the source size/rate with export-quality settings."""
    with tempfile.TemporaryDirectory(prefix="highlight_encode_") as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        clip_share = 0.97 / len(reel.moments)
        for index, moment in enumerate(reel.moments):
            part = tmp_dir / f"part_{index:03d}.mp4"
            on_source = moment_on_source(moment, source, clip_offsets=clip_offsets)
            _encode_clip(
                _moment_source(moment, source),
                on_source,
                part,
                duration_s=max(MIN_CLIP_S, on_source.end_s - on_source.start_s),
                width=width,
                fps=fps,
                use_gpu=use_gpu,
                high_quality=True,
                progress=(
                    None
                    if progress is None
                    else (lambda fraction, index=index: progress(clip_share * (index + fraction)))
                ),
            )
            parts.append(part)
        if progress is not None:
            progress(0.98)
        concat_copies(parts, output_path)
    if progress is not None:
        progress(1.0)
    return output_path


def export_reel(
    source: str | Path,
    reel: Reel,
    output_path: str | Path,
    *,
    width: int | None = None,
    fps: float | None = None,
    use_gpu: bool = True,
    progress=None,
    clip_offsets: dict[str, float] | None = None,
) -> Path:
    """Cuts the reel's moments out of ``source`` and joins them into one file.

    An H.264 source - the camera's own codec - is cut by stream copy, so the reel's picture and sound are the
    source's, not a downscaled or recompressed copy. Copy cuts land on keyframes; that is why moments carry
    lead-in and tail. When a copy-join cannot work (a manifest whose clips differ, another codec, a join that
    ffmpeg refuses), each part is re-encoded at the source's own size and rate with the high-quality settings
    (:data:`REEL_CRF`) rather than the cheap preview ones, so the cut lands exactly. ``width``/``fps`` override
    the source geometry for a deliberate downscale. ``progress`` covers the whole reel: the clips dominate, the
    join is the final few percent.

    ``clip_offsets`` maps a recording's path to where it starts inside ``source`` (from the game manifest). A moment
    found in another recording - a whistle scanned on a single camera file, say - is translated through it, so the
    reel cut from the combined game shows the moment rather than the same *number* of seconds of the wrong part.
    """
    source, output_path = Path(source), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not reel.moments:
        raise ValueError(f"reel {reel.tier!r} has no moments to export")
    probe = probe_source(source)
    if width is None:
        width = int(probe.width)
    if fps is None or fps <= 0:
        fps = float(probe.fps) or 30.0
    width = max(2, int(width))
    fps = max(1.0, float(fps))

    # The camera writes H.264; a lossless stream copy keeps the source's exact picture and sound and needs no
    # encode at all. A copy cuts on keyframes, which is why moments carry lead-in and tail. A manifest whose
    # clips differ, or an unusual codec, can fail to join - then the fallback re-encodes at source size/rate
    # with export quality rather than returning a broken reel.
    if probe.codec == "h264":
        try:
            return _copy_reel(
                source, reel, output_path, clip_offsets=clip_offsets, progress=progress
            )
        except (RuntimeError, OSError):
            output_path.unlink(missing_ok=True)
    return _encode_reel(
        source,
        reel,
        output_path,
        width=width,
        fps=fps,
        use_gpu=use_gpu,
        clip_offsets=clip_offsets,
        progress=progress,
    )


def preview_clip_name(moment: Moment, mode: str = "full") -> str:
    """A stable filename for a moment's preview clip, for one way of cutting it.

    The name is a hash of the window, what it shows and the *mode*, so the same tag reuses its cached clip (no
    re-encode on every rerun) while editing the tag - its time, type or note - produces a new file rather than a
    stale one. The mode is part of it because switching between a full-size clip and the sound alone is a different
    artifact, not a different look at the same one, and a cached file must never be served for the wrong kind.
    """
    if mode not in PREVIEW_SETTINGS:
        raise ValueError(f"unknown preview mode {mode!r}; expected one of {tuple(PREVIEW_SETTINGS)}")
    # The full-size clip keeps the name its window always had, so the previews a match has already cut stay usable
    # when the cheaper modes arrive beside it.
    scope = "" if mode == "full" else f"{mode}|"
    key = f"{scope}{moment.start_s:.3f}|{moment.end_s:.3f}|{moment.event_type}|{moment.team}|{moment.reason}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"preview_{digest}{PREVIEW_MODE_EXTENSIONS[mode]}"


def export_moment(
    source: str | Path,
    moment: Moment,
    output_path: str | Path,
    *,
    mode: str = "full",
    width: int | None = None,
    use_gpu: bool = True,
    progress=None,
    clip_offsets: dict[str, float] | None = None,
) -> Path:
    """Cut a single moment out of ``source`` for inline preview in the dashboard.

    Unlike :func:`export_reel` this does not pad the clip to ``MIN_CLIP_S`` or join parts - it is the raw window the
    moment describes, scaled down so it loads and seeks quickly in the browser. When the moment carries its own
    video (every detected candidate does), that file is the one cut from; ``source`` is only the fallback for tags
    and momentum moments, which belong to whatever the page has selected.

    ``clip_offsets`` maps a recording's path to where it starts inside ``source``, for the same reason as
    :func:`export_reel`.

    ``mode`` is one of :data:`PREVIEW_SETTINGS` and is the whole cost trade: ``"audio"`` writes an MP3 of the
    window (no video decode at all), ``"quick"`` a low-rate flipbook of the camera's own keyframes, ``"full"`` the
    size a shared clip wants. ``width`` overrides the mode's width where a caller has its own idea - the tests use a
    tiny one.
    """
    settings = PREVIEW_SETTINGS.get(mode)
    if settings is None:
        raise ValueError(f"unknown preview mode {mode!r}; expected one of {tuple(PREVIEW_SETTINGS)}")
    on_source = moment_on_source(moment, source, clip_offsets=clip_offsets)
    duration_s = max(1.0, on_source.end_s - on_source.start_s)
    return _encode_clip(
        _moment_source(moment, source),
        on_source,
        Path(output_path),
        duration_s=duration_s,
        width=int(width if width is not None else settings["width"]),
        fps=int(settings["fps"]),
        use_gpu=use_gpu,
        audio_codecs=tuple(settings["audio_codecs"]),
        audio_only=bool(settings["audio_only"]),
        skip_frame=settings["skip_frame"],
        progress=progress,
    )


def reel_manifest(reel: Reel, video_path: str | Path) -> dict:
    """A sidecar describing a reel, so a coach can see what went into it.

    Each moment carries its own recording where it has one: a reel can mix hand tags (cut from the video the page
    was showing) with candidates detected in another file, and the seconds of one are not the seconds of the other.
    """
    return {
        "tier": reel.tier,
        "video": str(video_path),
        "duration_s": reel.duration_s,
        "moments": [
            {
                "time_s": round(m.time_s, 2),
                "start_s": round(m.start_s, 2),
                "end_s": round(m.end_s, 2),
                "weight": round(m.weight, 2),
                "reason": m.reason,
                "event_type": m.event_type,
                "team": m.team,
                "video": m.video,
            }
            for m in reel.moments
        ],
    }


def write_manifest(reel: Reel, video_path: str | Path, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reel_manifest(reel, video_path), indent=2))
    return path

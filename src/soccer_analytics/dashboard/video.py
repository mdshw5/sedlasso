"""H.264 annotated match video: the same overlays as the MJPEG stream, at the source's own frame rate.

The MJPEG endpoint pushes one JPEG per *analysis* frame - five a second - which is honest for a live overlay
but leaves the source's detail (and its 60 fps) on the table. This module drives the same renderer at whatever
rate the caller asks for, interpolating between analysis samples (``AnnotatedMatch.render_at``), and hands the
frames to an H.264 encoder over a pipe:

* :func:`encode_clip` - a bounded window encoded into an MP4 with its index at the front, so a player can seek
  it (the server caches these and serves them with range requests);
* :func:`iter_live_chunks` - the endless variant: a fragmented MP4 streamed out as the frames are made, paced
  to real time, which browsers play from a plain ``<video>`` (no MSE, no HLS).

What the box can *sustain* is the caller's to know: this machine decodes full 4K at ~28 fps, so a live 4K60
stream cannot keep up, while a bounded 4K clip simply takes longer to encode. NVENC is preferred, x264 is the
fallback; the frames never touch disk - only a finished clip does.
"""

from __future__ import annotations

import hashlib
import os
import select
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path

import numpy as np

from soccer_analytics.ingest.ffmpeg_reader import FFmpegError, probe_video
from soccer_analytics.ingest.source import FileSource, as_source, open_reader, window_segments

# Widths the encoded routes will serve. Above the MJPEG cap on purpose: this is the endpoint for the full
# picture. 3840 is what the source cameras record; anything wider is a mistake, not a request.
MAX_VIDEO_WIDTH = 3840
# A live stream defaults to 1080p: the decode has to run ahead of the encoder for a stream to be live at all,
# and this box decodes 4K at ~28 fps - fine for a clip, not enough for 60 fps of anything.
DEFAULT_LIVE_WIDTH = 1920
DEFAULT_CLIP_SECONDS = 10.0
MAX_CLIP_SECONDS = 300.0
# The direct path (no overlays) targets this output timeline. The source is sampled at
# ``min(source_fps, DISPLAY_FPS / rate)``, so a 60 fps camera gives 60 fps at 1x, 30 fps at 2x,
# 15 fps at 4x and 7.5 fps at 8x - only the frames the viewer can actually see are encoded.
DIRECT_DISPLAY_FPS = 60.0
CLIP_CACHE_TTL_S = 900.0
CLIP_CACHE_MAX = 8
_CHUNK_BYTES = 64 * 1024


class VideoError(RuntimeError):
    """An encode that could not start, keep up, or finish - the route turns it into an error response."""


# The encoder candidates per codec, best first. ``codec=hevc`` is an explicit choice, not the default: HEVC in
# MP4 is smaller at equal quality on this hardware, but browsers only play it where the platform decodes it
# (Chromium on this box does not), so the compatible H.264 stays the default and HEVC is for downloads and
# players that know it.
ENCODERS = {
    "h264": ("h264_nvenc", "libx264"),
    "hevc": ("hevc_nvenc", "libx265"),
}
CODECS = tuple(ENCODERS)


class _NeverStops:
    stopped = False


_NEVER_STOPPED = _NeverStops()


@lru_cache(maxsize=4)
def pick_encoder(codec: str = "h264") -> str:
    """The encoder this ffmpeg can actually open for a codec: the GPU one when the driver takes it, else CPU.

    Probed with two tiny frames rather than assumed: a machine with an NVIDIA card but no encode permission
    (or a filled session table) must fall back to a working encoder, not fail every request. Cached per codec:
    the answer cannot change while the process lives.
    """
    candidates = ENCODERS.get(codec, ENCODERS["h264"])
    for encoder in candidates:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            # 320x240, not a thumbnail: HEVC's NVENC refuses frames below its minimum size (64x64 fails with
            # "dimensions are less than the minimum supported value"), which would silently pick the CPU encoder.
            "-f", "lavfi", "-i", "nullsrc=size=320x240:rate=10",
            "-frames:v", "2", "-c:v", encoder, "-f", "null", "-",
        ]
        if subprocess.run(command, capture_output=True).returncode == 0:
            return encoder
    raise VideoError(f"no usable {codec} encoder available (tried {', '.join(candidates)})")


def _atempo_chain(rate: float) -> str:
    """The audio tempo filter chain for a content rate: ``atempo`` only accepts 0.5-2.0 per stage.

    Slowing down and speeding up must not turn the match into chipmunks, so the audio is retimed by stretching
    it rather than resampling it; 4x is two 2x stages, 0.25x two halves.
    """
    remaining = max(0.25, min(8.0, float(rate)))
    stages = []
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.4f}")
    return ",".join(stages)


def encoder_quality_args(encoder: str, *, live: bool = False) -> list[str]:
    """Quality/rate-control flags for an encoder, shared by the raw-frame and direct-ffmpeg export paths."""
    if encoder.endswith("nvenc"):
        return ["-preset", "p4", "-tune", "ll" if live else "hq", "-rc", "vbr", "-cq", "23"]
    if "265" in encoder or "hevc" in encoder:
        return ["-preset", "veryfast" if live else "medium", "-crf", "25"]
    return ["-preset", "veryfast" if live else "medium", "-crf", "20"]


def direct_sample_fps(rate: float, source_fps: float = 0.0, display_fps: float = DIRECT_DISPLAY_FPS) -> float:
    """The source frames a second the direct path must sample to fill ``display_fps`` at this speed.

    A speed above 1 needs fewer *source* frames for the same output rate (8x needs 7.5 of them to make 60), so
    the frame-rate filter drops the rest before the encoder ever sees them. Slowing down cannot invent frames:
    the sample rate is capped by the source's own, and the output timeline is then source_fps * rate.
    """
    wanted = max(0.1, float(display_fps) / max(0.01, float(rate)))
    if source_fps and float(source_fps) > 0:
        wanted = min(wanted, float(source_fps))
    return max(1.0, round(wanted, 3))


def direct_skip_frame(rate: float) -> str | None:
    """Which source frames the decoder may skip entirely at this speed.

    Sampling with ``fps`` happens *after* decode, and this box decodes 4K HEVC at roughly 3.2x realtime - so 1x
    and 2x decode everything (2x needs 2 source seconds a real second and holds up), while 4x and 8x cannot
    without a backlog. Those speeds decode keyframes only; at the camera's one-second GOP that is still 4-8
    fresh pictures a second, plenty for the fast scrub they are used as. The frame-rate filter fills the 60 fps
    output from what is decoded.
    """
    rate = float(rate)
    if rate >= 8.0:
        return "nokey"
    if rate >= 4.0:
        return "nointra"  # every I frame, not only IDR keyframes - the most a fast scrub can show in time
    return None


def _direct_input_args(pieces, *, rate: float, pace: bool, hwaccel: str = "auto") -> list[str]:
    """One seekable input per clip piece: the clip resolver's contract (never seek the concat demuxer).

    ``hwaccel="cuda"`` keeps decoded frames on the GPU (``scale_cuda`` and NVENC consume them without a
    download/upload round trip); ``"auto"`` uses whatever decoder ffmpeg picks and copies frames back to
    system memory, which is the portable path and the fallback when CUDA cannot open the stream.
    """
    if hwaccel == "cuda":
        hardware = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    elif hwaccel == "auto":
        hardware = ["-hwaccel", "auto"]
    else:
        hardware = []
    skip = direct_skip_frame(rate)
    command: list[str] = []
    for piece in pieces:
        if pace:
            # Read the source at the playback speed: ffmpeg then produces the output timeline in real time,
            # which is what keeps the live stream near the live edge instead of buffering the whole match.
            command += ["-readrate", f"{float(rate):g}"]
        if skip:
            command += ["-skip_frame", skip]
        command += [
            *hardware,
            "-ss", f"{float(piece.start_s):.3f}",
            "-t", f"{max(0.05, float(piece.duration_s)):.3f}",
            "-i", str(piece.path),
        ]
    return command


def _direct_video_chain(sample_fps: float, width: int, hwaccel: str = "auto") -> str:
    """The per-input video filter: sample the needed frames, scale once, keep square pixels."""
    if hwaccel == "cuda":
        return f"fps={sample_fps:g},scale_cuda={max(2, int(width))}:-2,setsar=1"
    return f"fps={sample_fps:g},scale='min(iw,{max(2, int(width))})':-2,setsar=1"


def direct_encoder_command(
    pieces,
    *,
    display_fps: float = DIRECT_DISPLAY_FPS,
    rate: float = 1.0,
    width: int = 1920,
    live: bool = True,
    encoder: str = "libx264",
    codec: str = "h264",
    source_fps: float = 0.0,
    audio_pieces=None,
    pace: bool = True,
    output: str = "pipe:1",
    hwaccel: str = "auto",
) -> list[str]:
    """The direct playback command: source file(s) in, one H.264/HEVC MP4 out - no Python frames in between.

    Unlike :func:`encoder_command`, this reads the recording itself: ffmpeg seeks each clip, samples only the
    frames the output rate needs (``fps`` before the encoder), retimes video with ``setpts`` and audio with
    ``atempo``, and writes a fragmented MP4 straight to the pipe. ``pieces`` are game-clock windows resolved to
    local clip slices (see :func:`window_segments`); ``audio_pieces`` must be the same list when sound is
    wanted, or ``[]`` for a silent stream. The source's own soundtrack is always kept across a clip join by the
    audio ``concat`` filter, and no call uses ``-shortest`` - a soundtrack shorter than the video cannot cut the
    picture.

    ``rate`` is the content speed (0.25-8x): 2x samples 30 source fps for a 60 fps timeline and stretches the
    audio by 2, 8x samples 7.5 source fps and stretches it by 8. ``display_fps`` is the constant output rate.
    """
    video_pieces = [piece for piece in pieces if float(piece.duration_s) > 0]
    if not video_pieces:
        raise VideoError("the requested window has no footage")
    audio = list(audio_pieces or [])
    if len(audio) != len(video_pieces):
        audio = []  # a partial soundtrack would shift every later input index: stay silent instead
    has_audio = bool(audio)
    n = len(video_pieces)
    rate = max(0.01, float(rate))
    sample_fps = direct_sample_fps(rate, source_fps, display_fps)
    output_rate = max(1.0, sample_fps * rate)
    gop = max(1, int(round(output_rate * (1.0 if live else 2.0))))

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    command += _direct_input_args(video_pieces, rate=rate, pace=pace, hwaccel=hwaccel)

    chain = _direct_video_chain(sample_fps, width, hwaccel)
    if n == 1:
        command += ["-vf", f"{chain},setpts=PTS/{rate:g}", "-map", "0:v:0"]
        if has_audio:
            command += ["-map", "0:a:0?"]
            if abs(rate - 1.0) > 1e-6:
                command += ["-filter:a", _atempo_chain(rate)]
    else:
        filters = [f"[{index}:v:0]{chain}[v{index}]" for index in range(n)]
        joined = "".join(f"[v{index}]" for index in range(n))
        filters.append(f"{joined}concat=n={n}:v=1:a=0,setpts=PTS/{rate:g}[vout]")
        if has_audio:
            filters += [f"[{index}:a:0]aresample=async=1[a{index}]" for index in range(n)]
            audio_chain = "".join(f"[a{index}]" for index in range(n)) + f"concat=n={n}:v=0:a=1"
            if abs(rate - 1.0) > 1e-6:
                audio_chain += f",{_atempo_chain(rate)}"
            filters.append(f"{audio_chain}[aout]")
        command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
        if has_audio:
            command += ["-map", "[aout]"]
    if has_audio:
        command += ["-c:a", "aac", "-b:a", "128k"]
    else:
        command += ["-an"]

    command += [
        "-c:v", encoder, *encoder_quality_args(encoder, live=live),
        "-g", str(gop), "-keyint_min", str(gop), "-r", f"{output_rate:g}",
    ]
    if hwaccel != "cuda":
        command += ["-pix_fmt", "yuv420p"]
    if "265" in encoder or "hevc" in encoder:
        command += ["-tag:v", "hvc1"]
    command += (
        ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", output]
        if live
        else ["-movflags", "+faststart", "-f", "mp4", output]
    )
    return command


def direct_mjpeg_command(
    pieces,
    *,
    display_fps: float = DIRECT_DISPLAY_FPS,
    rate: float = 1.0,
    width: int = 1600,
    source_fps: float = 0.0,
    pace: bool = True,
) -> list[str]:
    """The MJPEG fallback's direct command: sampled frames straight to JPEG, no OpenCV render at all."""
    video_pieces = [piece for piece in pieces if float(piece.duration_s) > 0]
    if not video_pieces:
        raise VideoError("the requested window has no footage")
    n = len(video_pieces)
    rate = max(0.01, float(rate))
    sample_fps = direct_sample_fps(rate, source_fps, display_fps)
    output_rate = max(1.0, sample_fps * rate)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    command += _direct_input_args(video_pieces, rate=rate, pace=pace)
    chain = _direct_video_chain(sample_fps, width)
    if n == 1:
        command += ["-vf", f"{chain},setpts=PTS/{rate:g}", "-map", "0:v:0"]
    else:
        filters = [f"[{index}:v:0]{chain}[v{index}]" for index in range(n)]
        joined = "".join(f"[v{index}]" for index in range(n))
        filters.append(f"{joined}concat=n={n}:v=1:a=0,setpts=PTS/{rate:g}[vout]")
        command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
    command += [
        "-an", "-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj420p", "-r", f"{output_rate:g}",
        "-f", "image2pipe", "pipe:1",
    ]
    return command


def encoder_command(
    *,
    width: int,
    height: int,
    fps: float,
    output: str,
    live: bool,
    encoder: str,
    keyframe_s: float,
    rate: float = 1.0,
    source: str | Path | None = None,
    audio: bool = False,
    audio_start_s: float = 0.0,
    audio_duration_s: float | None = None,
    audio_inputs: list[tuple[str, float, float]] | None = None,
) -> list[str]:
    """The ffmpeg invocation both encoders share: raw BGR frames in, H.264 MP4 out.

    ``live`` picks the fragmented-MP4 flags - a stream a player can join mid-flight - and a low-latency tune;
    a clip gets the index at the front instead (``faststart``), which is what makes it seekable, and a slower,
    denser tune. The keyframe interval doubles as the fragment interval for a live stream: about a second of
    video per fragment is what keeps a viewer near the live edge without chopping the stream into confetti.

    ``rate`` is the *content* speed: the muxer's clock runs ``fps * rate`` per source second, so a 2x request
    plays twice as fast in any player - the audio, when asked for, is retimed to match (``atempo``), and the
    writer paces to the same product. ``source``/``audio`` add the match's own soundtrack from the recording,
    seek-together with the video window; ``1:a:0?`` keeps a source without an audio track working, and the
    output is bounded by ``-t`` (the window over the rate) rather than ``-shortest``: a soundtrack shorter than
    the video must not truncate the clip, and one longer must not extend a live stream past its window in a
    silent tail.

    ``audio_inputs`` is for a never-merged game, whose window may cross a clip join: one ``(path, local_start,
    local_duration)`` per clip the window touches, each an ordinary seek inside its own clip. One piece maps
    like the single source; several are stitched by ffmpeg's ``concat`` filter (video is piped, so only the
    sound is chained) and then retimed as a whole, so a join costs no second encoder and no time jump.
    """
    quality = encoder_quality_args(encoder, live=live)
    pieces = list(audio_inputs or [])
    if not pieces and audio and source is not None:
        legacy_duration = 0.0 if audio_duration_s is None else float(audio_duration_s)
        pieces = [(str(source), float(audio_start_s), legacy_duration)]
    has_audio = bool(pieces)
    output_rate = max(1.0, float(fps) * float(rate))
    gop = max(1, int(round(output_rate * keyframe_s)))
    finishing = (
        ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", output]
        if live
        else ["-movflags", "+faststart", "-f", "mp4", output]
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{output_rate:g}",
        "-i", "pipe:0",
    ]
    if has_audio:
        for path, local_start, local_duration in pieces:
            command += ["-ss", f"{float(local_start):.3f}"]
            # Per-input bounds exist to grab the right slice of each clip. The legacy single-source call has
            # always left this to the output ``-t``; keep that byte-for-byte, so only real clip sets bound here.
            if len(pieces) > 1 and local_duration:
                command += ["-t", f"{max(0.05, float(local_duration)):.3f}"]
            command += ["-i", str(path)]
    command += ["-map", "0:v:0"]
    if has_audio:
        if len(pieces) > 1:
            chains = "".join(f"[{index}:a:0]" for index in range(1, len(pieces) + 1))
            chain = f"{chains}concat=n={len(pieces)}:v=0:a=1"
            if abs(float(rate) - 1.0) > 1e-6:
                chain += f",{_atempo_chain(rate)}"
            command += ["-filter_complex", f"{chain}[aout]", "-map", "[aout]"]
        else:
            command += ["-map", "1:a:0?"]
            if abs(float(rate) - 1.0) > 1e-6:
                command += ["-filter:a", _atempo_chain(rate)]
        command += ["-c:a", "aac", "-b:a", "128k"]
        if audio_duration_s is not None:
            command += ["-t", f"{max(0.1, float(audio_duration_s) / max(0.01, float(rate))):.3f}"]
    else:
        command += ["-an"]
    command += [
        "-c:v", encoder, *quality, "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop),
    ]
    if "265" in encoder or "hevc" in encoder:
        # ffmpeg tags HEVC as hev1 by default; QuickTime and Safari only accept hvc1, and it costs nothing.
        command += ["-tag:v", "hvc1"]
    command += finishing
    return command


class _StderrTail:
    """The last lines an encoder printed, drained on a thread so a chatty process cannot block on a full pipe."""

    def __init__(self, stream, limit: int = 20):
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=limit)
        self._thread = threading.Thread(target=self._drain, name="match-video-stderr", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode(errors="replace").strip()
                if line:
                    self._lines.append(line)
        except (ValueError, OSError):
            pass

    def text(self) -> str:
        return " | ".join(self._lines)

    def close(self) -> None:
        with suppress(OSError, ValueError):
            self._stream.close()


def _chain(first, rest):
    """The first frame back in front of the rest of the reader."""
    yield first
    yield from rest


def _start(
    encoder: str,
    *,
    fps: float,
    width: int,
    height: int,
    output: str,
    live: bool,
    keyframe_s: float,
    rate: float = 1.0,
    source: str | Path | None = None,
    audio: bool = False,
    audio_start_s: float = 0.0,
    audio_duration_s: float | None = None,
    audio_inputs: list[tuple[str, float, float]] | None = None,
):
    command = encoder_command(
        width=width,
        height=height,
        fps=fps,
        output=output,
        live=live,
        encoder=encoder,
        keyframe_s=keyframe_s,
        rate=rate,
        source=source,
        audio=audio,
        audio_start_s=audio_start_s,
        audio_duration_s=audio_duration_s,
        audio_inputs=audio_inputs,
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _audio_inputs(match, audio: bool, start_s: float, duration_s: float) -> list[tuple[str, float, float]]:
    """The soundtrack pieces for the window: one entry per clip it crosses, else the single recording.

    A never-merged game's ``match.video`` is its manifest, which no ffmpeg can open; the pieces let the
    encoder take each clip's own slice (seek inside the clip - the resolver's contract) and stitch the sound
    with ffmpeg's ``concat`` filter, so the audio is continuous across a join. A simulated match has no file
    behind it and gets none.
    """
    if not audio:
        return []
    try:
        resolved = as_source(match.video)
    except OSError:
        return []
    if isinstance(resolved, FileSource):
        video = Path(match.video)
        return [(str(video), float(start_s), float(duration_s))] if video.exists() else []
    pieces = [
        (piece.path, piece.start_s, piece.duration_s)
        for piece in window_segments(resolved, start_s, duration_s)
        if Path(piece.path).exists()
    ]
    # The concat filter is not optional the way ``1:a:0?`` is: it must not be asked for streams that do not
    # exist, or the whole encode dies. One camera's clips share their format, so the first piece decides.
    if len(pieces) > 1 and not probe_video(pieces[0][0]).has_audio:
        return []
    return pieces


def direct_audio_pieces(source, start_s: float, duration_s: float) -> list:
    """The soundtrack slices for the direct path, or ``[]`` when the recording has no usable audio.

    One camera's clips share their format, so the first containing clip decides whether they all carry sound.
    The direct command maps the same inputs as the video, so a partial soundtrack would misalign every later
    input - silence is the honest answer then, not sound that drifts out of step with the picture.
    """
    try:
        pieces = [
            piece
            for piece in window_segments(source, start_s, duration_s)
            if Path(piece.path).exists()
        ]
        if not pieces:
            return []
        if not probe_video(pieces[0].path).has_audio:
            return []
    except (OSError, FFmpegError):
        return []
    return pieces


def _direct_source_fps(match) -> float:
    """The source's own rate for sampling, or 0 when the segment did not record it.

    ``AnnotatedMatch.native_fps`` falls back to the analysis rate (5 fps on a real build), which is not a
    playback rate; treating that as the cap would turn every stream into a slideshow, so only a plausible
    source rate is used.
    """
    try:
        fps = float(match.native_fps)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return fps if fps >= 10.0 else 0.0


def _direct_pieces(match, start_s: float, duration_s: float) -> list:
    """The clip slices a direct stream reads, resolved once per request."""
    source = as_source(match.video)
    pieces = window_segments(source, max(0.0, float(start_s)), max(0.1, float(duration_s)))
    if not pieces:
        raise VideoError("the requested window has no footage")
    return pieces


def _direct_process(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def iter_direct_live_chunks(
    match,
    *,
    start_s: float,
    width: int,
    control=None,
    encoder: str | None = None,
    pace: bool = True,
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
    display_fps: float = DIRECT_DISPLAY_FPS,
):
    """Yield a fragmented MP4 read straight from the recording - no decoded frame ever enters Python.

    The direct sibling of :func:`iter_live_chunks`, used when nothing has to be drawn on the picture. ffmpeg
    seeks each clip itself, samples only the frames the output rate needs, retimes video and audio, and encodes
    in the same process. ``-readrate`` makes it produce the stream at the requested speed, so the viewer stays
    near the live edge instead of buffering the whole window; ``control`` ends it the same way the MJPEG stream
    ends (the pane's stop token).
    """
    end_s = match.start_s + match.frame_count / match.fps
    duration_s = max(0.1, end_s - float(start_s))
    pieces = _direct_pieces(match, start_s, duration_s)
    resolved_audio = direct_audio_pieces(match.video, max(0.0, float(start_s)), duration_s) if audio else []
    chosen = encoder or pick_encoder(codec)
    # NVENC can consume CUDA frames directly, so the preferred path decodes, scales and encodes entirely on the
    # GPU. A driver that cannot open the stream falls back to the copied-frame path before any byte is sent.
    candidates = ["cuda", "auto"] if chosen.endswith("nvenc") else ["auto"]
    last_error = "the direct encoder produced no output"
    for hwaccel in candidates:
        command = direct_encoder_command(
            pieces,
            display_fps=display_fps,
            rate=rate,
            width=width,
            live=True,
            encoder=chosen,
            codec=codec,
            source_fps=_direct_source_fps(match),
            audio_pieces=resolved_audio,
            pace=pace,
            hwaccel=hwaccel,
        )
        process = _direct_process(command)
        stderr = _StderrTail(process.stderr)
        produced = 0
        stopped = False
        try:
            while True:
                if control is not None and control.stopped:
                    stopped = True
                    break
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), _CHUNK_BYTES)
                if not chunk:
                    break
                produced += len(chunk)
                yield chunk
        finally:
            with suppress(OSError, ValueError):
                process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            stderr.close()
        if produced or stopped or process.returncode == 0:
            return
        last_error = stderr.text() or last_error
    raise VideoError(f"the direct encoder failed: {last_error}")


def iter_direct_mjpeg_frames(
    match,
    *,
    start_s: float,
    width: int,
    control=None,
    pace: bool = True,
    rate: float = 1.0,
    display_fps: float = DIRECT_DISPLAY_FPS,
):
    """Yield JPEGs straight from ffmpeg's MJPEG encoder, one per sampled frame.

    This is the MJPEG fallback's direct path: the server still wraps each picture in a multipart part, but the
    decode, frame-rate sampling and JPEG encode all happen inside one ffmpeg process - Python only splits the
    byte stream on the JPEG markers. ``-readrate`` paces the pictures at the requested speed.
    """
    end_s = match.start_s + match.frame_count / match.fps
    duration_s = max(0.1, end_s - float(start_s))
    pieces = _direct_pieces(match, start_s, duration_s)
    command = direct_mjpeg_command(
        pieces,
        display_fps=display_fps,
        rate=rate,
        width=width,
        source_fps=_direct_source_fps(match),
        pace=pace,
    )
    process = _direct_process(command)
    stderr = _StderrTail(process.stderr)
    buffer = bytearray()
    sent = 0
    stopped = False
    try:
        while True:
            if control is not None and control.stopped:
                stopped = True
                break
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), _CHUNK_BYTES)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                begin = buffer.find(b"\xff\xd8")
                if begin < 0:
                    # Keep a trailing 0xFF: it may be the first half of an SOI split across two reads.
                    if buffer.endswith(b"\xff"):
                        buffer[:] = b"\xff"
                    else:
                        buffer.clear()
                    break
                if begin:
                    del buffer[:begin]
                end = buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    break  # the rest of this picture is still in the pipe
                jpeg = bytes(buffer[: end + 2])
                del buffer[: end + 2]
                sent += 1
                yield jpeg
    finally:
        with suppress(OSError, ValueError):
            process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        stderr.close()
    if sent == 0 and not stopped and process.returncode not in (0, None):
        raise VideoError(f"the direct MJPEG encoder failed: {stderr.text()}")


def _first_frame(reader):
    frames = reader.frames()
    first = next(frames, None)
    if first is None:
        raise VideoError("no frames decoded for the requested window")
    return first, frames


def encode_clip(
    match,
    *,
    start_s: float,
    duration_s: float,
    fps: float,
    width: int,
    overlays: dict,
    output: Path,
    encoder: str | None = None,
    reader_factory=open_reader,
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
    skip_frame: str | None = None,
    on_progress=None,
) -> int:
    """Encode ``[start_s, start_s + duration_s]`` of the annotated match into ``output``; returns frame count.

    The frame size is taken from the first decoded frame rather than computed: the height follows the source's
    aspect, and the reader owns that decision - a raw-video pipe that disagrees with the frames it receives
    corrupts the encode. Encoding to a file means nothing has to drain the encoder's output, so this is a
    straight loop: decode, draw, write. The match's own audio rides along when the recording is there.

    ``skip_frame`` (a decoder input option, "nokey") is for the whole-game navigator videos, where one picture
    per second is enough and decoding every frame would take the footage's own length. Only readers that accept
    it are given it.
    """
    encoder = encoder or pick_encoder(codec)
    reader_kwargs: dict = {"fps": fps, "width": width, "start_s": start_s, "duration_s": duration_s}
    if skip_frame:
        reader_kwargs["skip_frame"] = skip_frame
    reader = reader_factory(match.video, **reader_kwargs)
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder,
        fps=fps,
        width=actual_width,
        height=height,
        output=str(output),
        live=False,
        keyframe_s=2.0,
        rate=rate,
        audio_inputs=_audio_inputs(match, audio, start_s, duration_s),
        audio_duration_s=duration_s,
    )
    stderr = _StderrTail(process.stderr)
    written = 0
    try:
        for timestamp, frame in _chain(first, rest):
            match.render_at(frame, timestamp, **overlays)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
            if on_progress is not None:
                on_progress(float(np.clip((timestamp - start_s) / max(duration_s, 1e-6), 0.0, 1.0)))
        process.stdin.close()
        code = process.wait()
    except (BrokenPipeError, OSError) as error:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise VideoError(f"the encoder stopped early: {stderr.text() or error}") from error
    finally:
        stderr.close()
    if code != 0:
        output.unlink(missing_ok=True)
        raise VideoError(f"ffmpeg exited with code {code}: {stderr.text()}")
    return written


def iter_live_chunks(
    match,
    *,
    start_s: float,
    fps: float,
    width: int,
    overlays: dict,
    control=None,
    encoder: str | None = None,
    reader_factory=open_reader,
    pace: bool = True,
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
):
    """Yield the annotated match as a fragmented MP4, chunk by chunk, until the window ends or ``control`` stops.

    With no overlay layers the whole picture is the recording itself, so the direct path takes over: ffmpeg
    seeks the source clips, samples only the frames the output rate needs, keeps the soundtrack and encodes in
    one process (see :func:`iter_direct_live_chunks`). With any layer on, a writer thread renders and feeds the
    encoder as before; this generator drains the encoder's output - and so paces the pipeline: nothing is
    produced much faster than the client reads it, and the writer's own pacing keeps the stream near real time.

    ``fps`` is the source frames sampled a second. Above 1x it is reduced in step with ``rate`` so the output
    timeline stays at :data:`DIRECT_DISPLAY_FPS` rather than growing to 120/240 fps at 4x/8x.
    ``control`` is the MJPEG stream's ``StreamControl``, so ``/stop`` ends both kinds of stream the same way.
    """
    if not any((overlays or {}).values()) and reader_factory is open_reader:
        try:
            as_source(match.video)
        except (OSError, ValueError):
            pass  # a missing source falls through to the reader, whose error message is the better one
        else:
            yield from iter_direct_live_chunks(
                match,
                start_s=start_s,
                width=width,
                control=control,
                encoder=encoder,
                pace=pace,
                rate=rate,
                audio=audio,
                codec=codec,
            )
            return
    encoder = encoder or pick_encoder(codec)
    sample_fps = float(fps)
    if rate > 1.0:
        # Only the frames a viewer can see reach the encoder: 2x still needs 30 source fps for 60 output, 8x
        # needs 7.5. Leaving the old fps*rate arithmetic here would ask for 120 fps at 4x and 240 at 8x.
        sample_fps = max(1.0, min(sample_fps, DIRECT_DISPLAY_FPS / float(rate)))
    end_s = match.start_s + match.frame_count / match.fps
    stop = control if control is not None else _NEVER_STOPPED
    reader = reader_factory(
        match.video, fps=sample_fps, width=width, start_s=start_s, duration_s=max(0.1, end_s - start_s)
    )
    first, rest = _first_frame(reader)
    _timestamp, frame = first
    height, actual_width = frame.shape[:2]
    process = _start(
        encoder,
        fps=sample_fps,
        width=actual_width,
        height=height,
        output="pipe:1",
        live=True,
        keyframe_s=1.0,
        rate=rate,
        audio_inputs=_audio_inputs(match, audio, start_s, max(0.1, end_s - start_s)),
        audio_duration_s=max(0.1, end_s - start_s),
    )
    stderr = _StderrTail(process.stderr)
    failures: list[Exception] = []

    def produce() -> None:
        period = 1.0 / max(0.01, sample_fps * max(0.01, rate))  # the muxer's clock runs fps*rate per source second
        next_frame_at = time.monotonic()
        try:
            for timestamp, frame in _chain(first, rest):
                if stop.stopped:
                    break
                match.render_at(frame, timestamp, **overlays)
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                if pace:
                    next_frame_at += period
                    delay = next_frame_at - time.monotonic()
                    if delay > 0.0:
                        time.sleep(delay)
                    else:
                        next_frame_at = time.monotonic()  # the decode is the cap; just re-anchor
        except (BrokenPipeError, OSError, ValueError) as error:
            failures.append(error)  # reported below; the reader loop is the one that decides the response
        finally:
            with suppress(OSError, ValueError):
                process.stdin.close()

    writer = threading.Thread(target=produce, name="match-video-writer", daemon=True)
    writer.start()
    try:
        while True:
            if stop.stopped:
                break
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), _CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        writer.join(timeout=5.0)
        stderr.close()
    if failures and not stop.stopped:
        raise VideoError(f"the encoder failed: {stderr.text() or failures[0]}")


def clip_key(
    match_id: str,
    start_s: float,
    duration_s: float,
    fps: float,
    width: int,
    overlays: dict,
    rate: float = 1.0,
    audio: bool = True,
    codec: str = "h264",
) -> str:
    """The identity of an encode: any difference in any of these is a different file."""
    layers = ",".join(sorted(name for name, on in overlays.items() if on))
    return (
        f"{match_id}-{start_s:.2f}-{duration_s:.2f}-{fps:g}x{rate:g}-{width}-{codec}"
        f"-{layers or 'none'}-{'aud' if audio else 'mute'}"
    )


def _file_name(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in key)[:80]
    return f"{safe}-{digest}.mp4"


class ClipCache:
    """Finished clips on disk, kept for a while because a player asks for the same clip more than once.

    A clip request blocks for the whole encode, so the file it produces is worth keeping: the player comes back
    for byte ranges as the viewer scrubs, and a reload asks again. Entries expire by age and by count and their
    files are deleted then; an encode that is still running holds its key (see :meth:`claim`), so two requests
    for one clip never write the same file.
    """

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        ttl_s: float = CLIP_CACHE_TTL_S,
        max_entries: int = CLIP_CACHE_MAX,
    ):
        self.directory = (
            Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="match-video-"))
        )
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._entries: dict[str, tuple[Path, float]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def lookup(self, key: str) -> Path | None:
        """The finished clip for a key, if it is still there; expired entries are dropped as they are noticed."""
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            entry = self._entries.get(key)
        if entry is None:
            return None
        path, _stored_at = entry
        if not path.exists():
            with self._lock:
                self._entries.pop(key, None)
            return None
        return path

    def target(self, key: str) -> Path:
        """Where the encode for a key should be written (not visible to lookups until :meth:`keep`)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / _file_name(key)

    def keep(self, key: str, path: Path) -> None:
        now = time.monotonic()
        with self._lock:
            self._entries[key] = (path, now)
            self._purge(now)

    @contextmanager
    def claim(self, key: str):
        with self._lock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            yield

    def _purge(self, now: float) -> None:
        expired = [key for key, (_path, stored_at) in self._entries.items() if now - stored_at > self.ttl_s]
        for key in expired:
            self._drop(key)
        if len(self._entries) > self.max_entries:
            oldest = sorted(self._entries.items(), key=lambda item: item[1][1])
            for key, _entry in oldest[: len(self._entries) - self.max_entries]:
                self._drop(key)

    def _drop(self, key: str) -> None:
        path, _stored_at = self._entries.pop(key, (None, 0.0))
        if path is not None:
            path.unlink(missing_ok=True)
        self._locks.pop(key, None)

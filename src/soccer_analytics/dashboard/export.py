"""Export the full match video: raw passthrough-style re-encode, or the annotated renderer.

The default export is the match as filmed. That path must not decode 4K through Python frame by frame for an
80-minute game, so it is a single ffmpeg filter graph: seek each clip to its slice, scale/fps once, concat, and
encode with the requested codec. The overlay export is the expensive one and uses the same ``AnnotatedMatch``
renderer as the live stream, so the exported picture and the stream cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

from soccer_analytics.dashboard.video import encoder_quality_args, pick_encoder
from soccer_analytics.ingest.ffmpeg_reader import probe_video, run_ffmpeg_with_progress
from soccer_analytics.ingest.source import WindowPiece, probe_source, window_segments

# Codec names used by the export API. ``hevc`` is what ffmpeg calls H.265; the UI spells it H.265/HEVC.
EXPORT_CODECS = ("h264", "hevc")


class ExportError(RuntimeError):
    """The export could not be produced (bad source, missing stream, ffmpeg failure)."""


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def target_size(source_width: int, source_height: int, width: int) -> tuple[int, int]:
    """The requested width and the even height that preserves the source aspect."""
    target_width = _even(width)
    return target_width, _even(target_width * source_height / max(1, source_width))


def export_command(
    pieces: list[WindowPiece],
    *,
    source_width: int,
    source_height: int,
    duration_s: float,
    width: int,
    fps: float,
    codec: str,
    output: Path,
    has_audio: bool,
) -> list[str]:
    """The direct ffmpeg command for an unannotated window.

    Split out from :func:`export_source_window` so the filter graph is testable without running ffmpeg. Every
    input is trimmed before decoding, then the video (and, when present, the sound) is scaled/retimed and
    concatenated. A one-clip window skips the concat filter; a game window crossing clip joins uses it.
    """
    if codec not in EXPORT_CODECS:
        raise ExportError(f"unsupported codec {codec!r}; expected one of {EXPORT_CODECS}")
    if not pieces:
        raise ExportError("the requested window intersects no footage")
    target_width, target_height = target_size(source_width, source_height, width)
    inputs: list[str] = []
    for piece in pieces:
        inputs += [
            "-ss", f"{piece.start_s:.3f}",
            "-t", f"{piece.duration_s:.3f}",
            "-i", str(piece.path),
        ]

    video_filter = f"scale={target_width}:{target_height}:flags=lanczos,setsar=1,fps={fps:g},setpts=PTS-STARTPTS"
    audio_filter = (
        "aresample=async=1:first_pts=0,"
        "aformat=sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *inputs]
    if len(pieces) == 1:
        command += ["-vf", video_filter, "-map", "0:v:0"]
        if has_audio:
            command += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
        else:
            command += ["-an"]
    else:
        filters = [f"[{index}:v]{video_filter}[v{index}]" for index in range(len(pieces))]
        if has_audio:
            filters += [f"[{index}:a]{audio_filter}[a{index}]" for index in range(len(pieces))]
        filters.append(
            "".join(f"[v{index}]" for index in range(len(pieces)))
            + f"concat=n={len(pieces)}:v=1:a=0[vout]"
        )
        if has_audio:
            filters.append(
                "".join(f"[a{index}]" for index in range(len(pieces)))
                + f"concat=n={len(pieces)}:v=0:a=1[aout]"
            )
        command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
        if has_audio:
            command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "128k"]
        else:
            command += ["-an"]

    encoder = pick_encoder(codec)
    command += [
        "-c:v", encoder,
        *encoder_quality_args(encoder),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if codec == "hevc":
        command += ["-tag:v", "hvc1"]
    command += [
        "-t", f"{duration_s:.3f}",
        "-progress", "pipe:1",
        "-nostats",
        str(output),
    ]
    return command


def export_source_window(
    source: str | Path,
    start_s: float,
    duration_s: float,
    output: str | Path,
    *,
    width: int,
    fps: float,
    codec: str,
    on_progress=None,
) -> int:
    """Export ``[start_s, start_s + duration_s]`` of the source with no overlays; returns the frame estimate.

    ``source`` may be a single recording or a ``game.json`` manifest. The manifest is split into per-clip slices
    by :func:`window_segments`, so a whole-game export seeks each camera file once instead of decoding a
    never-merged game's clips sequentially through Python.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pieces = window_segments(source, start_s, duration_s)
    if not pieces:
        raise ExportError("the requested window intersects no footage")
    probe = probe_source(source)
    # One camera's clips share their format, but an archive can mix a silent camera file in. The concat filter
    # cannot be asked for audio that one input does not have, so fall back to a clean video-only export.
    has_audio = probe.has_audio and all(probe_video(piece.path).has_audio for piece in pieces)
    command = export_command(
        pieces,
        source_width=probe.width,
        source_height=probe.height,
        duration_s=duration_s,
        width=width,
        fps=fps,
        codec=codec,
        output=output,
        has_audio=has_audio,
    )
    returncode, tail = run_ffmpeg_with_progress(command, duration_s, on_progress)
    if returncode != 0 or not output.exists() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise ExportError(f"ffmpeg could not export the match: {tail or f'exit {returncode}'}")
    return int(round(duration_s * fps))

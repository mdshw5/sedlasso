"""The direct match-video export command: the filter graph, codec choice and audio fallback.

The annotated path is the stream renderer and is covered by the stream tests; these pin the fast path that must
not accidentally decode an 80-minute game through Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soccer_analytics.dashboard.export import ExportError, export_command, target_size
from soccer_analytics.ingest.source import WindowPiece


def _pieces(count: int = 2) -> list[WindowPiece]:
    return [WindowPiece(path=f"clip_{index}.mp4", start_s=2.0 * index, duration_s=10.0) for index in range(count)]


def test_target_size_preserves_aspect_and_forces_even_dimensions() -> None:
    assert target_size(3840, 2160, 1920) == (1920, 1080)
    assert target_size(3840, 2160, 1280) == (1280, 720)
    # A non-16:9 source keeps its shape; odd heights are rounded to an even number for yuv420p.
    width, height = target_size(1920, 803, 960)
    assert width == 960 and height % 2 == 0 and abs(height - 402) <= 1


def test_a_multi_clip_export_uses_one_filter_graph_to_concat_video_and_audio() -> None:
    command = export_command(
        _pieces(3),
        source_width=3840,
        source_height=2160,
        duration_s=30.0,
        width=1920,
        fps=30,
        codec="hevc",
        output=Path("export.mp4"),
        has_audio=True,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "scale=1920:1080" in graph
    assert "fps=30" in graph
    assert "concat=n=3:v=1:a=0[vout]" in graph
    assert "concat=n=3:v=0:a=1[aout]" in graph
    assert "aformat=sample_rates=48000:channel_layouts=stereo" in graph
    assert command.count("-i") == 3
    assert "-tag:v" in command and command[command.index("-tag:v") + 1] == "hvc1"
    assert command[command.index("-c:v") + 1]  # encoder was resolved
    assert command[-1].endswith("export.mp4")


def test_a_single_clip_export_skips_the_concat_filter() -> None:
    command = export_command(
        _pieces(1),
        source_width=1920,
        source_height=1080,
        duration_s=10.0,
        width=1280,
        fps=15,
        codec="h264",
        output=Path("export.mp4"),
        has_audio=True,
    )
    assert "-vf" in command
    assert "-filter_complex" not in command
    assert "scale=1280:720" in command[command.index("-vf") + 1]
    assert "-map" in command and "0:a:0?" in command


def test_a_silent_source_is_exported_video_only() -> None:
    command = export_command(
        _pieces(2),
        source_width=1920,
        source_height=1080,
        duration_s=20.0,
        width=1920,
        fps=30,
        codec="h264",
        output=Path("export.mp4"),
        has_audio=False,
    )
    assert "-an" in command
    assert "-c:a" not in command


def test_an_unknown_codec_is_refused() -> None:
    with pytest.raises(ExportError):
        export_command(
            _pieces(1),
            source_width=1920,
            source_height=1080,
            duration_s=10.0,
            width=1920,
            fps=30,
            codec="av1",
            output=Path("export.mp4"),
            has_audio=False,
        )

"""The direct match-video export command: the filter graph, codec choice and audio fallback.

The annotated path is the stream renderer and is covered by the stream tests; these pin the fast path that must
not accidentally decode an 80-minute game through Python.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from soccer_analytics.dashboard.export import (
    ExportError,
    export_command,
    preview_stream_url,
    target_size,
)
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


def test_a_preview_url_maps_every_export_overlay_and_keeps_audio() -> None:
    """The preview must exercise the same layer set the export will render: team boxes become the stream's
    ``boxes``, the scoreboard rides its own flag, and the soundtrack stays on. H.264 is deliberate - the file
    may be HEVC, but browsers cannot be assumed to decode that."""
    url = preview_stream_url(
        "2026-10-03_game",
        base="http://match-host:8510/",
        start_s=12.5,
        width=1920,
        fps=30,
        team=True,
        scoreboard=True,
        ball=True,
        numbers=True,
        nonce=2,
    )
    parsed = urlparse(url)
    assert parsed.scheme == "http" and parsed.netloc == "match-host:8510"
    assert parsed.path == "/live/2026-10-03_game.mp4"
    query = parse_qs(parsed.query)
    assert query["start"] == ["12.5"]
    assert query["fps"] == ["30"]
    assert query["width"] == ["1920"]
    assert query["overlays"] == ["boxes,numbers,ball"]
    assert query["scoreboard"] == ["1"]
    assert query["audio"] == ["1"]
    assert query["codec"] == ["h264"]
    assert query["_preview"] == ["2"], "the restart button must force a fresh player URL"


def test_a_clean_preview_asks_for_no_layers_and_no_scoreboard() -> None:
    url = preview_stream_url(
        "m", base="http://match-host:8510", start_s=0.0, width=1280, fps=60
    )
    query = parse_qs(urlparse(url).query)
    assert query["overlays"] == ["none"]
    assert "scoreboard" not in query
    assert query["audio"] == ["1"], "the export always carries the match audio when the source has it"
    assert query["codec"] == ["h264"]

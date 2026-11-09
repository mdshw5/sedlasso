"""Timeline scrubber: the time <-> frame mapping and the cached seek-friendly proxy.

The mapping is the part that is easy to get silently wrong (a frame that is off by a few is still a plausible
picture), and the proxy build is the part that has to leave the cache in a state the dashboard can trust.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest

from soccer_analytics.dashboard import timeline
from soccer_analytics.ingest.video_reader import VideoWriter


@pytest.fixture()
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    width, height, fps, frames = 96, 64, 10.0, 40
    with VideoWriter(path, fps=fps, width=width, height=height) as writer:
        for i in range(frames):
            frame = np.full((height, width, 3), fill_value=(i * 5) % 256, dtype=np.uint8)
            writer.write(frame)
    return path


def test_frame_time_clamps_and_indexes() -> None:
    times = np.array([0.0, 0.2, 0.4, 0.6])
    assert timeline.frame_time(times, 0) == 0.0
    assert timeline.frame_time(times, 2) == pytest.approx(0.4)
    assert timeline.frame_time(times, 99) == pytest.approx(0.6), "a frame past the end is clamped, not an error"
    assert timeline.frame_time(times, -5) == 0.0


def test_frame_time_rejects_an_empty_segment() -> None:
    with pytest.raises(ValueError):
        timeline.frame_time(np.array([]), 0)


def test_nearest_frame_index_snaps_to_the_closest_time() -> None:
    times = np.array([0.0, 0.2, 0.4, 0.6])
    assert timeline.nearest_frame_index(times, 0.31) == 2
    assert timeline.nearest_frame_index(times, 10.0) == 3
    assert timeline.nearest_frame_index(times, -1.0) == 0
    assert timeline.nearest_frame_index(np.array([]), 5.0) == 0


def test_scrubber_assets_live_under_the_segment(tmp_path) -> None:
    segment = tmp_path / "seg_123"
    assert timeline.scrubber_dir(segment) == segment / "scrubber"
    assert timeline.proxy_path(segment) == segment / "scrubber" / "proxy.mp4"
    assert not timeline.proxy_is_ready(segment)


def test_proxy_readiness_is_about_the_proxy_alone(tmp_path) -> None:
    segment = tmp_path / "seg"
    # The component is a single static folder now, not a per-segment copy, so readiness is only about the video.
    assert not timeline.proxy_is_ready(segment)
    timeline.scrubber_dir(segment).mkdir(parents=True)
    timeline.proxy_path(segment).write_bytes(b"not really a video")
    assert timeline.proxy_is_ready(segment)


def test_build_proxy_writes_a_seekable_mp4(tmp_path, sample_video) -> None:
    segment = tmp_path / "seg"
    proxy = timeline.build_proxy(sample_video, segment, start_s=0.0, duration_s=4.0, width=64, prefer_gpu=False)
    assert proxy.exists() and proxy.stat().st_size > 0
    assert timeline.proxy_is_ready(segment)
    # A half-written temporary must not be left behind next to the finished proxy.
    assert not proxy.with_name(proxy.name + ".tmp.mp4").exists()


def test_long_windows_are_built_from_keyframes_alone() -> None:
    """Decoding 4K60 is the whole cost of a proxy; keyframes-only measured 30-50x faster than every frame."""
    assert timeline.proxy_skip_frame(60.0) is None
    assert timeline.proxy_skip_frame(timeline.KEYFRAME_ONLY_AFTER_S) == timeline.KEYFRAME_SKIP
    assert timeline.proxy_skip_frame(80 * 60) == timeline.KEYFRAME_SKIP

    command = timeline.proxy_command(
        "in.mp4",
        "out.mp4",
        start_s=0.0,
        duration_s=4800.0,
        width=640,
        height=360,
        fps=4.0,
        gop=4,
        skip_frame=timeline.KEYFRAME_SKIP,
        use_gpu=True,
    )
    assert command[command.index("-skip_frame") + 1] == "nokey"
    # It is an input option: it has to come before the input it applies to.
    assert command.index("-skip_frame") < command.index("-i")


def test_picture_times_parse_drop_junk_and_dedupe() -> None:
    """ffprobe prints one timestamp per keyframe; blanks, N/A and duplicates must not become overlay samples."""
    output = "0.000000\n1.000000\nN/A\n\n1.000000\n2.500000\n-1.000000\n"
    assert timeline.parse_picture_times(output) == [0.0, 1.0, 2.5]


def test_picture_times_need_a_proxy(tmp_path) -> None:
    assert timeline.proxy_picture_times(tmp_path) is None


def test_picture_times_are_cached_until_the_proxy_changes(tmp_path) -> None:
    """Probing a whole game's proxy must happen once; a rebuilt proxy must be probed again."""
    segment = tmp_path / "segment"
    timeline.scrubber_dir(segment).mkdir(parents=True)
    proxy = timeline.proxy_path(segment)
    proxy.write_bytes(b"not really a video")
    stamp = proxy.stat().st_mtime_ns
    timeline.pictures_path(segment).write_text(json.dumps({"proxy_mtime_ns": stamp, "times": [1.5, 2.5]}))
    assert timeline.proxy_picture_times(segment) == [1.5, 2.5]
    # A new proxy mtime invalidates the cache; the fake file cannot be probed, so the probe fails rather than
    # serving the stale times.
    os.utime(proxy, ns=(stamp + 1_000_000, stamp + 1_000_000))
    assert timeline.proxy_picture_times(segment) is None


def test_a_short_proxy_decodes_every_frame() -> None:
    command = timeline.proxy_command(
        "in.mp4",
        "out.mp4",
        start_s=0.0,
        duration_s=300.0,
        width=640,
        height=360,
        fps=4.0,
        gop=4,
        skip_frame=None,
        use_gpu=True,
    )
    assert "-skip_frame" not in command


def test_build_proxy_accepts_keyframe_only_decoding(tmp_path, sample_video) -> None:
    segment = tmp_path / "seg"
    proxy = timeline.build_proxy(
        sample_video,
        segment,
        start_s=0.0,
        duration_s=4.0,
        width=64,
        prefer_gpu=False,
        skip_frame=timeline.KEYFRAME_SKIP,
    )
    assert proxy.exists() and proxy.stat().st_size > 0


def test_build_proxy_reports_encode_progress(tmp_path, sample_video) -> None:
    """The dashboard's building bar is fed from here; a build that never reports is a spinner for minutes."""
    segment = tmp_path / "seg"
    calls: list[float] = []
    timeline.build_proxy(
        sample_video,
        segment,
        start_s=0.0,
        duration_s=4.0,
        width=64,
        prefer_gpu=False,
        on_progress=lambda fraction: calls.append(float(fraction)),
    )
    assert calls and calls[-1] == 1.0, "the finished proxy must report 100%"
    assert all(0.0 <= call <= 1.0 for call in calls) and calls == sorted(calls)


def test_build_state_round_trips_and_merges(tmp_path) -> None:
    segment = tmp_path / "seg"
    assert timeline.read_build_state(segment) == {}
    timeline.write_build_state(segment, state="running", pid=1234)
    assert timeline.read_build_state(segment)["pid"] == 1234
    timeline.write_build_state(segment, state="done", error=None)
    state = timeline.read_build_state(segment)
    assert state["state"] == "done" and state["pid"] == 1234, "a later write must merge, not replace"
    # A corrupt state file is treated as "no idea", never as an exception.
    timeline.build_state_path(segment).write_text("{not json")
    assert timeline.read_build_state(segment) == {}


def test_a_finished_process_is_not_alive(tmp_path) -> None:
    finished = subprocess.Popen(["true"])
    finished.wait()
    assert timeline.process_alive(os.getpid())
    assert not timeline.process_alive(finished.pid)
    assert not timeline.process_alive(None)


def test_build_in_progress_needs_a_live_process(tmp_path) -> None:
    segment = tmp_path / "seg"
    timeline.write_build_state(segment, state="running", pid=os.getpid())
    assert timeline.build_in_progress(segment), "we are alive, so the build is treated as running"
    timeline.write_build_state(segment, state="running", pid=2_000_000_000)
    assert not timeline.build_in_progress(segment), "a dead pid means the build must be retried, not waited on"


def test_start_background_build_is_a_no_op_when_the_proxy_is_ready(tmp_path, sample_video) -> None:
    segment = tmp_path / "seg"
    timeline.build_proxy(sample_video, segment, start_s=0.0, duration_s=4.0, width=64, prefer_gpu=False)
    assert timeline.start_background_build(segment, sample_video, start_s=0.0, duration_s=4.0) is False

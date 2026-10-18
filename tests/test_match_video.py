"""The encoded video routes: full-rate rendering, the H.264 pipelines, and the clip cache.

Two kinds of test live here. The interpolation ones need no video at all - they feed an ``AnnotatedMatch``
hand-built frames and check that a position between two analysis samples draws *between* them. The encoder
ones fall into a fast group (command construction, the clip cache, the routes with injected encoders - no
ffmpeg) and a small integration group that really encodes a handful of tiny frames with x264 and probes the
result; those skip when the machine has no ffmpeg.
"""

from __future__ import annotations

import http.client
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.dashboard.stream import (
    AnnotatedMatch,
    MatchStreamServer,
    OVERLAY_NAMES,
    StreamControl,
    TeamStyle,
    iter_annotated_frames,
    slerp_rotation,
)
from soccer_analytics.dashboard.video import (
    ClipCache,
    VideoError,
    clip_key,
    direct_encoder_command,
    direct_mjpeg_command,
    direct_sample_fps,
    encode_clip,
    encoder_command,
    iter_live_chunks,
    pick_encoder,
)
from soccer_analytics.ingest.source import WindowPiece

NO_LAYERS = {name: False for name in OVERLAY_NAMES}


def _match(
    *,
    boxes_by_frame: dict[int, list[tuple]] | None = None,
    ball_records=(),
    q=None,
    focal=None,
    frame_count: int = 100,
    native: tuple[int, int, float] = (1920, 1080, 30.0),
    video: Path | None = None,
    start_s: float = 100.0,
) -> AnnotatedMatch:
    """A match assembled from hand-written observations - the leanest way to test interpolation."""
    tracks: dict[int, tuple[list[int], list[list[float]]]] = {}
    for frame, entries in (boxes_by_frame or {}).items():
        for x1, y1, x2, y2, track_id, team in entries:
            frames, positions = tracks.setdefault(track_id, ([], []))
            frames.append(frame)
            positions.append([x1, y1, x2, y2])
    players = [
        {"track_id": track_id, "team": 0, "frames": frames} for track_id, (frames, _positions) in tracks.items()
    ]
    boxes = {track_id: np.asarray(positions, dtype=np.float64) for track_id, (_f, positions) in tracks.items()}
    return AnnotatedMatch(
        match_id="synthetic",
        video=Path(video) if video is not None else Path("/nonexistent/synthetic.mp4"),
        fps=5.0,
        start_s=start_s,
        frame_count=frame_count,
        native=native,
        pitch=(105.0, 68.0),
        calibration=None,
        q=np.asarray(q) if q is not None else np.tile(np.eye(3), (frame_count, 1, 1)),
        focal=np.asarray(focal, dtype=np.float64) if focal is not None else np.ones(frame_count),
        players=players,
        boxes=boxes,
        numbers={},
        numbers_note=None,
        ball_records=list(ball_records),
        teams=[TeamStyle(name="A", bgr=(11, 22, 33)), TeamStyle(name="B", bgr=(44, 55, 66))],
        notes=[],
    )


# --------------------------------------------------------------------------------------------------------------
# Interpolation: a frame between two analysis samples
# --------------------------------------------------------------------------------------------------------------
def test_a_player_box_interpolates_between_the_samples_around_it() -> None:
    """The whole point of the full-rate render: at 60 fps, eleven of every twelve frames are between samples,
    and a box held back to the previous sample would lag the footage visibly during any sprint."""
    match = _match(
        boxes_by_frame={
            0: [(0.10, 0.20, 0.30, 0.40, 7, 0)],
            1: [(0.20, 0.30, 0.40, 0.50, 7, 0)],
        }
    )
    halfway = match.players_at(0.5)
    assert len(halfway) == 1
    assert halfway[0][:4] == pytest.approx((0.15, 0.25, 0.35, 0.45))
    assert halfway[0][4:] == (7, 0)


def test_a_box_holds_through_a_gap_and_a_track_does_not_appear_before_its_first_sample() -> None:
    """A player briefly unobserved keeps their last box rather than teleporting; a track whose first sample is
    still ahead is not drawn yet - the same appearance either has at the analysis rate."""
    holds = _match(boxes_by_frame={0: [(0.10, 0.10, 0.20, 0.20, 7, 0)], 1: []})
    assert holds.players_at(0.5)[0][:4] == pytest.approx((0.10, 0.10, 0.20, 0.20))
    late = _match(boxes_by_frame={0: [], 1: [(0.30, 0.30, 0.40, 0.40, 7, 0)]})
    assert late.players_at(0.5) == [], "the track appears when its sample arrives, not before"


def test_the_ball_interpolates_between_two_sightings_and_is_absent_everywhere_else() -> None:
    """The dot moves between two neighboring sightings - and a forecast is never drawn: where a coast interrupts
    the track, or the scan has not seen the ball yet, the picture carries no ball mark at all."""
    match = _match(
        ball_records=[
            {"i": 0, "status": "tracking", "u": 0.2, "v": 0.4},
            {"i": 1, "status": "tracking", "u": 0.6, "v": 0.8},
        ]
    )
    assert match.ball_at(0.4) == pytest.approx((0.36, 0.56, 1.0)), "between sightings the dot interpolates"
    assert match.ball_at(0.0) == pytest.approx((0.2, 0.4, 1.0)), "a sample a detector saw draws as seen"
    coasted = _match(
        ball_records=[
            {"i": 0, "status": "tracking", "u": 0.2, "v": 0.4},
            {"i": 1, "status": "coasting", "u": 0.6, "v": 0.8},
        ]
    )
    assert not np.isfinite(coasted.ball_at(0.4)[0]), "a forecast is not a detection: not drawn"
    gap = _match(ball_records=[{"i": 0, "status": "tracking", "u": 0.2, "v": 0.4}])
    assert not np.isfinite(gap.ball_at(0.5)[0]), "nothing is held across a later gap"
    future = _match(ball_records=[{"i": 1, "status": "tracking", "u": 0.2, "v": 0.4}])
    assert not np.isfinite(future.ball_at(0.5)[0]), "not yet seen at the earlier sample"


def test_the_camera_pose_interpolates_along_the_shortest_arc() -> None:
    """The pitch markings ride the camera pose, so between samples the pose slerps - a linear blend of a rotation
    would shrink the pitch's projection exactly when the camera pans fastest."""
    angle = np.pi / 2
    quarter = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    eighth = angle / 2
    expected = np.array(
        [[np.cos(eighth), -np.sin(eighth), 0.0], [np.sin(eighth), np.cos(eighth), 0.0], [0.0, 0.0, 1.0]]
    )
    assert slerp_rotation(np.eye(3), quarter, 0.5) == pytest.approx(expected, abs=1e-9)
    match = _match(frame_count=2, q=[np.eye(3), quarter], focal=[1.0, 2.0])
    q_half, focal_half = match.pose_at(0.5)
    assert q_half == pytest.approx(expected, abs=1e-9)
    assert focal_half == pytest.approx(1.5), "the focal interpolates"


def test_render_at_draws_the_interpolated_box_not_either_sample() -> None:
    """Rendered on a bare frame so the assertion is exact: the box edge lands where the midpoint says, which is
    a column no sample's box would have drawn."""
    match = _match(
        boxes_by_frame={
            0: [(0.10, 0.10, 0.30, 0.30, 7, 0)],
            1: [(0.30, 0.10, 0.50, 0.30, 7, 0)],
        }
    )
    frame = np.full((60, 80, 3), 60, dtype=np.uint8)
    match.render_at(frame, match.start_s + 0.1, pitch=False, numbers=False, ball=False, hud=False, debug=False)
    expected = match.teams[0].bgr
    # mid-sample box: left 0.2*80 = 16, top 0.1*80 = 8, bottom 0.3*80 = 24
    assert tuple(int(v) for v in frame[(8 + 24) // 2, 16]) == expected
    cleared = np.full((60, 80, 3), 60, dtype=np.uint8)
    match.render_at(cleared, match.start_s + 0.1, **NO_LAYERS)
    assert np.array_equal(cleared, np.full((60, 80, 3), 60, dtype=np.uint8)), "all layers off draws nothing"


# --------------------------------------------------------------------------------------------------------------
# The encoder pipelines
# --------------------------------------------------------------------------------------------------------------
def test_the_live_command_fragments_and_the_clip_command_seeks() -> None:
    """The flags are the difference between the two products: a fragmented MP4 a browser can join mid-flight,
    and a faststart MP4 a player can scrub - both fed raw frames, both H.264."""
    live = encoder_command(
        width=1920, height=1080, fps=60.0, output="pipe:1", live=True, encoder="h264_nvenc", keyframe_s=1.0
    )
    assert "+frag_keyframe+empty_moov+default_base_moof" in live
    assert live[live.index("-g") + 1] == "60", "one keyframe a second is also one fragment a second"
    assert live[2:6] == ["-loglevel", "error", "-nostdin", "-y"]
    assert "rawvideo" in live and "bgr24" in live and live[-1] == "pipe:1"
    assert "ll" in live, "a live stream wants the low-latency tune"

    clip = encoder_command(
        width=3840, height=2160, fps=30.0, output="/tmp/x.mp4", live=False, encoder="libx264", keyframe_s=2.0
    )
    assert "+faststart" in clip and clip[-1] == "/tmp/x.mp4"
    assert clip[clip.index("-g") + 1] == "60"
    assert "libx264" in clip and "medium" in clip, "the CPU fallback gets the denser preset"


def test_the_direct_playback_command_samples_the_frames_each_speed_needs_and_keeps_audio() -> None:
    """The direct path reads the recording itself and gives the encoder only the frames a viewer can see:
    a constant 60 fps timeline needs 60 source fps at 1x, 30 at 2x, 15 at 4x and 7.5 at 8x. Audio rides
    along at every speed (`atempo`), and high speeds also tell the decoder to skip frames it cannot show."""
    pieces = [WindowPiece("/archive/game.mp4", 1.25, 10.0)]
    for rate, sample in ((1.0, 60.0), (2.0, 30.0), (4.0, 15.0), (8.0, 7.5)):
        command = direct_encoder_command(
            pieces,
            rate=rate,
            width=1600,
            live=True,
            encoder="h264_nvenc",
            source_fps=60.0,
            audio_pieces=pieces,
        )
        joined = " ".join(command)
        assert f"fps={sample:g}" in joined, f"{rate:g}x must sample {sample:g} source fps"
        assert f"setpts=PTS/{rate:g}" in joined, "the output timeline is retimed to the requested speed"
        assert f"-readrate {rate:g}" in joined, "the source is read at playback speed, not buffered"
        assert "-map 0:a:0?" in joined and "-c:a aac" in joined, "the match audio rides along"
        if rate > 1:
            assert "atempo" in joined, "the soundtrack is retimed with the picture"
        else:
            assert "-filter:a" not in joined, "1x audio needs no retiming"
        assert command[-1] == "pipe:1"
    assert direct_sample_fps(8.0, 60.0) == pytest.approx(7.5)
    fast = " ".join(
        direct_encoder_command(pieces, rate=8.0, source_fps=60.0, encoder="libx264", audio_pieces=pieces)
    )
    assert "-skip_frame nokey" in fast, "8x decodes only keyframes; the encoder still fills 60 fps"
    medium = " ".join(
        direct_encoder_command(pieces, rate=4.0, source_fps=60.0, encoder="libx264", audio_pieces=pieces)
    )
    assert "-skip_frame nointra" in medium, "4x cannot decode every frame in time either"
    smooth = " ".join(
        direct_encoder_command(pieces, rate=2.0, source_fps=60.0, encoder="libx264", audio_pieces=pieces)
    )
    assert "-skip_frame" not in smooth, "2x decodes everything: it keeps up and looks best"


def test_the_direct_playback_command_joins_clips_and_keeps_audio_across_the_join() -> None:
    """A never-merged game is a clip set, so the direct path seeks each clip locally and lets ffmpeg's
    ``concat`` filter stitch both the picture and the sound - no second encode, no silent seam."""
    pieces = [WindowPiece("/clips/a.mp4", 1.25, 0.75), WindowPiece("/clips/b.mp4", 0.0, 1.5)]
    command = direct_encoder_command(
        pieces,
        rate=2.0,
        width=1600,
        live=True,
        encoder="h264_nvenc",
        source_fps=60.0,
        audio_pieces=pieces,
    )
    joined = " ".join(command)
    assert "-ss 1.250 -t 0.750 -i /clips/a.mp4" in joined
    assert "-ss 0.000 -t 1.500 -i /clips/b.mp4" in joined
    assert "[0:v:0]fps=30" in joined and "[1:v:0]fps=30" in joined
    assert "concat=n=2:v=1:a=0,setpts=PTS/2[vout]" in joined
    assert "[0:a:0]aresample=async=1[a0]" in joined
    assert "concat=n=2:v=0:a=1,atempo=2.0000[aout]" in joined
    assert "-map [vout]" in joined and "-map [aout]" in joined


def test_the_direct_mjpeg_command_drops_frames_a_high_speed_cannot_show() -> None:
    """The fallback transport uses the same sampling: 8x needs 7.5 source frames a second for a 60 fps
    MJPEG push, and the JPEG encoder only ever sees those."""
    pieces = [WindowPiece("/archive/game.mp4", 0.0, 10.0)]
    command = direct_mjpeg_command(pieces, rate=8.0, width=1280, source_fps=60.0)
    joined = " ".join(command)
    assert "fps=7.5" in joined and "setpts=PTS/8" in joined
    assert "-c:v mjpeg" in joined and "-f image2pipe pipe:1" in joined
    assert "-an" in joined, "MJPEG carries no audio; the <img> fallback is silent"


def test_the_source_audio_rides_along_and_retimes_with_the_rate() -> None:
    """Audio is the recording's own soundtrack, seeked to the same second as the video window; a content rate
    stretches it with atempo (never resamples it into chipmunks), and the output is bounded by -t so a short
    soundtrack cannot truncate the clip. No source, no audio - and no ``-an`` on the command would leave the
    silent case unmapped rather than silent."""
    with_sound = encoder_command(
        width=1280,
        height=720,
        fps=30.0,
        output="pipe:1",
        live=True,
        encoder="h264_nvenc",
        keyframe_s=1.0,
        rate=2.0,
        source="/archive/game.mp4",
        audio=True,
        audio_start_s=1460.0,
        audio_duration_s=10.0,
    )
    assert with_sound[with_sound.index("-framerate") + 1] == "60", "the muxer's clock runs fps*rate per source second"
    assert with_sound[with_sound.index("-ss") + 1] == "1460.000"
    assert "/archive/game.mp4" in with_sound and "1:a:0?" in with_sound, "the optional map keeps a silent source working"
    assert with_sound[with_sound.index("-c:a") + 1] == "aac"
    assert with_sound[with_sound.index("-filter:a") + 1] == "atempo=2.0000", "2x is one atempo stage"
    assert with_sound[with_sound.index("-t") + 1] == "5.000", "ten source seconds at 2x are five output seconds"
    assert "-shortest" not in with_sound and "-an" not in with_sound

    silent = encoder_command(
        width=1280, height=720, fps=30.0, output="pipe:1", live=True, encoder="h264_nvenc", keyframe_s=1.0
    )
    assert "-an" in silent and "-t" not in silent

    quarter = encoder_command(
        width=1280,
        height=720,
        fps=30.0,
        output="pipe:1",
        live=True,
        encoder="h264_nvenc",
        keyframe_s=1.0,
        rate=0.25,
        source="/archive/game.mp4",
        audio=True,
    )
    assert quarter[quarter.index("-framerate") + 1] == "7.5"
    assert quarter[quarter.index("-filter:a") + 1].count("atempo") >= 2, "0.25x needs two halving stages"


def test_the_hevc_variant_is_tagged_for_apple_players() -> None:
    """HEVC in MP4 is only accepted by QuickTime/Safari as hvc1, where ffmpeg's default is hev1 - the tag is
    what makes the smaller file playable on the whole Apple line, and it costs nothing anywhere else."""
    hevc = encoder_command(
        width=1920, height=1080, fps=60.0, output="x.mp4", live=False, encoder="hevc_nvenc", keyframe_s=2.0
    )
    assert hevc[hevc.index("-tag:v") + 1] == "hvc1"
    h264 = encoder_command(
        width=1920, height=1080, fps=60.0, output="x.mp4", live=False, encoder="h264_nvenc", keyframe_s=2.0
    )
    assert "-tag:v" not in h264, "the tag is an HEVC-only concern"
    cpu = encoder_command(
        width=1920, height=1080, fps=60.0, output="x.mp4", live=False, encoder="libx265", keyframe_s=2.0
    )
    assert "libx265" in cpu and cpu[cpu.index("-crf") + 1] == "25", "the CPU HEVC fallback gets its own CRF"


def test_a_clip_key_changes_with_any_parameter_that_changes_the_picture() -> None:
    """The cache key is the encode's identity: a different window, rate, size or layer set is a different file,
    and repeating every parameter reuses the one that exists."""
    layers = dict.fromkeys(OVERLAY_NAMES, True)
    key = clip_key("m", 10.0, 2.0, 30.0, 1920, layers)
    assert clip_key("m", 10.0, 2.0, 30.0, 1920, layers) == key
    assert clip_key("m", 11.0, 2.0, 30.0, 1920, layers) != key
    assert clip_key("m", 10.0, 2.0, 30.0, 1280, layers) != key
    assert clip_key("m", 10.0, 2.0, 30.0, 1920, {**layers, "ball": False}) != key
    assert clip_key("m", 10.0, 2.0, 30.0, 1920, layers, rate=2.0) != key, "a faster cut is a different file"
    assert clip_key("m", 10.0, 2.0, 30.0, 1920, layers, audio=False) != key, "sound and silence differ"
    assert clip_key("m", 10.0, 2.0, 30.0, 1920, layers, codec="hevc") != key, "the codec is part of the identity"


def test_the_clip_cache_expires_by_count_and_by_age(tmp_path) -> None:
    """Clips are large and their encodes are expensive, so a few live for a while - but not forever, and not
    unboundedly. Eviction deletes the file, not just the memory of it."""
    cache = ClipCache(directory=tmp_path, ttl_s=0.08, max_entries=2)
    paths = []
    for name in ("a", "b", "c"):
        path = cache.target(name)
        path.write_bytes(name.encode())
        cache.keep(name, path)
        paths.append(path)
    assert cache.lookup("a") is None and not paths[0].exists(), "the oldest entry made room"
    assert cache.lookup("b") is not None and cache.lookup("c") is not None
    time.sleep(0.09)
    assert cache.lookup("c") is None and not paths[2].exists(), "age expires the rest"


# --------------------------------------------------------------------------------------------------------------
# The HTTP surface (encoders injected: no ffmpeg, no decode)
# --------------------------------------------------------------------------------------------------------------
class _FakeClipEncoder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        match,
        *,
        start_s,
        duration_s,
        fps,
        width,
        overlays,
        output,
        encoder=None,
        reader_factory=None,
        rate=1.0,
        audio=True,
        codec="h264",
    ):
        self.calls.append(
            {
                "start_s": start_s,
                "duration_s": duration_s,
                "fps": fps,
                "width": width,
                "overlays": dict(overlays),
                "rate": rate,
                "audio": audio,
                "codec": codec,
            }
        )
        output.write_bytes(b"MOCKMP4-" + f"{width}x{fps:g}".encode())
        return 1


def _serve(tmp_path, match, **kwargs):
    server = MatchStreamServer(
        ("127.0.0.1", 0),
        root=tmp_path,
        pace=False,
        clip_cache=ClipCache(directory=tmp_path / "clips"),
        **kwargs,
    )
    server._sessions["synthetic"] = match
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_a_clip_is_encoded_once_then_served_with_ranges(tmp_path) -> None:
    """The route blocks on the encode and then behaves like a file server: a player scrubbing through a clip
    asks for byte ranges, and the cache is what keeps those answers free (and re-requests, too)."""
    match = _match()
    encoder = _FakeClipEncoder()
    server = _serve(tmp_path, match, clip_encoder=encoder)
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        url = "/video/synthetic.mp4?duration=2&fps=30&width=1920&rate=2&audio=0"
        connection.request("GET", url)
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "video/mp4"
        assert response.getheader("Accept-Ranges") == "bytes"
        assert body == b"MOCKMP4-1920x30"
        assert len(encoder.calls) == 1
        assert encoder.calls[0]["start_s"] == match.start_s
        assert encoder.calls[0]["duration_s"] == pytest.approx(2.0)
        assert encoder.calls[0]["rate"] == pytest.approx(2.0), "the speed rides through to the encode"
        assert encoder.calls[0]["audio"] is False, "audio can be turned off per request"

        connection.request("GET", url, headers={"Range": "bytes=2-5"})
        partial = connection.getresponse()
        assert partial.status == 206
        assert partial.getheader("Content-Range") == f"bytes 2-5/{len(body)}"
        assert partial.read() == body[2:6]
        assert len(encoder.calls) == 1, "a range request must not re-encode"

        connection.request("GET", url, headers={"Range": "bytes=9999-"})
        assert connection.getresponse().status == 416
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_the_clip_defaults_are_the_sources_own_rate_and_resolution(tmp_path) -> None:
    """'Full framerate and resolution' is the endpoint's reason to exist, so the defaults go there - a request
    that says nothing gets the source's 1920x1080 at 30 fps (on the real matches: 3840x2160 at 60)."""
    match = _match(native=(1920, 1080, 30.0))
    encoder = _FakeClipEncoder()
    server = _serve(tmp_path, match, clip_encoder=encoder)
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", "/video/synthetic.mp4?duration=1")
        assert connection.getresponse().status == 200
        assert encoder.calls[0]["width"] == 1920
        assert encoder.calls[0]["fps"] == pytest.approx(30.0)
        assert encoder.calls[0]["audio"] is True, "the match's audio rides along by default"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


class _FakeLiveChunks:
    def __init__(self, chunks=(b"one", b"two"), wait_stop: bool = False) -> None:
        self.calls: list[dict] = []
        self.chunks = chunks
        self.wait_stop = wait_stop

    def __call__(
        self,
        match,
        *,
        start_s,
        fps,
        width,
        overlays,
        control=None,
        encoder=None,
        reader_factory=None,
        pace=True,
        rate=1.0,
        audio=True,
        codec="h264",
    ):
        self.calls.append(
            {
                "start_s": start_s,
                "fps": fps,
                "width": width,
                "control": control,
                "rate": rate,
                "audio": audio,
                "codec": codec,
            }
        )
        if not self.wait_stop:
            yield from self.chunks
            return
        for index in range(200):  # ends when the token is stopped, not when the chunks run out
            if control is not None and control.stopped:
                return
            yield f"chunk{index}".encode()
            time.sleep(0.02)


def test_the_live_endpoint_streams_fragmented_mp4_and_registers_its_stop_token(tmp_path) -> None:
    """The live variant is the MJPEG stream's sibling: a streaming response with no length, carrying the same
    stop-token contract (the pane's beacons and seeks end it the same way)."""
    match = _match()
    live = _FakeLiveChunks()
    server = _serve(tmp_path, match, live_chunks=live)
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", "/live/synthetic.mp4?fps=30&width=1280&rate=2&audio=0&codec=hevc&token=tok-1")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "video/mp4"
        assert response.read() == b"onetwo"
        assert live.calls[0]["width"] == 1280
        assert live.calls[0]["fps"] == pytest.approx(30.0)
        assert live.calls[0]["rate"] == pytest.approx(2.0), "the animation's speed rides through to the encode"
        assert live.calls[0]["audio"] is False, "audio can be turned off per request"
        assert live.calls[0]["codec"] == "hevc", "the codec choice reaches the encoder"
        assert isinstance(live.calls[0]["control"], StreamControl), "the token must reach the stream"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_a_live_video_stream_ends_when_its_token_is_stopped(tmp_path) -> None:
    match = _match()
    live = _FakeLiveChunks(wait_stop=True)
    server = _serve(tmp_path, match, live_chunks=live)
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", "/live/synthetic.mp4?token=tok-2")
        response = connection.getresponse()
        assert response.status == 200
        first = response.read(20)
        assert b"chunk" in first
        stopper = http.client.HTTPConnection(host, port, timeout=10)
        stopper.request("GET", "/stop?token=tok-2")
        assert stopper.getresponse().status == 200
        stopper.close()
        rest = response.read()  # ends because the stream was stopped, not because the chunks ran out
        assert (first + rest).count(b"chunk") < 200
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_the_play_page_picks_the_element_each_format_plays_in(tmp_path) -> None:
    """MJPEG only plays in an <img>; the encoded variants only in a <video>. The page is also the formats'
    index: it links the three ways to watch and passes the request's parameters through."""
    match = _match()
    server = _serve(tmp_path, match, clip_encoder=_FakeClipEncoder(), live_chunks=_FakeLiveChunks())
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=15)
        connection.request("GET", "/play/synthetic?format=clip&duration=7")
        page = connection.getresponse().read().decode()
        assert "<video" in page and "/video/synthetic.mp4?duration=7" in page
        assert "duration=10&format=clip" in page, "the nav re-asks for a 10s clip without doubling the parameter"
        connection.request("GET", "/play/synthetic?format=live&fps=30")
        page = connection.getresponse().read().decode()
        assert "<video" in page and "/live/synthetic.mp4?fps=30&token=" in page
        assert "sendBeacon" in page, "a page that navigates away must end its own live encoder"
        connection.request("GET", "/play/synthetic")
        page = connection.getresponse().read().decode()
        assert "<img" in page and "/stream/synthetic.mjpg" in page
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------------------------------------------
# Real encodes (tiny frames, x264): the pipelines against ffmpeg itself
# --------------------------------------------------------------------------------------------------------------
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is what is under test")


def _tone_source(tmp_path: Path, *, seconds: float = 2.0) -> Path:
    """A tiny real recording with a soundtrack, for the direct path's ffmpeg integration tests."""
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=30:duration={seconds:g}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds:g}",
            "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(source),
        ],
        check=True,
        capture_output=True,
    )
    return source


class _TinyReader:
    """A reader with no video behind it: small frames on the match's own grid."""

    def __init__(self, path, *, fps, width, start_s=0.0, duration_s=None, sleep_s=0.0):
        self.fps = float(fps)
        self.start_s = float(start_s)
        self.duration_s = duration_s
        self.sleep_s = sleep_s
        self.width = width

    def frames(self):
        count = max(1, int(round((self.duration_s or 1.0) * self.fps)))
        for index in range(count):
            if self.sleep_s:
                time.sleep(self.sleep_s)
            yield self.start_s + index / self.fps, np.full((64, 96, 3), 30 + index % 100, dtype=np.uint8)


@needs_ffmpeg
def test_a_clip_really_encodes_to_an_h264_file_the_source_rate_reads_back(tmp_path) -> None:
    """The end of the pipeline, against ffmpeg itself: the artifact is an H.264 MP4 with the frame count, rate
    and size that were asked for - and it plays (the index is at the front, so a seek is a range, not an encode)."""
    from soccer_analytics.ingest.ffmpeg_reader import probe_video

    match = _match()
    output = tmp_path / "clip.mp4"
    frames = encode_clip(
        match,
        start_s=match.start_s,
        duration_s=1.0,
        fps=5.0,
        width=96,
        overlays=dict(NO_LAYERS),
        output=output,
        encoder="libx264",
        reader_factory=_TinyReader,
    )
    assert frames == 5
    probe = probe_video(output)
    assert probe.codec == "h264"
    assert (probe.width, probe.height) == (96, 64)
    assert probe.fps == pytest.approx(5.0, abs=0.1)
    assert probe.duration_s == pytest.approx(1.0, abs=0.3)


@needs_ffmpeg
def test_a_live_stream_really_emits_a_fragmented_mp4_and_ends_when_stopped(tmp_path) -> None:
    """The live artifact must be joinable mid-flight (ftyp + moof fragments) and must *stop* - a stream whose
    token is set stops feeding the encoder and closes out, instead of running the window to its end."""
    match = _match()
    payload = b"".join(
        iter_live_chunks(
            match,
            start_s=match.start_s,
            fps=5.0,
            width=96,
            overlays=dict(NO_LAYERS),
            encoder="libx264",
            reader_factory=_TinyReader,
            pace=False,
        )
    )
    assert payload[4:8] == b"ftyp", "an MP4 starts with its file-type box"
    assert b"moof" in payload, "fragmented, so a player can join mid-flight"

    control = StreamControl()
    started = time.monotonic()
    stopped = _stopped_live(match, control)
    elapsed = time.monotonic() - started
    assert stopped[4:8] == b"ftyp"
    assert elapsed < 5.0, "a stopped stream must not wait out the window"


@needs_ffmpeg
def test_the_matchs_audio_really_rides_along_in_a_clip(tmp_path) -> None:
    """The end of the audio path, against ffmpeg itself: a real (tiny, generated) recording with a tone is
    seeked alongside the video window, so the clip a viewer gets is the match, sound and all - and ``audio=0``
    really leaves it silent."""
    from soccer_analytics.ingest.ffmpeg_reader import probe_video

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(source),
        ],
        check=True,
        capture_output=True,
    )
    match = _match(video=source, start_s=0.0)
    with_sound = tmp_path / "with.mp4"
    encode_clip(
        match,
        start_s=0.0,
        duration_s=1.0,
        fps=5.0,
        width=96,
        overlays=dict(NO_LAYERS),
        output=with_sound,
        encoder="libx264",
        reader_factory=_TinyReader,
        audio=True,
    )
    assert probe_video(with_sound).has_audio, "the clip carries the source's soundtrack"
    muted = tmp_path / "muted.mp4"
    encode_clip(
        match,
        start_s=0.0,
        duration_s=1.0,
        fps=5.0,
        width=96,
        overlays=dict(NO_LAYERS),
        output=muted,
        encoder="libx264",
        reader_factory=_TinyReader,
        audio=False,
    )
    assert not probe_video(muted).has_audio, "audio=0 leaves the clip silent"


@needs_ffmpeg
def test_the_direct_live_path_streams_sampled_h264_and_keeps_the_audio(tmp_path) -> None:
    """The direct path end to end: a 2x request samples 30 source fps into a 60 fps fragmented MP4 and the
    recording's own soundtrack comes with it - no decoded frame passes through Python."""
    from soccer_analytics.ingest.ffmpeg_reader import probe_video

    source = _tone_source(tmp_path, seconds=2.0)
    match = _match(video=source, start_s=0.0, frame_count=10, native=(96, 64, 30.0))
    payload = b"".join(
        iter_live_chunks(
            match,
            start_s=0.0,
            fps=30.0,
            width=160,
            overlays=dict(NO_LAYERS),
            encoder="libx264",
            pace=False,
            rate=2.0,
            audio=True,
        )
    )
    assert payload[4:8] == b"ftyp" and b"moof" in payload
    output = tmp_path / "direct.mp4"
    output.write_bytes(payload)
    probe = probe_video(output)
    assert probe.fps == pytest.approx(60.0, abs=1.0), "the output timeline stays at 60 fps"
    assert probe.duration_s == pytest.approx(1.0, abs=0.25), "2 source seconds at 2x"
    assert probe.has_audio, "the match audio must ride along"


@needs_ffmpeg
def test_the_direct_mjpeg_path_really_emits_jpegs_without_a_python_render(tmp_path) -> None:
    """The MJPEG fallback gets the same treatment: ffmpeg decodes, samples and JPEG-encodes; Python only
    splits the byte stream into pictures the multipart writer can frame."""
    source = _tone_source(tmp_path, seconds=2.0)
    match = _match(video=source, start_s=0.0, frame_count=10, native=(96, 64, 30.0))
    frames = list(
        iter_annotated_frames(
            match, 0.0, rate=2.0, width=160, overlays=dict(NO_LAYERS), pace=False
        )
    )
    assert len(frames) >= 20, "the 2x push should carry most of a second of 60 fps pictures"
    assert all(jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9") for _index, jpeg in frames)


@needs_ffmpeg
def test_the_hevc_variant_really_encodes_when_the_box_can(tmp_path) -> None:
    """With an HEVC encoder present the clip really comes out as HEVC in an MP4 - the option is wired to the
    codec, not just to a flag; machines without one skip (the H.264 path covers the mechanics everywhere)."""
    from soccer_analytics.ingest.ffmpeg_reader import probe_video

    try:
        encoder = pick_encoder("hevc")
    except VideoError:
        pytest.skip("no HEVC encoder on this machine")
    match = _match()
    output = tmp_path / "clip_hevc.mp4"
    encode_clip(
        match,
        start_s=match.start_s,
        duration_s=1.0,
        fps=5.0,
        width=96,
        overlays=dict(NO_LAYERS),
        output=output,
        encoder=encoder,
        codec="hevc",
        reader_factory=_TinyReader,
    )
    probe = probe_video(output)
    assert probe.codec == "hevc", f"expected HEVC, got {probe.codec}"
    assert (probe.width, probe.height) == (96, 64)


def _stopped_live(match, control):
    """Start a live stream, stop it once the first bytes have arrived, and return everything it sent."""
    chunks = iter_live_chunks(
        match,
        start_s=match.start_s,
        fps=5.0,
        width=96,
        overlays=dict(NO_LAYERS),
        control=control,
        encoder="libx264",
        reader_factory=lambda *a, **k: _TinyReader(*a, sleep_s=0.05, **k),
        pace=False,
    )
    payload = b""
    for chunk in chunks:
        payload += chunk
        if payload and not control.stopped:
            control.request_stop()
    return payload


def test_multi_clip_audio_inputs_are_seeked_per_clip_and_joined_by_the_concat_filter() -> None:
    """A clip-set window's sound comes in as one nicely bounded input per clip, stitched by ``concat``.

    The shape matters as much as the sound: each input seeks inside its own clip (never the concat demuxer) and
    is bounded by ``-t``, and the retiming happens once, after the join - so the pieces cannot drift apart.
    """
    command = encoder_command(
        width=96,
        height=64,
        fps=5.0,
        output="pipe:1",
        live=False,
        encoder="libx264",
        keyframe_s=1.0,
        rate=2.0,
        audio_duration_s=3.0,
        audio_inputs=[("/clips/a.mp4", 1.25, 0.75), ("/clips/b.mp4", 0.0, 1.5)],
    )
    joined = " ".join(command)
    assert "-ss 1.250 -t 0.750 -i /clips/a.mp4" in joined
    assert "-ss 0.000 -t 1.500 -i /clips/b.mp4" in joined
    assert "[1:a:0][2:a:0]concat=n=2:v=0:a=1,atempo=2.0000[aout]" in joined
    assert "-map [aout]" in joined


def test_a_single_audio_piece_maps_exactly_like_a_single_source() -> None:
    command = encoder_command(
        width=96,
        height=64,
        fps=5.0,
        output="pipe:1",
        live=False,
        encoder="libx264",
        keyframe_s=1.0,
        audio_inputs=[("/clips/a.mp4", 3.0, 2.0)],
    )
    joined = " ".join(command)
    assert "-map 1:a:0?" in joined
    assert "concat" not in joined and "filter_complex" not in joined

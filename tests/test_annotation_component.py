"""The landmark component's contract with Python: what a gesture may, and may not, report.

The component is a static frontend, so these read the file rather than driving a browser. They exist because the
timeline's reactivity is easy to regress silently: scrubbing must stay in the browser, and the magnified crop must
only be re-read when the user aims at a point.
"""

from __future__ import annotations

from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "field_annotation_component"
    / "index.html"
)


def test_the_frontend_is_the_scrubbable_landmark_view() -> None:
    html = COMPONENT.read_text()
    assert "<video" in html and "streamlit:setComponentValue" in html
    assert 'id="overviewVideo"' in html, "the whole-frame viewport is the timeline"


def test_scrubbing_never_reports_to_python() -> None:
    """A scrub must not ask Python for anything: a round trip per release is what re-rendered the crop."""
    html = COMPONENT.read_text()
    assert "sendValue('scrub')" not in html
    assert "commitScrub" not in html


def test_an_aiming_gesture_carries_the_frame() -> None:
    """The frame must ride with the aim, or clicking after a scrub would crop a frame that is not on screen."""
    html = COMPONENT.read_text()
    assert "frame: currentFrame()" in html


def test_the_marker_picker_tags_the_next_click() -> None:
    """A landmark is named at the click, in a popover that appears where the click was made."""
    html = COMPONENT.read_text()
    assert 'id="marker-popover"' in html
    assert "function openPopover(" in html, "the picker appears at the click, not in a toolbar"
    assert "commitPoint(x, y, name)" in html, "the pick is what the click is tagged with"
    assert "openPopover(x, y)" in html, "a click on the magnified view opens the picker"


def test_a_click_commits_immediately_without_an_apply_button() -> None:
    """The click is its own commit: there is no Apply step to forget."""
    html = COMPONENT.read_text()
    assert 'id="apply-btn"' not in html, "the Apply button is gone"
    assert "sendValue('apply')" in html, "the commit still reports as an apply"
    commit = html.split("function commitPoint")[1].split("function addPoint")[0]
    assert "sendValue('apply')" in commit, "committing a click sends it to Python at once"
    drag = html.split("function endDrag")[1].split("canvas.addEventListener('mouseup'")[0]
    assert "sendValue('apply')" in drag, "a finished drag commits the same way"


def test_markers_are_only_drawn_on_the_exact_crop_frame() -> None:
    """The proxy holds one picture per second, so a marker drawn on it points where the corner was ages ago.

    The still Python sends is the exact frame the crop shows, and the markers may only be drawn over it - otherwise
    a perfectly good click reads as a misplaced one, which is exactly what happened on a whip-pan frame.
    """
    html = COMPONENT.read_text()
    assert "function pictureIsExact()" in html
    marker_block = html.index("if (exact) {")
    assert marker_block < html.index("(args.marker_points || []).forEach"), "markers sit inside the exact-frame branch"
    # and the still must be loaded in video mode too, not only as the no-proxy fallback
    assert "if (args.overview_data && args.overview_data !== overviewDataSrc)" in html
    # scrubbing off the crop's frame must drop the still immediately
    assert "drawOverview();" in html.split("scrub.addEventListener('input'")[1].split("});")[0]


def test_the_viewport_follows_a_frame_moved_from_python() -> None:
    """Jumping to a landmark's frame from the list is a Python-side move; the bar and picture follow it.

    The other half of the rule matters just as much: a scrub the user made must survive the next poll, which is why
    the follow is conditional on the frame having actually changed since the last render.
    """
    html = COMPONENT.read_text()
    assert "const movedByPython = mounted && videoMode && renderedFrom !== cropFrame;" in html
    assert "} else if (movedByPython) {" in html
    follow = html.split("const movedByPython")[1].split("}\n")[0]
    assert "pendingFrame = renderedFrom;" in follow and "applyPendingFrame();" in follow


def test_a_placed_marker_can_be_dragged_onto_the_marking_it_stands_for() -> None:
    """Re-anchoring is a nudge: the placed marker carries its label, and the drag is the whole measurement."""
    html = COMPONENT.read_text()
    assert "function featureNear(x, y)" in html
    assert "canvas.addEventListener('mousedown'" in html
    assert "suppressClick = true;" in html, "a drag must not leave a fresh click behind it"
    assert "ctx.strokeText(feature.label" in html, "a marker has to say which landmark it stands for"


def test_a_dragged_marker_survives_an_unrelated_rerun() -> None:
    """Python's copy of a marker only changes on a commit, so aiming the view must not yank a drag back."""
    html = COMPONENT.read_text()
    assert "feature.dirty = true;" in html
    assert "moved.get(feature.id)" in html, "the pending drag is matched to Python's copy by id"
    end_drag = html.split("function endDrag")[1].split("canvas.addEventListener('mouseup'")[0]
    assert "feature.dirty = false" in end_drag, "a finished drag makes Python's positions the canonical ones"


def test_clear_is_scoped_to_the_frame_being_viewed() -> None:
    """Clear removes the applied points in the frame of reference, and Python does the same to the stored clicks."""
    html = COMPONENT.read_text()
    clear = html.split("clearBtn.addEventListener('click'")[1].split("});")[0]
    assert "sendValue('clear')" in html, "Clear reports as its own action"
    assert "savedFeatures = [];" in clear, "the component empties its own copy"


def test_a_gesture_carries_its_identity_so_the_sticky_value_can_be_ignored() -> None:
    """With no Apply button to re-key the component, Python dedupes on the sequence number instead."""
    html = COMPONENT.read_text()
    assert "seq: seq," in html and "mount: typeof args.mount_nonce" in html
    assert "seq += 1;" in html, "every reported gesture is a new one"


def test_the_bar_maps_frames_through_the_analysis_rate() -> None:
    """The proxy can be seconds shorter than the segment, so its duration must not stretch the frame numbering.

    Mapping frame k onto ``k / (frames - 1) * proxy_duration`` put the bar seconds ahead of the picture whenever a
    proxy was rebuilt from an older window (measured 2.4 s at 61 minutes on a proxy 2.8 s short) - and the overlay,
    which follows the picture, then disagreed with the still at the crop frame.
    """
    html = COMPONENT.read_text()
    frame_to = html.split("function frameToTime(frame)")[1].split("function timeToFrame")[0]
    assert "analysisFps > 0" in frame_to and "clampFrame(frame) / analysisFps" in frame_to
    time_to = html.split("function timeToFrame(time)")[1].split("function updateTimelineReadout")[0]
    assert "clampFrame(time * analysisFps)" in time_to
    assert "args.analysis_fps" in html


def test_the_pitch_overlay_follows_the_timeline_in_the_browser() -> None:
    """Scrubbing and playing must redraw the pitch overlay without a round trip per frame.

    Python sends one homography per proxy picture plus the markings, and the component draws the sample that
    belongs to the picture on screen - the presented frame's media time, not the 30 fps scrub readout. Blending
    between pictures is only for a proxy that holds every frame (a short window), where the samples form a dense
    grid; between pictures a second apart it is wrong by the whole pan between them.
    """
    html = COMPONENT.read_text()
    assert "function overlayMatrix(time)" in html, "the sample is picked for the picture being shown"
    assert "function sampleAtOrBefore(time)" in html, "the picture shown is the sample at or before its time"
    assert "overlayMode === 'latest'" in html, "per-picture samples are used directly, not blended"
    assert "function drawPitchOverlay()" in html
    assert "drawPitchOverlay();" in html.split("function drawOverview")[1].split("function drawImage")[0], (
        "the overlay is drawn with the rest of the whole-frame view"
    )
    assert "args.overlay" in html and "args.overlay_polylines" in html
    assert "function decodeOverlay(payload)" in html and "Float32Array" in html, (
        "the matrices arrive base64-packed, not as JSON digits on every gesture"
    )
    # The two projections must never stack: the exact still already carries Python's own overlay, so the browser's
    # copy is only drawn when the still is not up. Drawing both put one geometry above the other.
    overview = html.split("function drawOverview")[1].split("function drawImage")[0]
    branches = overview.split("const stillUp = exact && overviewImage;")[1]
    still_branch, live_branch = branches.split("} else {")
    assert "overviewCtx.drawImage(overviewImage" in still_branch, "the exact still is drawn when it is up"
    assert "drawPitchOverlay();" in live_branch, "the browser projection is drawn only while the still is away"
    # The overlay's y must scale by the canvas/frame *height* ratio: v is width-normalized, and dividing by the
    # width squashed the geometry to 9/16 of its height - the pitch hovered far above the field.
    draw = html.split("function drawPitchOverlay")[1].split("function drawOverview")[0]
    assert "overview.height / frameHeight" in draw
    assert "overview.height / frameWidth" not in draw
    # While playing, redraws ride the video's presented frames (their media time), not the coarse `currentTime`
    # readout - the presented-frame clock is what keeps the geometry sitting exactly on the picture.
    assert "requestVideoFrameCallback" in html
    assert "metadata.mediaTime" in html
    time_of = html.split("function overlayTime()")[1].split("function drawPitchOverlay")[0]
    assert "presentedMediaTime" in time_of, "playing uses the presented frame's time"
    assert "overviewVideo.currentTime" in time_of, "paused uses the video clock, not the scrub readout"
    assert "clampFrame(Number(scrub.value))" not in time_of, "the bar is not the picture while paused"
    # The seek the scrub queued changes the picture; the overlay has to follow it, not the previous one.
    assert "overviewVideo.addEventListener('seeked'" in html
    assert "overlayTime()" in draw

"""The replay component's contract: playback stays in the browser, and what Python must supply.

Like the annotation component, this is a static frontend read as a file. The checks exist because the interesting
property is easy to lose: a playback control that round-trips to Python would re-render the whole page per frame.
The two deliberate exceptions are the tag bar and a right-click delete on the strip: each is a discrete gesture
whose whole point is to change the match's event log, so it posts one value and nothing else does.
"""

from __future__ import annotations

from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_analytics"
    / "dashboard"
    / "pitch_replay_component"
    / "index.html"
)
APP = Path(__file__).resolve().parents[1] / "src" / "soccer_analytics" / "dashboard" / "app.py"


def test_playback_controls_exist_and_run_locally() -> None:
    html = COMPONENT.read_text()
    assert 'id="play"' in html and 'id="scrub"' in html and "requestAnimationFrame" in html
    # Play/pause, stepping and scrubbing must not talk to Python: a component value per gesture (let alone per
    # frame) would re-render the whole page. The two deliberate exceptions are store gestures - a tag press and
    # a marker delete - and both go through the one ``postAction`` call, so the raw protocol call stays singular.
    assert html.count("Streamlit.setComponentValue(") == 1, "only store gestures may round-trip to Python"
    assert "function postAction(" in html
    assert "postAction({action: 'tag'" in html
    assert "postAction({action: 'delete'" in html
    assert html.index("Streamlit.setComponentValue(") > html.index("function postAction(")
    assert html.index("Streamlit.setComponentValue(") > html.index("function sendTag(")
    assert html.index("Streamlit.setComponentValue(") > html.index("function sendDelete(")


def test_an_event_marker_can_be_deleted_by_right_click() -> None:
    """The strip is a review queue: a right-click on one marker deletes that event, not a row chosen in a form.

    The gesture must not fall through to the strip's seek click, must name the event it drew (position and
    identity) rather than deleting whatever later occupies that position, and must use the same sequence +
    acknowledgment contract as the tag bar so a sticky component value cannot hide a marker forever.
    """
    html = COMPONENT.read_text()
    assert "addEventListener('contextmenu'" in html
    assert "event.preventDefault()" in html.split("addEventListener('contextmenu'", 1)[1]
    assert "function showEventMenu(" in html and 'id="event-menu-delete"' in html
    assert "function sendDelete(" in html
    send_delete = html.split("function sendDelete(", 1)[1]
    for field in ("event.index", "event.log_time_s", "event.type", "event.source", "event.video"):
        assert field in send_delete, f"a delete must carry {field} to prove the row it means"
    assert "state.pendingDeletes" in send_delete
    assert "args.ack_delete_seq" in html, "the acknowledgment is what clears a pending marker"
    assert "state.pendingDeletes = state.pendingDeletes.filter" in html


def test_the_strip_index_reaches_the_event_log_delete() -> None:
    """The component can only name a log row if Python puts its index on the strip; the action must be handled."""
    source = APP.read_text()
    assert '"index": index' in source, "each strip event must carry its position in the event log"
    assert '"log_time_s"' in source, "the delete matcher needs the recording second, not the strip second"
    assert "ack_delete_seq=int(st.session_state.get(f\"replay_delete_ack" in source
    assert "_apply_timeline_deletes(library, match_id, replay_action)" in source
    # The picker's widget exists before the delete is handled (Streamlit forbids writing its key mid-run), so the
    # repair has to be queued and applied at the start of the next run, before the widget is created again.
    assert 'f"event_reindex::{match_id}"' in source
    assert "_apply_event_picker_repair(match_id)" in source


def test_the_tag_bar_offers_every_event_type_on_the_playback_second() -> None:
    """The quick-tag buttons are the only thing that talks to Python, one press at a time.

    Each press posts the event type with the playback's own second *at the press* - reading the clock after the
    round trip would land the tag wherever playback got to meanwhile - and unacknowledged presses ride along in
    ``pendingTags``, so a slow rerun can neither drop a press nor cause it to be stored twice.
    """
    from soccer_analytics.analysis.events import EVENT_TYPES

    html = COMPONENT.read_text()
    assert 'id="tag-bar"' in html and 'id="tag-buttons"' in html and 'id="tag-team"' in html
    tag_types_block = html.split("const TAG_TYPES = [", 1)[1].split("];", 1)[0]
    for event_type in EVENT_TYPES:
        assert f"'{event_type}'" in tag_types_block, f"the tag bar has no button for {event_type}"
    send_tag = html.split("function sendTag(", 1)[1]
    assert "state.time" in send_tag, "the tag must be stamped with the playback's own second"
    assert "Math.min(state.data.duration_s" in send_tag, "a tag before a payload has loaded must not be sent"
    assert "pendingTags" in send_tag, "unacknowledged presses must ride along with the next one"
    assert "args.ack_tag_seq" in html, "the acknowledgment is what clears a stored press"
    assert "state.pendingTags = state.pendingTags.filter" in html


def test_a_swapped_attack_reopens_the_footage_so_its_scoreboard_agrees() -> None:
    """The pitch arrows redraw from a new payload, but the footage's scoreboard is drawn server-side.

    A swap must therefore re-open the footage pane as well; otherwise the animation shows the corrected ends
    while the scoreboard beside it still points the old way. The viewer's place is kept, since the correction
    is only about which way the arrows point.
    """
    html = COMPONENT.read_text()
    load_data = html.split("async function loadData", 1)[1].split("function renderLegend", 1)[0]
    assert "previousAttack" in load_data
    assert "attackChanged" in load_data
    assert "restartFootage()" in load_data
    assert "previousTime" in load_data, "a correction of the ends must not reset the playback position"


def test_the_component_fetches_the_replay_by_url() -> None:
    """The payload is megabytes; it must be fetched once, not pushed through component arguments."""
    html = COMPONENT.read_text()
    assert "fetch(url)" in html or "fetch(" in html
    assert "data_url" in html


def test_shirt_numbers_arrive_as_a_separate_argument() -> None:
    """Numbers can change (roster edits) without rebuilding the replay, so they travel outside the JSON."""
    html = COMPONENT.read_text()
    assert "args.numbers" in html
    assert "entry.number" in html


def test_the_view_can_be_sized_and_shows_a_legend() -> None:
    html = COMPONENT.read_text()
    assert "setFrameHeight" in html
    assert 'id="legend"' in html


def test_the_selected_player_comes_from_python() -> None:
    html = COMPONENT.read_text()
    assert "args.selected_track" in html
    assert "selected_track" in html


def test_the_scanned_ball_layer_exists_and_distinguishes_forecasts() -> None:
    """When the payload carries a ball track, it is drawn as its own layer - and a forecast is not a sighting.

    The scan's third element per frame is the measured flag; the component is where that honesty has to land, so
    it must both read it and show it - a drawn football for a detection, a dashed ring for the tracker's forecast.
    """
    html = COMPONENT.read_text()
    assert "state.data.ball" in html
    assert "BALL_COLOR" in html
    assert "ball[2]" in html, "the per-frame measured flag must drive the drawing"
    assert "drawSoccerBall" in html, "the ball must read as a football, not another filled circle"


def test_the_markers_wear_the_measured_kit_color() -> None:
    """The kit color is measured data, so the component takes it from the payload - palette only as fallback.

    The number inside a marker also has to stay readable *on* that color: a fixed white number disappears on a
    white kit, which is why the text color is chosen from the marker's own color rather than hard-coded.
    """
    html = COMPONENT.read_text()
    assert "state.data.team_colors" in html, "markers must read the measured kit colors from the payload"
    assert "TEAM_COLORS[team]" in html, "the fixed palette must remain as the fallback"
    assert "markerTextColor" in html, "the number color must follow the kit color, not be assumed white"


def test_the_camera_is_a_cartoon_pointing_where_it_aimed() -> None:
    """The camera is a small icon at its own ground position, turned to face the point the gimbal was aimed at -
    the direction the real camera pointed, readable on the pitch. It replaced the aim dot and direction line, so
    both facts ride the one icon.

    The camera sits off the near touchline, so the view bounds must widen to include it - otherwise the icon is
    clipped off the edge of the canvas.
    """
    html = COMPONENT.read_text()
    assert "state.data.camera" in html, "the icon is drawn at the camera position in the payload"
    assert "state.data.aim" in html, "the icon is oriented by the aim point in the payload"
    assert "drawCamera" in html, "the camera must be drawn as its own icon, not a dot"
    assert "CAMERA_COLOR" in html
    assert "AIM_COLOR" not in html, "the yellow aim dot was replaced by the camera icon"
    assert "viewBounds" in html, "the view must expand to include the camera position"
    assert "cam[0] - MARGIN_M" in html and "cam[1] - MARGIN_M" in html


def test_roles_are_color_coded_and_named_in_the_legend() -> None:
    """Referee and goalkeepers wear fixed-color rings; the legend names the colors, not the markers.

    The kit descriptor is not trusted for this: measured on the real game one keeper's kit read "red" while
    being orange to the eye. Role identity rides the payload's ``role`` field and a color the math cannot
    muddy - and the user asked for color coding, not text labels.
    """
    html = COMPONENT.read_text()
    assert "ROLE_COLORS" in html and "referee:" in html and "goalkeeper:" in html
    assert "player.role" in html, "the marker ring must read the role the payload carries"
    assert "referee (ring)" in html and "goalkeeper (ring)" in html, "the legend must name the ring colors"


def test_attack_arrows_carry_each_teams_direction_and_swap_at_half_time() -> None:
    """The arrows point the way each team attacks, drawn at the goal it defends, and swap on the payload's
    second-period frame; a payload from before directions existed simply draws nothing."""
    html = COMPONENT.read_text()
    assert "function drawAttackArrows(" in html and "drawAttackArrows(frame)" in html
    assert "attack.directions" in html and "attack.half_frame" in html
    assert "attacking direction" in html, "the legend must say what the arrows mean"


def test_the_timeline_strip_draws_momentum_and_events() -> None:
    """The strip above the pitch combines the momentum curve with the event markers, and seeks on click.

    The events and momentum arrive as component arguments (they change without the replay being rebuilt), so the
    component must read them from ``args`` and draw both - a momentum curve and one marker per event, with a manual
    tag filled and a detected candidate hollow so a machine's guess never reads as a human's certainty.
    """
    html = COMPONENT.read_text()
    assert 'id="timeline"' in html and 'id="tl"' in html
    assert "args.events" in html, "events travel as a component argument"
    assert "args.momentum" in html, "momentum travels as a component argument"
    assert "drawTimeline" in html
    assert "eventColor" in html, "each event type has its own color"
    assert "event.source === 'manual'" in html, "a manual tag must be drawn differently from a detected one"
    assert "timeline.addEventListener('click'" in html, "clicking the strip must seek the replay"


def test_each_event_type_gets_its_own_track_and_a_category_shape() -> None:
    """The strip splits events onto tracks: one row per event type present, offensive kinds first, then
    defensive, then the rest. The marker's shape says the category - triangle offensive (goal/shot/corner/
    penalty), square defensive (save/block/clearance), circle other - while the color stays the type's and the
    fill still means tagged vs detected."""
    html = COMPONENT.read_text()
    assert "timelineLanes" in html, "the layout is one track per event type"
    assert "lanes.yOf(event.type)" in html, "every event is drawn on its own type's track"
    assert "EVENT_CATEGORIES" in html
    assert "goal: 'offensive'" in html and "shot: 'offensive'" in html
    assert "corner: 'offensive'" in html and "penalty: 'offensive'" in html
    assert "save: 'defensive'" in html and "block: 'defensive'" in html and "clearance: 'defensive'" in html
    assert "function eventShapePath" in html and "drawEventMarker" in html
    assert "category === 'offensive'" in html and "category === 'defensive'" in html, "triangle and square"
    assert "TL_LANE_H" in html, "the tracks stack under the momentum band"


def test_the_arrows_step_between_timeline_events_instead_of_frames() -> None:
    """One analysis frame is nothing to look at between events; the events are the moments a review moves
    between. The arrows jump to the previous/next event on the strip - strictly past the current second, so a
    cluster of events at one moment takes one press - wrap at the ends, and pause like a strip click does."""
    html = COMPONENT.read_text()
    assert 'id="prev-event"' in html and 'id="next-event"' in html, "the arrows live beside play"
    assert 'id="step-back"' not in html and 'id="step-fwd"' not in html, "frame stepping is replaced, not kept"
    seek = html.split("function seekToEvent", 1)[1].split("\n      function ", 1)[0]
    assert "state.events" in seek and "event.time_s" in seek, "the steps are the events the strip draws"
    assert "> state.time + epsilon" in seek and "< state.time - epsilon" in seek, "strictly past the current second"
    assert "times[0]" in seek and "times[times.length - 1]" in seek, "wrap at both ends"
    assert "setPlaying(false)" in seek and "draw();" in seek, "pause and show the moment"
    assert "prevEventBtn.disabled" in html, "with no events there is nowhere to step"
    listeners = html.split("prevEventBtn.addEventListener", 1)[1].split("tl.addEventListener('mousemove'", 1)[0]
    assert "seekToEvent(-1)" in listeners and "seekToEvent(1)" in listeners


def test_timeline_markers_name_their_event_on_hover() -> None:
    """The strip is a canvas, so hovering needs a hit test in the drawing's own coordinates and a fixed-position
    popover (which the canvas cannot clip): the event's type, its second, the team, whether it was tagged or
    detected and any note - the glance the marker colors alone cannot give. The arrows flash the same popover
    for the event they land on."""
    html = COMPONENT.read_text()
    assert 'id="tl-popover"' in html and "#tl-popover {" in html
    assert "pointer-events: none" in html, "the popover must never eat the strip's clicks"
    hit = html.split("function eventAt", 1)[1].split("\n      function ", 1)[0]
    assert "tl.getBoundingClientRect()" in hit and "timelineMarkerX" in hit, "hit test in the drawing's coordinates"
    assert "timelineLanes" in hit and "clientY" in hit, "the hit test must know which track the pointer is on"
    assert "const TL_PAD" in html and html.count("const pad = TL_PAD") == 2, (
        "one padding for drawing, clicking and hit testing"
    )
    text = html.split("function eventPopoverText", 1)[1].split("\n      function ", 1)[0]
    assert "event.type" in text and "clockText(event.time_s)" in text, "type and time at a glance"
    assert "'tagged' : 'detected'" in text
    assert "event.note" in text and "player_number" in text
    assert "tl.addEventListener('mousemove'" in html and "tl.addEventListener('mouseleave'" in html
    assert "function hideEventPopover" in html
    assert "flashEventPopover" in html, "a jump with the arrows names what it landed on"


def test_the_legend_is_redrawn_when_events_arrive_without_a_reload() -> None:
    """The legend's event chips must follow the events argument, not only the replay payload.

    Events travel as their own argument so a new tag shows up without rebuilding the replay. The legend lists a chip
    per event type, so if it were only drawn from ``loadData`` the strip would gain markers while the legend still
    listed the types it saw when the payload loaded - which is exactly what happened when the detectors first ran.
    """
    html = COMPONENT.read_text()
    render_event = html.split("Streamlit.events.addEventListener(Streamlit.RENDER_EVENT", 1)[1]
    # The else branch is the "same payload, new arguments" path - the one a new tag takes.
    else_branch = render_event.split("} else {", 1)[1].split("Streamlit.setFrameHeight", 1)[0]
    assert "renderLegend()" in else_branch, "the legend must be redrawn when the arguments change"


def test_the_pitch_and_the_footage_stream_sit_side_by_side() -> None:
    """The footage is the annotated match stream beside the animation; the pitch is the whole window. They are
    read together, and the footage gets the larger half - the video is the match itself, read at a glance, while
    the pitch tolerates the narrower column - with the tag bar stacked under the animation, in the space the
    shorter pitch column leaves. The pitch sizes itself from its own pane, not the page."""
    html = COMPONENT.read_text()
    assert 'id="panes"' in html, "the two views must be laid out as panes"
    assert 'id="pitch-pane"' in html and 'id="footage-pane"' in html
    assert 'id="footage"' in html and "<img" in html, "only an <img> element plays an MJPEG stream"
    pitch_css = html.split("#pitch-pane {", 1)[1].split("}", 1)[0]
    footage_css = html.split("#footage-pane {", 1)[1].split("}", 1)[0]
    assert "flex: 1 1 45%" in pitch_css and "flex: 1 1 55%" in footage_css, "the footage gets the larger half"
    assert html.index('id="pitch-pane"') < html.index('id="tag-bar"') < html.index('id="footage-pane"'), (
        "the tag bar is stacked under the animation, not beside it"
    )
    assert "pitchPane.clientWidth" in html, "the pitch must measure its own pane, not the page"


def test_the_footage_can_take_the_whole_page_and_give_it_back() -> None:
    """The split is the working layout, but watching the match itself wants the footage alone. One button toggles
    between them client-side (a round trip would make the jump a page rerun): the pitch column steps aside and the
    same click brings it back. A paused pane refetches its still at the new width; a running stream keeps playing
    and is re-asked at the new width on its next restart."""
    html = COMPONENT.read_text()
    assert 'id="wide"' in html, "the toggle sits in the controls"
    wide_css = html.split("body.wide #pitch-pane {", 1)[1].split("}", 1)[0]
    assert "display: none" in wide_css, "full width means the pitch column steps aside"
    handler = html.split("wideBtn.addEventListener", 1)[1].split("resyncFootageBtn", 1)[0]
    assert "classList.toggle('wide'" in handler, "one click toggles both ways"
    assert "scheduleStill()" in handler, "a paused pane's still should match the new width"
    assert "setFrameHeight" in handler, "the iframe height must follow the layout change"


def test_playback_holds_until_the_streams_first_frame_but_never_forever() -> None:
    """The encoder takes a second or two to start; letting the animation run through it would put the two clocks
    out of step from the first second. A play press therefore holds the animation (freezing a run in progress,
    e.g. a seek) until the stream's first frame is up - loadeddata for the video, the <img> load in MJPEG mode -
    and the hold is capped, so a stream that never starts cannot wedge playback; giving up releases it too."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_PRIME_TIMEOUT_S" in html
    play = html.split("function setPlaying", 1)[1].split("function beginTicking", 1)[0]
    assert "state.playbackPending = footageEnabled() && state.footageSync && !state.footageError" in play
    assert "cancelAnimationFrame(state.raf)" in play, "a seek while playing freezes until the stream catches up"
    assert "beginTicking()" in play and "setTimeout(releasePlayback" in play, "the cap never lets the hold wedge"
    begin = html.split("function beginTicking", 1)[1].split("function releasePlayback", 1)[0]
    assert "requestAnimationFrame(tick)" in begin and "lastTimestamp = 0" in begin
    release = html.split("function releasePlayback", 1)[1].split("\n      function ", 1)[0]
    assert "state.playbackPending = false" in release and "beginTicking()" in release
    assert "!state.playbackPending || !state.playing" in release, "a pause while waiting cancels the release"
    loaded = html.split("footageVideo.addEventListener('loadeddata'", 1)[1]
    assert "releasePlayback()" in loaded, "the video's first frame releases the hold"
    load = html.split("function onFootageLoad", 1)[1].split("\n      function ", 1)[0]
    assert "state.footageMode === 'mjpeg' && state.footageKind === 'stream'" in load
    assert "releasePlayback()" in load, "the MJPEG stream's first frame releases the same hold"
    error_handler = html.split("function onFootageError", 1)[1].split("function onFootageLoad", 1)[0]
    assert "releasePlayback()" in error_handler, "giving up on the stream must not hold playback forever"
    assert "Starting the footage stream" in html, "the pane says why nothing is moving yet"


def test_the_footage_follows_the_animation_clock() -> None:
    """An MJPEG connection cannot be seeked, so the pane follows the animation in the two ways it can: one
    annotated *still* of the exact second while paused (debounced - a scrub must not fire a decode per pixel), and
    a *stream* opened at that second, at the animation's speed, while playing.

    Two things make the times comparable and are easy to lose: the strip's clock is translated onto the
    recording's (``streamOffset``) before it is asked for, and every path that moves the animation ends in a
    draw, which is where the pane is re-synced - so the animation drives, never the other way round.
    """
    html = COMPONENT.read_text()
    assert "function footageUrl" in html
    assert (
        "/live/" in html and "/frame/" in html and "/stream/" in html
    ), "the live encoded stream for playing, the MJPEG fallback, and the still endpoint for paused"
    assert "state.streamOffset" in html, "the strip clock must be translated onto the recording's"
    assert "function scheduleStill" in html and "FOOTAGE_STILL_DEBOUNCE_MS" in html, "scrubbing must be debounced"
    assert "function syncFootage" in html
    draw_body = html.split("function draw()", 1)[1].split("\n      function ", 1)[0]
    assert "syncFootage();" in draw_body, "every path that moves the animation must move the footage"


def test_a_stream_that_falls_behind_is_restarted_not_left_to_drift() -> None:
    """The decode is capped by how fast the 4K footage decodes, so a playing stream can fall behind the
    animation. The pane restarts it at the animation's own second once the gap is visible - rate-limited, so a
    decode slower than real time cannot turn the pane into a loop of restarts."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_DRIFT_TOLERANCE_S" in html, "a restart costs an encode: only correct a visible divergence"
    assert "FOOTAGE_MIN_RESTART_INTERVAL_MS" in html, "and never restart more often than this"
    assert "function restartFootage" in html
    assert "footageVideo.currentTime" in html, "the drift is measured off the video's own clock, not modeled"


def test_the_footage_sync_can_be_turned_off() -> None:
    """The footage is worth watching on its own - the animation is a reconstruction, the footage is the match -
    so the two can be uncoupled without losing the pane."""
    html = COMPONENT.read_text()
    assert 'id="sync"' in html, "there must be a control to uncouple the two"
    assert "state.footageSync" in html
    assert "footageSync: true" in html, "synced is the useful default"


def test_a_button_restarts_the_footage_at_the_animation_second() -> None:
    """The pane's one real control: drop the current stream and reopen it at the animation's second (the way
    back into step after a stall). It re-couples sync if it was switched off - a button that says "match the
    animation" while leaving them free to drift again would not do what it says."""
    html = COMPONENT.read_text()
    assert 'id="resync-footage"' in html, "the button sits beside play"
    handler = html.split("resyncFootageBtn.addEventListener", 1)[1]
    assert "restartFootage()" in handler, "the button restarts the stream at the animation's own second"
    assert "state.footageSync = true" in handler, "restarting re-couples sync if it was off"


def test_the_footage_opens_clean_and_offers_eight_times_speed() -> None:
    """The pane is for watching and manually tagging the game, so its default picture is the game itself -
    every layer off. That is also what lets the server take the direct ffmpeg path, and 8x is the speed that
    path exists for."""
    html = COMPONENT.read_text()
    assert (
        "footageOverlays: {pitch: false, boxes: false, numbers: false, ball: false, hud: false, debug: false}"
        in html
    ), "the footage must start without overlays"
    for ident in ("ov-pitch", "ov-boxes", "ov-numbers", "ov-ball", "ov-hud", "ov-debug"):
        line = next(line for line in html.splitlines() if f'id="{ident}"' in line)
        assert "checked" not in line, f"{ident} must default to off"
    assert '<option value="8">8x</option>' in html, "8x playback is part of the direct path"


def test_the_footage_overlay_toggles_choose_the_layers() -> None:
    """The footage is drawn server-side, so each toggle is a query parameter on the stream *and* the still URLs:
    switching one off reopens the pane at the animation's own second with the new layer set, rather than leaving
    the previous picture on screen until the next seek. ``none`` is the explicit spelling for a clean frame - an
    empty query value parses as \"absent\" on the server, which means everything on."""
    html = COMPONENT.read_text()
    for ident in ("ov-pitch", "ov-boxes", "ov-numbers", "ov-ball", "ov-hud", "ov-debug"):
        assert f'id="{ident}"' in html, f"a toggle must exist for {ident}"
    assert "state.footageOverlays" in html
    assert "overlays=${overlayParam()}" in html, "the layer set must travel on the URLs"
    assert html.count("overlays=${overlayParam()}") == 3, (
        "the encoded stream, the MJPEG fallback and the still all carry it"
    )
    assert "on.length ? on.join(',') : 'none'" in html, "all layers off is spelled none, not empty"
    handler = html.split("state.footageOverlays[name] = event.target.checked;", 1)[1]
    assert "restartFootage()" in handler, "a new layer set must reopen the footage now"


def test_the_markers_track_ids_follow_the_debug_toggle() -> None:
    """The ``#1234`` on a marker is the tracker's own key, not something read off a shirt: the stream prints
    track ids only in debug, so the animation's do too - the old "shirts" toggle is gone and debug governs.
    A *detected* number is a sighting, not debug information, so it stays on either way, exactly as the
    stream's number chips stay when its debug layer is off.
    """
    html = COMPONENT.read_text()
    assert 'id="labels"' not in html and "state.labels" not in html, "the shirts toggle must be gone"
    text_fn = html.split("function textFor", 1)[1].split("function labelFor", 1)[0]
    assert "state.footageOverlays.debug && major" in text_fn, "the track id must follow the debug toggle"
    assert "entry.number) return" in text_fn, "a detected number must not be caught by the gate"
    handler = html.split("state.footageOverlays[name] = event.target.checked;", 1)[1]
    assert "if (name === 'debug') draw();" in handler, (
        "toggling debug must redraw the pitch, not only reopen the footage"
    )


def test_a_busy_stream_server_is_retried_before_giving_up() -> None:
    """Restarting the footage can lose the race for the server's decode slot (the previous connection's slot is
    only freed when the server notices nobody is reading it): the pane retries with backoff instead of declaring
    the server absent on the first refusal, and a successful frame clears the state."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_RETRY_LIMIT" in html and "FOOTAGE_RETRY_BASE_MS" in html
    error_handler = html.split("function onFootageError", 1)[1].split("function onFootageLoad", 1)[0]
    assert "state.footageRetry < FOOTAGE_RETRY_LIMIT" in error_handler
    assert "openFootage();" in error_handler, "the retry reopens the footage itself, not through a counter reset"
    assert "state.footageError = true" in error_handler, "only exhausted retries call the server absent"
    load_handler = html.split("function onFootageLoad", 1)[1]
    assert "state.footageRetry = 0" in load_handler, "a frame that arrives clears the retry state"


def test_webkit_browsers_get_the_mjpeg_footage_instead_of_the_mp4() -> None:
    """Safari's media stack will not play the pane's endless fragmented MP4: it probes with ``Range: bytes=0-1``
    and expects a 206/Content-Range (Apple's documented requirement), which a stream with no length cannot
    answer - so the pane must not even try it there. WebKit-family browsers (desktop Safari, every iOS browser)
    start on the MJPEG stream in the pane's <img>; Chromium and Gecko keep the encoded stream."""
    html = COMPONENT.read_text()
    assert "function webkitFamily" in html
    detector = html.split("function webkitFamily", 1)[1].split("}", 1)[0]
    assert "/Safari\\//" in detector, "Safari identifies itself as WebKit"
    for rival in ("Chrome\\/", "Chromium\\/", "Edg\\/", "OPR\\/"):
        assert rival in detector, f"a Chromium browser wearing the Safari token must not match ({rival})"
    assert "footageMode: webkitFamily() ? 'mjpeg' : 'video'" in html, "the mode is decided up front"
    mjpeg_block = html.split("if (kind === 'mjpeg')", 1)[1].split("// The live encoded stream", 1)[0]
    assert "/stream/${state.matchId}.mjpg" in mjpeg_block
    for param in (
        "start=${source.toFixed(1)}",
        "rate=${state.speed}",
        "&width=${width}",
        "overlays=${overlayParam()}",
        "token=${state.footageToken}",
    ):
        assert param in mjpeg_block, f"the MJPEG URL is missing {param}"


def test_a_browser_that_refuses_the_video_falls_back_to_mjpeg() -> None:
    """WebKit starts on MJPEG, but any other stack can refuse /live too. After a few video errors - and only
    once a still has loaded, so a dead server still reads as "not running" - the pane switches to the MJPEG
    stream itself instead of calling the server absent; a decoded video frame clears the failure count."""
    html = COMPONENT.read_text()
    assert "FOOTAGE_VIDEO_FALLBACK_FAILS" in html
    error_handler = html.split("function onFootageError", 1)[1].split("function onFootageLoad", 1)[0]
    assert "event.target === footageVideo" in error_handler, "only the video element's failures flip the mode"
    assert "state.footageMode = 'mjpeg'" in error_handler
    assert "footage.complete &&" in error_handler and "footage.naturalWidth > 0" in error_handler, (
        "the still must prove the server is up before the mode is blamed"
    )
    loaded = html.split("footageVideo.addEventListener('loadeddata'", 1)[1]
    assert "state.footageVideoFails = 0" in loaded, "a decoded frame clears the failure count"


def test_the_mjpeg_mode_shares_the_pane_image_and_reports_no_clock() -> None:
    """In the fallback mode the pane's existing <img> carries both the MJPEG stream and the paused still (the
    video element goes unused), there is no currentTime to measure drift against, and the audio toggle is
    disabled - the fallback stream has no audio, and the note under the pane says so."""
    html = COMPONENT.read_text()
    opener = html.split("function openFootage", 1)[1].split("function restartFootage", 1)[0]
    assert "setFootageSource('stream', footageUrl('mjpeg', state.time))" in opener
    sync_body = html.split("function syncFootage", 1)[1].split("\n      function ", 1)[0]
    assert "state.footageMode === 'mjpeg'" in sync_body, "the drift check belongs to the video's own clock"
    visibility = html.split("function renderFootageVisibility", 1)[1].split("\n      function ", 1)[0]
    assert "state.footageMode === 'mjpeg'" in visibility
    assert "footage.style.display = enabled ? 'block'" in visibility, "the <img> is the stream and the still"
    assert "soundToggle.disabled = true" in html, "the fallback has no audio: the toggle must say so"
    assert "no audio" in html, "the pane note must be honest about what the fallback loses"


def test_a_superseded_stream_is_aborted_not_just_ignored() -> None:
    """Chromium keeps *draining* an MJPEG fetch while its <img> exists, so merely removing the src leaves every
    superseded stream decoding - here and on the server, holding one of its slots - with nobody watching. Every
    source change therefore clears the src (which aborts the fetch) and swaps in a fresh element."""
    html = COMPONENT.read_text()
    assert "function replaceFootageImage" in html
    body = html.split("function replaceFootageImage", 1)[1].split("function setFootageSource", 1)[0]
    assert "removeAttribute('src')" in body, "clearing the src is what aborts the fetch"
    assert "replaceWith(next)" in body, "the drained element must be replaced, not reused"
    setter = html.split("function setFootageSource", 1)[1].split("\n      function ", 1)[0]
    assert "replaceFootageImage()" in setter, "every source change goes through the swap"


def test_an_abandoned_stream_is_stopped_by_the_server_not_left_draining() -> None:
    """A browser will not abort an MJPEG fetch on command - Chromium keeps draining a replaced <img> - so the
    pane tells the server to end the stream it is abandoning: before every replacement, and via a pagehide
    beacon when the page goes away. Without it every seek/toggle would leave a decode running."""
    html = COMPONENT.read_text()
    assert "function stopFootageStream" in html
    assert "/stop?token=" in html
    assert "sendBeacon" in html, "the page says goodbye on unload"
    assert "footageToken" in html, "each stream identifies itself"
    opener = html.split("function openFootage", 1)[1].split("function restartFootage", 1)[0]
    assert "stopFootageStream();" in opener, "a stream must be stopped before its replacement starts"
    assert "newFootageToken()" in opener, "the replacement gets its own token"


def test_a_repeated_render_does_not_reload_the_still() -> None:
    """Streamlit re-sends the args while a component stays unchanged, and every payload redraws the pane. While
    paused that must not swap the still <img>: the reload would abort a decode in flight for a picture that is
    already on screen - a 4K seek spent per rerun. A still that *failed* is the exception: its retry must go
    through, so a loaded-with-no-pixels image is not skipped."""
    html = COMPONENT.read_text()
    setter = html.split("function setFootageSource", 1)[1].split("\n      function ", 1)[0]
    assert "url === footage.getAttribute('src')" in setter, "the same request twice is a no-op"
    assert "!failed" in setter, "except when the picture failed to load - that retry has to proceed"


def test_the_panes_clock_comes_from_the_video_itself() -> None:
    """The pane no longer *infers* where the stream is: a <video> element reports its own timeline, so the drift
    check compares the animation's clock against a measurement - the stream's zero is the strip second it was
    asked for, and its clock starts when the first *frame* arrives, not when the request was made."""
    html = COMPONENT.read_text()
    assert "footageVideo.currentTime * state.footageRate" in html, "the display position is measured, not modeled"
    assert "state.footageOpenStrip = state.time" in html, "the stream's zero is the strip second it was asked for"
    opener = html.split("function openFootage", 1)[1].split("function restartFootage", 1)[0]
    assert "state.footageAwaitingFirstFrame = true" in opener, "the wait begins with the request"
    load_handler = html.split("footageVideo.addEventListener('loadeddata'", 1)[1]
    assert "state.footageAwaitingFirstFrame = false" in load_handler, "frame one ends the wait"
    assert "state.footageOpenStrip = state.time" in load_handler, (
        "the encoder's start-up second is re-anchored away, or a stream in step reads as permanently behind"
    )
    sync_body = html.split("function syncFootage", 1)[1].split("\n      function ", 1)[0]
    assert "state.footageAwaitingFirstFrame || footageVideo.readyState < 2" in sync_body, (
        "no drift verdict before the first frame can arrive"
    )


def test_the_playing_footage_is_full_motion_video_with_optional_sound() -> None:
    """While the animation plays, the pane plays the encoded live stream - full frame rate and the match's own
    audio - in a <video> element; the still stays up until the stream's first frame, so an encoder start-up does
    not flash black. The sound starts off (browsers block audible autoplay) and the box is the gesture that
    allows it."""
    html = COMPONENT.read_text()
    assert '<video id="footage-video"' in html, "the live stream needs a video element"
    assert "function openLiveVideo" in html and "function closeLiveVideo" in html
    assert "footageVideo.src = url" in html and "footageVideo.load()" in html, "clearing the src aborts the fetch"
    assert "const FOOTAGE_FPS = 30" in html
    assert "&fps=${FOOTAGE_FPS}&rate=${state.speed}" in html, "the live URL carries the pane's own rate"
    assert "readyState >= 2" in html, "the still gives way only once the stream has a frame"
    assert 'id="sound"' in html and "footageVideo.muted = !state.footageSound" in html


def test_the_moment_jump_follows_changing_sequences_only() -> None:
    """Python's moment picker moves the *animation* to the chosen second - the stream follows the animation, so
    one jump moves both. The jump arrives as a target plus a sequence number, and only a changed sequence acts:
    an unrelated rerun (a scan's progress, a table redraw) must not yank the playback back to the moment."""
    html = COMPONENT.read_text()
    assert "args.seek_to_s" in html and "args.seek_seq" in html, "the jump arrives as target plus sequence"
    assert "state.seekSeq" in html
    assert "pendingSeek" in html, "a jump that arrives before the payload finishes loading must still land"
    handler = html.split("seekSeq && seekSeq !== state.seekSeq", 1)
    assert len(handler) == 2, "the sequence, not the mere presence of a target, decides when to act"
    assert "setPlaying(true)" in handler[1], "jumping to a moment starts playback at it"

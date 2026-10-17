"""Smoke test: the dashboard must render end to end without raising.

The page does real work as it renders - it probes the video, reads the segment status and loads saved artifacts - so
this catches the class of error that only appears when the script actually runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "src" / "soccer_analytics" / "dashboard" / "app.py"


def test_the_export_button_callback_takes_every_argument_positionally() -> None:
    """Streamlit's ``on_click=...`` passes ``args`` positionally.

    A keyword-only signature here looks tidier but fails at click time with "takes 5 positional arguments but 12
    were given" - the export button must be callable with exactly the tuple it passes.
    """
    tree = ast.parse(APP.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_video_export"
    )
    assert not function.args.kwonlyargs, "the export callback must accept the button's arguments positionally"
    assert len(function.args.args) == 12


def _no_footage(app) -> bool:
    """Whether the page gave up because this machine has no video to analyze.

    Checked by message rather than by `app.error` being non-empty: the page also raises *content* errors, such as a
    pitch calibration that came out unusable, and those are the page working, not the page failing.
    """
    return any("No video files found" in str(block.value) for block in app.error)


def test_dashboard_renders_without_error() -> None:
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]

    titles = [t.value for t in app.title]
    assert "Match analysis" in titles

    # Either there is no footage (reported honestly) or Step 1 offers footage and formats downstream.
    if _no_footage(app):
        pytest.skip("no footage available in this environment")
    headers = [h.value for h in app.header]
    assert any("Step 1" in h for h in headers)
    assert any("Videos (newest first)" == m.label for m in app.multiselect)


def test_dashboard_offers_the_next_step_either_way() -> None:
    """The page must either offer Step 3 or say what Step 3 is waiting for - never just be silent.

    Deliberately not asserting which: the answer depends on what has already been analyzed on this machine, and a
    test that only passes on a clean workspace is a test that fails the moment someone uses the tool.
    """
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    if _no_footage(app):
        pytest.skip("no footage available in this environment")

    messages = " ".join(str(block.value) for block in app.info)
    messages += " ".join(str(block.value) for block in app.warning)
    buttons = [button.label for button in app.button]
    actionable = "Build report" in buttons
    explained = any(phrase in messages for phrase in ("Run Step 1 first", "Steps 1 and 2", "Not analyzed yet"))
    assert actionable or explained, f"Step 3 said neither. buttons={buttons} messages={messages!r}"


def test_the_option_sidebar_is_gone_and_the_one_press_build_is_offered() -> None:
    """The preview-detail sidebar was removed - every clip is full quality now - and Step 3 offers the one-press
    "Build report + run all detections" beside "Build report" whenever it can build at all."""
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert not list(app.sidebar.radio), "the sidebar still offers options - it was to be removed entirely"
    if _no_footage(app):
        pytest.skip("no footage available in this environment")
    buttons = [button.label for button in app.button]
    if "Build report" not in buttons:
        pytest.skip("this machine has no analyzed, calibrated match to build a report for")
    assert "Build report + run all detections" in buttons


def test_the_unique_players_panel_renders_when_a_report_exists() -> None:
    """The centered-clip panel must render against real artifacts - or say why it cannot.

    Data-dependent on purpose: on a machine with a match whose report and replay are built, the panel appears with
    its player list; on a clean machine there is nothing to show and the test skips. What it guards is the wiring
    between the replay payload (which carries each player's own boxes), the grouping by shirt number, and the
    clip-cutting controls - the parts that only fail when they are actually rendered.
    """
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    if _no_footage(app):
        pytest.skip("no footage available in this environment")
    clip_box = next((m for m in app.multiselect if m.label == "Videos (newest first)"), None)
    if clip_box is None:
        pytest.skip("the page did not offer footage to analyze")
    # The session's footage is picked in Step 1 now; the segment for the combined game is the one the whole
    # pipeline runs on, and any of its clips opens it.
    game = next((option for option in clip_box.options if "game_" in str(option)), None)
    if game is None:
        pytest.skip("no combined game video on this machine")
    clip_box.set_value([game])
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    labels = [expander.label for expander in app.expander]
    if not any("Unique players" in label for label in labels):
        pytest.skip("this machine has no built report for the game video")
    panel = next(expander for expander in app.expander if "Unique players" in expander.label)
    player_box = next(box for box in panel.selectbox if box.label == "Player to cut a clip around")
    assert len(player_box.options) > 0, "the panel offers no player to cut a clip around"
    # The appearances offered are the chosen player's own tracks, so the second list is never empty either.
    appearance_box = next(box for box in panel.selectbox if box.label == "Which appearance to cut")
    assert len(appearance_box.options) > 0
    # The page is roster-centred: one roster link per team (the scan supplies numbers, the linked team roster
    # supplies names), and the per-track table is only the correction path.
    assert any("Team rosters" in label for label in labels), "the roster editors must render beside the replay"
    roster_boxes = [box for box in app.selectbox if box.label == "Roster"]
    assert len(roster_boxes) == 2, "one roster choice per team"
    assert all("(no roster)" in box.options for box in roster_boxes), "a team can be left without a roster"


def _media_url_variables(tree) -> set[str]:  # noqa: ANN001 - ast.Module
    """Names bound to the result of ``_served_video_url(...)`` anywhere in a module.

    The bug's shape was ``url = _served_video_url(path, key) or ""`` - the ``or ""`` wrap is why the check looks
    through boolean operands instead of only at a direct call.
    """
    import ast

    def holds_url(value) -> bool:  # noqa: ANN001
        if isinstance(value, ast.Call):
            return getattr(value.func, "id", "") == "_served_video_url"
        if isinstance(value, ast.BoolOp):
            return any(holds_url(operand) for operand in value.values)
        return False

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BoolOp | ast.Call):
            if holds_url(node.value):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def test_the_media_url_check_catches_the_bug_it_exists_for() -> None:
    """A guard that cannot fail is not a guard: the detector must flag the exact code that broke the page."""
    import ast

    snippet = 'url = _served_video_url(target, "k") or ""\nif target.exists():\n    st.video(url)\n'
    tree = ast.parse(snippet)
    assert _media_url_variables(tree) == {"url"}
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "video"
        and getattr(node.func.value, "id", "") == "st"
        and any(isinstance(arg, ast.Name) and arg.id in _media_url_variables(tree) for arg in node.args)
    ]
    assert offenders == [3], "the guard did not catch the pattern that produced the live error"


def test_streamlits_video_player_is_never_given_a_media_url() -> None:
    """``st.video`` takes a path (or a real URL): a ``/media/<hash>`` string is read as a *local file path*.

    Learned live, not in theory: the centered-clip panel registered a clip through the media endpoint and then
    handed that URL to ``st.video``, which tried to open the URL as a path and failed with
    ``MediaFileStorageError: Error opening '/media/<hash>.mp4'`` - so a clip that cut perfectly was reported as a
    page-breaking error on every rerun. The endpoint URL is only for callers that fetch it themselves (the replay
    component's clip pane); ``st.video`` must be given the file. This pins that no call site does it again.
    """
    import ast

    tree = ast.parse(APP.read_text())
    # There may be no registered media URLs at all (the marking proxy's was removed with the never-merge
    # workflow); the detector itself is pinned by the test above, so an empty set is fine here.
    url_names = _media_url_variables(tree)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "video"
        and getattr(node.func.value, "id", "") == "st"
        and any(isinstance(arg, ast.Name) and arg.id in url_names for arg in node.args)
    ]
    assert not offenders, f"st.video() was given a media URL at line(s) {offenders}: pass the file path instead"


def test_the_marking_step_offers_the_stream_start_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screen that needs the footage server must offer the one press that starts it, not just name a script.

    The marking step renders frames straight from the camera clips through the stream server; a fresh machine -
    or one after a reboot - has no server yet, and being told to go run a script, with no button and no path
    from here, is a dead end. ``SOCCER_STREAM_PORT`` is pointed at a port nothing listens on so the
    not-running branch renders even on a machine where a server happens to be up.
    """
    monkeypatch.setenv("SOCCER_STREAM_PORT", "8599")
    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    if _no_footage(app):
        pytest.skip("no footage available in this environment")
    if not any("marking stream" in str(block.value) for block in app.info):
        pytest.skip("this machine has no game to mark a clock on")
    keys = [str(button.key) for button in app.button]
    assert any(key.startswith("start_mark_stream::") for key in keys), f"no start button in the marking step: {keys}"

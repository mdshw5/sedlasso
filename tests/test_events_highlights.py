"""Whistle detection, highlight selection/export and the match archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soccer_analytics.analysis.events import (
    DETECTED_EVENT_TYPES,
    EVENT_TYPES,
    LOW_GAIN_MAX,
    VERDICT_FALSE,
    VERDICT_TRUE,
    VOICE_SHARE_MAX,
    Event,
    EventLog,
    Whistle,
    detect_whistles,
    whistles_to_events,
)
from soccer_analytics.analysis import highlights as highlights_lib
from soccer_analytics.analysis.highlights import (
    LEAD_S,
    PREVIEW_SETTINGS,
    TAIL_S,
    TIER_SECONDS,
    build_moments,
    clamp_moment,
    export_moment,
    export_reel,
    moment_for_event,
    moment_on_source,
    preview_clip_name,
    reel_manifest,
    select_reel,
    write_manifest,
)
from soccer_analytics.analysis.library import MatchLibrary
from soccer_analytics.geometry.pitch_calibration import PitchCalibration
from soccer_analytics.ingest.ffmpeg_reader import probe_video
from soccer_analytics.ingest.video_reader import VideoWriter

SR = 16000


def _tone(freq: float, duration: float, amplitude: float = 0.35) -> np.ndarray:
    time = np.arange(int(duration * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _speech_like(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Harmonic stacks with formant-ish peaks: what the real match audio mostly is."""
    time = np.arange(int(duration * SR)) / SR
    out = np.zeros_like(time)
    for harmonic in range(1, 9):
        out += (0.25 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * time + rng.uniform(0, 6))
    out *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * time)
    out += rng.normal(0.0, 0.02, len(time))
    return (0.3 * out).astype(np.float32)


def _shouted_vowel(duration: float = 0.6, f0: float = 190.0) -> np.ndarray:
    """A coach's yell: a voiced harmonic stack with formants that put real energy in the whistle band.

    The formants at 1.8 kHz and 3.6 kHz lift the upper harmonics well above the 1/k rolloff, so this is loud in the
    2.2-4.6 kHz band and passes the prominence gate exactly like the false positives on the real footage.
    """
    time = np.arange(int(duration * SR)) / SR
    out = np.zeros_like(time)
    harmonic = 1
    while harmonic * f0 < 7800.0:
        formant = 1.0 + 4.0 * np.exp(-((harmonic * f0 - 1800.0) / 900.0) ** 2) + 2.0 * np.exp(-((harmonic * f0 - 3600.0) / 900.0) ** 2)
        out += (1.0 / harmonic) * formant * np.sin(2 * np.pi * harmonic * f0 * time)
        harmonic += 1
    out *= 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * time)  # the syllable envelope of a shout
    return (0.4 * out / np.max(np.abs(out))).astype(np.float32)


def test_detects_whistles_in_speech_heavy_audio() -> None:
    """The real match is dominated by speech near the microphone, so loudness alone finds nothing."""
    rng = np.random.default_rng(0)
    audio = _speech_like(12.0, rng)
    for at, freq, length in ((2.0, 3800.0, 0.45), (7.5, 3600.0, 0.35)):
        start = int(at * SR)
        audio[start : start + len(_tone(freq, length))] += _tone(freq, length)
    # Speech is loud; make sure the detector is not simply picking the loudest moments.
    audio[int(4.5 * SR) : int(5.5 * SR)] *= 3.0

    whistles = detect_whistles(audio, SR)
    times = [w.time_s for w in whistles]
    assert len(whistles) >= 2, f"expected both whistles, got {times}"
    assert any(abs(t - 2.0) < 0.25 for t in times), times
    assert any(abs(t - 7.5) < 0.25 for t in times), times
    assert all(2500 < w.frequency_hz < 4500 for w in whistles)
    assert not any(4.5 < t < 5.6 for t in times), f"flagged the loud speech section: {times}"


def test_a_quiet_tonal_blip_is_not_a_whistle() -> None:
    """The gate that matters, and the numbers come from the real 4K sample.

    On five minutes of that footage the old detector reported 191 candidates: tonal blips sitting around 17x the
    match's own level in the whistle band - shouts, kicks, and whistles from the pitches next door - while the loud
    sustained blasts from the pitch being filmed ran from 100x to over 1000x. Both ends of that are built here, with
    the amplitudes calibrated so the prominences match what was measured.
    """
    rng = np.random.default_rng(4)
    quiet = _speech_like(14.0, rng)
    blip = _tone(3800.0, 0.5, amplitude=0.015)  # measured: ~11x the match median, i.e. the false-positive range
    quiet[int(6.0 * SR) : int(6.0 * SR) + len(blip)] += blip
    assert detect_whistles(quiet, SR) == [], "a blip this quiet is not a whistle on this pitch"

    rng = np.random.default_rng(4)
    loud = _speech_like(14.0, rng)
    blast = _tone(3800.0, 0.5, amplitude=0.20)  # measured: ~1800x, the range real blasts sit in
    loud[int(6.0 * SR) : int(6.0 * SR) + len(blast)] += blast
    found = detect_whistles(loud, SR)
    assert len(found) == 1, [w.time_s for w in found]
    assert found[0].prominence > 100.0, "the prominence is what tells the user why it was reported"


def test_blasts_close_together_are_one_candidate() -> None:
    """A whistle and the shouting that follows it are one event happening, not a fan of candidates.

    Reporting each blip separately is what made the feature unusable: on the real sample the candidates arrived in
    clusters of three to six within a few seconds of each other.
    """
    def with_two_blasts(gap_s: float) -> list:
        rng = np.random.default_rng(7)
        audio = _speech_like(16.0, rng)
        for at in (6.0, 6.0 + gap_s):
            tone = _tone(3800.0, 0.4, amplitude=0.2)
            start = int(at * SR)
            audio[start : start + len(tone)] += tone
        return detect_whistles(audio, SR)

    close = with_two_blasts(0.6)
    assert len(close) == 1, f"a second blast 0.6 s later is the same event: {[w.time_s for w in close]}"
    apart = with_two_blasts(2.5)
    assert len(apart) == 2, f"two blasts 2.5 s apart are two events: {[w.time_s for w in apart]}"


def test_a_short_click_is_not_a_whistle() -> None:
    """A blast is held; a click is not. 0.06 s used to be enough, which is about the length of a kick."""
    rng = np.random.default_rng(11)
    audio = _speech_like(10.0, rng)
    click = _tone(3500.0, 0.06, amplitude=0.4)
    start = int(5.0 * SR)
    audio[start : start + len(click)] += click
    assert detect_whistles(audio, SR) == []


def test_quiet_audio_yields_no_whistles() -> None:
    rng = np.random.default_rng(3)
    assert detect_whistles(_speech_like(6.0, rng), SR) == []
    assert detect_whistles(np.zeros(1000, dtype=np.float32), SR) == []


def _with_shout(rng: np.random.Generator, duration: float = 4.0) -> np.ndarray:
    audio = rng.normal(0.0, 0.001, int(duration * SR)).astype(np.float32)
    shout = _shouted_vowel()
    start = int(1.5 * SR)
    audio[start : start + len(shout)] += shout
    return audio


def test_a_shouted_vowel_is_rejected_even_though_it_clears_the_prominence_gate() -> None:
    """The yell must pass the old gates first, otherwise this test would prove nothing.

    A shouted vowel has real harmonics in the whistle band and is loud, which is exactly why the detector kept
    reporting coaches yelling as stoppages. What separates it from a whistle is where the rest of the energy is.
    """
    rng = np.random.default_rng(5)
    audio = _with_shout(rng)

    # The low-frequency gate is switched off here so this checks the *older* gates: the shout has to be caught by
    # the voice-share test, which is what this test is about.
    unfiltered = detect_whistles(audio, SR, min_prominence=5.0, max_voice_share=None, max_low_gain=None)
    assert any(abs(w.time_s - 1.5) < 0.4 for w in unfiltered), (
        "the shouted vowel should clear the prominence/tonality gates; this test needs that to be true"
    )

    filtered = detect_whistles(audio, SR, min_prominence=5.0)
    assert not any(abs(w.time_s - 1.5) < 0.4 for w in filtered), "a coach's yell was reported as a whistle"


def test_a_pure_tone_survives_the_voice_rejection() -> None:
    rng = np.random.default_rng(6)
    audio = rng.normal(0.0, 0.001, int(4.0 * SR)).astype(np.float32)
    tone = _tone(3800.0, 0.5)
    start = int(1.5 * SR)
    audio[start : start + len(tone)] += tone

    kept = [w for w in detect_whistles(audio, SR, min_prominence=5.0) if abs(w.time_s - 1.5) < 0.4]
    assert kept, "a loud pure tone must survive the voice rejection"
    assert kept[0].voice_share < VOICE_SHARE_MAX, kept[0].voice_share
    assert kept[0].low_gain < LOW_GAIN_MAX, kept[0].low_gain
    assert kept[0].frequency_hz == pytest.approx(3800.0, abs=120.0)


def _voiced_sound(f0: float, duration: float, formants: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    """A voiced sound: harmonics of ``f0`` shaped by (center Hz, width Hz, boost) formants."""
    time = np.arange(int(duration * SR)) / SR
    out = np.zeros_like(time)
    harmonic = 1
    while harmonic * f0 < 7800.0:
        frequency = harmonic * f0
        gain = 1.0 / harmonic
        for center, width, boost in formants:
            gain += boost * np.exp(-((frequency - center) / width) ** 2) / harmonic
        out += gain * np.sin(2 * np.pi * frequency * time)
        harmonic += 1
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 3.5 * time)  # a syllable's own shape
    out = out * envelope
    return (0.35 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def test_a_female_shout_is_rejected_by_the_low_frequency_gate() -> None:
    """The case the old voice gate missed, and the user's labels named: a *female* coach shouting.

    A female voice has higher formants, so a shout can put most of its energy inside the whistle band - the old
    "share of energy below 1.2 kHz" test reads under its ceiling and lets it through. What still gives her away is
    that the shout *brings* low-frequency energy with it: the gate compares the region below the band during the
    blast against the seconds around it, and a lone tone adds nothing there.
    """
    rng = np.random.default_rng(8)
    audio = _speech_like(6.0, rng)
    shout = _voiced_sound(240.0, 0.4, ((900.0, 250.0, 2.0), (2300.0, 300.0, 3.0), (3300.0, 400.0, 9.0)))
    start = int(3.0 * SR)
    audio[start : start + len(shout)] += shout

    # It clears every gate that was there before this test was written, or the test proves nothing.
    old_gates = detect_whistles(audio, SR, min_prominence=5.0, max_voice_share=None, max_low_gain=None)
    assert any(abs(w.time_s - 3.0) < 0.4 for w in old_gates), "the shout must clear the older gates"

    kept = detect_whistles(audio, SR, min_prominence=5.0, max_voice_share=None)
    assert not any(abs(w.time_s - 3.0) < 0.4 for w in kept), "a female shout was reported as a whistle"

    # And with the gate switched off it comes back, so the gate is what removed it.
    assert any(
        abs(w.time_s - 3.0) < 0.4
        for w in detect_whistles(audio, SR, min_prominence=5.0, max_voice_share=None, max_low_gain=None)
    )


def test_a_raptor_call_is_rejected_by_the_low_frequency_gate() -> None:
    """A bird of prey's call: a rich, gliding tone whose fundamental sits *above* the voice ceiling.

    The user's labels named birds as the other false positive. Their call is tonal and sustained, so the tonality,
    duration and loudness gates all pass, and a fundamental at 1.4-1.6 kHz is above the 1.2 kHz voice ceiling - the
    old share gate cannot see it at all. It still drags its own harmonics up in the region below the whistle band,
    which is what the differential gate measures.
    """
    rng = np.random.default_rng(9)
    audio = _speech_like(6.0, rng)
    duration, sweep = 0.45, np.linspace(1600.0, 1350.0, int(0.45 * SR))  # a descending mew
    time = np.arange(sweep.size) / SR
    phase = 2 * np.pi * np.cumsum(sweep) / SR
    call = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    envelope = np.minimum(1.0, time / 0.06) * np.minimum(1.0, np.maximum(0.0, (duration - time) / 0.1))
    call = (0.32 * call * envelope).astype(np.float32)
    start = int(3.0 * SR)
    audio[start : start + call.size] += call

    old_gates = detect_whistles(audio, SR, min_prominence=5.0, max_voice_share=None, max_low_gain=None)
    assert any(abs(w.time_s - 3.0) < 0.4 for w in old_gates), "the call must clear the older gates"

    kept = detect_whistles(audio, SR, min_prominence=5.0)
    assert not any(abs(w.time_s - 3.0) < 0.4 for w in kept), "a bird's call was reported as a whistle"
    assert any(
        abs(w.time_s - 3.0) < 0.4
        for w in detect_whistles(audio, SR, min_prominence=5.0, max_low_gain=None)
    ), "switching the gate off must bring the call back"


def test_whistles_become_review_candidates_not_goals() -> None:
    rng = np.random.default_rng(1)
    audio = _speech_like(8.0, rng)
    audio[int(3.0 * SR) : int(3.0 * SR) + len(_tone(3900, 0.25))] += _tone(3900, 0.25)
    events = whistles_to_events(detect_whistles(audio, SR))
    assert events, "no candidate produced"
    assert all(event.source == "audio" and event.type == "other" for event in events)
    assert all("candidate" in event.note for event in events)


def test_event_log_round_trips_and_validates() -> None:
    log = EventLog()
    log.add(Event(time_s=10.0, type="goal", team=0, note="header"))
    log.add(Event(time_s=5.0, type="other", source="audio", confidence=0.4))
    assert [e.time_s for e in log.events] == [5.0, 10.0]  # sorted
    with pytest.raises(ValueError, match="unknown event type"):
        log.add(Event(time_s=1.0, type="touchdown"))


def test_the_detected_event_types_are_all_in_the_vocabulary() -> None:
    """The detectors' output has to be storable: every type they emit must be a known event type."""
    assert set(DETECTED_EVENT_TYPES) <= set(EVENT_TYPES)
    for event_type in DETECTED_EVENT_TYPES:
        EventLog().add(Event(time_s=1.0, type=event_type, source="ball"))


def test_player_attribution_survives_the_json_round_trip() -> None:
    """A detected event names the player it is attributed to; the number is optional and stays optional."""
    event = Event(time_s=12.0, type="tackle", team=0, source="ball", player_track=7, player_number=9)
    restored = Event.from_json(event.to_json())
    assert restored.player_track == 7 and restored.player_number == 9

    anonymous = Event.from_json(Event(time_s=1.0, type="shot", source="ball").to_json())
    assert anonymous.player_track is None and anonymous.player_number is None


def test_a_penalty_outranks_a_shot_and_a_goal_outranks_both() -> None:
    """The new types have to be ranked, or the reels would treat a penalty like any other moment."""
    goal = moment_for_event(Event(time_s=10.0, type="goal", source="ball"))
    penalty = moment_for_event(Event(time_s=10.0, type="penalty", source="ball"))
    shot = moment_for_event(Event(time_s=10.0, type="shot", source="ball"))
    corner = moment_for_event(Event(time_s=10.0, type="corner", source="ball"))
    tackle = moment_for_event(Event(time_s=10.0, type="tackle", source="ball"))
    assert goal.weight > penalty.weight > shot.weight > corner.weight > tackle.weight


def test_the_goals_reel_uses_penalties_when_there_are_no_goals() -> None:
    moments = [
        moment_for_event(Event(time_s=10.0, type="penalty", source="ball")),
        moment_for_event(Event(time_s=100.0, type="shot", source="ball")),
    ]
    reel = select_reel("goals", moments)
    assert [m.event_type for m in reel.moments] == ["penalty"]


def test_rescanning_does_not_duplicate_detected_events() -> None:
    """The scan is deterministic, so a second pass must not append a second copy of every candidate."""
    log = EventLog()
    first = [Event(time_s=10.0, type="other", source="audio"), Event(time_s=40.0, type="other", source="audio")]
    assert log.add_detected(first) == 2

    # The same audio scanned again, with the times a little different: nothing new.
    again = [Event(time_s=10.3, type="other", source="audio"), Event(time_s=40.1, type="other", source="audio")]
    assert log.add_detected(again) == 0
    assert len(log.events) == 2

    # A genuinely new stoppage still lands, and the list stays time-ordered.
    assert log.add_detected([Event(time_s=12.0, type="other", source="audio")]) == 1
    assert [e.time_s for e in log.events] == [10.0, 12.0, 40.0]


def test_a_manual_tag_does_not_hide_a_detected_candidate() -> None:
    """A goal tagged by hand and a whistle at the same moment are two events, not one."""
    log = EventLog()
    log.add(Event(time_s=10.0, type="goal", team=0))
    assert log.add_detected([Event(time_s=10.2, type="other", source="audio")]) == 1
    assert len(log.events) == 2

    with pytest.raises(ValueError, match="add_detected"):
        log.add_detected([Event(time_s=30.0, type="goal")])


def test_discarding_auto_detected_events_keeps_the_manual_tags() -> None:
    """The scan's candidates are a review queue: clearing them must not touch what a human tagged."""
    log = EventLog()
    log.add(Event(time_s=10.0, type="goal", team=0, note="header"))
    log.add(Event(time_s=30.0, type="other", source="audio", confidence=0.4))
    log.add(Event(time_s=31.5, type="other", source="audio", confidence=0.5))
    log.add(Event(time_s=45.0, type="other", source="model", confidence=0.2))
    log.add(Event(time_s=90.0, type="save", team=1))

    # Anything a human did not tag counts as detected, not only what the whistle scan produced.
    assert [e.time_s for e in log.detected()] == [30.0, 31.5, 45.0]
    assert [e.time_s for e in log.manual()] == [10.0, 90.0]

    assert log.discard_detected() == 3
    assert [e.time_s for e in log.events] == [10.0, 90.0]
    assert log.detected() == []

    # Nothing left to discard is not an error - the option is simply not offered.
    assert log.discard_detected() == 0
    assert [e.time_s for e in log.events] == [10.0, 90.0]


def test_moment_for_event_shapes_one_window_around_the_tag() -> None:
    event = Event(time_s=42.0, type="goal", team=1, note="header")
    moment = moment_for_event(event)
    assert moment.start_s == pytest.approx(42.0 - LEAD_S)
    assert moment.end_s == pytest.approx(42.0 + TAIL_S)
    assert moment.time_s == pytest.approx(42.0)
    assert moment.event_type == "goal" and moment.team == 1
    assert "header" in moment.reason
    # The preview and the reel must cut the same seconds, so build_moments has to use the same mapping.
    assert build_moments([event])[0].start_s == moment.start_s
    assert build_moments([event])[0].end_s == moment.end_s
    # A tag before the lead-in must not produce a negative start.
    assert moment_for_event(Event(time_s=1.0, type="shot")).start_s == 0.0


def test_a_detected_whistle_is_clipped_like_any_other_moment() -> None:
    """Ten seconds, the blast four seconds in. The window is the same for every source.

    Speed for a review comes from the preview *mode* instead (`PREVIEW_SETTINGS`): a longer window made every clip
    bigger without answering the question the review asks, which is only whether the sound is a whistle.
    """
    detected = moment_for_event(Event(time_s=100.0, type="other", source="audio", note="whistle"))
    tagged = moment_for_event(Event(time_s=100.0, type="goal", team=0))
    for moment in (detected, tagged):
        assert moment.start_s == pytest.approx(100.0 - LEAD_S)
        assert moment.end_s == pytest.approx(100.0 + TAIL_S)
    assert detected.end_s - detected.start_s == pytest.approx(10.0)
    # The reel cuts the same seconds as the preview.
    assert build_moments([Event(time_s=100.0, type="other", source="audio")])[0].end_s == detected.end_s


def test_preview_modes_trade_the_picture_for_speed() -> None:
    """The three costs the review can pick from, and the cached files they must not share.

    Lowering the *resolution* is not what makes a preview cheap - measured on the real 4K60 game, a 640 px clip
    costs the same ten seconds as a 1280 px one, because decoding is the cost. Dropping frames is the lever, so the
    cheap picture mode must actually skip frames.
    """
    assert set(PREVIEW_SETTINGS) == {"audio", "quick", "full"}
    assert PREVIEW_SETTINGS["audio"]["audio_only"] is True
    assert PREVIEW_SETTINGS["quick"]["skip_frame"] == "nokey", "the cheap picture mode must skip frames"
    assert PREVIEW_SETTINGS["full"]["skip_frame"] is None, "the full mode must decode every frame"
    assert PREVIEW_SETTINGS["quick"]["width"] < PREVIEW_SETTINGS["full"]["width"]
    assert PREVIEW_SETTINGS["quick"]["fps"] < PREVIEW_SETTINGS["full"]["fps"]
    # The sound alone must be MP3: the browser this runs in cannot decode AAC audio.
    assert PREVIEW_SETTINGS["audio"]["audio_codecs"] == ("mp3",)

    moment = moment_for_event(Event(time_s=10.0, type="other", source="audio"))
    names = {mode: preview_clip_name(moment, mode) for mode in PREVIEW_SETTINGS}
    assert names["audio"].endswith(".mp3") and names["full"].endswith(".mp4")
    assert len(set(names.values())) == 3, "the modes must not serve each other's cached file"
    with pytest.raises(ValueError, match="unknown preview mode"):
        preview_clip_name(moment, "cinema")


def test_whistle_candidates_remember_the_recording_they_came_from(tmp_path: Path) -> None:
    """A candidate's seconds belong to the file that was scanned - not to whatever video is selected later."""
    scanned = whistles_to_events(
        [Whistle(time_s=5.0, duration_s=0.3, frequency_hz=3800.0, score=1.0, prominence=200.0)],
        video=tmp_path / "game.mp4",
    )
    assert scanned and all(e.video == str((tmp_path / "game.mp4").resolve()) for e in scanned)
    assert moment_for_event(scanned[0]).video == scanned[0].video
    # Without a video the candidate carries nothing, and a hand tag has no recording of its own either: it was
    # tagged on whatever the page was showing.
    assert whistles_to_events([Whistle(time_s=1.0, duration_s=0.3, frequency_hz=3000.0, score=1.0)])[0].video == ""
    assert moment_for_event(Event(time_s=5.0, type="goal", team=0)).video == ""


def test_clamp_moment_trims_to_the_recording_or_refuses() -> None:
    """A window running off the end of the file used to come back as a silently short clip."""
    moment = moment_for_event(Event(time_s=100.0, type="other", source="audio"))
    assert clamp_moment(moment, 1000.0) is moment  # fully inside: untouched

    trimmed = clamp_moment(moment, moment.end_s - 5.0)
    assert trimmed is not None
    assert trimmed.end_s == pytest.approx(moment.end_s - 5.0)
    assert trimmed.end_s < moment.end_s
    assert trimmed.start_s == pytest.approx(moment.start_s)

    # The window starts inside the recording but the recording is where the clip has to stop: allowed, and short.
    tail_end = clamp_moment(moment, moment.start_s + 2.0)
    assert tail_end is not None and tail_end.end_s - tail_end.start_s == pytest.approx(2.0)

    # Wholly past the end of the file (a candidate from another recording) is refused, not cut into nonsense.
    assert clamp_moment(moment, moment.start_s) is None
    assert clamp_moment(moment, 0.0) is None


def test_verdicts_round_trip_and_old_files_read_as_unreviewed(tmp_path: Path) -> None:
    log = EventLog()
    log.add(Event(time_s=10.0, type="other", source="audio", note="whistle"))
    log.set_verdict(0, VERDICT_TRUE)
    path = tmp_path / "events.json"
    log.save(path)
    assert EventLog.load(path).events[0].verdict == VERDICT_TRUE

    # An archive written before verdicts existed: unreviewed, and still readable.
    legacy = json.loads(
        '[{"time_s": 3.0, "type": "other", "team": -1, "note": "", "source": "audio", "confidence": 0.4}]'
    )
    old = EventLog([Event.from_json(item) for item in legacy])
    assert old.events[0].verdict == "" and old.review_counts()["unreviewed"] == 1


def test_set_verdict_validates_the_index_and_the_value() -> None:
    log = EventLog()
    log.add(Event(time_s=10.0, type="other", source="audio"))
    with pytest.raises(ValueError, match="unknown verdict"):
        log.set_verdict(0, "maybe")
    with pytest.raises(IndexError):
        log.set_verdict(3, VERDICT_TRUE)
    assert log.review_counts() == {VERDICT_TRUE: 0, VERDICT_FALSE: 0, "unreviewed": 1}

    log.set_verdict(0, VERDICT_FALSE)
    assert log.review_counts() == {VERDICT_TRUE: 0, VERDICT_FALSE: 1, "unreviewed": 0}
    log.set_verdict(0, "")  # clearing a verdict is not the same as deciding against it
    assert log.review_counts()["unreviewed"] == 1


def test_remove_deletes_exactly_one_event_and_validates_the_index() -> None:
    """An individual tag can be deleted without disturbing the other events or their order."""
    log = EventLog()
    log.add(Event(time_s=10.0, type="goal", team=0))
    log.add(Event(time_s=20.0, type="shot", team=1))
    log.add(Event(time_s=30.0, type="other", source="audio"))
    removed = log.remove(1)
    assert removed.time_s == 20.0 and removed.type == "shot"
    assert [event.time_s for event in log.events] == [10.0, 30.0]
    with pytest.raises(IndexError):
        log.remove(5)


def test_discarding_rejected_candidates_keeps_everything_else() -> None:
    log = EventLog()
    log.add(Event(time_s=10.0, type="other", source="audio"))
    log.add(Event(time_s=20.0, type="other", source="audio"))
    log.add(Event(time_s=30.0, type="other", source="audio"))
    log.add(Event(time_s=40.0, type="goal", team=0, note="header"))
    log.set_verdict(0, VERDICT_FALSE)
    log.set_verdict(1, VERDICT_TRUE)
    # The last auto candidate stays unreviewed; the goal is manual and can never be swept up by this.

    assert log.discard_false() == 1
    assert [e.time_s for e in log.events] == [20.0, 30.0, 40.0]
    assert log.discard_false() == 0


def test_rescanning_prunes_candidates_the_detector_no_longer_reports() -> None:
    """Refining the detector has to be able to shorten the queue it was refined for.

    A re-scan used to only ever add, so the rows a stricter detector no longer reported stayed in the list for ever.
    What must survive is anything a human put there or *confirmed* - that verdict is data, not a detector output.
    """
    log = EventLog()
    log.add(Event(time_s=5.0, type="goal", team=0, note="header"))                    # manual: never touched
    log.add(Event(time_s=100.0, type="other", source="audio"))                        # unreviewed, still detected
    log.add(Event(time_s=200.0, type="other", source="audio"))                        # unreviewed, no longer detected
    log.add(Event(time_s=300.0, type="other", source="audio"))                        # confirmed, no longer detected
    log.add(Event(time_s=400.0, type="other", source="audio"))                        # rejected, no longer detected
    log.set_verdict(3, VERDICT_TRUE)
    log.set_verdict(4, VERDICT_FALSE)

    fresh = [Event(time_s=100.2, type="other", source="audio"), Event(time_s=500.0, type="other", source="audio")]
    added, dropped = log.reconcile_detected(fresh)
    assert (added, dropped) == (1, 2)
    assert [e.time_s for e in log.events] == [5.0, 100.0, 300.0, 500.0]
    assert log.events[2].verdict == VERDICT_TRUE, "a confirmed candidate survived a re-scan that lost it"
    # Running it again changes nothing: the list now matches the detector exactly.
    assert log.reconcile_detected(fresh) == (0, 0)
    assert [e.time_s for e in log.events] == [5.0, 100.0, 300.0, 500.0]


def test_reconciling_one_detector_leaves_the_other_detectors_rows_alone() -> None:
    """The whistle scan and the ball detectors share one review queue, so neither may sweep the other's rows.

    This is what a corrected time base looks like from the queue's point of view: the ball rows move by the
    kick-off offset, so the old rows are nowhere near the new ones and only a source-scoped sweep removes them.
    Without the scope, re-running the ball detectors would delete every whistle candidate as a side effect.
    """
    log = EventLog()
    log.add(Event(time_s=100.0, type="other", source="audio"))   # whistle candidate: not this detector's business
    log.add(Event(time_s=131.6, type="tackle", source="ball"))   # stale: written before the offset fix
    log.add(Event(time_s=166.2, type="tackle", source="ball"))   # stale too

    fresh = [Event(time_s=672.4, type="tackle", source="ball"), Event(time_s=707.0, type="tackle", source="ball")]
    added, dropped = log.reconcile_detected(fresh, source="ball")
    assert (added, dropped) == (2, 2)
    assert [e.time_s for e in log.events] == [100.0, 672.4, 707.0]
    assert log.events[0].source == "audio", "the whistle candidate survived a ball re-scan"
    # And the reverse: a whistle re-scan must not touch the ball rows.
    assert log.reconcile_detected([Event(time_s=100.0, type="other", source="audio")], source="audio") == (0, 0)
    assert [e.time_s for e in log.events] == [100.0, 672.4, 707.0]


def test_a_rejected_candidate_is_left_out_of_the_reels() -> None:
    """The point of reviewing is that the reels change: a false positive is not worth watching again."""
    events = [
        Event(time_s=20.0, type="other", source="audio"),
        Event(time_s=120.0, type="other", source="audio"),
    ]
    events[0].verdict = VERDICT_FALSE
    moments = build_moments(events)
    assert [m.time_s for m in moments] == [120.0]
    # Confirmed and unreviewed candidates both stay in.
    events[0].verdict = VERDICT_TRUE
    assert [m.time_s for m in build_moments(events)] == [20.0, 120.0]


def test_preview_clip_name_is_stable_and_changes_with_the_moment() -> None:
    same_a = moment_for_event(Event(time_s=10.0, type="shot", team=0))
    same_b = moment_for_event(Event(time_s=10.0, type="shot", team=0))
    other_team = moment_for_event(Event(time_s=10.0, type="shot", team=1))
    assert preview_clip_name(same_a) == preview_clip_name(same_b)  # cache hit on re-selection
    assert preview_clip_name(same_a) != preview_clip_name(other_team)  # an edited tag re-cuts
    assert preview_clip_name(same_a).startswith("preview_") and preview_clip_name(same_a).endswith(".mp4")


def test_highlights_rank_manual_tags_above_audio_candidates() -> None:
    events = [
        Event(time_s=100.0, type="other", source="audio", confidence=0.5),
        Event(time_s=200.0, type="goal", team=0),
        Event(time_s=300.0, type="shot", team=1),
    ]
    moments = build_moments(events)
    assert moments[0].event_type == "goal", "a tagged goal should outrank an audio hint"
    assert moments[0].start_s < 200.0 < moments[0].end_s
    assert any("manual" in m.reason for m in moments)


def test_momentum_swings_add_candidates() -> None:
    momentum = {0: {"team_0": 0.9, "team_1": 0.1, "action_x": 40.0}, 1: {"team_0": 0.5, "team_1": 0.5, "action_x": 30.0}}
    moments = build_moments([], momentum)
    assert len(moments) == 1  # only the minute with a real swing
    assert "momentum" in moments[0].reason


def test_tiers_pick_appropriate_lengths_and_respect_gaps() -> None:
    events = [Event(time_s=20.0 * i, type="shot", team=i % 2) for i in range(1, 30)]
    moments = build_moments(events)
    clip = select_reel("clip", moments)
    goals = select_reel("goals", moments)
    match = select_reel("match", moments)
    assert 15.0 <= clip.duration_s <= 30.0
    assert clip.duration_s <= TIER_SECONDS["clip"]
    assert goals.duration_s <= TIER_SECONDS["goals"]
    assert match.duration_s <= TIER_SECONDS["match"]
    # Nothing overlaps, and the picks are in chronological order.
    for reel in (clip, goals, match):
        times = [m.time_s for m in reel.moments]
        assert times == sorted(times)
        assert all(b - a >= 20.0 for a, b in zip(times, times[1:]))
    # The long reel should use more of the match than the short one.
    assert len(match.moments) > len(clip.moments)


def test_goal_tier_uses_goals_when_present() -> None:
    events = [Event(time_s=10.0, type="other", source="audio"), Event(time_s=120.0, type="goal", team=0)]
    moments = build_moments(events)
    reel = select_reel("goals", moments)
    assert [m.event_type for m in reel.moments] == ["goal"]


def test_saves_tier_uses_only_saves() -> None:
    """The saves reel is a dedicated option: goals and shots must not pad it out."""
    events = [
        Event(time_s=10.0, type="save", team=1),
        Event(time_s=40.0, type="goal", team=0),
        Event(time_s=70.0, type="shot", team=0),
    ]
    reel = select_reel("saves", build_moments(events))
    assert [m.event_type for m in reel.moments] == ["save"]
    assert reel.duration_s <= TIER_SECONDS["saves"]


def test_export_moment_writes_a_playable_preview(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    width, height, fps = 320, 180, 15
    with VideoWriter(source, fps=float(fps), width=width, height=height) as writer:
        for i in range(fps * 30):  # 30 s
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i * 5 % 255, 40, 90)
            writer.write(frame)

    moment = moment_for_event(Event(time_s=12.0, type="goal", team=0))
    out = export_moment(source, moment, tmp_path / "previews" / "moment.mp4", width=320, use_gpu=False)
    assert out.exists() and out.stat().st_size > 1000

    import cv2

    capture = cv2.VideoCapture(str(out))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    # The preview is the moment's own window (~10 s), not the whole 30 s source. The export normalizes the rate to
    # 30 fps, so measure the duration rather than the frame count.
    duration_s = frames / out_fps
    assert 4.0 < duration_s <= 12.0, f"preview is not the moment window ({duration_s:.1f}s)"


def test_a_preview_is_cut_from_the_recording_the_candidate_was_found_in(tmp_path: Path) -> None:
    """The mismatch that produced short clips: candidates from the game, clips cut out of one camera file.

    A candidate scanned in one recording must be previewed against *that* file. Its seconds are seconds of the
    scanned recording, so cutting them out of another one shows a different part of the match - and near the end of
    the shorter file there is nothing left to cut, which is how a moment came back as a clip with no whistle in it.
    """
    import cv2

    first, second = tmp_path / "camera_a.mp4", tmp_path / "camera_b.mp4"
    width, height, fps = 160, 90, 10
    for path, color in ((first, (0, 0, 255)), (second, (0, 255, 0))):  # BGR: red and green
        with VideoWriter(path, fps=float(fps), width=width, height=height) as writer:
            for _ in range(fps * 20):  # 20 s
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :] = color
                writer.write(frame)

    def first_frame(path: Path):
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        return ok, frame

    detected = Event(time_s=5.0, type="other", source="audio", video=str(second))
    out = export_moment(first, moment_for_event(detected), tmp_path / "preview.mp4", width=160, use_gpu=False)
    ok, frame = first_frame(out)
    assert ok, "the preview was not written"
    blue, green, _red = frame[45, 80]
    assert green > 128 and blue < 128, "the clip was cut from the selected video, not the scanned one"

    # A candidate that knows nothing about its origin still falls back to the video the caller passes - that is a
    # hand tag, which was made while looking at it.
    manual = Event(time_s=5.0, type="goal", team=0)
    out2 = export_moment(first, moment_for_event(manual), tmp_path / "preview2.mp4", width=160, use_gpu=False)
    ok, frame = first_frame(out2)
    assert ok and frame[45, 80][2] > 128, "a hand tag should be cut from the selected video"


def test_export_moment_reports_encode_progress(tmp_path: Path) -> None:
    """The export bar is fed by ffmpeg's own ``out_time_us``; it must arrive, bounded and forward.

    A spinner that lies about being halfway is worse than the old plain spinner, so the contract is checked against
    the real encoder rather than a stub.
    """
    source = tmp_path / "clip.mp4"
    width, height, fps = 320, 180, 15
    with VideoWriter(source, fps=float(fps), width=width, height=height) as writer:
        for i in range(fps * 30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i * 5 % 255, 40, 90)
            writer.write(frame)

    calls: list[float] = []
    moment = moment_for_event(Event(time_s=12.0, type="goal", team=0))
    export_moment(
        source,
        moment,
        tmp_path / "previews" / "progress.mp4",
        width=320,
        use_gpu=False,
        progress=lambda fraction: calls.append(float(fraction)),
    )
    assert calls, "ffmpeg reported no progress at all"
    assert all(0.0 <= call <= 1.0 for call in calls)
    assert calls == sorted(calls), "the bar must not jump backwards"
    assert max(calls) > 0.5, "the encode did not report most of its run"


def test_preview_clip_audio_is_not_aac(tmp_path: Path) -> None:
    """The browser the dashboard runs in has no AAC decoder, so a preview's audio must be MP3.

    Measured on the machine this was written for: an MP4 whose only audio is AAC plays its picture but is silent
    (the browser reports no supported streams for the audio), whereas the same clip with an MP3 track plays with
    sound. Guard the choice here so a later 'tidy-up' back to AAC cannot silently mute the previews again.
    """
    import json
    import shutil
    import subprocess

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        pytest.skip("ffmpeg/ffprobe not available to check the audio codec")

    # A source with an AAC audio track, like the real footage, so the re-encode is exercised end to end.
    source = tmp_path / "source.mp4"
    made = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=20",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
        ],
        capture_output=True,
        text=True,
    )
    if made.returncode != 0:
        pytest.skip(f"could not synthesize a source clip: {made.stderr.strip()[:120]}")

    moment = moment_for_event(Event(time_s=10.0, type="shot", team=0))
    out = export_moment(source, moment, tmp_path / "preview.mp4", width=320, use_gpu=False)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "json", str(out)],
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    assert streams, f"preview has no audio track at all: {probe.stderr.strip()}"
    assert streams[0]["codec_name"] == "mp3", (
        f"preview audio is {streams[0]['codec_name']!r}; the dashboard's browser cannot decode AAC audio"
    )


def test_an_audio_only_preview_is_a_playable_mp3_of_the_window(tmp_path: Path) -> None:
    """The cheap mode for a review: the sound alone, in a codec this browser has (MP3, never AAC), no picture.

    This is the mode that makes classifying ninety candidates quick - it decodes no video at all - so it is worth
    guarding that it really is audio-only and really is the window, rather than an empty file that plays silence.
    """
    import json
    import shutil
    import subprocess

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        pytest.skip("ffmpeg/ffprobe not available to check the audio-only preview")

    source = tmp_path / "source.mp4"
    made = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=20",
            "-f", "lavfi", "-i", "sine=frequency=3000:duration=20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
        ],
        capture_output=True,
        text=True,
    )
    if made.returncode != 0:
        pytest.skip(f"could not synthesize a source clip: {made.stderr.strip()[:120]}")

    moment = moment_for_event(Event(time_s=10.0, type="other", source="audio"))
    out = export_moment(source, moment, tmp_path / "preview.mp3", mode="audio", use_gpu=False)
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name",
            "-show_entries", "format=duration", "-of", "json", str(out),
        ],
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    kinds = [stream["codec_type"] for stream in payload["streams"]]
    assert kinds == ["audio"], f"the audio-only preview carries {kinds}"
    assert payload["streams"][0]["codec_name"] == "mp3"  # the browser cannot decode AAC
    duration_s = float(payload["format"]["duration"])
    assert 8.0 <= duration_s <= 11.0, f"not the ten second window ({duration_s:.1f}s)"


def test_a_light_preview_keeps_the_timeline_but_loses_the_frames(tmp_path: Path) -> None:
    """Keyframe-only decoding keeps a ten second clip ten seconds long, with a fraction of the pictures in it.

    A slideshow that is the wrong length would be useless for judging how long after a whistle something happened -
    the frame *rate* is what the mode trades away, not the timeline.
    """
    import shutil
    import subprocess

    import cv2

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available to check the light preview")

    source = tmp_path / "source.mp4"
    made = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=20",
            "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p",  # one keyframe a second, like the camera
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if made.returncode != 0:
        pytest.skip(f"could not synthesize a source clip: {made.stderr.strip()[:120]}")

    moment = moment_for_event(Event(time_s=10.0, type="other", source="audio"))
    # Same width for both, so the comparison is about the frames rather than the pixels.
    light = export_moment(source, moment, tmp_path / "light.mp4", mode="quick", width=320, use_gpu=False)
    full = export_moment(source, moment, tmp_path / "full.mp4", width=320, use_gpu=False)

    def duration_and_frames(path: Path) -> tuple[float, int]:
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        return frames / rate, frames

    light_s, light_frames = duration_and_frames(light)
    full_s, _full_frames = duration_and_frames(full)
    assert light_s >= 9.0, f"the light preview is the wrong length ({light_s:.1f}s)"
    assert full_s >= 9.0, f"the full preview is the wrong length ({full_s:.1f}s)"
    # The encode itself is the small part of the cost; the file is the visible proof that the frames are gone.
    assert light.stat().st_size < 0.6 * full.stat().st_size
    assert light_frames < 90, f"the light preview still encodes {light_frames} frames for ten seconds"


def _h264_clip(path: Path, *, seconds: int = 20, width: int = 320, height: int = 180, fps: int = 15) -> None:
    """A small real H.264 file, so the copy path is exercised with the camera's own codec."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available to synthesize an H.264 clip")
    made = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
            "-c:v", "libx264", "-g", str(fps), "-pix_fmt", "yuv420p", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if made.returncode != 0:
        pytest.skip(f"could not synthesize an H.264 clip: {made.stderr.strip()[:120]}")


def test_an_h264_reel_is_stream_copied_not_reencoded(tmp_path: Path, monkeypatch) -> None:
    """The camera writes H.264, so a reel can keep the exact source bitstream instead of losing quality."""
    source = tmp_path / "clip.mp4"
    _h264_clip(source)
    reel = select_reel("saves", build_moments([Event(time_s=5.0, type="save", team=0)]))

    def must_not_encode(*args, **kwargs):
        pytest.fail("an H.264 reel must be stream-copied")

    monkeypatch.setattr(highlights_lib, "_encode_clip", must_not_encode)
    out = export_reel(source, reel, tmp_path / "out" / "saves.mp4", use_gpu=False)
    probe = probe_video(out)
    assert (probe.width, probe.height) == (320, 180)
    assert probe.fps == pytest.approx(15, abs=0.5)


def test_a_failed_copy_join_falls_back_to_a_high_quality_encode(tmp_path: Path, monkeypatch) -> None:
    """If a copy cannot be joined, the export still succeeds - at source geometry and export quality."""
    source = tmp_path / "clip.mp4"
    _h264_clip(source)
    reel = select_reel("saves", build_moments([Event(time_s=5.0, type="save", team=0)]))
    calls: list[dict] = []
    original = highlights_lib._encode_clip  # noqa: SLF001 - the fallback's settings are under test

    def fail_copy(*args, **kwargs):
        raise RuntimeError("copy join failed")

    def record_encode(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(highlights_lib, "_copy_reel", fail_copy)
    monkeypatch.setattr(highlights_lib, "_encode_clip", record_encode)
    out = export_reel(source, reel, tmp_path / "out" / "saves.mp4", use_gpu=False)
    assert calls and calls[0]["high_quality"] is True
    probe = probe_video(out)
    assert (probe.width, probe.height) == (320, 180)
    assert probe.fps == pytest.approx(15, abs=0.5)


def test_export_reel_writes_a_playable_file(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    width, height, fps = 320, 180, 15
    with VideoWriter(source, fps=float(fps), width=width, height=height) as writer:
        for i in range(fps * 40):  # 40 s
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i * 3 % 255, 60, 120)
            writer.write(frame)

    events = [Event(time_s=5.0, type="shot", team=0), Event(time_s=25.0, type="goal", team=1)]
    reel = select_reel("clip", build_moments(events))
    # No width/fps: an export follows the source, so this 320x180/15 fps clip must stay that way.
    out = export_reel(source, reel, tmp_path / "out" / "clip.mp4", use_gpu=False)
    assert out.exists() and out.stat().st_size > 1000
    probe = probe_video(out)
    assert probe.duration_s >= 14.0, f"exported reel is too short ({probe.duration_s:.1f}s)"
    assert (probe.width, probe.height) == (width, height)
    assert probe.fps == pytest.approx(fps, abs=0.5)

    manifest = write_manifest(reel, source, tmp_path / "out" / "clip.json")
    payload = json.loads(manifest.read_text())
    assert payload["tier"] == "clip" and len(payload["moments"]) == len(reel.moments)
    assert reel_manifest(reel, source)["moments"][0]["reason"]


@pytest.mark.parametrize(("gpu", "quality_flag", "preset"), [(False, "-crf", "medium"), (True, "-cq", "p6")])
def test_high_quality_reel_cuts_use_the_export_presets(
    tmp_path: Path, monkeypatch, gpu: bool, quality_flag: str, preset: str
) -> None:
    """A reel is not a preview: the encoder gets the near-lossless preset, not the cheap default."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    commands: list[list[str]] = []

    def fake_run(command, expected_s, progress):
        commands.append(command)
        Path(command[-1]).write_bytes(b"x" * 2048)
        return 0, ""

    monkeypatch.setattr(highlights_lib, "run_ffmpeg_with_progress", fake_run)
    moment = moment_for_event(Event(time_s=1.0, type="save", team=0))
    highlights_lib._encode_clip(  # noqa: SLF001 - the encoder's flags are the unit under test
        source,
        moment,
        tmp_path / "out.mp4",
        duration_s=1.0,
        width=320,
        fps=60,
        use_gpu=gpu,
        high_quality=True,
    )
    command = commands[0]
    assert command[command.index(quality_flag) + 1] == "18"
    assert command[command.index("-preset") + 1] == preset
    assert "scale=320:-2:flags=lanczos,fps=60" in command


def test_export_rejects_an_empty_reel(tmp_path: Path) -> None:
    from soccer_analytics.analysis.highlights import Reel

    with pytest.raises(ValueError, match="no moments"):
        export_reel("whatever.mp4", Reel("clip", (), 0.0), tmp_path / "out.mp4")


def test_match_library_archives_artifacts(tmp_path: Path) -> None:
    library = MatchLibrary(tmp_path / "matches")
    record = library.create("/videos/match.MP4", format="9v9", pitch_length_m=60.0, pitch_width_m=40.0)
    assert record.match_id.startswith("2")

    calibration = PitchCalibration(np.array([30.0, -7.0, 4.5]), np.eye(3), 1.0, 0.5625, 0.42, (0.1, 0.2, 0.3, 0.4))
    library.save_calibration(record.match_id, calibration)
    loaded = library.load_calibration(record.match_id)
    assert loaded is not None and np.allclose(loaded.position, calibration.position)
    # A bad fit has to be removable, or a wrong camera stays on the match forever.
    assert library.clear_calibration(record.match_id) is True
    assert library.load_calibration(record.match_id) is None
    assert library.clear_calibration(record.match_id) is False
    library.save_calibration(record.match_id, calibration)

    log = EventLog([Event(time_s=12.0, type="goal", team=0)])
    library.save_events(record.match_id, log)
    assert len(library.events(record.match_id).events) == 1

    library.save_report(record.match_id, {"teams": [], "notes": ["n"]})
    assert library.load_report(record.match_id)["notes"] == ["n"]

    (library.highlights_dir(record.match_id) / "clip.mp4").write_bytes(b"x" * 2000)
    summary = library.summaries()[0]
    assert summary["has_calibration"] and summary["has_report"]
    assert summary["events"] == 1 and summary["highlights"] == 1
    names = [item["path"] for item in library.artifacts(record.match_id)]
    assert any(name.startswith("highlights/") for name in names)

    # A rewritten file must not leave a .tmp behind for the archive listing.
    library.save_report(record.match_id, {"teams": [], "notes": ["again"]})
    assert not any(item["path"].endswith(".tmp") for item in library.artifacts(record.match_id))


def test_a_segment_is_recorded_against_its_match(tmp_path: Path) -> None:
    """The archive has to know what has been produced for a match, or a later visit cannot tell."""
    library = MatchLibrary(tmp_path / "matches")
    record = library.create("/videos/match.MP4")
    assert library.summaries()[0]["segments"] == 0

    library.add_segment(record.match_id, "/segments/match_123")
    library.add_segment(record.match_id, "/segments/match_123")  # re-running Stage A must not duplicate it
    assert library.load(record.match_id).segments == ["/segments/match_123"]

    library.add_segment(record.match_id, "/segments/match_456")
    assert library.summaries()[0]["segments"] == 2


def test_creating_the_same_match_twice_never_resets_it(tmp_path: Path) -> None:
    """The id comes from the video's name and the date, so a second create lands on the same match.

    Overwriting the record there would silently discard the format, the team names and the segment list that were
    saved against it - and the page reads that record back on every run.
    """
    library = MatchLibrary(tmp_path / "matches")
    first = library.create("/videos/match.MP4", format="9v9", pitch_length_m=60.0, pitch_width_m=40.0)
    library.add_segment(first.match_id, "/segments/match_123")

    again = library.create("/videos/match.MP4", format="11v11", pitch_length_m=100.0, pitch_width_m=64.0)
    assert again.match_id == first.match_id
    assert again.format == "9v9", "a second create must not rewrite the archive"
    assert again.pitch_length_m == 60.0 and again.pitch_width_m == 40.0
    assert again.segments == ["/segments/match_123"]
    assert len(library.list_ids()) == 1


# --------------------------------------------------------------------------------------------------------------
# Clocks: a moment's seconds are seconds of its own recording, and a reel is cut from one file.
# --------------------------------------------------------------------------------------------------------------
def test_a_moment_from_another_recording_is_mapped_onto_the_source() -> None:
    """A whistle scanned on a camera clip is a time on that clip's clock.

    Reels are cut from one file - usually the selected video. When a moment's own recording is a different file
    (a camera clip combined into a game), its window has to be translated through the clip's offset or the reel
    shows the same *number* of seconds of the wrong part of the match.
    """
    moment = moment_for_event(
        Event(time_s=5.0, type="other", source="audio", video="/srv/x/clip2.mp4")
    )
    # clip2 starts 1800 s into the combined game.
    offsets = {"/srv/x/clip2.mp4": 1800.0}
    game = "/srv/x/game.mp4"
    mapped = moment_on_source(moment, game, clip_offsets=offsets)
    assert mapped.clip_start_s == moment.start_s + 1800.0
    assert mapped.clip_end_s == moment.end_s + 1800.0
    # The original window is untouched: the manifest still describes the moment in its own recording's terms.
    assert mapped.start_s == moment.start_s and mapped.end_s == moment.end_s


def test_a_moment_cut_from_its_own_recording_is_not_mapped() -> None:
    """A candidate previewed against the file it was found in needs no translation."""
    moment = moment_for_event(Event(time_s=100.0, type="goal", source="ball", video="/srv/x/game.mp4"))
    mapped = moment_on_source(moment, "/srv/x/game.mp4", clip_offsets={"/srv/x/game.mp4": 0.0})
    assert mapped is moment, "a moment cut from its own recording must come back untouched"


def test_a_manual_tag_with_no_recording_is_not_mapped() -> None:
    """Manual tags and momentum swings belong to whatever the caller passes; there is nothing to translate."""
    moment = moment_for_event(Event(time_s=42.0, type="goal", team=0, note="header"))
    assert moment.video == ""
    mapped = moment_on_source(moment, "/srv/x/game.mp4", clip_offsets={"/srv/x/other.mp4": 60.0})
    assert mapped is moment


def test_a_moment_whose_recording_is_unknown_cut_is_left_alone() -> None:
    """No offset for the moment's recording: leave the window alone rather than inventing one.

    The caller (the preview) already refuses a moment past the end of the file it is cutting from; silently
    shifting a window by a guessed offset would be worse than leaving it and saying so.
    """
    moment = moment_for_event(Event(time_s=5.0, type="other", source="audio", video="/srv/x/clip9.mp4"))
    mapped = moment_on_source(moment, "/srv/x/game.mp4", clip_offsets={"/srv/x/clip2.mp4": 1800.0})
    assert mapped.clip_start_s is None and mapped.clip_end_s is None
    assert mapped.start_s == moment.start_s

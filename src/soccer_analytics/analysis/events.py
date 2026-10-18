"""Match events: what the audio can tell us, and what only a human can.

The whistle is the one match event that is reliably detectable from this footage. It is a narrowband tone, so it is
found by tracking how much a single frequency bin stands out from its neighborhood in the 2-4.5 kHz band over time,
rather than by loudness - on the real sample the whole match is dominated by speech near the microphone, and a
loudness threshold flagged nothing at all.

Everything else is a manual tag. The ball has its own dedicated scan (``analysis.ball``), but no event is inferred
from it yet, so goals, shots, saves and blocks stay human calls, and the report says so rather than inventing them.
Audio-derived moments are therefore *candidates for review*, presented to the user as such.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# Whistle band and analysis window. A referee's whistle is 2-4 kHz; 32 ms windows resolve a short blast.
BAND_LOW_HZ = 2200.0
BAND_HIGH_HZ = 4600.0
WINDOW = 512
HOP = 128
FFT_BLOCK = 8192  # windows transformed per block: bounds the temporary copy and lets a scan report progress
MIN_DURATION_S = 0.20
MIN_GAP_S = 1.0
# Two auto-detected candidates closer together than this are the same stoppage recorded twice - a re-run of the
# scan after changing the detector's settings, or the same blast found from two windows. The scan is deterministic,
# so without this a second pass appended a second copy of every candidate (the real archive grew from 28 rows to
# 44 for the same 28 moments).
DUPLICATE_WINDOW_S = 1.0
_PEAK_RATIO_K = 6.0  # robust sigmas above the match's own baseline

# How far above the match's own median level in the whistle band a blast has to sit. This is the gate that matters,
# and it is measured, not guessed: on the real 4K sample the bulk of the tonal blips sit around 17x the match median
# (shouts, kicks, whistles from the pitches next door) while the loud sustained blasts run from 100x to over 1000x.
# It is a *relative* measure, so it does not care about the recording level.
#
# The default sits at the bottom of that gap rather than in the middle of it, and the reason is the camera's
# position: it is on the sideline, so a whistle blown in the far half arrives far quieter than one blown five meters
# away - the same blast can differ by 30x on distance alone. A stricter gate would drop exactly the far-half
# whistles, which are the ones on the pitch being filmed. Better to report a few more and let the level in the note
# tell the user which is which; raise it for a venue where the neighbors dominate.
MIN_PROMINENCE = 50.0

# Below this the recording is effectively silent and "prominence" would be a ratio against nothing.
_SILENCE_FLOOR = 1e-9

# Where a shout keeps its own energy, and how much of a candidate's energy may sit there.
#
# Loudness and tonality alone cannot separate a whistle from a coach's yell: a shouted vowel has strong harmonics up
# in the whistle band, so it passes the prominence gate. But a shout is *voiced* - its energy is dominated by the
# harmonics of the speaker's pitch (100-300 Hz) and their formants, all well below the whistle band - while a
# whistle is a lone tone with almost nothing down there. Measured on synthesized shouts (f0 120/190/300 Hz, formants
# in the band) that all cleared the prominence gate: 53-67% of their energy sat below 1.2 kHz, against 0% for a pure
# tone. On the real footage the loudest blasts (up to 11000x the match level) measure 0.0-0.4, while the candidates
# that sound like yelling measure 0.7-0.9. So a candidate whose median energy share below the ceiling is at least
# VOICE_SHARE_MAX is a voice, not a whistle.
VOICE_CEILING_HZ = 1200.0
VOICE_SHARE_MAX = 0.5

# How much a blast is allowed to lift the region *below* the whistle band, relative to the same region either side
# of it.
#
# This is the gate a user's own labels produced (the first sixteen confirmed/rejected candidates, 2026-10-05). They
# named the two things that survive the loudness and voice gates: a female coach's shout and birds of prey. Both
# carry their own energy well below the whistle band - a shouted vowel keeps its lower harmonics and formants there
# however high the voice is, and a raptor's call is a rich, gliding tone rather than a lone whistle - while a
# referee's blast puts everything into its own narrow band and leaves the region below untouched.
#
# The measure is deliberately *differential*. The crowd's speech sits in that region constantly, so its absolute
# share says very little - a loud passage raises the whistle band and the voice band together, which is why the
# share gate let a female shout through. What separates a voice or a call from a whistle is whether the *blast
# itself* brought low frequencies with it, so the region is measured over the blast (with a tenth of a second of
# grace either side, so a very short blast is not judged on twenty frames) against the second either side of it.
#
# Measured with this definition on those labels: every *true* whistle stayed between 0.54x and 1.91x, and nine of
# the twelve false ones ran from 4.7x to 205x - a shout's or a call's own low energy rides in with it. The
# threshold sits in that wide gap rather than on the 1.9-2.0 knife edge two of the candidates sit on: the cost of
# keeping a candidate that turns out to be a call is one review click, and the cost of dropping a real whistle is
# a stoppage nobody ever sees. The three the gate does not catch are pure high tones with nothing below them, which
# is what a distant whistle on a neighboring pitch also looks like: one microphone cannot tell those apart, and
# they are the reviewer's call.
LOW_GAIN_MAX = 3.0
LOW_GAIN_BAND_HZ = (150.0, 1500.0)
LOW_GAIN_SPAN_S = 0.10  # the grace around the blast that still counts as the blast
LOW_GAIN_CONTEXT_S = 1.0

# The event vocabulary. The first group is what the audio and the ball scan can now infer (see
# ``analysis.event_detection``); the rest stay human calls. ``penalty`` is the award, ``goal`` the outcome - a
# converted penalty is both, and they are tagged separately because they are different things to watch.
EVENT_TYPES = (
    "goal",
    "shot",
    "save",
    "block",
    "corner",
    "penalty",
    "clearance",
    "tackle",
    "foul",
    "substitution",
    "other",
)

# Which events the ball scan and the player tracks can put on the timeline by themselves. Everything else is a
# human's call, and the report says so rather than inventing it.
DETECTED_EVENT_TYPES = ("goal", "shot", "corner", "penalty", "clearance", "tackle")

# What a human has decided about a detected candidate. "" means nobody has looked at it yet, which is why it is
# the default: a review queue that starts out "reviewed" is not a review queue.
VERDICT_TRUE = "true"
VERDICT_FALSE = "false"
VERDICTS = ("", VERDICT_TRUE, VERDICT_FALSE)


@dataclass(frozen=True)
class Whistle:
    time_s: float
    duration_s: float
    frequency_hz: float
    score: float
    prominence: float = 0.0  # band level relative to the match's own median, the gate that decides this is a whistle
    voice_share: float = 0.0  # share of the blast's energy below VOICE_CEILING_HZ; set by the voice rejection
    low_gain: float = 0.0  # how much the blast lifted LOW_GAIN_BAND_HZ against its own surroundings


@dataclass
class Event:
    """A moment in the match. ``source`` records whether a human or the audio put it there.

    ``video`` is the file the moment was found in. It matters because a candidate's timestamp is a time on *that*
    recording's clock: a scan of the combined game and a scan of one camera file both describe this match, and a
    clip cut from the wrong one is a clip of the wrong moment. Empty for manual tags - a human tags what is on
    screen, so the page's own video is meant.

    ``verdict`` is the human's review of a detected candidate: ``""`` (not looked at yet), ``VERDICT_TRUE`` or
    ``VERDICT_FALSE``.

    ``player_track`` and ``player_number`` name the player the event is attributed to, when one can be. The track id
    is the pipeline's own identity (stable within a match, meaningless across matches); the number is the shirt
    number the roster or the OCR scan gave that track, and is ``None`` when the track has no number. Both are
    optional because a detected event often has no single player to name - a goal off a deflection, a clearance
    nobody was near - and a guess would be worse than an honest blank.
    """

    time_s: float
    type: str
    team: int = -1
    note: str = ""
    source: str = "manual"
    confidence: float = 1.0
    video: str = ""
    verdict: str = ""
    player_track: int | None = None
    player_number: int | None = None

    def to_json(self) -> dict:
        return {
            "time_s": round(float(self.time_s), 3),
            "type": self.type,
            "team": int(self.team),
            "note": self.note,
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "video": self.video,
            "verdict": self.verdict,
            "player_track": None if self.player_track is None else int(self.player_track),
            "player_number": None if self.player_number is None else int(self.player_number),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Event":
        return cls(
            time_s=float(data["time_s"]),
            type=str(data.get("type", "other")),
            team=int(data.get("team", -1)),
            note=str(data.get("note", "")),
            source=str(data.get("source", "manual")),
            confidence=float(data.get("confidence", 1.0)),
            video=str(data.get("video", "")),
            verdict=str(data.get("verdict", "")),
            player_track=None if data.get("player_track") is None else int(data["player_track"]),
            player_number=None if data.get("player_number") is None else int(data["player_number"]),
        )


def _voice_share(spectra: np.ndarray, freqs: np.ndarray, start: int, end: int) -> float:
    """Median share of a candidate's energy that sits below :data:`VOICE_CEILING_HZ`.

    Measured per frame and then medianed, so a single noisy frame cannot decide the question. A whistle coming from
    far away still wears its shape - the tone dominates its own band - whereas a shout always carries its low
    harmonics, which is why this works where loudness does not.
    """
    voice = (freqs >= 80.0) & (freqs <= VOICE_CEILING_HZ)
    span = spectra[max(0, start) : max(0, end)]
    if span.size == 0 or not voice.any():
        return 0.0
    totals = span.sum(axis=1) + 1e-12
    return float(np.median(span[:, voice].sum(axis=1) / totals))


def _low_gain(
    spectra: np.ndarray, freqs: np.ndarray, start: int, end: int, *, sample_rate: int
) -> float:
    """How much the blast lifted the region below the whistle band, against the seconds either side of it.

    The frames are the ones the detector already isolated, so this asks a question about the *event* rather than
    about the passage it happened in: a whistle adds energy in its own band and nothing below it, whereas a shout or
    a bird's call brings its lower harmonics and formants with it. Speech from the crowd around the pitch is always
    in that region, which is why the reference is taken from the blast's own surroundings rather than the whole
    match - it is the *rise* that identifies the sound, not the level.
    """
    low = (freqs >= LOW_GAIN_BAND_HZ[0]) & (freqs <= LOW_GAIN_BAND_HZ[1])
    start = max(0, start - int(LOW_GAIN_SPAN_S * sample_rate / HOP))
    end = max(start + 1, end + int(LOW_GAIN_SPAN_S * sample_rate / HOP))
    span = spectra[max(0, start) : max(0, end)]
    if span.size == 0 or not low.any():
        return 0.0
    guard = max(1, int(0.1 * sample_rate / HOP))  # a tenth of a second either side still belongs to the blast
    context_frames = max(guard + 1, int(LOW_GAIN_CONTEXT_S * sample_rate / HOP))
    before = spectra[max(0, start - context_frames) : max(0, start - guard)]
    after = spectra[end + guard : end + context_frames]
    blocks = [block for block in (before, after) if block.size]
    if not blocks:
        return 0.0
    reference = float(np.median(np.vstack(blocks)[:, low].sum(axis=1)))
    if reference <= 0.0:
        return 0.0
    return float(np.median(span[:, low].sum(axis=1)) / reference)


def detect_whistles(
    samples: np.ndarray,
    sample_rate: int,
    min_prominence: float = MIN_PROMINENCE,
    max_voice_share: float | None = VOICE_SHARE_MAX,
    max_low_gain: float | None = LOW_GAIN_MAX,
    on_progress=None,
) -> list[Whistle]:
    """Finds whistle blasts: a narrow band peak that dominates the band, loud, held for a moment, and not a voice.

    Five things have to be true, and each was measured on real footage rather than assumed. The peak has to stand
    out from nearby frequencies (so it is a tone and not a shout); it has to last (a blast, not a click); it has to
    be *loud* relative to the rest of the match; most of its energy has to sit in the band rather than in the voice
    range below it (so a coach's yell is not reported as a whistle); and the blast must not drag the region *below*
    the band up with it, which is what a shout or a bird of prey's call does and a lone tone does not
    (:data:`LOW_GAIN_MAX` - the gate a user's own true/false labels produced). Without the loudness gate this
    reported 191 candidates in five minutes - mostly whistles from neighboring pitches and shouts - because a
    tonal blip from 30 m away looks exactly like a referee's whistle, only quieter. ``min_prominence`` is that
    gate, in multiples of the match's own median band level; ``max_voice_share`` and ``max_low_gain`` are the two
    "is it a voice or a call" gates, and ``None`` switches either off.

    ``on_progress(fraction)`` follows the transform, which is the whole cost of a scan on a real recording; the
    fraction is of the analysis windows, and the thresholding that follows it is quick.
    """
    if samples.size < WINDOW * 4 or sample_rate <= 0:
        return []
    window = np.hanning(WINDOW)
    frames = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP]
    # The transform is the whole cost here - a full game is over half a million windows - so it runs in blocks.
    # That is what a progress bar can report on, and it also stops the windowed copy of the signal from
    # materializing in one go (2.5 GB on a full game). The numbers are those of the single-call version.
    spectra = np.empty((frames.shape[0], WINDOW // 2 + 1), dtype=np.float32)
    for start_row in range(0, frames.shape[0], FFT_BLOCK):
        block = frames[start_row : start_row + FFT_BLOCK]
        spectra[start_row : start_row + block.shape[0]] = np.abs(np.fft.rfft(block * window, axis=1)) ** 2
        if on_progress is not None:
            on_progress(min(1.0, (start_row + block.shape[0]) / frames.shape[0]))
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / sample_rate)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if band.sum() < 8:
        return []
    banded = spectra[:, band]
    band_freqs = freqs[band]

    # How much does the loudest bin stand out from the median of the band? Speech is spread out, a whistle is not.
    peak = banded.max(axis=1)
    local = np.median(banded, axis=1) + 1e-12
    ratio = peak / local
    # Require it to be a sustained tone, not a click: average over ~5 windows (40 ms at a 16 kHz rate / 128 hop).
    smooth = np.convolve(ratio, np.ones(5) / 5.0, mode="same")

    # The loudness reference: what this match normally sounds like in the whistle band, so the gate is about the
    # recording rather than the microphone.
    band_level = banded.sum(axis=1)
    reference = float(np.median(band_level))
    if reference < _SILENCE_FLOOR:
        return []

    baseline = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - baseline))) * 1.4826 + 1e-9
    threshold = baseline + _PEAK_RATIO_K * mad
    active = smooth > threshold
    if not active.any():
        return []

    frame_times = np.arange(len(active)) * HOP / sample_rate
    out: list[Whistle] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append(_make_whistle(start, index, frame_times, banded, band_freqs, smooth, band_level, reference))
            start = None
    if start is not None:
        out.append(
            _make_whistle(start, len(active), frame_times, banded, band_freqs, smooth, band_level, reference)
        )

    # One candidate per event. A whistle heard at 12.7 s and the shouting that follows it, or the same blast arriving
    # off a wall, are one thing happening, not four - and a fan of near-identical candidates is exactly what made
    # this unusable. The strongest blast describes the event.
    merged: list[Whistle] = []
    for whistle in out:
        if merged and whistle.time_s - (merged[-1].time_s + merged[-1].duration_s) < MIN_GAP_S:
            if whistle.score > merged[-1].score:
                merged[-1] = whistle
        else:
            merged.append(whistle)

    kept: list[Whistle] = []
    for whistle in merged:
        if whistle.duration_s < MIN_DURATION_S or whistle.prominence < min_prominence:
            continue
        start = max(0, int(round(whistle.time_s * sample_rate / HOP)) - 1)
        end = max(start + 1, int(round((whistle.time_s + whistle.duration_s) * sample_rate / HOP)))
        share = _voice_share(spectra, freqs, start, end)
        if max_voice_share is not None and share >= max_voice_share:
            continue
        gain = _low_gain(spectra, freqs, start, end, sample_rate=sample_rate)
        if max_low_gain is not None and gain >= max_low_gain:
            continue
        kept.append(replace(whistle, voice_share=share, low_gain=gain))
    return kept


def _make_whistle(start, end, frame_times, banded, band_freqs, smooth, band_level, reference) -> Whistle:
    span = slice(start, max(end, start + 1))
    peak_frame = start + int(np.argmax(smooth[span]))
    bins = np.where(banded[peak_frame] > 0.5 * banded[peak_frame].max())[0]
    frequency = float(np.mean(band_freqs[bins])) if len(bins) else float(band_freqs[int(np.argmax(banded[peak_frame]))])
    return Whistle(
        time_s=float(frame_times[start]),
        duration_s=float((end - start) * (frame_times[1] - frame_times[0])) if len(frame_times) > 1 else 0.0,
        frequency_hz=frequency,
        score=float(smooth[peak_frame]),
        prominence=float(np.median(band_level[span])) / reference,
    )


def whistles_to_events(whistles: list[Whistle], *, video: str | Path = "") -> list[Event]:
    """Whistles as review candidates. They mark stoppages, not goals: a human decides what actually happened.

    The note carries the prominence as well as the pitch of the blast, because "13x the rest of the match" and "900x"
    mean very different things to whoever is reviewing them.

    ``video`` is stamped on every candidate so its clip is later cut out of the recording the time was measured on,
    rather than out of whatever the page happens to have selected. The times of two scans of the same match are not
    comparable - one camera file's 10:00 is not the combined game's 10:00 - so without it a preview can silently
    show the wrong part of the match.
    """
    origin = str(Path(video).resolve()) if str(video) else ""
    return [
        Event(
            time_s=whistle.time_s,
            type="other",
            note=(
                f"whistle at {whistle.frequency_hz:.0f} Hz ({whistle.duration_s:.2f}s, "
                f"{whistle.prominence:.0f}x the match level) - stoppage candidate"
            ),
            source="audio",
            confidence=float(np.clip(whistle.prominence / 300.0, 0.0, 0.9)),
            video=origin,
        )
        for whistle in whistles
    ]


@dataclass
class EventLog:
    """All events for a match, human and audio-derived, kept in one place and serializable."""

    events: list[Event] = field(default_factory=list)

    def add(self, event: Event) -> None:
        if event.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event.type!r}; expected one of {EVENT_TYPES}")
        self.events.append(event)
        self.events.sort(key=lambda e: e.time_s)

    def add_detected(self, events: Iterable[Event], *, within_s: float = DUPLICATE_WINDOW_S) -> int:
        """Add auto-detected candidates, skipping any already recorded within ``within_s`` of one of them.

        Returns how many were actually added, so the page can say "already recorded" rather than silently doing
        nothing. Only auto-detected events are compared against: a human tagging a goal at the moment a whistle
        sounds means two things happened, not one, so manual tags never suppress a candidate (and vice versa).
        """
        added = 0
        for event in events:
            if event.source == "manual":
                raise ValueError("add_detected is for auto-detected events; use add() for manual tags")
            if any(
                other.source != "manual" and abs(other.time_s - event.time_s) < within_s
                for other in self.events
            ):
                continue
            self.add(event)
            added += 1
        return added

    def manual(self) -> list[Event]:
        return [e for e in self.events if e.source == "manual"]

    def detected(self) -> list[Event]:
        """Events no human put there - currently the whistle candidates from the audio scan.

        Defined as "everything that is not manual" rather than "everything that is audio", so a future detector
        cannot quietly make its output survive a discard.
        """
        return [e for e in self.events if e.source != "manual"]

    def discard_detected(self) -> int:
        """Drop every auto-detected event and keep the human's own tags. Returns how many were removed.

        The candidates are a review queue, not a record: once they have been read they are noise in the list, and
        re-running the scan brings them straight back, so clearing them is cheap. Manual tags are never touched.
        """
        kept = self.manual()
        removed = len(self.events) - len(kept)
        self.events = kept
        return removed

    def remove(self, index: int) -> Event:
        """Drop the event at ``index`` and return it; raises :class:`IndexError` for an unknown row.

        Removal is by position because the page's pickers index into this same list. It is deliberately a hard
        delete of one row - unlike :meth:`discard_detected` and :meth:`discard_false`, which are bulk review
        sweeps that keep a human's own tags.
        """
        if not 0 <= int(index) < len(self.events):
            raise IndexError(f"no event at index {index} of {len(self.events)}")
        return self.events.pop(int(index))

    def set_verdict(self, index: int, verdict: str) -> Event:
        """Record the review verdict for the event at ``index`` and return it.

        By position rather than by time: the page's picker is an index into this same list, and two candidates a
        few seconds apart are still two different events. Raising on an out-of-range index is deliberate - a
        silently ignored verdict is a review that looks done and is not.
        """
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
        if not 0 <= index < len(self.events):
            raise IndexError(f"no event at index {index} of {len(self.events)}")
        event = self.events[index]
        event.verdict = verdict
        return event

    def review_counts(self) -> dict[str, int]:
        """How the review is going: ``{"true": n, "false": n, "unreviewed": n}``."""
        counts = {VERDICT_TRUE: 0, VERDICT_FALSE: 0, "unreviewed": 0}
        for event in self.events:
            counts[event.verdict if event.verdict in (VERDICT_TRUE, VERDICT_FALSE) else "unreviewed"] += 1
        return counts

    def discard_false(self) -> int:
        """Drop the auto-detected events a human rejected, keeping everything else. Returns how many were removed.

        Manual tags are kept whether or not anything was decided about them - a verdict only ever applies to a
        detected candidate, and dropping a human's own tag is never what "clear the false positives" means.
        """
        kept = [e for e in self.events if e.source == "manual" or e.verdict != VERDICT_FALSE]
        removed = len(self.events) - len(kept)
        self.events = kept
        return removed

    def reconcile_detected(
        self,
        events: Iterable[Event],
        *,
        within_s: float = DUPLICATE_WINDOW_S,
        source: str | None = None,
    ) -> tuple[int, int]:
        """Bring the auto-detected rows in line with a fresh scan; returns ``(added, dropped)``.

        A re-scan used to only ever *add*: rows the detector no longer reports stayed in the list for ever, so
        refining the detector could not shorten the queue it was meant to clean up. Now a row that no fresh
        candidate is anywhere near is dropped - unless a human *confirmed* it, because that verdict is data rather
        than a detector output, and losing it on a re-scan would make reviewing pointless. Manual tags are never
        touched. Rows that are still detected keep their time, note and verdict.

        ``source`` limits the sweep to one detector's own rows. The whistle scan and the ball detectors write into
        the same queue, so re-running one of them must not delete the other's candidates. It also has to delete its
        *own* rows when their times have moved, which is exactly what a corrected time base does - otherwise the
        old rows survive alongside the new ones and the queue shows every event twice.
        """
        fresh = list(events)
        added = self.add_detected(fresh, within_s=within_s)
        kept: list[Event] = []
        dropped = 0
        for event in self.events:
            if event.source == "manual" or event.verdict == VERDICT_TRUE:
                kept.append(event)
                continue
            if source is not None and event.source != source:
                kept.append(event)
                continue
            if any(abs(event.time_s - candidate.time_s) < within_s for candidate in fresh):
                kept.append(event)
                continue
            dropped += 1
        self.events = kept
        return added, dropped

    def before(self, time_s: float) -> list[Event]:
        return [e for e in self.events if e.time_s <= time_s]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([e.to_json() for e in self.events], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EventLog":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls([Event.from_json(item) for item in json.loads(path.read_text())])

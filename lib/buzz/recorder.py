"""
Automatic .wav capture of interference events.

EventRecorder watches the analyzer's state machine and writes each event it sees
to its own file, so an interesting burst can be replayed through buzz.playback
later — analysed again on the same displays, at real speed, with no receiver
attached and no chance of missing it live.

A recording spans more than the event itself:

    |<-- lead-in -->|<---------- event ---------->|<-- trailer -->|
    buffered audio   LOCKED, sampled continuously   stop_after_seconds
    already captured                                without a lock
    when lock hit

The lead-in is free.  The ring buffer is always holding the last several seconds
of audio, so at the instant of lock the run-up to the event has already been
captured — the recorder just reads the buffer from its oldest surviving sample
rather than from the live tail.  Without that, every recording would begin with
the pulse train mid-stride, which is the least useful part to look at.

The trailer costs nothing either.  A recording ends because the signal has been
gone for stop_after_seconds, and that audio is written as it arrives rather than
being held back, so by the time the timeout expires the trailer is already in the
file.  The same timeout is what lets a flickering signal stay one recording: any
lock inside the window continues the event instead of splitting it in two.

Both ends are faded so the file starts and finishes at exactly zero and cannot
click — see fade_ramp for the shape, FADE_SECONDS for the length, and _write for
how the fade-out reaches audio whose lastness is only known afterwards.

Lock is not polled.  The analyzer publishes each state change to a listener (see
ContinuousAnalyzer.add_state_listener) and the recorder's thread does the writing,
so the two never wait on each other: analysis is never behind disk I/O, and a lock
that comes and goes between two polls still starts a recording.

Everything is measured in absolute sample positions rather than wall-clock time —
the same audio clock the analyzer's drift tracker uses.  A recording's length is
then exactly what its audio contains, regardless of when the polling thread
happened to run, and a stalled capture device cannot time out a recording that
has not actually gone quiet.
"""

import logging
import threading
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from buzz import __version__, wavmeta
from buzz.analyzer import AnalyzerState, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.sampler import RingBufferPipeline

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2   # 16-bit PCM, matching the int16 capture format end to end

_EMPTY = np.empty(0, dtype=np.int16)

# How each `ended` token written into a file's metadata reads in the log.  The two
# tokens produced by a limit are described with that limit's value instead, by
# _end_description.
_END_DESCRIPTIONS = {
    'operator': 'stopped by operator',
    'shutdown': 'monitor shutting down',
}


@dataclass(frozen=True)
class RecorderStatus:
    """Immutable snapshot of the recorder, for the toolbar to poll and draw."""

    armed: bool
    recording: bool
    # None when max_events is 0 — record every event until switched off by hand.
    events_remaining: int | None
    # Audio written to the current file so far, lead-in included; 0 when idle.
    elapsed_seconds: float
    filename: str | None


def fade_ramp(n: int) -> np.ndarray:
    """An n-point raised-cosine ramp rising from exactly 0 to exactly 1.

    A file that begins or ends on a non-zero sample steps to or from silence, and a
    step is broadband: it clicks, and clicks at the seams when files are played back
    to back.  Sound cards carry a DC offset (the reason LevelStream removes one), so
    this happens even where the recording contains nothing but noise floor.

    Raised cosine rather than an exponential, which the ends of a file argue for on
    both counts.  An exponential approaches zero without reaching it, so it has to be
    truncated — and the truncation is itself a step, exactly what the fade was for.
    This shape hits 0 and 1 exactly, and meets both of them with zero slope, so the
    join to silence and the join to full-scale audio are each smooth.  That
    continuity is worth 6 dB/octave of splatter rolloff over a linear ramp's corner,
    for the same cost.

    Reverse it for the fade-out; the last sample is then exactly zero.
    """
    return 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))


def event_filename(when: datetime) -> str:
    """Return the .wav filename for an event that locked at `when`.

    Local time with the UTC offset attached, so a file is unambiguous a year later
    and across a DST change.  ISO 8601's colons are illegal in Windows filenames and
    its T separator is hard to read at a glance, so date and time are joined with a
    dash instead: event-20260729-143307-0700.wav.
    """
    return f'event-{when.strftime("%Y%m%d-%H%M%S%z")}.wav'


def unique_path(path: Path) -> Path:
    """Return `path`, or the first free -2, -3, ... variant if it already exists.

    Two events cannot normally share a filename — stop_after_seconds keeps their
    lock instants at least a second apart — but replaying an old recording directory
    or a clock adjustment could collide, and silently overwriting a capture would
    destroy the one thing this module exists to keep.
    """
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f'{path.stem}-{n}{path.suffix}')
        n += 1
    return candidate


class EventRecorder:
    """Records locked events to .wav files from its own polling thread.

    Armed/disarmed at runtime (toolbar button, or --enable-recording at startup).
    While armed it starts a file on the first LOCKED tick and closes it once the
    signal has been gone for stop_after_seconds, then counts the event against the
    remaining budget and disarms itself when that budget runs out.
    """

    # Matches the analyzer's LOCKED tick cadence: polling faster cannot see a lock
    # sooner, and the audio itself is never sampled at this rate — each pass writes
    # every sample captured since the previous one, whenever it happens to run.
    POLL_INTERVAL = 0.2

    # Length of the fade at each end of a file (see fade_ramp for the shape).  A fade
    # of duration T spreads the transition it replaces over a bandwidth of about 1/T,
    # so a handful of samples — 0.25 ms, 4 kHz — merely turns a click into a quieter
    # click.  Audibility falls off past a millisecond and is gone by about five, which
    # is where audio editors put their default fades at edit points.
    #
    # Sized for a cut through full-scale audio, because both ends can be one.  A file
    # usually opens in quiet lead-in and closes in the silence the event faded into,
    # but not always: an arc already buzzing when the monitor starts is locked onto
    # within a second or two, so the lead-in is a live pulse train from its first
    # sample, and a max_seconds cap ends a file mid-event the same way.  Even then the
    # fade gives up well under one pulse out of the 120 in that second.
    FADE_SECONDS = 0.005

    def __init__(self, pipeline: RingBufferPipeline, analyzer: ContinuousAnalyzer,
                 config: BuzzConfig) -> None:
        self._pipeline = pipeline
        recording = config.recording
        self._directory   = recording.directory_path(config.station)
        self._zone        = ZoneInfo(config.station.timezone)
        self._sample_rate = config.audio.sample_rate
        self._max_events  = recording.max_events
        # Kept for the file's metadata.  The pulse rate and the dB calibration are
        # the two settings a replay cannot recover from the audio itself, and getting
        # either wrong changes what the replay measures — see wavmeta.
        self._callsign         = config.station.callsign
        self._pulse_rate       = config.audio.pulse_rate
        self._rf_conversion_db = config.station.audio_rf_conversion_db
        # Both limits in samples, on the audio clock.  0 means uncapped length.
        self._max_samples     = round(recording.max_seconds * self._sample_rate)
        self._timeout_samples = round(recording.stop_after_seconds * self._sample_rate)
        self._fade_in = fade_ramp(round(self.FADE_SECONDS * self._sample_rate))

        self._armed = recording.enabled
        self._events_remaining = self._initial_budget()

        # Lock state is pushed from the analyzer rather than read back from it (see
        # ContinuousAnalyzer.add_state_listener), seeded here with the state the
        # analyzer is in at wiring time.  _lock_acquired is sticky until the next
        # tick consumes it, so an event that locks and drops again inside a single
        # poll interval still starts a recording instead of vanishing between polls.
        self._locked = analyzer.state == AnalyzerState.LOCKED
        self._lock_acquired = self._locked
        analyzer.add_state_listener(self._on_analyzer_state)

        # Current recording, all None/0 while idle.
        self._writer: wave.Wave_write | None = None
        self._path: Path | None = None
        self._frames_accepted = 0       # audio taken into the recording, tail included
        self._lead_in = 0               # samples captured before the lock, i.e. the cue point
        self._started_at: datetime | None = None    # wall-clock time of the lock
        # The most recent samples, held back from the file so the fade-out can be
        # applied to whichever ones turn out to be last.  See _write().
        self._tail = _EMPTY
        self._position = 0              # next unread sample position in the pipeline
        self._event_start = 0           # position at the moment of lock (max_seconds origin)
        self._last_lock = 0             # position as of the most recent LOCKED tick
        # Set when a recording ends, cleared by the first tick that sees no lock.  A
        # capped recording ends with the signal still present, and one event should
        # produce one file: without this the next tick would immediately open another
        # and a long event would come back as a pile of max_seconds fragments.
        self._await_relock = False

        # One lock for all of the above: tick() runs on the recorder thread while
        # arm(), disarm() and status() are called from the Qt thread, and a toggle
        # arriving mid-write must not tear the file's bookkeeping.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='recorder')

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and close any recording in progress.

        Finalising matters: a .wav's header carries its length, and a file whose
        writer never closed reports zero frames no matter how much audio is in it.
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._writer is not None:
                self._finish('shutdown')

    def arm(self) -> None:
        """Enable recording and refill the event budget."""
        with self._lock:
            self._armed = True
            self._events_remaining = self._initial_budget()
            # An explicit re-arm means "record now", even part-way through the event
            # whose recording was just capped or stopped by hand.
            self._await_relock = False
        logger.info('Recording armed — %s', self._budget_description())

    def disarm(self) -> None:
        """Disable recording, closing any recording in progress at its current length."""
        with self._lock:
            self._armed = False
            if self._writer is not None:
                self._finish('operator')
        logger.info('Recording disarmed')

    def toggle(self) -> bool:
        """Flip armed state (the toolbar button and R key); returns the new state."""
        if self.status().armed:
            self.disarm()
            return False
        self.arm()
        return True

    def status(self) -> RecorderStatus:
        with self._lock:
            return RecorderStatus(
                armed=self._armed,
                recording=self._writer is not None,
                events_remaining=self._events_remaining,
                elapsed_seconds=self._frames_accepted / self._sample_rate,
                filename=self._path.name if self._path is not None else None,
            )

    def tick(self) -> None:
        """Advance the recorder by one poll: capture audio, start and stop files.

        Public so tests can drive the state machine deterministically without
        running the thread.
        """
        with self._lock:
            self._tick()

    # ----------------------------------------------------------------- internal

    def _initial_budget(self) -> int | None:
        return self._max_events if self._max_events > 0 else None

    def _budget_description(self) -> str:
        remaining = self._events_remaining
        return 'every event' if remaining is None else f'{remaining} event(s)'

    def _run(self) -> None:  # pragma: no cover -- thread body; tick() is tested directly
        # Mirrors ContinuousAnalyzer._run(): a transient failure (a full disk, a
        # numerical edge case) must not silently kill the thread and leave recording
        # looking armed while nothing is ever written again.
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception('Recorder tick failed — retrying at the next poll.')
            self._stop.wait(self.POLL_INTERVAL)

    def _on_analyzer_state(self, state: AnalyzerState) -> None:
        """Analyzer state change, delivered on the analyzer thread.

        Deliberately trivial: it records what happened and returns.  Anything more —
        opening a file, writing audio — would run analysis-critical work behind disk
        I/O.  The recorder's own thread picks this up on its next tick.

        No lock is taken, for the same reason.  Both fields are plain flags, and a
        rebinding is atomic; the worst a race can do is have the tick that consumes
        _lock_acquired clear a lock the analyzer set microseconds earlier, in which
        case _locked is True anyway and the tick sees the lock regardless.
        """
        self._locked = state == AnalyzerState.LOCKED
        if self._locked:
            self._lock_acquired = True

    def _tick(self) -> None:
        # Either a lock right now, or one that came and went since the last tick.
        locked = self._locked or self._lock_acquired
        self._lock_acquired = False
        if self._writer is None:
            if not locked:
                self._await_relock = False
            elif self._armed and not self._await_relock:
                self._begin()
            return

        self._capture()
        if locked:
            self._last_lock = self._position
        if self._max_samples and self._position - self._event_start >= self._max_samples:
            self._finish('capped')
        elif self._position - self._last_lock >= self._timeout_samples:
            self._finish('timeout')

        if self._writer is None:
            # A recording that just ended with the signal still present is a capped
            # one; the event is not over, so hold off until it truly is.  One that
            # ended on the timeout is already over and the next lock is a new event.
            self._await_relock = locked

    def _begin(self) -> None:
        """Open a file for a newly locked event and write everything buffered so far."""
        now = datetime.now(self._zone)
        path = self._directory / event_filename(now)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._path = unique_path(path)
            self._writer = wave.open(str(self._path), 'wb')
            self._writer.setnchannels(1)
            self._writer.setsampwidth(_BYTES_PER_SAMPLE)
            self._writer.setframerate(self._sample_rate)
        except OSError:
            logger.exception('Cannot start recording in %s — disarming.', self._directory)
            self._writer, self._path, self._armed = None, None, False
            return

        # Position 0 reads the whole buffer, which is the lead-in: audio captured
        # before the lock that the ring buffer has not yet discarded.
        span = self._pipeline.read_from(0)
        self._frames_accepted, self._tail = 0, _EMPTY
        self._lead_in, self._started_at = len(span.samples), now
        self._event_start = self._last_lock = self._position = span.end
        self._write(span.samples)
        logger.info('Recording %s (%.1f s lead-in)',
                    self._path.name, self._lead_in / self._sample_rate)

    def _capture(self) -> None:
        """Write every sample captured since the previous poll, up to any length cap."""
        span = self._pipeline.read_from(self._position)
        if span.start > self._position:
            logger.warning('Recorder fell behind the ring buffer — %d samples lost.',
                           span.start - self._position)
        samples, end = span.samples, span.end
        if self._max_samples:
            # Trim the tail so a capped recording is exactly max_seconds long rather
            # than however far past the cap this poll happened to land.
            limit = self._event_start + self._max_samples
            if end > limit:
                samples, end = samples[:len(samples) - (end - limit)], limit
        self._write(samples)
        self._position = end

    def _write(self, samples: np.ndarray) -> None:
        """Take audio into the recording, holding back enough of it to fade out with.

        The end of a recording is only known after the fact — the tick that decides
        to stop has already been handed the audio that turned out to be last.  So a
        fade's worth of the newest samples never goes straight to the file; it waits
        here until either more audio arrives behind it, or the recording ends and
        _flush_tail() ramps it down to silence.  The file therefore trails the
        capture by 5 ms, which nothing depends on.
        """
        if samples.size == 0:
            return
        samples = self._faded_in(samples)
        self._frames_accepted += len(samples)
        pending = np.concatenate((self._tail, samples))
        held = len(self._fade_in)
        if len(pending) > held:
            self._emit(pending[:len(pending) - held])
            self._tail = pending[len(pending) - held:]
        else:
            self._tail = pending

    def _faded_in(self, samples: np.ndarray) -> np.ndarray:
        """Ramp up whatever part of `samples` falls inside the opening fade.

        Applied by position within the recording rather than per write, because the
        lead-in arrives as one large span and everything after it in small ones — the
        fade has to span whatever split the polling happens to produce.
        """
        remaining = len(self._fade_in) - self._frames_accepted
        if remaining <= 0:
            return samples
        n = min(remaining, len(samples))
        faded = samples.copy()
        ramp = self._fade_in[self._frames_accepted:self._frames_accepted + n]
        faded[:n] = np.rint(faded[:n] * ramp)
        return faded

    def _flush_tail(self) -> None:
        """Write the held-back samples, ramped down so the file ends at exactly zero.

        The ramp is built to the tail's own length, so a recording too short to have
        filled it still ends on silence rather than on a step.
        """
        self._emit(np.rint(self._tail * fade_ramp(len(self._tail))[::-1]).astype(np.int16))
        self._tail = _EMPTY

    def _emit(self, samples: np.ndarray) -> None:
        self._writer.writeframes(samples.astype('<i2', copy=False).tobytes())

    def _end_description(self, ended: str) -> str:
        """The log's version of an `ended` token, with the limit that produced it."""
        if ended == 'timeout':
            return f'no lock for {self._timeout_samples / self._sample_rate:g} s'
        if ended == 'capped':
            return f'reached the {self._max_samples / self._sample_rate:g} s limit'
        return _END_DESCRIPTIONS.get(ended, ended)

    def _write_metadata(self, ended: str) -> None:
        """Tag the finished file with what it is and how to read it back.

        Never allowed to fail the recording.  The audio is closed and safe by this
        point, and an untagged recording is still a perfectly good one — losing it
        over a metadata write would be a poor trade.
        """
        settings = wavmeta.format_settings({
            'sample_rate': self._sample_rate,
            'pulse_rate': self._pulse_rate,
            'audio_rf_conversion_db': self._rf_conversion_db,
            'lead_in_seconds': round(self._lead_in / self._sample_rate, 2),
            'ended': ended,
        })
        started = self._started_at.replace(microsecond=0).isoformat()
        try:
            wavmeta.append_metadata(
                self._path,
                {
                    'INAM': f'{self._callsign} powerline QRM event {started}',
                    'IART': self._callsign,
                    # Nominally a date; the full timestamp is more use and is widely
                    # accepted, and it carries the offset the filename also records.
                    'ICRD': started,
                    'ISFT': f'n6ol-powerline-qrm-monitor {__version__}',
                    'ICMT': settings,
                },
                {self._lead_in: 'LOCK'},
            )
        except OSError:
            logger.exception('Could not tag %s — the audio itself is unaffected.',
                             self._path.name)

    def _finish(self, ended: str) -> None:
        """Close the current file, tag it, count the event, and disarm if spent.

        `ended` is a short token naming why the recording stopped; it is written into
        the file's metadata, where it is the only way to tell a recording that ran its
        course from one the length cap cut short.
        """
        self._flush_tail()
        self._writer.close()
        self._writer = None
        self._write_metadata(ended)
        logger.info('Recorded %s — %.1f s (%s)', self._path.name,
                    self._frames_accepted / self._sample_rate, self._end_description(ended))
        self._path, self._frames_accepted = None, 0
        self._await_relock = True

        if self._events_remaining is None:
            return
        self._events_remaining -= 1
        if self._events_remaining <= 0:
            self._armed = False
            logger.info('Recording disarmed — event budget spent.')

"""
Automatic .wav capture of interference events.

EventRecorder watches the analyzer's state machine and writes each event it sees
to its own file, so an interesting burst can be replayed through buzz.playback
later - analyzed again on the same displays, at real speed, with no receiver
attached and no chance of missing it live.

A recording spans more than the event itself:

    |<-- lead-in -->|<---------- event ---------->|<-- trailer -->|
    buffered audio   LOCKED, sampled continuously   stop_after_seconds
    already captured                                without a lock
    when lock hit

The lead-in is free.  The ring buffer is always holding the last several seconds
of audio, so at the instant of lock the run-up to the event has already been
captured - the recorder just reads the buffer from its oldest surviving sample
rather than from the live tail.  Without that, every recording would begin with
the pulse train mid-stride, which is the least useful part to look at.

The trailer costs nothing either.  A recording ends because the signal has been
gone for stop_after_seconds, and that audio is written as it arrives rather than
being held back, so by the time the timeout expires the trailer is already in the
file.  The same timeout is what lets a flickering signal stay one recording: any
lock inside the window continues the event instead of splitting it in two.

Both ends are faded so the file starts and finishes at exactly zero and cannot
click - see fade_ramp for the shape, FADE_SECONDS for the length, and _write for
how the fade-out reaches audio whose lastness is only known afterwards.

The event budget can be a rate rather than a one-off.  With rearm_reset_minutes
set, max_events refills on that cycle - 10 and 1440 give ten events a day, every
day, unattended.  The cycle is measured from when the budget was last filled
rather than from when it ran out, so it keeps its time of day instead of sliding
later by however long each day's events took to arrive.

Lock is not polled.  The analyzer publishes each state change to a listener (see
ContinuousAnalyzer.add_state_listener) and the recorder's thread does the writing,
so a lock that comes and goes between two polls still starts a recording and
analysis never ends up behind disk I/O - the listener sets two flags under a lock
held for nothing longer than that, and does no work of its own.

Everything is measured in absolute sample positions rather than wall-clock time -
the same audio clock the analyzer's drift tracker uses.  A recording's length is
then exactly what its audio contains, regardless of when the polling thread
happened to run, and a stalled capture device cannot time out a recording that
has not actually gone quiet.
"""

import logging
import threading
import time
import wave
from collections import deque
from contextlib import suppress
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

# How each `ended` token written into a file's metadata reads in the log.  These are
# also exactly the two ends that are not the recorder's own doing - each is asked for
# from outside and announced by whoever asked - which is why the budget stays quiet
# about them in _finish().  The two tokens produced by a limit are described with
# that limit's value instead, by _end_description.
_END_DESCRIPTIONS = {
    'operator': 'stopped by operator',
    'shutdown': 'monitor shutting down',
}


@dataclass(frozen=True)
class RecorderStatus:
    """Immutable snapshot of the recorder, for the toolbar to poll and draw."""

    armed: bool
    recording: bool
    # None when max_events is 0 - record every event until switched off by hand.
    events_remaining: int | None
    # Seconds of audio since the lock that started the recording; 0 when idle.
    #
    # Timed from the lock rather than from the first sample in the file, for the same
    # reason max_seconds is: the lead-in is audio the monitor already had, so counting
    # it would start the display at whatever the ring buffer happened to hold, and
    # would disagree with the limit the recording is actually measured against.
    elapsed_seconds: float
    filename: str | None
    # Seconds until the event budget is refilled, or None when no cycle is running
    # (rearm_reset_minutes is 0, or recording was switched off by hand).
    rearm_in_seconds: float | None = None


def fade_ramp(n: int) -> np.ndarray:
    """An n-point raised-cosine ramp rising from exactly 0 to exactly 1.

    A file that begins or ends on a non-zero sample steps to or from silence, and a
    step is broadband: it clicks, and clicks at the seams when files are played back
    to back.  Sound cards carry a DC offset (the reason LevelStream removes one), so
    this happens even where the recording contains nothing but noise floor.

    Raised cosine rather than an exponential, which the ends of a file argue for on
    both counts.  An exponential approaches zero without reaching it, so it has to be
    truncated - and the truncation is itself a step, exactly what the fade was for.
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

    Two events cannot normally share a filename - stop_after_seconds keeps their
    lock instants at least a second apart - but replaying an old recording directory
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
    # sooner, and the audio itself is never sampled at this rate - each pass writes
    # every sample captured since the previous one, whenever it happens to run.
    POLL_INTERVAL = 0.2

    # Length of the fade at each end of a file (see fade_ramp for the shape).  A fade
    # of duration T spreads the transition it replaces over a bandwidth of about 1/T,
    # so a handful of samples - 0.25 ms, 4 kHz - merely turns a click into a quieter
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

    # How many published results min_lock_snr is judged over.  Five at the analyzer's
    # publishing cadence is about a second: long enough that one loud tick cannot let
    # a weak event through, short enough to follow a signal that is still building.
    SNR_WINDOW = 5

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
        # either wrong changes what the replay measures - see wavmeta.
        self._callsign         = config.station.callsign
        self._pulse_rate       = config.audio.pulse_rate
        self._rf_conversion_db = config.station.audio_rf_conversion_db
        # Both limits in samples, on the audio clock, and both clamped: a nonsensical
        # setting should degrade to the nearest sensible behavior rather than into
        # something surprising.  A negative cap would drive the write position back
        # behind itself and re-read audio already written, so anything at or below
        # zero means uncapped.  A zero timeout would end every recording on the tick
        # after it began, signal or no signal, so it floors at a single sample -
        # which reads as "stop as soon as the lock is lost".
        self._max_samples     = max(0, round(recording.max_seconds * self._sample_rate))
        self._timeout_samples = max(1, round(recording.stop_after_seconds * self._sample_rate))
        self._min_lock_samples = self._qualifying_lock_samples(recording.min_lock_seconds)
        self._fade_in = fade_ramp(round(self.FADE_SECONDS * self._sample_rate))

        # The analyzer, kept for its published levels rather than its state - the
        # state arrives by push because lock is an edge, but SNR is a level, and a
        # level is exactly the thing it is right to read when you happen to want it.
        self._analyzer = analyzer
        self._min_lock_snr = recording.min_lock_snr
        # Recent SNR readings taken while a lock is being judged.  A rolling window
        # rather than an average over the whole wait: an event that starts quiet and
        # builds should be recorded from the moment it is loud enough, and an average
        # dragged down by how it began would hold it off long after that.
        self._recent_snr: deque[float] = deque(maxlen=self.SNR_WINDOW)

        # Negative would leave the deadline permanently in the past, re-arming on
        # every tick and burying the log five lines a second; it means "never", as 0 does.
        self._rearm_period = max(0.0, recording.rearm_reset_minutes * 60)
        # Arming is what creates the directory - here, and in arm() the same way -
        # rather than the first event doing it.  A bad path is a configuration
        # mistake, and the moment to find out about one is while somebody is still
        # watching, not at the end of an unattended day from an empty folder that
        # explains nothing about why it is empty.
        #
        # The short-circuit matters.  Recording that is off reaches for
        # nothing at all, so a monitor run without it leaves no stray directory
        # behind and cannot complain about a path it was never going to use; the
        # check happens instead when the operator presses Record.
        self._armed = recording.enabled and self._can_record()
        self._events_remaining = self._initial_budget()
        # Monotonic deadline for the next budget reset, or None when no cycle is
        # running.  Set whenever the budget is filled, which is what makes the cycle
        # a fixed one: ten events per day means ten per day, not ten per day plus
        # however long the tenth event took to arrive.
        self._next_reset = self._reset_deadline() if self._armed else None

        # Lock state is pushed from the analyzer rather than read back from it (see
        # ContinuousAnalyzer.add_state_listener), seeded here with the state the
        # analyzer is in at wiring time.  _lock_acquired is sticky until the next
        # tick consumes it, so an event that locks and drops again inside a single
        # poll interval still starts a recording instead of vanishing between polls.
        self._locked = analyzer.state == AnalyzerState.LOCKED
        self._lock_acquired = self._locked
        self._lock_lost = False
        # Guards those two flags and nothing else, deliberately separate from the
        # _lock below: the analyzer thread has to take this one on every transition,
        # and the whole point of the push is that analysis never waits on the
        # recorder's disk I/O.  Only ever taken innermost, so the two cannot deadlock.
        self._state_lock = threading.Lock()
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
        # Pipeline position when the current unbroken lock began, or None when there
        # is no lock to be timing.  See _lock_has_held.
        # Which gate last held a recording back, for the opening log line to explain
        # itself with; None once nothing is holding it.  See _not_yet.
        self._held_back_by: str | None = None
        self._locked_since: int | None = None
        self._position = 0              # next unread sample position in the pipeline
        self._event_start = 0           # position where recording began (max_seconds origin)
        self._last_lock = 0             # position as of the most recent LOCKED tick
        # Set when a recording ends, cleared by the first tick that sees no lock.  A
        # capped recording ends with the signal still present, and one event should
        # produce one file: without this the next tick would immediately open another
        # and a long event would come back as a pile of max_seconds fragments.
        self._await_relock = False

        # One lock for the recording state above, and for the budget and the re-arm
        # cycle: tick() runs on the recorder thread while arm(), disarm() and status()
        # are called from the Qt thread, and a toggle arriving mid-write must not tear
        # the file's bookkeeping.  The two lock flags are the exception, on
        # _state_lock, because the analyzer thread touches those and must not end up
        # behind a disk write.
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
        """Enable recording and refill the event budget, restarting the reset cycle.

        Re-checks the directory, so this is also how an operator retries after fixing
        a bad path: arming is refused, loudly, while there is nowhere to record to.
        """
        if not self._can_record():
            return
        with self._lock:
            self._fill_budget()
            # An explicit re-arm means "record now", even part-way through the event
            # whose recording was just capped or stopped by hand.
            self._await_relock = False
        logger.info('Recording armed - %s', self._budget_description())

    def disarm(self) -> None:
        """Disable recording, closing any recording in progress at its current length.

        Cancels the reset cycle as well.  Switching recording off by hand has to mean
        off: coming back to a monitor that re-armed itself overnight, because it was
        turned off during a cycle rather than between two, would be a nasty surprise.
        """
        with self._lock:
            self._armed = False
            self._next_reset = None
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
        """A snapshot of the recorder's current state, for the toolbar to poll and draw."""
        with self._lock:
            return RecorderStatus(
                armed=self._armed,
                recording=self._writer is not None,
                events_remaining=self._events_remaining,
                # Explicitly zero when idle: the positions below are left where the
                # last recording ended, so the subtraction would otherwise keep
                # reporting that recording's length long after it closed.
                elapsed_seconds=(0.0 if self._writer is None else
                                 (self._position - self._event_start) / self._sample_rate),
                filename=self._path.name if self._path is not None else None,
                rearm_in_seconds=(None if self._next_reset is None
                                  else max(0.0, self._next_reset - time.monotonic())),
            )

    def tick(self) -> None:
        """Advance the recorder by one poll: capture audio, start and stop files.

        Public so tests can drive the state machine deterministically without
        running the thread.
        """
        with self._lock:
            self._tick()

    # ----------------------------------------------------------------- internal

    def _max_lead_in_samples(self) -> int:
        """The most lead-in this configuration can produce, for the metadata.

        Recorded alongside lead_in_seconds because without it the figure is censored
        data wearing the clothes of a measurement.  The buffer is a sliding window and
        the wait for min_lock_seconds runs while it slides, so the lead-in can never
        exceed the buffer's capacity less that wait.  A recording sitting exactly at
        that bound is saying "everything there was", not "the analyzer took this long
        to lock" -- and from the file alone the two are indistinguishable, since
        neither the buffer size nor min_lock_seconds is otherwise recorded.

        Observed across fourteen real recordings: eight clustered at 6.56-6.59 s with a
        9.60 s buffer and min_lock_seconds of 3, which is this bound to within a poll.
        The rest, from 0.0 to 3.26, are true measurements.  Nothing in the file said
        which was which.
        """
        return max(0, self._pipeline.capacity_samples - self._min_lock_samples)

    def _qualifying_lock_samples(self, seconds: float) -> int:
        """How long a lock must hold before it is worth a file, in samples.

        Two ceilings, and the wait is clamped to whichever is lower.

        The ring buffer, because waiting is not free: the buffer is a sliding window,
        so every second spent deciding is a second of run-up that has fallen off the
        far end by the time the file opens.  Wait longer than the buffer and the
        recording would begin *after* the lock, missing the onset of the very event
        it exists to capture.

        And max_seconds, because the wait is counted against it - those seconds are
        part of the event, so asking for three of them and ten of recording buys ten
        and not thirteen.  A wait longer than the whole allowance is a contradiction:
        left alone it would reach back past the audio the file opens with and throw
        away its newest seconds to stay inside a limit already spent.  Clamped, the
        two settings meet at the sensible end of it - the recording is exactly the
        allowance, all of it from what the buffer was already holding.
        """
        capacity = self._pipeline.capacity_samples
        wanted = max(0, round(seconds * self._sample_rate))
        if wanted > capacity:
            logger.warning(
                'min_lock_seconds of %g is longer than the %.1f s of audio the buffer '
                'holds; using %.1f s, beyond which a recording would start after the '
                'event it is recording.',
                seconds, capacity / self._sample_rate, capacity / self._sample_rate)
            wanted = capacity
        if self._max_samples and wanted > self._max_samples:
            logger.warning(
                'min_lock_seconds of %g is longer than max_seconds of %g; using %g, '
                'since the wait counts against the recording length and cannot be '
                'longer than the whole of it.',
                seconds, self._max_samples / self._sample_rate,
                self._max_samples / self._sample_rate)
            wanted = self._max_samples
        return wanted

    def _can_record(self) -> bool:
        """Whether recording is possible at all, before anything is armed.

        The sample rate is checked because every duration here is derived by dividing
        by it, and because wave refuses to write a file without a valid one - a check
        at the point of arming turns what would otherwise be a failure per poll, plus
        a stray file for each, into one message and a recorder that stays off.
        """
        if self._sample_rate <= 0:
            logger.error('Cannot record at a sample rate of %s - check sample_rate in '
                         'the [audio] section of the config.', self._sample_rate)
            return False
        return self._ensure_directory()

    def _ensure_directory(self) -> bool:
        """Create the recording directory if needed; report whether it is usable.

        Deliberately not fatal.  Measuring and logging the interference is the
        monitor's job and recording is an extra, so a directory nobody can write to
        costs the operator their recordings, not their day's data.
        """
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:
            logger.error('Cannot create the recording directory %s (%s) - recording '
                         'is off.  Check the directory setting in the [recording] '
                         'section of the config, and permissions on that path.',
                         self._directory, exc)
            return False

    def _initial_budget(self) -> int | None:
        return self._max_events if self._max_events > 0 else None

    def _reset_deadline(self) -> float | None:
        """When the budget should next be refilled, or None if it never should."""
        return time.monotonic() + self._rearm_period if self._rearm_period else None

    def _fill_budget(self) -> None:
        """Arm with a full budget and start the cycle again (caller holds the lock)."""
        self._armed = True
        self._events_remaining = self._initial_budget()
        self._next_reset = self._reset_deadline()

    def _reset_budget(self) -> None:
        """Refill the budget because the cycle came round (caller holds the lock).

        Deliberately not a call to _fill_budget: the next deadline advances by exactly
        one period from the last one rather than from now, so a cycle cannot drift by
        the poll interval each time round.  A cycle missed entirely - a suspended
        machine, a very long stall - restarts from now rather than firing repeatedly
        to catch up on events that could not have happened anyway.

        _await_relock is left alone.  Unlike arming by hand, this is not somebody
        asking to record right now, so an event already in progress stays finished as
        far as the recorder is concerned and the next one starts the next file.
        """
        now = time.monotonic()
        self._armed = True
        self._events_remaining = self._initial_budget()
        self._next_reset += self._rearm_period
        if self._next_reset <= now:
            self._next_reset = now + self._rearm_period
        logger.info('Recording re-armed - %s', self._budget_description())

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
                logger.exception('Recorder tick failed - retrying at the next poll.')
            self._stop.wait(self.POLL_INTERVAL)

    def _on_analyzer_state(self, state: AnalyzerState) -> None:
        """Analyzer state change, delivered on the analyzer thread.

        Deliberately trivial: it records what happened and returns.  Anything more -
        opening a file, writing audio - would run analysis-critical work behind disk
        I/O.  The recorder's own thread picks this up on its next tick.

        The lock it does take is _state_lock, which guards these two flags alone and
        is never held across anything slower than an assignment, so the analyzer is
        not waiting on the recorder in any sense that matters.
        """
        with self._state_lock:
            self._locked = state == AnalyzerState.LOCKED
            if self._locked:
                self._lock_acquired = True
            else:
                self._lock_lost = True

    def _not_yet(self) -> str | None:
        """Which gate is still holding a recording back, phrased for the log.

        Recorded rather than inferred afterwards.  By the time a recording starts
        both gates are satisfied and nothing in its state says which of them the
        waiting was for - a question that has already cost three wrong guesses and a
        measurement rig to answer from the outside.
        """
        if not self._lock_has_held():
            return 'waiting out min_lock_seconds'
        if not self._is_loud_enough():
            return 'waiting for the signal to reach min_lock_snr'
        return None

    def _lock_has_held(self) -> bool:
        """Whether the current lock has lasted long enough to be worth a file.

        Counted in captured samples from the moment the lock appeared, so a stalled
        capture device cannot age a lock that no audio is arriving to support.

        Any reported loss restarts the count, including one that comes and goes
        between two polls.  The analyzer debounces a real loss across several checks
        before it reports one, so a loss that reaches here is a true one - and
        without this, a stream of blips accumulates: no tick ever observes the gaps,
        because the next blip has set the flag again before it runs.
        """
        held = self._pipeline.total_samples - self._locked_since
        return held >= self._min_lock_samples

    def _sample_snr(self) -> None:
        """Note the analyzer's latest SNR, while a lock is waiting to be judged.

        Only locked results carry a level - an unlocked one reports zero by
        convention (see AnalysisResult.unlocked), and averaging those in would hold
        off a perfectly loud event for the sake of a reading that measured nothing.
        """
        if not self._min_lock_snr:
            return
        result = self._analyzer.latest_result()
        if result is not None and result.locked:
            self._recent_snr.append(result.snr)

    def _is_loud_enough(self) -> bool:
        """Whether the signal has reached min_lock_snr, judged over recent readings.

        An event below the bar is watched, not abandoned: powerline arcs commonly
        start quiet and build, and one that crosses ten seconds in is still worth
        recording from the moment it does - at the cost of that much lead-in, which
        is the trade this setting makes and the reason to keep it modest.

        The window must be full before it decides anything.  Judged on one or two
        readings, a single loud tick averages away a weak event and lets it through -
        and the first readings after a cold lock are the ones least worth trusting
        anyway, since the analyzer's drift estimate has not converged and levels read
        several dB low until it does.  Waiting for a full window costs about a second
        and buys a measurement worth thresholding on.
        """
        if not self._min_lock_snr:
            return True
        if len(self._recent_snr) < self.SNR_WINDOW:
            return False
        return sum(self._recent_snr) / self.SNR_WINDOW >= self._min_lock_snr

    def _consume_lock_state(self) -> tuple[bool, bool]:
        """(locked since the previous tick, lost since the previous tick).

        Reading the live flag and clearing the sticky ones are one operation because
        they have to be.  Done separately, a lock and a loss arriving in the gap
        between them would clear the flag that recorded the lock while leaving the
        level saying there is none - and a brief lock vanishing between two polls is
        precisely the failure the push exists to prevent.

        Both edges are reported, not just the acquisition.  A run of brief locks
        looks identical to one long lock if all you know is that there was a lock
        since the last poll: no tick ever sees the gaps, because each blip sets the
        flag again before the next one runs.  Knowing a loss happened is what tells
        min_lock_seconds that the clock has to start again.
        """
        with self._state_lock:
            locked = self._locked or self._lock_acquired
            lost = self._lock_lost
            self._lock_acquired = self._lock_lost = False
        return locked, lost

    def _tick(self) -> None:
        # Before anything else, so a budget coming back round can record an event
        # that is already under way rather than waiting for the next poll.
        if self._next_reset is not None and time.monotonic() >= self._next_reset:
            self._reset_budget()

        # Either a lock right now, or one that came and went since the last tick.
        locked, lost = self._consume_lock_state()
        if self._writer is None:
            if not locked:
                self._locked_since = None
                self._recent_snr.clear()
                self._held_back_by = None
                self._await_relock = False
            else:
                if lost or self._locked_since is None:
                    self._locked_since = self._pipeline.total_samples
                    self._recent_snr.clear()
                    self._held_back_by = None
                self._sample_snr()
                if self._armed and not self._await_relock:
                    # Remembered only while something is blocking.  Overwriting it on
                    # the tick that finally starts would clear the answer just as the
                    # question is asked, since nothing is blocking by then; acquiring
                    # a lock is what clears it, above.
                    blocking = self._not_yet()
                    if blocking is None:
                        self._begin()
                    else:
                        self._held_back_by = blocking
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
        writer = None
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._path = unique_path(path)
            writer = wave.open(str(self._path), 'wb')
            writer.setnchannels(1)
            writer.setsampwidth(_BYTES_PER_SAMPLE)
            writer.setframerate(self._sample_rate)
        except Exception:
            # Any failure, not just OSError: wave rejects a bad frame rate with its
            # own exception, and an escape from here would leave a half-configured
            # writer in place for the next tick to trip over - and drop a stray file
            # per poll for as long as the signal lasts.  Built into a local and only
            # published on success, so a failure cannot leave one half-installed.
            logger.exception('Cannot start recording in %s - disarming.', self._directory)
            if writer is not None:
                # Closing a writer that never got its header settings raises in turn,
                # and the reason we are here is already logged.
                with suppress(Exception):
                    writer.close()
            self._writer, self._path, self._armed = None, None, False
            return
        self._writer = writer

        # Position 0 reads the whole buffer: everything still held from before this
        # moment, which is the run-up the file opens with.
        span = self._pipeline.read_from(0)
        self._frames_accepted, self._tail = 0, _EMPTY
        # Where the lock actually sits inside that, which is not where the file
        # starts and not where this decision is being taken.  min_lock_seconds puts
        # those seconds between the two, and they are part of the event: the cue
        # marker, the metadata and the length cap all measure from the lock, so all
        # three would be wrong by exactly that much if this used either end instead.
        locked_at = span.end if self._locked_since is None else self._locked_since
        self._lead_in, self._started_at = max(0, locked_at - span.start), now
        # The cap runs from the moment recording begins, less whatever was spent
        # waiting out min_lock_seconds - those seconds are the event too, and asking
        # for three of them and ten of recording should not quietly buy thirteen.
        # (_lock_has_held guarantees at least that much has passed, so the
        # subtraction never reaches back further than the wait actually was.)
        #
        # Only that wait is charged.  A min_lock_snr wait is open-ended, and charging
        # it would spend the whole cap before the file was opened - which is how a
        # real event once came to be saved as a nought-second recording.  What the
        # buffer holds beyond the deliberate wait is lead-in, and lead-in is free.
        #
        # The charge is the configured wait rather than the observed one, so a
        # recording overruns max_seconds by however late the poll was in noticing it -
        # up to POLL_INTERVAL.  Charging what was observed would be exact here and
        # unbounded for min_lock_snr, so the overrun is documented rather than chased.
        self._event_start = span.end - self._min_lock_samples
        # Capped here too, not only in _capture: with min_lock_seconds this opening
        # write already contains audio from after the lock, so a cap shorter than the
        # wait would otherwise be overrun before the first poll ever looked at it.
        samples, end = self._clamp_to_cap(span.samples, span.end)
        self._last_lock = self._position = end
        self._write(samples)
        # Both of the reasons a recording is not what the settings might suggest, said
        # plainly, because neither is recoverable from the file afterwards.  A monitor
        # that has just started has not filled its buffer, so an arc already buzzing
        # when it did gets a shorter run-up than the same arc would an hour later -
        # which reads as a bug.  And a start delayed by a gate is a number the
        # operator would otherwise have to work back to from the length of the file.
        notes = []
        if span.start == 0:
            notes.append('all the audio captured so far')
        if self._held_back_by is not None:
            notes.append(f'started {(span.end - locked_at) / self._sample_rate:.1f} s '
                         f'after the lock, {self._held_back_by}')
        logger.info('Recording %s (%.1f s lead-in%s)', self._path.name,
                    self._lead_in / self._sample_rate,
                    '; ' + '; '.join(notes) if notes else '')
        self._held_back_by = None

    def _capture(self) -> None:
        """Write every sample captured since the previous poll, up to any length cap."""
        span = self._pipeline.read_from(self._position)
        if span.start > self._position:
            logger.warning('Recorder fell behind the ring buffer - %d samples lost.',
                           span.start - self._position)
        samples, end = self._clamp_to_cap(span.samples, span.end)
        self._write(samples)
        self._position = end

    def _clamp_to_cap(self, samples: np.ndarray, end: int) -> tuple[np.ndarray, int]:
        """Trim a span so a capped recording ends exactly at the cap.

        Without this a recording runs to wherever the poll that noticed happened to
        fall, rather than to the length that was asked for.

        max(0) because the span can begin past the cap outright: a thread stalled for
        longer than the ring buffer holds - a suspend and resume - comes back to find
        the oldest surviving sample already beyond it.  A bare negative index would
        silently trim from the wrong end.
        """
        if not self._max_samples:
            return samples, end
        limit = self._event_start + self._max_samples
        if end <= limit:
            return samples, end
        return samples[:max(0, len(samples) - (end - limit))], limit

    def _write(self, samples: np.ndarray) -> None:
        """Take audio into the recording, holding back enough of it to fade out with.

        The end of a recording is only known after the fact - the tick that decides
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
        lead-in arrives as one large span and everything after it in small ones - the
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
        point, and an untagged recording is still a perfectly good one - losing it
        over a metadata write would be a poor trade.
        """
        settings = wavmeta.format_settings({
            'sample_rate': self._sample_rate,
            'pulse_rate': self._pulse_rate,
            'audio_rf_conversion_db': self._rf_conversion_db,
            'lead_in_seconds': round(self._lead_in / self._sample_rate, 2),
            'lead_in_max_seconds': round(self._max_lead_in_samples() / self._sample_rate, 2),
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
            logger.exception('Could not tag %s - the audio itself is unaffected.',
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
        # Broken down, because the total is not the number any setting names and
        # working out why takes knowing that max_seconds runs from the lock while the
        # lead-in sits outside it - which is a lot to ask of somebody reading a log at
        # the end of a night.
        logger.info('Recorded %s - %.1f s: %.1f s lead-in + %.1f s from the lock (%s)',
                    self._path.name,
                    self._frames_accepted / self._sample_rate,
                    self._lead_in / self._sample_rate,
                    (self._frames_accepted - self._lead_in) / self._sample_rate,
                    self._end_description(ended))
        self._path, self._frames_accepted = None, 0
        self._await_relock = True

        if self._events_remaining is None:
            return
        self._events_remaining -= 1
        if self._events_remaining > 0:
            return
        self._armed = False
        # Announced only when the budget is why recording is now off.  A recording
        # stopped by hand or at shutdown spends its event too, but whatever asked for
        # that has already disarmed and said so, and a second line naming a different
        # cause reads as the monitor contradicting itself.
        if ended not in _END_DESCRIPTIONS:
            logger.info('Recording disarmed - event budget spent.')

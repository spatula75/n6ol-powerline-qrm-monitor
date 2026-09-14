"""
Automatic .wav capture of interference events.

Two responsibilities, deliberately in two classes.  RecordingTrigger decides *when* an
event is worth a file and when that file should end.  An AbstractEventRecorder subclass
handles *how* one gets written: which pipeline the samples come from, how they are
shaped into a frame, and what metadata the finished file carries.  The trigger drives
one or more recorders through begin, capture and finish, and has no opinion about any
of those answers.

The split exists so that a second format needs no second state machine.  Raw IQ capture
is the case it was made for, and docs-notebook/iq-recording-design.md records the
reasoning.  One trigger owning the lock gating and the event budget is what stops two
recorders ever disagreeing about whether an event happened, or about why it ended.

The trigger watches the analyzer's state machine and gives each event its own file, so
an interesting burst can be replayed through buzz.playback later: analyzed again on the
same displays, at real speed, with no receiver attached and no chance of missing it
live.

A recording spans more than the event itself:

    |<-- lead-in -->|<---------- event ---------->|<-- trailer -->|
    buffered audio   LOCKED, sampled continuously   stop_after_seconds
    already captured                                without a lock
    when lock hit

The lead-in is free.  The ring buffer always holds the last several seconds of
audio, so at the instant of lock the run-up to the event is already captured.  The
recorder just reads the buffer from its oldest surviving sample rather than from the
live tail.  Without that, every recording would begin with the pulse train
mid-stride, which is the least useful part to look at.

The trailer costs nothing either.  A recording ends because the signal has been gone
for stop_after_seconds, and the recorder writes that audio as it arrives rather than
holding it back, so by the time the timeout expires the trailer is already in the
file.  The same timeout is what lets a flickering signal stay one recording: any
lock inside the window continues the event instead of splitting it in two.

Both ends are faded so the file starts and finishes at exactly zero and cannot
click - see fade_ramp for the shape, FADE_SECONDS for the length, and _write for
how the fade-out reaches audio whose lastness is only known afterward.

The event budget can be a rate rather than a one-off.  With rearm_reset_minutes set,
max_events refills on that cycle: 10 and 1440 give ten events a day, every day,
unattended.  The cycle is measured from when the budget was last filled rather than
from when it ran out, so it keeps its time of day instead of sliding later by
however long each day's events took to arrive.

Lock is not polled.  The analyzer publishes each state change to a listener (see
ContinuousAnalyzer.add_state_listener), and the trigger's own thread does the driving,
so a lock that comes and goes between two polls still starts a recording and analysis
never ends up behind disk I/O.  The listener sets two flags under a lock held for
nothing longer than that, and does no work of its own.

Everything is measured in absolute sample positions rather than wall-clock time, the
same audio clock the analyzer's drift tracker uses.  A recording's length is then
exactly what its audio contains, regardless of when the polling thread happened to
run, and a stalled capture device cannot time out a recording that has not actually
gone quiet.
"""


import logging
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from buzz import __version__, wavmeta
from buzz.analyzer import AnalyzerState, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.sampler import RingBufferPipeline

logger = logging.getLogger(__name__)



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
    # reason max_seconds is.  The lead-in is audio the monitor already had, so counting
    # it would start the display at whatever the ring buffer happened to hold, and
    # would disagree with the limit the recording is actually measured against.
    elapsed_seconds: float
    filename: str | None
    # Seconds until the event budget is refilled, or None when no cycle is running
    # (rearm_reset_minutes is 0, or recording was switched off by hand).
    rearm_in_seconds: float | None = None


class AbstractEventRecorder(ABC):
    """Writes one event to one file.  Told when to start and stop, and decides neither.

    A subclass answers what the trigger has no opinion about: which pipeline the
    samples come from, how they are shaped into a frame, how long the fade at each end
    runs, and what the finished file's metadata says.  Everything else here - the
    lead-in read, the length cap, the held-back fade tail, the RIFF tagging - is the
    same whatever those answers are.

    The three static methods below were module functions.  Only a recorder builds an
    event filename, resolves a collision, or shapes a fade, so they sit on the class
    that does those things, per the rule in CLAUDE.md about encapsulating by default.
    """

    # Length of the fade at each end of a file (see fade_ramp for the shape).  A fade
    # of duration T spreads the transition it replaces over a bandwidth of about 1/T,
    # so a handful of samples - 0.25 ms, 4 kHz - merely turns a click into a quieter
    # click.  Audibility falls off past a millisecond and is gone by about five, which
    # is where audio editors put their default fades at edit points.
    #
    # Sized for a cut through full-scale audio, because both ends can be one.  A file
    # usually opens in quiet lead-in and closes in the silence the event faded into,
    # but not always.  An arc already buzzing when the monitor starts is locked onto
    # within a second or two, so the lead-in is a live pulse train from its first
    # sample, and a max_seconds cap ends a file mid-event the same way.  Even then the
    # fade gives up well under one pulse out of the 120 in that second.
    #
    # A subclass wanting no fade at all sets this to zero rather than branching:
    # fade_ramp(0) is empty, so nothing is held back and every write goes straight out.
    FADE_SECONDS = 0.005

    # The frame format, which a subclass states in full because the three have to
    # agree.  The width in bytes is what `wave` writes per sample, and the dtype is
    # how the samples are packed to reach it.
    CHANNELS: int
    SAMPLE_WIDTH_BYTES: int
    SAMPLE_DTYPE: str

    # What the file holds, for the INAM tag.  Raw IQ and audio are not interchangeable
    # to whoever opens one later, so each says which it is.
    KIND: str

    # Appended to the filename, so that two recorders of one event are one glance
    # apart in the directory.  Empty for the audio, which is the one an operator
    # reaches for and the one every other part of this program reads.
    FILENAME_SUFFIX = ''

    def __init__(self, pipeline: RingBufferPipeline, sample_rate: int, directory: Path,
                 callsign: str, max_seconds: float, charged_wait_seconds: float) -> None:
        self._pipeline = pipeline
        self._sample_rate = sample_rate
        self._directory = directory
        self._callsign = callsign
        # The cap in this recorder's own samples, clamped the way the trigger's is:
        # anything at or below zero means uncapped, since a negative would drive the
        # write position back behind itself and re-read audio already written.
        self._max_samples = max(0, round(max_seconds * self._sample_rate))
        # How much of the wait before the lock is charged against that cap.  The
        # trigger works the figure out once, clamped against the buffer and the cap,
        # and hands it over in seconds so that every recorder converts it at its own
        # rate.  See RecordingTrigger._qualifying_lock_samples for why it is clamped.
        self._charged_wait_samples = max(0, round(charged_wait_seconds * self._sample_rate))
        self._fade_in = self.fade_ramp(round(self.FADE_SECONDS * self._sample_rate))

        # Current recording, all None/0 while idle.
        self._writer: wave.Wave_write | None = None
        self._path: Path | None = None
        self._frames_accepted = 0       # audio taken into the recording, tail included
        self._lead_in = 0               # samples captured before the lock, i.e. the cue point
        self._started_at: datetime | None = None    # wall-clock time of the lock
        # The most recent samples, held back from the file so the fade-out can be
        # applied to whichever ones turn out to be last.  See _write().
        self._tail = self._no_frames
        self._position = 0              # next unread sample position in the pipeline
        self._event_start = 0           # position where recording began (max_seconds origin)

    # ------------------------------------------------------------------ public

    @property
    def _no_frames(self) -> np.ndarray:
        """An empty run of frames, shaped the way this recorder's own samples arrive.

        Mono audio is a flat array and stereo IQ is one row per frame.  The two cannot
        be concatenated with each other, so the empty the tail starts as has to match
        whatever will be appended to it.
        """
        return np.empty(0 if self.CHANNELS == 1 else (0, self.CHANNELS),
                        dtype=self.SAMPLE_DTYPE)

    @property
    def is_recording(self) -> bool:
        """Whether a file is open right now."""
        return self._writer is not None

    @property
    def filename(self) -> str | None:
        """The name of the file being written, or None while idle."""
        return self._path.name if self._path is not None else None

    def can_record(self) -> bool:
        """Whether recording is possible at all, before anything is armed.

        This checks the sample rate because every duration here is derived by
        dividing by it, and because wave refuses to write a file without a valid
        one.  A check at the point of arming turns what would otherwise be a
        failure per poll, plus a stray file for each, into one message and a
        recorder that stays off.
        """
        if self._sample_rate <= 0:
            logger.error('Cannot record at a sample rate of %s - check sample_rate in '
                         'the [audio] section of the config.', self._sample_rate)
            return False
        return self._ensure_directory()

    @staticmethod
    def fade_ramp(n: int) -> np.ndarray:
        """An n-point raised-cosine ramp rising from exactly 0 to exactly 1.

        A file that begins or ends on a non-zero sample steps to or from silence, and a
        step is broadband: it clicks, and clicks at the seams when files play back to
        back.  Sound cards carry a DC offset (the reason LevelStream removes one), so
        this happens even where the recording contains nothing but noise floor.

        This uses a raised cosine rather than an exponential, and the ends of a file
        argue for that on both counts.  An exponential approaches zero without reaching
        it, so it has to be truncated, and the truncation is itself a step, exactly what
        the fade was for.  This shape hits 0 and 1 exactly, and meets both of them with
        zero slope, so the join to silence and the join to full-scale audio are each
        smooth.  That continuity is worth 6 dB/octave of splatter rolloff over a linear
        ramp's corner, for the same cost.

        Reverse it for the fade-out; the last sample is then exactly zero.
        """
        return 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n))

    @staticmethod
    def event_filename(when: datetime, suffix: str = '') -> str:
        """Return the .wav filename for an event that locked at `when`.

        The suffix is what keeps two recorders of one event apart.  Both open at the
        same instant, so both would otherwise ask for the same name and the second
        would be pushed to a -2 that says nothing about which file it is.

        Local time with the UTC offset attached, so a file stays unambiguous a year
        later and across a DST change.  ISO 8601's colons are illegal in Windows
        filenames and its T separator is hard to read at a glance, so date and time are
        joined with a dash instead: event-20260729-143307-0700.wav.
        """
        return f'event-{when.strftime("%Y%m%d-%H%M%S%z")}{suffix}.wav'

    @staticmethod
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

    def _max_lead_in_samples(self) -> int:
        """The most lead-in this configuration can produce, for the metadata.

        This is recorded alongside lead_in_seconds because without it the figure is
        censored data wearing the clothes of a measurement.  The buffer is a sliding
        window, and the wait for min_lock_seconds runs while it slides, so the
        lead-in can never exceed the buffer's capacity less that wait.  A recording
        sitting exactly at that bound is saying "everything there was", not "the
        analyzer took this long to lock" - and from the file alone the two are
        indistinguishable, since neither the buffer size nor min_lock_seconds is
        otherwise recorded.

        Observed across fourteen real recordings: eight clustered at 6.56-6.59 s with
        a 9.60 s buffer and min_lock_seconds of 3, which is this bound to within a
        poll.  The rest, from 0.0 to 3.26, are true measurements.  Nothing in the
        file said which was which.
        """
        return max(0, self._pipeline.capacity_samples - self._charged_wait_samples)

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

    def begin(self, started_at: datetime, lock_age_seconds: float | None,
              held_back_by: str | None) -> bool:
        """Open a file for a newly locked event and write everything buffered so far.

        Returns whether a file was opened.  The caller decides what a failure means;
        this only reports it, having already said why in the log.

        The age of the lock arrives in seconds rather than as a position, because a
        recorder reading a different pipeline at a different rate has to place the cue
        marker in its own samples.  See docs-notebook/iq-recording-design.md.
        """
        path = self._directory / self.event_filename(started_at, self.FILENAME_SUFFIX)
        writer = None
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._path = self.unique_path(path)
            writer = wave.open(str(self._path), 'wb')
            writer.setnchannels(self.CHANNELS)
            writer.setsampwidth(self.SAMPLE_WIDTH_BYTES)
            writer.setframerate(self._sample_rate)
        except Exception:
            # Any failure, not just OSError: wave rejects a bad frame rate with its
            # own exception.  An escape from here would leave a half-configured
            # writer in place for the next tick to trip over, dropping a stray file
            # per poll for as long as the signal lasts.  This builds the writer into
            # a local and only publishes it on success, so a failure cannot leave
            # one half-installed.
            logger.exception('Cannot start recording in %s - disarming.', self._directory)
            if writer is not None:
                # Closing a writer that never got its header settings raises in turn,
                # and the reason we are here is already logged.
                with suppress(Exception):
                    writer.close()
            self._writer, self._path = None, None
            return False
        self._writer = writer

        # Position 0 reads the whole buffer: everything still held from before this
        # moment, which is the run-up the file opens with.
        span = self._pipeline.read_from(0)
        self._frames_accepted, self._tail = 0, self._no_frames
        # Where the lock actually sits inside that span, which is not where the file
        # starts and not where this decision is being taken.  min_lock_seconds puts
        # those seconds between the two, and they are part of the event: the cue
        # marker, the metadata and the length cap all measure from the lock, so all
        # three would be wrong by exactly that much if this used either end instead.
        locked_at = (span.end if lock_age_seconds is None
                     else span.end - round(lock_age_seconds * self._sample_rate))
        self._lead_in, self._started_at = max(0, locked_at - span.start), started_at
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
        # recording overruns max_seconds by however late the poll was in noticing it,
        # up to POLL_INTERVAL.  Charging what was observed would be exact here and
        # unbounded for min_lock_snr, so the overrun is documented rather than chased.
        self._event_start = span.end - self._charged_wait_samples
        # Capped here too, not only in _capture: with min_lock_seconds this opening
        # write already contains audio from after the lock, so a cap shorter than the
        # wait would otherwise be overrun before the first poll ever looked at it.
        samples, end = self._clamp_to_cap(span.samples, span.end)
        self._position = end
        self._write(samples)
        # Both of the reasons a recording is not what the settings might suggest, said
        # plainly, because neither is recoverable from the file afterward.  A monitor
        # that has just started has not filled its buffer, so an arc already buzzing
        # when it did gets a shorter run-up than the same arc would an hour later,
        # which reads as a bug.  A start delayed by a gate is also a number the
        # operator would otherwise have to work back to from the length of the file.
        notes = []
        if span.start == 0:
            notes.append('all the audio captured so far')
        if held_back_by is not None:
            notes.append(f'started {(span.end - locked_at) / self._sample_rate:.1f} s '
                         f'after the lock, {held_back_by}')
        logger.info('Recording %s (%.1f s lead-in%s)', self._path.name,
                    self._lead_in / self._sample_rate,
                    '; ' + '; '.join(notes) if notes else '')
        return True

    def capture(self) -> None:
        """Write every sample captured since the previous poll, up to any length cap."""
        span = self._pipeline.read_from(self._position)
        if span.start > self._position:
            logger.warning('Recorder fell behind the ring buffer - %d samples lost.',
                           span.start - self._position)
        samples, end = self._clamp_to_cap(span.samples, span.end)
        self._write(samples)
        self._position = end

    def finish(self, ended: str, description: str) -> None:
        """Close the current file and tag it.

        `ended` is a short token naming why the recording stopped.  It is written
        into the file's metadata, where it is the only way to tell a recording that
        ran its course from one the length cap cut short.  `description` is the same
        reason phrased for the log, which the caller words because the limits that
        produced it are the caller's.
        """
        self._flush_tail()
        self._writer.close()
        self._writer = None
        self._write_metadata(ended)
        # Broken down, because the total is not the number any setting names, and
        # working out why takes knowing that max_seconds runs from the lock while the
        # lead-in sits outside it - a lot to ask of somebody reading a log at the end
        # of a night.
        logger.info('Recorded %s - %.1f s: %.1f s lead-in + %.1f s from the lock (%s)',
                    self._path.name,
                    self._frames_accepted / self._sample_rate,
                    self._lead_in / self._sample_rate,
                    (self._frames_accepted - self._lead_in) / self._sample_rate,
                    description)
        self._path, self._frames_accepted = None, 0

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

        The end of a recording is only known after the fact: the tick that decides to
        stop has already been handed the audio that turned out to be last.  So a
        fade's worth of the newest samples never goes straight to the file.  It waits
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
        lead-in arrives as one large span and everything after it in small ones, so
        the fade has to span whatever split the polling happens to produce.
        """
        remaining = len(self._fade_in) - self._frames_accepted
        if remaining <= 0:
            return samples
        n = min(remaining, len(samples))
        faded = samples.copy()
        ramp = self._ramp_for(self._fade_in[self._frames_accepted:self._frames_accepted + n],
                              samples)
        faded[:n] = np.rint(faded[:n] * ramp)
        return faded

    def _flush_tail(self) -> None:
        """Write the held-back samples, ramped down so the file ends at exactly zero.

        The ramp is built to the tail's own length, so a recording too short to have
        filled it still ends on silence rather than on a step.

        Nothing is held back at all by a recorder that does not fade, so the early
        return is the whole of that case.  It also keeps the arithmetic below away
        from an empty stereo tail, whose shape a flat ramp cannot broadcast against.
        """
        if not len(self._tail):
            return
        ramp = self._ramp_for(self.fade_ramp(len(self._tail))[::-1], self._tail)
        self._emit(np.rint(self._tail * ramp).astype(self.SAMPLE_DTYPE))
        self._tail = self._no_frames

    @staticmethod
    def _ramp_for(ramp: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """A fade ramp shaped to multiply against `frames`.

        One value per frame, applied to every channel of it.  Mono frames are a flat
        array and the ramp already matches; anything with channels is a row per frame,
        which a flat ramp cannot broadcast against and would raise on instead.

        Nothing multi-channel fades today, since the one stereo recorder is raw IQ and
        deliberately does not.  This is here so that the fade length stays a knob a
        subclass can simply set, rather than one that works for mono and raises for
        everything else.
        """
        return ramp if frames.ndim == 1 else ramp[:, None]

    def _emit(self, samples: np.ndarray) -> None:
        """Write frames out in this recorder's own sample format.

        The dtype is the subclass's rather than a literal, and the two .wav
        conventions line up with numpy's: 8-bit is unsigned, like a receiver's raw
        bytes, and everything wider is signed little-endian, like int16 audio.  So
        the cast is a no-op whenever the samples already arrived in the right form.
        """
        self._writer.writeframes(samples.astype(self.SAMPLE_DTYPE, copy=False).tobytes())

    def _write_metadata(self, ended: str) -> None:
        """Tag the finished file with what it is and how to read it back.

        Never allowed to fail the recording.  The audio is closed and safe by this
        point, and an untagged recording is still a perfectly good one, so losing it
        over a metadata write would be a poor trade.
        """
        settings = wavmeta.format_settings(self._metadata_settings(ended))
        started = self._started_at.replace(microsecond=0).isoformat()
        try:
            wavmeta.append_metadata(
                self._path,
                {
                    'INAM': f'{self._callsign} {self.KIND} {started}',
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

    @abstractmethod
    def _metadata_settings(self, ended: str) -> dict[str, Any]:
        """The key=value settings this format's ICMT tag carries.

        Everything a later reading cannot recover from the file itself, and nothing it
        can: the sample rate is already in the format header.  What belongs here
        differs entirely by format, which is why the subclass answers it.
        """


class AudioEventRecorder(AbstractEventRecorder):
    """The monitor's own audio, as 16-bit mono PCM at the audio sample rate.

    The format the rest of this program already reads: buzz.playback replays one of
    these through the whole pipeline, and buzz.render turns one into video.
    """

    CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2      # 16-bit PCM, matching the int16 capture format end to end
    SAMPLE_DTYPE = '<i2'
    KIND = 'powerline QRM event'

    def __init__(self, pipeline: RingBufferPipeline, config: BuzzConfig,
                 charged_wait_seconds: float) -> None:
        recording = config.recording
        super().__init__(
            pipeline,
            sample_rate=config.audio.sample_rate,
            directory=recording.directory_path(config.station),
            callsign=config.station.callsign,
            max_seconds=recording.max_seconds,
            charged_wait_seconds=charged_wait_seconds,
        )
        # Kept for the file's metadata.  The pulse rate and the dB calibration are
        # the two settings a replay cannot recover from the audio itself, and getting
        # either wrong changes what the replay measures - see wavmeta.
        self._pulse_rate = config.audio.pulse_rate
        self._rf_conversion_db = config.level_offset_db

    def _metadata_settings(self, ended: str) -> dict[str, Any]:
        return {
            'sample_rate': self._sample_rate,
            'pulse_rate': self._pulse_rate,
            'audio_rf_conversion_db': self._rf_conversion_db,
            'lead_in_seconds': round(self._lead_in / self._sample_rate, 2),
            'lead_in_max_seconds': round(self._max_lead_in_samples() / self._sample_rate, 2),
            'ended': ended,
        }


class IqEventRecorder(AbstractEventRecorder):
    """The receiver's raw IQ, exactly as it came off the device.

    Stereo, I on the left channel and Q on the right, at the receiver's own sample
    rate.  Nothing is scaled, levelled or faded: the file is the bytes the converter
    produced, so that whoever opens it is looking at the measurement rather than at
    this program's opinion of it.  Nothing here reads one back either - the point is
    handing the raw data to somebody with their own tools.

    No fade, where the audio recorder has one.  A fade exists to stop a click when a
    person plays the file, and this is not for playing.  Altering the samples at each
    end would be the one edit this recorder makes to data it exists to pass through.

    The width comes from the buffer rather than from a constant here, because it is a
    fact about the device: an RTL-SDR delivers unsigned bytes, and a wider converter
    would deliver something else.  The two conventions happen to line up, which is
    what makes the copy a copy - 8-bit .wav is unsigned like the device's bytes, and
    every wider .wav is signed like a wider converter's samples.
    """

    CHANNELS = 2
    KIND = 'raw IQ capture'
    FILENAME_SUFFIX = '-iq'
    # No fade.  See the class docstring; fade_ramp(0) is empty, so the machinery in
    # AbstractEventRecorder holds nothing back and every write goes straight out.
    FADE_SECONDS = 0.0

    def __init__(self, pipeline: RingBufferPipeline, config: BuzzConfig,
                 charged_wait_seconds: float) -> None:
        recording = config.recording
        settings = config.rtlsdr
        # Before the base constructor, which shapes its first empty frame buffer from
        # these.  Per instance rather than per class, unlike every other recorder's,
        # because the answer belongs to the hardware that filled the buffer.
        self.SAMPLE_DTYPE = pipeline.dtype
        self.SAMPLE_WIDTH_BYTES = pipeline.dtype.itemsize
        super().__init__(
            pipeline,
            sample_rate=settings.iq_sample_rate,
            directory=recording.directory_path(config.station),
            callsign=config.station.callsign,
            max_seconds=recording.max_seconds,
            charged_wait_seconds=charged_wait_seconds,
        )
        # What a reader needs to make sense of the samples, none of which the file
        # itself carries.  The center frequency matters most: it is where DC sits in
        # this capture, and without it the numbers describe an unknown piece of
        # spectrum.
        self._pulse_rate = config.audio.pulse_rate
        self._listening_hz = settings.frequency_hz
        self._tuned_hz = settings.frequency_hz + settings.tuning_offset_hz
        self._tuning_offset_hz = settings.tuning_offset_hz
        self._gain_db = settings.gain_db
        self._rf_conversion_db = config.level_offset_db

    def _metadata_settings(self, ended: str) -> dict[str, Any]:
        return {
            'sample_rate': self._sample_rate,
            'pulse_rate': self._pulse_rate,
            # Where the hardware actually sat, which is DC in this file.  The
            # frequency the monitor measures is the other one, offset from it on
            # purpose so that the receiver's own spur misses the measured band.
            'center_frequency_hz': self._tuned_hz,
            'listening_frequency_hz': self._listening_hz,
            'tuning_offset_hz': self._tuning_offset_hz,
            'gain_db': self._gain_db,
            'rf_conversion_db': self._rf_conversion_db,
            'lead_in_seconds': round(self._lead_in / self._sample_rate, 2),
            'lead_in_max_seconds': round(self._max_lead_in_samples() / self._sample_rate, 2),
            'ended': ended,
        }


class RecordingTrigger:
    """Decides when an event is worth recording, and drives the recorders that write it.

    Armed and disarmed at runtime (toolbar button, or --enable-recording at startup).
    While armed it starts a file on the first LOCKED tick and closes it once the
    signal has been gone for stop_after_seconds, then counts the event against the
    remaining budget and disarms itself when that budget runs out.

    Everything here is a decision.  No audio is read and no file is touched: the
    counters this works in come from the pipeline's own total_samples, and the
    recorders do the reading and the writing when they are told to.  That is what lets
    a second format arrive without a second copy of the gating below, and what keeps
    one event counting once against the budget however many files it produced.
    """

    # Matches the analyzer's LOCKED tick cadence: polling faster cannot see a lock
    # sooner, and the audio itself is never sampled at this rate - each pass writes
    # every sample captured since the previous one, whenever it happens to run.
    POLL_INTERVAL = 0.2

    # How many published results min_lock_snr is judged over.  Five at the analyzer's
    # publishing cadence is about a second: long enough that one loud tick cannot let
    # a weak event through, short enough to follow a signal that is still building.
    SNR_WINDOW = 5

    def __init__(self, pipeline: RingBufferPipeline, analyzer: ContinuousAnalyzer,
                 config: BuzzConfig) -> None:
        self._pipeline = pipeline
        recording = config.recording
        self._zone = ZoneInfo(config.station.timezone)
        self._sample_rate = config.audio.sample_rate
        self._max_events  = recording.max_events
        # Both limits in samples, on the audio clock, and both clamped: a nonsensical
        # setting should degrade to the nearest sensible behavior rather than into
        # something surprising.  A negative cap would drive the write position back
        # behind itself and re-read audio already written, so anything at or below
        # zero means uncapped.  A zero timeout would end every recording on the tick
        # after it began, signal or no signal, so it floors at a single sample -
        # which reads as "stop as soon as the lock is lost".
        self._max_samples     = max(0, round(recording.max_seconds * self._sample_rate))
        self._timeout_samples = max(1, round(recording.stop_after_seconds * self._sample_rate))
        self._min_lock_samples = self._qualifying_lock_samples(
            recording.min_lock_seconds, self._sample_rate,
            pipeline.capacity_samples, self._max_samples)

        # Built here rather than passed in, so that everything wiring the monitor
        # together keeps building one object.  Which formats apply is a property of
        # the configuration rather than of the caller.
        # Guarded because a misconfigured rate of zero reaches here before anything
        # has had the chance to refuse it: can_record() is what reports that, and it
        # cannot run until the recorder below exists.  The wait is zero samples at a
        # zero rate anyway, so there is nothing to carry across.
        charged_wait = (self._min_lock_samples / self._sample_rate
                        if self._sample_rate > 0 else 0.0)
        self._recorders: list[AbstractEventRecorder] = [
            AudioEventRecorder(pipeline, config, charged_wait_seconds=charged_wait),
        ]
        # A second file of the same event, when the source kept the raw IQ to write it
        # from.  Both are driven by the decisions below, so they start and stop
        # together and one event spends one from the budget.
        if pipeline.iq_buffer is not None:
            self._recorders.append(
                IqEventRecorder(pipeline.iq_buffer, config,
                                charged_wait_seconds=charged_wait))

        # The analyzer, kept for its published levels rather than its state.  The
        # state arrives by push because lock is an edge, but SNR is a level, and a
        # level is exactly the thing it is right to read when you happen to want it.
        self._analyzer = analyzer
        self._min_lock_snr = recording.min_lock_snr
        # Recent SNR readings taken while a lock is being judged.  This is a rolling
        # window rather than an average over the whole wait: an event that starts
        # quiet and builds should be recorded from the moment it is loud enough, and
        # an average dragged down by how it began would hold it off long after that.
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
        # The short-circuit matters.  Recording that is off reaches for nothing at
        # all, so a monitor run without it leaves no stray directory behind and
        # cannot complain about a path it was never going to use.  The check happens
        # instead when the operator presses Record.
        self._armed = recording.enabled and self._can_record()
        self._events_remaining = self._initial_budget()
        # Monotonic deadline for the next budget reset, or None when no cycle is
        # running.  Set whenever the budget is filled, which is what makes the cycle
        # a fixed one: ten events per day means ten per day, not ten per day plus
        # however long the tenth event took to arrive.
        self._next_reset = self._reset_deadline() if self._armed else None

        # Lock state is pushed from the analyzer rather than read back from it (see
        # ContinuousAnalyzer.add_state_listener), seeded here with the state the
        # analyzer is in at wiring time.  _lock_acquired stays sticky until the next
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

        # Pipeline position when the current unbroken lock began, or None when there
        # is no lock to be timing.  See _lock_has_held.
        # Which gate last held a recording back, for the opening log line to explain
        # itself with; None once nothing is holding it.  See _not_yet.
        self._held_back_by: str | None = None
        self._locked_since: int | None = None
        self._position = 0              # pipeline position as of the current tick
        self._event_start = 0           # position where recording began (max_seconds origin)
        self._last_lock = 0             # position as of the most recent LOCKED tick
        # Set when a recording ends, cleared by the first tick that sees no lock.  A
        # capped recording ends with the signal still present, and one event should
        # produce one file: without this the next tick would immediately open another
        # and a long event would come back as a pile of max_seconds fragments.
        self._await_relock = False

        # One lock for the recording state above, and for the budget and the re-arm
        # cycle: tick() runs on this thread while arm(), disarm() and status() are
        # called from the Qt thread, and a toggle arriving mid-write must not tear the
        # file's bookkeeping.  The two lock flags are the exception, on _state_lock,
        # because the analyzer thread touches those and must not end up behind a disk
        # write.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name='recorder')

    # ------------------------------------------------------------------ public

    @property
    def _recording(self) -> bool:
        """Whether any recorder has a file open."""
        return any(recorder.is_recording for recorder in self._recorders)

    def _can_record(self) -> bool:
        """Whether every recorder is ready to write."""
        return all(recorder.can_record() for recorder in self._recorders)

    def _begin(self) -> None:
        """Open a file on every recorder for a newly locked event.

        The age of the lock goes out in seconds rather than as a position, so that a
        recorder reading a different pipeline at a different rate can place its own
        cue marker.  See docs-notebook/iq-recording-design.md.
        """
        now = datetime.now(self._zone)
        lock_age_seconds = (None if self._locked_since is None
                            else (self._position - self._locked_since) / self._sample_rate)
        # The cap runs from the moment recording begins, less whatever was spent
        # waiting out min_lock_seconds; each recorder applies the same figure to its
        # own clock.  See AbstractEventRecorder.__init__.
        self._event_start = self._position - self._min_lock_samples
        self._last_lock = self._position
        for recorder in self._recorders:
            if not recorder.begin(now, lock_age_seconds, self._held_back_by):
                # What a single recorder already did on a failed open: a file that
                # cannot be created is a configuration problem rather than a passing
                # one, so recording stops instead of dropping a stray file per poll.
                #
                # This avoids _finish_all deliberately.  Nothing was recorded, so the
                # event must not be counted against the budget - the operator would
                # otherwise pay for a file they never got.
                for opened in self._recorders:
                    if opened.is_recording:
                        opened.finish('failed', self._end_description('failed'))
                self._armed = False
                break
        self._held_back_by = None

    def _finish_all(self, ended: str) -> None:
        """Close every open file, and count the event once against the budget."""
        description = self._end_description(ended)
        for recorder in self._recorders:
            if recorder.is_recording:
                recorder.finish(ended, description)
        self._await_relock = True
        self._spend_event(ended)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and close any recording in progress.

        Finalizing matters: a .wav's header carries its length, and a file whose
        writer never closed reports zero frames no matter how much audio is in it.
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._recording:
                self._finish_all('shutdown')

    def arm(self) -> None:
        """Enable recording and refill the event budget, restarting the reset cycle.

        This re-checks the directory, so it is also how an operator retries after
        fixing a bad path: arming is refused, loudly, while there is nowhere to
        record to.
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

        This cancels the reset cycle as well.  Switching recording off by hand has
        to mean off: coming back to a monitor that re-armed itself overnight, because
        it was turned off during a cycle rather than between two, would be a nasty
        surprise.
        """
        with self._lock:
            self._armed = False
            self._next_reset = None
            if self._recording:
                self._finish_all('operator')
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
                recording=self._recording,
                events_remaining=self._events_remaining,
                # Explicitly zero when idle: the positions below are left where the
                # last recording ended, so the subtraction would otherwise keep
                # reporting that recording's length long after it closed.
                elapsed_seconds=(0.0 if not self._recording else
                                 (self._position - self._event_start) / self._sample_rate),
                filename=self._recorders[0].filename,
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

    @staticmethod
    def _qualifying_lock_samples(seconds: float, sample_rate: int, capacity: int,
                                 max_samples: int) -> int:
        """How long a lock must hold before it is worth a file, in samples.

        Two ceilings apply, and the wait is clamped to whichever is lower.

        The first is the ring buffer, because waiting is not free.  The buffer is a
        sliding window, so every second spent deciding is a second of run-up that
        has fallen off the far end by the time the file opens.  Wait longer than the
        buffer and the recording would begin *after* the lock, missing the onset of
        the very event it exists to capture.

        The second is max_seconds, because the wait is counted against it.  Those
        seconds are part of the event, so asking for three of them and ten of
        recording buys ten and not thirteen.  A wait longer than the whole allowance
        is a contradiction: left alone it would reach back past the audio the file
        opens with and throw away its newest seconds to stay inside a limit already
        spent.  Clamped, the two settings meet at the sensible end of it - the
        recording is exactly the allowance, all of it from what the buffer was
        already holding.
        """
        wanted = max(0, round(seconds * sample_rate))
        if wanted > capacity:
            logger.warning(
                'min_lock_seconds of %g is longer than the %.1f s of audio the buffer '
                'holds.  Using %.1f s instead.  A longer wait would start the '
                'recording after the event it records.',
                seconds, capacity / sample_rate, capacity / sample_rate)
            wanted = capacity
        if max_samples and wanted > max_samples:
            logger.warning(
                'min_lock_seconds of %g is longer than max_seconds of %g.  Using %g '
                'instead.  The wait counts against the recording length, so it cannot '
                'be longer than the whole of it.',
                seconds, max_samples / sample_rate,
                max_samples / sample_rate)
            wanted = max_samples
        return wanted

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

    def _end_description(self, ended: str) -> str:
        """The log's version of an `ended` token, with the limit that produced it."""
        if ended == 'timeout':
            return f'no lock for {self._timeout_samples / self._sample_rate:g} s'
        if ended == 'capped':
            return f'reached the {self._max_samples / self._sample_rate:g} s limit'
        return _END_DESCRIPTIONS.get(ended, ended)

    def _spend_event(self, ended: str) -> None:
        """Count one finished event against the budget, and disarm if it is spent.

        Once per event rather than once per file.  That is what one trigger owning the
        budget buys: two recorders each counting their own would drift apart the first
        time one of them failed to open a file.
        """
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

        Deliberately trivial: it records what happened and returns.  Anything more,
        such as opening a file or writing audio, would run analysis-critical work
        behind disk I/O.  The recorder's own thread picks this up on its next tick.

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

        This is recorded rather than inferred afterward.  By the time a recording
        starts, both gates are satisfied and nothing in its state says which of them
        the waiting was for - a question that has already cost three wrong guesses
        and a measurement rig to answer from the outside.
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
        before it reports one, so a loss that reaches here is a true one.  Without
        this, a stream of blips would accumulate: no tick would ever observe the
        gaps, because the next blip has set the flag again before it runs.
        """
        held = self._pipeline.total_samples - self._locked_since
        return held >= self._min_lock_samples

    def _sample_snr(self) -> None:
        """Note the analyzer's latest SNR, while a lock is waiting to be judged.

        Only locked results carry a level.  An unlocked one reports zero by
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

        An event below the bar is watched, not abandoned.  Powerline arcs commonly
        start quiet and build, and one that crosses ten seconds in is still worth
        recording from the moment it does, at the cost of that much lead-in.  That
        is the trade this setting makes, and the reason to keep it modest.

        The window must be full before it decides anything.  Judged on one or two
        readings, a single loud tick averages away a weak event and lets it through.
        The first readings after a cold lock are also the least worth trusting
        anyway, since the analyzer's drift estimate has not converged and levels
        read several dB low until it does.  Waiting for a full window costs about a
        second and buys a measurement worth thresholding on.
        """
        if not self._min_lock_snr:
            return True
        if len(self._recent_snr) < self.SNR_WINDOW:
            return False
        return sum(self._recent_snr) / self.SNR_WINDOW >= self._min_lock_snr

    def _consume_lock_state(self) -> tuple[bool, bool]:
        """(locked since the previous tick, lost since the previous tick).

        Reading the live flag and clearing the sticky ones has to be one operation.
        Done separately, a lock and a loss arriving in the gap between them would
        clear the flag that recorded the lock while leaving the level saying there is
        none, and a brief lock vanishing between two polls is precisely the failure
        the push exists to prevent.

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
        # One snapshot of the clock per tick, so every decision below is taken
        # against the same instant.  A counter, never a read: the recorders do the
        # reading when they are told to.
        self._position = self._pipeline.total_samples
        locked, lost = self._consume_lock_state()
        if not self._recording:
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

        for recorder in self._recorders:
            recorder.capture()
        if locked:
            self._last_lock = self._position
        if self._max_samples and self._position - self._event_start >= self._max_samples:
            self._finish_all('capped')
        elif self._position - self._last_lock >= self._timeout_samples:
            self._finish_all('timeout')

        if not self._recording:
            # A recording that just ended with the signal still present is a capped
            # one; the event is not over, so hold off until it truly is.  One that
            # ended on the timeout is already over and the next lock is a new event.
            self._await_relock = locked

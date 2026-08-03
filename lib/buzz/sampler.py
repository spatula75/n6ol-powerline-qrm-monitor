"""
Audio input for the powerline QRM monitor.

Pure audio I/O - the pulse-train analysis lives in buzz.dsp and buzz.analyzer.
RingBufferPipeline holds the buffering all audio sources share; AudioPipeline
adds a PortAudio callback that fills it live, and buzz.playback adds a
file-backed source that replays a recorded .wav through the same interface.
Multiple consumers (continuous analyzer, waterfall display, event recorder) read
overlapping snapshots without removing data.  AudioSampler resolves the
configured device by name and owns the live pipeline.  LevelStream provides
real-time broadband level readings for the calibration/level-meter tool.
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import Self

import numpy as np
import sounddevice as sd

from buzz.config import BuzzConfig
from buzz.dsp import SILENCE_DBFS, amplitude_to_dbm

logger = logging.getLogger(__name__)

# Ring buffer capacity, in seconds of audio rather than in samples.
#
# 9.6 s is exactly what 300 chunks of 512 came to at 16 kHz, so nothing changes at the
# rate this program records at.  Expressing it as a duration is what makes it mean the
# same thing at any other rate: a fixed sample count held 9.6 s at 16 kHz but only 3.5 s
# at 44.1 kHz, silently shrinking both the analyzer's history and the lead-in an event
# recording opens with, in proportion to a setting nobody would connect to either.
#
# Ample headroom for the continuous analyzer's 1 s aligned windows and the waterfall's
# per-frame reads, and it doubles as the lead-in an event recording opens with.
_BUFFER_SECONDS = 9.6
# What the buffer costs at the top of the supported range: 48 kHz needs 900 chunks of
# 512 int16 samples, which is under a megabyte.  Sizing by duration is affordable
# precisely because the audio is mono and narrow-band.
_DEFAULT_SAMPLE_RATE = 16000

# How often a continuing run of dropped audio is summarized.  A minute is chosen to
# match the collector's cycle, so a log reporting dropouts lines up with the CSV rows
# they affected.  The first one is always reported immediately; see DropoutReporter.
_DROPOUT_REPORT_SECONDS = 60.0


def buffer_chunks(sample_rate: int, chunk_size: int) -> int:
    """How many chunks hold _BUFFER_SECONDS of audio at `sample_rate`.

    Rounded up, so the buffer is never shorter than the duration it promises.
    """
    return ceil(_BUFFER_SECONDS * sample_rate / chunk_size)


@dataclass(frozen=True)
class AudioSpan:
    """A contiguous run of samples, tagged with its absolute position in the stream.

    `start` and `end` are counted in samples since the stream began, on the same
    monotonic clock as total_samples, so a sequential reader can tell the difference
    between "nothing new yet" (start == end) and "I fell behind and the buffer
    discarded audio I never read" (start > the position it asked for).
    """

    samples: np.ndarray
    start: int
    end: int


class RingBufferPipeline:
    """Ring buffer of fixed-size chunks, shared by every audio source.

    Whatever produces the audio appends each chunk with _append(), which notifies a
    Condition so consumers can block-wait for new data.  Multiple independent
    consumers (analysis thread, waterfall display, event recorder) read from the
    buffer via get_snapshot() or read_from() without removing data; the deque's
    maxlen acts as a sliding window that discards audio older than ~10 seconds.

    CHUNK_SIZE is a power of two so FFT-based consumers get clean window boundaries
    without padding or resampling.
    """

    CHUNK_SIZE = 512  # samples per callback block; 32 ms at 16 kHz

    def __init__(self, sample_rate: int = _DEFAULT_SAMPLE_RATE) -> None:
        # The rate is taken here only to size the buffer: this class never looks at the
        # audio, and a subclass that knows the real rate passes it up.  Sizing in
        # seconds is what keeps the analyzer's history and a recording's lead-in
        # meaning the same thing whatever the audio arrives at.
        self._chunks = buffer_chunks(sample_rate, self.CHUNK_SIZE)
        self._buffer: deque[np.ndarray] = deque(maxlen=self._chunks)
        self._condition = threading.Condition()
        # Monotonic count of samples ever captured; keeps growing after the deque
        # starts discarding old chunks.  Global sample positions derived from this
        # are what make phase-aligned snapshots possible.
        self._total_samples = 0

    def _append(self, chunk: np.ndarray) -> None:
        """Add one chunk of captured audio and wake anything waiting on it."""
        with self._condition:
            self._buffer.append(chunk)
            self._total_samples += len(chunk)
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard buffered audio, as if capture had only just started.

        The sample counter keeps going.  It is the audio clock the analyzer measures
        drift against and the origin every phase is expressed in, so winding it back
        would not read as "no audio yet" but as time running backwards.
        """
        with self._condition:
            self._buffer.clear()

    def get_snapshot(self, n_samples: int, align: int = 1) -> np.ndarray:
        """Return the most recent n_samples of audio, optionally phase-aligned.

        With align > 1 the window ends at the greatest multiple of align samples
        since the stream started, rather than at the live tail.  Every aligned
        window then has the same start position modulo align, at the cost of being
        up to align-1 samples staler than the newest audio.  The analyzer depends
        on this: it compares pulse phases across snapshots, and an unaligned
        window's phase origin moves with the tail (512-sample chunks are not a
        whole number of pulse periods), silently invalidating stored phases.

        Caller should ensure wait_for_data(n_samples + align) has returned True.
        """
        n_chunks = ceil((n_samples + align - 1) / self.CHUNK_SIZE)
        with self._condition:
            chunks = list(self._buffer)[-n_chunks:]
            total = self._total_samples
        if not chunks:
            return np.zeros(n_samples, dtype=np.int16)
        arr = np.concatenate(chunks)
        end = len(arr) - total % align
        if end <= 0:
            return np.zeros(n_samples, dtype=np.int16)
        return arr[max(0, end - n_samples):end]

    def read_from(self, position: int) -> AudioSpan:
        """Return every buffered sample from absolute `position` to the live tail.

        This is the sequential counterpart to get_snapshot(): where a display wants
        the most recent N samples and does not care what it skipped, a recorder needs
        each sample exactly once, in order, with nothing dropped or repeated.  Passing
        back the previous span's `end` on each call gives that.

        A reader slower than the buffer's ~10 second window gets what survives rather
        than an error, with the loss visible as span.start > the requested position.
        Passing 0 therefore reads everything still buffered, which is how a recording
        picks up its lead-in: the audio leading to the moment of lock is already here.
        """
        with self._condition:
            chunks = list(self._buffer)
            end = self._total_samples
        buffered = sum(len(c) for c in chunks)
        oldest = end - buffered
        start = max(position, oldest)
        if start >= end:
            return AudioSpan(np.empty(0, dtype=np.int16), end, end)

        # Only the chunks the span actually touches are joined.  A caller reading
        # sequentially asks for the fraction of a second that arrived since its last
        # call, so joining the whole ~10 second buffer and then slicing would copy
        # several hundred kilobytes to keep a few, five times a second, for as long
        # as a recording lasts.  Taken from the newest end, which is the one the span
        # always reaches, and without assuming every chunk is the same length.
        wanted = end - start
        kept, taken = [], 0
        for chunk in reversed(chunks):
            kept.append(chunk)
            taken += len(chunk)
            if taken >= wanted:
                break
        # taken overshoots wanted by however far into its oldest chunk `start` falls:
        # the run of chunks begins on a chunk boundary and a position rarely does.
        return AudioSpan(np.concatenate(kept[::-1])[taken - wanted:], start, end)

    @property
    def capacity_samples(self) -> int:
        """The most audio the buffer ever holds, and so the longest lead-in possible.

        Anything that waits before starting a recording is spending this: the window
        slides, so a second spent waiting is a second of run-up that has fallen off
        the far end by the time the file opens.
        """
        return self._chunks * self.CHUNK_SIZE

    @property
    def total_samples(self) -> int:
        """Monotonic count of samples captured since the stream started.

        Lets consumers detect a stalled stream (count stops advancing) without
        comparing audio content.
        """
        with self._condition:
            return self._total_samples

    def wait_for_data(self, n_samples: int, timeout: float | None = None) -> bool:
        """Block until at least n_samples worth of chunks are in the buffer.

        Returns True if sufficient data is available, False on timeout.
        On first startup this blocks while the buffer fills; thereafter it
        returns immediately.
        """
        n_chunks = ceil(n_samples / self.CHUNK_SIZE)
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self._buffer) >= n_chunks,
                timeout=timeout,
            )

    def start(self) -> None:
        """Begin producing audio, for a source that does not start on construction.

        Live capture has no use for this - its device is running by the time the
        constructor returns - but a file-backed replay must not begin before the
        caller has somewhere to show it (see FilePlaybackPipeline.start), and a
        consumer holding a pipeline should not have to know which kind it has.
        """

    def close(self) -> None:
        """Stop producing audio.  Subclasses shut down whatever fills the buffer."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DropoutReporter:
    """Rate-limits the warning for audio the input device dropped.

    Separate from AudioPipeline so it can be tested without opening a device, and
    because the decision of when to speak is worth stating on its own.

    The first dropout is reported at once, since one on a machine that has never had
    one is worth seeing immediately.  After that a continuing run costs a single line
    per interval carrying the count, rather than one per callback: this is called from
    the PortAudio callback, on PortAudio's own thread, where a warning per block would
    both flood the log and spend time the audio path does not have -- and where the
    logging could itself provoke the next dropout.

    A run that stops part-way through an interval takes its suppressed count with it:
    five dropouts over two seconds and then quiet are reported as one.  Only the next
    dropout can print anything, and there is no timer here to flush the remainder on.
    That is the right way round for a warning whose job is to say a fault is ongoing.
    """

    def __init__(self, report_interval_seconds: float = _DROPOUT_REPORT_SECONDS) -> None:
        self._interval = report_interval_seconds
        self._pending = 0
        self._reported_at: float | None = None

    def record(self, now: float) -> int | None:
        """Count one dropout; return how many to report, or None to stay quiet.

        `now` is passed in rather than read here so a test can drive the clock.
        """
        self._pending += 1
        if self._reported_at is not None and now - self._reported_at < self._interval:
            return None
        self._reported_at = now
        count, self._pending = self._pending, 0
        return count


class AudioPipeline(RingBufferPipeline):
    """Live audio input: a PortAudio callback filling the shared ring buffer."""

    def __init__(self, config: BuzzConfig, device_index: int) -> None:
        super().__init__(config.audio.sample_rate)
        self._dropouts = DropoutReporter()

        def _callback(indata: np.ndarray, frames: int,
                      time: object, status: sd.CallbackFlags) -> None:
            if status:
                # PortAudio reports every input fault through `status`, and they are
                # all handled the same way because the recovery is the same.  On a
                # capture stream it is in practice an overflow: the device captured
                # faster than this callback collected, and the driver discarded the
                # difference.
                #
                # Note what that does and does not do, because the obvious guess is
                # wrong.  It leaves no gap and no silence -- this callback still gets a
                # full block of current audio -- so nothing reads low.  What arrives is
                # a *splice* of two runs that were never adjacent, and since
                # _total_samples advances only by what is appended, the audio clock
                # under-counts the time that really passed.  The pulse train did move
                # through the discarded samples, so the next phase measurement jumps,
                # and ContinuousAnalyzer's least-squares fit reads that step as drift.
                # The grid frequency is therefore the reading to distrust, not the
                # levels.
                #
                # Logged rather than raised.  This monitor runs unattended all day, and
                # losing every later measurement to protect one polluted minute is the
                # wrong trade; every other failure here degrades and carries on for the
                # same reason.  The analyzer recovers by itself: a phase that stops
                # making sense drops it to SEARCHING and it re-acquires.
                dropped = self._dropouts.record(monotonic())
                if dropped is not None:
                    logger.warning(
                        'The audio input reported %d error(s) (%s), almost always an '
                        'overflow: the device captured faster than this program '
                        'collected it, so the driver discarded the difference. The '
                        'splice that leaves makes the pulse phase jump, so the grid '
                        'frequency is the reading to distrust until the analyzer '
                        're-acquires, which it does on its own; the levels are '
                        'unaffected. Usually transient load - if it repeats, close '
                        'whatever else on this machine is using audio.',
                        dropped, status)
            self._append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=config.audio.sample_rate,
            dtype='int16',
            blocksize=self.CHUNK_SIZE,
            callback=_callback,
        )
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class AudioSampler:
    def __init__(self, config: BuzzConfig) -> None:
        """Resolve the PortAudio device to record from and start the pipeline.

        Always resolves the device by name, not by the stored index.  PortAudio
        device indices are reassigned by Windows on every reboot; the name is stable.
        """
        self._config = config
        device = sd.query_devices(config.audio.input_device_name, 'input')
        self._device_index = device['index']
        self._pipeline = AudioPipeline(config, self._device_index)

    @property
    def pipeline(self) -> AudioPipeline:
        return self._pipeline

    def close(self) -> None:
        self._pipeline.close()

    def level_stream(self, blocksize: int = 320) -> 'LevelStream':
        """Open a persistent input stream for real-time level monitoring.

        Returns a context manager whose .read() method blocks until one block
        of audio is available and returns the broadband signal level in dBm.
        Default blocksize of 320 samples = 20 ms at 16 kHz (one Windows CPU quantum).
        """
        return LevelStream(self._config, self._device_index, blocksize)


class LevelStream:
    """Persistent input stream for real-time level monitoring.

    Uses a PortAudio callback rather than blocking read() because DirectSound on
    Windows does not support PortAudio's blocking I/O reliably.  The callback fires
    whenever the hardware delivers a new buffer; read() blocks on a threading.Event
    until that happens, then returns immediately with the latest dBm level.

    The sound card's DC offset is removed before rectification, using the same
    EMA-smoothed median estimate the analyzer applies (see ContinuousAnalyzer._capture
    for why the median rather than the mean).  It matters more here than anywhere
    else: this is the reading the operator calibrates audio_rf_conversion_db against,
    so an uncorrected offset would be baked into every measurement the monitor ever
    reports.  Smoothing is what makes it viable at this block size - a 320-sample
    block is far too short to estimate an offset from on its own.

    Use as a context manager:
        with sampler.level_stream() as stream:
            dbm = stream.read()
    """

    # ~10 s time constant at the 50 Hz callback rate of the default 320-sample block.
    DC_EMA_ALPHA = 0.002

    def __init__(self, config: BuzzConfig, device_index: int, blocksize: int) -> None:
        self._event = threading.Event()
        self._latest_dbm: float = SILENCE_DBFS
        self._offset_db = config.station.audio_rf_conversion_db
        self._dc: float | None = None   # None until the first block seeds it

        def _callback(indata: np.ndarray, frames: int,
                      time: object, status: sd.CallbackFlags) -> None:
            block = indata[:, 0].astype(np.float32)
            block_dc = float(np.median(block))
            self._dc = (block_dc if self._dc is None
                        else self._dc + self.DC_EMA_ALPHA * (block_dc - self._dc))
            amplitude = float(np.mean(np.abs(block - self._dc)))
            self._latest_dbm = amplitude_to_dbm(amplitude, self._offset_db)
            self._event.set()

        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=config.audio.sample_rate,
            dtype='int16',
            blocksize=blocksize,
            latency='low',
            callback=_callback,
        )
        self._stream.start()

    def read(self) -> float:
        """Block until the next hardware callback fires and return the level in dBm."""
        self._event.wait()
        self._event.clear()
        return self._latest_dbm

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> 'LevelStream':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

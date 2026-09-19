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
# 9.6 s is exactly what 300 chunks of 512 came to at 16 kHz, so nothing changes at
# the rate this program records at.  Expressing it as a duration is what makes it
# mean the same thing at any other rate.  A fixed sample count held 9.6 s at 16 kHz
# but only 3.5 s at 44.1 kHz, silently shrinking both the analyzer's history and the
# lead-in an event recording opens with, in proportion to a setting nobody would
# connect to either.
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

    def __init__(self, sample_rate: int = _DEFAULT_SAMPLE_RATE, *,
                 chunk_size: int | None = None,
                 dtype: np.dtype | type = np.int16) -> None:
        # The rate is taken here only to size the buffer: this class never looks at the
        # audio, and a subclass that knows the real rate passes it up.  Sizing in
        # seconds is what keeps the analyzer's history and a recording's lead-in
        # meaning the same thing whatever the audio arrives at.
        #
        # chunk_size and dtype exist for a buffer holding something other than the
        # monitor's audio.  The raw IQ buffer holds whatever its receiver delivers and
        # is appended one whole device block at a time, because nothing reads it by
        # chunk count the way the analyzer reads audio.  Both default to what every
        # audio buffer has always used, so no existing caller changes.
        self._chunk_size = self.CHUNK_SIZE if chunk_size is None else chunk_size
        # Normalized, so that itemsize and str are available to anything asking
        # what this buffer holds.  A recorder sizes its .wav frames from it.
        self._dtype = np.dtype(dtype)
        self._chunks = buffer_chunks(sample_rate, self._chunk_size)
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

        The caller should ensure wait_for_data(n_samples + align) has returned True.
        """
        n_chunks = ceil((n_samples + align - 1) / self._chunk_size)
        with self._condition:
            chunks = list(self._buffer)[-n_chunks:]
            total = self._total_samples
        if not chunks:
            return np.zeros(n_samples, dtype=self._dtype)
        arr = np.concatenate(chunks)
        end = len(arr) - total % align
        if end <= 0:
            return np.zeros(n_samples, dtype=self._dtype)
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
            return AudioSpan(np.empty(0, dtype=self._dtype), end, end)

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
    def dtype(self) -> np.dtype:
        """What one sample of this buffer is, which decides a recording's frame size."""
        return self._dtype

    @property
    def effective_bits(self) -> int:
        """How many bits of the int16 samples this source really carries.

        Everything downstream sees int16 whatever the source, so a source of fewer
        bits arrives in coarser steps rather than in a smaller range.  The scope uses
        this to decide how far it will magnify before it would be drawing the source's
        own quantization noise at full height.  See scope.minimum_full_scale.

        Sixteen here, because a sound card delivers int16 and means all of it.  A
        receiver answers for its own converter.
        """
        return 16

    @property
    def scope_floor_steps(self) -> float:
        """How many of this source's own steps the scope refuses to magnify past.

        This answers one, which never clamps a signal that is really present and is
        therefore the answer to give where nobody has measured.  A sound card is that
        case for good: its dead level depends on the operator's AF gain, so no figure
        stated here would hold across two stations.  A receiver measured against its
        own dead channel answers larger.  See scope.minimum_full_scale.
        """
        return 1.0

    @property
    def iq_buffer(self) -> 'RingBufferPipeline | None':
        """The raw pre-conversion samples this source kept, or None when it keeps none.

        Only a receiver has anything to keep: a sound card delivers the audio itself,
        with nothing upstream of it to hold on to.  Answering None here rather than
        making the caller test the source keeps that question off every call site.
        """
        return None

    @property
    def capacity_samples(self) -> int:
        """The most audio the buffer ever holds, and so the longest lead-in possible.

        Anything that waits before starting a recording is spending this: the window
        slides, so a second spent waiting is a second of run-up that has fallen off
        the far end by the time the file opens.
        """
        return self._chunks * self._chunk_size

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
        n_chunks = ceil(n_samples / self._chunk_size)
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
                # PortAudio reports every input fault through `status`, and all of
                # them get the same handling, because the recovery is the same.  On a
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
                # This is logged rather than raised.  The monitor runs unattended all
                # day, and losing every later measurement to protect one polluted
                # minute is the wrong trade; every other failure here degrades and
                # carries on for the same reason.  The analyzer recovers on its own: a
                # phase that stops making sense drops it to SEARCHING and it
                # re-acquires.
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

        This always resolves the device by name, not by the stored index.
        PortAudio device indices are reassigned by Windows on every reboot; the name
        is stable.
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

    def level_stream(self, blocksize: int = 320) -> 'SoundCardLevelStream':
        """Open a persistent input stream for real-time level monitoring.

        Returns a context manager whose .read() method blocks until one block
        of audio is available and returns the broadband signal level in dBm.
        Default blocksize of 320 samples = 20 ms at 16 kHz (one Windows CPU quantum).
        """
        return SoundCardLevelStream(self._config, self._device_index, blocksize)


class LevelStream:
    """A live broadband level in dBm, from whatever produces audio.

    Subclasses supply blocks; everything that turns a block into a reading lives
    here.  The split matters more than it looks: the figure this produces is what an
    operator calibrates `audio_rf_conversion_db` against, so if a second source
    estimated DC differently, or rectified differently, or converted to dBm
    differently, they would calibrate against a number the monitor never reports and
    bake the difference into every level that station ever logs.  See
    `SoundCardLevelStream` and `buzz.sdr.SdrLevelStream`, and the test that puts the
    same samples through both.

    DC is removed before rectification, using the same EMA-smoothed median estimate
    the analyzer applies (see ContinuousAnalyzer._capture for why the median rather
    than the mean).  Smoothing is what makes it viable at this block size, since a
    320-sample block is far too short to estimate an offset from on its own.

    offset_db is public and safe to write from outside while the stream runs: the
    setup program's calibration dialog nudges it live as the operator adjusts the
    offset, and amplitude_to_dbm() applies it fresh on every block, so a write from
    the UI thread takes effect on the very next one with no restart needed.  A plain
    float assignment races the producing thread only in the Python-level sense of
    "which value it reads this block or the next" - never a torn read - which is
    precise enough for a live display somebody is watching, not a value anything
    logs or averages.

    Use as a context manager:
        with sampler.level_stream() as stream:
            dbm = stream.read()
    """

    # How long the DC estimate takes to follow a change, in seconds.  Ten is long
    # against the couple of seconds an operator spends turning a knob, which is the
    # point: the estimate should track the card's offset, not the signal.
    DC_TIME_CONSTANT_SECONDS = 10.0

    @staticmethod
    def dc_ema_alpha(sample_rate: int, blocksize: int, seconds: float) -> float:
        """The EMA weight that gives a `seconds`-long time constant at this block rate.

        Read it as one over the number of blocks the time constant spans, which is
        what the arithmetic comes to: 320 samples at 16 kHz is 500 blocks in ten
        seconds, so 0.002.  It is written as a single division rather than a
        division inside one.

        The weight applies once per block, so what it means in seconds depends on
        how long a block is.  It was written as a bare 0.002 with a comment saying
        "~10 s at the 50 Hz callback rate of the default 320-sample block", which
        was true of exactly one configuration.  A second source delivering blocks at
        another rate would have silently had a different time constant, on the
        estimate an operator calibrates against.

        Deriving it from both figures is the same move as `_PANEL_WIDTH_MULTIPLE`: a
        value that has to satisfy a constraint gets computed from the constraint,
        not hand-worked for the case that happens to be current.
        """
        return blocksize / (seconds * sample_rate)

    def __init__(self, offset_db: float, sample_rate: int, blocksize: int) -> None:
        self.offset_db = offset_db
        self._alpha = self.dc_ema_alpha(sample_rate, blocksize, self.DC_TIME_CONSTANT_SECONDS)
        self._event = threading.Event()
        self._latest_dbm: float = SILENCE_DBFS
        self._dc: float | None = None   # None until the first block seeds it

    def _on_block(self, block: np.ndarray) -> None:
        """Fold one block of mono audio into the reading, and wake any reader.

        Called on whichever thread the source uses, which is a PortAudio callback
        for a sound card and an ordinary worker for a receiver.  It does no I/O and
        allocates one block, so either is fine.
        """
        samples = block.astype(np.float32)
        block_dc = float(np.median(samples))
        self._dc = (block_dc if self._dc is None
                    else self._dc + self._alpha * (block_dc - self._dc))
        amplitude = float(np.mean(np.abs(samples - self._dc)))
        self._latest_dbm = amplitude_to_dbm(amplitude, self.offset_db)
        self._event.set()

    # A healthy source delivers a block every 20 to 64 ms, so a second is dozens of
    # blocks late.  Long enough that a loaded machine never trips it, short enough
    # that a caller can say the reading went stale instead of freezing on it.
    READ_TIMEOUT_SECONDS = 1.0

    def read(self, timeout: float | None = None) -> float | None:
        """Block until the next block arrives and return the level in dBm.

        None means nothing arrived in time, which is what an unplugged receiver or a
        sound card that went away looks like from here.  Returning it rather than
        waiting forever is the difference between a meter that says so and a dialog
        that freezes on a number that stopped being true.  `SdrSource.read`
        returns None on the same grounds.
        """
        if not self._event.wait(self.READ_TIMEOUT_SECONDS if timeout is None else timeout):
            return None
        self._event.clear()
        return self._latest_dbm

    def _stop(self) -> None:
        """Stop producing blocks.  Subclasses shut down whatever fills this."""

    def close(self) -> None:
        self._stop()
        # Wakes a read() blocked in the wait above, which the setup program's
        # calibration dialogs call via asyncio.to_thread() - see calibration.py.
        # Cancelling that asyncio Task does not stop the thread pool worker
        # actually running read(): whatever would normally set this event has just
        # been stopped, so without this, that thread blocks in Event.wait() forever,
        # on an event nothing will ever set again.  A thread stuck like that is not
        # merely leaked - CPython's own ThreadPoolExecutor registers an atexit hook
        # that joins every worker thread it ever created, so one stuck thread hangs
        # the entire process on exit, not just the dialog that orphaned it.
        # Confirmed live: closing the setup program after opening either calibration
        # dialog hung instead of exiting, until this line was added.
        self._event.set()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SoundCardLevelStream(LevelStream):
    """A level stream fed by a sound card.

    This uses a PortAudio callback rather than blocking read(), because DirectSound
    on Windows does not support PortAudio's blocking I/O reliably.  The callback
    fires whenever the hardware delivers a new buffer.
    """

    def __init__(self, config: BuzzConfig, device_index: int, blocksize: int) -> None:
        super().__init__(config.level_offset_db,
                         config.audio.sample_rate, blocksize)
        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=config.audio.sample_rate,
            dtype='int16',
            blocksize=blocksize,
            latency='low',
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata: np.ndarray, frames: int,
                  time: object, status: sd.CallbackFlags) -> None:
        self._on_block(indata[:, 0])

    def _stop(self) -> None:
        self._stream.stop()
        self._stream.close()

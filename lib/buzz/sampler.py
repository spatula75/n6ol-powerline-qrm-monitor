"""
Audio input for the powerline QRM monitor.

Pure audio I/O — the pulse-train analysis lives in buzz.dsp and buzz.analyzer.
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
from typing import Self

import numpy as np
import sounddevice as sd

from buzz.config import BuzzConfig
from buzz.dsp import SILENCE_DBFS, amplitude_to_dbm

logger = logging.getLogger(__name__)

# Ring buffer capacity in chunks.  300 × 512 samples at 16 kHz ≈ 9.6 seconds — ample
# headroom for the continuous analyzer's 1 s aligned windows and the waterfall's
# per-frame reads, and it doubles as the lead-in an event recording opens with.
_BUFFER_CHUNKS = 300


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

    def __init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=_BUFFER_CHUNKS)
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
        return _BUFFER_CHUNKS * self.CHUNK_SIZE

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

        Live capture has no use for this — its device is running by the time the
        constructor returns — but a file-backed replay must not begin before the
        caller has somewhere to show it (see FilePlaybackPipeline.start), and a
        consumer holding a pipeline should not have to know which kind it has.
        """

    def close(self) -> None:
        """Stop producing audio.  Subclasses shut down whatever fills the buffer."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AudioPipeline(RingBufferPipeline):
    """Live audio input: a PortAudio callback filling the shared ring buffer."""

    def __init__(self, config: BuzzConfig, device_index: int) -> None:
        super().__init__()

        def _callback(indata: np.ndarray, frames: int,
                      time: object, status: sd.CallbackFlags) -> None:
            if status:
                logger.warning('PortAudio callback status: %s', status)
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
    reports.  Smoothing is what makes it viable at this block size — a 320-sample
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

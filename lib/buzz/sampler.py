"""
Audio input for the powerline QRM monitor.

Pure audio I/O — the pulse-train analysis lives in buzz.dsp and buzz.analyzer.
AudioPipeline continuously fills a ring buffer from a PortAudio callback so
multiple consumers (continuous analyzer, waterfall display) can read
overlapping snapshots.  AudioSampler resolves the configured device by name
and owns the pipeline.  LevelStream provides real-time broadband level
readings for the calibration/level-meter tool.
"""

import logging
import threading
from collections import deque
from math import ceil

import numpy as np
import sounddevice as sd

from buzz.config import BuzzConfig
from buzz.dsp import SILENCE_DBFS, amplitude_to_dbm

logger = logging.getLogger(__name__)

# Ring buffer capacity in chunks.  300 × 512 samples at 16 kHz ≈ 160 seconds — enough
# headroom for the continuous analyzer's 1 s windows and any display consumer that
# wants a few seconds of history.
_BUFFER_CHUNKS = 300


class AudioPipeline:
    """Continuously-running audio input that fills a ring buffer of fixed-size chunks.

    A PortAudio callback appends each chunk to a deque and notifies a Condition so
    consumers can block-wait for new data.  Multiple independent consumers (analysis
    thread, waterfall display, etc.) read from the buffer via get_snapshot() without
    removing data; the deque's maxlen acts as a sliding window that discards audio
    older than ~160 seconds.

    CHUNK_SIZE is a power of two so FFT-based consumers get clean window boundaries
    without padding or resampling.
    """

    CHUNK_SIZE = 512  # samples per callback block; 32 ms at 16 kHz

    def __init__(self, config: BuzzConfig, device_index: int) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=_BUFFER_CHUNKS)
        self._condition = threading.Condition()

        def _callback(indata: np.ndarray, frames: int,
                      time: object, status: sd.CallbackFlags) -> None:
            if status:
                logger.warning('PortAudio callback status: %s', status)
            chunk = indata[:, 0].copy()
            with self._condition:
                self._buffer.append(chunk)
                self._condition.notify_all()

        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=config.audio.sample_rate,
            dtype='int16',
            blocksize=self.CHUNK_SIZE,
            callback=_callback,
        )
        self._stream.start()

    def get_snapshot(self, n_samples: int, offset: int = 0) -> np.ndarray:
        """Return n_samples ending offset samples before the current tail.

        offset=0 (default) returns the most recent n_samples.
        offset=n_samples returns the window immediately before that, and so on.
        Caller should ensure wait_for_data(n_samples + offset) has returned True.
        """
        n_chunks = ceil((n_samples + offset) / self.CHUNK_SIZE)
        with self._condition:
            chunks = list(self._buffer)[-n_chunks:]
        if not chunks:
            return np.zeros(n_samples, dtype=np.int16)
        arr = np.concatenate(chunks)
        end = len(arr) - offset
        return arr[max(0, end - n_samples):end]

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

    def latest_chunk(self) -> np.ndarray | None:
        """Return a copy of the most recently captured chunk, or None if the buffer is empty."""
        with self._condition:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> 'AudioPipeline':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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

    Use as a context manager:
        with sampler.level_stream() as stream:
            dbm = stream.read()
    """

    def __init__(self, config: BuzzConfig, device_index: int, blocksize: int) -> None:
        self._event = threading.Event()
        self._latest_dbm: float = SILENCE_DBFS
        self._offset_db = config.station.audio_rf_conversion_db

        def _callback(indata: np.ndarray, frames: int,
                      time: object, status: sd.CallbackFlags) -> None:
            amplitude = float(np.mean(np.abs(indata.astype(np.int32))))
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

"""
File-backed audio source: replays a recorded .wav through the live pipeline.

FilePlaybackPipeline fills the same ring buffer AudioPipeline does, from a thread
that feeds chunks at the file's own sample rate instead of a PortAudio callback.
Everything downstream — analyzer, scope, waterfall, meters — is unchanged and
cannot tell the difference, which is the point: an event captured by the recorder
can be replayed later and analysed exactly as it was live, with the display
running at real speed for a screen recording.

Playback deliberately opens no audio device, so a recording can be reviewed on a
machine with no receiver attached.

At the end of the file the feeder simply stops.  The buffer keeps its last few
seconds, so the display holds its final frame rather than going blank, and the
analyzer drops to SEARCHING on its own once the audio stops advancing.
"""

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np

from buzz.sampler import RingBufferPipeline

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2   # 16-bit PCM, matching what the recorder writes


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM .wav into (int16 samples, sample rate).

    Multi-channel files are reduced to channel 0 rather than mixed down, matching
    what the live pipeline does with a stereo input device.

    Sample width is the one property worth refusing outright: the whole signal
    chain is int16 end to end, and silently converting a 24- or 32-bit file would
    invite a mismatch between what the analyzer measures here and what it measured
    live.  Recordings this program produces are always 16-bit.
    """
    with wave.open(str(path), 'rb') as wav:
        if wav.getsampwidth() != _BYTES_PER_SAMPLE:
            raise ValueError(
                f'{path}: expected 16-bit PCM audio, got {wav.getsampwidth() * 8}-bit')
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype='<i2')[::channels]
    # astype() rather than the frombuffer view: that view is read-only and borrows
    # the file's byte order, and consumers expect a plain writable native int16 array.
    return samples.astype(np.int16), sample_rate


def resolve_playback_path(name: Path | str, directory: Path) -> Path:
    """Resolve a --playback argument to a file path.

    A bare filename is looked up in the recordings directory, so replaying a
    capture is just `--playback event-20260729-143307-0700.wav`; anything with a
    directory component in it is used as given.
    """
    path = Path(name)
    return directory / path if str(name) == path.name else path


class FilePlaybackPipeline(RingBufferPipeline):
    """Replays a .wav into the ring buffer at real-time speed.

    The feeder thread appends one CHUNK_SIZE chunk at a time against a deadline
    schedule computed from the file's sample rate.  Deadlines accumulate from a
    fixed origin rather than by sleeping a fixed interval each pass, so the small
    per-chunk overhead of the sleep and the append cannot accumulate into playback
    running progressively slower than the audio it represents.
    """

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._samples, self.sample_rate = load_wav(path)
        self.path = Path(path)
        self.duration = len(self._samples) / self.sample_rate
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(
            target=self._feed, daemon=True, name='playback')
        self._thread.start()

    @property
    def finished(self) -> bool:
        """True once the last chunk of the file has been fed into the buffer."""
        return self._finished.is_set()

    def _feed(self) -> None:
        chunk_period = self.CHUNK_SIZE / self.sample_rate
        origin = time.monotonic()
        # Partial trailing chunks are dropped: consumers read whole chunks, and up
        # to 32 ms of audio at the very end of a file is not worth a special case.
        n_chunks = len(self._samples) // self.CHUNK_SIZE
        logger.info('Playing back %s — %.1f s at %d Hz',
                    self.path.name, self.duration, self.sample_rate)
        for i in range(n_chunks):
            if self._stop.wait(max(0.0, origin + i * chunk_period - time.monotonic())):
                return
            self._append(self._samples[i * self.CHUNK_SIZE:(i + 1) * self.CHUNK_SIZE])
        self._finished.set()
        logger.info('Playback finished: %s', self.path.name)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

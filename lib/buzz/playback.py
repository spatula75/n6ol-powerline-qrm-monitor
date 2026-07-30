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
    """Replays a .wav into the ring buffer at real-time speed, with transport control.

    The feeder thread appends one CHUNK_SIZE chunk at a time against a deadline
    schedule computed from the file's sample rate.  Deadlines accumulate from an
    origin rather than by sleeping a fixed interval each pass, so the small
    per-chunk overhead of the sleep and the append cannot accumulate into playback
    running progressively slower than the audio it represents.  Pausing and
    restarting re-base that origin, since the schedule is only meaningful for a
    stretch of uninterrupted play.

    pause(), resume() and restart() are called from the Qt thread while the feeder
    runs, so the play position and the origin live behind a Condition the feeder
    waits on.  Appending takes the ring buffer's own lock inside that one, which is
    the only place the two nest, and always in that order.
    """

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._samples, self.sample_rate = load_wav(path)
        self.path = Path(path)
        self.duration = len(self._samples) / self.sample_rate
        # Partial trailing chunks are dropped: consumers read whole chunks, and up
        # to 32 ms of audio at the very end of a file is not worth a special case.
        self._chunks = len(self._samples) // self.CHUNK_SIZE
        self._chunk_period = self.CHUNK_SIZE / self.sample_rate

        self._state = threading.Condition()
        self._index = 0             # next chunk to feed
        self._paused = False
        self._origin = 0.0          # monotonic time that _origin_index was due at
        self._origin_index = 0
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._rebase()

        self._thread = threading.Thread(
            target=self._feed, daemon=True, name='playback')
        self._thread.start()

    # ------------------------------------------------------------------ public

    @property
    def finished(self) -> bool:
        """True once the last chunk of the file has been fed into the buffer."""
        return self._finished.is_set()

    @property
    def paused(self) -> bool:
        with self._state:
            return self._paused

    @property
    def position(self) -> float:
        """How far into the file playback has reached, in seconds."""
        with self._state:
            return self._index * self._chunk_period

    def pause(self) -> None:
        with self._state:
            self._paused = True
            self._state.notify_all()

    def resume(self) -> None:
        """Carry on from the current position.  Does nothing at the end of the file."""
        with self._state:
            if self._index >= self._chunks:
                return
            self._paused = False
            self._rebase()
            self._state.notify_all()

    def toggle_pause(self) -> None:
        self.pause() if not self.paused else self.resume()

    def restart(self) -> None:
        """Play again from the beginning, whether paused, playing, or finished.

        Restarting resumes rather than staying paused: a restart button that leaves
        a paused file paused at zero looks like it did nothing at all.

        The ring buffer is emptied too.  It still holds the last several seconds of
        the pass just abandoned, and leaving that in place would hand an analyzer
        that has just been reset a locked-on pulse train to find immediately —
        producing a second pass that opens already locked, which is precisely what
        restarting is meant to avoid.  The cost is a few seconds of empty display
        while the new pass refills it, the same as at startup.
        """
        with self._state:
            self._index = 0
            self._paused = False
            self._finished.clear()
            self._rebase()
            self.clear()
            self._state.notify_all()

    # ----------------------------------------------------------------- internal

    def _rebase(self) -> None:
        """Start the deadline schedule again from now (caller holds _state)."""
        self._origin, self._origin_index = time.monotonic(), self._index

    def _wait_until_playable(self) -> bool:
        """Block while paused or sitting at the end; False once stopping."""
        with self._state:
            while not self._stop.is_set() and (self._paused or self._index >= self._chunks):
                self._state.wait()
            return not self._stop.is_set()

    def _feed(self) -> None:
        logger.info('Playing back %s — %.1f s at %d Hz',
                    self.path.name, self.duration, self.sample_rate)
        while self._wait_until_playable():
            with self._state:
                index = self._index
                due = self._origin + (index - self._origin_index) * self._chunk_period
            delay = due - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                return
            with self._state:
                # A pause or a restart while this chunk was waiting its turn makes
                # it the wrong chunk to play; go back and read the new position.
                if self._paused or self._index != index:
                    continue
                self._append(
                    self._samples[index * self.CHUNK_SIZE:(index + 1) * self.CHUNK_SIZE])
                self._index += 1
                if self._index >= self._chunks:
                    self._finished.set()
                    logger.info('Playback finished: %s', self.path.name)

    def close(self) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()
        self._thread.join(timeout=1.0)

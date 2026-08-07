"""Shared apparatus for the integration suite: real components, driven at real speed.

Kept out of conftest.py because these are things the tests import and call, not
fixtures pytest injects - conftest.py next door holds the fixtures built on top of
them.  Importable as a bare module name because pytest puts each test file's own
directory on sys.path when there is no __init__.py beside it.

Nothing here opens an audio device.  Playback is muted by default and the monitor
is fed by appending to the ring buffer directly, so the whole of this tier runs on
a machine with no sound card at all - which is what a CI runner is.
"""

import threading
import time
from pathlib import Path

import numpy as np

from buzz.analyzer import AnalyzerState, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.playback import FilePlaybackPipeline
from buzz.recorder import EventRecorder
from buzz.sampler import RingBufferPipeline

RATE = 16000
PULSE_RATE = 120
CHUNK = RingBufferPipeline.CHUNK_SIZE
BUFFERED_SECONDS = RingBufferPipeline().capacity_samples / RATE

# Amplitudes chosen from what the analyzer actually reports for them: roughly 18 dB
# SNR and 52 dB.  Straddling a 30 dB threshold by twelve either way keeps the tests
# about the components under test rather than about the last decibel of the DSP.
QUIET_PULSES = 400
LOUD_PULSES = 20_000


def audio(seconds: float, amplitude: int | None, seed: int = 1) -> np.ndarray:
    """Noise, optionally carrying a 120 pps pulse train of the given amplitude."""
    n = int(seconds * RATE)
    data = np.random.default_rng(seed).integers(-100, 101, size=n, dtype=np.int16)
    if amplitude is None:
        return data
    spacing = RATE / PULSE_RATE
    for i in range(int(n / spacing)):
        at = round(i * spacing)
        if at + 3 < n:
            data[at:at + 3] = amplitude
    return data


def config_for(directory: Path | None = None, **recording) -> BuzzConfig:
    """A config wired to the synthetic signal these tests generate."""
    config = BuzzConfig()
    config.audio.sample_rate, config.audio.pulse_rate = RATE, PULSE_RATE
    if directory is not None:
        config.recording.directory = str(directory)
        config.recording.enabled = True
        config.recording.max_events = 1
    for key, value in recording.items():
        setattr(config.recording, key, value)
    return config


class StateLog:
    """Every state the analyzer passed through, in order.

    Registered as a listener rather than polled, for the reason add_state_listener()
    gives: lock is an event, not a level.  A test that polls `analyzer.state` can
    miss a lock that comes and goes between two polls - and one of the things this
    suite exists to prove is that a lock *dropped*, which is precisely the kind of
    brief transition polling is blind to.

    Runs on the analyzer thread, so it does the least it can: append under a lock
    and notify.  Register before start(); the listener list is not synchronized.
    """

    def __init__(self) -> None:
        self._states: list[AnalyzerState] = []
        self._changed = threading.Condition()

    def __call__(self, state: AnalyzerState) -> None:
        with self._changed:
            self._states.append(state)
            self._changed.notify_all()

    @property
    def states(self) -> list[AnalyzerState]:
        with self._changed:
            return list(self._states)

    def wait_for(self, state: AnalyzerState, timeout: float) -> bool:
        """Block until the analyzer next enters `state`.  True if it did in time.

        Waits for a transition rather than a level: a caller that has just watched a
        lock and wants to see the *next* one calls mark() first, so an already-locked
        analyzer cannot satisfy the wait without changing anything.
        """
        deadline = time.monotonic() + timeout
        with self._changed:
            while state not in self._states:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._changed.wait(remaining):
                    return state in self._states
            return True

    def mark(self) -> None:
        """Forget the history, so wait_for() speaks only about what happens next."""
        with self._changed:
            self._states.clear()


class Monitor:
    """A real pipeline, analyzer and recorder, fed audio at the speed of the clock."""

    def __init__(self, directory: Path, **recording) -> None:
        config = config_for(directory, **recording)
        self.directory = directory
        self.pipeline = RingBufferPipeline()
        self.analyzer = ContinuousAnalyzer(self.pipeline, config)
        self.recorder = EventRecorder(self.pipeline, self.analyzer, config)
        self.log = StateLog()
        self.analyzer.add_state_listener(self.log)
        self.analyzer.start()
        self.recorder.start()

    def play(self, seconds: float, amplitude: int | None) -> None:
        """Feed audio in real time, so every timer under test runs as it would live."""
        samples = audio(seconds, amplitude)
        for i in range(len(samples) // CHUNK):
            self.pipeline._append(samples[i * CHUNK:(i + 1) * CHUNK])
            time.sleep(CHUNK / RATE)

    def stop(self) -> None:
        self.recorder.stop()
        self.analyzer.stop()

    def recordings(self) -> list[Path]:
        return sorted(self.directory.glob('*.wav'))


class Replay:
    """A recording played back through a real analyzer, wired as the display wires it.

    Muted, so no output stream is ever opened and playback stays paced by its own
    deadline schedule.  That is the same clock a machine with no sound card uses, but
    it does mean this class cannot say anything about the other one - see the note in
    the module docstring of test_replay_transport.py.
    """

    def __init__(self, path: Path) -> None:
        self.playback = FilePlaybackPipeline(path)
        self.analyzer = ContinuousAnalyzer(self.playback, config_for())
        self.log = StateLog()
        self.analyzer.add_state_listener(self.log)

    def __enter__(self) -> 'Replay':
        self.playback.start()
        self.analyzer.start()
        return self

    def __exit__(self, *exc) -> None:
        self.analyzer.stop()
        self.playback.close()

    def restart(self) -> None:
        """Exactly what the Restart button does - see waterfall.RecordingBarWidget.

        Reset before restart, in that order, because the order is part of what is
        being tested: reset while the abandoned pass is still the newest audio, so a
        tick already in flight cannot publish a lock against fresh samples that the
        reset had just revoked.
        """
        self.analyzer.reset()
        self.playback.restart()

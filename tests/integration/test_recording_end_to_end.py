"""End-to-end recording, over the real analyzer and real threads at real speed.

Deselected by default because it costs tens of seconds.  Run it with:

    pytest -m integration --no-cov

These drive the components the unit tests stub out: a real ContinuousAnalyzer
measuring real synthetic audio, a real EventRecorder polling on its own thread, and
a real ring buffer filling and discarding in real time.  That combination is where
this project's costly bugs have lived — an analyzer reset that silently did
nothing, a recording that came out empty because a limit was measured from the
wrong end — none of which a stubbed analyzer would have shown.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from buzz import wavmeta
from buzz.analyzer import AnalyzerState, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.playback import load_wav
from buzz.recorder import EventRecorder
from buzz.sampler import RingBufferPipeline

RATE = 16000
PULSE_RATE = 120
CHUNK = RingBufferPipeline.CHUNK_SIZE
BUFFERED_SECONDS = RingBufferPipeline().capacity_samples / RATE

# Amplitudes chosen from what the analyzer actually reports for them: roughly 18 dB
# SNR and 52 dB.  Straddling a 30 dB threshold by twelve either way keeps the test
# about the recorder rather than about the last decibel of the DSP.
QUIET_PULSES = 400
LOUD_PULSES = 20_000


def _audio(seconds: float, amplitude: int | None, seed: int = 1) -> np.ndarray:
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


class Monitor:
    """A real pipeline, analyzer and recorder, fed audio at the speed of the clock."""

    def __init__(self, directory: Path, **recording) -> None:
        config = BuzzConfig()
        config.audio.sample_rate, config.audio.pulse_rate = RATE, PULSE_RATE
        config.recording.directory = str(directory)
        config.recording.enabled = True
        config.recording.max_events = 1
        for key, value in recording.items():
            setattr(config.recording, key, value)
        self.directory = directory
        self.pipeline = RingBufferPipeline()
        self.analyzer = ContinuousAnalyzer(self.pipeline, config)
        self.recorder = EventRecorder(self.pipeline, self.analyzer, config)
        self.analyzer.start()
        self.recorder.start()

    def play(self, seconds: float, amplitude: int | None) -> None:
        """Feed audio in real time, so every timer under test runs as it would live."""
        audio = _audio(seconds, amplitude)
        for i in range(len(audio) // CHUNK):
            self.pipeline._append(audio[i * CHUNK:(i + 1) * CHUNK])
            time.sleep(CHUNK / RATE)

    def stop(self) -> None:
        self.recorder.stop()
        self.analyzer.stop()

    def recordings(self) -> list[Path]:
        return sorted(self.directory.glob('*.wav'))


@pytest.mark.integration
class TestThresholdCrossing:
    """A faint arc that grows loud: what min_lock_snr exists for.

    The recording must be everything the buffer was holding when the level crossed,
    plus max_seconds of the loud part on top.  Both of the ways this was got wrong
    were invisible to unit tests with a stubbed analyzer: measuring the cap from the
    lock spent it before the file opened and saved a real event as silence, and
    measuring it from the first buffered sample let free lead-in eat the cap and
    left a fraction of a second of the part worth hearing.
    """

    @pytest.fixture(scope='class')
    def crossed(self, tmp_path_factory):
        monitor = Monitor(tmp_path_factory.mktemp('crossing'),
                          min_lock_snr=30.0, max_seconds=5.0, stop_after_seconds=3.0)
        try:
            monitor.play(2, None)                   # quiet: nothing to lock onto
            monitor.play(12, QUIET_PULSES)          # locked, but below the threshold
            monitor.play(8, LOUD_PULSES)            # crosses; recording starts here
            monitor.play(4, None)
        finally:
            monitor.stop()
        return monitor

    def test_the_event_is_recorded(self, crossed):
        assert len(crossed.recordings()) == 1

    def test_the_whole_buffer_is_kept_as_lead_in(self, crossed):
        """Not merely the audio after the crossing: the approach to an event is the
        part that shows what the arc was doing before it got loud."""
        samples = load_wav(crossed.recordings()[0])[0]
        assert len(samples) / RATE == pytest.approx(BUFFERED_SECONDS + 5.0, abs=0.3)

    def test_the_lead_in_holds_the_quiet_approach(self, crossed):
        """Proof that the buffer really was kept, rather than the length coming out
        right by some other route: its opening seconds are the faint pulse train."""
        samples = load_wav(crossed.recordings()[0])[0]
        opening = np.abs(samples[:int(BUFFERED_SECONDS * RATE) - RATE]).max()
        assert QUIET_PULSES // 2 < opening < LOUD_PULSES // 2

    def test_the_loud_part_is_there_too(self, crossed):
        samples = load_wav(crossed.recordings()[0])[0]
        assert np.abs(samples[-int(4 * RATE):]).max() > LOUD_PULSES // 2

    def test_it_stopped_because_of_the_cap(self, crossed):
        settings = wavmeta.read_settings(crossed.recordings()[0])
        assert settings['ended'] == 'capped'

    def test_the_lock_is_older_than_the_file(self, crossed):
        """The lock happened while the arc was still faint, long enough ago that it
        has fallen out of the buffer — which is exactly the case that used to save
        the event as a nought-second recording."""
        settings = wavmeta.read_settings(crossed.recordings()[0])
        assert float(settings['lead_in_seconds']) == 0.0


@pytest.mark.integration
class TestPlaybackRelocksARecording:
    """A recording this monitor made must be analysable by the same monitor."""

    def test_a_recorded_event_locks_again_on_replay(self, tmp_path):
        from buzz.playback import FilePlaybackPipeline

        monitor = Monitor(tmp_path, max_seconds=6.0, stop_after_seconds=2.0)
        try:
            monitor.play(2, None)
            monitor.play(10, LOUD_PULSES)
            monitor.play(3, None)
        finally:
            monitor.stop()
        recording = monitor.recordings()[0]

        config = BuzzConfig()
        config.audio.sample_rate, config.audio.pulse_rate = RATE, PULSE_RATE
        with FilePlaybackPipeline(recording) as playback:
            analyzer = ContinuousAnalyzer(playback, config)
            playback.start()
            analyzer.start()
            deadline = time.monotonic() + playback.duration + 2.0
            locked = False
            while time.monotonic() < deadline and not locked:
                locked = analyzer.state == AnalyzerState.LOCKED
                time.sleep(0.05)
            analyzer.stop()
        assert locked

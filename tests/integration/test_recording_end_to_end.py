"""End-to-end recording, over the real analyzer and real threads at real speed.

Deselected by default because it costs tens of seconds.  Run it with:

    pytest -m integration --no-cov

These drive the components the unit tests stub out: a real ContinuousAnalyzer
measuring real synthetic audio, a real EventRecorder polling on its own thread, and
a real ring buffer filling and discarding in real time.  That combination is where
this project's costly bugs have lived - an analyzer reset that silently did
nothing, a recording that came out empty because a limit was measured from the
wrong end - none of which a stubbed analyzer would have shown.
"""

import numpy as np
import pytest
from harness import BUFFERED_SECONDS, LOUD_PULSES, QUIET_PULSES, RATE, Monitor

from buzz import wavmeta
from buzz.playback import load_wav


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
    @staticmethod
    def crossed(tmp_path_factory):
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
        has fallen out of the buffer - which is exactly the case that used to save
        the event as a nought-second recording."""
        settings = wavmeta.read_settings(crossed.recordings()[0])
        assert float(settings['lead_in_seconds']) == 0.0

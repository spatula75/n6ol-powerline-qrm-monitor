"""Replaying a recording: re-acquisition, restart, and the transport's own clock.

    pytest -m integration --no-cov

Tier 1 — no audio device is opened.  Playback is muted throughout, which means the
feeder is paced by its internal deadline schedule rather than by a sound card's
write().  Everything below is true of both clocks, but only the deadline one is
exercised here; the switch between them happens when an output stream opens, so it
cannot be tested on a machine without one and belongs to the manual tier instead.
"""

import time

import pytest
from harness import Replay

from buzz.analyzer import AnalyzerState
from buzz.playback import FilePlaybackPipeline

# Generous, and deliberately so.  Acquisition takes a few seconds on an idle machine
# and rather more on a loaded runner; every assertion here is about whether something
# happened at all, never about how quickly.  A timeout that fails under load is a
# test that gets ignored, which is worse than a slow one.
ACQUIRE_TIMEOUT = 30.0


@pytest.mark.integration
class TestReplayRelocks:
    """A recording this monitor made must be analysable by the same monitor."""

    def test_a_recorded_event_locks_again_on_replay(self, recorded_event):
        with Replay(recorded_event) as replay:
            assert replay.log.wait_for(AnalyzerState.LOCKED, ACQUIRE_TIMEOUT)


@pytest.mark.integration
class TestRestartAcquiresCold:
    """Restart must throw away what the analyzer learned, not just rewind the audio.

    The bug this exists for got through twice.  The analyzer that has just watched an
    event through knows where the pulse train is and how fast it is drifting, so a
    restart that fails to clear that opens the second pass already locked — at a
    drift rate learned from the pass before.  Watching the acquisition is usually the
    whole point of replaying an event, so a restart that skips it is a restart that
    silently did nothing, and it looks identical to one that worked.

    Both halves matter and they fail differently.  A reset that does not clear the
    tracker never leaves LOCKED; one that clears it but leaves the old audio in the
    ring buffer re-locks instantly off samples that were supposed to be abandoned.
    Asserting the lock dropped *and* came back distinguishes them from a test that
    merely ends up locked either way.
    """

    def test_restart_drops_the_lock_and_acquires_from_scratch(self, recorded_event):
        with Replay(recorded_event) as replay:
            assert replay.log.wait_for(AnalyzerState.LOCKED, ACQUIRE_TIMEOUT), \
                'never locked on the first pass, so there is nothing to reset'

            # Only what happens after the button, so an analyzer that is already
            # locked cannot satisfy the waits below without having changed anything.
            replay.log.mark()
            replay.restart()

            assert replay.log.wait_for(AnalyzerState.SEARCHING, ACQUIRE_TIMEOUT), \
                'the lock survived the restart: the analyzer was never really reset'
            assert replay.log.wait_for(AnalyzerState.LOCKED, ACQUIRE_TIMEOUT), \
                'the second pass never re-acquired the signal it had just been shown'

    def test_the_restarted_pass_plays_from_the_beginning(self, recorded_event):
        """The audio rewinds too, and not merely the analysis."""
        with Replay(recorded_event) as replay:
            assert replay.log.wait_for(AnalyzerState.LOCKED, ACQUIRE_TIMEOUT)
            assert replay.playback.position > 0
            replay.restart()
            assert replay.playback.position < 1.0


@pytest.mark.integration
class TestTransportPosition:
    """Pausing holds the position; resuming carries on from it rather than catching up.

    The deadline schedule accumulates from an origin, so a resume that failed to
    re-base it would leave the feeder behind by however long the pause lasted — and
    it would then dump chunks as fast as the loop could run until it caught up.  That
    is audible as a burst of fast-forward on resume, and it is invisible to a unit
    test, where nothing is paced by a clock in the first place.
    """

    @pytest.fixture(scope='class')
    @staticmethod
    def transport(recorded_event):
        """One pass through pause and resume, sampled as it goes.

        The measurements are taken once and asserted separately, because taking them
        costs several seconds of real time and re-running the transport per assertion
        would multiply that by four for no extra coverage.
        """
        playback = FilePlaybackPipeline(recorded_event)
        with playback:
            playback.start()
            deadline = time.monotonic() + 5.0
            while playback.position == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            started_at = playback.position

            time.sleep(1.0)
            while_playing = playback.position

            playback.pause()
            # The feeder may be part-way through a chunk when the pause lands, and it
            # finishes that one before looking; sampling immediately would catch the
            # position still moving for reasons that have nothing to do with the test.
            time.sleep(0.2)
            paused_at = playback.position
            time.sleep(1.5)
            still_paused_at = playback.position

            playback.resume()
            time.sleep(1.0)
            after_resume = playback.position
        return {'started_at': started_at, 'while_playing': while_playing,
                'paused_at': paused_at, 'still_paused_at': still_paused_at,
                'after_resume': after_resume}

    def test_it_plays(self, transport):
        assert transport['started_at'] < transport['while_playing']

    def test_it_plays_at_about_real_speed(self, transport):
        """A second of wall clock is a second of audio, give or take a chunk either
        side — the whole reason replay has a feeder thread rather than a loop."""
        played = transport['while_playing'] - transport['started_at']
        assert played == pytest.approx(1.0, abs=0.3)

    def test_pause_stops_the_position(self, transport):
        assert transport['still_paused_at'] == transport['paused_at']

    def test_resume_carries_on_rather_than_catching_up(self, transport):
        """The failure this rules out is a leap forward of about the paused duration:
        the second of audio played after resuming must be about a second of audio."""
        played = transport['after_resume'] - transport['paused_at']
        assert played == pytest.approx(1.0, abs=0.3)

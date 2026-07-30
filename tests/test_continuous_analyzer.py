"""Tests for ContinuousAnalyzer state machine and DSP integration.

Tier methods (_full_analysis, _quick_check, _noise_check, _phase_search) measure
and publish, then return the state they propose; _transition() applies it.  The
_step() helper below mirrors what _run()'s tick methods do in production.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from buzz.analyzer import (
    AnalysisResult, AnalyzerState, ContinuousAnalyzer, TriggerSync, fit_drift_rate,
)
from buzz.config import BuzzConfig
from buzz.dsp import pulse_phase_period
from buzz.sampler import AudioPipeline

SAMPLE_RATE = 16000
PULSE_RATE  = 120
MIN_FIT     = ContinuousAnalyzer.DRIFT_FIT_MIN_POINTS


def _make_config() -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.sample_rate   = SAMPLE_RATE
    cfg.audio.pulse_rate    = PULSE_RATE
    cfg.audio.input_device_name = 'Test'
    return cfg


def _make_analyzer() -> ContinuousAnalyzer:
    cfg      = _make_config()
    pipeline = MagicMock(spec=AudioPipeline)
    pipeline.wait_for_data.return_value = True
    # Drift is tracked against the audio clock.  Freezing it means no audio time
    # passes between calls, so these tests see no drift prediction at all — the
    # tracker is exercised separately with _AlignedStreamPipeline, which advances it.
    pipeline.total_samples = 0
    return ContinuousAnalyzer(pipeline, cfg)


def _step(az: ContinuousAnalyzer, tier_method) -> str:
    """Run one tier method and apply its proposed state, as _run's tick methods do."""
    proposed = tier_method()
    az._transition(proposed)
    return proposed


def _pulse_audio(n: int = SAMPLE_RATE, amplitude: int = 20000, phase: int = 5) -> np.ndarray:
    """1 second of int16 audio with a clear 120 pps pulse train.

    Noise is bipolar and zero-mean, matching LSB receiver audio (a frequency-
    translated slice of RF), so _capture's DC removal is a no-op on the fixture
    rather than shifting its level.  Pulse positions use round(), the grid the
    kernel and the amplitude averager both work on.
    """
    data = np.random.default_rng(42).integers(-100, 101, size=n, dtype=np.int16)
    spp  = SAMPLE_RATE / PULSE_RATE
    for i in range(int(n / spp)):
        pos = phase + round(i * spp)
        if pos + 3 < n:
            data[pos] = data[pos + 1] = data[pos + 2] = amplitude
    return data


def _noise_audio(n: int = SAMPLE_RATE) -> np.ndarray:
    """1 second of flat, zero-mean random noise with no pulse structure."""
    return np.random.default_rng(0).integers(-100, 101, size=n, dtype=np.int16)


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_is_frozen(self):
        r = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        with pytest.raises((AttributeError, TypeError)):
            r.snr = 0.0  # type: ignore[misc]

    def test_fields(self):
        r = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        assert r.signal_dbm == -70.0
        assert r.noise_dbm  == -90.0
        assert r.snr        == 20.0
        assert r.locked     is True

    def test_unlocked_signal_coincides_with_noise(self):
        """The plotter and meter panel rely on this convention — see AnalysisResult.unlocked."""
        r = AnalysisResult.unlocked(-92.5)
        assert r.signal_dbm == r.noise_dbm == -92.5
        assert r.snr == 0.0
        assert r.locked is False


class TestAnalyzerState:
    def test_states_compare_equal_to_their_names(self):
        """StrEnum keeps string comparison working for logs and tests."""
        assert AnalyzerState.LOCKED == 'LOCKED'
        assert AnalyzerState.SEARCHING == 'SEARCHING'
        assert AnalyzerState.SIGNAL_LOST == 'SIGNAL_LOST'


# ---------------------------------------------------------------------------
# ContinuousAnalyzer initialisation
# ---------------------------------------------------------------------------

class TestContinuousAnalyzerInit:
    def test_initial_state_is_searching(self):
        assert _make_analyzer()._state == 'SEARCHING'

    def test_initial_result_is_none(self):
        assert _make_analyzer().latest_result() is None

    def test_initial_phases_not_valid(self):
        assert _make_analyzer()._phases_valid is False

    def test_stop_event_not_set_initially(self):
        assert not _make_analyzer()._stop.is_set()

    def test_stop_sets_event(self):
        az = _make_analyzer()
        az.stop()
        assert az._stop.is_set()


# ---------------------------------------------------------------------------
# _transition
# ---------------------------------------------------------------------------

class TestTransition:
    def test_same_state_is_noop(self):
        az = _make_analyzer()
        az._consecutive_low_snr = 2
        az._transition('SEARCHING')
        assert az._state == 'SEARCHING'
        assert az._consecutive_low_snr == 2   # untouched on no-op

    def test_entering_locked_sets_phases_valid(self):
        az = _make_analyzer()
        az._transition('LOCKED')
        assert az._phases_valid is True

    def test_entering_locked_stamps_refine_timer(self):
        az = _make_analyzer()
        assert az._last_refine == 0.0
        az._transition('LOCKED')
        assert az._last_refine > 0.0

    def test_any_transition_resets_debounce_counter(self):
        az = _make_analyzer()
        az._transition('LOCKED')
        az._consecutive_low_snr = 2
        az._transition('SIGNAL_LOST')
        assert az._consecutive_low_snr == 0

    def test_state_is_updated(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        assert az._state == 'SIGNAL_LOST'


class TestReset:
    """Replaying a file from the top must not start with what the last pass learned."""

    def _locked_analyzer(self):
        """An analyzer carrying everything a pass over a real event leaves behind."""
        az = _make_analyzer()
        az._transition(AnalyzerState.LOCKED)
        az._peak_phase, az._noise_phase = 41.0, 97.0
        az._phase_drift_rate = 0.35
        az._phases_measured_at = 12.0
        az._unwrapped_phase = 806.0
        az._previous_measured_phase = 41
        az._phase_history.extend([(1.0, 40.0), (2.0, 40.5)])
        az._dc = 3.5
        az._consecutive_low_snr = 2
        az._last_refine = az._last_full_fft = time.monotonic()
        az._publish(AnalysisResult(signal_dbm=-70.0, noise_dbm=-95.0, snr=25.0, locked=True))
        return az

    def test_returns_to_searching(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert az.state == AnalyzerState.SEARCHING

    def test_forgets_the_phase_pair(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert (az._peak_phase, az._noise_phase, az._phases_valid) == (0.0, 0.0, False)

    def test_forgets_the_drift_rate(self):
        """Unlike a signal outage, which keeps the rate as a good starting guess:
        after a reset there is nothing to be right about yet."""
        az = self._locked_analyzer()
        az._apply_reset()
        assert az._phase_drift_rate == 0.0

    def test_forgets_the_fitted_history(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert len(az._phase_history) == 0 and az._previous_measured_phase is None

    def test_forgets_the_dc_estimate(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert az._dc is None

    def test_drops_published_results(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert az.latest_result() is None and az.drain_results() == []

    def test_clears_the_tier_timers(self):
        az = self._locked_analyzer()
        az._apply_reset()
        assert (az._last_refine, az._last_full_fft) == (0.0, 0.0)

    def test_tells_listeners_the_lock_is_gone(self):
        az = self._locked_analyzer()
        seen = []
        az.add_state_listener(seen.append)
        az._apply_reset()
        assert seen == [AnalyzerState.SEARCHING]

    def test_says_nothing_when_already_searching(self):
        az = _make_analyzer()
        seen = []
        az.add_state_listener(seen.append)
        az._apply_reset()
        assert seen == []

    def test_request_is_deferred_to_the_analysis_thread(self):
        """reset() is called from the Qt thread, so it must not touch anything."""
        az = self._locked_analyzer()
        az.reset()
        assert az.state == AnalyzerState.LOCKED

    def test_request_is_recorded(self):
        az = self._locked_analyzer()
        az.reset()
        assert az._reset_requested.is_set()

    def test_the_scope_stops_trusting_the_phase_immediately(self):
        """Not on the next tick: a tick can sit in wait_for_data for a second, which
        is long enough for replayed audio to re-lock and for the operator to conclude
        the restart did nothing at all."""
        az = self._locked_analyzer()
        az.reset()
        assert az.trigger_phase()[1] == TriggerSync.FREE

    def test_the_last_result_is_dropped_immediately(self):
        az = self._locked_analyzer()
        az.reset()
        assert az.latest_result() is None

    def test_buffered_results_are_dropped_immediately(self):
        az = self._locked_analyzer()
        az.reset()
        assert az.drain_results() == []

    def test_applying_clears_the_request(self):
        az = self._locked_analyzer()
        az.reset()
        az._apply_reset()
        assert not az._reset_requested.is_set()


class TestStateListeners:
    def test_listener_is_called_on_a_transition(self):
        az = _make_analyzer()
        seen = []
        az.add_state_listener(seen.append)
        az._transition(AnalyzerState.LOCKED)
        assert seen == [AnalyzerState.LOCKED]

    def test_listener_is_not_called_when_the_state_is_unchanged(self):
        az = _make_analyzer()
        seen = []
        az.add_state_listener(seen.append)
        az._transition(AnalyzerState.SEARCHING)   # already SEARCHING
        assert seen == []

    def test_every_listener_is_called(self):
        az = _make_analyzer()
        first, second = [], []
        az.add_state_listener(first.append)
        az.add_state_listener(second.append)
        az._transition(AnalyzerState.LOCKED)
        assert (first, second) == ([AnalyzerState.LOCKED], [AnalyzerState.LOCKED])

    def test_listener_sees_the_bookkeeping_already_applied(self):
        az = _make_analyzer()
        az.add_state_listener(lambda _: seen.append(az._phases_valid))
        seen = []
        az._transition(AnalyzerState.LOCKED)
        assert seen == [True]

    def test_a_raising_listener_does_not_break_the_transition(self):
        az = _make_analyzer()

        def explode(_):
            raise RuntimeError('boom')

        az.add_state_listener(explode)
        az._transition(AnalyzerState.LOCKED)
        assert az.state == AnalyzerState.LOCKED

    def test_a_raising_listener_does_not_stop_the_others(self):
        az = _make_analyzer()
        seen = []

        def explode(_):
            raise RuntimeError('boom')

        az.add_state_listener(explode)
        az.add_state_listener(seen.append)
        az._transition(AnalyzerState.LOCKED)
        assert seen == [AnalyzerState.LOCKED]

    def test_state_property_reports_the_current_state(self):
        az = _make_analyzer()
        az._transition(AnalyzerState.SIGNAL_LOST)
        assert az.state == AnalyzerState.SIGNAL_LOST


# ---------------------------------------------------------------------------
# Tick methods — per-state scheduling (tier cadence and transition wiring)
# ---------------------------------------------------------------------------

class TestTickMethods:
    def test_searching_tick_runs_full_analysis(self):
        az = _make_analyzer()
        with patch.object(az, '_full_analysis', return_value='SEARCHING') as fa:
            interval = az._searching_tick()
        fa.assert_called_once()
        assert interval == ContinuousAnalyzer.SEARCH_INTERVAL
        assert az._last_full_fft > 0.0

    def test_locked_tick_quick_checks_between_refines(self):
        az = _make_analyzer()
        az._transition('LOCKED')   # stamps _last_refine = now
        with patch.object(az, '_quick_check', return_value='LOCKED') as qc, \
             patch.object(az, '_phase_search') as ps:
            interval = az._locked_tick()
        qc.assert_called_once()
        ps.assert_not_called()
        assert interval == ContinuousAnalyzer.FAST_TICK_INTERVAL

    def test_locked_tick_refines_when_interval_elapsed(self):
        az = _make_analyzer()
        az._transition('LOCKED')
        az._last_refine = 0.0   # refine long overdue
        with patch.object(az, '_phase_search', return_value='LOCKED') as ps, \
             patch.object(az, '_quick_check') as qc:
            az._locked_tick()
        ps.assert_called_once()
        qc.assert_not_called()
        assert az._last_refine > 0.0

    def test_signal_lost_tick_noise_checks_every_tick(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        with patch.object(az, '_noise_check', return_value='SIGNAL_LOST') as nc, \
             patch.object(az, '_phase_search', return_value='SIGNAL_LOST'), \
             patch.object(az, '_fast_scan', return_value=False), \
             patch.object(az, '_full_analysis', return_value='SIGNAL_LOST'):
            interval = az._signal_lost_tick()
        nc.assert_called_once()
        assert interval == ContinuousAnalyzer.FAST_TICK_INTERVAL

    def test_signal_lost_tick_skips_lower_tiers_after_tier1_relock(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        with patch.object(az, '_noise_check', return_value='LOCKED'), \
             patch.object(az, '_phase_search') as ps, \
             patch.object(az, '_fast_scan') as fs:
            az._signal_lost_tick()
        ps.assert_not_called()
        fs.assert_not_called()
        assert az._state == 'LOCKED'

    def test_signal_lost_tick_runs_full_fft_on_fast_scan_hit(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        with patch.object(az, '_noise_check', return_value='SIGNAL_LOST'), \
             patch.object(az, '_phase_search', return_value='SIGNAL_LOST'), \
             patch.object(az, '_fast_scan', return_value=True), \
             patch.object(az, '_full_analysis', return_value='LOCKED') as fa:
            az._signal_lost_tick()
        fa.assert_called_once()
        assert az._state == 'LOCKED'

    def test_signal_lost_tick_skips_full_fft_without_hit_before_backstop(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        az._last_full_fft = time.monotonic()   # backstop not yet due
        with patch.object(az, '_noise_check', return_value='SIGNAL_LOST'), \
             patch.object(az, '_phase_search', return_value='SIGNAL_LOST'), \
             patch.object(az, '_fast_scan', return_value=False), \
             patch.object(az, '_full_analysis') as fa:
            az._signal_lost_tick()
        fa.assert_not_called()

    def test_signal_lost_tick_respects_narrow_scan_cadence(self):
        az = _make_analyzer()
        az._transition('SIGNAL_LOST')
        az._last_narrow_scan = time.monotonic()   # Tier 2 not yet due
        with patch.object(az, '_noise_check', return_value='SIGNAL_LOST'), \
             patch.object(az, '_phase_search') as ps, \
             patch.object(az, '_fast_scan', return_value=False), \
             patch.object(az, '_full_analysis', return_value='SIGNAL_LOST'):
            az._signal_lost_tick()
        ps.assert_not_called()


# ---------------------------------------------------------------------------
# Result ring buffer
# ---------------------------------------------------------------------------

class TestResultBuffer:
    def test_buffer_initially_empty(self):
        assert _make_analyzer().drain_results() == []

    def test_publish_appends_to_buffer(self):
        az = _make_analyzer()
        r = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        az._publish(r)
        assert az.drain_results() == [r]

    def test_drain_clears_the_buffer(self):
        """Each collection cycle must average a disjoint set of results — a
        non-draining read would re-average the previous minute's tail."""
        az = _make_analyzer()
        az._publish(AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True))
        az.drain_results()
        assert az.drain_results() == []

    def test_drain_does_not_affect_latest_result(self):
        az = _make_analyzer()
        r = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        az._publish(r)
        az.drain_results()
        assert az.latest_result() == r   # the meters keep their reading

    def test_buffer_bounded_to_600(self):
        az = _make_analyzer()
        r = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        for _ in range(700):
            az._publish(r)
        assert len(az.drain_results()) == 600

    def test_multiple_results_preserved_in_order(self):
        az = _make_analyzer()
        r1 = AnalysisResult(signal_dbm=-70.0, noise_dbm=-90.0, snr=20.0, locked=True)
        r2 = AnalysisResult(signal_dbm=-75.0, noise_dbm=-92.0, snr=17.0, locked=False)
        az._publish(r1)
        az._publish(r2)
        drained = az.drain_results()
        assert drained[0] is r1
        assert drained[1] is r2


# ---------------------------------------------------------------------------
# _full_analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def test_strong_signal_transitions_to_locked(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'

    def test_strong_signal_proposes_locked(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        assert az._full_analysis() == 'LOCKED'

    def test_strong_signal_sets_phases_valid(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._phases_valid is True

    def test_strong_signal_publishes_locked_result(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        result = az.latest_result()
        assert result is not None
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_flat_noise_stays_searching(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._full_analysis)
        assert az._state == 'SEARCHING'

    def test_flat_noise_publishes_unlocked_result(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._full_analysis)
        result = az.latest_result()
        assert result is not None
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_no_data_returns_without_publishing(self):
        az = _make_analyzer()
        az._pipeline.wait_for_data.return_value = False
        _step(az, az._full_analysis)
        assert az.latest_result() is None

    def test_stores_peak_and_noise_phase_on_lock(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=5)
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        spp = SAMPLE_RATE / PULSE_RATE
        assert 0 <= az._peak_phase  < spp
        assert 0 <= az._noise_phase < spp

    def test_full_analysis_silent_when_signal_lost_no_lock(self):
        """From SIGNAL_LOST, a failed FFT fit does not overwrite the noise-check result."""
        az = _make_analyzer()
        # Lock, then force SIGNAL_LOST
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        az._transition('SIGNAL_LOST')
        # Publish a known noise-check result
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._noise_check)
        noise_result = az.latest_result()
        assert noise_result is not None and noise_result.locked is False
        # Now run full_analysis with flat noise — should NOT overwrite
        _step(az, az._full_analysis)
        assert az.latest_result() == noise_result


# ---------------------------------------------------------------------------
# _quick_check
# ---------------------------------------------------------------------------

class TestQuickCheck:
    def _locked_analyzer(self) -> ContinuousAnalyzer:
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        return az

    def test_strong_signal_stays_locked(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._quick_check)
        assert az._state == 'LOCKED'

    def test_strong_signal_publishes_locked_result(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._quick_check)
        assert az.latest_result().locked is True

    def test_single_failure_does_not_lose_lock(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        _step(az, az._quick_check)
        assert az._state == 'LOCKED'

    def test_silence_loses_lock_after_consecutive_failures(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            _step(az, az._quick_check)
        assert az._state == 'SIGNAL_LOST'

    def test_silence_publishes_unlocked_result(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            _step(az, az._quick_check)
        result = az.latest_result()
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_phases_valid_preserved_after_signal_lost(self):
        """_phases_valid stays True so SIGNAL_LOST can reuse stored phases."""
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            _step(az, az._quick_check)
        assert az._state == 'SIGNAL_LOST'
        assert az._phases_valid is True

    def test_recovery_resets_failure_count(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        _step(az, az._quick_check)
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._quick_check)
        assert az._consecutive_low_snr == 0
        assert az._state == 'LOCKED'

    def test_no_data_returns_without_state_change(self):
        az = self._locked_analyzer()
        az._pipeline.wait_for_data.return_value = False
        _step(az, az._quick_check)
        assert az._state == 'LOCKED'

    def test_hysteresis_acquire_threshold_exceeds_lose_threshold(self):
        """Hysteresis contract: acquiring lock requires higher SNR than losing it."""
        assert ContinuousAnalyzer.LOCK_ACQUIRE_SNR > ContinuousAnalyzer.LOCK_LOSE_SNR

    def test_lose_lock_count_requires_multiple_failures(self):
        """Debounce contract: at least two consecutive failures required to lose lock."""
        assert ContinuousAnalyzer.LOSE_LOCK_COUNT >= 2


# ---------------------------------------------------------------------------
# _noise_check  (SIGNAL_LOST state)
# ---------------------------------------------------------------------------

def _signal_lost_analyzer() -> ContinuousAnalyzer:
    """Analyzer that was LOCKED and has transitioned to SIGNAL_LOST."""
    az = _make_analyzer()
    az._pipeline.get_snapshot.return_value = _pulse_audio()
    _step(az, az._full_analysis)
    assert az._state == 'LOCKED'
    az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
    for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
        _step(az, az._quick_check)
    assert az._state == 'SIGNAL_LOST'
    return az


class TestNoiseCheck:
    def test_noise_check_publishes_noise_only_when_signal_absent(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._noise_check)
        result = az.latest_result()
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_noise_check_relocks_when_signal_returns(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._noise_check)
        assert az._state == 'LOCKED'

    def test_noise_check_publishes_locked_result_on_recovery(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._noise_check)
        result = az.latest_result()
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_noise_check_no_data_returns_without_change(self):
        az = _signal_lost_analyzer()
        az._pipeline.wait_for_data.return_value = False
        _step(az, az._noise_check)
        assert az._state == 'SIGNAL_LOST'


# ---------------------------------------------------------------------------
# _phase_search  (SIGNAL_LOST narrow scan)
# ---------------------------------------------------------------------------

class TestPhaseSearch:
    def test_phase_search_relocks_on_signal_at_stored_phase(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        assert _step(az, az._phase_search) == 'LOCKED'
        assert az._state == 'LOCKED'

    def test_phase_search_publishes_locked_result_on_recovery(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._phase_search)
        result = az.latest_result()
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_phase_search_proposes_current_state_without_signal(self):
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        assert _step(az, az._phase_search) == 'SIGNAL_LOST'
        assert az._state == 'SIGNAL_LOST'

    def test_phase_search_does_not_publish_on_failure(self):
        """On failure _phase_search() must not overwrite the noise-check result."""
        az = _signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._noise_check)
        before = az.latest_result()
        _step(az, az._phase_search)
        assert az.latest_result() == before

    def test_phase_search_finds_signal_at_offset_phase(self):
        """Signal shifted by 2 samples should still trigger re-lock."""
        az = _signal_lost_analyzer()
        spp_int   = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        new_phase = int(az._peak_phase + 2) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        assert _step(az, az._phase_search) == 'LOCKED'
        assert az._state == 'LOCKED'
        assert az._peak_phase == new_phase

    def test_phase_search_updates_noise_phase_independently(self):
        """Noise phase is searched within its own radius, not co-moved with the signal."""
        az = _signal_lost_analyzer()
        spp_int        = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        original_noise = int(az._noise_phase)
        new_peak       = int(az._peak_phase + 2) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_peak)
        _step(az, az._phase_search)
        assert az._state == 'LOCKED'
        # Noise phase must land within the search radius of where it started,
        # confirming it was searched independently rather than shifted by the
        # signal delta.
        r     = ContinuousAnalyzer.PHASE_SEARCH_RADIUS
        delta = min(
            abs(az._noise_phase - original_noise),
            spp_int - abs(az._noise_phase - original_noise),
        )
        assert delta <= r

    def test_phase_search_no_data_proposes_current_state(self):
        az = _signal_lost_analyzer()
        az._pipeline.wait_for_data.return_value = False
        assert _step(az, az._phase_search) == 'SIGNAL_LOST'
        assert az._state == 'SIGNAL_LOST'

    def test_phase_search_radius_contract(self):
        """Search radius must cover at least a few samples of phase drift."""
        assert ContinuousAnalyzer.PHASE_SEARCH_RADIUS >= 5

    def test_latest_signal_correction_initial_zero(self):
        assert _make_analyzer().latest_signal_correction() == 0

    def test_phase_search_records_signal_offset(self):
        """Correction is updated to the winning offset when phase_search finds the signal."""
        az = _signal_lost_analyzer()
        spp_int   = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        new_phase = int(az._peak_phase + 3) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        _step(az, az._phase_search)
        assert az._state == 'LOCKED'
        assert az.latest_signal_correction() == 3

    def test_phase_search_updates_correction_even_without_relock(self):
        """Correction field is written on every phase_search call, not only on relock."""
        az = _signal_lost_analyzer()
        az._latest_signal_correction = 999   # sentinel — must be overwritten
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._phase_search)
        assert az._state == 'SIGNAL_LOST'
        assert az.latest_signal_correction() != 999

    def test_latest_noise_correction_initial_zero(self):
        assert _make_analyzer().latest_noise_correction() == 0

    def test_phase_search_records_noise_offset_independently_of_signal(self):
        """Noise correction reflects the noise scan's own winning offset, not the signal's."""
        az = _signal_lost_analyzer()
        spp_int   = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        new_phase = int(az._peak_phase + 3) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        _step(az, az._phase_search)
        assert az._state == 'LOCKED'
        # Noise data is random, so the winning offset isn't predictable, but it must
        # be a real result of the scan: within the search radius, and not simply
        # mirroring the (different, known) signal offset of 3.
        r = ContinuousAnalyzer.PHASE_SEARCH_RADIUS
        assert -r <= az.latest_noise_correction() <= r

    def test_phase_search_updates_noise_correction_even_without_relock(self):
        """Noise correction is written on every phase_search call, not only on relock."""
        az = _signal_lost_analyzer()
        az._latest_noise_correction = 999   # sentinel — must be overwritten
        az._pipeline.get_snapshot.return_value = _noise_audio()
        _step(az, az._phase_search)
        assert az._state == 'SIGNAL_LOST'
        assert az.latest_noise_correction() != 999

    def test_signal_lost_refine_interval_much_longer_than_search(self):
        """FFT fallback should be infrequent compared to the cheap narrow scan."""
        assert ContinuousAnalyzer.SIGNAL_LOST_REFINE >= 5 * ContinuousAnalyzer.SEARCH_INTERVAL


class TestPhaseSearchTrackingThreshold:
    """Acquire/track hysteresis: acquiring a lock from SIGNAL_LOST demands
    LOCK_ACQUIRE_SNR, but tracking drift while LOCKED only requires
    LOCK_LOSE_SNR — otherwise signals in the 2–6 dB band would hold lock
    without being able to follow drift, guaranteeing eventual loss."""

    WEAK_AMPLITUDE = 85   # vs ~50 mean-abs noise → SNR ≈ 5 dB, between the thresholds

    def test_weak_signal_tracks_drift_while_locked(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        moved_phase = int(az._peak_phase) + 2
        az._pipeline.get_snapshot.return_value = _pulse_audio(
            amplitude=self.WEAK_AMPLITUDE, phase=moved_phase)
        _step(az, az._phase_search)
        assert az._state == 'LOCKED'
        assert az._peak_phase == moved_phase

    def test_weak_signal_does_not_acquire_from_signal_lost(self):
        az = _signal_lost_analyzer()
        original_phase = int(az._peak_phase)
        az._pipeline.get_snapshot.return_value = _pulse_audio(
            amplitude=self.WEAK_AMPLITUDE, phase=original_phase + 2)
        _step(az, az._phase_search)
        assert az._state == 'SIGNAL_LOST'
        assert az._peak_phase == original_phase


class TestScanPhaseHysteresis:
    """The incumbent phase keeps its place unless a challenger beats it by
    PHASE_MOVE_MARGIN, so phases (and the correction indicators) don't dance
    on measurement noise."""

    def _two_train_data(self, phase_a: int, amp_a: int, phase_b: int, amp_b: int) -> np.ndarray:
        """Silent background with two interleaved pulse trains at fixed phases."""
        data = np.zeros(SAMPLE_RATE, dtype=np.int32)
        spp = SAMPLE_RATE / PULSE_RATE
        for i in range(int(SAMPLE_RATE / spp)):
            base = round(i * spp)
            for phase, amp in ((phase_a, amp_a), (phase_b, amp_b)):
                pos = base + phase
                if pos + 3 < SAMPLE_RATE:
                    data[pos:pos + 3] = amp
        return data

    def test_marginally_louder_challenger_does_not_move_the_phase(self):
        az = _make_analyzer()
        data = self._two_train_data(30, 10000, 35, 10300)   # +3 % — inside the margin
        phase, offset = az._scan_phase(data, center=30, anchor=100, minimize=False)
        assert (phase, offset) == (30, 0)

    def test_clearly_louder_challenger_moves_the_phase(self):
        az = _make_analyzer()
        data = self._two_train_data(30, 10000, 35, 11000)   # +10 % — beats the margin
        phase, offset = az._scan_phase(data, center=30, anchor=100, minimize=False)
        assert (phase, offset) == (35, 5)

    def test_noise_scan_keeps_incumbent_on_flat_noise(self):
        """The noise scan picks the quietest of 21 noisy measurements; without the
        deadband the winner is essentially random every call and the NF correction
        indicator dances.  On featureless noise it must stay put."""
        az = _make_analyzer()
        data = np.abs(_noise_audio().astype(np.int32))
        phase, offset = az._scan_phase(data, center=60, anchor=5, minimize=True)
        assert (phase, offset) == (60, 0)


# ---------------------------------------------------------------------------
# Snapshot phase alignment (regression)
# ---------------------------------------------------------------------------

class _AlignedStreamPipeline:
    """Reproduces AudioPipeline snapshot semantics over a synthetic stream: the
    window ends at the chunk-quantised tail, adjusted down to a multiple of
    `align`.  Honouring `align` is the behaviour under test — with align=1 the
    window's phase origin moves with the tail and stored phases go stale."""

    CHUNK = 512

    def __init__(self, stream: np.ndarray, start_chunks: int = 40):
        self._stream = stream
        self.chunks = start_chunks

    @property
    def total_samples(self) -> int:
        """Audio clock: advancing `chunks` advances simulated time, as real capture does."""
        return self.chunks * self.CHUNK

    def wait_for_data(self, n_samples: int, timeout: float | None = None) -> bool:
        return self.chunks * self.CHUNK >= n_samples

    def get_snapshot(self, n_samples: int, align: int = 1) -> np.ndarray:
        tail = self.chunks * self.CHUNK
        end = tail - tail % align
        return self._stream[max(0, end - n_samples):end]


class TestShortestPhaseMove:
    """Phase differences must fold to the shortest equivalent move, both directions."""

    def test_small_forward_move_is_unchanged(self):
        assert _make_analyzer()._shortest_phase_move(10) == pytest.approx(10)

    def test_small_backward_move_is_unchanged(self):
        assert _make_analyzer()._shortest_phase_move(-10) == pytest.approx(-10)

    def test_wrap_forward_reads_as_short_backward_move(self):
        """395 -> 5 is +10 forward, not -390 backward."""
        az = _make_analyzer()
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        assert az._shortest_phase_move(-(period - 10)) == pytest.approx(10)

    def test_wrap_backward_reads_as_short_forward_move(self):
        az = _make_analyzer()
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        assert az._shortest_phase_move(period - 10) == pytest.approx(-10)

    def test_result_always_within_half_a_period(self):
        az = _make_analyzer()
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        for raw in range(-2 * period, 2 * period, 7):
            assert -period / 2 <= az._shortest_phase_move(raw) <= period / 2


class TestPredictedPhases:
    """Phases are projected forward by the drift rate rather than held constant."""

    def _analyzer_at(self, rate: float, peak: int = 50, noise: int = 100):
        az = _make_analyzer()
        az._peak_phase, az._noise_phase = float(peak), float(noise)
        az._phase_drift_rate = rate
        az._phases_measured_at = 0.0
        return az

    def test_zero_drift_predicts_the_stored_phase(self):
        az = self._analyzer_at(0.0)
        az._pipeline.total_samples = 10 * SAMPLE_RATE
        assert az._predicted_phases() == (50, 100)

    def test_positive_drift_moves_phases_forward(self):
        az = self._analyzer_at(5.0)
        az._pipeline.total_samples = 2 * SAMPLE_RATE      # 2 s of audio
        assert az._predicted_phases() == (60, 110)        # +10 samples each

    def test_negative_drift_moves_phases_backward(self):
        """A grid running the other way must be tracked just as well."""
        az = self._analyzer_at(-5.0)
        az._pipeline.total_samples = 2 * SAMPLE_RATE
        assert az._predicted_phases() == (40, 90)         # -10 samples each

    def test_negative_prediction_wraps_into_range(self):
        az = self._analyzer_at(-5.0, peak=2, noise=4)
        az._pipeline.total_samples = 2 * SAMPLE_RATE      # -10 would go below zero
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        peak, noise = az._predicted_phases()
        assert (peak, noise) == (period - 8, period - 6)
        assert 0 <= peak < period and 0 <= noise < period

    def test_both_phases_shift_together(self):
        """Signal and noise phases are anchored to the same power cycle."""
        az = self._analyzer_at(7.0)
        az._pipeline.total_samples = 3 * SAMPLE_RATE
        peak, noise = az._predicted_phases()
        assert (peak - 50) == (noise - 100)

    def test_stalled_audio_freezes_the_prediction(self):
        """The audio clock, not the wall clock: no new audio means no new drift."""
        az = self._analyzer_at(20.0)
        az._pipeline.total_samples = 0
        time.sleep(0.05)
        assert az._predicted_phases() == (50, 100)


class TestPhaseHoldTimeout:
    """SIGNAL_LOST gives up and returns to SEARCHING once the stored phases go stale.

    Keeps the state machine and the display in step: trigger_phase() reports HOLD
    purely on _phases_valid, so an unbounded SIGNAL_LOST would claim a synchronised
    sweep indefinitely — all night, if the arc quits at dusk.
    """

    def _lost(self, phase_age_s):
        """Analyzer sitting in SIGNAL_LOST with phases measured `phase_age_s` ago."""
        az = _make_analyzer()
        az._transition(AnalyzerState.LOCKED)      # marks phases valid
        az._transition(AnalyzerState.SIGNAL_LOST)
        az._phases_measured_at = 0.0
        az._pipeline.total_samples = int(phase_age_s * SAMPLE_RATE)
        # Tier methods can't measure anything from a MagicMock pipeline, so they
        # leave the state alone and only the timeout can move it.
        az._capture = lambda *a, **kw: None
        return az

    def test_holds_while_phases_are_fresh(self):
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT - 5.0)
        az._signal_lost_tick()
        assert az._state == AnalyzerState.SIGNAL_LOST
        assert az._phases_valid is True

    def test_falls_back_to_searching_once_stale(self):
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT + 5.0)
        az._signal_lost_tick()
        assert az._state == AnalyzerState.SEARCHING

    def test_giving_up_invalidates_the_phases(self):
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT + 5.0)
        az._signal_lost_tick()
        assert az._phases_valid is False

    def test_display_reports_free_after_giving_up(self):
        """The parity that motivates this: no state claiming sync it cannot deliver."""
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT + 5.0)
        assert az.trigger_phase()[1] is TriggerSync.HOLD
        az._signal_lost_tick()
        assert az.trigger_phase() == (0, TriggerSync.FREE)

    def test_stalled_audio_does_not_age_the_phases(self):
        """Measured on the audio clock: no audio captured means no drift happened,
        so the stored phases are exactly as good as when they were taken."""
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT + 600.0)
        az._pipeline.total_samples = 0            # device stalled; clock frozen at 0
        az._signal_lost_tick()
        assert az._state == AnalyzerState.SIGNAL_LOST
        assert az._phases_valid is True

    def test_searching_re_entry_clears_the_fit_history(self):
        az = self._lost(ContinuousAnalyzer.PHASE_HOLD_TIMEOUT + 5.0)
        az._phase_history.append((0.0, 50.0))
        az._signal_lost_tick()
        assert len(az._phase_history) == 0


class TestFitDriftRate:
    """The least-squares slope itself, independent of the analyzer."""

    def test_perfect_line_recovers_its_slope_exactly(self):
        history = [(t * 0.6, 100.0 + 7.5 * t * 0.6) for t in range(10)]
        assert fit_drift_rate(history, MIN_FIT) == pytest.approx(7.5)

    def test_negative_slope(self):
        history = [(t * 0.6, 100.0 - 7.5 * t * 0.6) for t in range(10)]
        assert fit_drift_rate(history, MIN_FIT) == pytest.approx(-7.5)

    def test_flat_history_reads_as_no_drift(self):
        assert fit_drift_rate([(t * 0.6, 42.0) for t in range(10)], MIN_FIT) == pytest.approx(0.0)

    def test_too_few_points_is_none(self):
        assert fit_drift_rate([(0.0, 1.0), (0.6, 2.0)], MIN_FIT) is None

    def test_min_points_is_configurable(self):
        history = [(0.0, 1.0), (0.6, 2.0)]
        assert fit_drift_rate(history, min_points=2) == pytest.approx(1 / 0.6)

    def test_all_points_at_one_instant_is_none(self):
        """A stalled audio clock gives a vertical line, which has no slope."""
        assert fit_drift_rate([(4.0, 1.0), (4.0, 2.0), (4.0, 3.0)], MIN_FIT) is None

    def test_averages_zero_mean_noise_away(self):
        """The whole reason for fitting rather than taking the last two points."""
        rng = np.random.default_rng(7)
        times = np.arange(10) * 0.6
        noisy = [(t, 50.0 + 9.0 * t + rng.uniform(-0.5, 0.5)) for t in times]
        assert fit_drift_rate(noisy, MIN_FIT) == pytest.approx(9.0, abs=0.35)

    def test_beats_the_two_point_slope_it_replaced(self):
        """Same quantised points, both estimators: the fit must be closer to truth."""
        true_rate = 9.0
        times = np.arange(10) * 0.6
        quantised = [(t, float(round(50.0 + true_rate * t))) for t in times]
        two_point = (quantised[-1][1] - quantised[-2][1]) / (times[-1] - times[-2])
        fitted = fit_drift_rate(quantised, MIN_FIT)
        assert abs(fitted - true_rate) < abs(two_point - true_rate)


class TestDriftRateLearning:
    """_record_phase_measurement maintains the history the rate is fitted through."""

    def _analyzer(self):
        az = _make_analyzer()
        az._peak_phase, az._noise_phase = 50.0, 100.0
        az._phases_measured_at = 0.0
        return az

    def _feed(self, az, true_rate, steps=40, start_phase=50, interval=None):
        """Feed whole-sample phase measurements drifting at true_rate."""
        interval = interval or ContinuousAnalyzer.REFINE_INTERVAL
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        at = 0.0
        for _ in range(steps):
            at += interval
            phase = int(round(start_phase + true_rate * at)) % period
            az._record_phase_measurement(phase, 100, measured_at=at)
        return az

    def test_starts_at_zero(self):
        assert _make_analyzer().phase_drift_rate() == 0.0

    def test_no_rate_until_min_points_are_available(self):
        az = self._analyzer()
        for i in range(ContinuousAnalyzer.DRIFT_FIT_MIN_POINTS - 1):
            az._record_phase_measurement(50 + 6 * (i + 1), 100, measured_at=(i + 1) * 1.0)
        assert az.phase_drift_rate() == 0.0

    def test_fits_a_rate_once_min_points_arrive(self):
        az = self._analyzer()
        for i in range(ContinuousAnalyzer.DRIFT_FIT_MIN_POINTS):
            az._record_phase_measurement(50 + 6 * (i + 1), 100, measured_at=(i + 1) * 1.0)
        assert az.phase_drift_rate() == pytest.approx(6.0)

    @pytest.mark.parametrize('true_rate', [6.0, -6.0, 15.0, -15.0, 1.5, -1.5])
    def test_converges_to_the_true_rate(self, true_rate):
        """Both directions, through whole-sample phases.

        The old per-refine estimator was bounded by 0.5 / REFINE_INTERVAL ~= 0.83
        samples/s, because it divided one quantisation error by one interval.  The fit
        spans (DRIFT_FIT_POINTS - 1) intervals = 5.4 s, over which a half-sample error
        at each end implies at most 1.0 / 5.4 = 0.185 samples/s.

        The 0.15 bound is measured rather than assumed: swept across 1201 drift rates
        from -30 to +30, this construction peaks at 0.146 (at +26.6) and averages
        0.042.  Rounding is deterministic per rate, so some rates genuinely fare worse
        than others — hence a bound near the measured maximum rather than the mean.
        """
        az = self._feed(self._analyzer(), true_rate)
        assert az.phase_drift_rate() == pytest.approx(true_rate, abs=0.15)

    def test_unwraps_across_the_period_boundary(self):
        """Measurements fold into [0, period); a line cannot be fitted to a sawtooth,
        so the history has to accumulate the true move instead."""
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        az = self._feed(self._analyzer(), 40.0, steps=60, start_phase=period - 20)
        assert az.phase_drift_rate() == pytest.approx(40.0, abs=0.15)

    def test_a_long_gap_restarts_the_history(self):
        """Across a gap the phase may have wrapped any number of times, so the old
        points must not be fitted against the new one."""
        az = self._feed(self._analyzer(), 10.0)
        before = az.phase_drift_rate()
        assert before == pytest.approx(10.0, abs=0.15)
        gap_at = az._phases_measured_at + ContinuousAnalyzer.MAX_PHASE_HISTORY_GAP + 1.0
        az._record_phase_measurement(123, 100, measured_at=gap_at)
        assert len(az._phase_history) == 1
        # The rate itself survives the gap — the grid kept drifting while we were away.
        assert az.phase_drift_rate() == pytest.approx(before)

    def test_reset_keeps_the_rate_but_drops_the_points(self):
        az = self._feed(self._analyzer(), 10.0)
        rate = az.phase_drift_rate()
        az._reset_phase_history()
        assert len(az._phase_history) == 0
        assert az.phase_drift_rate() == pytest.approx(rate)

    def test_leaving_locked_clears_the_history(self):
        az = self._feed(self._analyzer(), 10.0)
        az._state = AnalyzerState.LOCKED
        az._transition(AnalyzerState.SIGNAL_LOST)
        assert len(az._phase_history) == 0

    def test_history_is_bounded(self):
        az = self._feed(self._analyzer(), 6.0, steps=200)
        assert len(az._phase_history) == ContinuousAnalyzer.DRIFT_FIT_POINTS

    @pytest.mark.parametrize('sign', [1, -1])
    def test_rate_is_clamped_to_a_physically_plausible_range(self, sign):
        """A scan latching onto the wrong peak must not compound into a runaway."""
        az = self._analyzer()
        for i in range(50):
            az._record_phase_measurement(50 + sign * 300 * (i % 2), 100,
                                         measured_at=(i + 1) * 0.6)
        assert abs(az.phase_drift_rate()) <= ContinuousAnalyzer.MAX_PHASE_DRIFT_RATE

    def test_measurement_updates_the_stored_phases_and_timestamp(self):
        az = self._analyzer()
        az._record_phase_measurement(77, 99, measured_at=2.5)
        assert (az._peak_phase, az._noise_phase, az._phases_measured_at) == (77.0, 99.0, 2.5)


class TestGridFrequency:
    """The drift estimate doubles as a line-frequency measurement."""

    def test_zero_drift_reads_nominal(self):
        assert _make_analyzer().grid_frequency_hz() == pytest.approx(60.0)

    def test_positive_drift_reads_below_nominal(self):
        """Pulses arriving later means fewer per second, so the grid is slow."""
        az = _make_analyzer()
        az._phase_drift_rate = SAMPLE_RATE * (PULSE_RATE - 119.95) / 119.95
        assert az.grid_frequency_hz() == pytest.approx(59.975, abs=0.001)

    def test_negative_drift_reads_above_nominal(self):
        az = _make_analyzer()
        az._phase_drift_rate = SAMPLE_RATE * (PULSE_RATE - 120.05) / 120.05
        assert az.grid_frequency_hz() == pytest.approx(60.025, abs=0.001)


class TestDriftRateLifecycle:
    def test_cold_acquisition_starts_from_zero_drift(self):
        az = _make_analyzer()
        az._phase_drift_rate = 12.0          # stale value with no prior lock
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        assert az.phase_drift_rate() == 0.0

    def test_reacquisition_after_outage_keeps_the_drift_estimate(self):
        """The grid kept drifting while our signal was gone, so the old rate is a
        better starting guess than zero."""
        az = _signal_lost_analyzer()
        az._phase_drift_rate = 4.0
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        assert az.phase_drift_rate() == pytest.approx(4.0)


class TestSnapshotPhaseAlignment:
    """Regression for the moving-phase-origin bug: 512-sample chunks are not a
    whole number of 133.33-sample pulse periods, so an unaligned snapshot's
    phase origin cycles through 25 offsets (multiples of 5.33 samples) as the
    tail advances — and phases learned in one snapshot point at the wrong
    samples in the next.  With aligned capture, lock must survive arbitrary
    tail movement on a perfect, drift-free signal."""

    def _pulse_stream(self, seconds: int = 30, pulse_rate: float = PULSE_RATE) -> np.ndarray:
        n = seconds * SAMPLE_RATE
        spp = SAMPLE_RATE / pulse_rate
        data = np.random.default_rng(7).integers(-100, 101, size=n).astype(np.int16)
        for i in range(int(n / spp)):
            pos = 5 + round(i * spp)
            if pos + 3 < n:
                data[pos:pos + 3] = 20000
        return data

    def test_lock_survives_chunkwise_tail_movement(self):
        pipe = _AlignedStreamPipeline(self._pulse_stream())
        az = ContinuousAnalyzer(pipe, _make_config())
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        # 200 ms ticks advance ~6.25 chunks (6,6,6,7 pattern) — plus deliberate
        # jitter so many residues of the 25-chunk misalignment cycle are visited.
        for advance in [6, 6, 6, 7, 7, 6, 8, 5, 31, 6, 13, 6, 6, 7, 9, 25]:
            pipe.chunks += advance
            _step(az, az._quick_check)
            assert az._state == 'LOCKED'
            assert az.latest_result().snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    # The drift estimate needs ~20 refines (~12 s) to converge from a standing start,
    # so bias comparisons are made after it has settled.
    SETTLED = slice(30, None)

    def _run_locked_cadence(self, pulse_rate: float, refine_cycles: int = 60,
                            track_drift: bool = True):
        """Drive the LOCKED cadence over a drifting stream.

        Mirrors production timing: two _quick_check ticks then a _phase_search per
        REFINE_INTERVAL, with the pipeline advanced ~200 ms of audio each tick.
        With track_drift=False the analyzer never learns a rate, which reproduces
        the behaviour before drift tracking existed and gives the comparisons below
        a control to measure against.

        Returns (analyzer, mean published level per refine cycle).
        """
        pipe = _AlignedStreamPipeline(self._pulse_stream(seconds=60, pulse_rate=pulse_rate))
        az = ContinuousAnalyzer(pipe, _make_config())
        if not track_drift:
            # Demand more points than the history can ever hold, so no fit is ever
            # produced and the rate stays at its initial zero — reproducing the
            # behaviour from before drift tracking existed.
            az.DRIFT_FIT_MIN_POINTS = 10 ** 9
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        per_cycle = []
        for _ in range(refine_cycles):
            levels = []
            for i, advance in enumerate([6, 6, 7]):
                pipe.chunks += advance
                _step(az, az._quick_check if i < 2 else az._phase_search)
                assert az._state == 'LOCKED'
                result = az.latest_result()
                if result is not None and result.locked:
                    levels.append(result.signal_dbm)
            per_cycle.append(np.mean(levels) if levels else np.nan)
        return az, np.array(per_cycle)

    @pytest.mark.parametrize('pulse_rate', [120.05, 119.95])
    def test_drift_tracking_removes_most_of_the_level_bias(self, pulse_rate):
        """The reason drift tracking exists, measured against an untracked control.

        Untracked, a drifting signal reads several dB low: the phase goes stale
        between refines, and worse, the sampling grid walks away from the pulses
        within each analysis window.  Tracking fixes both — it predicts the phase
        forward and spaces the samples at the measured rate.

        A residual of about a dB remains and is expected: phases are whole samples,
        so the prediction is always up to half a sample off.  Closing that would
        need sub-sample interpolation.
        """
        _, reference = self._run_locked_cadence(PULSE_RATE)
        _, tracked   = self._run_locked_cadence(pulse_rate)
        _, untracked = self._run_locked_cadence(pulse_rate, track_drift=False)

        base = np.nanmean(reference[self.SETTLED])
        tracked_bias   = base - np.nanmean(tracked[self.SETTLED])
        untracked_bias = base - np.nanmean(untracked[self.SETTLED])

        assert untracked_bias > 4.0, (
            f'control should show the problem, saw only {untracked_bias:.2f} dB')
        assert tracked_bias < 2.0, f'tracked bias still {tracked_bias:.2f} dB'
        assert tracked_bias < untracked_bias / 3

    @pytest.mark.parametrize('pulse_rate', [120.05, 119.95])
    def test_drift_estimate_converges_from_a_standing_start(self, pulse_rate):
        """Starting from zero, the rate must reach the truth in a reasonable time."""
        az, _ = self._run_locked_cadence(pulse_rate, refine_cycles=30)
        expected = SAMPLE_RATE * (PULSE_RATE - pulse_rate) / pulse_rate
        assert az.phase_drift_rate() == pytest.approx(expected, abs=1.0)

    @pytest.mark.parametrize('pulse_rate,expected_sign', [(120.05, -1), (119.95, 1)])
    def test_drift_rate_is_learned_with_the_right_sign(self, pulse_rate, expected_sign):
        """A grid running fast and one running slow must both be tracked."""
        pipe = _AlignedStreamPipeline(self._pulse_stream(pulse_rate=pulse_rate))
        az = ContinuousAnalyzer(pipe, _make_config())
        _step(az, az._full_analysis)
        for _ in range(30):
            for i, advance in enumerate([6, 6, 7]):
                pipe.chunks += advance
                _step(az, az._quick_check if i < 2 else az._phase_search)
        # Matching the sampling grid to a true rate r needs a slip of
        # sample_rate * (pulse_rate - r) / r -- negative when the grid runs fast.
        expected = SAMPLE_RATE * (PULSE_RATE - pulse_rate) / pulse_rate
        assert az.phase_drift_rate() * expected_sign > 0
        assert az.phase_drift_rate() == pytest.approx(expected, abs=1.5)
        assert az.grid_frequency_hz() == pytest.approx(pulse_rate / 2, abs=0.01)

    def test_lock_tracks_realistic_grid_drift(self):
        """A +0.05 Hz pulse-rate error (ordinary grid drift) slips the phase
        ~6.7 samples/s.  Emulating the LOCKED cadence — two quick checks then a
        refine per REFINE_INTERVAL — the analyzer must hold lock continuously
        for ~25 simulated seconds and keep every correction well inside
        PHASE_SEARCH_RADIUS."""
        pipe = _AlignedStreamPipeline(self._pulse_stream(pulse_rate=120.05))
        az = ContinuousAnalyzer(pipe, _make_config())
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        for _ in range(40):                      # 40 refine cycles ≈ 24 s
            for i, advance in enumerate([6, 6, 7]):   # 3 ticks ≈ REFINE_INTERVAL
                pipe.chunks += advance
                if i < 2:
                    _step(az, az._quick_check)
                else:
                    _step(az, az._phase_search)
                    assert abs(az.latest_signal_correction()) <= 6
                assert az._state == 'LOCKED'


# ---------------------------------------------------------------------------
# _fast_scan  (SIGNAL_LOST Tier-3a screening)
# ---------------------------------------------------------------------------

class TestFastScan:
    def test_fast_scan_returns_true_with_signal(self):
        az = _signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _pulse_audio(n=n)
        assert az._fast_scan() is True

    def test_fast_scan_returns_false_without_signal(self):
        az = _signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _noise_audio(n=n)
        assert az._fast_scan() is False

    def test_fast_scan_does_not_publish_or_change_state(self):
        az = _signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _pulse_audio(n=n)
        before = az.latest_result()
        az._fast_scan()
        assert az._state == 'SIGNAL_LOST'
        assert az.latest_result() == before

    def test_fast_scan_no_data_returns_false(self):
        az = _signal_lost_analyzer()
        az._pipeline.wait_for_data.return_value = False
        assert az._fast_scan() is False
        assert az._state == 'SIGNAL_LOST'

    def test_fast_scan_interval_shorter_than_signal_lost_refine(self):
        assert ContinuousAnalyzer.FAST_SCAN_INTERVAL < ContinuousAnalyzer.SIGNAL_LOST_REFINE

    def test_fast_scan_kernel_shorter_than_full_kernel(self):
        az = _make_analyzer()
        assert len(az._fast_kernel) < len(az._kernel)

    def test_fast_scan_pulses_less_than_scan_pulses(self):
        az = _make_analyzer()
        assert ContinuousAnalyzer.FAST_SCAN_PULSES < az._scan_pulses


class TestRunResilience:
    """_run() must not let a single bad tick silently kill the daemon thread —
    see the exception guard's comment for why that matters for an unattended
    monitor."""

    def test_run_survives_a_tick_that_raises(self):
        az = _make_analyzer()
        calls = []

        def flaky_searching_tick():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('simulated transient failure')
            az._stop.set()   # let the loop exit cleanly on the second pass
            return 0.0

        az._searching_tick = flaky_searching_tick
        az._run()   # would raise and never reach the second call if unguarded
        assert len(calls) == 2

    def test_run_logs_the_failure(self, caplog):
        az = _make_analyzer()
        calls = []

        def flaky_searching_tick():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError('boom')
            az._stop.set()
            return 0.0

        az._searching_tick = flaky_searching_tick
        with caplog.at_level('ERROR'):
            az._run()
        assert 'Analyzer tick failed' in caplog.text


class TestTriggerPhase:
    """The sync source for the scope display (buzz.scope).

    Everything the scope draws hangs off this returning the *predicted* phase, so
    these cover the projection and the three trust levels rather than just the
    happy path.
    """

    PHASE_ALIGN = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)   # 400

    def _tracking(self, state=AnalyzerState.LOCKED, peak=100.0,
                  drift=0.0, measured_at=0.0, elapsed_s=0.0):
        az = _make_analyzer()
        az._state = state
        az._phases_valid = True
        az._peak_phase = peak
        az._phase_drift_rate = drift
        az._phases_measured_at = measured_at
        az._pipeline.total_samples = int(elapsed_s * SAMPLE_RATE)
        return az

    def test_free_before_any_lock(self):
        az = _make_analyzer()
        assert az.trigger_phase() == (0, TriggerSync.FREE)

    def test_free_even_while_searching_with_a_stale_peak(self):
        """_peak_phase holds whatever the last FFT saw even when nothing locked, so
        validity has to gate the answer rather than the phase being non-zero."""
        az = _make_analyzer()
        az._peak_phase = 250.0
        assert az.trigger_phase() == (0, TriggerSync.FREE)

    def test_lock_when_locked(self):
        _, sync = self._tracking(state=AnalyzerState.LOCKED).trigger_phase()
        assert sync is TriggerSync.LOCK

    def test_hold_when_signal_lost(self):
        """The case that makes a fading signal reappear on the persistence display
        before the analyzer formally re-locks."""
        _, sync = self._tracking(state=AnalyzerState.SIGNAL_LOST).trigger_phase()
        assert sync is TriggerSync.HOLD

    def test_hold_when_searching_but_phases_still_valid(self):
        _, sync = self._tracking(state=AnalyzerState.SEARCHING).trigger_phase()
        assert sync is TriggerSync.HOLD

    def test_returns_stored_phase_when_nothing_has_drifted(self):
        phase, _ = self._tracking(peak=137.0).trigger_phase()
        assert phase == 137

    def test_projects_forward_at_the_drift_rate(self):
        phase, _ = self._tracking(peak=100.0, drift=10.0, elapsed_s=2.0).trigger_phase()
        assert phase == 120

    def test_projects_backward_on_negative_drift(self):
        """A grid running fast walks the phase the other way; the modulo must not
        turn that into a huge positive jump."""
        phase, _ = self._tracking(peak=100.0, drift=-10.0, elapsed_s=2.0).trigger_phase()
        assert phase == 80

    def test_wraps_on_the_phase_period(self):
        phase, _ = self._tracking(peak=390.0, drift=10.0, elapsed_s=2.0).trigger_phase()
        assert phase == 10

    def test_always_within_one_phase_period(self):
        for drift in (-60.0, -7.5, 0.0, 3.3, 60.0):
            for elapsed in (0.0, 0.7, 5.0, 60.0):
                phase, _ = self._tracking(peak=250.0, drift=drift,
                                          elapsed_s=elapsed).trigger_phase()
                assert 0 <= phase < self.PHASE_ALIGN

    def test_agrees_with_the_analyzers_own_prediction(self):
        """The property that actually matters: the scope must sync to the same place
        the analyzer samples.  If these two ever diverge, the trace would sit at a
        fixed offset from the pulses it claims to be showing."""
        for drift in (-25.0, 0.0, 12.5):
            for elapsed in (0.0, 1.3, 9.0):
                az = self._tracking(peak=173.0, drift=drift, elapsed_s=elapsed)
                assert az.trigger_phase()[0] == az._predicted_phases()[0]

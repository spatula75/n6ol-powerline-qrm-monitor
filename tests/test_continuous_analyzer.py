"""Tests for ContinuousAnalyzer state machine and DSP integration.

Tier methods (_full_analysis, _quick_check, _noise_check, _phase_search) measure
and publish, then return the state they propose; _transition() applies it.  The
_step() helper below mirrors what _run()'s tick methods do in production.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.analyzer import AnalysisResult, AnalyzerState, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline

SAMPLE_RATE = 16000
PULSE_RATE  = 120


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
    return ContinuousAnalyzer(pipeline, cfg)


def _step(az: ContinuousAnalyzer, tier_method) -> str:
    """Run one tier method and apply its proposed state, as _run's tick methods do."""
    proposed = tier_method()
    az._transition(proposed)
    return proposed


def _pulse_audio(n: int = SAMPLE_RATE, amplitude: int = 20000, phase: int = 5) -> np.ndarray:
    """1 second of int16 audio with a clear 120 pps pulse train."""
    data = np.random.default_rng(42).integers(50, 150, size=n, dtype=np.int16)
    spp  = SAMPLE_RATE / PULSE_RATE
    for i in range(int(n / spp)):
        pos = phase + int(i * spp)
        if pos + 3 < n:
            data[pos] = data[pos + 1] = data[pos + 2] = amplitude
    return data


def _noise_audio(n: int = SAMPLE_RATE) -> np.ndarray:
    """1 second of flat random noise with no pulse structure."""
    return np.random.default_rng(0).integers(50, 150, size=n, dtype=np.int16)


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
        spp_int   = int(SAMPLE_RATE / PULSE_RATE)
        new_phase = (az._peak_phase + 2) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        assert _step(az, az._phase_search) == 'LOCKED'
        assert az._state == 'LOCKED'
        assert az._peak_phase == new_phase

    def test_phase_search_updates_noise_phase_independently(self):
        """Noise phase is searched within its own radius, not co-moved with the signal."""
        az = _signal_lost_analyzer()
        spp_int        = int(SAMPLE_RATE / PULSE_RATE)
        original_noise = az._noise_phase
        new_peak       = (az._peak_phase + 2) % spp_int
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
        spp_int   = int(SAMPLE_RATE / PULSE_RATE)
        new_phase = (az._peak_phase + 3) % spp_int
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
        spp_int   = int(SAMPLE_RATE / PULSE_RATE)
        new_phase = (az._peak_phase + 3) % spp_int
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

    WEAK_AMPLITUDE = 160   # vs ~100 mean noise → SNR ≈ 4 dB, between the thresholds

    def test_weak_signal_tracks_drift_while_locked(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        _step(az, az._full_analysis)
        assert az._state == 'LOCKED'
        moved_phase = az._peak_phase + 2
        az._pipeline.get_snapshot.return_value = _pulse_audio(
            amplitude=self.WEAK_AMPLITUDE, phase=moved_phase)
        _step(az, az._phase_search)
        assert az._state == 'LOCKED'
        assert az._peak_phase == moved_phase

    def test_weak_signal_does_not_acquire_from_signal_lost(self):
        az = _signal_lost_analyzer()
        original_phase = az._peak_phase
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
            base = int(i * spp)
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

    def wait_for_data(self, n_samples: int, timeout: float | None = None) -> bool:
        return self.chunks * self.CHUNK >= n_samples

    def get_snapshot(self, n_samples: int, align: int = 1) -> np.ndarray:
        tail = self.chunks * self.CHUNK
        end = tail - tail % align
        return self._stream[max(0, end - n_samples):end]


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
        data = np.random.default_rng(7).integers(50, 150, size=n).astype(np.int16)
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

    def test_lock_tracks_realistic_mains_drift(self):
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

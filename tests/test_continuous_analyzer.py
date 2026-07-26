"""Tests for ContinuousAnalyzer state machine and DSP integration."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from buzz.analyzer import AnalysisResult, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline

SAMPLE_RATE = 16000
PULSE_RATE  = 120


def _make_config() -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.sample_rate   = SAMPLE_RATE
    cfg.audio.pulse_rate    = PULSE_RATE
    cfg.audio.input_device_name = 'Test'
    cfg.audio.duration      = 3
    return cfg


def _make_analyzer() -> ContinuousAnalyzer:
    cfg      = _make_config()
    pipeline = MagicMock(spec=AudioPipeline)
    pipeline.wait_for_data.return_value = True
    return ContinuousAnalyzer(pipeline, cfg)


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
# _full_analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def test_strong_signal_transitions_to_locked(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._state == 'LOCKED'

    def test_strong_signal_sets_phases_valid(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._phases_valid is True

    def test_strong_signal_publishes_locked_result(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        result = az.latest_result()
        assert result is not None
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_flat_noise_stays_searching(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._full_analysis()
        assert az._state == 'SEARCHING'

    def test_flat_noise_publishes_unlocked_result(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._full_analysis()
        result = az.latest_result()
        assert result is not None
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_no_data_returns_without_publishing(self):
        az = _make_analyzer()
        az._pipeline.wait_for_data.return_value = False
        az._full_analysis()
        assert az.latest_result() is None

    def test_stores_peak_and_noise_phase_on_lock(self):
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=5)
        az._full_analysis()
        assert az._state == 'LOCKED'
        spp = SAMPLE_RATE / PULSE_RATE
        assert 0 <= az._peak_phase  < spp
        assert 0 <= az._noise_phase < spp

    def test_full_analysis_silent_when_signal_lost_no_lock(self):
        """From SIGNAL_LOST, a failed FFT fit does not overwrite the noise-check result."""
        az = _make_analyzer()
        # Lock, then force SIGNAL_LOST
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        az._state = 'SIGNAL_LOST'
        # Publish a known noise-check result
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._noise_check()
        noise_result = az.latest_result()
        assert noise_result is not None and noise_result.locked is False
        # Now run full_analysis with flat noise — should NOT overwrite
        az._full_analysis()
        assert az.latest_result() == noise_result


# ---------------------------------------------------------------------------
# _quick_check
# ---------------------------------------------------------------------------

class TestQuickCheck:
    def _locked_analyzer(self) -> ContinuousAnalyzer:
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._state == 'LOCKED'
        return az

    def test_strong_signal_stays_locked(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._quick_check()
        assert az._state == 'LOCKED'

    def test_strong_signal_publishes_locked_result(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._quick_check()
        assert az.latest_result().locked is True

    def test_single_failure_does_not_lose_lock(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        az._quick_check()
        assert az._state == 'LOCKED'

    def test_silence_loses_lock_after_consecutive_failures(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        assert az._state == 'SIGNAL_LOST'

    def test_silence_publishes_unlocked_result(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        result = az.latest_result()
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_phases_valid_preserved_after_signal_lost(self):
        """_phases_valid stays True so SIGNAL_LOST can reuse stored phases."""
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        assert az._state == 'SIGNAL_LOST'
        assert az._phases_valid is True

    def test_recovery_resets_failure_count(self):
        az = self._locked_analyzer()
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        az._quick_check()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._quick_check()
        assert az._consecutive_low_snr == 0
        assert az._state == 'LOCKED'

    def test_no_data_returns_without_state_change(self):
        az = self._locked_analyzer()
        az._pipeline.wait_for_data.return_value = False
        az._quick_check()
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

class TestNoiseCheck:
    def _signal_lost_analyzer(self) -> ContinuousAnalyzer:
        """Analyzer that was LOCKED and has transitioned to SIGNAL_LOST."""
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._state == 'LOCKED'
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        assert az._state == 'SIGNAL_LOST'
        return az

    def test_noise_check_publishes_noise_only_when_signal_absent(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._noise_check()
        result = az.latest_result()
        assert result.locked is False
        assert result.snr == 0.0
        assert result.signal_dbm == result.noise_dbm

    def test_noise_check_relocks_when_signal_returns(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._noise_check()
        assert az._state == 'LOCKED'

    def test_noise_check_publishes_locked_result_on_recovery(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._noise_check()
        result = az.latest_result()
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_noise_check_no_data_returns_without_change(self):
        az = self._signal_lost_analyzer()
        az._pipeline.wait_for_data.return_value = False
        az._noise_check()
        assert az._state == 'SIGNAL_LOST'


# ---------------------------------------------------------------------------
# _phase_search  (SIGNAL_LOST narrow scan)
# ---------------------------------------------------------------------------

class TestPhaseSearch:
    def _signal_lost_analyzer(self) -> ContinuousAnalyzer:
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._state == 'LOCKED'
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        assert az._state == 'SIGNAL_LOST'
        return az

    def test_phase_search_relocks_on_signal_at_stored_phase(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        assert az._phase_search() is True
        assert az._state == 'LOCKED'

    def test_phase_search_publishes_locked_result_on_recovery(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._phase_search()
        result = az.latest_result()
        assert result.locked is True
        assert result.snr >= ContinuousAnalyzer.LOCK_ACQUIRE_SNR

    def test_phase_search_returns_false_without_signal(self):
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        assert az._phase_search() is False
        assert az._state == 'SIGNAL_LOST'

    def test_phase_search_does_not_publish_on_failure(self):
        """On failure _phase_search() must not overwrite the noise-check result."""
        az = self._signal_lost_analyzer()
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._noise_check()
        before = az.latest_result()
        az._phase_search()
        assert az.latest_result() == before

    def test_phase_search_finds_signal_at_offset_phase(self):
        """Signal shifted by 2 samples should still trigger re-lock."""
        az = self._signal_lost_analyzer()
        spp_int   = int(SAMPLE_RATE / PULSE_RATE)
        new_phase = (az._peak_phase + 2) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        assert az._phase_search() is True
        assert az._state == 'LOCKED'
        assert az._peak_phase == new_phase

    def test_phase_search_updates_noise_phase_independently(self):
        """Noise phase is searched within its own radius, not co-moved with the signal."""
        az = self._signal_lost_analyzer()
        spp_int        = int(SAMPLE_RATE / PULSE_RATE)
        original_noise = az._noise_phase
        new_peak       = (az._peak_phase + 2) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_peak)
        az._phase_search()
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

    def test_phase_search_no_data_returns_false(self):
        az = self._signal_lost_analyzer()
        az._pipeline.wait_for_data.return_value = False
        assert az._phase_search() is False
        assert az._state == 'SIGNAL_LOST'

    def test_phase_search_radius_contract(self):
        """Search radius must cover at least a few samples of phase drift."""
        assert ContinuousAnalyzer.PHASE_SEARCH_RADIUS >= 5

    def test_latest_correction_initial_zero(self):
        assert _make_analyzer().latest_correction() == 0

    def test_phase_search_records_signal_offset(self):
        """Correction is updated to the winning offset when phase_search finds the signal."""
        az = self._signal_lost_analyzer()
        spp_int   = int(SAMPLE_RATE / PULSE_RATE)
        new_phase = (az._peak_phase + 3) % spp_int
        az._pipeline.get_snapshot.return_value = _pulse_audio(phase=new_phase)
        az._phase_search()
        assert az._state == 'LOCKED'
        assert az.latest_correction() == 3

    def test_phase_search_updates_correction_even_without_relock(self):
        """Correction field is written on every phase_search call, not only on relock."""
        az = self._signal_lost_analyzer()
        az._last_correction = 999   # sentinel — must be overwritten
        az._pipeline.get_snapshot.return_value = _noise_audio()
        az._phase_search()
        assert az._state == 'SIGNAL_LOST'
        assert az.latest_correction() != 999

    def test_signal_lost_refine_interval_much_longer_than_search(self):
        """FFT fallback should be infrequent compared to the cheap narrow scan."""
        assert ContinuousAnalyzer.SIGNAL_LOST_REFINE >= 5 * ContinuousAnalyzer.SEARCH_INTERVAL


# ---------------------------------------------------------------------------
# _fast_scan  (SIGNAL_LOST Tier-3a screening)
# ---------------------------------------------------------------------------

class TestFastScan:
    def _signal_lost_analyzer(self) -> ContinuousAnalyzer:
        az = _make_analyzer()
        az._pipeline.get_snapshot.return_value = _pulse_audio()
        az._full_analysis()
        assert az._state == 'LOCKED'
        az._pipeline.get_snapshot.return_value = np.zeros(SAMPLE_RATE, dtype=np.int16)
        for _ in range(ContinuousAnalyzer.LOSE_LOCK_COUNT):
            az._quick_check()
        assert az._state == 'SIGNAL_LOST'
        return az

    def test_fast_scan_returns_true_with_signal(self):
        az = self._signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _pulse_audio(n=n)
        assert az._fast_scan() is True

    def test_fast_scan_returns_false_without_signal(self):
        az = self._signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _noise_audio(n=n)
        assert az._fast_scan() is False

    def test_fast_scan_does_not_publish_or_change_state(self):
        az = self._signal_lost_analyzer()
        n = ContinuousAnalyzer.FAST_SCAN_SAMPLES
        az._pipeline.get_snapshot.return_value = _pulse_audio(n=n)
        before = az.latest_result()
        az._fast_scan()
        assert az._state == 'SIGNAL_LOST'
        assert az.latest_result() == before

    def test_fast_scan_no_data_returns_false(self):
        az = self._signal_lost_analyzer()
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

"""Tests for the DSP core (buzz.dsp) and AudioSampler: kernel, fit array, averaging, and golden files."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy import uint32, zeros

from buzz.config import BuzzConfig
from buzz.dsp import (
    amplitude_to_dbfs,
    analyze_window,
    average_pulse_amplitude,
    build_pulse_kernel,
    calculate_pps_fit_array,
)
from buzz.sampler import AudioPipeline, AudioSampler

SAMPLE_RATE = 16000
PULSE_RATE = 120
RESOURCES = Path(__file__).resolve().parent / 'resources'


def _make_pulse_signal(n: int = 48000, phase: int = 0, amplitude: int = 10000) -> np.ndarray:
    data = np.zeros(n, dtype=np.int32)
    spp = SAMPLE_RATE / PULSE_RATE
    for i in range(int(n / spp)):
        pos = phase + int(i * spp)
        if pos + 3 < n:
            data[pos] = data[pos + 1] = data[pos + 2] = amplitude
    return data


def _sampler_config(device_index: int | None = None, device_name: str = 'Test, DirectSound') -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.device_index = device_index
    cfg.audio.input_device_name = device_name
    cfg.audio.sample_rate = SAMPLE_RATE
    cfg.audio.pulse_rate = PULSE_RATE
    return cfg


class TestBuildPulseKernel:
    def test_first_three_samples_are_one(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert k[0] == 1 and k[1] == 1 and k[2] == 1

    def test_is_palindrome(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        np.testing.assert_array_equal(k, k[::-1])

    def test_sum_equals_scan_pulses_times_three(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        scan_pulses = PULSE_RATE // 2
        assert k.sum() == scan_pulses * 3

    def test_expected_length(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        spp = SAMPLE_RATE / PULSE_RATE
        scan_pulses = PULSE_RATE // 2
        expected = int((scan_pulses - 1) * spp) + 3
        assert len(k) == expected

    def test_50hz_grid_sum(self):
        k = build_pulse_kernel(SAMPLE_RATE, 100)
        assert k.sum() == (100 // 2) * 3

    def test_all_values_zero_or_one(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert set(k.tolist()).issubset({0, 1})

    def test_custom_n_pulses_sum(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        assert k.sum() == 15 * 3

    def test_custom_n_pulses_palindrome(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        np.testing.assert_array_equal(k, k[::-1])

    def test_custom_n_pulses_shorter_than_default(self):
        k_short = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        k_full  = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert len(k_short) < len(k_full)


class TestCalculatePpsFitArray:
    def test_zero_input_gives_zero_output(self):
        data = np.zeros(48000, dtype=np.int32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.sum() == 0

    def test_output_is_1d(self):
        data = np.zeros(48000, dtype=np.int32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.ndim == 1

    def test_uniform_input_gives_constant_output(self):
        data = np.full(48000, 100, dtype=np.int32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.min() == out.max()

    @pytest.mark.parametrize('phase', [0, 5, 10, 50])
    def test_peak_at_correct_phase(self, phase):
        data = _make_pulse_signal(phase=phase, amplitude=10000)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        spp = SAMPLE_RATE / PULSE_RATE
        detected_phase = int(out.argmax() % spp)
        assert detected_phase == phase

    def test_stronger_signal_gives_higher_peak(self):
        weak = _make_pulse_signal(phase=5, amplitude=1000)
        strong = _make_pulse_signal(phase=5, amplitude=10000)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out_weak = calculate_pps_fit_array(weak, k, PULSE_RATE // 2)
        out_strong = calculate_pps_fit_array(strong, k, PULSE_RATE // 2)
        assert out_strong.max() > out_weak.max()


class TestAveragePulseAmplitude:
    def test_zero_input_returns_zero(self):
        data = zeros(48000, dtype=uint32)
        assert average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 0

    def test_uniform_amplitude_returns_that_amplitude(self):
        data = np.full(48000, 1000, dtype=uint32)
        assert average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 1000

    def test_spikes_only_at_pulse_positions(self):
        data = zeros(48000, dtype=uint32)
        spp = SAMPLE_RATE / PULSE_RATE
        for i in range(60):
            pos = round(i * spp)
            data[pos] = data[pos + 1] = data[pos + 2] = 5000
        assert average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 5000

    @pytest.mark.parametrize('start_index', [0, 10, 50, 133, 266])
    def test_various_start_indices_match_direct_sum(self, start_index):
        rng = np.random.default_rng(7)
        data = rng.integers(0, 32768, size=48000, dtype=uint32)
        result = int(average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, start_index))
        # Direct reference sum
        spp = SAMPLE_RATE / PULSE_RATE
        total = sum(
            int(data[start_index + round(i * spp)]) +
            int(data[start_index + round(i * spp) + 1]) +
            int(data[start_index + round(i * spp) + 2])
            for i in range(60)
        )
        expected = total // (3 * 60)
        assert result == expected


class TestAmplitudeToDbfs:
    def test_full_scale_is_zero_dbfs(self):
        assert amplitude_to_dbfs(32768.0) == pytest.approx(0.0)

    def test_half_scale_is_minus_six_dbfs(self):
        assert amplitude_to_dbfs(16384.0) == pytest.approx(-6.02, abs=0.01)

    def test_zero_amplitude_returns_sentinel(self):
        assert amplitude_to_dbfs(0.0) == -128.0

    def test_negative_amplitude_returns_sentinel(self):
        assert amplitude_to_dbfs(-1.0) == -128.0


class TestAnalyzeWindow:
    def _kernel(self):
        return build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)

    def test_detects_pulse_phase(self):
        data = _make_pulse_signal(phase=5, amplitude=10000)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window is not None
        assert window.peak_phase == 5

    def test_signal_amplitude_above_noise_for_strong_pulse(self):
        data = _make_pulse_signal(phase=5, amplitude=10000)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window.signal_amplitude > window.noise_amplitude

    def test_flat_noise_gives_similar_amplitudes(self):
        rng = np.random.default_rng(0)
        data = np.abs(rng.integers(50, 150, size=48000).astype(np.int32))
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        sig_db = amplitude_to_dbfs(window.signal_amplitude)
        noise_db = amplitude_to_dbfs(window.noise_amplitude)
        assert abs(sig_db - noise_db) < 5.0

    def test_too_short_window_returns_none(self):
        data = np.zeros(10, dtype=np.int32)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window is None

    def test_empty_window_returns_none(self):
        data = np.zeros(0, dtype=np.int32)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window is None

    def test_phases_within_one_pulse_period(self):
        data = _make_pulse_signal(phase=100, amplitude=10000)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        spp = SAMPLE_RATE / PULSE_RATE
        assert 0 <= window.peak_phase < spp
        assert 0 <= window.noise_phase < spp


class TestAudioSamplerInit:
    def test_init_resolves_device_by_name(self):
        cfg = _sampler_config()
        device = {'index': 3, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 3

    def test_init_always_uses_name_even_when_index_configured(self):
        cfg = _sampler_config(device_index=2, device_name='Test, DirectSound')
        device = {'index': 7, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 7  # index from name lookup, not the stored 2

    def test_init_creates_pipeline(self):
        cfg = _sampler_config()
        device = {'index': 0, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
            sampler = AudioSampler(cfg)
        assert isinstance(sampler._pipeline, AudioPipeline)

    def test_pipeline_property_returns_pipeline(self):
        sampler = _make_sampler()
        assert sampler.pipeline is sampler._pipeline

    def test_close_stops_pipeline_stream(self):
        cfg = _sampler_config(device_index=0, device_name='Test, DirectSound')
        device = {'index': 0, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_stream = MagicMock()
            mock_cls.return_value = mock_stream
            sampler = AudioSampler(cfg)
        sampler.close()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


def _make_sampler() -> AudioSampler:
    cfg = _sampler_config(device_index=0, device_name='Test, DirectSound')
    device = {'index': 0, 'name': 'Test', 'hostapi': 0}
    with patch('buzz.sampler.sd.query_devices', return_value=device), \
         patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
        return AudioSampler(cfg)


class TestGoldenFiles:
    @pytest.fixture(autouse=True)
    def require_goldens(self):
        if not (RESOURCES / 'synthetic_audio.npy').exists():
            pytest.skip('golden files not generated — run tests/resources/generate_goldens.py')

    def test_fit_array_matches_golden(self):
        audio = np.load(RESOURCES / 'synthetic_audio.npy')
        mono = np.abs(audio[:, 0].astype(np.int32))
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        fit = calculate_pps_fit_array(mono, k, PULSE_RATE // 2)
        golden = np.load(RESOURCES / 'fit_array_golden.npy')
        np.testing.assert_array_equal(fit, golden)

    def test_analyze_window_matches_golden(self):
        golden = np.load(RESOURCES / 'analysis_golden.npy')  # [snr, signal_dbfs, noise_dbfs]
        mono = np.abs(np.load(RESOURCES / 'synthetic_audio.npy')[:, 0].astype(np.int32))
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        window = analyze_window(mono, SAMPLE_RATE, PULSE_RATE, k, PULSE_RATE // 2)
        signal = amplitude_to_dbfs(window.signal_amplitude)
        noise = amplitude_to_dbfs(window.noise_amplitude)
        assert signal - noise == pytest.approx(golden[0], abs=0.1)
        assert signal == pytest.approx(golden[1], abs=0.1)
        assert noise == pytest.approx(golden[2], abs=0.1)

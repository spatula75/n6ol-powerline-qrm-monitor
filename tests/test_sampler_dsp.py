"""Tests for audio DSP functions and AudioSampler: kernel, fit array, averaging, and golden files."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy import uint32, zeros

from buzz.config import BuzzConfig
from buzz.sampler import (
    AudioSampler, _average_pulse_amplitude, _build_pulse_kernel, _calculate_pps_fit_array,
)

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
    cfg.audio.duration = 3
    return cfg


class TestBuildPulseKernel:
    def test_first_three_samples_are_one(self):
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert k[0] == 1 and k[1] == 1 and k[2] == 1

    def test_is_palindrome(self):
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        np.testing.assert_array_equal(k, k[::-1])

    def test_sum_equals_scan_pulses_times_three(self):
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        scan_pulses = PULSE_RATE // 2
        assert k.sum() == scan_pulses * 3

    def test_expected_length(self):
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        spp = SAMPLE_RATE / PULSE_RATE
        scan_pulses = PULSE_RATE // 2
        expected = int((scan_pulses - 1) * spp) + 3
        assert len(k) == expected

    def test_50hz_grid_sum(self):
        k = _build_pulse_kernel(SAMPLE_RATE, 100)
        assert k.sum() == (100 // 2) * 3

    def test_all_values_zero_or_one(self):
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert set(k.tolist()).issubset({0, 1})


class TestCalculatePpsFitArray:
    def test_zero_input_gives_zero_output(self):
        data = np.zeros(48000, dtype=np.int32)
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = _calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.sum() == 0

    def test_output_is_1d(self):
        data = np.zeros(48000, dtype=np.int32)
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = _calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.ndim == 1

    def test_uniform_input_gives_constant_output(self):
        data = np.full(48000, 100, dtype=np.int32)
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = _calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.min() == out.max()

    @pytest.mark.parametrize('phase', [0, 5, 10, 50])
    def test_peak_at_correct_phase(self, phase):
        data = _make_pulse_signal(phase=phase, amplitude=10000)
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = _calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        spp = SAMPLE_RATE / PULSE_RATE
        detected_phase = int(out.argmax() % spp)
        assert detected_phase == phase

    def test_stronger_signal_gives_higher_peak(self):
        weak = _make_pulse_signal(phase=5, amplitude=1000)
        strong = _make_pulse_signal(phase=5, amplitude=10000)
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out_weak = _calculate_pps_fit_array(weak, k, PULSE_RATE // 2)
        out_strong = _calculate_pps_fit_array(strong, k, PULSE_RATE // 2)
        assert out_strong.max() > out_weak.max()


class TestAveragePulseAmplitude:
    def test_zero_input_returns_zero(self):
        data = zeros(48000, dtype=uint32)
        assert _average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 0

    def test_uniform_amplitude_returns_that_amplitude(self):
        data = np.full(48000, 1000, dtype=uint32)
        assert _average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 1000

    def test_spikes_only_at_pulse_positions(self):
        data = zeros(48000, dtype=uint32)
        spp = SAMPLE_RATE / PULSE_RATE
        for i in range(60):
            pos = int(i * spp)
            data[pos] = data[pos + 1] = data[pos + 2] = 5000
        assert _average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, 0) == 5000

    @pytest.mark.parametrize('start_index', [0, 10, 50, 133, 266])
    def test_various_start_indices_match_direct_sum(self, start_index):
        rng = np.random.default_rng(7)
        data = rng.integers(0, 32768, size=48000, dtype=uint32)
        result = int(_average_pulse_amplitude(data, SAMPLE_RATE, PULSE_RATE, 60, start_index))
        # Direct reference sum
        spp = SAMPLE_RATE / PULSE_RATE
        total = sum(
            int(data[start_index + int(i * spp)]) +
            int(data[start_index + int(i * spp) + 1]) +
            int(data[start_index + int(i * spp) + 2])
            for i in range(60)
        )
        expected = total // (3 * 60)
        assert result == expected


class TestAudioSamplerInit:
    def test_init_by_name_resolves_device_index(self):
        cfg = _sampler_config()
        device = {'index': 3, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 3

    def test_init_by_index_uses_stored_index(self):
        cfg = _sampler_config(device_index=2, device_name='Test, DirectSound')
        device = {'index': 2, 'name': 'Test', 'hostapi': 0}
        hostapis = [{'name': 'DirectSound'}]
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.query_hostapis', return_value=hostapis):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 2

    def test_init_by_index_name_mismatch_logs_warning(self, caplog):
        import logging
        cfg = _sampler_config(device_index=2, device_name='Old Device Name')
        device = {'index': 2, 'name': 'New Device', 'hostapi': 0}
        hostapis = [{'name': 'DirectSound'}]
        with caplog.at_level(logging.WARNING, logger='buzz.sampler'), \
             patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.query_hostapis', return_value=hostapis):
            AudioSampler(cfg)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_init_builds_kernel(self):
        cfg = _sampler_config()
        device = {'index': 0, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device):
            sampler = AudioSampler(cfg)
        assert sampler._kernel is not None
        assert len(sampler._kernel) > 0

    def test_scan_pulses_is_half_pulse_rate(self):
        cfg = _sampler_config()
        device = {'index': 0, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device):
            sampler = AudioSampler(cfg)
        assert sampler._scan_pulses == PULSE_RATE // 2


def _make_sampler() -> AudioSampler:
    cfg = _sampler_config(device_index=0, device_name='Test, DirectSound')
    device = {'index': 0, 'name': 'Test', 'hostapi': 0}
    hostapis = [{'name': 'DirectSound'}]
    with patch('buzz.sampler.sd.query_devices', return_value=device), \
         patch('buzz.sampler.sd.query_hostapis', return_value=hostapis):
        return AudioSampler(cfg)


def _synthetic_recording(phase: int = 5, amplitude: int = 20000) -> np.ndarray:
    rng = np.random.default_rng(42)
    audio = rng.integers(50, 150, size=(48000, 1), dtype=np.int16)
    spp = SAMPLE_RATE / PULSE_RATE
    for i in range(int(48000 / spp)):
        pos = phase + int(i * spp)
        if pos + 3 < 48000:
            audio[pos, 0] = amplitude
            audio[pos + 1, 0] = amplitude
            audio[pos + 2, 0] = amplitude
    return audio


class TestAudioSamplerTakeSample:
    def test_returns_three_float_values(self):
        sampler = _make_sampler()
        recording = _synthetic_recording()
        with patch('buzz.sampler.sd.rec', return_value=recording):
            result = sampler.take_sample()
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_snr_positive_for_strong_pulse(self):
        sampler = _make_sampler()
        recording = _synthetic_recording(amplitude=20000)
        with patch('buzz.sampler.sd.rec', return_value=recording):
            snr, _, _ = sampler.take_sample()
        assert snr > 0

    def test_signal_above_noise_for_strong_pulse(self):
        sampler = _make_sampler()
        recording = _synthetic_recording(amplitude=20000)
        with patch('buzz.sampler.sd.rec', return_value=recording):
            snr, signal, noise = sampler.take_sample()
        assert signal > noise

    def test_snr_near_zero_for_flat_noise(self):
        sampler = _make_sampler()
        rng = np.random.default_rng(0)
        recording = rng.integers(50, 150, size=(48000, 1), dtype=np.int16)
        with patch('buzz.sampler.sd.rec', return_value=recording):
            snr, _, _ = sampler.take_sample()
        assert abs(snr) < 5.0


class TestGoldenFiles:
    @pytest.fixture(autouse=True)
    def require_goldens(self):
        if not (RESOURCES / 'synthetic_audio.npy').exists():
            pytest.skip('golden files not generated — run tests/resources/generate_goldens.py')

    def test_fit_array_matches_golden(self):
        audio = np.load(RESOURCES / 'synthetic_audio.npy')
        mono = np.abs(audio[:, 0].astype(np.int32))
        k = _build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        fit = _calculate_pps_fit_array(mono, k, PULSE_RATE // 2)
        golden = np.load(RESOURCES / 'fit_array_golden.npy')
        np.testing.assert_array_equal(fit, golden)

    def test_take_sample_matches_golden(self):
        audio_data = np.load(RESOURCES / 'synthetic_audio.npy')
        golden = np.load(RESOURCES / 'take_sample_golden.npy')
        sampler = _make_sampler()
        with patch('buzz.sampler.sd.rec', return_value=audio_data):
            snr, signal, noise = sampler.take_sample()
        assert snr == pytest.approx(golden[0], abs=0.1)
        assert signal == pytest.approx(golden[1], abs=0.1)
        assert noise == pytest.approx(golden[2], abs=0.1)

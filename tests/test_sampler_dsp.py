"""Tests for the DSP core (buzz.dsp) and AudioSampler: kernel, fit array, averaging, and golden files."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy import float32, zeros

from buzz.config import BuzzConfig
from buzz.dsp import (
    SILENCE_DBFS,
    amplitude_to_dbfs,
    amplitude_to_dbm,
    analyze_window,
    average_pulse_amplitude,
    build_pulse_kernel,
    calculate_pps_fit_array,
    pulse_phase_period,
)
from buzz.sampler import AudioPipeline, AudioSampler

SAMPLE_RATE = 16000
PULSE_RATE = 120
RESOURCES = Path(__file__).resolve().parent / 'resources'


def _make_pulse_signal(n: int = 48000, phase: int = 0, amplitude: int = 10000) -> np.ndarray:
    """Rectified audio with a 120 pps pulse train at `phase`.

    Pulse positions use round(), the same rule build_pulse_kernel and
    average_pulse_amplitude use, so the fixture models the signal the DSP
    actually looks for rather than one offset from it.
    """
    data = np.zeros(n, dtype=np.float32)
    spp = SAMPLE_RATE / PULSE_RATE
    for i in range(int(n / spp)):
        pos = phase + round(i * spp)
        if pos + 3 < n:
            data[pos] = data[pos + 1] = data[pos + 2] = amplitude
    return data


def _sampler_config(device_name: str = 'Test, DirectSound') -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.input_device_name = device_name
    cfg.audio.sample_rate = SAMPLE_RATE
    cfg.audio.pulse_rate = PULSE_RATE
    return cfg


class TestBuildPulseKernel:
    def test_first_three_samples_are_one(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert k[0] == 1 and k[1] == 1 and k[2] == 1

    def test_positions_match_average_pulse_amplitude(self):
        """The kernel's ones must sit exactly where average_pulse_amplitude reads.

        Both use round(); truncating in either place would put a third of the
        positions one sample off, costing a third of each affected pulse's energy
        and making the FFT optimise a phase the averager never samples.
        """
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        spp = SAMPLE_RATE / PULSE_RATE
        expected = {round(i * spp) + j for i in range(PULSE_RATE // 2) for j in range(3)}
        assert set(np.where(k == 1)[0].tolist()) == expected

    def test_sum_equals_scan_pulses_times_three(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        scan_pulses = PULSE_RATE // 2
        assert k.sum() == scan_pulses * 3

    def test_expected_length(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        spp = SAMPLE_RATE / PULSE_RATE
        scan_pulses = PULSE_RATE // 2
        expected = round((scan_pulses - 1) * spp) + 3
        assert len(k) == expected

    def test_no_trailing_or_leading_zeros(self):
        """Last pulse group sits flush against the end - no padding to convolve through."""
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert k[0] == 1 and k[-1] == 1

    def test_50hz_grid_sum(self):
        k = build_pulse_kernel(SAMPLE_RATE, 100)
        assert k.sum() == (100 // 2) * 3

    def test_all_values_zero_or_one(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert set(k.tolist()).issubset({0, 1})

    def test_custom_n_pulses_sum(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        assert k.sum() == 15 * 3

    def test_custom_n_pulses_positions_match_average_pulse_amplitude(self):
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        spp = SAMPLE_RATE / PULSE_RATE
        expected = {round(i * spp) + j for i in range(15) for j in range(3)}
        assert set(np.where(k == 1)[0].tolist()) == expected

    def test_custom_n_pulses_shorter_than_default(self):
        k_short = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE, n_pulses=15)
        k_full  = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert len(k_short) < len(k_full)


class TestCalculatePpsFitArray:
    def test_zero_input_gives_zero_output(self):
        data = np.zeros(48000, dtype=np.float32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        np.testing.assert_allclose(out, 0.0, atol=1e-9)

    def test_output_is_1d(self):
        data = np.zeros(48000, dtype=np.float32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert out.ndim == 1

    def test_output_is_float_not_quantised(self):
        """Scores stay in float: at 1 LSB mean amplitude one integer step is 6 dB,
        so flooring here would coarsen exactly the weak-signal end that matters."""
        data = _make_pulse_signal(phase=5, amplitude=10000)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        assert np.issubdtype(out.dtype, np.floating)
        assert np.any(out != np.rint(out))   # truly fractional, not floats-holding-ints

    def test_uniform_input_gives_constant_output(self):
        data = np.full(48000, 100, dtype=np.float32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        np.testing.assert_allclose(out, 100.0, rtol=1e-9)

    @pytest.mark.parametrize('start_index', [0, 1, 133, 5000, 20000])
    def test_fit_score_equals_average_pulse_amplitude(self, start_index):
        """The FFT fit at position n must equal what average_pulse_amplitude
        measures at start_index = n - they are two routes to the same sum, and
        the whole state machine depends on the FFT's chosen phase measuring the
        same way the cheap path later re-measures it."""
        rng = np.random.default_rng(5)
        data = np.abs(rng.normal(0, 800, 48000)).astype(np.float32)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        fit = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        direct = average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE,
                                         PULSE_RATE // 2, start_index)
        assert fit[start_index] == pytest.approx(direct, abs=1e-6)

    @pytest.mark.parametrize('phase', [0, 5, 10, 50])
    def test_peak_at_correct_phase(self, phase):
        data = _make_pulse_signal(phase=phase, amplitude=10000)
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        out = calculate_pps_fit_array(data, k, PULSE_RATE // 2)
        detected_phase = int(out.argmax()) % pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
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
        data = zeros(48000, dtype=float32)
        assert average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE, 60, 0) == 0

    def test_uniform_amplitude_returns_that_amplitude(self):
        data = np.full(48000, 1000, dtype=np.float32)
        assert average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE, 60, 0) == 1000

    def test_spikes_only_at_pulse_positions(self):
        data = zeros(48000, dtype=float32)
        spp = SAMPLE_RATE / PULSE_RATE
        for i in range(60):
            pos = round(i * spp)
            data[pos] = data[pos + 1] = data[pos + 2] = 5000
        assert average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE, 60, 0) == 5000

    @pytest.mark.parametrize('start_index', [0, 10, 50, 133, 266])
    def test_various_start_indices_match_direct_sum(self, start_index):
        rng = np.random.default_rng(7)
        data = rng.integers(0, 32768, size=48000).astype(np.float32)
        result = average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE, 60, start_index)
        # Direct reference sum
        spp = SAMPLE_RATE / PULSE_RATE
        total = sum(
            int(data[start_index + round(i * spp)]) +
            int(data[start_index + round(i * spp) + 1]) +
            int(data[start_index + round(i * spp) + 2])
            for i in range(60)
        )
        assert result == pytest.approx(total / (3 * 60))

    def test_returns_unquantised_mean(self):
        """A non-integer mean must survive as a fraction, not floor to an int."""
        data = zeros(48000, dtype=float32)
        spp = SAMPLE_RATE / PULSE_RATE
        for i in range(60):
            pos = round(i * spp)
            data[pos] = data[pos + 1] = 1000
            data[pos + 2] = 1001          # mean is 1000.333…, not an integer
        result = average_pulse_amplitude(data, SAMPLE_RATE / PULSE_RATE, 60, 0)
        assert result == pytest.approx(1000 + 1 / 3)


class TestAmplitudeToDbfs:
    def test_full_scale_is_zero_dbfs(self):
        assert amplitude_to_dbfs(32768.0) == pytest.approx(0.0)

    def test_half_scale_is_minus_six_dbfs(self):
        assert amplitude_to_dbfs(16384.0) == pytest.approx(-6.02, abs=0.01)

    def test_zero_amplitude_returns_sentinel(self):
        assert amplitude_to_dbfs(0.0) == -128.0

    def test_negative_amplitude_returns_sentinel(self):
        assert amplitude_to_dbfs(-1.0) == -128.0


class TestAmplitudeToDbm:
    def test_offset_is_applied(self):
        assert amplitude_to_dbm(32768.0, -32.0) == pytest.approx(-32.0)

    def test_matches_dbfs_plus_offset(self):
        assert amplitude_to_dbm(1000.0, -32.0) == pytest.approx(amplitude_to_dbfs(1000.0) - 32.0)

    def test_silence_sentinel_is_not_offset(self):
        assert amplitude_to_dbm(0.0, -32.0) == SILENCE_DBFS

    def test_negative_amplitude_returns_sentinel(self):
        assert amplitude_to_dbm(-1.0, 10.0) == SILENCE_DBFS


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
        data = np.abs(rng.integers(50, 150, size=48000).astype(np.float32))
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        sig_db = amplitude_to_dbfs(window.signal_amplitude)
        noise_db = amplitude_to_dbfs(window.noise_amplitude)
        assert abs(sig_db - noise_db) < 5.0

    def test_too_short_window_returns_none(self):
        data = np.zeros(10, dtype=np.float32)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window is None

    def test_empty_window_returns_none(self):
        data = np.zeros(0, dtype=np.float32)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window is None

    def test_phases_within_one_phase_period(self):
        data = _make_pulse_signal(phase=100, amplitude=10000)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        period = pulse_phase_period(SAMPLE_RATE, PULSE_RATE)
        assert 0 <= window.peak_phase < period
        assert 0 <= window.noise_phase < period

    @pytest.mark.parametrize('phase', [0, 5, 100, 399])
    def test_detected_phase_is_exact_across_the_period(self, phase):
        """A perfectly periodic signal makes every position one phase period apart an
        equal-best fit, so FFT round-off decides which one argmax returns.  Reducing
        by the exact integer period maps them all back to the true phase; reducing by
        the fractional samples_per_pulse truncated 4.999999999997698 to 4."""
        data = _make_pulse_signal(phase=phase, amplitude=10000)
        window = analyze_window(data, SAMPLE_RATE, PULSE_RATE, self._kernel(), PULSE_RATE // 2)
        assert window.peak_phase == phase


class TestAudioSamplerInit:
    def test_init_resolves_device_by_name(self):
        cfg = _sampler_config()
        device = {'index': 3, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 3

    def test_the_index_comes_from_the_lookup_not_from_the_config(self):
        """There is nowhere in the config to put an index any more, and this is why:
        the answer has to be whatever the name resolves to right now. The device the
        name finds is at 7 here, and 7 is what the sampler must use."""
        cfg = _sampler_config(device_name='Test, DirectSound')
        device = {'index': 7, 'name': 'Test', 'hostapi': 0}
        with patch('buzz.sampler.sd.query_devices', return_value=device), \
             patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
            sampler = AudioSampler(cfg)
        assert sampler._device_index == 7

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
        cfg = _sampler_config(device_name='Test, DirectSound')
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
    cfg = _sampler_config(device_name='Test, DirectSound')
    device = {'index': 0, 'name': 'Test', 'hostapi': 0}
    with patch('buzz.sampler.sd.query_devices', return_value=device), \
         patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
        return AudioSampler(cfg)


class TestGoldenFiles:
    @pytest.fixture(autouse=True)
    def require_goldens(self):
        if not (RESOURCES / 'synthetic_audio.npy').exists():
            pytest.skip('golden files not generated - run tests/resources/generate_goldens.py')

    def test_fit_array_matches_golden(self):
        audio = np.load(RESOURCES / 'synthetic_audio.npy')
        mono = np.abs(audio[:, 0].astype(np.float32))
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        fit = calculate_pps_fit_array(mono, k, PULSE_RATE // 2)
        golden = np.load(RESOURCES / 'fit_array_golden.npy')
        # Scores are float now, so pin to FFT round-off rather than bit equality.
        np.testing.assert_allclose(fit, golden, rtol=1e-9, atol=1e-6)

    def test_analyze_window_matches_golden(self):
        golden = np.load(RESOURCES / 'analysis_golden.npy')  # [snr, signal_dbfs, noise_dbfs]
        mono = np.abs(np.load(RESOURCES / 'synthetic_audio.npy')[:, 0].astype(np.float32))
        k = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        window = analyze_window(mono, SAMPLE_RATE, PULSE_RATE, k, PULSE_RATE // 2)
        signal = amplitude_to_dbfs(window.signal_amplitude)
        noise = amplitude_to_dbfs(window.noise_amplitude)
        assert signal - noise == pytest.approx(golden[0], abs=0.1)
        assert signal == pytest.approx(golden[1], abs=0.1)
        assert noise == pytest.approx(golden[2], abs=0.1)

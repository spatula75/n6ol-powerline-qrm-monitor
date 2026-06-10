"""
Verifies the mathematical properties that justify replacing np.correlate
(padded kernel) with scipy.signal.fftconvolve (symmetric trimmed kernel).

Properties under test:
1. The trimmed kernel (trailing zeros removed) is an exact palindrome.
2. Trimmed and padded kernels produce identical correlation values over
   the padded kernel's valid range — trailing zeros contribute nothing.
3. For a symmetric kernel, fftconvolve == np.correlate within float tolerance
   (convolution == correlation when the kernel reads the same both ways).
4. The combined optimization produces the same argmax and argmin as the
   original on synthetic pulse data.
"""

import numpy as np
import pytest
from math import ceil
from numpy import uint32, zeros
from scipy.signal import fftconvolve

SAMPLE_RATE = 16000
PULSE_RATE = 120
SCAN_PULSES = PULSE_RATE // 2


# ---------------------------------------------------------------------------
# Reference implementations (pinning the OLD behaviour)
# ---------------------------------------------------------------------------

def _build_padded_kernel(sample_rate):
    """Original ceil-based kernel — has trailing zeros."""
    pf = sample_rate / PULSE_RATE
    coeffs = zeros(ceil(SCAN_PULSES * pf), dtype=uint32)
    for i in range(SCAN_PULSES):
        pos = int(i * pf)
        coeffs[pos] = 1
        coeffs[pos + 1] = 1
        coeffs[pos + 2] = 1
    return coeffs


def _build_symmetric_kernel(sample_rate):
    """Trimmed kernel: length = floor((N-1)*pf) + 3, which is an exact palindrome for N=60."""
    pf = sample_rate / PULSE_RATE
    last_pos = int((SCAN_PULSES - 1) * pf)
    coeffs = zeros(last_pos + 3, dtype=uint32)
    for i in range(SCAN_PULSES):
        pos = int(i * pf)
        coeffs[pos] = 1
        coeffs[pos + 1] = 1
        coeffs[pos + 2] = 1
    return coeffs


def _old_scan(data, sample_rate):
    """Original scan: np.correlate with padded kernel, integer floor division."""
    kernel = _build_padded_kernel(sample_rate)
    output = np.correlate(data.astype(np.int64), kernel.astype(np.int64), mode='valid')
    return output // (3 * SCAN_PULSES)


def _new_scan(data, sample_rate):
    """Proposed scan: fftconvolve with symmetric kernel, round then floor divide."""
    kernel = _build_symmetric_kernel(sample_rate)
    raw = fftconvolve(data.astype(np.float64), kernel.astype(np.float64), mode='valid')
    return np.rint(raw).astype(np.int64) // (3 * SCAN_PULSES)


def _make_pulse_signal(offset, noise_amplitude=50, pulse_amplitude=30000):
    """48000-sample signal: low noise with a strong 120pps pulse train at `offset`."""
    rng = np.random.default_rng(42)
    signal = rng.integers(0, noise_amplitude, size=48000, dtype=uint32)
    pf = SAMPLE_RATE / PULSE_RATE
    for i in range(SCAN_PULSES):
        pos = offset + int(i * pf)
        if pos + 3 <= len(signal):
            signal[pos] = pulse_amplitude
            signal[pos + 1] = pulse_amplitude
            signal[pos + 2] = pulse_amplitude
    return signal


def _random_data(seed, length=48000):
    return np.random.default_rng(seed).integers(0, 32768, size=length, dtype=uint32)


# ---------------------------------------------------------------------------
# 1. Kernel symmetry properties
# ---------------------------------------------------------------------------

class TestSymmetricKernelProperties:

    def test_kernel_is_palindrome(self):
        kernel = _build_symmetric_kernel(SAMPLE_RATE)
        np.testing.assert_array_equal(kernel, kernel[::-1])

    def test_kernel_shorter_than_padded(self):
        assert len(_build_symmetric_kernel(SAMPLE_RATE)) < len(_build_padded_kernel(SAMPLE_RATE))

    def test_padded_kernel_trailing_elements_are_zeros(self):
        sym = _build_symmetric_kernel(SAMPLE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        assert np.all(padded[len(sym):] == 0)

    def test_pulse_positions_identical_in_both_kernels(self):
        sym = _build_symmetric_kernel(SAMPLE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        assert set(np.where(sym == 1)[0].tolist()) == set(np.where(padded == 1)[0].tolist())

    def test_no_trailing_zeros(self):
        assert _build_symmetric_kernel(SAMPLE_RATE)[-1] == 1

    def test_no_leading_zeros(self):
        assert _build_symmetric_kernel(SAMPLE_RATE)[0] == 1


# ---------------------------------------------------------------------------
# 2. Trimmed kernel gives same correlation values as padded kernel
# ---------------------------------------------------------------------------

class TestTrimmedKernelEquivalence:

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_valid_range_values_identical(self, seed):
        data = _random_data(seed)
        sym = _build_symmetric_kernel(SAMPLE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        n_padded = len(data) - len(padded) + 1

        out_sym = np.correlate(data.astype(np.int64), sym.astype(np.int64), mode='valid')
        out_padded = np.correlate(data.astype(np.int64), padded.astype(np.int64), mode='valid')

        np.testing.assert_array_equal(out_sym[:n_padded], out_padded)


# ---------------------------------------------------------------------------
# 3. fftconvolve == np.correlate for a symmetric kernel
# ---------------------------------------------------------------------------

class TestFftConvolveEquivalence:

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_fftconvolve_matches_correlate(self, seed):
        data = _random_data(seed).astype(np.float64)
        kernel = _build_symmetric_kernel(SAMPLE_RATE).astype(np.float64)
        np.testing.assert_allclose(fftconvolve(data, kernel, mode='valid'),
                                   np.correlate(data, kernel, mode='valid'), atol=0.5)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_integer_results_identical_after_rounding(self, seed):
        data = _random_data(seed)
        kernel = _build_symmetric_kernel(SAMPLE_RATE)

        corr = np.correlate(data.astype(np.int64), kernel.astype(np.int64), mode='valid')
        conv = fftconvolve(data.astype(np.float64), kernel.astype(np.float64), mode='valid')

        n = min(len(corr), len(conv))
        np.testing.assert_array_equal(np.rint(conv[:n]).astype(np.int64) // (3 * SCAN_PULSES),
                                      corr[:n] // (3 * SCAN_PULSES))


# ---------------------------------------------------------------------------
# 4. argmax / argmin preserved end-to-end (old scan vs new scan)
# ---------------------------------------------------------------------------

class TestScanPreservesExtrema:

    @pytest.mark.parametrize("offset", [0, 50, 133, 266, 500])
    def test_argmax_matches_old_scan(self, offset):
        data = _make_pulse_signal(offset)
        old_out = _old_scan(data, SAMPLE_RATE)
        new_out = _new_scan(data, SAMPLE_RATE)
        assert old_out.argmax() == new_out[:len(old_out)].argmax()

    @pytest.mark.parametrize("seed", range(10))
    def test_argmax_matches_on_random_data(self, seed):
        data = _random_data(seed)
        old_out = _old_scan(data, SAMPLE_RATE)
        new_out = _new_scan(data, SAMPLE_RATE)
        assert old_out.argmax() == new_out[:len(old_out)].argmax()

    @pytest.mark.parametrize("seed", range(10))
    def test_argmin_matches_on_random_data(self, seed):
        data = _random_data(seed)
        old_out = _old_scan(data, SAMPLE_RATE)
        new_out = _new_scan(data, SAMPLE_RATE)
        assert old_out.argmin() == new_out[:len(old_out)].argmin()

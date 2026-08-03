"""
Verifies the mathematical properties that justify computing the pps fit with
scipy.signal.fftconvolve over a reversed, trimmed kernel instead of np.correlate.

Properties under test:
1. Kernel pulse positions use round(), matching average_pulse_amplitude, and the
   kernel carries no leading or trailing zeros.
2. Reversing the kernel is what turns convolution into correlation.  The kernel is
   not a palindrome, so skipping the reversal gives a different - wrong - answer.
3. fftconvolve(x, kernel[::-1]) == np.correlate(x, kernel) within float tolerance.
4. The optimization produces the same argmax and argmin as a direct correlation on
   synthetic pulse data.

The kernel used to be built with truncated positions, which happened to make it an
exact palindrome so the reversal could be skipped.  That symmetry was a coincidence
of particular sample-rate/pulse-rate pairs, and it cost a one-sample disagreement
with average_pulse_amplitude on a third of the pulse positions; round() plus an
explicit reversal is correct for any rate pair.
"""

import numpy as np
import pytest
from math import ceil
from numpy import uint32, zeros
from scipy.signal import fftconvolve

from buzz.dsp import (
    PULSE_WIDTH_SAMPLES,
    build_pulse_kernel,
    calculate_pps_fit_array,
    pulse_phase_period,
)

SAMPLE_RATE = 16000
PULSE_RATE = 120
SCAN_PULSES = PULSE_RATE // 2


def _build_padded_kernel(sample_rate):
    """Kernel with trailing zeros - the trimmed kernel must match it over the valid range."""
    pf = sample_rate / PULSE_RATE
    coeffs = zeros(ceil(SCAN_PULSES * pf), dtype=uint32)
    for i in range(SCAN_PULSES):
        pos = round(i * pf)
        coeffs[pos:pos + PULSE_WIDTH_SAMPLES] = 1
    return coeffs


def _reference_scan(data, kernel):
    """Reference: direct cross-correlation, no FFT."""
    return (np.correlate(data.astype(np.int64), kernel.astype(np.int64), mode='valid')
            / (PULSE_WIDTH_SAMPLES * SCAN_PULSES))


def _make_pulse_signal(offset, noise_amplitude=50, pulse_amplitude=30000):
    """48000-sample signal: low noise with a strong 120pps pulse train at `offset`."""
    rng = np.random.default_rng(42)
    signal = rng.integers(0, noise_amplitude, size=48000, dtype=uint32)
    pf = SAMPLE_RATE / PULSE_RATE
    for i in range(SCAN_PULSES):
        pos = offset + round(i * pf)
        if pos + PULSE_WIDTH_SAMPLES <= len(signal):
            signal[pos:pos + PULSE_WIDTH_SAMPLES] = pulse_amplitude
    return signal


def _random_data(seed, length=48000):
    return np.random.default_rng(seed).integers(0, 32768, size=length, dtype=uint32)


# ---------------------------------------------------------------------------
# 1. Kernel shape
# ---------------------------------------------------------------------------

class TestKernelShape:

    def test_positions_are_rounded_not_truncated(self):
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        pf = SAMPLE_RATE / PULSE_RATE
        rounded = {round(i * pf) + j
                   for i in range(SCAN_PULSES) for j in range(PULSE_WIDTH_SAMPLES)}
        truncated = {int(i * pf) + j
                     for i in range(SCAN_PULSES) for j in range(PULSE_WIDTH_SAMPLES)}
        assert set(np.where(kernel == 1)[0].tolist()) == rounded
        assert rounded != truncated          # the two rules truly disagree here

    def test_no_leading_or_trailing_zeros(self):
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert kernel[0] == 1 and kernel[-1] == 1

    def test_kernel_shorter_than_padded(self):
        assert len(build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)) < len(_build_padded_kernel(SAMPLE_RATE))

    def test_padded_kernel_trailing_elements_are_zeros(self):
        trimmed = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        assert np.all(padded[len(trimmed):] == 0)

    def test_pulse_positions_identical_in_both_kernels(self):
        trimmed = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        assert set(np.where(trimmed == 1)[0].tolist()) == set(np.where(padded == 1)[0].tolist())


# ---------------------------------------------------------------------------
# 2. The reversal is what makes convolution into correlation
# ---------------------------------------------------------------------------

class TestReversalIsNecessary:

    def test_kernel_is_not_a_palindrome(self):
        """round()ed positions are not symmetric, so symmetry cannot be relied on."""
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert not np.array_equal(kernel, kernel[::-1])

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_unreversed_convolution_would_be_wrong(self, seed):
        data = _random_data(seed).astype(np.float64)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE).astype(np.float64)
        correct = np.correlate(data, kernel, mode='valid')
        unreversed = fftconvolve(data, kernel, mode='valid')
        assert not np.allclose(unreversed, correct, atol=1.0)


# ---------------------------------------------------------------------------
# 3. fftconvolve over the reversed kernel == np.correlate
# ---------------------------------------------------------------------------

class TestFftConvolveEquivalence:

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_matches_direct_correlation(self, seed):
        data = _random_data(seed)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        np.testing.assert_allclose(calculate_pps_fit_array(data, kernel, SCAN_PULSES),
                                   _reference_scan(data, kernel), rtol=1e-9, atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_trimmed_matches_padded_over_valid_range(self, seed):
        data = _random_data(seed)
        trimmed = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        padded = _build_padded_kernel(SAMPLE_RATE)
        n_padded = len(data) - len(padded) + 1
        np.testing.assert_allclose(_reference_scan(data, trimmed)[:n_padded],
                                   _reference_scan(data, padded), rtol=1e-9)


# ---------------------------------------------------------------------------
# 4. argmax / argmin preserved end-to-end
# ---------------------------------------------------------------------------

class TestScanPreservesExtrema:

    @pytest.mark.parametrize("offset", [0, 50, 133, 266, 500])
    def test_argmax_matches_reference(self, offset):
        data = _make_pulse_signal(offset)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert calculate_pps_fit_array(data, kernel, SCAN_PULSES).argmax() == \
               _reference_scan(data, kernel).argmax()

    @pytest.mark.parametrize("seed", range(10))
    def test_argmax_matches_on_random_data(self, seed):
        data = _random_data(seed)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert calculate_pps_fit_array(data, kernel, SCAN_PULSES).argmax() == \
               _reference_scan(data, kernel).argmax()

    @pytest.mark.parametrize("seed", range(10))
    def test_argmin_matches_on_random_data(self, seed):
        data = _random_data(seed)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        assert calculate_pps_fit_array(data, kernel, SCAN_PULSES).argmin() == \
               _reference_scan(data, kernel).argmin()

    @pytest.mark.parametrize("offset", [0, 5, 50, 100])
    def test_detected_phase_is_exact(self, offset):
        """With kernel and signal on the same round()ed grid the peak falls exactly
        on the true phase - the one-sample bias from the old truncated kernel is gone."""
        data = _make_pulse_signal(offset)
        kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
        fit = calculate_pps_fit_array(data, kernel, SCAN_PULSES)
        assert int(fit.argmax()) % pulse_phase_period(SAMPLE_RATE, PULSE_RATE) == offset

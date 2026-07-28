"""
Confirms that _sum_pulse_train produces the same result as directly summing
sample values at the expected pulse positions.

The cross-correlation of a signal with a 0/1 coefficient vector is identical
to summing the signal at the positions where the coefficient is 1.  These tests
pin that equivalence so we can trust the direct-sum path in AudioSampler.
"""

import numpy as np
import pytest
from numpy import uint32, zeros

from buzz.dsp import average_pulse_amplitude as _sum_pulse_train

SAMPLE_RATE = 16000
PULSE_RATE = 120


def _direct_sum(data, sample_rate, n_pulses, start_index):
    """Reference: walk the pulse positions and sum the samples by hand."""
    pf = sample_rate / 120
    total = 0
    for i in range(n_pulses):
        pos = round(i * pf)
        total += int(data[start_index + pos])
        total += int(data[start_index + pos + 1])
        total += int(data[start_index + pos + 2])
    return total / (3 * n_pulses)


def _fast_result(data, n_pulses, start_index):
    return _sum_pulse_train(data, SAMPLE_RATE / PULSE_RATE, n_pulses, start_index)


def _random_data(seed, length=48000):
    return np.random.default_rng(seed).integers(0, 32768, size=length, dtype=uint32)


class TestSumPulseTrainEquivalence:

    def test_all_zeros(self):
        data = zeros(48000, dtype=uint32)
        assert _fast_result(data, 60, 0) == _direct_sum(data, SAMPLE_RATE, 60, 0) == 0

    def test_uniform_amplitude(self):
        data = np.full(48000, 1000, dtype=uint32)
        assert _fast_result(data, 60, 0) == _direct_sum(data, SAMPLE_RATE, 60, 0) == 1000

    def test_pulse_only_at_expected_positions(self):
        data = zeros(48000, dtype=uint32)
        pf = SAMPLE_RATE / 120
        for i in range(60):
            pos = round(i * pf)
            data[pos] = data[pos + 1] = data[pos + 2] = 5000
        assert _fast_result(data, 60, 0) == _direct_sum(data, SAMPLE_RATE, 60, 0) == 5000

    @pytest.mark.parametrize("start_index", [0, 10, 50, 133, 266])
    def test_various_start_indices(self, start_index):
        data = _random_data(7)
        assert _fast_result(data, 60, start_index) == _direct_sum(data, SAMPLE_RATE, 60, start_index)

    @pytest.mark.parametrize("n_pulses", [30, 60, 90, 110])
    def test_various_pulse_counts(self, n_pulses):
        data = _random_data(99)
        assert _fast_result(data, n_pulses, 0) == _direct_sum(data, SAMPLE_RATE, n_pulses, 0)

    @pytest.mark.parametrize("seed", range(20))
    def test_twenty_random_seeds(self, seed):
        data = _random_data(seed)
        assert _fast_result(data, 60, 0) == _direct_sum(data, SAMPLE_RATE, 60, 0)

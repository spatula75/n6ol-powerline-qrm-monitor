"""Tests for the pure math in tools/pulse_probe.py (no audio hardware required)."""
import numpy as np
import pytest

from tools.pulse_probe import (
    drift_rate_to_grid_hz,
    estimate_drift_rate,
    find_source_peaks,
    fold_pulses,
    matched_width_sweep,
    profile_sharpness,
    rectify,
)

SAMPLE_RATE = 16000
PULSE_RATE = 120
SPP = SAMPLE_RATE / PULSE_RATE
HALF = int(SPP // 2)


def _pulse_train(seconds: float = 4.0, drift_rate: float = 0.0, width: int = 3,
                 amplitude: float = 6000.0, noise: float = 200.0,
                 phases: tuple[int, ...] = (0,), dc: float = 0.0, seed: int = 3) -> np.ndarray:
    """Synthetic bipolar audio with one pulse train per entry in `phases`.

    drift_rate slips each successive pulse, in samples per second, the same way a
    utility line frequency error does.
    """
    n = int(seconds * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    audio = rng.normal(dc, noise, n)
    for i in range(int(n / SPP)):
        base = round(i * SPP) + round(drift_rate * i / PULSE_RATE)
        for phase in phases:
            start = base + phase
            if 0 <= start and start + width < n:
                # alternating sign: an LSB-demodulated impulse is bipolar
                audio[start:start + width] = amplitude * (-1.0) ** np.arange(width)
    return audio


class TestRectify:
    def test_removes_dc_offset(self):
        clean = rectify(_pulse_train(seconds=1.0, dc=0.0))
        offset = rectify(_pulse_train(seconds=1.0, dc=500.0))
        assert clean.mean() == pytest.approx(offset.mean(), rel=0.02)

    def test_output_is_non_negative(self):
        assert rectify(_pulse_train(seconds=1.0)).min() >= 0

    def test_median_not_mean_so_pulses_do_not_bias_it(self):
        """A dense pulse train pulls a mean far off; the median ignores it."""
        audio = _pulse_train(seconds=1.0, amplitude=20000.0, noise=50.0, width=3)
        unipolar = np.abs(audio)                     # make the train one-sided
        assert abs(np.median(unipolar) - np.mean(unipolar)) > 100   # mean is badly skewed
        assert rectify(unipolar).min() >= 0


class TestFoldPulses:
    def test_profile_length_is_one_pulse_period(self):
        profile, _ = fold_pulses(rectify(_pulse_train()), SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        assert len(profile) == 2 * HALF + 1

    def test_averages_many_pulses(self):
        _, n = fold_pulses(rectify(_pulse_train(seconds=4.0)), SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        assert n > 400          # ~480 pulses in 4 s

    def test_pulse_lands_at_profile_centre_when_undrifted(self):
        profile, _ = fold_pulses(rectify(_pulse_train()), SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        assert abs(int(profile.argmax()) - HALF) <= 1

    def test_too_short_input_returns_empty(self):
        profile, n = fold_pulses(np.zeros(50), SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        assert n == 0 and len(profile) == 2 * HALF + 1

    def test_correcting_for_drift_recovers_sharpness(self):
        """The whole point of the tool: an uncorrected fold of drifting audio is blurred."""
        rectified = rectify(_pulse_train(seconds=4.0, drift_rate=6.0))
        blurred, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        sharp, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 6.0, HALF)
        assert profile_sharpness(sharp) > 2 * profile_sharpness(blurred)


class TestProfileSharpness:
    def test_flat_profile_scores_about_one(self):
        assert profile_sharpness(np.full(133, 50.0)) == pytest.approx(1.0)

    def test_peaked_profile_scores_higher_than_flat(self):
        peaked = np.full(133, 50.0)
        peaked[66] = 500.0
        assert profile_sharpness(peaked) > profile_sharpness(np.full(133, 50.0))

    def test_all_zero_profile_is_safe(self):
        assert profile_sharpness(np.zeros(133)) == 0.0


class TestEstimateDriftRate:
    @pytest.mark.parametrize('true_rate', [0.0, 5.0, -5.0, 12.0])
    def test_recovers_known_drift(self, true_rate):
        rectified = rectify(_pulse_train(seconds=4.0, drift_rate=true_rate))
        assert estimate_drift_rate(rectified, SAMPLE_RATE, PULSE_RATE, HALF) == pytest.approx(
            true_rate, abs=1.0)

    def test_undrifted_signal_reports_near_zero(self):
        rectified = rectify(_pulse_train(seconds=4.0, drift_rate=0.0))
        assert abs(estimate_drift_rate(rectified, SAMPLE_RATE, PULSE_RATE, HALF)) < 1.0


class TestDriftRateToGridHz:
    def test_zero_drift_is_nominal(self):
        assert drift_rate_to_grid_hz(0.0, SAMPLE_RATE, PULSE_RATE) == pytest.approx(60.0)

    def test_positive_drift_means_grid_running_slow(self):
        """Positive slip = pulses arriving later = fewer per second = slow grid."""
        assert drift_rate_to_grid_hz(5.0, SAMPLE_RATE, PULSE_RATE) < 60.0

    def test_negative_drift_means_grid_running_fast(self):
        assert drift_rate_to_grid_hz(-5.0, SAMPLE_RATE, PULSE_RATE) > 60.0

    def test_known_value(self):
        # A true rate of 120.05 pps requires a slip of SR*(120-120.05)/120.05
        drift = SAMPLE_RATE * (PULSE_RATE - 120.05) / 120.05
        assert drift_rate_to_grid_hz(drift, SAMPLE_RATE, PULSE_RATE) == pytest.approx(60.025)


class TestFindSourcePeaks:
    def test_single_source_gives_one_peak(self):
        rectified = rectify(_pulse_train(seconds=4.0, phases=(0,)))
        profile, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        assert len(find_source_peaks(profile)) == 1

    def test_three_phases_give_three_peaks_a_third_apart(self):
        third = int(round(SPP / 3))
        rectified = rectify(_pulse_train(seconds=4.0, phases=(0, third, 2 * third)))
        profile, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        peaks = sorted(find_source_peaks(profile))
        assert len(peaks) == 3
        gaps = [peaks[i + 1] - peaks[i] for i in range(2)]
        assert all(abs(gap - third) <= 2 for gap in gaps)

    def test_flat_profile_has_no_peaks(self):
        assert find_source_peaks(np.full(133, 7.0)) == []

    def test_peaks_are_returned_strongest_first(self):
        rectified = rectify(_pulse_train(seconds=4.0, phases=(0, 44)))
        profile, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        peaks = find_source_peaks(profile)
        assert profile[peaks[0]] >= profile[peaks[-1]]


class TestMatchedWidthSweep:
    def _profile_for(self, width: int) -> np.ndarray:
        rectified = rectify(_pulse_train(seconds=4.0, width=width, noise=400.0))
        profile, _ = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)
        return profile

    def test_covers_every_candidate_width(self):
        widths = [w for w, _, _ in matched_width_sweep(self._profile_for(3))]
        assert widths == list(range(1, 21))

    @pytest.mark.parametrize('true_width', [3, 6, 10])
    def test_best_width_tracks_the_real_pulse_width(self, true_width):
        sweep = matched_width_sweep(self._profile_for(true_width))
        best = max(sweep, key=lambda row: row[2])[0]
        assert abs(best - true_width) <= 2

    def test_score_turns_over_rather_than_growing_forever(self):
        """A resolved pulse has an interior optimum — that is what makes the
        answer meaningful.  Monotonic growth means the pulse was not resolved."""
        scores = [fom for _, _, fom in matched_width_sweep(self._profile_for(3))]
        assert scores.index(max(scores)) < len(scores) - 1

    def test_wider_pulse_prefers_wider_window(self):
        narrow = max(matched_width_sweep(self._profile_for(3)), key=lambda r: r[2])[0]
        wide = max(matched_width_sweep(self._profile_for(10)), key=lambda r: r[2])[0]
        assert wide > narrow

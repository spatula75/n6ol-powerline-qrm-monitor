"""Tests for tools/pulse_probe.py (no audio hardware required).

The pure math is exercised directly.  The one place that touches a sound card,
capture(), is mocked at the AudioSampler boundary, and the report sections are
checked through capsys - they are the tool's entire output, so what they say is
the behaviour worth pinning.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from buzz.config import BuzzConfig
from buzz.dsp import pulse_phase_period

from tools.pulse_probe import (
    _print_active_phases_section,
    _print_drift_section,
    _print_profile_section,
    _print_width_sweep_section,
    capture,
    drift_rate_to_grid_hz,
    estimate_drift_rate,
    find_source_peaks,
    fold_pulses,
    main,
    matched_width_sweep,
    profile_sharpness,
    rectify,
    report,
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
        """A resolved pulse has an interior optimum - that is what makes the
        answer meaningful.  Monotonic growth means the pulse was not resolved."""
        scores = [fom for _, _, fom in matched_width_sweep(self._profile_for(3))]
        assert scores.index(max(scores)) < len(scores) - 1

    def test_wider_pulse_prefers_wider_window(self):
        narrow = max(matched_width_sweep(self._profile_for(3)), key=lambda r: r[2])[0]
        wide = max(matched_width_sweep(self._profile_for(10)), key=lambda r: r[2])[0]
        assert wide > narrow


# --------------------------------------------------------------------------- edge cases

class TestFoldPulsesEdges:
    def test_drift_that_walks_every_pulse_out_of_the_recording_returns_empty(self):
        """A drift candidate large enough to push every window past the end of the
        audio leaves nothing to average.  The search in estimate_drift_rate scores
        candidates blindly, so this has to return a profile rather than raise."""
        rectified = rectify(_pulse_train(seconds=0.4))
        profile, n = fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 1e6, HALF)
        assert n == 0
        assert len(profile) == 2 * HALF + 1
        assert not profile.any()


class TestFindSourcePeaksGrouping:
    def test_adjacent_maxima_are_one_peak_not_several(self):
        """A single pulse a few samples wide can have more than one local maximum on
        its own shoulder.  Counting those separately would report one arcing source
        as two or three, which is the number this section exists to state."""
        profile = np.full(133, 10.0)
        profile[60] = 100.0
        profile[61] = 90.0          # dip, then back up: two maxima 2 samples apart
        profile[62] = 100.0
        assert len(find_source_peaks(profile)) == 1

    def test_peaks_further_apart_than_the_gap_stay_separate(self):
        profile = np.full(133, 10.0)
        profile[40] = 100.0
        profile[90] = 95.0
        assert len(find_source_peaks(profile)) == 2


# --------------------------------------------------------------------------- capture

class _FakePipeline:
    def __init__(self, snapshot, ready=True):
        self.snapshot = snapshot
        self.ready = ready
        self.waits = 0

    def wait_for_data(self, samples, timeout):
        self.waits += 1
        return self.ready

    def get_snapshot(self, samples, align):
        self.requested = (samples, align)
        return self.snapshot


class TestCapture:
    def _sampler(self, pipeline):
        sampler = MagicMock()
        sampler.pipeline = pipeline
        return sampler

    def test_returns_the_snapshot_the_pipeline_hands_back(self):
        audio = _pulse_train(seconds=1.0)
        pipeline = _FakePipeline(audio)
        config = BuzzConfig()
        with patch('tools.pulse_probe.AudioSampler', return_value=self._sampler(pipeline)):
            assert capture(config, 1.0) is audio

    def test_the_snapshot_is_aligned_to_the_pulse_phase(self):
        """An unaligned snapshot would put the fold's window boundaries in an
        arbitrary place within the pulse period, blurring the profile for no reason."""
        config = BuzzConfig()
        pipeline = _FakePipeline(_pulse_train(seconds=1.0))
        with patch('tools.pulse_probe.AudioSampler', return_value=self._sampler(pipeline)):
            capture(config, 2.0)
        wanted, align = pipeline.requested
        assert wanted == int(2.0 * config.audio.sample_rate)
        assert align == pulse_phase_period(config.audio.sample_rate, config.audio.pulse_rate)

    def test_the_device_is_closed_even_when_the_snapshot_raises(self):
        """The sampler holds the input device open; leaking it would leave the sound
        card claimed by a diagnostic that has already exited."""
        pipeline = _FakePipeline(None)
        pipeline.get_snapshot = MagicMock(side_effect=RuntimeError('device vanished'))
        sampler = self._sampler(pipeline)
        with patch('tools.pulse_probe.AudioSampler', return_value=sampler):
            with pytest.raises(RuntimeError):
                capture(BuzzConfig(), 1.0)
        sampler.close.assert_called_once()

    def test_it_gives_up_at_the_deadline_rather_than_waiting_forever(self):
        """wait_for_data returning False every time must not spin until the heat
        death of the shack - the loop is bounded by its own deadline."""
        audio = _pulse_train(seconds=1.0)
        pipeline = _FakePipeline(audio, ready=False)
        with patch('tools.pulse_probe.AudioSampler', return_value=self._sampler(pipeline)):
            with patch('tools.pulse_probe.time.monotonic', side_effect=[0.0, 1.0, 10_000.0]):
                assert capture(BuzzConfig(), 1.0) is audio
        assert pipeline.waits == 1, 'it should stop asking once the deadline has passed'


# --------------------------------------------------------------------------- report sections

def _folded(phases=(0,), width=3, seconds=4.0):
    rectified = rectify(_pulse_train(seconds=seconds, phases=phases, width=width))
    return fold_pulses(rectified, SAMPLE_RATE, PULSE_RATE, 0.0, HALF)


class TestDriftSection:
    def test_it_reports_the_measured_grid_frequency(self, capsys):
        rectified = rectify(_pulse_train(seconds=4.0, drift_rate=0.0))
        profile, n_pulses = _print_drift_section(rectified, SAMPLE_RATE, PULSE_RATE, HALF)
        out = capsys.readouterr().out
        assert 'UTILITY LINE DRIFT' in out
        line = next(ln for ln in out.splitlines() if 'grid frequency' in ln)
        measured = float(line.split(':')[1].split('Hz')[0])
        assert measured == pytest.approx(60.0, abs=0.05), 'an undrifted train is a 60 Hz grid'
        assert n_pulses > 400 and len(profile) == 2 * HALF + 1

    def test_it_shows_what_the_drift_correction_bought(self, capsys):
        """The corrected-vs-uncorrected sharpness is the evidence that the drift
        search found something real, so it is printed rather than just used."""
        rectified = rectify(_pulse_train(seconds=4.0, drift_rate=6.0))
        _print_drift_section(rectified, SAMPLE_RATE, PULSE_RATE, HALF)
        out = capsys.readouterr().out
        assert 'sharpness' in out and 'uncorrected' in out
        assert 'smear avoided' in out


class TestActivePhasesSection:
    def test_one_source_is_reported_without_the_overlap_warning(self, capsys):
        profile, _ = _folded(phases=(0,))
        _print_active_phases_section(profile, SPP)
        out = capsys.readouterr().out
        assert 'distinct peaks    : 1' in out
        assert 'NOTE' not in out

    def test_three_phases_are_named_as_separate_utility_phases(self, capsys):
        third = int(round(SPP / 3))
        profile, _ = _folded(phases=(0, third, 2 * third))
        _print_active_phases_section(profile, SPP)
        out = capsys.readouterr().out
        assert 'distinct peaks    : 3' in out
        assert 'separate utility phase' in out
        assert 'NOTE' in out, 'overlapping sources make the width sweep unreliable'


class TestProfileSection:
    def test_it_marks_the_peak_and_says_how_many_pulses_were_averaged(self, capsys):
        profile, n_pulses = _folded()
        _print_profile_section(profile, n_pulses)
        out = capsys.readouterr().out
        assert '<-PEAK' in out
        assert '{} pulses averaged'.format(n_pulses) in out
        assert out.count('\n') > len(profile), 'one line per sample, plus the header'


class TestWidthSweepSection:
    def test_it_recommends_a_pulse_width(self, capsys):
        profile, _ = _folded(width=6)
        _print_width_sweep_section(profile, SAMPLE_RATE)
        out = capsys.readouterr().out
        assert 'best PULSE_WIDTH_SAMPLES' in out
        assert 'implied pulse duration' in out
        assert '<- current' in out, 'the width in use has to be visible for comparison'

    def test_an_unresolved_pulse_is_called_out_as_no_answer(self, capsys):
        """When the score never turns over, the best width is just the widest one
        tried, which is not a recommendation - saying so is the whole point."""
        profile = np.full(133, 10.0)
        profile[40:100] = 100.0          # a plateau far wider than MAX_PULSE_WIDTH
        _print_width_sweep_section(profile, SAMPLE_RATE)
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'no answer' in out


class TestReport:
    def test_it_prints_every_section(self, capsys):
        rectified = rectify(_pulse_train(seconds=4.0))
        report(rectified, BuzzConfig(), show_profile=True)
        out = capsys.readouterr().out
        assert 'UTILITY LINE DRIFT' in out
        assert 'ACTIVE UTILITY PHASES' in out
        assert 'FOLDED PULSE PROFILE' in out
        assert 'MATCHED WIDTH SWEEP' in out

    def test_quiet_mode_drops_only_the_sample_by_sample_listing(self, capsys):
        rectified = rectify(_pulse_train(seconds=4.0))
        report(rectified, BuzzConfig(), show_profile=False)
        out = capsys.readouterr().out
        assert 'FOLDED PULSE PROFILE' not in out
        assert 'MATCHED WIDTH SWEEP' in out, 'the recommendation is the point of the tool'


# --------------------------------------------------------------------------- entry point

class TestMain:
    def _run(self, argv_extra=()):
        config = BuzzConfig()
        audio = _pulse_train(seconds=4.0)
        argv = ['pulse_probe.py']
        argv.extend(argv_extra)
        with patch('sys.argv', argv):
            with patch('tools.pulse_probe.BuzzConfig.from_toml', return_value=config):
                with patch('tools.pulse_probe.capture', return_value=audio) as captured:
                    main()
        return captured

    def test_it_names_the_device_and_rates_before_capturing(self, capsys):
        """Printed before the capture blocks for several seconds, so whoever ran it
        can see it is listening to the right input while they wait."""
        self._run()
        out = capsys.readouterr().out
        assert 'device :' in out
        assert 'samples per pulse' in out
        assert 'capturing' in out

    def test_it_reports_on_what_it_captured(self, capsys):
        self._run()
        out = capsys.readouterr().out
        assert 'captured' in out
        assert 'UTILITY LINE DRIFT' in out

    def test_seconds_is_passed_through_to_the_capture(self):
        captured = self._run(['--seconds', '2.5'])
        assert captured.call_args.args[1] == 2.5

    def test_quiet_suppresses_the_profile_listing(self, capsys):
        self._run(['--quiet'])
        assert 'FOLDED PULSE PROFILE' not in capsys.readouterr().out

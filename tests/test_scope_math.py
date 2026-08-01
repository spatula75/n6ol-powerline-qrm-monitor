"""Tests for pure-numpy functions in scope.py (no Qt required)."""
import numpy as np
import pytest

from buzz.scope import (
    SCOPE_H, accumulate_trace, auto_range_full_scale, build_graticule,
    build_phosphor_colormap, extract_sweeps, full_scale_dbfs, n_complete_sweeps,
    resample_to_columns, sweep_start_offset, trace_rows, update_running_average,
    _HEADER_H, _TRACE_H, _H_DIVISIONS, _V_DIVISIONS, _GRATICULE_LINE, _GRATICULE_AXIS,
    _PRETRIGGER_DIVISIONS,
    _MIN_FULL_SCALE, _RANGE_HEADROOM, _RANGE_PERCENTILE, _RANGE_EMA_ALPHA,
)
from buzz.waterfall import _AXIS_H, _WATERFALL_H

SWEEP = 400          # pulse_phase_period(16000, 120)
PULSE_OFFSETS = (0, 133, 267)   # round(i * 16000/120) for i = 0, 1, 2
# Derived from the shipped constant rather than hardcoded, so these tests keep
# exercising the pretrigger production actually uses.
PRETRIGGER = round(SWEEP * _PRETRIGGER_DIVISIONS / _H_DIVISIONS)


def pulse_train(n_samples, phase, amplitude=1000.0):
    """Synthetic 120 pps train whose pulses sit at `phase` within each 400-sample period."""
    signal = np.zeros(n_samples, dtype=np.float32)
    for position in range(len(signal)):
        if (position - phase) % SWEEP in PULSE_OFFSETS:
            signal[position] = amplitude
    return signal


class TestSweepStartOffset:
    def test_backs_off_by_pretrigger(self):
        assert sweep_start_offset(200, pretrigger=40, sweep_samples=SWEEP) == 160

    def test_wraps_when_pretrigger_exceeds_phase(self):
        assert sweep_start_offset(10, pretrigger=40, sweep_samples=SWEEP) == 370

    def test_result_always_within_one_sweep(self):
        for phase in range(0, SWEEP, 7):
            assert 0 <= sweep_start_offset(phase, 40, SWEEP) < SWEEP

    def test_phase_at_origin_with_no_pretrigger_is_zero(self):
        assert sweep_start_offset(0, pretrigger=0, sweep_samples=SWEEP) == 0


class TestNCompleteSweeps:
    def test_exact_multiple(self):
        assert n_complete_sweeps(SWEEP * 3, 0, SWEEP) == 3

    def test_partial_sweep_is_dropped(self):
        assert n_complete_sweeps(SWEEP * 3 - 1, 0, SWEEP) == 2

    def test_start_offset_consumes_capacity(self):
        assert n_complete_sweeps(SWEEP * 3, SWEEP // 2, SWEEP) == 2

    def test_start_past_end_is_zero(self):
        assert n_complete_sweeps(100, 200, SWEEP) == 0

    def test_zero_sweep_width_is_zero(self):
        assert n_complete_sweeps(1000, 0, 0) == 0


class TestExtractSweeps:
    def test_shape(self):
        sweeps = extract_sweeps(np.zeros(SWEEP * 4), 0, SWEEP)
        assert sweeps.shape == (4, SWEEP)

    def test_empty_when_nothing_fits(self):
        sweeps = extract_sweeps(np.zeros(SWEEP - 1), 0, SWEEP)
        assert sweeps.shape == (0, SWEEP)
        assert sweeps.size == 0

    def test_content_starts_at_offset(self):
        samples = np.arange(SWEEP * 3, dtype=np.float32)
        sweeps = extract_sweeps(samples, 10, SWEEP)
        assert sweeps[0, 0] == 10
        assert sweeps[1, 0] == 10 + SWEEP


class TestSynchronisation:
    """The property the whole display exists for: a pulse falls on the same column
    regardless of where in the pulse period the signal happens to sit."""

    @pytest.mark.parametrize('phase', [0, 37, 133, 200, 267, 399])
    def test_triggered_pulse_lands_at_pretrigger_column(self, phase):
        signal = pulse_train(SWEEP * 6, phase)
        start = sweep_start_offset(phase, PRETRIGGER, SWEEP)
        sweeps = extract_sweeps(signal, start, SWEEP)
        assert len(sweeps) >= 4
        for sweep in sweeps:
            assert int(np.argmax(sweep)) == PRETRIGGER

    @pytest.mark.parametrize('phase', [0, 37, 133, 200, 267, 399])
    def test_all_three_pulses_land_on_fixed_columns(self, phase):
        signal = pulse_train(SWEEP * 6, phase)
        start = sweep_start_offset(phase, PRETRIGGER, SWEEP)
        sweeps = extract_sweeps(signal, start, SWEEP)
        expected = sorted((PRETRIGGER + offset) % SWEEP for offset in PULSE_OFFSETS)
        for sweep in sweeps:
            assert sorted(np.flatnonzero(sweep).tolist()) == expected

    def test_untriggered_pulse_position_follows_the_signal(self):
        """Control: without the trigger phase the pulse moves, so the test above is
        actually measuring synchronisation rather than an artefact of the fixture."""
        positions = set()
        for phase in (0, 37, 133, 200):
            signal = pulse_train(SWEEP * 6, phase)
            sweeps = extract_sweeps(signal, sweep_start_offset(0, PRETRIGGER, SWEEP), SWEEP)
            positions.add(int(np.argmax(sweeps[0])))
        assert len(positions) > 1


class TestAutoRangeFullScale:
    def test_empty_input_holds_previous(self):
        assert auto_range_full_scale(np.empty((0, SWEEP)), 500.0) == 500.0

    def test_never_below_minimum(self):
        silent = np.zeros((4, SWEEP), dtype=np.float32)
        assert auto_range_full_scale(silent, _MIN_FULL_SCALE) >= _MIN_FULL_SCALE

    def test_silence_cannot_collapse_the_scale(self):
        """The failure mode the floor exists for: near-silent input must not shrink
        full scale toward zero, which would amplify dither to full deflection."""
        scale = 2048.0
        dither = np.random.default_rng(0).normal(0, 0.5, size=(4, SWEEP))
        for _ in range(500):
            scale = auto_range_full_scale(dither, scale)
        assert scale == pytest.approx(_MIN_FULL_SCALE)

    def test_moves_toward_but_not_onto_the_measurement(self):
        loud = np.full((4, SWEEP), 8000.0)
        moved = auto_range_full_scale(loud, 1000.0)
        assert moved > 1000.0
        assert moved < 8000.0 * _RANGE_HEADROOM

    def test_single_step_matches_ema_weight(self):
        loud = np.full((4, SWEEP), 8000.0)
        target = 8000.0 * _RANGE_HEADROOM
        expected = 1000.0 + _RANGE_EMA_ALPHA * (target - 1000.0)
        assert auto_range_full_scale(loud, 1000.0) == pytest.approx(expected)

    def test_converges_to_percentile_times_headroom(self):
        rng = np.random.default_rng(1)
        sweeps = rng.normal(0, 1000.0, size=(4, SWEEP))
        target = float(np.percentile(np.abs(sweeps), _RANGE_PERCENTILE)) * _RANGE_HEADROOM
        scale = 100.0
        for _ in range(500):
            scale = auto_range_full_scale(sweeps, scale)
        assert scale == pytest.approx(target, rel=0.01)

    def test_reaches_into_the_pulse_tail(self):
        """Pulses occupy ~2% of samples, so the percentile has to sit above the noise
        population or the pulses would be scaled clean off the screen."""
        sweeps = np.tile(pulse_train(SWEEP, 0, amplitude=10000.0), (4, 1))
        sweeps = sweeps + np.random.default_rng(2).normal(0, 100.0, sweeps.shape)
        scale = 100.0
        for _ in range(500):
            scale = auto_range_full_scale(sweeps, scale)
        assert scale > 1000.0


class TestFullScaleDbfs:
    def test_int16_full_scale_is_zero_dbfs(self):
        assert full_scale_dbfs(32768.0) == pytest.approx(0.0)

    def test_halving_the_scale_drops_six_db(self):
        assert full_scale_dbfs(16384.0) == pytest.approx(-6.02, abs=0.01)

    def test_minimum_scale_is_about_minus_sixty(self):
        """_MIN_FULL_SCALE's comment claims ~-60 dBFS; hold it to that."""
        assert full_scale_dbfs(_MIN_FULL_SCALE) == pytest.approx(-60.2, abs=0.1)

    def test_goes_positive_when_chasing_a_clipping_signal(self):
        """Headroom past the rail is reported rather than clamped, so an overdriven
        input reads as overdriven instead of pinned at 0."""
        assert full_scale_dbfs(32768.0 * _RANGE_HEADROOM) > 0.0

    def test_monotonic_in_full_scale(self):
        scales = np.array([32.0, 256.0, 2048.0, 16384.0, 32768.0])
        values = [full_scale_dbfs(s) for s in scales]
        assert np.all(np.diff(values) > 0)

    def test_every_auto_ranged_value_is_representable(self):
        """auto_range_full_scale is floored at _MIN_FULL_SCALE, so the log can never
        be handed a zero or negative argument."""
        silent = np.zeros((4, SWEEP), dtype=np.float32)
        assert full_scale_dbfs(auto_range_full_scale(silent, _MIN_FULL_SCALE)) < 0.0


class TestResampleToColumns:
    def test_length(self):
        assert len(resample_to_columns(np.zeros(SWEEP), 641)) == 641

    def test_endpoints_preserved(self):
        sweep = np.arange(SWEEP, dtype=np.float64)
        out = resample_to_columns(sweep, 641)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(SWEEP - 1)

    def test_linear_ramp_stays_linear(self):
        out = resample_to_columns(np.arange(SWEEP, dtype=np.float64), 641)
        assert np.allclose(np.diff(out), np.diff(out)[0])

    def test_upsamples_without_overshoot(self):
        """Linear interpolation must not invent values outside the signal's range."""
        sweep = pulse_train(SWEEP, 0, amplitude=1000.0)
        out = resample_to_columns(sweep, 641)
        assert out.min() >= sweep.min()
        assert out.max() <= sweep.max()


class TestTraceRows:
    HEIGHT = 101   # odd, so the centre row is exact

    def test_bipolar_zero_is_centred(self):
        rows = trace_rows(np.array([0.0]), 1000.0, self.HEIGHT)
        assert rows[0] == (self.HEIGHT - 1) // 2

    def test_bipolar_positive_full_scale_is_top(self):
        rows = trace_rows(np.array([1000.0]), 1000.0, self.HEIGHT)
        assert rows[0] == 0

    def test_bipolar_negative_full_scale_is_bottom(self):
        rows = trace_rows(np.array([-1000.0]), 1000.0, self.HEIGHT)
        assert rows[0] == self.HEIGHT - 1

    def test_overdrive_clips_to_rails_rather_than_wrapping(self):
        rows = trace_rows(np.array([50000.0, -50000.0]), 1000.0, self.HEIGHT)
        assert rows[0] == 0
        assert rows[1] == self.HEIGHT - 1

    def test_unipolar_zero_is_bottom(self):
        rows = trace_rows(np.array([0.0]), 1000.0, self.HEIGHT, bipolar=False)
        assert rows[0] == self.HEIGHT - 1

    def test_unipolar_full_scale_is_top(self):
        rows = trace_rows(np.array([1000.0]), 1000.0, self.HEIGHT, bipolar=False)
        assert rows[0] == 0

    def test_always_indexable(self):
        rng = np.random.default_rng(3)
        rows = trace_rows(rng.normal(0, 50000.0, 5000), 1000.0, self.HEIGHT)
        assert rows.min() >= 0
        assert rows.max() < self.HEIGHT
        assert rows.dtype == np.intp


class TestAccumulateTrace:
    def test_flat_trace_fills_one_row(self):
        buffer = np.zeros((21, 10), dtype=np.float32)
        accumulate_trace(buffer, np.full(11, 10, dtype=np.intp), 1.0)
        assert np.all(buffer[10] == 1.0)
        assert buffer.sum() == pytest.approx(10.0)

    def test_steep_segment_is_connected(self):
        """The reason spans are filled at all: a fast-moving trace must not break
        into disconnected dots."""
        buffer = np.zeros((21, 2), dtype=np.float32)
        accumulate_trace(buffer, np.array([0, 20, 0], dtype=np.intp), 1.0)
        assert np.all(buffer[:, 0] > 0)

    def test_intensity_accumulates_across_sweeps(self):
        buffer = np.zeros((21, 10), dtype=np.float32)
        rows = np.full(11, 10, dtype=np.intp)
        accumulate_trace(buffer, rows, 0.5)
        accumulate_trace(buffer, rows, 0.5)
        assert buffer[10, 0] == pytest.approx(1.0)

    def test_span_touching_bottom_edge_does_not_overflow(self):
        buffer = np.zeros((21, 4), dtype=np.float32)
        accumulate_trace(buffer, np.full(5, 20, dtype=np.intp), 1.0)
        assert np.all(buffer[20] == 1.0)
        assert buffer[:20].sum() == 0.0

    def test_leaves_untouched_rows_at_zero(self):
        buffer = np.zeros((21, 10), dtype=np.float32)
        accumulate_trace(buffer, np.full(11, 5, dtype=np.intp), 1.0)
        assert buffer[6:].sum() == 0.0
        assert buffer[:5].sum() == 0.0


class TestBuildGraticule:
    def test_shape(self):
        assert build_graticule(_TRACE_H, 640).shape == (_TRACE_H, 640)

    def test_values_within_intensity_range(self):
        grid = build_graticule(_TRACE_H, 640)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_centre_rules_are_brighter(self):
        grid = build_graticule(_TRACE_H, 640)
        assert grid[0, 320] == pytest.approx(_GRATICULE_AXIS)
        assert grid[round(_TRACE_H / 2), 0] == pytest.approx(_GRATICULE_AXIS)

    def test_division_lines_are_dimmer_than_centre(self):
        grid = build_graticule(_TRACE_H, 640)
        assert grid[0, 64] == pytest.approx(_GRATICULE_LINE)
        assert _GRATICULE_LINE < _GRATICULE_AXIS

    def test_division_counts(self):
        grid = build_graticule(_TRACE_H, 640)
        lit_columns = np.flatnonzero(grid[0] > 0)
        assert len(lit_columns) == _H_DIVISIONS - 1
        lit_rows = np.flatnonzero(grid[:, 0] > 0)
        assert len(lit_rows) == _V_DIVISIONS - 1

    def test_graticule_stays_below_a_real_trace(self):
        """Composited with a maximum, so any drawn trace must outrank the grid."""
        assert _GRATICULE_AXIS < 0.55   # _SWEEP_INTENSITY


class TestBuildPhosphorColormap:
    def test_shape_and_dtype(self):
        lut = build_phosphor_colormap()
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_dark_end_is_near_black(self):
        assert int(build_phosphor_colormap()[0].sum()) < 50

    def test_hot_end_is_near_white(self):
        assert int(build_phosphor_colormap()[255].min()) > 200

    def test_brightness_increases_monotonically(self):
        brightness = build_phosphor_colormap().astype(np.int32).sum(axis=1)
        assert np.all(np.diff(brightness) >= 0)

    def test_is_blue_green_not_red(self):
        """The whole point of the palette: green and blue lead red everywhere except
        the white-hot core."""
        lut = build_phosphor_colormap().astype(np.int32)
        mid = lut[:216]
        assert np.all(mid[:, 1] >= mid[:, 0])
        assert np.all(mid[:, 2] >= mid[:, 0])

    def test_dim_end_leans_teal(self):
        """The 'blue' half of blue-green: below the crossover at index 68 the ramp is
        blue-biased, which is what the faint persistence halo renders in."""
        lut = build_phosphor_colormap().astype(np.int32)
        assert np.all(lut[:64, 2] >= lut[:64, 1])

    def test_green_leads_above_the_crossover(self):
        """...and the 'green' half: a well-driven trace reads green, not blue."""
        lut = build_phosphor_colormap().astype(np.int32)
        assert np.all(lut[72:, 1] >= lut[72:, 2])

    def test_no_red_through_the_normal_trace_range(self):
        """What actually makes this a blue-green phosphor rather than a heat map: red
        is entirely absent between the teal floor and the white-hot core, so ordinary
        signal can never render warm."""
        lut = build_phosphor_colormap().astype(np.int32)
        assert np.all(lut[64:140, 0] == 0)


class TestUpdateRunningAverage:
    def test_seeds_from_first_frame(self):
        sweeps = np.array([[1.0, -3.0], [3.0, -1.0]])
        assert np.allclose(update_running_average(None, sweeps, 0.05), [2.0, 2.0])

    def test_empty_input_holds_previous(self):
        previous = np.array([5.0, 5.0])
        assert update_running_average(previous, np.empty((0, 2)), 0.05) is previous

    def test_none_and_empty_stays_none(self):
        assert update_running_average(None, np.empty((0, 2)), 0.05) is None

    def test_rectifies_before_averaging(self):
        """A negative-going pulse must contribute its magnitude, not cancel a
        positive one - the envelope is what carries the signal."""
        sweeps = np.array([[-1000.0], [1000.0]])
        assert update_running_average(None, sweeps, 0.05)[0] == pytest.approx(1000.0)

    def test_ema_blends_toward_new_value(self):
        previous = np.array([0.0])
        blended = update_running_average(previous, np.array([[100.0]]), 0.25)
        assert blended[0] == pytest.approx(25.0)

    def test_random_carrier_phase_preserves_the_envelope(self):
        """The physics the docstring claims: bursts whose carrier phase is random
        sweep-to-sweep survive rectified averaging, and would not survive raw
        averaging."""
        rng = np.random.default_rng(4)
        t = np.arange(SWEEP)
        envelope = pulse_train(SWEEP, 0, amplitude=1.0)
        bursts = np.array([
            envelope * np.cos(2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi))
            for _ in range(200)
        ])
        raw_mean = np.abs(bursts.mean(axis=0)).max()

        average = None
        for sweep in bursts:
            average = update_running_average(average, sweep[None, :], 0.05)

        assert average.max() > 5 * raw_mean

    def test_averaging_pulls_noise_down(self):
        rng = np.random.default_rng(5)
        average = None
        for _ in range(400):
            average = update_running_average(
                average, rng.normal(0, 1.0, size=(4, SWEEP)), 0.05)
        # Mean |x| of unit Gaussian noise is sqrt(2/pi) ~ 0.798; averaging shrinks the
        # scatter around it without moving the mean.
        assert average.mean() == pytest.approx(0.798, rel=0.05)
        assert average.std() < 0.15


class TestPretrigger:
    """Where the triggering pulse sits, and why it isn't a whole division."""

    WIDTH = 640   # ScopeWidget takes its width from the waterfall; 128 bins x 5 px

    def _column(self):
        pretrigger = round(SWEEP * _PRETRIGGER_DIVISIONS / _H_DIVISIONS)
        return pretrigger * self.WIDTH / SWEEP

    def test_lands_clear_of_any_graticule_line(self):
        """A whole number of divisions would draw the pulse directly under a grid
        line, hiding its leading edge and reading as pinned to the grid."""
        division_px = self.WIDTH / _H_DIVISIONS
        distance_to_line = abs((self._column() / division_px) % 1)
        assert 0.25 < distance_to_line < 0.75

    def test_leaves_room_for_the_leading_edge_and_halo(self):
        assert self._column() >= 0.5 * self.WIDTH / _H_DIVISIONS

    def test_does_not_push_the_train_off_the_right(self):
        """Three pulses have to remain visible after the lead-in."""
        assert _PRETRIGGER_DIVISIONS < _H_DIVISIONS / 3

    def test_matches_the_documented_position(self):
        assert self._column() == pytest.approx(96.0, abs=1.0)


class TestPanelGeometry:
    def test_scope_header_matches_waterfall_axis(self):
        """The two header bars are drawn at the same height so they read as one
        design; the constant is duplicated to avoid a circular import."""
        assert _HEADER_H == _AXIS_H

    def test_scope_and_waterfall_are_the_same_height(self):
        assert SCOPE_H == _WATERFALL_H

    def test_scope_height_is_header_plus_trace(self):
        assert SCOPE_H == _HEADER_H + _TRACE_H

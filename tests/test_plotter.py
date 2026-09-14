"""Tests for Plotter: moving average, daily graph, and summary graph generation."""

import gc
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest

from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter, _bar_color, _gc_guarded, _smooth

_TZ = ZoneInfo('America/Los_Angeles')


def _make_plotter(tmp_path: Path) -> tuple[Plotter, CsvStore]:
    cfg = BuzzConfig()
    cfg.station.path = str(tmp_path)
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.station.noise_floor = -98.0
    cfg.station.noise_min_snr = 12.0
    cfg.station.audio_rf_conversion_db = -32.0
    cfg.audio.pulse_rate = 120
    store = CsvStore(cfg)
    return Plotter(cfg, store), store


def _write_csv(path: Path, n_rows: int = 10) -> None:
    tz_offset = '-08:00'
    lines = ['ISO datetime,120pps SNR,120pps signal dB,Noise floor dB,T,H,S,W,G,B']
    for i in range(n_rows):
        ts = f'2024-01-15T10:{i:02d}:00{tz_offset}'
        lines.append(f'{ts},15.0,-80.0,-95.0,68,52,300,7,12,225')
    path.write_text('\n'.join(lines) + '\n')


def _write_csv_with_lock(path: Path, lock_statuses: list[str]) -> None:
    """Write a new-format CSV with Signal Lock Status column."""
    tz_offset = '-08:00'
    lines = ['ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,T,H,S,W,G,B']
    for i, status in enumerate(lock_statuses):
        ts = f'2024-01-15T10:{i:02d}:00{tz_offset}'
        sig = '-95.0' if status == 'none' else '-80.0'
        lines.append(f'{ts},15.0,{sig},-95.0,{status},68,52,300,7,12,225')
    path.write_text('\n'.join(lines) + '\n')


class TestGcGuarded:
    """_gc_guarded disables GC around the call (PySide6/shiboken crash workaround -
    see the decorator's docstring in plotter.py) and forces a collect() afterward
    (matplotlib leak workaround). These tests exercise the decorator directly rather
    than through Plotter, since the behavior has nothing to do with plotting."""

    @pytest.fixture(autouse=True)
    def restore_gc_state(self):
        was_enabled = gc.isenabled()
        yield
        if was_enabled:
            gc.enable()
        else:
            gc.disable()

    def test_gc_disabled_during_call(self):
        observed = {}

        @_gc_guarded
        def func():
            observed['enabled'] = gc.isenabled()

        gc.enable()
        func()
        assert observed['enabled'] is False

    def test_gc_re_enabled_after_call_when_previously_enabled(self):
        @_gc_guarded
        def func():
            pass

        gc.enable()
        func()
        assert gc.isenabled() is True

    def test_gc_left_disabled_if_already_disabled_before_call(self):
        @_gc_guarded
        def func():
            pass

        gc.disable()
        func()
        assert gc.isenabled() is False

    def test_gc_re_enabled_even_if_wrapped_function_raises(self):
        @_gc_guarded
        def func():
            raise ValueError('boom')

        gc.enable()
        with pytest.raises(ValueError):
            func()
        assert gc.isenabled() is True

    def test_return_value_passes_through(self):
        @_gc_guarded
        def func():
            return 42

        assert func() == 42

    def test_collect_runs_after_call(self):
        calls = []

        @_gc_guarded
        def func():
            assert calls == []   # not yet called during the guarded call

        with patch('buzz.plotter.gc.collect', side_effect=lambda: calls.append(1)):
            func()
        assert calls == [1]


class TestSmooth:
    def test_simple_moving_average(self):
        result = _smooth([1.0, 2.0, 3.0, 4.0, 5.0], points=3)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_points_equal_one_returns_same_values(self):
        data = [10.0, 20.0, 30.0]
        result = _smooth(data, points=1)
        np.testing.assert_allclose(result, data)

    def test_output_shorter_than_input(self):
        result = _smooth([1.0, 2.0, 3.0, 4.0, 5.0], points=3)
        assert len(result) == 3  # len(data) - points + 1

    def test_uniform_data_unchanged_by_smoothing(self):
        data = [5.0] * 10
        result = _smooth(data, points=3)
        np.testing.assert_allclose(result, [5.0] * 8)


class TestBarColor:
    def test_max_is_firebrick(self):
        assert _bar_color(100) == 'firebrick'

    def test_above_high_threshold_is_indianred(self):
        assert _bar_color(93) == 'indianred'

    def test_above_elevated_threshold_is_lightcoral(self):
        assert _bar_color(86) == 'lightcoral'

    def test_at_elevated_threshold_starts_gradient(self):
        assert _bar_color(85) == '#87ceeb'   # skyblue, matching the legend

    def test_zero_fades_to_near_white(self):
        assert _bar_color(0) == '#fefefe'


class TestGenerateGraphFromCsv:
    def test_creates_png_file(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=10)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output)
        assert output.exists()

    def test_creates_png_with_smoothing(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=20)
        output = tmp_path / 'out_smooth.png'
        plotter.generate_graph_from_csv(csv_path, output, smooth=6)
        assert output.exists()

    def test_returns_early_if_too_few_rows_for_smooth(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=4)   # fewer than smooth=6
        output = tmp_path / 'should_not_exist.png'
        plotter.generate_graph_from_csv(csv_path, output, smooth=6)
        assert not output.exists()

    def test_header_lines_skipped_without_error(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=5)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output)
        assert output.exists()

    def test_accepts_path_or_string(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=5)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(str(csv_path), str(output))
        assert output.exists()

    def test_old_format_csv_without_lock_column_still_renders(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv(csv_path, n_rows=10)   # no Signal Lock Status column
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output)
        assert output.exists()

    def test_all_none_lock_status_renders(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv_with_lock(csv_path, ['none'] * 10)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output)
        assert output.exists()

    def test_mixed_lock_status_renders(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv_with_lock(csv_path, ['full'] * 5 + ['none'] * 5)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output)
        assert output.exists()

    def test_smooth_with_lock_status_renders(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_csv_with_lock(csv_path, ['full'] * 10 + ['none'] * 10)
        output = tmp_path / 'out.png'
        plotter.generate_graph_from_csv(csv_path, output, smooth=6)
        assert output.exists()


def _write_frequency_csv(path: Path, values: list[str], day: str = '2024-01-15') -> None:
    """A new-format CSV whose grid-frequency column holds `values`, blank for no lock."""
    lines = ['ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
             'Grid frequency (Hz),Phase drift (samples/s),Temperature (F),Humidity (%),'
             'Solar radiation (w/m^2),Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)']
    for minute, value in enumerate(values):
        lines.append(f'{day}T10:{minute:02d}:00-08:00,15.0,-80.0,-95.0,full,'
                     f'{value},-6.1,68,52,300,7,12,225')
    path.write_text('\n'.join(lines) + '\n')


class TestFrequencySeries:
    """Splitting readings into a drawable trace and the times that left the band."""

    def _series(self, tmp_path, readings, low=59.9, high=60.1):
        plotter, _ = _make_plotter(tmp_path)
        pairs = [(datetime(2024, 1, 15, 10, i, tzinfo=_TZ), v) for i, v in enumerate(readings)]
        return plotter._frequency_series(pairs, low, high)

    def test_a_minute_with_no_lock_becomes_a_gap(self, tmp_path):
        """NaN is what makes matplotlib break the line instead of spanning the gap.

        Joining across an unlocked stretch would draw a straight segment through
        readings nobody took, which on a chart of a wandering figure reads as an hour
        of unusual stability.
        """
        series = self._series(tmp_path, [60.0, None, 60.02])
        assert series.trace[0] == 60.0
        assert np.isnan(series.trace[1])
        assert series.trace[2] == 60.02

    def test_a_reading_above_the_band_is_clamped_and_recorded(self, tmp_path):
        series = self._series(tmp_path, [60.5])
        assert series.trace[0] == 60.1
        assert len(series.above_at) == 1 and series.below_at == []

    def test_a_reading_below_the_band_is_clamped_and_recorded(self, tmp_path):
        series = self._series(tmp_path, [59.0])
        assert series.trace[0] == 59.9
        assert len(series.below_at) == 1 and series.above_at == []

    def test_a_reading_exactly_on_the_edge_is_not_an_excursion(self, tmp_path):
        """The band is inclusive, so an edge value is drawn as itself, not marked."""
        series = self._series(tmp_path, [59.9, 60.1])
        assert series.outside_count == 0

    def test_the_count_covers_both_edges(self, tmp_path):
        series = self._series(tmp_path, [60.5, 59.0, 60.0])
        assert series.outside_count == 2


class TestGenerateFrequencyGraph:
    def _limits(self, tmp_path, values, pulse_rate=120):
        """Render a chart and report the axis limits matplotlib ended up with."""
        plotter, _ = _make_plotter(tmp_path)
        plotter._config.audio.pulse_rate = pulse_rate
        csv_path = tmp_path / 'data.csv'
        _write_frequency_csv(csv_path, values)
        captured = {}
        real_subplots = plt.subplots

        def capture(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured['axes'] = axes
            return figure, axes

        with patch('buzz.plotter.plt.subplots', capture):
            plotter.generate_frequency_graph(csv_path, tmp_path / 'out.png',
                                             datetime(2024, 1, 15, 10, 30, tzinfo=_TZ))
        return captured['axes']

    def test_creates_png_file(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_frequency_csv(csv_path, ['60.01', '60.02'])
        output = tmp_path / 'out.png'
        plotter.generate_frequency_graph(csv_path, output,
                                         datetime(2024, 1, 15, 10, 30, tzinfo=_TZ))
        assert output.exists()

    def test_a_day_with_no_readings_still_writes_a_chart(self, tmp_path):
        """The output has one fixed name and is overwritten in place.

        Returning early would leave yesterday's chart sitting there under today's
        name, looking current, which is the staleness the fixed name invites.  An
        empty chart for an empty morning is the honest answer.
        """
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_frequency_csv(csv_path, ['', '', ''])
        output = tmp_path / 'out.png'
        output.write_bytes(b'yesterday')
        plotter.generate_frequency_graph(csv_path, output,
                                         datetime(2024, 1, 15, 0, 1, tzinfo=_TZ))
        assert output.read_bytes() != b'yesterday', (
            'The stale chart from the previous day was left in place under a name '
            'that says it is current.'
        )

    def test_the_y_axis_is_the_nominal_frequency_plus_or_minus_a_tenth(self, tmp_path):
        axes = self._limits(tmp_path, ['60.01'])
        low, high = axes.get_ylim()
        assert (round(low, 6), round(high, 6)) == (59.9, 60.1)

    def test_a_fifty_hertz_grid_centres_on_fifty(self, tmp_path):
        """An arc fires on both peaks, so 100 pps is a 50 Hz grid."""
        axes = self._limits(tmp_path, ['50.01'], pulse_rate=100)
        low, high = axes.get_ylim()
        assert (round(low, 6), round(high, 6)) == (49.9, 50.1)

    def test_the_axis_does_not_stretch_to_fit_the_data(self, tmp_path):
        """Fixed scale is what lets one hour's chart be compared with the next."""
        axes = self._limits(tmp_path, ['60.5', '59.4'])
        assert (round(axes.get_ylim()[0], 6), round(axes.get_ylim()[1], 6)) == (59.9, 60.1)

    def test_the_x_axis_runs_from_midnight_to_now_with_a_pad_at_each_end(self, tmp_path):
        """The newest reading belongs at the right-hand end, not partway along an axis
        mostly waiting to be used.

        The pad is 5% of the span either side, which is what the reference chart this
        layout follows leaves matplotlib's default margin at.  Taking it as a fraction
        rather than as a fixed number of minutes keeps it proportionate at any hour.
        """
        axes = self._limits(tmp_path, ['60.01', '60.02'])   # rendered as at 10:30
        start, end = (mdates.num2date(v).astimezone(_TZ) for v in axes.get_xlim())
        midnight = datetime(2024, 1, 15, tzinfo=_TZ)
        now = datetime(2024, 1, 15, 10, 30, tzinfo=_TZ)
        expected_pad = (now - midnight).total_seconds() * 0.05
        assert (midnight - start).total_seconds() == pytest.approx(expected_pad, rel=0.001)
        assert (end - now).total_seconds() == pytest.approx(expected_pad, rel=0.001)

    def test_just_after_midnight_the_axis_still_has_a_width(self, tmp_path):
        """An axis of zero width cannot be drawn, and at 00:00 the day is zero wide.

        The chart is rendered on the hour, so this is the state it is in for the first
        render of every day.
        """
        plotter, _ = _make_plotter(tmp_path)
        csv_path = tmp_path / 'data.csv'
        _write_frequency_csv(csv_path, [''])
        output = tmp_path / 'out.png'
        plotter.generate_frequency_graph(csv_path, output,
                                         datetime(2024, 1, 15, 0, 0, tzinfo=_TZ))
        assert output.exists()

    def test_the_trace_is_dark_orange(self, tmp_path):
        axes = self._limits(tmp_path, ['60.01', '60.02'])
        assert axes.get_lines()[0].get_color() == 'darkorange'

    def test_horizontal_grid_lines_every_25_millihertz(self, tmp_path):
        """Set explicitly rather than left to the autolocator.

        Both axes have a fixed span here, so the lines can sit on round numbers and
        stay there from one hour's chart to the next.  An autolocator is free to
        choose differently as the data changes.
        """
        axes = self._limits(tmp_path, ['60.01'])
        low, high = axes.get_ylim()
        ticks = [t for t in axes.get_yticks() if low - 1e-9 <= t <= high + 1e-9]
        steps = {round(b - a, 6) for a, b in zip(ticks, ticks[1:])}
        assert steps == {0.025}, f'Horizontal grid lines were spaced {steps}, not 0.025 Hz.'

    def test_the_hours_are_labelled_and_ruled_every_two_hours_on_even_hours(self, tmp_path):
        """One locator sets both, so the labels are on the lines rather than between.

        Left to the autolocator this labels every three hours, and none of those
        labels would fall on a two-hourly grid line.
        """
        axes = self._limits(tmp_path, ['60.01'])
        start, end = axes.get_xlim()
        hours = [mdates.num2date(t).astimezone(_TZ).hour
                 for t in axes.get_xticks() if start <= t <= end]
        # Rendered as at 10:30, so the marks run 00:00 through 10:00.
        assert hours == [0, 2, 4, 6, 8, 10], (
            f'Vertical grid lines and hour labels fell on {hours}, not the even hours '
            'from midnight to the last one before now.'
        )

    def test_the_grid_is_drawn_behind_the_trace(self, tmp_path):
        """A light gray line crossing the orange one reads as a break in the trace."""
        axes = self._limits(tmp_path, ['60.01'])
        assert axes.get_axisbelow() is True

    def test_the_axes_box_matches_the_reference_chart(self, tmp_path):
        """The layout is measured off kestrelgrid.com's WECC frequency panel, so the
        two can be read side by side.

        Its axes box is 1861 x 579 px inside a 1999 px figure, with the left edge at
        123 px.  Those are the numbers a reader would notice if they drifted, since
        the point is that one chart overlays the other.
        """
        axes = self._limits(tmp_path, ['60.01'])
        figure = axes.get_figure()
        width, height = (round(v * figure.dpi) for v in figure.get_size_inches())
        box = axes.get_position()
        assert (width, height) == (1999, 742)
        assert round(box.x0 * width) == 123
        assert (round(box.width * width), round(box.height * height)) == (1861, 579)

    def test_the_labels_are_monospace(self, tmp_path):
        """The reference is monospace throughout, and a column of proportional
        timestamps beside a column of monospace ones is the difference a reader
        notices first."""
        axes = self._limits(tmp_path, ['60.01'])
        assert axes.get_xticklabels()[0].get_fontfamily() == ['monospace']

    def test_the_hour_labels_are_rotated_clear_of_each_other(self, tmp_path):
        """Twelve HH:MM:SS labels across the day collide when drawn flat."""
        axes = self._limits(tmp_path, ['60.01'])
        label = axes.get_xticklabels()[0]
        assert label.get_rotation() == 45
        assert label.get_text().count(':') == 2


class TestGenerateSummaryGraph:
    def _time_data(self) -> dict[time, int]:
        return {
            time(10, 0): 5,
            time(10, 15): 10,
            time(10, 30): 8,
            time(14, 30): 3,
            time(18, 0): 10,  # max → 100%
        }

    def test_creates_png_file(self, tmp_path):
        plotter, store = _make_plotter(tmp_path)
        store.read_range_scores = MagicMock(return_value=self._time_data())
        output = tmp_path / 'summary.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(output, start)
        assert output.exists()

    def test_returns_early_when_no_data(self, tmp_path):
        plotter, store = _make_plotter(tmp_path)
        store.read_range_scores = MagicMock(return_value={})
        output = tmp_path / 'should_not_exist.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(output, start)
        assert not output.exists()

    def test_no_data_early_return_leaves_no_open_figures(self, tmp_path):
        import matplotlib.pyplot as plt
        plotter, store = _make_plotter(tmp_path)
        store.read_range_scores = MagicMock(return_value={})
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(tmp_path / 'nope.png', start)
        assert plt.get_fignums() == []

    def test_accepts_path_or_string(self, tmp_path):
        plotter, store = _make_plotter(tmp_path)
        store.read_range_scores = MagicMock(return_value=self._time_data())
        output = tmp_path / 'summary.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(str(output), start)
        assert output.exists()

    def test_passes_date_range_to_store(self, tmp_path):
        plotter, store = _make_plotter(tmp_path)
        store.read_range_scores = MagicMock(return_value={})
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        output = tmp_path / 'summary.png'
        plotter.generate_summary_graph(output, start)
        store.read_range_scores.assert_called_once()
        call_start = store.read_range_scores.call_args[0][0]
        assert call_start == start

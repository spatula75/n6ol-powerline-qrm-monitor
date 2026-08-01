"""Tests for Plotter: moving average, daily graph, and summary graph generation."""

import gc
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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
    cfg.station.distance_attenuation = 29.54
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

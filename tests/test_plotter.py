"""Tests for Plotter: moving average, daily graph, and summary graph generation."""

from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter

_TZ = ZoneInfo('America/Los_Angeles')


def _make_plotter(tmp_path: Path) -> tuple[Plotter, MagicMock]:
    cfg = BuzzConfig()
    cfg.station.path = str(tmp_path)
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.station.noise_floor = -98.0
    cfg.station.noise_min_snr = 12.0
    cfg.station.audio_rf_conversion_db = -32.0
    cfg.station.distance_attenuation = 29.54
    cfg.audio.pulse_rate = 120
    store_mock = MagicMock(spec=CsvStore)
    return Plotter(cfg, store_mock), store_mock


def _write_csv(path: Path, n_rows: int = 10) -> None:
    tz_offset = '-08:00'
    lines = ['ISO datetime,120pps SNR,120pps signal dB,Noise floor dB,T,H,S,W,G,B']
    for i in range(n_rows):
        ts = f'2024-01-15T10:{i:02d}:00{tz_offset}'
        lines.append(f'{ts},15.0,-80.0,-95.0,68,52,300,7,12,225')
    path.write_text('\n'.join(lines) + '\n')


class TestSmooth:
    def test_simple_moving_average(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        result = plotter._smooth([1.0, 2.0, 3.0, 4.0, 5.0], points=3)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_points_equal_one_returns_same_values(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        data = [10.0, 20.0, 30.0]
        result = plotter._smooth(data, points=1)
        np.testing.assert_allclose(result, data)

    def test_output_shorter_than_input(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        result = plotter._smooth([1.0, 2.0, 3.0, 4.0, 5.0], points=3)
        assert len(result) == 3  # len(data) - points + 1

    def test_uniform_data_unchanged_by_smoothing(self, tmp_path):
        plotter, _ = _make_plotter(tmp_path)
        data = [5.0] * 10
        result = plotter._smooth(data, points=3)
        np.testing.assert_allclose(result, [5.0] * 8)


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
        plotter, store_mock = _make_plotter(tmp_path)
        store_mock.read_range_to_time_dict.return_value = self._time_data()
        output = tmp_path / 'summary.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(output, start)
        assert output.exists()

    def test_returns_early_when_no_data(self, tmp_path):
        plotter, store_mock = _make_plotter(tmp_path)
        store_mock.read_range_to_time_dict.return_value = {}
        output = tmp_path / 'should_not_exist.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(output, start)
        assert not output.exists()

    def test_accepts_path_or_string(self, tmp_path):
        plotter, store_mock = _make_plotter(tmp_path)
        store_mock.read_range_to_time_dict.return_value = self._time_data()
        output = tmp_path / 'summary.png'
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        plotter.generate_summary_graph(str(output), start)
        assert output.exists()

    def test_passes_date_range_to_store(self, tmp_path):
        plotter, store_mock = _make_plotter(tmp_path)
        store_mock.read_range_to_time_dict.return_value = {}
        start = datetime(2024, 1, 1, tzinfo=_TZ)
        output = tmp_path / 'summary.png'
        plotter.generate_summary_graph(output, start)
        store_mock.read_range_to_time_dict.assert_called_once()
        call_start = store_mock.read_range_to_time_dict.call_args[0][0]
        assert call_start == start

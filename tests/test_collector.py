"""Tests for Collector: measurement averaging, hourly summaries, uploads, and loop resilience."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from buzz.collector import Collector
from buzz.config import BuzzConfig

_TZ = ZoneInfo('America/Los_Angeles')


def _make_config(tmp_path: Path, server_enabled: bool = False) -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.station.path = str(tmp_path)
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.station.audio_rf_conversion_db = -32.0
    cfg.server.enabled = server_enabled
    cfg.audio.measurements_to_take = 2
    return cfg


def _make_collector(cfg: BuzzConfig) -> Collector:
    return Collector(
        config=cfg,
        sampler=MagicMock(),
        weather=MagicMock(),
        store=MagicMock(),
        plotter=MagicMock(),
        publisher=MagicMock(),
    )


def _setup_defaults(collector: Collector, tmp_path: Path, minute: int = 30) -> datetime:
    collector._sampler.take_sample.return_value = (10.0, -80.0, -90.0)
    collector._weather.fetch.return_value = ('72', '45', '120', '5', '8', '180')
    collector._store.append.return_value = 'csv_row'
    collector._store.filename_for_date.return_value = tmp_path / 'data.csv'
    return datetime(2024, 1, 15, 10, minute, 0, tzinfo=_TZ)


class TestRunCollectionAveraging:
    def test_calls_take_sample_n_times(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.audio.measurements_to_take = 3
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        collector._sampler.take_sample.side_effect = [
            (10.0, -80.0, -90.0),
            (12.0, -78.0, -88.0),
            (14.0, -76.0, -86.0),
        ]
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        assert collector._sampler.take_sample.call_count == 3

    def test_averaged_snr_passed_to_store(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.audio.measurements_to_take = 2
        collector = _make_collector(cfg)
        collector._sampler.take_sample.side_effect = [
            (10.0, -80.0, -90.0),
            (20.0, -70.0, -80.0),
        ]
        collector._weather.fetch.return_value = ('72', '45', '120', '5', '8', '180')
        collector._store.append.return_value = 'csv'
        collector._store.filename_for_date.return_value = tmp_path / 'data.csv'
        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        args = collector._store.append.call_args[0]
        assert args[1] == pytest.approx(15.0)   # mean snr

    def test_calls_weather_fetch_once(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._weather.fetch.assert_called_once()


class TestRunCollectionPlotting:
    def test_generates_two_plots_per_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        assert collector._plotter.generate_graph_from_csv.call_count == 2

    def test_smooth_plot_uses_smooth_6(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        calls = collector._plotter.generate_graph_from_csv.call_args_list
        smooth_calls = [c for c in calls if c.kwargs.get('smooth') == 6
                        or (len(c.args) > 2 and c.args[2] == 6)]
        assert len(smooth_calls) == 1

    def test_no_summary_graphs_when_minute_not_zero(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path, minute=30)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._plotter.generate_summary_graph.assert_not_called()

    def test_three_summary_graphs_generated_on_the_hour(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path, minute=0)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        assert collector._plotter.generate_summary_graph.call_count == 3


class TestRunCollectionUploads:
    def test_no_uploads_when_server_disabled(self, tmp_path):
        cfg = _make_config(tmp_path, server_enabled=False)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._publisher.generate_index.assert_not_called()
        collector._publisher.scp_to_server.assert_not_called()

    def test_none_publisher_is_safe_when_server_disabled(self, tmp_path):
        # main.py passes publisher=None when server.enabled is False;
        # _run_collection must not attempt to call methods on it
        cfg = _make_config(tmp_path, server_enabled=False)
        collector = Collector(
            config=cfg,
            sampler=MagicMock(),
            weather=MagicMock(),
            store=MagicMock(),
            plotter=MagicMock(),
            publisher=None,
        )
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()  # must not raise AttributeError

    def test_uploads_when_server_enabled(self, tmp_path):
        cfg = _make_config(tmp_path, server_enabled=True)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._publisher.generate_index.assert_called_once()
        collector._publisher.scp_to_server.assert_called_once()

    def test_scp_includes_csv_and_both_plots(self, tmp_path):
        cfg = _make_config(tmp_path, server_enabled=True)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        # scp_to_server is called with a list of (file, prefix) pairs;
        # the CSV + 2 plots + index = at least 4 entries
        call_args = collector._publisher.scp_to_server.call_args[0][0]
        assert len(call_args) >= 4


class TestCollectionLoop:
    def test_keyboard_interrupt_exits_loop(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        t0 = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        t1 = datetime(2024, 1, 15, 10, 31, 1, tzinfo=_TZ)
        with patch.object(collector, '_run_collection', side_effect=KeyboardInterrupt), \
             patch('buzz.collector.sleep'), \
             patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.side_effect = [t0, t1]
            mock_dt.fromisoformat = datetime.fromisoformat
            collector.collection_loop()   # must return, not propagate

    def test_runtime_exception_is_caught_and_loop_continues(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        call_count = [0]

        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError('transient hardware error')
            raise KeyboardInterrupt

        t0 = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        t1 = datetime(2024, 1, 15, 10, 31, 1, tzinfo=_TZ)
        t2 = datetime(2024, 1, 15, 10, 31, 30, tzinfo=_TZ)
        t3 = datetime(2024, 1, 15, 10, 32, 1, tzinfo=_TZ)
        with patch.object(collector, '_run_collection', side_effect=side_effect), \
             patch('buzz.collector.sleep'), \
             patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.side_effect = [t0, t1, t2, t3]
            mock_dt.fromisoformat = datetime.fromisoformat
            collector.collection_loop()

        assert call_count[0] == 2   # first call raised, loop continued, second raised KeyboardInterrupt

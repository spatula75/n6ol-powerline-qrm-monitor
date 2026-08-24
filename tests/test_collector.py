"""Tests for Collector: measurement averaging, hourly summaries, uploads, and loop resilience."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from zoneinfo import ZoneInfo

import pytest

from buzz.analyzer import AnalysisResult
from buzz.collector import ALL_TIME_SUMMARY_NAME, Collector
from buzz.config import BuzzConfig

_TZ = ZoneInfo('America/Los_Angeles')


def _make_config(tmp_path: Path, server_enabled: bool = False) -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.station.path = str(tmp_path)
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.station.audio_rf_conversion_db = -32.0
    cfg.server.enabled = server_enabled
    return cfg


_LOCKED_RESULT   = AnalysisResult(signal_dbm=-80.0, noise_dbm=-90.0, snr=10.0, locked=True)
_UNLOCKED_RESULT = AnalysisResult(signal_dbm=-90.0, noise_dbm=-90.0, snr=0.0,  locked=False)


def _make_collector(cfg: BuzzConfig) -> Collector:
    analyzer = MagicMock()
    # The collector formats these as numbers, so the mock has to return real ones.
    # Defaults here keep tests that aren't about drift tracking from having to
    # configure it; the tests that are override them.
    analyzer.grid_frequency_hz.return_value = 60.0
    analyzer.phase_drift_rate.return_value = 0.0
    return Collector(
        config=cfg,
        analyzer=analyzer,
        weather=MagicMock(),
        store=MagicMock(),
        plotter=MagicMock(),
        publisher=MagicMock(),
    )


def _setup_defaults(collector: Collector, tmp_path: Path, minute: int = 30) -> datetime:
    collector._analyzer.drain_results.return_value = [_LOCKED_RESULT] * 60
    collector._weather.fetch.return_value = ('72', '45', '120', '5', '8', '180')
    collector._store.append.return_value = 'csv_row'
    collector._store.filename_for_date.return_value = tmp_path / 'data.csv'
    return datetime(2024, 1, 15, 10, minute, 0, tzinfo=_TZ)


class TestRunCollectionAveraging:
    def test_calls_drain_results_once(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._analyzer.drain_results.assert_called_once()

    def test_averaged_snr_from_locked_results(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        results = [
            AnalysisResult(signal_dbm=-80.0, noise_dbm=-90.0, snr=10.0, locked=True),
            AnalysisResult(signal_dbm=-70.0, noise_dbm=-80.0, snr=20.0, locked=True),
        ]
        collector._analyzer.drain_results.return_value = results
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

    def test_weather_failure_still_writes_csv_row(self, tmp_path):
        """A weather API failure must not cost us the noise measurement."""
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        collector._weather.fetch.side_effect = OSError('weather server down')
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        collector._store.append.assert_called_once()
        args = collector._store.append.call_args[0]
        assert args[5:] == ('', '', '', '', '', '')   # blank weather fields


class TestRunCollectionGridFrequency:
    """The collector reads the drift tracker's by-products and logs them."""

    def _run(self, collector: Collector, tmp_path: Path) -> dict:
        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        collector._weather.fetch.return_value = ('72', '45', '120', '5', '8', '180')
        collector._store.append.return_value = 'csv'
        collector._store.filename_for_date.return_value = tmp_path / 'data.csv'
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        return collector._store.append.call_args[1]

    def test_logged_when_the_minute_had_a_lock(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_LOCKED_RESULT] * 60
        collector._analyzer.grid_frequency_hz.return_value = 60.0234
        collector._analyzer.phase_drift_rate.return_value = -6.123
        kwargs = self._run(collector, tmp_path)
        assert kwargs['grid_frequency'] == '60.023'
        assert kwargs['phase_drift'] == '-6.12'

    def test_three_decimals_for_frequency(self, tmp_path):
        """One digit past the absolute accuracy, so relative changes survive; see
        the note in Collector._run_collection about the sound-card timebase."""
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_LOCKED_RESULT] * 60
        collector._analyzer.grid_frequency_hz.return_value = 60.0
        collector._analyzer.phase_drift_rate.return_value = 0.0
        assert self._run(collector, tmp_path)['grid_frequency'] == '60.000'

    def test_blank_when_no_lock_was_held(self, tmp_path):
        """With no lock the tracker's rate is stale - a number here would be a lie."""
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_UNLOCKED_RESULT] * 60
        collector._analyzer.grid_frequency_hz.return_value = 60.0234
        kwargs = self._run(collector, tmp_path)
        assert kwargs['grid_frequency'] == ''
        assert kwargs['phase_drift'] == ''

    def test_blank_when_no_results_at_all(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = []
        kwargs = self._run(collector, tmp_path)
        assert kwargs['grid_frequency'] == ''
        assert kwargs['phase_drift'] == ''

    def test_partial_lock_still_logs_a_frequency(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_LOCKED_RESULT, _UNLOCKED_RESULT]
        collector._analyzer.grid_frequency_hz.return_value = 59.9812
        collector._analyzer.phase_drift_rate.return_value = 5.01
        assert self._run(collector, tmp_path)['grid_frequency'] == '59.981'


class TestRunCollectionLockStatus:
    def _run(self, collector: Collector, tmp_path: Path) -> tuple:
        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        collector._weather.fetch.return_value = ('72', '45', '120', '5', '8', '180')
        collector._store.append.return_value = 'csv'
        collector._store.filename_for_date.return_value = tmp_path / 'data.csv'
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        return collector._store.append.call_args[0]

    def test_all_locked_gives_full(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_LOCKED_RESULT] * 60
        args = self._run(collector, tmp_path)
        assert args[4] == 'full'

    def test_all_unlocked_gives_none(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_UNLOCKED_RESULT] * 60
        args = self._run(collector, tmp_path)
        assert args[4] == 'none'

    def test_mixed_gives_partial(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = (
            [_LOCKED_RESULT] * 30 + [_UNLOCKED_RESULT] * 30
        )
        args = self._run(collector, tmp_path)
        assert args[4] == 'partial'

    def test_no_results_gives_none(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = []
        args = self._run(collector, tmp_path)
        assert args[4] == 'none'

    def test_no_lock_signal_equals_noise(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        collector._analyzer.drain_results.return_value = [_UNLOCKED_RESULT] * 60
        args = self._run(collector, tmp_path)
        assert args[2] == args[3]   # signal_mean == noise_mean


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

    def test_two_summary_graphs_generated_on_the_hour(self, tmp_path):
        """7-day and 30-day only, because the all-time graph is off by default."""
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path, minute=0)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        assert collector._plotter.generate_summary_graph.call_count == 2


class TestTheAllTimeSummaryIsOptional:
    """The rolling windows need no setting; the all-time graph does.

    7 days and 30 days each end today, so they follow the station's current situation
    on their own.  The all-time graph keeps averaging in months a station may have
    stopped resembling long ago, which is why it is the one that has to be asked for.
    """

    def _run_on_the_hour(self, collector, tmp_path):
        now = _setup_defaults(collector, tmp_path, minute=0)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()

    def _summary_paths(self, collector) -> list[str]:
        return [Path(c.args[0]).name
                for c in collector._plotter.generate_summary_graph.call_args_list]

    def test_it_is_off_by_default(self, tmp_path):
        assert BuzzConfig().station.enable_all_time_summary is False, (
            'A station running for more than a few months is the common case, and '
            'averaging its whole history describes neither the fault nor the repair.'
        )

    def test_the_all_time_graph_is_skipped_when_off(self, tmp_path):
        collector = _make_collector(_make_config(tmp_path))
        self._run_on_the_hour(collector, tmp_path)
        assert self._summary_paths(collector) == ['_noise_probability_summary_7d.png',
                                                  '_noise_probability_summary_30d.png']

    def test_turning_it_on_adds_the_all_time_graph(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.station.enable_all_time_summary = True
        collector = _make_collector(cfg)
        self._run_on_the_hour(collector, tmp_path)
        assert ALL_TIME_SUMMARY_NAME in self._summary_paths(collector)

    def test_the_all_time_graph_starts_at_the_configured_date(self, tmp_path):
        """The start date is what the flag turns back on, so it has to still be used."""
        cfg = _make_config(tmp_path)
        cfg.station.enable_all_time_summary = True
        cfg.station.summary_start_date_iso = '2019-03-04T00:00:00+0000'
        collector = _make_collector(cfg)
        self._run_on_the_hour(collector, tmp_path)
        all_time = [c for c in collector._plotter.generate_summary_graph.call_args_list
                    if Path(c.args[0]).name == ALL_TIME_SUMMARY_NAME][0]
        assert all_time.args[1] == datetime.fromisoformat('2019-03-04T00:00:00+0000')

    def test_the_all_time_graph_is_uploaded_when_on(self, tmp_path):
        """One flag governs both, since a chart nobody publishes helps nobody."""
        cfg = _make_config(tmp_path, server_enabled=True)
        cfg.station.enable_all_time_summary = True
        collector = _make_collector(cfg)
        self._run_on_the_hour(collector, tmp_path)
        uploaded = [Path(local).name for local, _ in
                    collector._publisher.scp_to_server.call_args.args[0]]
        assert ALL_TIME_SUMMARY_NAME in uploaded

    def test_the_all_time_graph_is_not_uploaded_when_off(self, tmp_path):
        cfg = _make_config(tmp_path, server_enabled=True)
        collector = _make_collector(cfg)
        self._run_on_the_hour(collector, tmp_path)
        uploaded = [Path(local).name for local, _ in
                    collector._publisher.scp_to_server.call_args.args[0]]
        assert ALL_TIME_SUMMARY_NAME not in uploaded


class TestAChartLeftBehindByTurningItOff:
    """Turning the summary off stops it being updated; it does not delete it.

    Nothing else in this program removes a file it published, and the operator may want
    to keep the last one.  But a chart that stops updating and says nothing is a chart
    that goes on looking current in the archive, which is the very thing this setting
    exists to prevent, so startup mentions it once.
    """

    def _stale_chart(self, tmp_path: Path) -> Path:
        chart = tmp_path / ALL_TIME_SUMMARY_NAME
        chart.write_bytes(b'png')
        return chart

    def test_an_existing_chart_is_reported_once_at_startup(self, tmp_path, caplog):
        self._stale_chart(tmp_path)
        with caplog.at_level(logging.INFO, logger='buzz.collector'):
            _make_collector(_make_config(tmp_path))
        assert ALL_TIME_SUMMARY_NAME in caplog.text
        assert 'no longer be updated' in caplog.text, (
            'The operator has to be told the chart is now stale, and where to delete '
            'it.  Otherwise it sits in the archive looking current for years.'
        )

    def test_the_chart_is_never_deleted(self, tmp_path, caplog):
        chart = self._stale_chart(tmp_path)
        with caplog.at_level(logging.INFO, logger='buzz.collector'):
            _make_collector(_make_config(tmp_path))
        assert chart.exists(), (
            'Reporting is not permission to delete.  Nothing else in this program '
            'removes a published file, and the last chart may be wanted.'
        )

    def test_nothing_is_said_when_the_summary_is_on(self, tmp_path, caplog):
        self._stale_chart(tmp_path)
        cfg = _make_config(tmp_path)
        cfg.station.enable_all_time_summary = True
        with caplog.at_level(logging.INFO, logger='buzz.collector'):
            _make_collector(cfg)
        assert ALL_TIME_SUMMARY_NAME not in caplog.text, (
            'The chart is being kept current, so there is nothing to report.'
        )

    def test_nothing_is_said_when_there_is_no_chart(self, tmp_path, caplog):
        """The ordinary case for a new station, which must start up quietly."""
        with caplog.at_level(logging.INFO, logger='buzz.collector'):
            _make_collector(_make_config(tmp_path))
        assert ALL_TIME_SUMMARY_NAME not in caplog.text


class TestRunCollectionUploads:
    def test_no_uploads_when_publisher_none(self, tmp_path):
        # main.py passes publisher=None when server.enabled is False; the
        # publisher's presence is the gate, so this must run without uploads
        # and without AttributeError
        cfg = _make_config(tmp_path, server_enabled=False)
        collector = _make_collector(cfg)
        collector._publisher = None
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()  # must not raise AttributeError

    def test_uploads_when_publisher_present(self, tmp_path):
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

    def test_the_smoothed_chart_is_the_one_published_as_current(self, tmp_path):
        """The page's fixed URL must show the smoothed chart, matching what it showed
        before the page stopped naming a file of its own.

        The raw chart is uploaded too and stays browsable, so publishing the wrong one
        would still produce a working page showing a noisier trace, which nothing but
        this assertion would catch.
        """
        cfg = _make_config(tmp_path, server_enabled=True)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        current = collector._publisher.scp_to_server.call_args.kwargs['current_chart']
        assert current.name == f'noise_plot_movavg.{now.strftime("%Y-%m-%d")}.png'

    def test_the_index_is_rendered_without_a_chart_name_or_timestamp(self, tmp_path):
        """generate_index takes only an output path now.

        The page reads the filename and the update time from the fixed URL's headers,
        so passing either in would reintroduce the staleness this design removed.
        """
        cfg = _make_config(tmp_path, server_enabled=True)
        collector = _make_collector(cfg)
        now = _setup_defaults(collector, tmp_path)
        with patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromisoformat = datetime.fromisoformat
            collector._run_collection()
        args, kwargs = collector._publisher.generate_index.call_args
        assert len(args) == 1 and not kwargs


class TestCollectionLoop:
    def test_keyboard_interrupt_exits_loop(self, tmp_path):
        cfg = _make_config(tmp_path)
        collector = _make_collector(cfg)
        t0 = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        t1 = datetime(2024, 1, 15, 10, 31, 1, tzinfo=_TZ)
        run_collection = Mock(side_effect=KeyboardInterrupt)
        with patch.object(collector, '_run_collection', run_collection), \
             patch('buzz.collector.sleep'), \
             patch('buzz.collector.datetime') as mock_dt:
            mock_dt.now.side_effect = [t0, t1]
            mock_dt.fromisoformat = datetime.fromisoformat
            collector.collection_loop()   # must return, not propagate
        # Not just that this returned without raising: it must have stopped on the
        # first interrupt rather than swallowing it somewhere and looping again, which
        # a `return` misplaced one level too shallow would do silently.
        run_collection.assert_called_once()

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

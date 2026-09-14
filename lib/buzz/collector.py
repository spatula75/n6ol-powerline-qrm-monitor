"""
Measurement loop: samples audio, stores results, generates plots, and uploads files.

Collector.collection_loop() runs forever (until KeyboardInterrupt), waking at the
top of each minute to call _run_collection(). _run_collection() averages several
audio samples, appends a CSV row, renders daily plots, and, if uploads are enabled,
generates an HTML index and SCPs everything to the configured web server.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

from buzz.analyzer import AnalysisResult, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore
from buzz.dsp import SILENCE_DBFS
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.weather import EMPTY_WEATHER, WeatherClient

logger = logging.getLogger(__name__)

# Named because two places need it: the hourly render writes it, and startup looks for
# one left behind by a station that has since turned the all-time summary off.
ALL_TIME_SUMMARY_NAME = '_noise_probability_summary.png'

# The grid-frequency chart covers the current day and is overwritten in place, so its
# name carries no date: there is only ever one, and it is always today's.
FREQUENCY_CHART_NAME = 'current_frequency_estimate.png'


class Collector:
    def __init__(self, config: BuzzConfig, analyzer: ContinuousAnalyzer, weather: WeatherClient,
                 store: CsvStore, plotter: Plotter, publisher: Publisher | None) -> None:
        self._config = config
        self._analyzer = analyzer
        self._weather = weather
        self._store = store
        self._plotter = plotter
        self._publisher = publisher
        # Parsed here rather than at the top of the hour, so a station that asked for
        # the all-time chart learns at startup that its start date is unreadable.
        # Parsed only when that chart is on, because the setup program hides the field
        # while it is off: a station carrying an empty or malformed date would
        # otherwise die on every start with no way left to correct it.
        self._summary_start_date = (self._parse_summary_start_date()
                                    if config.station.enable_all_time_summary else None)
        self._report_any_stale_chart(enabled=config.station.enable_all_time_summary,
                                     name=ALL_TIME_SUMMARY_NAME,
                                     description='the all-time summary',
                                     setting='enable_all_time_summary')
        self._report_any_stale_chart(enabled=config.station.enable_frequency_chart,
                                     name=FREQUENCY_CHART_NAME,
                                     description='the grid frequency chart',
                                     setting='enable_frequency_chart')

    def _parse_summary_start_date(self) -> datetime:
        """The date the all-time summary begins, refusing an unreadable one by name.

        fromisoformat reports nothing but the text it choked on, which does not say
        which setting the text came from or where to change it.
        """
        configured = self._config.station.summary_start_date_iso
        try:
            return datetime.fromisoformat(configured)
        except ValueError as exc:
            raise ValueError(
                f'The [station] summary_start_date_iso setting is "{configured}", '
                'which is not an ISO 8601 date.  The all-time summary graph starts '
                'from that date and cannot be drawn without it.  Set it in '
                '~/.buzz/config.toml in the form 2024-01-01T00:00:00+0000, or turn '
                'off station.enable_all_time_summary.') from exc

    def _report_any_stale_chart(self, enabled: bool, name: str, description: str, setting: str) -> None:
        """Say once that an optional chart is present but no longer being updated.

        Turning a chart off does not delete what it already wrote, here or on the web
        server.  Silence would leave a chart that looks current sitting in the archive
        for as long as the station runs, which is the very thing the setting exists to
        avoid.  Deleting it unasked is worse: the operator may want to keep the last
        one, and nothing else in this program removes a published file.

        The grid frequency chart is the sharper case, because its name claims it is
        current whatever its age.
        """
        if enabled:
            return
        stale = Path(self._config.station.path) / name
        if stale.exists():
            logger.info(
                '%s exists, and %s is off, so it will no longer be updated.  Delete it '
                'here and on the web server if you do not want a stale chart served.  '
                'Set station.%s to keep it current instead.', stale, description, setting)

    def _average_minute_results(self, results: list[AnalysisResult]) -> tuple[float, float, float, str]:
        """Average one minute's analyzer results into (snr, signal, noise, lock_status).

        Signal and SNR are averaged only over locked results, so an intermittent
        signal's unlocked ticks don't drag the level down mid-minute. Noise floor
        is averaged over every result regardless of lock. With no locked results
        at all, signal mirrors noise, mirroring AnalysisResult.unlocked()'s
        convention, so the plotted traces coincide instead of gapping. With no
        results at all, the minute reads as silence with status 'none'.
        """
        if not results:
            return 0.0, SILENCE_DBFS, SILENCE_DBFS, 'none'
        locked_results = [r for r in results if r.locked]
        if not locked_results:
            noise_mean = round(sum(r.noise_dbm for r in results) / len(results), 2)
            return 0.0, noise_mean, noise_mean, 'none'
        signal_mean = round(sum(r.signal_dbm for r in locked_results) / len(locked_results), 2)
        snr_mean    = round(sum(r.snr        for r in locked_results) / len(locked_results), 2)
        noise_mean  = round(sum(r.noise_dbm  for r in results)        / len(results),        2)
        lock_status = 'full' if len(locked_results) == len(results) else 'partial'
        return snr_mean, signal_mean, noise_mean, lock_status

    def _grid_frequency_fields(self, results: list[AnalysisResult]) -> tuple[str, str]:
        """CSV-ready (grid_frequency, phase_drift) strings for this minute.

        Grid frequency and drift come from the analyzer's phase tracker, which only
        has a meaningful estimate while it is following a pulse train. With no
        locked results this minute the stored rate is stale, so this returns blanks
        rather than a number that looks like a measurement.

        Three decimal places is one digit past what the absolute accuracy supports.
        The reading is scaled by the sound card's sample-clock error (50-100 ppm on
        typical hardware, or 0.003-0.006 Hz at 60 Hz), so the third digit is only
        meaningful for how the frequency *changes*, not for what it is. That error
        is a single multiplicative constant, so if the card is ever calibrated the
        whole logged history can be corrected by one scale factor, which is also
        why the raw drift rate is worth keeping alongside the derived frequency.
        """
        if not any(r.locked for r in results):
            return '', ''
        return f'{self._analyzer.grid_frequency_hz():.3f}', f'{self._analyzer.phase_drift_rate():.2f}'

    def _fetch_weather_or_blank(self) -> tuple:
        """Fetch current weather, degrading to blank fields on any failure.

        Weather is decoration on the noise measurement. A failed fetch must not
        cost this program the CSV row.
        """
        try:
            return self._weather.fetch()
        except Exception:
            logger.warning(
                'Could not fetch weather, so this measurement is recorded with its '
                'weather columns blank; the noise figures themselves are unaffected. '
                'Usually the station is unreachable or slow to answer. Check the '
                '[weather] section of the config if every row comes out blank.',
                exc_info=True)
            return EMPTY_WEATHER

    def _render_daily_plots(self, csv_filename: Path, plot_filename: Path,
                            smooth_plot_filename: Path) -> None:
        """Render the raw and 6-point-smoothed daily signal-vs-noise-floor charts."""
        self._plotter.generate_graph_from_csv(csv_filename, plot_filename)
        # smooth=6 applies a 6-point moving average (6 minutes); reduces noise in the
        # displayed trace without obscuring true interference events.
        self._plotter.generate_graph_from_csv(csv_filename, smooth_plot_filename, smooth=6)

    def _render_hourly_summaries(self, zone: ZoneInfo, output_dir: Path) -> list[Path]:
        """On the hour, regenerate the 7-day and 30-day summary graphs, and the all-time
        one where the station asked for it.

        Only the all-time graph has a setting.  The other two cover a fixed span ending
        today, so what they average changes with the station and no start date has to be
        maintained.  The all-time span only grows, and averages every day since the
        configured start date however little the recent ones resemble the first.

        Returns the paths just written, for the caller to add to its upload list.
        """
        today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)

        summaries: list[Path] = []
        if self._config.station.enable_all_time_summary:
            summaries.append(self._write_summary(output_dir, ALL_TIME_SUMMARY_NAME,
                                                 self._summary_start_date))
        summaries.append(self._write_summary(output_dir, '_noise_probability_summary_7d.png',
                                             today - timedelta(days=7)))
        summaries.append(self._write_summary(output_dir, '_noise_probability_summary_30d.png',
                                             today - timedelta(days=30)))
        return summaries

    def _render_frequency_chart(self, csv_filename: Path, output_dir: Path,
                                now: datetime) -> list[Path]:
        """Redraw the current day's grid-frequency chart, where the station asked for it.

        One file, overwritten in place, rather than one per day.  It shows today only,
        and yesterday's is not kept: the daily CSVs hold the readings, and this chart is
        for watching the figure move rather than for keeping.

        Redrawn every cycle, with the daily charts, rather than on the hour with the
        summaries.  It costs 434 ms and 144 kB per minute, measured, which is 0.7% of
        the minute it has to work in.

        Returns the path written, for the caller to add to its upload list, or nothing
        at all when the chart is off.
        """
        if not self._config.station.enable_frequency_chart:
            return []
        chart = output_dir / FREQUENCY_CHART_NAME
        self._plotter.generate_frequency_graph(csv_filename, chart, now)
        return [chart]

    def _write_summary(self, output_dir: Path, name: str, start: datetime) -> Path:
        """Generate one summary graph covering `start` to now, and return where it went."""
        path = output_dir / name
        self._plotter.generate_summary_graph(path, start)
        return path

    def _publish_outputs(self, output_dir: Path, smooth_plot_filename: Path,
                         upload_files: list[Path]) -> None:
        """Render the HTML index and SCP everything to the configured web server.

        The index is re-uploaded every cycle even though its content only changes when
        the config does.  It costs about 2 kB per minute, and uploading it unconditionally
        means a cycle whose upload failed repairs itself on the next one, where a
        once-at-startup upload would leave the page missing until a restart.
        """
        index_filename = output_dir / 'index.html'
        self._publisher.generate_index(index_filename)
        self._publisher.scp_to_server(
            [(f, 'data/') for f in upload_files] + [(index_filename, '')],
            current_chart=smooth_plot_filename,
        )

    def _run_collection(self) -> None:
        """Take one complete measurement cycle and write all outputs.

        Drains the AnalysisResult objects the analyzer published since the previous
        cycle and averages them (draining keeps consecutive rows from re-averaging
        each other's data), appends a CSV row, and generates the raw and smoothed
        daily plots.  On the hour it also regenerates the 7-day and 30-day summary
        graphs, and the all-time one where the station asked for it.  If server uploads
        are enabled, it also renders the HTML index and SCPs all changed files.
        """
        station = self._config.station
        zone = ZoneInfo(station.timezone)
        now = datetime.now(zone).replace(second=0, microsecond=0)

        results = self._analyzer.drain_results()
        snr_mean, signal_mean, noise_mean, lock_status = self._average_minute_results(results)
        grid_frequency, phase_drift = self._grid_frequency_fields(results)
        weather_data = self._fetch_weather_or_blank()

        csv_str = self._store.append(now, snr_mean, signal_mean, noise_mean,
                                     lock_status, *weather_data,
                                     grid_frequency=grid_frequency, phase_drift=phase_drift)

        output_dir = Path(station.path)
        now_date_str = now.strftime('%Y-%m-%d')
        csv_filename = self._store.filename_for_date(now)
        plot_filename = output_dir / f'noise_plot.{now_date_str}.png'
        smooth_plot_filename = output_dir / f'noise_plot_movavg.{now_date_str}.png'
        self._render_daily_plots(csv_filename, plot_filename, smooth_plot_filename)

        upload_files = [csv_filename, plot_filename, smooth_plot_filename]
        upload_files.extend(self._render_frequency_chart(csv_filename, output_dir, now))
        if now.minute == 0:
            upload_files.extend(self._render_hourly_summaries(zone, output_dir))

        # main.py wires a Publisher only when uploads are enabled, so the publisher's
        # presence is the single source of truth for whether to upload.
        if self._publisher is not None:
            self._publish_outputs(output_dir, smooth_plot_filename, upload_files)

        logger.info(csv_str)

    def collection_loop(self) -> None:
        """Run _run_collection() at the top of every minute until interrupted.

        Sleeps until the next whole minute, then calls _run_collection().
        Exceptions (other than KeyboardInterrupt) are logged and the loop continues,
        so a transient hardware or network error doesn't kill the monitor.
        """
        while True:
            try:
                zone = ZoneInfo(self._config.station.timezone)
                now = datetime.now(zone)
                next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                # Loop rather than a single sleep because sleep() can return early
                # on some platforms, and to skip cleanly if a collection runs long
                while now.timestamp() < next_minute.timestamp():
                    sleep(next_minute.timestamp() - now.timestamp())
                    now = datetime.now(zone)
                self._run_collection()
            except KeyboardInterrupt:
                return
            except Exception:
                logger.exception(
                    'Collection cycle failed - likely a transient hardware or network error; '
                    'will retry at the next minute boundary.'
                )

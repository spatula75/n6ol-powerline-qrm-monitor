"""
Measurement loop: samples audio, stores results, generates plots, and uploads files.

Collector.collection_loop() runs forever (until KeyboardInterrupt), waking at the
top of each minute to call _run_collection().  _run_collection() averages several
audio samples, appends a CSV row, renders daily plots, and — if uploads are enabled
— generates an HTML index and SCPs everything to the configured web server.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import WeatherClient

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, config: BuzzConfig, sampler: AudioSampler, weather: WeatherClient,
                 store: CsvStore, plotter: Plotter, publisher: Publisher) -> None:
        self._config = config
        self._sampler = sampler
        self._weather = weather
        self._store = store
        self._plotter = plotter
        self._publisher = publisher
        self._summary_start_date = datetime.fromisoformat(config.station.summary_start_date_iso)

    def _run_collection(self) -> None:
        """Take one complete measurement cycle and write all outputs.

        Averages config.audio.measurements_to_take samples, appends a CSV row,
        generates the raw and smoothed daily plots, and on the hour also regenerates
        the all-time, 7-day, and 30-day summary graphs.  If server uploads are
        enabled, also renders the HTML index and SCPs all changed files.
        """
        station = self._config.station
        zone = ZoneInfo(station.timezone)
        now = datetime.now(zone).replace(second=0, microsecond=0)

        n = self._config.audio.measurements_to_take
        snrs, signals, noises = zip(*[self._sampler.take_sample() for _ in range(n)])
        snr_mean = round(sum(snrs) / n, 2)
        # take_sample() returns levels in dBFS (relative to digital full-scale); adding the
        # station's calibration offset converts to approximate dBm at the receiver input.
        signal_mean = round(sum(signals) / n, 2) + station.audio_rf_conversion_db
        noise_mean = round(sum(noises) / n, 2) + station.audio_rf_conversion_db

        temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing = self._weather.fetch()

        csv_str = self._store.append(now, snr_mean, signal_mean, noise_mean,
                                     temperature, humidity, solar_radiation,
                                     wind_speed, wind_gust, wind_bearing)

        output_dir = Path(station.path)
        now_date_str = now.strftime('%Y-%m-%d')
        csv_filename = self._store.filename_for_date(now)
        plot_filename = output_dir / f'noise_plot.{now_date_str}.png'
        smooth_plot_filename = output_dir / f'noise_plot_movavg.{now_date_str}.png'

        self._plotter.generate_graph_from_csv(csv_filename, plot_filename)
        # smooth=6 applies a 6-point moving average (6 minutes); reduces noise in the
        # displayed trace without obscuring genuine interference events.
        self._plotter.generate_graph_from_csv(csv_filename, smooth_plot_filename, smooth=6)

        upload_files = [csv_filename, plot_filename, smooth_plot_filename]

        if now.minute == 0:
            today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)

            summary_all = output_dir / '_noise_probability_summary.png'
            self._plotter.generate_summary_graph(summary_all, self._summary_start_date)

            summary_7d = output_dir / '_noise_probability_summary_7d.png'
            self._plotter.generate_summary_graph(summary_7d, today - timedelta(days=7))

            summary_30d = output_dir / '_noise_probability_summary_30d.png'
            self._plotter.generate_summary_graph(summary_30d, today - timedelta(days=30))

            upload_files.extend([summary_all, summary_7d, summary_30d])

        if self._config.server.enabled:
            index_filename = output_dir / 'index.html'
            image_path = 'data/' + smooth_plot_filename.name
            self._publisher.generate_index(index_filename, now, image_path)
            self._publisher.scp_to_server(
                [(f, 'data/') for f in upload_files] + [(index_filename, '')]
            )

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
                    'Collection cycle failed — likely a transient hardware or network error; '
                    'will retry at the next minute boundary.'
                )

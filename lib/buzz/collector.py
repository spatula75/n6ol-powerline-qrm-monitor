import traceback
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


class Collector:
    def __init__(self, config: BuzzConfig, sampler: AudioSampler, weather: WeatherClient,
                 store: CsvStore, plotter: Plotter, publisher: Publisher):
        self._config = config
        self._sampler = sampler
        self._weather = weather
        self._store = store
        self._plotter = plotter
        self._publisher = publisher
        self._summary_start_date = datetime.fromisoformat(config.station.summary_start_date_iso)

    def run_collection(self):
        station = self._config.station
        zone = ZoneInfo(station.timezone)
        now = datetime.now(zone).replace(second=0, microsecond=0)

        snr_total = signal_total = noise_total = 0.0
        for _ in range(self._config.audio.measurements_to_take):
            snr, signal_db, noise_db = self._sampler.take_sample()
            snr_total += snr
            signal_total += signal_db
            noise_total += noise_db

        n = self._config.audio.measurements_to_take
        snr_mean = round(snr_total / n, 2)
        signal_mean = round(signal_total / n, 2) + station.audio_rf_conversion_db
        noise_mean = round(noise_total / n, 2) + station.audio_rf_conversion_db

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

        print(csv_str)

    def collection_loop(self):
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
                self.run_collection()
            except KeyboardInterrupt:
                return
            except Exception:
                traceback.print_exc()
                print('Unexpected error — will retry next minute.')

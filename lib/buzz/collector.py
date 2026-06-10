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

    def run_collection(self):
        zone = ZoneInfo(self._config.timezone)
        now = datetime.now(zone).replace(second=0, microsecond=0)

        snr_total = signal_total = noise_total = 0.0
        for _ in range(self._config.measurements_to_take):
            snr, signal_db, noise_db = self._sampler.sample_data
            snr_total += snr
            signal_total += signal_db
            noise_total += noise_db

        n = self._config.measurements_to_take
        snr_mean = round(snr_total / n, 2)
        signal_mean = round(signal_total / n, 2) + self._config.audio_rf_conversion_db
        noise_mean = round(noise_total / n, 2) + self._config.audio_rf_conversion_db

        temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing = self._weather.fetch()

        csv_str = self._store.append(now, snr_mean, signal_mean, noise_mean,
                                     temperature, humidity, solar_radiation,
                                     wind_speed, wind_gust, wind_bearing)

        now_date_str = now.strftime('%Y-%m-%d')
        csv_filename = self._store.filename_for_date(now)
        plot_filename = f'{self._config.path}/noise_plot.{now_date_str}.png'
        smooth_plot_filename = f'{self._config.path}/noise_plot_movavg.{now_date_str}.png'

        self._plotter.generate_graph_from_csv(csv_filename, plot_filename)
        self._plotter.generate_graph_from_csv(csv_filename, smooth_plot_filename, smooth=6)

        upload_files = [csv_filename, plot_filename, smooth_plot_filename]

        if now.minute == 0:
            zone = ZoneInfo(self._config.timezone)
            today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)

            all_time_start = datetime.fromisoformat(self._config.summary_start_date_iso)
            summary_all = f'{self._config.path}/_noise_probability_summary.png'
            self._plotter.generate_summary_graph(summary_all, all_time_start)

            summary_7d = f'{self._config.path}/_noise_probability_summary_7d.png'
            self._plotter.generate_summary_graph(summary_7d, today - timedelta(days=7))

            summary_30d = f'{self._config.path}/_noise_probability_summary_30d.png'
            self._plotter.generate_summary_graph(summary_30d, today - timedelta(days=30))

            upload_files.extend([summary_all, summary_7d, summary_30d])

        self._publisher.scp_to_server(upload_files, prefix='data/')

        index_filename = f'{self._config.path}/index.html'
        image_path = 'data/' + Path(smooth_plot_filename).name
        self._publisher.generate_index(index_filename, now, image_path)
        self._publisher.scp_to_server([index_filename])

        print(csv_str)

    def collection_loop(self):
        while True:
            try:
                zone = ZoneInfo(self._config.timezone)
                now = datetime.now(zone)
                next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                while now.timestamp() < next_minute.timestamp():
                    sleep(next_minute.timestamp() - now.timestamp())
                    now = datetime.now(zone)
                self.run_collection()
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f'Unexpected exception {e}, ignoring to try again next time.')

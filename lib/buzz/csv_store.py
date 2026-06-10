from collections import defaultdict
from datetime import datetime, timedelta
from math import log
from pathlib import Path
from zoneinfo import ZoneInfo

from buzz.config import BuzzConfig


class CsvStore:
    def __init__(self, config: BuzzConfig):
        self._config = config
        pps = config.audio.pulse_rate
        self._headers = (f'ISO datetime,{pps}pps SNR,{pps}pps signal dB,Noise floor dB,'
                         f'Temperature (F),Humidity (%),Solar radiation (w/m^2),'
                         f'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)\n')

    def filename_for_date(self, date: datetime) -> Path:
        return Path(self._config.station.path) / f'noise_data.{date.strftime("%Y-%m-%d")}.csv'

    def append(self, now: datetime, snr: float, signal: float, noise: float,
               temperature, humidity, solar_radiation,
               wind_speed, wind_gust, wind_bearing) -> str:
        csv_filename = self.filename_for_date(now)
        write_headers = not csv_filename.exists()
        csv_str = (f'{now.isoformat()},{snr:.2f},{signal:.2f},{noise:.2f},'
                   f'{temperature},{humidity},{solar_radiation},'
                   f'{wind_speed},{wind_gust},{wind_bearing}')
        with open(csv_filename, 'a') as f:
            if write_headers:
                f.write(self._headers)
            f.write(f'{csv_str}\n')
        return csv_str

    def read_date_to_time_dict(self, input_filename: Path | str) -> dict:
        time_to_score = defaultdict(int)
        station = self._config.station
        # +3 dB above the detection threshold: a just-qualifying event (SNR exactly
        # at snr_gate) contributes log(snr_gate, snr_gate) = 1.0 to the score.
        snr_gate = station.noise_min_snr + 3
        with open(input_filename, 'r') as f:
            for line in f:
                parts = line.split(',')
                try:
                    timestamp = datetime.fromisoformat(parts[0]).astimezone(ZoneInfo(station.timezone))
                    snr = float(parts[1])
                    signal = float(parts[2])
                except (ValueError, IndexError):
                    continue
                # Bucket timestamp to the nearest 15-minute interval
                t = timestamp.time().replace(
                    minute=int(15 * (timestamp.minute // 15)),
                    second=0, microsecond=0,
                )
                if signal >= station.noise_threshold and snr >= snr_gate:
                    time_to_score[t] += log(snr, snr_gate)
        return {k: int(v) for k, v in time_to_score.items()}

    def read_range_to_time_dict(self, start_date: datetime, end_date: datetime) -> dict:
        time_to_score = defaultdict(int)
        now_date = start_date
        while now_date <= end_date:
            csv_filename = self.filename_for_date(now_date)
            now_date += timedelta(days=1)
            try:
                for t, val in self.read_date_to_time_dict(csv_filename).items():
                    time_to_score[t] += val
            except FileNotFoundError:
                pass
        return dict(time_to_score)

"""
CSV persistence layer for noise measurements.

Each day's data lives in a separate file named noise_data.YYYY-MM-DD.csv in the
configured output directory.  CsvStore owns the file format end to end: writing
new rows, parsing files back into typed CsvRow records (including old-format
files that predate the Signal Lock Status column), and aggregating a date range
into the time-bucketed score dict the summary graphs consume.

Column order is: timestamp, SNR, signal, noise floor, lock status, grid frequency,
phase drift, then the six weather fields.  New columns belong immediately after
lock status, because read_rows() stops reading at index 4 and every field past
that point is written for humans and external tools rather than parsed here.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from math import log
from pathlib import Path
from zoneinfo import ZoneInfo

from buzz.config import BuzzConfig

CsvValue = str | float

# Summary scores are bucketed to intervals of this many minutes.  The summary
# graph builds its time axis from the same constant so the two can't drift apart.
BUCKET_MINUTES = 15


@dataclass(frozen=True)
class CsvRow:
    """One measurement row, with the timestamp converted to the station timezone."""
    timestamp: datetime
    snr: float
    signal: float
    noise: float
    lock_status: str


class CsvStore:
    def __init__(self, config: BuzzConfig) -> None:
        self._config = config
        pps = config.audio.pulse_rate
        self._header_line = (f'ISO datetime,{pps}pps SNR,{pps}pps signal (dBm),Noise floor (dBm),'
                             f'Signal Lock Status,'
                             f'Grid frequency (Hz),Phase drift (samples/s),'
                             f'Temperature (F),Humidity (%),Solar radiation (w/m^2),'
                             f'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)\n')

    def filename_for_date(self, date: datetime) -> Path:
        return Path(self._config.station.path) / f'noise_data.{date.strftime("%Y-%m-%d")}.csv'

    def append(self, now: datetime, snr: float, signal: float, noise: float,
               lock_status: str,
               temperature: CsvValue, humidity: CsvValue, solar_radiation: CsvValue,
               wind_speed: CsvValue, wind_gust: CsvValue, wind_bearing: CsvValue,
               *, grid_frequency: CsvValue = '', phase_drift: CsvValue = '') -> str:
        """Append one measurement row, writing the header first if the file is new.

        grid_frequency and phase_drift are keyword-only and default to blank: they are
        by-products of the analyzer's drift tracking rather than measurements the
        monitor depends on, and a row with no pulse-train lock has nothing to report
        for them.  They are written after Signal Lock Status, ahead of the weather
        fields - read_rows() only ever reads up to index 4, so inserting there leaves
        parsing of both older and newer files completely unaffected.
        """
        csv_filename = self.filename_for_date(now)
        write_header = not csv_filename.exists()
        csv_str = (f'{now.isoformat()},{snr:.2f},{signal:.2f},{noise:.2f},'
                   f'{lock_status},'
                   f'{grid_frequency},{phase_drift},'
                   f'{temperature},{humidity},{solar_radiation},'
                   f'{wind_speed},{wind_gust},{wind_bearing}')
        with open(csv_filename, 'a') as f:
            if write_header:
                f.write(self._header_line)
            f.write(f'{csv_str}\n')
        return csv_str

    def read_rows(self, input_filename: Path | str) -> list[CsvRow]:
        """Parse one CSV file into CsvRow records, skipping headers and malformed lines.

        Timestamps are converted to the station timezone.  Old-format files without
        the Signal Lock Status column get a non-'none' value at index 4 (either the
        temperature field or nothing), which correctly reads as locked; rows too
        short to hold the measurement fields default the status to 'full'.

        Nothing beyond index 4 is read.  That is deliberate and is what lets columns
        be added after Signal Lock Status without breaking files written by older
        versions - the fields that shift are ones this parser never looks at.
        """
        zone = ZoneInfo(self._config.station.timezone)
        rows: list[CsvRow] = []
        with open(input_filename, newline='') as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                try:
                    rows.append(CsvRow(
                        timestamp=datetime.fromisoformat(row[0]).astimezone(zone),
                        snr=float(row[1]),
                        signal=float(row[2]),
                        noise=float(row[3]),
                        lock_status=row[4].strip() if len(row) > 4 else 'full',
                    ))
                except ValueError:
                    continue
        return rows

    def _read_day_scores(self, input_filename: Path | str) -> dict[time, int]:
        """Read one day's CSV file and return a {time: score} dict bucketed to 15-minute intervals.

        Only rows where the signal is at or above the noise threshold AND the SNR is at
        or above snr_gate are counted.  Each qualifying row contributes log(snr, snr_gate)
        to its bucket so stronger events weigh more than just-threshold events.  The
        returned dict maps datetime.time keys (minute is a multiple of 15) to integer scores.
        """
        time_to_score = defaultdict(int)
        station = self._config.station
        # +3 dB above the detection threshold: a just-qualifying event (SNR exactly
        # at snr_gate) contributes log(snr_gate, snr_gate) = 1.0 to the score.
        snr_gate = station.noise_min_snr + 3
        for row in self.read_rows(input_filename):
            if row.signal < station.noise_threshold or row.snr < snr_gate:
                continue
            # Bucket timestamp down to the enclosing BUCKET_MINUTES interval
            t = row.timestamp.time().replace(
                minute=BUCKET_MINUTES * (row.timestamp.minute // BUCKET_MINUTES),
                second=0, microsecond=0,
            )
            time_to_score[t] += log(row.snr, snr_gate)
        return {k: int(v) for k, v in time_to_score.items()}

    def read_range_scores(self, start_date: datetime, end_date: datetime) -> dict[time, int]:
        """Aggregate scores across a date range into a single {time: score} dict.

        Missing CSV files (days with no data) are silently skipped.  The returned
        dict is the sum of all per-day dicts, suitable for passing directly to the
        summary graph generator.
        """
        time_to_score = defaultdict(int)
        day = start_date
        while day <= end_date:
            csv_filename = self.filename_for_date(day)
            day += timedelta(days=1)
            try:
                for t, score in self._read_day_scores(csv_filename).items():
                    time_to_score[t] += score
            except FileNotFoundError:
                pass
        return dict(time_to_score)

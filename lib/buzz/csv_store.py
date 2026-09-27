"""
CSV persistence layer for noise measurements.

Each day's data lives in a separate file named noise_data.YYYY-MM-DD.csv in the
configured output directory.  CsvStore owns the file format end to end: writing
new rows, parsing files back into typed CsvRow records (including old-format
files that predate the Signal Lock Status column), and aggregating a date range
into the time-bucketed score dict the summary graphs consume.

Column order is: timestamp, SNR, signal, noise floor, lock status, grid frequency,
phase drift, then the weather columns.  read_rows() stops reading at index 4, and
read_grid_frequencies() finds its column by name, so every other field is written for
people and external tools rather than parsed here.

A new weather column goes at the end of _WEATHER_COLUMNS, so that an older file's
header stays a prefix of a newer one.  A new core column needs more care.  It would
move the start of the weather section, and every file written before it would then
have its first weather heading read as a core column.

The weather columns follow the file's own header.  A new file gets every weather
column this version knows, in the configured units.  A row added to an existing file
gets one cell for each weather heading already there, in that heading's order and in
the units its label names.  So a file keeps its columns and its units until midnight,
whatever changed in between: a new version with more columns, or a change to
`[weather] units` after a restart.  The core columns, timestamp through phase drift,
stay positional, because their headings carry the pulse rate and a change to it must
not blank the measurements.
"""

import csv
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from math import log
from pathlib import Path
from zoneinfo import ZoneInfo

from buzz.config import BuzzConfig
from buzz.weather import CsvValue, WeatherData, WeatherUnits

logger = logging.getLogger(__name__)

# Summary scores are bucketed to intervals of this many minutes.  The summary
# graph builds its time axis from the same constant so the two can't drift apart.
BUCKET_MINUTES = 15

# The header cell naming the grid-frequency column.  _core_headings writes it and
# read_grid_frequencies() reads it back, so the two cannot drift apart and a rename
# cannot quietly turn every frequency chart empty.
_GRID_FREQUENCY_HEADING = 'Grid frequency (Hz)'


# A weather heading is a name with an optional unit label in parentheses, such as
# "Temperature (F)" or "Humidity (%)".
_WEATHER_HEADING = re.compile(r'(?P<name>.+?)(?: \((?P<label>[^()]*)\))?')


@dataclass(frozen=True)
class _WeatherColumn:
    """One weather column: its heading, and the WeatherData field it holds.

    A heading is the column's name with its unit in parentheses, such as
    "Temperature (F)".  `labels` lists every unit the heading may show, with the
    system each one belongs to: "F" is imperial and "C" is metric.  Humidity's "%" is
    the same in both systems, so its entry is None.  A new file's header shows the
    first entry that fits the configured system.
    """
    name: str
    field: str  # the WeatherData field the column holds
    labels: Mapping[str, WeatherUnits | None]

    def heading(self, units: WeatherUnits) -> str:
        """The heading a new file gives this column in `units`."""
        label = next(label for label, meant in self.labels.items() if meant in (units, None))
        return f'{self.name} ({label})'

    def value(self, weather: WeatherData) -> CsvValue:
        """The figure for this column, from weather the caller has converted to this column's units."""
        return getattr(weather, self.field)


# Every weather column this version writes, in the order a new header lists them.  The
# labels for temperature and wind come from WeatherUnits, so a label cannot change in one
# place and not the other.
_WEATHER_COLUMNS = (
    _WeatherColumn('Temperature', 'temperature', {units.temperature_label: units for units in WeatherUnits}),
    _WeatherColumn('Humidity', 'humidity', {'%': None}),
    # Files written before the watt got its capital W say "w/m^2", and still match.
    _WeatherColumn('Solar radiation', 'solar_radiation', {'W/m^2': None, 'w/m^2': None}),
    _WeatherColumn('Wind speed', 'wind_speed', {units.wind_speed_label: units for units in WeatherUnits}),
    _WeatherColumn('Wind gust', 'wind_gust', {units.wind_speed_label: units for units in WeatherUnits}),
    _WeatherColumn('Wind bearing', 'wind_bearing', {'deg': None}),
)

# A file's weather columns, one entry for each weather heading in its header.  An entry
# holds the column that heading names and the system to write it in, or None when this
# version does not recognize the heading.
_WeatherLayout = list[tuple[_WeatherColumn, WeatherUnits] | None]


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
        self._weather_units = WeatherUnits.from_setting(config.weather.units)
        # The file whose weather layout the log has already explained, so that the
        # explanation appears once per file rather than once a minute.
        self._explained_layout_of: Path | None = None

    def filename_for_date(self, date: datetime) -> Path:
        return Path(self._config.station.path) / f'noise_data.{date.strftime("%Y-%m-%d")}.csv'

    def append(self, now: datetime, snr: float, signal: float, noise: float,
               lock_status: str, weather: WeatherData,
               *, grid_frequency: CsvValue = '', phase_drift: CsvValue = '') -> str:
        """Append one measurement row, writing the header first if the file is new.

        The weather section follows the file's header.  A new file gets every weather
        column in the configured units, and an existing one keeps the columns and the
        units its header names.

        grid_frequency and phase_drift are keyword-only and default to blank.  They
        are by-products of the analyzer's drift tracking rather than measurements the
        monitor depends on, and a row with no pulse-train lock has nothing to report
        for them.  They are written after Signal Lock Status, ahead of the weather
        fields: read_rows() only ever reads up to index 4, so inserting there leaves
        parsing of both older and newer files completely unaffected.
        """
        csv_filename = self.filename_for_date(now)
        write_header = not csv_filename.exists()
        layout = self._new_weather_layout() if write_header else self._weather_layout_of(csv_filename)
        cells = [now.isoformat(), f'{snr:.2f}', f'{signal:.2f}', f'{noise:.2f}', lock_status,
                 str(grid_frequency), str(phase_drift), *self._weather_cells(weather, layout)]
        csv_str = ','.join(cells)
        with open(csv_filename, 'a') as f:
            if write_header:
                weather_headings = [column.heading(units) for column, units in filter(None, layout)]
                f.write(','.join(self._core_headings() + weather_headings) + '\n')
            f.write(f'{csv_str}\n')
        return csv_str

    def _core_headings(self) -> list[str]:
        """The headings of the columns every row writes by position, ahead of the weather."""
        pps = self._config.audio.pulse_rate
        return ['ISO datetime', f'{pps}pps SNR', f'{pps}pps signal (dBm)', 'Noise floor (dBm)',
                'Signal Lock Status', _GRID_FREQUENCY_HEADING, 'Phase drift (samples/s)']

    def _new_weather_layout(self) -> _WeatherLayout:
        """Every weather column, in the configured units, for a file this row creates."""
        return [(column, self._weather_units) for column in _WEATHER_COLUMNS]

    def _weather_layout_of(self, csv_filename: Path) -> _WeatherLayout:
        """The weather layout an existing file's header describes.

        This reads the header on every append.  An append happens once a minute and the
        header is one short line, so the cost is too small to matter.  A layout read
        fresh also stays right if somebody edits or replaces the file.

        The weather section is every heading after the core columns, counted by
        position.  A file written before the grid frequency columns existed has two
        fewer core columns, so its first two weather headings are counted as core.  Its
        rows then come out as misaligned as they always did on the day of that upgrade.
        """
        with open(csv_filename, newline='') as f:
            header = next(csv.reader(f), [])
        headings = header[len(self._core_headings()):]
        layout = [self._column_headed(heading) for heading in headings]
        if csv_filename != self._explained_layout_of:
            self._explained_layout_of = csv_filename
            self._explain_layout(csv_filename, headings, layout)
        return layout

    def _column_headed(self, heading: str) -> tuple[_WeatherColumn, WeatherUnits] | None:
        """Look up the column a heading names, and the system its unit belongs to.

        This returns None when the table does not have the heading's name, or has it
        with a different unit.  A column such as humidity, whose unit is the same in
        both systems, comes back with the configured system.  Either system would do,
        because humidity reads the same in both.
        """
        match = _WEATHER_HEADING.fullmatch(heading.strip())
        if match is None:
            return None
        column = next((column for column in _WEATHER_COLUMNS if column.name == match['name']), None)
        if column is None or match['label'] not in column.labels:
            return None
        meant = column.labels[match['label']]
        return column, self._weather_units if meant is None else meant

    def _explain_layout(self, csv_filename: Path, headings: list[str], layout: _WeatherLayout) -> None:
        """Log what an existing file's layout does that the configuration would not."""
        unknown = [heading for heading, slot in zip(headings, layout) if slot is None]
        if unknown:
            logger.warning(
                '%s has weather headings this version does not recognize: %s.  The monitor '
                'leaves those columns blank in the rows it adds to that file.  The next new '
                'file gets the current columns.', csv_filename.name, ', '.join(unknown))
        kept = sorted({units for _, units in filter(None, layout)} - {self._weather_units})
        if kept:
            logger.info(
                '[weather] units is %s, but %s already records weather in %s units.  The '
                'monitor keeps that file in %s units.  The next new file uses %s units.',
                self._weather_units, csv_filename.name, kept[0], kept[0], self._weather_units)

    @staticmethod
    def _weather_cells(weather: WeatherData, layout: _WeatherLayout) -> list[str]:
        """One cell per heading in `layout`, each in the units its heading names."""
        converted = {units: weather.in_units(units) for units in {units for _, units in filter(None, layout)}}
        return ['' if slot is None else str(slot[0].value(converted[slot[1]])) for slot in layout]

    def read_rows(self, input_filename: Path | str) -> list[CsvRow]:
        """Parse one CSV file into CsvRow records, skipping headers and malformed lines.

        It converts timestamps to the station timezone.  Old-format files without
        the Signal Lock Status column get a non-'none' value at index 4 (either the
        temperature field or nothing), which correctly reads as locked.  Rows too
        short to hold the measurement fields default the status to 'full'.

        It reads nothing beyond index 4.  That is deliberate, and is what lets columns
        be added after Signal Lock Status without breaking files written by older
        versions: the fields that shift are ones this parser never looks at.
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

    def read_grid_frequencies(self, input_filename: Path | str) -> list[tuple[datetime, float | None]]:
        """One day's (timestamp, grid frequency) pairs, with None where there was no lock.

        This finds the column by its header name rather than by its position, and that
        is not fussiness.  Grid frequency sits at index 5, which in a file written
        before that column existed holds Temperature.  Reading by position would plot
        degrees as hertz with nothing on the chart to say so, which is the failure
        this module's own docstring warns about when it says nothing past index 4 is
        read.  A file whose header does not name the column reports no readings at
        all, so an older day comes out empty rather than wrong.

        One case still comes out empty that holds real data: a file created by a
        version without the column and appended to by a version with it, which is the
        single day an upgrade happens on.  The header is written once, when the file
        is created, so it describes the older rows.  Showing nothing for that day is
        the safe side of the same trade.
        """
        zone = ZoneInfo(self._config.station.timezone)
        readings: list[tuple[datetime, float | None]] = []
        with open(input_filename, newline='') as f:
            rows = csv.reader(f)
            header = next(rows, None)
            if header is None or _GRID_FREQUENCY_HEADING not in header:
                return []
            column = header.index(_GRID_FREQUENCY_HEADING)
            for row in rows:
                if len(row) <= column:
                    continue
                try:
                    timestamp = datetime.fromisoformat(row[0]).astimezone(zone)
                except ValueError:
                    continue
                # An unlocked minute writes the field blank rather than a number, so
                # a blank is a real reading of "nothing to report" and not a defect.
                text = row[column].strip()
                try:
                    readings.append((timestamp, float(text) if text else None))
                except ValueError:
                    readings.append((timestamp, None))
        return readings

    def _read_day_scores(self, input_filename: Path | str) -> dict[time, int]:
        """Read one day's CSV file and return a {time: score} dict bucketed to 15-minute intervals.

        It counts only rows where the signal is at or above the noise threshold AND the SNR is at
        or above snr_gate.  Each qualifying row contributes log(snr, snr_gate)
        to its bucket, so stronger events weigh more than just-threshold events.  The
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

        It silently skips missing CSV files (days with no data).  The returned
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

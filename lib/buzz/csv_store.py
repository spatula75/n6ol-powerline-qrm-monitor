"""
CSV persistence layer for noise measurements.

Each day's data lives in a separate file named noise_data.YYYY-MM-DD.csv in the
configured output directory.  CsvStore owns the file format end to end: writing
new rows, parsing files back into typed CsvRow records (including old-format
files that predate the Signal Lock Status column), and aggregating a date range
into the time-bucketed score dict the summary graphs consume.

Every row follows its file's own header.  _COLUMNS lists each column this version
knows, with its heading and how to fill it, and a new file gets all of them.  A row
added to an existing file gets one cell for each heading already there, in that
heading's order, and in the units its heading names.  So a file keeps its columns and
its units until midnight, whatever changed in between: a new version with more
columns, or a change to `[weather] units` after a restart.

read_rows() reads the first five columns by position, so those stay first.  A new
column can go anywhere after them, and a file written before it simply goes without
it.  Putting a new column at the end keeps an older file's header a prefix of a newer
one, which suits external tools that read by position.
"""

import csv
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from math import log
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from zoneinfo import ZoneInfo

from buzz.config import BuzzConfig
from buzz.weather import CsvValue, WeatherData, WeatherUnits

logger = logging.getLogger(__name__)

# Summary scores are bucketed to intervals of this many minutes.  The summary
# graph builds its time axis from the same constant so the two can't drift apart.
BUCKET_MINUTES = 15

# The header cell naming the grid-frequency column.  _COLUMNS writes it and
# read_grid_frequencies() reads it back, and a test holds the two together, so a rename
# cannot quietly turn every frequency chart empty.
_GRID_FREQUENCY_HEADING = 'Grid frequency (Hz)'


# The heading format: a name, then an optional qualifier in square brackets, then an
# optional unit in parentheses, as in "SNR [120 pps] (dB)", "Noise floor (dBm)" and
# "ISO datetime".  The parentheses only ever hold a unit.  The brackets hold anything
# else a reader needs to know about what the column measured.
_HEADING_FORMAT = re.compile(
    r'(?P<name>[^\[\]()]+?)(?: \[(?P<qualifier>[^\[\]()]*)\])?(?: \((?P<unit>[^\[\]()]*)\))?')


class _Row(NamedTuple):
    """Everything one row reports, before it is laid out under a header."""
    now: datetime
    snr: float
    signal: float
    noise: float
    lock_status: str
    grid_frequency: CsvValue
    phase_drift: CsvValue
    weather: WeatherData


@dataclass(frozen=True)
class _Qualifier:
    """What a column's heading says in square brackets.

    `pattern` accepts the texts an existing file may carry there, and may capture the
    pulse rate as `rate`.  `text` writes the qualifier for a new file, from the
    configured pulse rate.
    """
    pattern: re.Pattern[str]
    text: Callable[[int], str]


# The qualifier of a column measured on the pulse train, as in "[120 pps]".  Any rate
# is accepted, so a file headed for another rate keeps its measurements.
_PULSE_RATE = _Qualifier(re.compile(r'(?P<rate>\d+) pps'), lambda pulse_rate: f'{pulse_rate} pps')


class _HeadingMatch(NamedTuple):
    """What a heading says about its column: the system its unit belongs to, and its pulse rate."""
    system: WeatherUnits | None  # None where the unit reads the same in both systems
    rate: str | None             # None where the heading shows no pulse rate


# The labels of a column whose heading shows no unit, such as "ISO datetime".
_NO_UNIT: Mapping[str | None, WeatherUnits | None] = MappingProxyType({None: None})


# Each column is one entry in _COLUMNS, so columns compare by identity, and eq=False
# keeps them hashable although `labels` is a dict.
@dataclass(frozen=True, eq=False)
class _Column:
    """One CSV column: its heading, and how to fill its cell from a row.

    A new file heads the column in the heading format, `name [qualifier] (unit)`.
    `labels` lists every unit the heading may show, with the system each one belongs
    to: "F" is imperial and "C" is metric.  Humidity's "%" is the same in both systems,
    so its entry is None.  A heading that shows no unit at all has the single entry
    None: None.  A new file's header shows the first entry that fits the configured
    system.

    `qualifier` is what the heading says in square brackets, for a column that says
    anything there.  `deprecated_formats` are the heading formats earlier versions wrote
    for this column, each a pattern for the whole heading.  A pattern may capture `rate`
    and `unit`.  One that captures no unit stands for a unit that reads the same in both
    systems.  A row added to a file an earlier version started finds the column through
    these.

    `cell` makes the column's text from the row, and from the row's weather already
    converted to the units the heading names.  A `required` column holds a measurement
    that no row may lose.
    """
    name: str
    cell: Callable[[_Row, WeatherData], str]
    labels: Mapping[str | None, WeatherUnits | None] = _NO_UNIT
    qualifier: _Qualifier | None = None
    deprecated_formats: tuple[re.Pattern[str], ...] = ()
    required: bool = False

    @classmethod
    def weather(cls, name: str, field: str, labels: Mapping[str | None, WeatherUnits | None]) -> '_Column':
        """A column that holds the WeatherData field named `field`."""
        return cls(name, lambda row, weather: str(getattr(weather, field)), labels)

    def heading(self, units: WeatherUnits, pulse_rate: int) -> str:
        """The heading a new file gives this column, in the heading format."""
        label = next(label for label, meant in self.labels.items() if meant in (units, None))
        qualifier = '' if self.qualifier is None else f' [{self.qualifier.text(pulse_rate)}]'
        unit = '' if label is None else f' ({label})'
        return f'{self.name}{qualifier}{unit}'

    def match(self, heading: str) -> _HeadingMatch | None:
        """Read `heading` as this column's, in the heading format or a deprecated one.

        This returns None when the heading names another column.  It also returns None
        when the heading shows a unit this column does not list, or a qualifier it does
        not take.
        """
        heading = heading.strip()
        current = _HEADING_FORMAT.fullmatch(heading)
        if current is not None and current['name'] == self.name:
            return self._match_current(current)
        for deprecated in self.deprecated_formats:
            old = deprecated.fullmatch(heading)
            if old is not None:
                groups = old.groupdict()
                if 'unit' not in groups:
                    return _HeadingMatch(None, groups.get('rate'))
                return self._match_unit(groups['unit'], groups.get('rate'))
        return None

    def _match_current(self, parsed: re.Match[str]) -> _HeadingMatch | None:
        """Check the qualifier and the unit of a heading already known to carry this name."""
        written = parsed['qualifier']
        if self.qualifier is None:
            return None if written is not None else self._match_unit(parsed['unit'], None)
        fits = None if written is None else self.qualifier.pattern.fullmatch(written)
        if fits is None:
            return None
        return self._match_unit(parsed['unit'], fits.groupdict().get('rate'))

    def _match_unit(self, unit: str | None, rate: str | None) -> _HeadingMatch | None:
        """The match for `unit` and `rate`, when `unit` is one this column lists."""
        if unit not in self.labels:
            return None
        return _HeadingMatch(self.labels[unit], rate)


# Every column this version writes, in the order a new header lists them.  The labels
# for temperature and wind come from WeatherUnits, so a label cannot change in one place
# and not the other.
_COLUMNS = (
    _Column('ISO datetime', lambda row, _: row.now.isoformat(), required=True),
    # Every release from 1.0.0 through 2.1.0 headed SNR as "120pps SNR", with no unit.
    _Column('SNR', lambda row, _: f'{row.snr:.2f}', {'dB': None}, qualifier=_PULSE_RATE,
            deprecated_formats=(re.compile(r'(?P<rate>\d+)pps SNR'),), required=True),
    # The same releases headed the signal as "120pps signal (dBm)".
    _Column('Signal', lambda row, _: f'{row.signal:.2f}', {'dBm': None}, qualifier=_PULSE_RATE,
            deprecated_formats=(re.compile(r'(?P<rate>\d+)pps signal \((?P<unit>dBm)\)'),), required=True),
    _Column('Noise floor', lambda row, _: f'{row.noise:.2f}', {'dBm': None}, required=True),
    _Column('Signal Lock Status', lambda row, _: row.lock_status, required=True),
    _Column('Grid frequency', lambda row, _: str(row.grid_frequency), {'Hz': None}),
    _Column('Phase drift', lambda row, _: str(row.phase_drift), {'samples/s': None}),
    _Column.weather('Temperature', 'temperature', {units.temperature_label: units for units in WeatherUnits}),
    _Column.weather('Humidity', 'humidity', {'%': None}),
    # Files written before the watt got its capital W say "w/m^2", and still match.
    _Column.weather('Solar radiation', 'solar_radiation', {'W/m^2': None, 'w/m^2': None}),
    _Column.weather('Wind speed', 'wind_speed', {units.wind_speed_label: units for units in WeatherUnits}),
    _Column.weather('Wind gust', 'wind_gust', {units.wind_speed_label: units for units in WeatherUnits}),
    _Column.weather('Wind bearing', 'wind_bearing', {'deg': None}),
)

# A file's columns, one entry for each heading in its header.  An entry holds the
# column that heading names and the system to write it in, or None when this version
# does not recognize the heading.
_Layout = list[tuple[_Column, WeatherUnits] | None]


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

        The row follows the file's header.  A new file gets every column in the
        configured units, and an existing one keeps the columns and the units its header
        names.

        grid_frequency and phase_drift are keyword-only and default to blank.  They
        are by-products of the analyzer's drift tracking rather than measurements the
        monitor depends on, and a row with no pulse-train lock has nothing to report
        for them.
        """
        csv_filename = self.filename_for_date(now)
        write_header = not csv_filename.exists()
        layout = self._new_layout() if write_header else self._layout_of(csv_filename)
        row = _Row(now, snr, signal, noise, lock_status, grid_frequency, phase_drift, weather)
        csv_str = ','.join(self._cells(row, layout))
        with open(csv_filename, 'a') as f:
            if write_header:
                pulse_rate = self._config.audio.pulse_rate
                f.write(','.join(column.heading(units, pulse_rate) for column, units in filter(None, layout)) + '\n')
            f.write(f'{csv_str}\n')
        return csv_str

    def _new_layout(self) -> _Layout:
        """Every column, in the configured units, for a file this row creates."""
        return [(column, self._weather_units) for column in _COLUMNS]

    def _layout_of(self, csv_filename: Path) -> _Layout:
        """The layout an existing file's header describes.

        This reads the header on every append.  An append happens once a minute and the
        header is one short line, so the cost is too small to matter.  A layout read
        fresh also stays right if somebody edits or replaces the file.

        A header without a heading for every required measurement falls back to the
        current layout, because following it would drop a measurement from every row.
        """
        with open(csv_filename, newline='') as f:
            header = next(csv.reader(f), [])
        layout = [self._column_headed(heading) for heading in header]
        found = {slot[0] for slot in filter(None, layout)}
        missing = [column for column in _COLUMNS if column.required and column not in found]
        if csv_filename != self._explained_layout_of:
            self._explained_layout_of = csv_filename
            self._explain_layout(csv_filename, header, layout, missing)
        return self._new_layout() if missing else layout

    def _column_headed(self, heading: str) -> tuple[_Column, WeatherUnits] | None:
        """Look up the column a heading names, and the system its unit belongs to.

        This returns None for a heading no column in the table recognizes.  A column
        such as humidity, whose unit is the same in both systems, comes back with the
        configured system.  Either system would do, because humidity reads the same in
        both.
        """
        found = self._match_heading(heading)
        if found is None:
            return None
        column, match = found
        return column, self._weather_units if match.system is None else match.system

    @staticmethod
    def _match_heading(heading: str) -> tuple[_Column, _HeadingMatch] | None:
        """The column in the table that recognizes `heading`, and what the heading says about it."""
        return next(((column, match) for column in _COLUMNS if (match := column.match(heading))), None)

    def _explain_layout(self, csv_filename: Path, header: list[str], layout: _Layout,
                        missing: list[_Column]) -> None:
        """Log what an existing file's header does that the configuration would not."""
        pulse_rate = self._config.audio.pulse_rate
        if missing:
            logger.warning(
                '%s has no heading for %s, so the monitor cannot tell where those '
                'measurements go.  It writes each row in the current layout instead, which '
                'may not match that header.  The next new file gets a matching header.',
                csv_filename.name, ', '.join(column.heading(self._weather_units, pulse_rate)
                                             for column in missing))
            return
        unknown = [heading for heading, slot in zip(header, layout) if slot is None]
        if unknown:
            logger.warning(
                '%s has headings this version does not recognize: %s.  The monitor leaves '
                'those columns blank in the rows it adds to that file.  The next new file '
                'gets the current columns.', csv_filename.name, ', '.join(unknown))
        kept = sorted({units for _, units in filter(None, layout)} - {self._weather_units})
        if kept:
            logger.info(
                '[weather] units is %s, but %s already records weather in %s units.  The '
                'monitor keeps that file in %s units.  The next new file uses %s units.',
                self._weather_units, csv_filename.name, kept[0], kept[0], self._weather_units)
        rates = sorted({found[1].rate for heading in header
                        if (found := self._match_heading(heading)) and found[1].rate})
        if rates and rates != [str(pulse_rate)]:
            logger.info(
                '[audio] pulse_rate is %d, but %s is headed for %s pps.  The monitor keeps '
                'writing measurements under those headings.  The next new file is headed '
                'for %d pps.', pulse_rate, csv_filename.name, ' and '.join(rates), pulse_rate)

    @staticmethod
    def _cells(row: _Row, layout: _Layout) -> list[str]:
        """One cell per heading in `layout`, each in the units its heading names."""
        converted = {units: row.weather.in_units(units) for units in {units for _, units in filter(None, layout)}}
        return ['' if slot is None else slot[0].cell(row, converted[slot[1]]) for slot in layout]

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
        degrees as hertz with nothing on the chart to say so.  A file whose header
        does not name the column reports no readings at all, so an older day comes out
        empty rather than wrong.

        One case comes out empty that holds real data.  Before rows followed their
        file's header, the version that added this column wrote it into files an older
        version had started, and those headers do not name it.  That happened on the
        single day of an upgrade, and showing nothing for such a day is the safe side of
        the same trade.  A file started without the column now simply goes without it.
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

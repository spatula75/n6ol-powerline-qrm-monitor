"""Tests for CsvStore: filename generation, row append, time bucketing, and range aggregation."""

import logging
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from buzz.config import BuzzConfig
from buzz.csv_store import _COLUMNS, _GRID_FREQUENCY_HEADING, _HEADING_FORMAT, CsvRow, CsvStore, _Row
from buzz.weather import EMPTY_WEATHER, WeatherData, WeatherUnits

_TZ = ZoneInfo('America/Los_Angeles')

# One observation in the units every weather client returns, degrees C, km/h and mm.  In
# the default imperial units it comes out as 68.0 F, 7.5 MPH and 12.0 MPH.  The rain and
# the weather timestamp have no column yet.
_WEATHER = WeatherData(20.0, 52.0, 300.0, 12.0, 19.3, 225, 5.08,
                       datetime(2024, 1, 15, 18, 29, tzinfo=UTC))


def _make_store(tmp_path: Path) -> CsvStore:
    cfg = BuzzConfig()
    cfg.station.path = str(tmp_path)
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.station.noise_floor = -98.0
    cfg.station.noise_min_snr = 12.0
    return CsvStore(cfg)


def _ts(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=_TZ)


class TestFilenameForDate:
    def test_filename_contains_date(self, tmp_path):
        store = _make_store(tmp_path)
        path = store.filename_for_date(_ts(2024, 1, 15, 10, 0))
        assert '2024-01-15' in path.name

    def test_filename_in_configured_directory(self, tmp_path):
        store = _make_store(tmp_path)
        path = store.filename_for_date(_ts(2024, 1, 15, 10, 0))
        assert path.parent == tmp_path

    def test_different_dates_give_different_filenames(self, tmp_path):
        store = _make_store(tmp_path)
        p1 = store.filename_for_date(_ts(2024, 1, 15, 10, 0))
        p2 = store.filename_for_date(_ts(2024, 1, 16, 10, 0))
        assert p1 != p2


class TestGridFrequencyColumns:
    """Grid frequency and phase drift are logged after Signal Lock Status.

    That position is chosen so the change is invisible to read_rows(), which stops
    at index 4 - files written before and after the change parse identically.
    """

    def _row(self, tmp_path, **kwargs) -> list[str]:
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER, **kwargs)
        lines = store.filename_for_date(now).read_text().splitlines()
        return lines[1].split(',')

    def test_header_names_both_columns(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', EMPTY_WEATHER)
        header = store.filename_for_date(now).read_text().splitlines()[0].split(',')
        assert header[5] == 'Grid frequency (Hz)'
        assert header[6] == 'Phase drift (samples/s)'

    def test_values_land_immediately_after_lock_status(self, tmp_path):
        fields = self._row(tmp_path, grid_frequency='60.023', phase_drift='-6.12')
        assert fields[4] == 'full'
        assert fields[5] == '60.023'
        assert fields[6] == '-6.12'

    def test_weather_still_follows_them(self, tmp_path):
        fields = self._row(tmp_path, grid_frequency='60.023', phase_drift='-6.12')
        assert fields[7:] == ['68.0', '52.0', '300.0', '7.5', '12.0', '225']

    def test_rain_and_the_weather_timestamp_are_not_written_yet(self, tmp_path):
        """The header has no column for either, so the row stops at wind bearing."""
        fields = self._row(tmp_path)
        assert len(fields) == 13
        assert fields[-1] == '225'

    def test_default_is_blank_not_zero(self, tmp_path):
        """A minute with no lock has nothing to report, and 0.000 Hz would be a lie."""
        fields = self._row(tmp_path)
        assert fields[5] == '' and fields[6] == ''

    def test_row_still_parses_with_the_new_columns(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'partial', _WEATHER,
                     grid_frequency='60.023', phase_drift='-6.12')
        rows = store.read_rows(store.filename_for_date(now))
        assert len(rows) == 1
        assert (rows[0].snr, rows[0].signal, rows[0].noise) == (15.0, -80.0, -95.0)
        assert rows[0].lock_status == 'partial'

    def test_old_format_rows_written_before_the_change_still_parse(self, tmp_path):
        """The compatibility guarantee: rows with the pre-change column layout."""
        path = tmp_path / 'old.csv'
        path.write_text(
            'ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
            'Temperature (F),Humidity (%),Solar radiation (w/m^2),'
            'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)\n'
            '2024-01-15T10:30:00-08:00,15.00,-80.00,-95.00,partial,68.0,52.0,300.0,7.5,12.0,225\n'
        )
        rows = _make_store(tmp_path).read_rows(path)
        assert len(rows) == 1
        assert (rows[0].snr, rows[0].signal, rows[0].noise) == (15.0, -80.0, -95.0)
        assert rows[0].lock_status == 'partial'


class TestReadGridFrequencies:
    """Reading the frequency column back, which is only safe by header name.

    Grid frequency sits at index 5, and in a file written before that column existed
    index 5 holds Temperature.  Every test here is really about telling those apart.
    """

    _NEW_HEADER = ('ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
                   'Grid frequency (Hz),Phase drift (samples/s),Temperature (F),Humidity (%),'
                   'Solar radiation (W/m^2),Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)')
    # Pre-grid-frequency layout: the sixth column is Temperature, not frequency.
    _OLD_HEADER = ('ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
                   'Temperature (F),Humidity (%),Solar radiation (w/m^2),'
                   'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)')

    def _write(self, path: Path, header: str, rows: list[str]) -> Path:
        path.write_text('\n'.join([header, *rows]) + '\n')
        return path

    def test_it_reads_the_values(self, tmp_path):
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'a.csv', self._NEW_HEADER, [
            '2024-01-15T10:00:00-08:00,15.0,-80.0,-95.0,full,60.021,-6.1,68,52,300,7,12,225',
            '2024-01-15T10:01:00-08:00,15.0,-80.0,-95.0,full,59.987,-6.2,68,52,300,7,12,225',
        ])
        assert [value for _, value in store.read_grid_frequencies(path)] == [60.021, 59.987]

    def test_an_unlocked_minute_reads_as_no_value(self, tmp_path):
        """A blank is a real reading of "nothing to report", not a damaged row."""
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'a.csv', self._NEW_HEADER, [
            '2024-01-15T10:00:00-08:00,0.00,-95.0,-95.0,none,,,68,52,300,7,12,225',
            '2024-01-15T10:01:00-08:00,15.0,-80.0,-95.0,full,60.010,-6.1,68,52,300,7,12,225',
        ])
        assert [value for _, value in store.read_grid_frequencies(path)] == [None, 60.010]

    def test_an_old_file_reports_nothing_rather_than_temperatures(self, tmp_path):
        """The whole reason this reads by header name.

        In the old layout the sixth column is Temperature.  Reading by position would
        chart 68 degrees as 68 Hz, put it far outside the band, and mark it as an
        excursion, with nothing anywhere to say the number was never a frequency.
        """
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'old.csv', self._OLD_HEADER, [
            '2024-01-15T10:00:00-08:00,15.0,-80.0,-95.0,full,68,52,300,7,12,225',
            '2024-01-15T10:01:00-08:00,15.0,-80.0,-95.0,full,69,52,300,7,12,225',
        ])
        assert store.read_grid_frequencies(path) == [], (
            'An old-format file gave up readings it does not have, so the sixth '
            'column was read by position rather than by name.'
        )

    def test_a_file_with_no_header_reports_nothing(self, tmp_path):
        """Without a header there is no way to know which column is which."""
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'bare.csv',
                           '2024-01-15T10:00:00-08:00,15.0,-80.0,-95.0,full,60.021,-6.1,68',
                           ['2024-01-15T10:01:00-08:00,15.0,-80.0,-95.0,full,60.022,-6.1,68'])
        assert store.read_grid_frequencies(path) == []

    def test_an_unparseable_value_reads_as_no_value(self, tmp_path):
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'a.csv', self._NEW_HEADER, [
            '2024-01-15T10:00:00-08:00,15.0,-80.0,-95.0,full,not-a-number,-6.1,68,52,300,7,12,225',
        ])
        assert [value for _, value in store.read_grid_frequencies(path)] == [None]

    def test_a_short_row_is_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'a.csv', self._NEW_HEADER, [
            '2024-01-15T10:00:00-08:00,15.0,-80.0',
            '2024-01-15T10:01:00-08:00,15.0,-80.0,-95.0,full,60.010,-6.1,68,52,300,7,12,225',
        ])
        assert [value for _, value in store.read_grid_frequencies(path)] == [60.010]

    def test_timestamps_come_back_in_the_station_timezone(self, tmp_path):
        store = _make_store(tmp_path)
        path = self._write(tmp_path / 'a.csv', self._NEW_HEADER, [
            '2024-01-15T18:00:00+00:00,15.0,-80.0,-95.0,full,60.010,-6.1,68,52,300,7,12,225',
        ])
        when, _ = store.read_grid_frequencies(path)[0]
        assert when.tzinfo is not None and when.hour == 10   # 18:00 UTC is 10:00 PST

    def test_what_it_writes_is_what_it_reads_back(self, tmp_path):
        """The header constant is shared by both sides, and this is what pins that.

        A rename on one side alone would leave every frequency chart silently empty.
        """
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER,
                     grid_frequency='60.023', phase_drift='-6.12')
        readings = store.read_grid_frequencies(store.filename_for_date(now))
        assert [value for _, value in readings] == [60.023]


class TestAppend:
    def test_creates_file_with_headers_on_first_call(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        content = store.filename_for_date(now).read_text()
        assert 'ISO datetime' in content
        assert 'SNR [120 pps] (dB)' in content

    def test_no_headers_on_subsequent_calls(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        store.append(now, 16.0, -81.0, -96.0, 'full', _WEATHER)
        lines = store.filename_for_date(now).read_text().strip().split('\n')
        header_count = sum(1 for l in lines if 'ISO datetime' in l)
        assert header_count == 1

    def test_returns_csv_string_without_newline(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        assert '\n' not in result

    def test_csv_string_contains_snr(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 17.5, -80.0, -95.0, 'full', _WEATHER)
        assert '17.50' in result

    def test_multiple_appends_produce_multiple_rows(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        for i in range(3):
            store.append(now, 15.0 + i, -80.0, -95.0, 'full', EMPTY_WEATHER)
        lines = [l for l in store.filename_for_date(now).read_text().strip().split('\n')
                 if 'ISO datetime' not in l]
        assert len(lines) == 3

    def test_accepts_string_weather_values(self, tmp_path):
        """Blank strings for every weather field - what a collection with weather
        disabled actually passes - must not raise trying to format them as numbers,
        and must come back as the blank fields they were, not dropped or defaulted."""
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'full', EMPTY_WEATHER)
        assert result == f'{now.isoformat()},15.00,-80.00,-95.00,full,,,,,,,,'

    def test_lock_status_in_header(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        content = store.filename_for_date(now).read_text()
        assert 'Signal Lock Status' in content

    def test_lock_status_written_to_row(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'partial', _WEATHER)
        assert 'partial' in result

    @pytest.mark.parametrize('status', ['full', 'partial', 'none'])
    def test_all_lock_statuses_accepted(self, tmp_path, status):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 0.0, -90.0, -90.0, status, EMPTY_WEATHER)
        assert status in result


class TestReadRows:
    def test_round_trips_appended_row(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'partial', _WEATHER)
        rows = store.read_rows(store.filename_for_date(now))
        assert rows == [CsvRow(timestamp=now, snr=15.0, signal=-80.0, noise=-95.0,
                               lock_status='partial')]

    def test_header_row_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', EMPTY_WEATHER)
        rows = store.read_rows(store.filename_for_date(now))
        assert len(rows) == 1

    def test_timestamp_converted_to_station_timezone(self, tmp_path):
        store = _make_store(tmp_path)
        utc_now = datetime(2024, 1, 15, 18, 30, tzinfo=ZoneInfo('UTC'))
        store.append(utc_now, 15.0, -80.0, -95.0, 'full', EMPTY_WEATHER)
        rows = store.read_rows(store.filename_for_date(utc_now))
        assert rows[0].timestamp.tzinfo == ZoneInfo('America/Los_Angeles')
        assert rows[0].timestamp == utc_now

    def test_old_format_row_without_lock_column_reads_as_locked(self, tmp_path):
        store = _make_store(tmp_path)
        path = tmp_path / 'old.csv'
        # Old format: temperature directly after noise floor, no lock column
        path.write_text(f'{_ts(2024, 1, 15, 10, 30).isoformat()},15.0,-80.0,-95.0\n')
        rows = store.read_rows(path)
        assert rows[0].lock_status == 'full'

    def test_malformed_rows_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        path = tmp_path / 'bad.csv'
        path.write_text('not,valid,data\nalso bad\n')
        assert store.read_rows(path) == []


class TestReadDateToTimeDict:
    def _write_csv(self, path: Path, rows: list[str]) -> None:
        path.write_text('\n'.join(rows) + '\n')

    def test_qualifying_row_appears_in_dict(self, tmp_path):
        store = _make_store(tmp_path)
        # signal >= -86, snr >= 15
        row = f'{_ts(2024, 1, 15, 10, 23).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, ['header,line', row])
        result = store._read_day_scores(csv_path)
        assert time(10, 15) in result

    def test_low_snr_row_excluded(self, tmp_path):
        store = _make_store(tmp_path)
        # snr=10 < 15 (snr_gate)
        row = f'{_ts(2024, 1, 15, 10, 23).isoformat()},10.0,-80.0,-95.0,72,50,300,5,8,180'
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, [row])
        result = store._read_day_scores(csv_path)
        assert len(result) == 0

    def test_low_signal_row_excluded(self, tmp_path):
        store = _make_store(tmp_path)
        # signal=-90 < -86 (noise_threshold)
        row = f'{_ts(2024, 1, 15, 10, 23).isoformat()},20.0,-90.0,-95.0,72,50,300,5,8,180'
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, [row])
        result = store._read_day_scores(csv_path)
        assert len(result) == 0

    def test_bucket_to_15_minute_interval(self, tmp_path):
        store = _make_store(tmp_path)
        row_10_23 = f'{_ts(2024, 1, 15, 10, 23).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        row_10_14 = f'{_ts(2024, 1, 15, 10, 14).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        row_10_45 = f'{_ts(2024, 1, 15, 10, 45).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, [row_10_23, row_10_14, row_10_45])
        result = store._read_day_scores(csv_path)
        assert time(10, 15) in result   # 10:23 → 10:15
        assert time(10, 0) in result    # 10:14 → 10:00
        assert time(10, 45) in result   # 10:45 → 10:45

    def test_two_rows_same_bucket_accumulate(self, tmp_path):
        store = _make_store(tmp_path)
        row1 = f'{_ts(2024, 1, 15, 10, 20).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        row2 = f'{_ts(2024, 1, 15, 10, 25).isoformat()},20.0,-80.0,-95.0,72,50,300,5,8,180'
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, [row1, row2])
        result = store._read_day_scores(csv_path)
        one_row_store = _make_store(tmp_path)
        one_csv = tmp_path / 'single.csv'
        self._write_csv(one_csv, [row1])
        one_result = one_row_store._read_day_scores(one_csv)
        assert result[time(10, 15)] > one_result[time(10, 15)]

    def test_header_line_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, ['ISO datetime,120pps SNR,120pps signal dB,Noise floor dB,...'])
        result = store._read_day_scores(csv_path)
        assert len(result) == 0

    def test_bad_lines_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        csv_path = store.filename_for_date(_ts(2024, 1, 15, 0, 0))
        self._write_csv(csv_path, ['not,valid,data', 'also bad'])
        result = store._read_day_scores(csv_path)
        assert len(result) == 0


class TestReadRangeToTimeDict:
    def _write_qualifying_row(self, store: CsvStore, when: datetime) -> None:
        store.append(when, 20.0, -80.0, -95.0, 'full', _WEATHER)

    def test_missing_files_silently_skipped(self, tmp_path):
        """No file exists for any day in the range - every day hits the
        FileNotFoundError branch, so the aggregate is empty rather than raising."""
        store = _make_store(tmp_path)
        start = _ts(2024, 1, 15, 0, 0)
        end = _ts(2024, 1, 17, 0, 0)
        assert store.read_range_scores(start, end) == {}

    def test_single_day_aggregated(self, tmp_path):
        store = _make_store(tmp_path)
        when = _ts(2024, 1, 15, 10, 20)
        self._write_qualifying_row(store, when)
        result = store.read_range_scores(
            _ts(2024, 1, 15, 0, 0),
            _ts(2024, 1, 15, 23, 59),
        )
        assert time(10, 15) in result

    def test_multiple_days_summed(self, tmp_path):
        store = _make_store(tmp_path)
        self._write_qualifying_row(store, _ts(2024, 1, 15, 10, 20))
        self._write_qualifying_row(store, _ts(2024, 1, 16, 10, 20))
        single_day_result = store.read_range_scores(
            _ts(2024, 1, 15, 0, 0), _ts(2024, 1, 15, 23, 59),
        )
        two_day_result = store.read_range_scores(
            _ts(2024, 1, 15, 0, 0), _ts(2024, 1, 16, 23, 59),
        )
        assert two_day_result[time(10, 15)] > single_day_result[time(10, 15)]

    def test_returns_plain_dict_not_defaultdict(self, tmp_path):
        store = _make_store(tmp_path)
        start = _ts(2024, 1, 15, 0, 0)
        end = _ts(2024, 1, 15, 23, 59)
        result = store.read_range_scores(start, end)
        assert type(result) is dict


class TestWeatherUnitsInTheCsv:
    """The header names the weather units, and every row in a file matches its header."""

    def _store(self, tmp_path: Path, units: str) -> CsvStore:
        store = _make_store(tmp_path)
        store._config.weather.units = units
        return CsvStore(store._config)

    def _lines(self, store: CsvStore, now: datetime) -> list[list[str]]:
        return [line.split(',') for line in store.filename_for_date(now).read_text().splitlines()]

    def test_a_metric_station_names_metric_units_and_writes_them(self, tmp_path):
        store = self._store(tmp_path, 'metric')
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        header, row = self._lines(store, now)
        assert header[7:] == ['Temperature (C)', 'Humidity (%)', 'Solar radiation (W/m^2)',
                              'Wind speed (km/h)', 'Wind gust (km/h)', 'Wind bearing (deg)']
        assert row[7:] == ['20.0', '52.0', '300.0', '12.0', '19.3', '225']

    def test_an_imperial_file_stays_imperial_after_the_setting_changes(self, tmp_path):
        """A restart with the setting changed must not put Celsius under an F header."""
        now = _ts(2024, 1, 15, 10, 30)
        self._store(tmp_path, 'imperial').append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        metric = self._store(tmp_path, 'metric')
        metric.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        header, first, second = self._lines(metric, now)
        assert header[7] == 'Temperature (F)'
        assert first[7:] == second[7:] == ['68.0', '52.0', '300.0', '7.5', '12.0', '225']

    def test_the_next_new_file_uses_the_new_setting(self, tmp_path):
        today, tomorrow = _ts(2024, 1, 15, 23, 59), _ts(2024, 1, 16, 0, 0)
        self._store(tmp_path, 'imperial').append(today, 15.0, -80.0, -95.0, 'full', _WEATHER)
        metric = self._store(tmp_path, 'metric')
        metric.append(today, 15.0, -80.0, -95.0, 'full', _WEATHER)
        metric.append(tomorrow, 15.0, -80.0, -95.0, 'full', _WEATHER)
        header, row = self._lines(metric, tomorrow)
        assert (header[7], row[7]) == ('Temperature (C)', '20.0')

    def test_the_log_explains_a_kept_unit_once_per_file(self, tmp_path, caplog):
        now = _ts(2024, 1, 15, 10, 30)
        self._store(tmp_path, 'imperial').append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        metric = self._store(tmp_path, 'metric')
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            for _ in range(3):
                metric.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        explanations = [r for r in caplog.records if 'already records weather in imperial' in r.getMessage()]
        assert len(explanations) == 1

    def test_a_first_line_that_is_not_a_header_gets_the_full_current_row(self, tmp_path):
        """Following it would drop every measurement, so the row is written whole instead."""
        store = self._store(tmp_path, 'metric')
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text('not a header\n')
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        row = self._lines(store, now)[1]
        assert row[1:5] == ['15.00', '-80.00', '-95.00', 'full']
        assert row[7:] == ['20.0', '52.0', '300.0', '12.0', '19.3', '225']


# The core headings a file started before SNR moved its pulse rate begins with.  Rows added
# to one of these must still fill every column.
_CORE = ('ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
         'Grid frequency (Hz),Phase drift (samples/s)')


class TestHeaderDrivenWeatherColumns:
    """A row fills the weather columns its file's header lists, in the units each heading names."""

    def _append_to(self, tmp_path: Path, weather_headings: str, units: str = 'imperial') -> tuple[list[str], str]:
        """Append one row to a file that already has `weather_headings`, returning header and row."""
        store = _make_store(tmp_path)
        store._config.weather.units = units
        store = CsvStore(store._config)
        now = _ts(2024, 1, 15, 10, 30)
        header = f'{_CORE},{weather_headings}' if weather_headings else _CORE
        store.filename_for_date(now).write_text(header + '\n')
        row = store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER, grid_frequency='60.010', phase_drift='-6.1')
        return header.split(','), row

    def test_a_new_file_gets_exactly_this_header(self, tmp_path):
        """Pinned as text, so a change to the table that alters the header shows up here."""
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        assert store.filename_for_date(now).read_text().splitlines()[0] == (
            'ISO datetime,SNR [120 pps] (dB),Signal [120 pps] (dBm),Noise floor (dBm),Signal Lock Status,'
            'Grid frequency (Hz),Phase drift (samples/s),Temperature (F),Humidity (%),Solar radiation (W/m^2),'
            'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)')

    @pytest.mark.parametrize('units, weather_cells', [
        ('imperial', ['68.0', '52.0', '300.0', '7.5', '12.0', '225']),
        ('metric', ['20.0', '52.0', '300.0', '12.0', '19.3', '225']),
    ])
    def test_the_next_row_in_a_file_this_version_started_fills_every_column(self, tmp_path, caplog, units,
                                                                            weather_cells):
        """The case that runs every minute: the second row is read against the header the first one wrote.

        The rows alone cannot show that the header was read.  A header the reader could
        not follow falls back to the full current row, which here looks the same, so the
        test also requires that the file drew no warning.
        """
        store = _make_store(tmp_path)
        store._config.weather.units = units
        store = CsvStore(store._config)
        first, second = _ts(2024, 1, 15, 10, 30), _ts(2024, 1, 15, 10, 31)
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            store.append(first, 15.0, -80.0, -95.0, 'full', _WEATHER, grid_frequency='60.010', phase_drift='-6.1')
            store.append(second, 17.5, -81.0, -96.0, 'partial', _WEATHER, grid_frequency='59.990',
                         phase_drift='-5.9')
        assert caplog.records == [], 'a file this version started should need no explaining'
        header, *rows = [line.split(',') for line in store.filename_for_date(first).read_text().splitlines()]
        assert rows == [
            [first.isoformat(), '15.00', '-80.00', '-95.00', 'full', '60.010', '-6.1', *weather_cells],
            [second.isoformat(), '17.50', '-81.00', '-96.00', 'partial', '59.990', '-5.9', *weather_cells],
        ]
        assert all(len(row) == len(header) for row in rows)

    def test_a_file_with_fewer_weather_columns_keeps_them_until_midnight(self, tmp_path):
        """What stops a new column from appearing in a file an older version started."""
        header, row = self._append_to(tmp_path, 'Temperature (F),Humidity (%)')
        assert row.split(',')[7:] == ['68.0', '52.0']
        assert len(row.split(',')) == len(header)

    def test_the_columns_come_in_the_order_the_header_gives(self, tmp_path):
        _, row = self._append_to(tmp_path, 'Wind bearing (deg),Temperature (F),Wind speed (MPH)')
        assert row.split(',')[7:] == ['225', '68.0', '7.5']

    def test_the_lowercase_watt_of_older_files_is_still_filled(self, tmp_path):
        _, row = self._append_to(tmp_path, 'Solar radiation (w/m^2)')
        assert row.split(',')[7:] == ['300.0']

    def test_each_column_takes_the_units_its_own_label_names(self, tmp_path):
        """A hand-edited header can mix systems, and each cell still matches its heading."""
        _, row = self._append_to(tmp_path, 'Temperature (C),Wind speed (MPH)', units='imperial')
        assert row.split(',')[7:] == ['20.0', '7.5']

    @pytest.mark.parametrize('heading', ['Dew point (F)', 'Temperature (K)', 'Temperature', 'Humidity (percent)'])
    def test_an_unknown_heading_or_label_gets_a_blank_cell(self, tmp_path, heading):
        """Writing a number under a heading this version cannot read would be the silent error the header prevents."""
        _, row = self._append_to(tmp_path, f'Temperature (F),{heading},Humidity (%)')
        assert row.split(',')[7:] == ['68.0', '', '52.0']

    def test_an_empty_heading_from_a_stray_comma_gets_a_blank_cell(self, tmp_path):
        header, row = self._append_to(tmp_path, 'Temperature (F),')
        assert row.split(',')[7:] == ['68.0', '']
        assert len(row.split(',')) == len(header)

    def test_an_unknown_heading_is_reported_once_per_file(self, tmp_path, caplog):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text(f'{_CORE},Temperature (F),Dew point (F)\n')
        with caplog.at_level(logging.WARNING, logger='buzz.csv_store'):
            for _ in range(3):
                store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        warnings = [r.getMessage() for r in caplog.records if 'does not recognize' in r.getMessage()]
        assert len(warnings) == 1
        assert 'does not recognize: Dew point (F).' in warnings[0]

    @pytest.mark.parametrize('weather_headings', [
        '', 'Temperature (F)', 'Dew point (F),Wind gust (km/h)',
        'Temperature (F),Humidity (%),Solar radiation (W/m^2),Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)',
    ])
    def test_every_row_has_as_many_cells_as_its_header(self, tmp_path, weather_headings):
        header, row = self._append_to(tmp_path, weather_headings)
        assert len(row.split(',')) == len(header)


class TestTheColumnTable:
    """The table that writes a new header and reads an old one back."""

    def test_every_column_can_fill_a_cell(self):
        """A weather column names its WeatherData field as text, so a misspelled field only fails here."""
        row = _Row(_ts(2024, 1, 15, 10, 30), 15.0, -80.0, -95.0, 'full', '60.010', '-6.1', _WEATHER)
        assert [column.cell(row, _WEATHER) for column in _COLUMNS][1:] == [
            '15.00', '-80.00', '-95.00', 'full', '60.010', '-6.1',
            '20.0', '52.0', '300.0', '12.0', '19.3', '225']

    @pytest.mark.parametrize('units', ['imperial', 'metric'])
    @pytest.mark.parametrize('pulse_rate', [120, 100])
    def test_every_heading_a_new_file_writes_is_read_back_as_itself(self, tmp_path, units, pulse_rate):
        """The drift pin between writing a header and reading it: each heading names its own column and units."""
        store = _make_store(tmp_path)
        store._config.weather.units = units
        store._config.audio.pulse_rate = pulse_rate
        store = CsvStore(store._config)
        written_in = WeatherUnits(units)
        for column in _COLUMNS:
            found, read_in = store._column_headed(column.heading(written_in, pulse_rate))
            assert found is column
            assert read_in is written_in

    def test_the_grid_frequency_heading_is_the_one_the_chart_reads(self):
        """read_grid_frequencies finds its column by this text, so a change to one must change both."""
        grid = next(column for column in _COLUMNS if column.name == 'Grid frequency')
        assert grid.heading(WeatherUnits.IMPERIAL, 120) == _GRID_FREQUENCY_HEADING

    def test_the_first_five_columns_are_the_ones_read_rows_reads(self):
        """read_rows reads timestamp, SNR, signal, noise and lock status by position."""
        assert [column.name for column in _COLUMNS[:5]] == [
            'ISO datetime', 'SNR', 'Signal', 'Noise floor', 'Signal Lock Status']


class TestHeadersFromOlderFiles:
    """Rows added to a file that an older version, or another setting, started."""

    def _append(self, tmp_path: Path, header: str, pulse_rate: int = 120) -> str:
        store = _make_store(tmp_path)
        store._config.audio.pulse_rate = pulse_rate
        store = CsvStore(store._config)
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text(header + '\n')
        return store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER, grid_frequency='60.010', phase_drift='-6.1')

    def test_a_file_from_before_the_grid_frequency_columns_stays_aligned(self, tmp_path):
        """Positional writing put drift figures under Temperature and Humidity on the day of that upgrade."""
        header = ('ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
                  'Temperature (F),Humidity (%),Solar radiation (w/m^2),'
                  'Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)')
        row = self._append(tmp_path, header).split(',')
        assert row[4:] == ['full', '68.0', '52.0', '300.0', '7.5', '12.0', '225']
        assert len(row) == len(header.split(','))

    def test_a_file_headed_for_another_pulse_rate_keeps_its_measurements(self, tmp_path, caplog):
        header = 'ISO datetime,120pps SNR,120pps signal (dBm),Noise floor (dBm),Signal Lock Status'
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            row = self._append(tmp_path, header, pulse_rate=100)
        assert row.split(',')[1:] == ['15.00', '-80.00', '-95.00', 'full']
        assert '[audio] pulse_rate is 100, but noise_data.2024-01-15.csv is headed for 120 pps.' in caplog.text

    def test_a_header_missing_a_measurement_gets_the_full_row_and_one_warning(self, tmp_path, caplog):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text('ISO datetime,120pps SNR,Signal Lock Status\n')
        with caplog.at_level(logging.WARNING, logger='buzz.csv_store'):
            rows = [store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER) for _ in range(3)]
        assert all(len(row.split(',')) == 13 for row in rows)
        warnings = [r.getMessage() for r in caplog.records if 'has no heading for' in r.getMessage()]
        assert len(warnings) == 1
        assert 'has no heading for Signal [120 pps] (dBm), Noise floor (dBm), so' in warnings[0]

    @pytest.mark.parametrize('heading', ['SNR', '120pps Temperature (F)'])
    def test_a_pulse_rate_only_goes_with_a_column_that_has_one(self, tmp_path, heading):
        store = _make_store(tmp_path)
        assert store._column_headed(heading) is None


# The whole header 2.1.0 wrote, from its csv_store.py, at each pulse rate.  Every release
# from 1.0.0 onward wrote the same core headings.
_RELEASED_HEADER = ('ISO datetime,{pps}pps SNR,{pps}pps signal (dBm),Noise floor (dBm),Signal Lock Status,'
                    'Grid frequency (Hz),Phase drift (samples/s),Temperature (F),Humidity (%),'
                    'Solar radiation (w/m^2),Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)')


class TestTheHeadingFormat:
    """Headings are written as `name [qualifier] (unit)`, and the formats they replaced still read."""

    @staticmethod
    def _column_named(name: str):
        return next(column for column in _COLUMNS if column.name == name)

    @pytest.mark.parametrize('heading, name, qualifier, unit', [
        ('SNR [120 pps] (dB)', 'SNR', '120 pps', 'dB'),
        ('Noise floor (dBm)', 'Noise floor', None, 'dBm'),
        ('ISO datetime', 'ISO datetime', None, None),
        ('Rain [since midnight] (in)', 'Rain', 'since midnight', 'in'),
    ])
    def test_the_format_splits_a_heading_into_its_three_parts(self, heading, name, qualifier, unit):
        parsed = _HEADING_FORMAT.fullmatch(heading)
        assert (parsed['name'], parsed['qualifier'], parsed['unit']) == (name, qualifier, unit)

    @pytest.mark.parametrize('heading', ['SNR (dB) [120 pps]', 'SNR [120 pps] [fast] (dB)', 'SNR ((dB))', ''])
    def test_the_format_takes_one_qualifier_then_one_unit_in_that_order(self, heading):
        assert _HEADING_FORMAT.fullmatch(heading) is None

    @pytest.mark.parametrize('heading, rate', [
        ('SNR [120 pps] (dB)', '120'), ('SNR [100 pps] (dB)', '100'),
        ('120pps SNR', '120'), ('100pps SNR', '100'),
    ])
    def test_snr_reads_in_the_heading_format_and_the_released_one(self, heading, rate):
        match = self._column_named('SNR').match(heading)
        assert match is not None and match.rate == rate

    @pytest.mark.parametrize('heading, rate', [
        ('Signal [120 pps] (dBm)', '120'), ('Signal [100 pps] (dBm)', '100'),
        ('120pps signal (dBm)', '120'), ('100pps signal (dBm)', '100'),
    ])
    def test_signal_reads_in_the_heading_format_and_the_released_one(self, heading, rate):
        match = self._column_named('Signal').match(heading)
        assert match is not None and match.rate == rate

    @pytest.mark.parametrize('heading', [
        'SNR',                      # no pulse rate at all
        'SNR (dB)',                 # the heading format needs the qualifier
        'SNR [120 pps]',            # and the unit, since no version wrote SNR that way
        'SNR (120 pps)',            # a rate in the unit's place, which never shipped
        'SNR [fast] (dB)',          # a qualifier that is not a pulse rate
        'SNR [120 pps] (dBm)',      # a unit SNR does not have
        '120pps SNR [120 pps] (dB)',  # both formats at once
        '120pps SNR (dB)',          # the released format never carried a unit
    ])
    def test_snr_is_not_found_in_a_heading_no_version_wrote(self, heading):
        assert self._column_named('SNR').match(heading) is None

    @pytest.mark.parametrize('heading', [
        'signal [120 pps] (dBm)',   # the old name in the new format
        '120pps Signal (dBm)',      # the new name in the old format
        'SIGNAL [120 pps] (dBm)',   # a capitalization nobody wrote
        'Signal [120 pps] (dB)',    # the signal is a level in dBm, not a ratio
        '120pps signal',            # the released format always carried dBm
    ])
    def test_signal_is_not_found_in_a_heading_no_version_wrote(self, heading):
        assert self._column_named('Signal').match(heading) is None

    @pytest.mark.parametrize('heading', ['Noise floor [120 pps] (dBm)', 'Temperature [120 pps] (F)',
                                         'Humidity [outdoor] (%)'])
    def test_a_column_that_takes_no_qualifier_refuses_one(self, tmp_path, heading):
        assert _make_store(tmp_path)._column_headed(heading) is None

    @pytest.mark.parametrize('pulse_rate', [120, 100])
    def test_a_new_file_writes_the_heading_format_and_no_deprecated_one(self, tmp_path, pulse_rate):
        store = _make_store(tmp_path)
        store._config.audio.pulse_rate = pulse_rate
        store = CsvStore(store._config)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        header = store.filename_for_date(now).read_text().splitlines()[0].split(',')
        assert header[1:4] == [f'SNR [{pulse_rate} pps] (dB)', f'Signal [{pulse_rate} pps] (dBm)', 'Noise floor (dBm)']
        for heading in header:
            assert _HEADING_FORMAT.fullmatch(heading), f'{heading!r} is not in the heading format'
            assert not any(old.fullmatch(heading) for column in _COLUMNS for old in column.deprecated_formats)

    @pytest.mark.parametrize('pulse_rate', [120, 100])
    def test_a_file_started_by_the_last_release_is_filled_in_every_column(self, tmp_path, caplog, pulse_rate):
        """The upgrade day: a file 2.1.0 started, appended to by this version at the same rate.

        2.1.0's columns are the current ones in the current order, so a header the reader
        could not follow would fall back to a row that looks the same.  The test also
        requires that the file drew no warning, which is what shows the header was read.
        """
        store = _make_store(tmp_path)
        store._config.audio.pulse_rate = pulse_rate
        store = CsvStore(store._config)
        now = _ts(2024, 1, 15, 10, 30)
        header = _RELEASED_HEADER.format(pps=pulse_rate)
        store.filename_for_date(now).write_text(header + '\n')
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            row = store.append(now, 17.5, -80.0, -95.0, 'full', _WEATHER, grid_frequency='60.010',
                               phase_drift='-6.1')
        assert caplog.records == [], 'a file the last release started should need no explaining'
        assert row.split(',')[1:] == ['17.50', '-80.00', '-95.00', 'full', '60.010', '-6.1',
                                      '68.0', '52.0', '300.0', '7.5', '12.0', '225']
        assert len(row.split(',')) == len(header.split(','))

    @pytest.mark.parametrize('header', [
        _RELEASED_HEADER.format(pps=120),
        'ISO datetime,SNR [120 pps] (dB),Signal [120 pps] (dBm),Noise floor (dBm),Signal Lock Status',
    ])
    def test_a_file_headed_for_another_rate_is_noted_in_either_format(self, tmp_path, caplog, header):
        store = _make_store(tmp_path)
        store._config.audio.pulse_rate = 100
        store = CsvStore(store._config)
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text(header + '\n')
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            row = store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        assert row.split(',')[1:5] == ['15.00', '-80.00', '-95.00', 'full']
        assert '[audio] pulse_rate is 100, but noise_data.2024-01-15.csv is headed for 120 pps.' in caplog.text

    def test_a_file_headed_for_the_configured_rate_gets_no_note(self, tmp_path, caplog):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.filename_for_date(now).write_text(_RELEASED_HEADER.format(pps=120) + '\n')
        with caplog.at_level(logging.INFO, logger='buzz.csv_store'):
            store.append(now, 15.0, -80.0, -95.0, 'full', _WEATHER)
        assert 'pulse_rate' not in caplog.text

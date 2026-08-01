"""Tests for CsvStore: filename generation, row append, time bucketing, and range aggregation."""

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from buzz.config import BuzzConfig
from buzz.csv_store import CsvRow, CsvStore

_TZ = ZoneInfo('America/Los_Angeles')


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
        store.append(now, 15.0, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225, **kwargs)
        lines = store.filename_for_date(now).read_text().splitlines()
        return lines[1].split(',')

    def test_header_names_both_columns(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', '', '', '', '', '', '')
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

    def test_default_is_blank_not_zero(self, tmp_path):
        """A minute with no lock has nothing to report, and 0.000 Hz would be a lie."""
        fields = self._row(tmp_path)
        assert fields[5] == '' and fields[6] == ''

    def test_row_still_parses_with_the_new_columns(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'partial', 68.0, 52.0, 300.0, 7.5, 12.0, 225,
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


class TestAppend:
    def test_creates_file_with_headers_on_first_call(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        content = store.filename_for_date(now).read_text()
        assert 'ISO datetime' in content
        assert '120pps SNR' in content

    def test_no_headers_on_subsequent_calls(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        store.append(now, 16.0, -81.0, -96.0, 'full', 69.0, 53.0, 310.0, 8.0, 13.0, 230)
        lines = store.filename_for_date(now).read_text().strip().split('\n')
        header_count = sum(1 for l in lines if 'ISO datetime' in l)
        assert header_count == 1

    def test_returns_csv_string_without_newline(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        assert '\n' not in result

    def test_csv_string_contains_snr(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 17.5, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        assert '17.50' in result

    def test_multiple_appends_produce_multiple_rows(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        for i in range(3):
            store.append(now, 15.0 + i, -80.0, -95.0, 'full', '', '', '', '', '', '')
        lines = [l for l in store.filename_for_date(now).read_text().strip().split('\n')
                 if 'ISO datetime' not in l]
        assert len(lines) == 3

    def test_accepts_string_weather_values(self, tmp_path):
        """Blank strings for every weather field - what a collection with weather
        disabled actually passes - must not raise trying to format them as numbers,
        and must come back as the blank fields they were, not dropped or defaulted."""
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'full', '', '', '', '', '', '')
        assert result == f'{now.isoformat()},15.00,-80.00,-95.00,full,,,,,,,,'

    def test_lock_status_in_header(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        content = store.filename_for_date(now).read_text()
        assert 'Signal Lock Status' in content

    def test_lock_status_written_to_row(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 15.0, -80.0, -95.0, 'partial', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        assert 'partial' in result

    @pytest.mark.parametrize('status', ['full', 'partial', 'none'])
    def test_all_lock_statuses_accepted(self, tmp_path, status):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        result = store.append(now, 0.0, -90.0, -90.0, status, '', '', '', '', '', '')
        assert status in result


class TestReadRows:
    def test_round_trips_appended_row(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'partial', 68.0, 52.0, 300.0, 7.5, 12.0, 225)
        rows = store.read_rows(store.filename_for_date(now))
        assert rows == [CsvRow(timestamp=now, snr=15.0, signal=-80.0, noise=-95.0,
                               lock_status='partial')]

    def test_header_row_skipped(self, tmp_path):
        store = _make_store(tmp_path)
        now = _ts(2024, 1, 15, 10, 30)
        store.append(now, 15.0, -80.0, -95.0, 'full', '', '', '', '', '', '')
        rows = store.read_rows(store.filename_for_date(now))
        assert len(rows) == 1

    def test_timestamp_converted_to_station_timezone(self, tmp_path):
        store = _make_store(tmp_path)
        utc_now = datetime(2024, 1, 15, 18, 30, tzinfo=ZoneInfo('UTC'))
        store.append(utc_now, 15.0, -80.0, -95.0, 'full', '', '', '', '', '', '')
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
        store.append(when, 20.0, -80.0, -95.0, 'full', 72, 50, 300, 5, 8, 180)

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

"""Tests for weather data clients, and for the units they convert between."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from buzz.weather import (
    EMPTY_WEATHER, CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient, WeatherData, WeatherUnits,
)

_RESOURCES = Path(__file__).parent / 'resources'

# The operator's station keeps Los Angeles time, and so does the computer CumulusMX runs on.
_STATION_ZONE = 'America/Los_Angeles'

# A moment to stand in for a weather timestamp where the test does not care which.
_WHEN = datetime(2026, 9, 27, 21, 0, tzinfo=UTC)


def _weather(temperature=10.0, humidity=52.0, solar_radiation=300.0, wind_speed=16.09344,
             wind_gust=32.18688, wind_bearing=225, rain_since_midnight=25.4, timestamp=_WHEN) -> WeatherData:
    return WeatherData(temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing,
                       rain_since_midnight, timestamp)


class TestWeatherData:
    def test_the_six_csv_fields_come_first(self):
        """The six columns the CSV writes today come first, in the CSV's order."""
        assert WeatherData._fields == ('temperature', 'humidity', 'solar_radiation', 'wind_speed',
                                       'wind_gust', 'wind_bearing', 'rain_since_midnight', 'timestamp')

    def test_named_field_access(self):
        data = _weather(temperature=68.2, wind_bearing=225)
        assert data.temperature == 68.2
        assert data.wind_bearing == 225
        assert data.timestamp == _WHEN

    def test_empty_weather_has_blank_values_and_no_timestamp(self):
        """A blank value writes an empty CSV cell, and a missing time is None rather than a fake one."""
        assert all(value == '' for value in EMPTY_WEATHER[:-1])
        assert EMPTY_WEATHER.timestamp is None


class TestNullWeatherClient:
    def test_fetch_returns_empty_weather(self):
        assert NullWeatherClient().fetch() == EMPTY_WEATHER

    def test_fetch_type_matches_weather_data(self):
        assert isinstance(NullWeatherClient().fetch(), WeatherData)


def _mock_urlopen(reply: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = reply
    return mock_resp


class TestWeatherUnits:
    """The conversions from the units every client returns, degrees C, km/h and mm."""

    def test_imperial_converts_the_fixed_points_of_the_scales(self):
        # Water freezes at 32 F and boils at 212 F, figures that owe nothing to the code.
        assert WeatherUnits.IMPERIAL.temperature_from_celsius(0.0) == pytest.approx(32.0)
        assert WeatherUnits.IMPERIAL.temperature_from_celsius(100.0) == pytest.approx(212.0)

    def test_imperial_wind_uses_the_statute_mile(self):
        assert WeatherUnits.IMPERIAL.wind_speed_from_kmh(1.609344) == pytest.approx(1.0)

    def test_imperial_rain_uses_the_inch(self):
        assert WeatherUnits.IMPERIAL.rain_from_mm(25.4) == pytest.approx(1.0)

    def test_metric_passes_everything_through(self):
        assert WeatherUnits.METRIC.temperature_from_celsius(21.3) == 21.3
        assert WeatherUnits.METRIC.wind_speed_from_kmh(17.0) == 17.0
        assert WeatherUnits.METRIC.rain_from_mm(3.2) == 3.2

    def test_the_labels_name_each_system(self):
        imperial, metric = WeatherUnits.IMPERIAL, WeatherUnits.METRIC
        assert (imperial.temperature_label, imperial.wind_speed_label, imperial.rain_label) == ('F', 'MPH', 'in')
        assert (metric.temperature_label, metric.wind_speed_label, metric.rain_label) == ('C', 'km/h', 'mm')

    def test_a_known_setting_is_read_as_itself(self):
        assert WeatherUnits.from_setting('metric') is WeatherUnits.METRIC

    def test_an_unknown_setting_falls_back_to_imperial_and_says_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            assert WeatherUnits.from_setting('kelvin') is WeatherUnits.IMPERIAL
        assert "[weather] units is 'kelvin'" in caplog.text


class TestWeatherDataInUnits:
    def test_imperial_converts_temperature_both_winds_and_rain(self):
        assert _weather().in_units(WeatherUnits.IMPERIAL) == (50.0, 52.0, 300.0, 10.0, 20.0, 225, 1.0, _WHEN)

    def test_a_converted_figure_is_rounded_to_tenths(self):
        """21 C is 69.8 F, which float arithmetic gives as 69.80000000000001."""
        assert _weather(temperature=21.0).in_units(WeatherUnits.IMPERIAL).temperature == 69.8

    def test_one_tip_of_a_gauge_survives_the_round_trip(self):
        """A tipping bucket counts 0.01 inch at a time.  Tenths would round a whole tip away."""
        one_tip_in_mm = 0.01 * 25.4
        assert _weather(rain_since_midnight=one_tip_in_mm).in_units(WeatherUnits.IMPERIAL).rain_since_midnight == 0.01

    def test_blank_fields_stay_blank(self):
        assert EMPTY_WEATHER.in_units(WeatherUnits.IMPERIAL) == EMPTY_WEATHER

    def test_metric_leaves_the_figures_as_they_were(self):
        data = _weather(temperature=21.0, wind_speed=12.0, wind_gust=19.0, rain_since_midnight=3.2)
        assert data.in_units(WeatherUnits.METRIC) == data


class TestCumulusMXWeatherClient:
    """A CumulusMX station serves figures in its own units, and says which they are."""

    # The operator's station answered the client's template with this on 2026-09-27.  The
    # two clock readings came from the same station on 2026-09-28, when its UTC clock read
    # 02:29:08 and its own clock 19:29:08.  CumulusMX returns every value as a string.
    _REPLY = {'temp': '64.0', 'hum': '70', 'SolarRad': '445', 'wspeed': '9', 'wgust': '15',
              'avgbearing': '319', 'rmidnight': '0.00', 'tempunitnodeg': 'F', 'windunit': 'mph',
              'rainunit': 'in', 'LastDataReadT': '1790549853',
              'timeUnix': '1790648948', 'timehhmmss': '19:29:08'}

    def _fetch(self, **changes: str) -> WeatherData:
        client = CumulusMXWeatherClient('http://fake', _STATION_ZONE)
        reply = json.dumps({**self._REPLY, **changes}).encode()
        with patch('buzz.weather.urllib.request.urlopen', return_value=_mock_urlopen(reply)):
            return client.fetch()

    _FULL = 'http://cumulusmx.local:8998/api/tags/process.txt'

    _ADDRESSES = ['http://cumulusmx.local:8998/', 'http://cumulusmx.local:8998', 'cumulusmx.local:8998']
    _OLDER_FORMS = [
        'http://cumulusmx.local:8998/api/tags/process.json',
        'http://cumulusmx.local:8998/api/tags/process.json?temp&hum&SolarRad&wspeed&wgust&avgbearing',
        'http://cumulusmx.local:8998/api/tags/process.json?rc&temp&hum&SolarRad&wspeed&wgust&avgbearing'
        '&tempunitnodeg&windunit',
        'http://cumulusmx.local:8998/somewhere/else.json?temp',
    ]

    @pytest.mark.parametrize('setting', _ADDRESSES + _OLDER_FORMS)
    def test_every_form_of_the_setting_builds_the_same_url(self, setting):
        """The station's address alone, and every longer form an older config holds."""
        assert CumulusMXWeatherClient(setting, _STATION_ZONE)._url == self._FULL

    def test_a_https_station_keeps_its_scheme(self):
        assert CumulusMXWeatherClient('https://wx.example.net/', _STATION_ZONE)._url.startswith('https://wx.example.net/api/')

    @pytest.mark.parametrize('setting', _OLDER_FORMS)
    def test_anything_after_the_port_is_dropped_with_a_warning(self, setting, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            CumulusMXWeatherClient(setting, _STATION_ZONE)
        assert 'set url to http://cumulusmx.local:8998/.' in caplog.text

    @pytest.mark.parametrize('setting', _ADDRESSES)
    def test_no_warning_for_the_address_alone(self, setting, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            CumulusMXWeatherClient(setting, _STATION_ZONE)
        assert caplog.records == []

    def test_the_template_is_posted_to_the_text_api(self):
        """Only the text API passes a tag its parameters, and `format=Unix` needs that."""
        client = CumulusMXWeatherClient('http://cumulusmx.local:8998/', _STATION_ZONE)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=_mock_urlopen(json.dumps(self._REPLY).encode())) as urlopen:
            client.fetch()
        request = urlopen.call_args.args[0]
        assert (request.get_method(), request.full_url) == ('POST', self._FULL)
        assert json.loads(request.data) == json.loads(CumulusMXWeatherClient._TEMPLATE)

    def test_the_template_asks_for_every_field_the_reply_needs(self):
        """The template's keys are the reply's keys, so a tag missing from one is missing from both."""
        assert set(json.loads(CumulusMXWeatherClient._TEMPLATE)) == set(self._REPLY)

    def test_every_value_tag_removes_commas(self):
        """Without `rc=y`, a station in a decimal-comma locale serves "20,5", and every fetch fails."""
        template = json.loads(CumulusMXWeatherClient._TEMPLATE)
        unit_tags = {'tempunitnodeg', 'windunit', 'rainunit', 'LastDataReadT', 'timeUnix', 'timehhmmss'}
        missing = [key for key, tag in template.items() if key not in unit_tags and 'rc=y' not in tag]
        assert missing == [], f'these tags would pass a locale comma through to float(): {missing}'

    def test_the_last_data_time_is_asked_for_in_epoch_seconds(self):
        """The default format follows the host's locale and names no timezone."""
        assert json.loads(CumulusMXWeatherClient._TEMPLATE)['LastDataReadT'] == '<#LastDataReadT format=Unix>'

    def test_the_stations_real_reply_is_read(self):
        result = self._fetch()
        assert result.temperature == pytest.approx((64.0 - 32) * 5 / 9)
        assert (result.wind_speed, result.wind_gust) == (pytest.approx(9 * 1.609344), pytest.approx(15 * 1.609344))
        assert (result.humidity, result.solar_radiation, result.wind_bearing) == ('70', '445', '319')
        assert result.rain_since_midnight == 0.0

    def test_the_weather_timestamp_is_when_cumulusmx_last_heard_from_the_station(self):
        # GNU `date -u -d @1790549853` gives 2026-09-27 22:57:33 UTC.  The request that
        # captured this reply went out at 1790549855, two seconds later.
        assert self._fetch().timestamp == datetime(2026, 9, 27, 22, 57, 33, tzinfo=UTC)

    def test_no_data_yet_gives_no_timestamp_and_keeps_the_rest(self):
        """Before CumulusMX has heard from the station, `LastDataReadT` reads "----"."""
        result = self._fetch(LastDataReadT='----')
        assert result.timestamp is None
        assert result.humidity == '70'

    def test_a_fahrenheit_station_is_converted_to_celsius(self):
        assert self._fetch(temp='50.0', tempunitnodeg='F').temperature == pytest.approx(10.0)

    def test_a_celsius_station_is_passed_through(self):
        assert self._fetch(temp='10.0', tempunitnodeg='C').temperature == pytest.approx(10.0)

    @pytest.mark.parametrize('unit, kmh', [
        ('km/h', 10.0), ('mph', 16.09344), ('m/s', 36.0), ('kts', 18.52),
    ])
    def test_each_wind_unit_is_converted_to_kmh(self, unit, kmh):
        result = self._fetch(wspeed='10', wgust='20', windunit=unit)
        assert (result.wind_speed, result.wind_gust) == (pytest.approx(kmh), pytest.approx(2 * kmh))

    @pytest.mark.parametrize('unit, mm', [('in', 25.4), ('mm', 1.0)])
    def test_each_rain_unit_is_converted_to_mm(self, unit, mm):
        assert self._fetch(rmidnight='1.00', rainunit=unit).rain_since_midnight == pytest.approx(mm)

    def test_a_missing_temperature_unit_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match='temperature unit as None'):
            self._fetch(tempunitnodeg=None)

    def test_an_unknown_wind_unit_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match="wind speed unit as 'Bft'"):
            self._fetch(windunit='Bft')

    def test_an_unknown_rain_unit_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match="rain unit as 'cm'.*the rainunit webtag"):
            self._fetch(rainunit='cm')


class TestAMissingReadingFromCumulusMX:
    """CumulusMX 5 answers "-" for a tag whose sensor has no reading."""

    def _fetch(self, **changes: str) -> WeatherData:
        client = CumulusMXWeatherClient('http://fake', _STATION_ZONE)
        reply = json.dumps({**TestCumulusMXWeatherClient._REPLY, **changes}).encode()
        with patch('buzz.weather.urllib.request.urlopen', return_value=_mock_urlopen(reply)):
            return client.fetch()

    def test_a_station_without_a_solar_sensor_gets_a_blank_rather_than_a_dash(self):
        assert self._fetch(SolarRad='-').solar_radiation == ''

    @pytest.mark.parametrize('tag, field', [
        ('temp', 'temperature'), ('wspeed', 'wind_speed'), ('wgust', 'wind_gust'),
        ('rmidnight', 'rain_since_midnight'), ('hum', 'humidity'), ('avgbearing', 'wind_bearing'),
    ])
    def test_any_missing_reading_is_blank_and_the_rest_still_arrive(self, tag, field):
        """A figure that is converted would otherwise fail float() and blank the whole row."""
        result = self._fetch(**{tag: '-'})
        assert getattr(result, field) == ''
        assert result.solar_radiation == '445'


class TestTheCumulusMXHostClock:
    """CumulusMX starts its rain total at its own midnight, which has to be the station's."""

    def _first_fetch(self, zone: str, utc_epoch: int, host_clock: str, caplog, fetches: int = 1) -> str:
        client = CumulusMXWeatherClient('http://fake', zone)
        reply = json.dumps({**TestCumulusMXWeatherClient._REPLY, 'timeUnix': str(utc_epoch),
                            'timehhmmss': host_clock}).encode()
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            for _ in range(fetches):
                with patch('buzz.weather.urllib.request.urlopen', return_value=_mock_urlopen(reply)):
                    client.fetch()
        return caplog.text

    # 2026-09-28 19:29:08 in Los Angeles is 02:29:08 UTC the next day, epoch 1790648948.
    _UTC_EPOCH = 1790648948

    def test_the_operators_own_reading_raises_no_warning(self, caplog):
        """The reading straddles UTC midnight, so the difference has to wrap round the day."""
        assert self._first_fetch(_STATION_ZONE, self._UTC_EPOCH, '19:29:08', caplog) == ''

    def test_a_second_that_ticks_between_the_two_readings_does_not_count(self, caplog):
        assert self._first_fetch(_STATION_ZONE, self._UTC_EPOCH, '19:29:09', caplog) == ''

    def test_a_host_on_another_zones_time_is_reported_with_the_hour_the_rain_restarts(self, caplog):
        """New York is UTC-04:00 on that date, so Los Angeles midnight falls at 03:00 there."""
        text = self._first_fetch('America/New_York', self._UTC_EPOCH, '19:29:08', caplog)
        assert ('CumulusMX runs on a clock at UTC-07:00, and [station] timezone America/New_York is at '
                'UTC-04:00.') in text
        assert 'restarts at 03:00 station time.' in text

    def test_a_zone_on_a_half_hour_is_told_apart(self, caplog):
        """India is UTC+05:30, and 07:00 UTC, the host's midnight, is 12:30 there."""
        text = self._first_fetch('Asia/Kolkata', self._UTC_EPOCH, '19:29:08', caplog)
        assert 'is at UTC+05:30.' in text
        assert 'restarts at 12:30 station time.' in text

    def test_a_host_on_utc_is_caught(self, caplog):
        text = self._first_fetch(_STATION_ZONE, self._UTC_EPOCH, '02:29:08', caplog)
        assert 'runs on a clock at UTC+00:00' in text
        assert 'restarts at 17:00 station time.' in text

    def test_the_warning_appears_once_however_many_fetches_follow(self, caplog):
        text = self._first_fetch('America/New_York', self._UTC_EPOCH, '19:29:08', caplog, fetches=3)
        assert text.count('CumulusMX runs on a clock') == 1


def _quarters_from_midnight(zone: str, year: int, month: int, day: int, count: int) -> list[int]:
    """Epoch times every 15 minutes from a local midnight, as Open-Meteo labels its series."""
    midnight = int(datetime(year, month, day, tzinfo=ZoneInfo(zone)).timestamp())
    return [midnight + 900 * step for step in range(count)]


class TestOpenMeteoWeatherClient:
    """Open-Meteo's current conditions, and its rain since the station's midnight."""

    # A real reply for Bangkok, captured on 2026-09-27 from the URL the client builds,
    # while it was raining there.
    _BANGKOK = (_RESOURCES / 'open_meteo_bangkok.json').read_bytes()

    def _fetch(self, reply: bytes, zone: str = 'Asia/Bangkok') -> WeatherData:
        client = OpenMeteoWeatherClient(13.75, 100.5, zone)
        with patch('buzz.weather.urllib.request.urlopen', return_value=_mock_urlopen(reply)):
            return client.fetch()

    def _reply(self, times: list[int], amounts: list[float | None], interval_end: int) -> bytes:
        payload = json.loads(self._BANGKOK)
        payload['current']['time'] = interval_end
        payload['minutely_15'] = {'time': times, 'precipitation': amounts}
        return json.dumps(payload).encode()

    def test_url_contains_latitude_longitude(self):
        client = OpenMeteoWeatherClient(37.8, -122.4, 'America/Los_Angeles')
        assert 'latitude=37.8&' in client._url
        assert 'longitude=-122.4&' in client._url

    def test_url_asks_for_the_units_weather_data_holds(self):
        url = OpenMeteoWeatherClient(37.8, -122.4, 'America/Los_Angeles')._url
        for unit in ('temperature_unit=celsius', 'wind_speed_unit=kmh', 'precipitation_unit=mm'):
            assert f'&{unit}&' in url

    def test_url_asks_for_the_stations_day_in_15_minute_steps_as_epoch_seconds(self):
        url = OpenMeteoWeatherClient(37.8, -122.4, 'America/Los_Angeles')._url
        for part in ('minutely_15=precipitation', 'timezone=America%2FLos_Angeles', 'forecast_days=1',
                     'timeformat=unixtime'):
            assert f'&{part}' in url

    def test_the_real_reply_is_read(self):
        result = self._fetch(self._BANGKOK)
        assert (result.temperature, result.humidity, result.solar_radiation) == (24.7, 95, 0.0)
        assert (result.wind_speed, result.wind_gust, result.wind_bearing) == (17.1, 42.1, 165)

    def test_the_weather_timestamp_is_the_end_of_the_current_interval(self):
        # GNU `date -u -d @1790549100` gives 22:45 UTC, which is 05:45 in Bangkok.
        assert self._fetch(self._BANGKOK).timestamp == datetime(2026, 9, 27, 22, 45, tzinfo=UTC)

    def test_the_real_replys_rain_is_its_quarters_after_midnight_through_05_45(self):
        """The series starts at Bangkok's midnight, so 00:15 to 05:45 are positions 1 to 23."""
        amounts = json.loads(self._BANGKOK)['minutely_15']['precipitation']
        assert self._fetch(self._BANGKOK).rain_since_midnight == pytest.approx(sum(amounts[1:24]))

    def test_the_quarter_labeled_midnight_fell_the_day_before(self):
        times = _quarters_from_midnight('Asia/Bangkok', 2026, 9, 28, 96)
        amounts = [100.0] + [1.0] * 95
        assert self._fetch(self._reply(times, amounts, times[4])).rain_since_midnight == pytest.approx(4.0)

    def test_quarters_after_the_current_one_are_forecast_and_left_out(self):
        times = _quarters_from_midnight('Asia/Bangkok', 2026, 9, 28, 96)
        amounts = [0.0] + [1.0] * 4 + [1000.0] * 91
        assert self._fetch(self._reply(times, amounts, times[4])).rain_since_midnight == pytest.approx(4.0)

    def test_at_midnight_nothing_has_fallen_yet(self):
        times = _quarters_from_midnight('Asia/Bangkok', 2026, 9, 28, 96)
        assert self._fetch(self._reply(times, [5.0] * 96, times[0])).rain_since_midnight == 0.0

    def test_a_gap_in_the_quarters_so_far_leaves_the_total_blank(self):
        """A total with a quarter missing would be short, and nothing in the CSV would say so."""
        times = _quarters_from_midnight('Asia/Bangkok', 2026, 9, 28, 96)
        amounts = [0.0, 1.0, None, 1.0, 1.0] + [0.0] * 91
        assert self._fetch(self._reply(times, amounts, times[4])).rain_since_midnight == ''

    def test_a_gap_in_the_forecast_does_not_matter(self):
        times = _quarters_from_midnight('Asia/Bangkok', 2026, 9, 28, 96)
        amounts = [0.0] + [1.0] * 4 + [None] * 91
        assert self._fetch(self._reply(times, amounts, times[4])).rain_since_midnight == pytest.approx(4.0)

    @pytest.mark.parametrize('hours_after_midnight, quarters', [(2.5, 10), (4.0, 16)])
    def test_the_night_clocks_fall_back_counts_the_repeated_hour_twice(self, hours_after_midnight, quarters):
        """On 2026-11-01 Los Angeles repeats 01:00 to 02:00, so its day runs 25 hours.

        2.5 hours after midnight is 01:30 for the second time, and 4 hours after is 03:00.
        A sum by local clock label would count 6 and 12 quarters, missing the repeated hour.
        """
        times = _quarters_from_midnight('America/Los_Angeles', 2026, 11, 1, 100)
        interval_end = times[0] + int(hours_after_midnight * 3600)
        reply = self._reply(times, [0.0] + [1.0] * 99, interval_end)
        assert self._fetch(reply, 'America/Los_Angeles').rain_since_midnight == pytest.approx(quarters)

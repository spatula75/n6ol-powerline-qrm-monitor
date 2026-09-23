"""Tests for weather data clients, and for the units they convert between."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from buzz.weather import (
    EMPTY_WEATHER, CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient, WeatherData, WeatherUnits,
)


class TestWeatherData:
    def test_fields_in_csv_order(self):
        assert WeatherData._fields == ('temperature', 'humidity', 'solar_radiation',
                                       'wind_speed', 'wind_gust', 'wind_bearing')

    def test_named_field_access(self):
        data = WeatherData(68.2, 52.0, 320.0, 7.5, 12.0, 225)
        assert data.temperature == 68.2
        assert data.wind_bearing == 225

    def test_empty_weather_is_all_blank(self):
        assert all(v == '' for v in EMPTY_WEATHER)


class TestNullWeatherClient:
    def test_fetch_returns_six_tuple(self):
        result = NullWeatherClient().fetch()
        assert len(result) == 6

    def test_fetch_returns_empty_weather(self):
        assert NullWeatherClient().fetch() == EMPTY_WEATHER

    def test_fetch_type_matches_weather_data(self):
        result = NullWeatherClient().fetch()
        assert isinstance(result, WeatherData)


def _mock_urlopen(data: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps(data).encode()
    return mock_resp


class TestWeatherUnits:
    """The conversions from the units every client returns, degrees C and km/h."""

    def test_imperial_converts_the_fixed_points_of_the_scales(self):
        # Water freezes at 32 F and boils at 212 F, figures that owe nothing to the code.
        assert WeatherUnits.IMPERIAL.temperature_from_celsius(0.0) == pytest.approx(32.0)
        assert WeatherUnits.IMPERIAL.temperature_from_celsius(100.0) == pytest.approx(212.0)

    def test_imperial_wind_uses_the_statute_mile(self):
        assert WeatherUnits.IMPERIAL.wind_speed_from_kmh(1.609344) == pytest.approx(1.0)

    def test_metric_passes_both_through(self):
        assert WeatherUnits.METRIC.temperature_from_celsius(21.3) == 21.3
        assert WeatherUnits.METRIC.wind_speed_from_kmh(17.0) == 17.0

    def test_the_labels_name_each_system(self):
        assert (WeatherUnits.IMPERIAL.temperature_label, WeatherUnits.IMPERIAL.wind_speed_label) == ('F', 'MPH')
        assert (WeatherUnits.METRIC.temperature_label, WeatherUnits.METRIC.wind_speed_label) == ('C', 'km/h')

    def test_a_known_setting_is_read_as_itself(self):
        assert WeatherUnits.from_setting('metric') is WeatherUnits.METRIC

    def test_an_unknown_setting_falls_back_to_imperial_and_says_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            assert WeatherUnits.from_setting('kelvin') is WeatherUnits.IMPERIAL
        assert "[weather] units is 'kelvin'" in caplog.text


class TestWeatherDataInUnits:
    def test_imperial_converts_temperature_and_both_winds(self):
        data = WeatherData(10.0, 52.0, 300.0, 16.09344, 32.18688, 225)
        assert data.in_units(WeatherUnits.IMPERIAL) == (50.0, 52.0, 300.0, 10.0, 20.0, 225)

    def test_a_converted_figure_is_rounded_to_tenths(self):
        """21 C is 69.8 F, which float arithmetic gives as 69.80000000000001."""
        assert WeatherData(21.0, '', '', '', '', '').in_units(WeatherUnits.IMPERIAL).temperature == 69.8

    def test_blank_fields_stay_blank(self):
        assert EMPTY_WEATHER.in_units(WeatherUnits.IMPERIAL) == EMPTY_WEATHER

    def test_metric_leaves_the_figures_as_they_were(self):
        data = WeatherData(21.0, 52.0, 300.0, 12.0, 19.0, 225)
        assert data.in_units(WeatherUnits.METRIC) == data


class TestCumulusMXWeatherClient:
    """A CumulusMX station serves figures in its own units, and says which they are."""

    # The webtag API returns each value as a string.
    _READING = {'hum': '52', 'SolarRad': '320', 'avgbearing': '225'}

    def _fetch(self, **fields: str) -> WeatherData:
        client = CumulusMXWeatherClient('http://fake')
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=_mock_urlopen({**self._READING, **fields})):
            return client.fetch()

    _FULL = ('http://cumulusmx.local:8998/api/tags/process.json'
             '?temp&hum&SolarRad&wspeed&wgust&avgbearing&tempunitnodeg&windunit')

    _ADDRESSES = ['http://cumulusmx.local:8998/', 'http://cumulusmx.local:8998', 'cumulusmx.local:8998']
    _OLDER_FORMS = [
        'http://cumulusmx.local:8998/api/tags/process.json',
        'http://cumulusmx.local:8998/api/tags/process.json?temp&hum&SolarRad&wspeed&wgust&avgbearing',
        'http://cumulusmx.local:8998/somewhere/else.json?temp',
        _FULL,
    ]

    @pytest.mark.parametrize('setting', _ADDRESSES + _OLDER_FORMS)
    def test_every_form_of_the_setting_builds_the_same_url(self, setting):
        """The station's address alone, and every longer form an older config holds."""
        assert CumulusMXWeatherClient(setting)._url == self._FULL

    def test_a_https_station_keeps_its_scheme(self):
        assert CumulusMXWeatherClient('https://wx.example.net/')._url.startswith('https://wx.example.net/api/')

    @pytest.mark.parametrize('setting', _OLDER_FORMS)
    def test_anything_after_the_port_is_dropped_with_a_warning(self, setting, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            CumulusMXWeatherClient(setting)
        assert 'set url to http://cumulusmx.local:8998/.' in caplog.text

    @pytest.mark.parametrize('setting', _ADDRESSES)
    def test_no_warning_for_the_address_alone(self, setting, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz.weather'):
            CumulusMXWeatherClient(setting)
        assert caplog.records == []

    def test_a_fahrenheit_station_is_converted_to_celsius(self):
        result = self._fetch(temp='50.0', tempunitnodeg='F', wspeed='0', wgust='0', windunit='km/h')
        assert result.temperature == pytest.approx(10.0)

    def test_a_celsius_station_is_passed_through(self):
        result = self._fetch(temp='10.0', tempunitnodeg='C', wspeed='0', wgust='0', windunit='km/h')
        assert result.temperature == pytest.approx(10.0)

    @pytest.mark.parametrize('unit, kmh', [
        ('km/h', 10.0), ('mph', 16.09344), ('m/s', 36.0), ('kts', 18.52),
    ])
    def test_each_wind_unit_is_converted_to_kmh(self, unit, kmh):
        result = self._fetch(temp='10', tempunitnodeg='C', wspeed='10', wgust='20', windunit=unit)
        assert (result.wind_speed, result.wind_gust) == (pytest.approx(kmh), pytest.approx(2 * kmh))

    def test_the_unconverted_fields_pass_through(self):
        result = self._fetch(temp='10', tempunitnodeg='C', wspeed='0', wgust='0', windunit='km/h')
        assert (result.humidity, result.solar_radiation, result.wind_bearing) == ('52', '320', '225')

    def test_a_missing_temperature_unit_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match='temperature unit as None'):
            self._fetch(temp='10', wspeed='0', wgust='0', windunit='km/h')

    def test_an_unknown_wind_unit_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match="wind speed unit as 'Bft'"):
            self._fetch(temp='10', tempunitnodeg='C', wspeed='0', wgust='0', windunit='Bft')


class TestOpenMeteoWeatherClient:
    _SAMPLE_RESPONSE = {
        'current': {
            'temperature_2m': 72.5,
            'relative_humidity_2m': 48.0,
            'shortwave_radiation': 410.0,
            'wind_speed_10m': 8.3,
            'wind_gusts_10m': 14.7,
            'wind_direction_10m': 270,
        }
    }

    def test_url_contains_latitude_longitude(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        assert '37.8' in client._url
        assert '-122.4' in client._url

    def test_url_asks_for_the_units_weather_data_holds(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        assert '&temperature_unit=celsius&' in client._url
        assert client._url.endswith('&wind_speed_unit=kmh')

    def test_fetch_returns_six_values(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=_mock_urlopen(self._SAMPLE_RESPONSE)):
            result = client.fetch()
        assert len(result) == 6

    def test_fetch_returns_correct_temperature(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=_mock_urlopen(self._SAMPLE_RESPONSE)):
            temp, *_ = client.fetch()
        assert temp == pytest.approx(72.5)

    def test_fetch_returns_fields_in_correct_order(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=_mock_urlopen(self._SAMPLE_RESPONSE)):
            result = client.fetch()
        assert result == (72.5, 48.0, 410.0, 8.3, 14.7, 270)

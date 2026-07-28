"""Tests for weather data clients: Null, CumulusMX, and Open-Meteo."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from buzz.weather import (
    EMPTY_WEATHER, CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient, WeatherData,
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


class TestCumulusMXWeatherClient:
    _SAMPLE = {
        'temp': 68.2, 'hum': 52.0, 'SolarRad': 320.0,
        'wspeed': 7.5, 'wgust': 12.0, 'avgbearing': 225,
    }

    def _mock_urlopen(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(data).encode()
        return mock_resp

    def test_fetch_returns_six_values(self):
        client = CumulusMXWeatherClient('http://fake/realtime.json')
        with patch('buzz.weather.urllib.request.urlopen', return_value=self._mock_urlopen(self._SAMPLE)):
            result = client.fetch()
        assert len(result) == 6

    def test_fetch_returns_correct_temperature(self):
        client = CumulusMXWeatherClient('http://fake/realtime.json')
        with patch('buzz.weather.urllib.request.urlopen', return_value=self._mock_urlopen(self._SAMPLE)):
            temp, hum, solar, wspeed, wgust, bearing = client.fetch()
        assert temp == pytest.approx(68.2)

    def test_fetch_returns_all_fields_in_order(self):
        client = CumulusMXWeatherClient('http://fake/realtime.json')
        with patch('buzz.weather.urllib.request.urlopen', return_value=self._mock_urlopen(self._SAMPLE)):
            result = client.fetch()
        assert result == (68.2, 52.0, 320.0, 7.5, 12.0, 225)

    def test_url_is_stored(self):
        url = 'http://weather.local/realtime.json'
        client = CumulusMXWeatherClient(url)
        assert client._url == url


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

    def _mock_urlopen(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(data).encode()
        return mock_resp

    def test_url_contains_latitude_longitude(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        assert '37.8' in client._url
        assert '-122.4' in client._url

    def test_url_uses_fahrenheit(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        assert 'fahrenheit' in client._url

    def test_url_uses_mph(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        assert 'mph' in client._url

    def test_fetch_returns_six_values(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=self._mock_urlopen(self._SAMPLE_RESPONSE)):
            result = client.fetch()
        assert len(result) == 6

    def test_fetch_returns_correct_temperature(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=self._mock_urlopen(self._SAMPLE_RESPONSE)):
            temp, *_ = client.fetch()
        assert temp == pytest.approx(72.5)

    def test_fetch_returns_fields_in_correct_order(self):
        client = OpenMeteoWeatherClient(37.8, -122.4)
        with patch('buzz.weather.urllib.request.urlopen',
                   return_value=self._mock_urlopen(self._SAMPLE_RESPONSE)):
            result = client.fetch()
        assert result == (72.5, 48.0, 410.0, 8.3, 14.7, 270)

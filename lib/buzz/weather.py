"""
Weather data clients for annotating noise measurements with environmental conditions.

All clients implement WeatherClient and return a six-tuple
(temperature_f, humidity_pct, solar_w_m2, wind_speed_mph, wind_gust_mph, wind_bearing_deg).

CumulusMXWeatherClient — reads from a local CumulusMX weather station's JSON API.
OpenMeteoWeatherClient — fetches current conditions from the free Open-Meteo API
                         (no key required, requires internet access).
NullWeatherClient      — returns empty strings for all fields; use when no weather
                         source is configured.
"""

import json
import urllib.request
from abc import ABC, abstractmethod

CsvValue = str | float
WeatherData = tuple[CsvValue, CsvValue, CsvValue, CsvValue, CsvValue, CsvValue]

# Weather annotates the measurements but must never stall them: a hung server
# would otherwise block the collector thread indefinitely (urlopen's default
# is no timeout at all).
_FETCH_TIMEOUT_S = 10


class WeatherClient(ABC):
    @abstractmethod
    def fetch(self) -> WeatherData:
        """Returns (temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing)."""
        pass


class CumulusMXWeatherClient(WeatherClient):
    def __init__(self, url: str) -> None:
        self._url = url

    def fetch(self) -> WeatherData:
        with urllib.request.urlopen(self._url, timeout=_FETCH_TIMEOUT_S) as response:
            data = json.loads(response.read())
            return (data['temp'], data['hum'], data['SolarRad'],
                    data['wspeed'], data['wgust'], data['avgbearing'])


class OpenMeteoWeatherClient(WeatherClient):
    _BASE = 'https://api.open-meteo.com/v1/forecast'
    _FIELDS = ('temperature_2m,relative_humidity_2m,shortwave_radiation,'
               'wind_speed_10m,wind_gusts_10m,wind_direction_10m')

    def __init__(self, latitude: float, longitude: float) -> None:
        self._url = (f'{self._BASE}?latitude={latitude}&longitude={longitude}'
                     f'&current={self._FIELDS}'
                     f'&temperature_unit=fahrenheit&wind_speed_unit=mph')

    def fetch(self) -> WeatherData:
        with urllib.request.urlopen(self._url, timeout=_FETCH_TIMEOUT_S) as response:
            current = json.loads(response.read())['current']
            return (current['temperature_2m'], current['relative_humidity_2m'],
                    current['shortwave_radiation'], current['wind_speed_10m'],
                    current['wind_gusts_10m'], current['wind_direction_10m'])


class NullWeatherClient(WeatherClient):
    def fetch(self) -> WeatherData:
        return ('', '', '', '', '', '')

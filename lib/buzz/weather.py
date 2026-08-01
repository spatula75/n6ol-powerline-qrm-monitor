"""
Weather data clients for annotating noise measurements with environmental conditions.

All clients implement WeatherClient and return a WeatherData record.

CumulusMXWeatherClient - reads from a local CumulusMX weather station's JSON API.
OpenMeteoWeatherClient - fetches current conditions from the free Open-Meteo API
                         (no key required, requires internet access).
NullWeatherClient      - returns EMPTY_WEATHER; use when no weather source is
                         configured.
"""

import json
import urllib.request
from abc import ABC, abstractmethod
from typing import NamedTuple

CsvValue = str | float


class WeatherData(NamedTuple):
    """One weather observation.  Fields appear in the CSV in this order; a field
    is an empty string when the source doesn't provide it."""
    temperature: CsvValue      # °F
    humidity: CsvValue         # %
    solar_radiation: CsvValue  # W/m²
    wind_speed: CsvValue       # MPH
    wind_gust: CsvValue        # MPH
    wind_bearing: CsvValue     # degrees


# The record used whenever no weather data is available (no source configured,
# or a fetch failed): all fields blank in the CSV.
EMPTY_WEATHER = WeatherData('', '', '', '', '', '')

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
            return WeatherData(
                temperature=data['temp'], humidity=data['hum'],
                solar_radiation=data['SolarRad'], wind_speed=data['wspeed'],
                wind_gust=data['wgust'], wind_bearing=data['avgbearing'],
            )


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
            return WeatherData(
                temperature=current['temperature_2m'],
                humidity=current['relative_humidity_2m'],
                solar_radiation=current['shortwave_radiation'],
                wind_speed=current['wind_speed_10m'],
                wind_gust=current['wind_gusts_10m'],
                wind_bearing=current['wind_direction_10m'],
            )


class NullWeatherClient(WeatherClient):
    def fetch(self) -> WeatherData:
        return EMPTY_WEATHER

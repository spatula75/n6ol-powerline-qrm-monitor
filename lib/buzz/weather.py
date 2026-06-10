import json
import urllib.request
from abc import ABC, abstractmethod


class WeatherClient(ABC):
    @abstractmethod
    def fetch(self) -> tuple:
        """Returns (temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing)."""
        pass


class CumulusMXWeatherClient(WeatherClient):
    def __init__(self, url: str):
        self._url = url

    def fetch(self) -> tuple:
        with urllib.request.urlopen(self._url) as response:
            data = json.loads(response.read())
            return (data['temp'], data['hum'], data['SolarRad'],
                    data['wspeed'], data['wgust'], data['avgbearing'])

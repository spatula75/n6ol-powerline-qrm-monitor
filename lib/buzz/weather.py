"""
Weather data clients, which annotate noise measurements with the conditions at the time.

Every client implements WeatherClient and returns a WeatherData record in one fixed set
of units, degrees Celsius and km/h.  `WeatherData.in_units` converts a record to the
units the station logs in, and `buzz.csv_store` is the one caller that does so.  Fixing
the units at the client means a source's own unit settings stop at the client that
reads it.

CumulusMXWeatherClient - reads from a local CumulusMX weather station's JSON API.
OpenMeteoWeatherClient - fetches current conditions from the free Open-Meteo API.  It
                         needs no key, and it needs internet access.
NullWeatherClient      - returns EMPTY_WEATHER.  Use it when no weather source is
                         configured.
"""

import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

CsvValue = str | float

# These three are exact by definition: an international mile is 1.609344 km, a knot
# is 1.852 km/h, and one meter per second is 3.6 km/h.
_KMH_PER_MPH = 1.609344
_KMH_PER_KNOT = 1.852
_KMH_PER_METER_PER_SECOND = 3.6

# A converted figure is rounded to tenths.  Both sources report tenths at best, and
# without rounding a conversion writes 69.80000000000001 into the CSV.
_CONVERTED_DECIMALS = 1


class WeatherUnits(StrEnum):
    """The units a station logs temperature and wind speed in.

    Humidity, solar radiation and wind bearing are percent, W/m^2 and degrees in both
    systems, so only three of the six weather fields depend on this.
    """
    IMPERIAL = 'imperial'
    METRIC = 'metric'

    @classmethod
    def from_setting(cls, setting: str) -> 'WeatherUnits':
        """Read `[weather] units`, falling back to imperial on a value it does not know.

        Weather is decoration on the noise measurement, so a mistyped setting costs the
        operator a warning rather than the monitor.
        """
        try:
            return cls(setting)
        except ValueError:
            offered = ', '.join(repr(units.value) for units in cls)
            logger.warning(
                '[weather] units is %r, and it must be one of %s.  The monitor logs weather '
                'in imperial units until somebody corrects the setting in the config file.',
                setting, offered)
            return cls.IMPERIAL

    @property
    def temperature_label(self) -> str:
        """The temperature unit as the CSV header writes it."""
        return 'F' if self is WeatherUnits.IMPERIAL else 'C'

    @property
    def wind_speed_label(self) -> str:
        """The wind speed unit as the CSV header writes it."""
        return 'MPH' if self is WeatherUnits.IMPERIAL else 'km/h'

    def temperature_from_celsius(self, celsius: float) -> float:
        """Convert a Celsius temperature to these units."""
        return celsius * 9 / 5 + 32 if self is WeatherUnits.IMPERIAL else celsius

    def wind_speed_from_kmh(self, kmh: float) -> float:
        """Convert a wind speed in km/h to these units."""
        return kmh / _KMH_PER_MPH if self is WeatherUnits.IMPERIAL else kmh


class WeatherData(NamedTuple):
    """One weather observation.  Fields appear in the CSV in this order.

    A field is an empty string when the source does not provide it.  The units stay
    fixed whatever the station logs in, so that every client returns the same thing.
    """
    temperature: CsvValue      # °C
    humidity: CsvValue         # %
    solar_radiation: CsvValue  # W/m²
    wind_speed: CsvValue       # km/h
    wind_gust: CsvValue        # km/h
    wind_bearing: CsvValue     # degrees

    def in_units(self, units: WeatherUnits) -> 'WeatherData':
        """Return this observation with temperature and wind speed converted to `units`.

        It rounds the converted fields to tenths.  A blank field stays blank, and the
        three fields that need no conversion pass through untouched.
        """
        return self._replace(
            temperature=self._converted(self.temperature, units.temperature_from_celsius),
            wind_speed=self._converted(self.wind_speed, units.wind_speed_from_kmh),
            wind_gust=self._converted(self.wind_gust, units.wind_speed_from_kmh),
        )

    @staticmethod
    def _converted(value: CsvValue, convert: Callable[[float], float]) -> CsvValue:
        """Apply `convert` to a weather field, leaving a blank field blank."""
        if value == '':
            return value
        return round(convert(float(value)), _CONVERTED_DECIMALS)


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
        """Return the current conditions, in degrees Celsius and km/h."""
        pass


class CumulusMXWeatherClient(WeatherClient):
    """Reads a CumulusMX station, in whatever units the station is set to.

    The operator gives the station's address, such as `http://cumulusmx.local:8998/`,
    and the client builds the rest of the URL itself.  The client reads each figure by
    its webtag name, so a query that leaves one out makes every fetch fail and every
    row's weather blank.

    CumulusMX has unit settings of its own, and its JSON endpoint serves every figure in
    them.  The client asks the station which units those are on every fetch, through
    the `tempunitnodeg` and `windunit` webtags, and converts from them.  Asking each time
    means a station whose owner changes its units is read correctly from the next fetch.
    """

    _ENDPOINT_PATH = 'api/tags/process.json'
    # The webtags for the six weather fields, then the two that name the station's units.
    # CumulusMX formats a decimal in its host's locale, so a station in Germany serves
    # "20,5" and float() refuses it.  The leading `rc` is CumulusMX's "remove commas"
    # switch for the whole request, and it works only as the first parameter.  See
    # docs-notebook/cumulusmx-json-api.md.
    _QUERY = 'rc&temp&hum&SolarRad&wspeed&wgust&avgbearing&tempunitnodeg&windunit'

    # The four wind units CumulusMX offers, spelled as `windunit` returns them.  The
    # spellings come from CumulusMX's source, where Cumulus.cs sets `Units.WindText` to
    # exactly one of these, and `tempunitnodeg` returns the letter after the degree
    # sign in "°C" or "°F".  Its master branch confirmed both on 2026-09-27.
    _KMH_PER_WIND_UNIT: dict[str, float] = {
        'km/h': 1.0,
        'mph': _KMH_PER_MPH,
        'm/s': _KMH_PER_METER_PER_SECOND,
        'kts': _KMH_PER_KNOT,
    }

    def __init__(self, url: str) -> None:
        self._url = self._endpoint_url(url)

    @classmethod
    def _endpoint_url(cls, url: str) -> str:
        """Build the full endpoint URL from the operator's `[weather] url`.

        The setting should hold the station's scheme, host and port alone.  Configs
        written before that carry the endpoint path and a query string as well, so the
        client keeps the scheme, host and port and builds the rest itself.  Anything
        after the port no longer has any effect, so the log says what to trim.  An
        address with no scheme gets `http://`, which is what CumulusMX serves by
        default.
        """
        if '://' not in url:
            url = f'http://{url}'
        parts = urlsplit(url)
        base = f'{parts.scheme}://{parts.netloc}/'
        if url.rstrip('/') != base.rstrip('/'):
            logger.warning(
                '[weather] url is %s, and the monitor uses only the start of it, %s.  The '
                'monitor builds the rest of the address itself.  To remove this warning, '
                'set url to %s.', url, base, base)
        return f'{base}{cls._ENDPOINT_PATH}?{cls._QUERY}'

    def fetch(self) -> WeatherData:
        with urllib.request.urlopen(self._url, timeout=_FETCH_TIMEOUT_S) as response:
            data = json.loads(response.read())
        kmh_per_unit = self._kmh_per_wind_unit(data.get('windunit'))
        return WeatherData(
            temperature=self._celsius(float(data['temp']), data.get('tempunitnodeg')),
            humidity=data['hum'],
            solar_radiation=data['SolarRad'],
            wind_speed=float(data['wspeed']) * kmh_per_unit,
            wind_gust=float(data['wgust']) * kmh_per_unit,
            wind_bearing=data['avgbearing'],
        )

    @staticmethod
    def _celsius(temperature: float, unit: str | None) -> float:
        """Convert a temperature from the station's unit to Celsius."""
        if unit == 'C':
            return temperature
        if unit == 'F':
            return (temperature - 32) * 5 / 9
        raise ValueError(
            f'CumulusMX reported its temperature unit as {unit!r}, and this program '
            f'expects C or F.  The likely cause is a CumulusMX version without the '
            f'tempunitnodeg webtag.  The weather columns stay blank until the station reports '
            f'its unit.')

    @classmethod
    def _kmh_per_wind_unit(cls, unit: str | None) -> float:
        """The factor that converts a wind speed in the station's unit to km/h."""
        try:
            return cls._KMH_PER_WIND_UNIT[unit]
        except KeyError:
            offered = ', '.join(cls._KMH_PER_WIND_UNIT)
            raise ValueError(
                f'CumulusMX reported its wind speed unit as {unit!r}, and this program '
                f'expects one of {offered}.  The likely cause is a CumulusMX version without '
                f'the windunit webtag.  The weather columns stay blank until the station '
                f'reports a unit from that list.') from None


class OpenMeteoWeatherClient(WeatherClient):
    _BASE = 'https://api.open-meteo.com/v1/forecast'
    _FIELDS = ('temperature_2m,relative_humidity_2m,shortwave_radiation,'
               'wind_speed_10m,wind_gusts_10m,wind_direction_10m')

    def __init__(self, latitude: float, longitude: float) -> None:
        # Celsius and km/h are Open-Meteo's own defaults.  The query states them anyway,
        # because WeatherData depends on them and a changed default would not announce
        # itself.
        self._url = (f'{self._BASE}?latitude={latitude}&longitude={longitude}'
                     f'&current={self._FIELDS}'
                     f'&temperature_unit=celsius&wind_speed_unit=kmh')

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

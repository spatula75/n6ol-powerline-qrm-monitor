"""
Weather data clients, which annotate noise measurements with the conditions at the time.

Every client implements WeatherClient and returns a WeatherData record in one fixed set
of units: degrees Celsius, km/h and millimeters.  `WeatherData.in_units` converts a
record to the units the station logs in, and `buzz.csv_store` is the one caller that
does so.  Fixing the units at the client means a source's own unit settings stop at the
client that reads it.

CumulusMXWeatherClient - reads from a local CumulusMX weather station's text API.
OpenMeteoWeatherClient - fetches current conditions from the free Open-Meteo API.  It
                         needs no key, and it needs internet access.
NullWeatherClient      - returns EMPTY_WEATHER.  Use it when no weather source is
                         configured.
"""

import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CsvValue = str | float

# These four are exact by definition: an international mile is 1.609344 km, a knot is
# 1.852 km/h, one meter per second is 3.6 km/h, and an inch is 25.4 mm.
_KMH_PER_MPH = 1.609344
_KMH_PER_KNOT = 1.852
_KMH_PER_METER_PER_SECOND = 3.6
_MM_PER_INCH = 25.4

# A converted temperature or wind speed is rounded to tenths.  Both sources report
# tenths at best, and without rounding a conversion writes 69.80000000000001 into the
# CSV.
_CONVERTED_DECIMALS = 1
# Rain keeps hundredths instead, because a tipping-bucket gauge counts in steps of
# 0.01 inch, and tenths would round a whole step away.
_RAIN_DECIMALS = 2


class WeatherUnits(StrEnum):
    """The units a station logs temperature, wind speed and rain in.

    Humidity, solar radiation and wind bearing are percent, W/m^2 and degrees in both
    systems, so they do not depend on this.
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

    @property
    def rain_label(self) -> str:
        """The rain unit as the CSV header will write it."""
        return 'in' if self is WeatherUnits.IMPERIAL else 'mm'

    def temperature_from_celsius(self, celsius: float) -> float:
        """Convert a Celsius temperature to these units."""
        return celsius * 9 / 5 + 32 if self is WeatherUnits.IMPERIAL else celsius

    def wind_speed_from_kmh(self, kmh: float) -> float:
        """Convert a wind speed in km/h to these units."""
        return kmh / _KMH_PER_MPH if self is WeatherUnits.IMPERIAL else kmh

    def rain_from_mm(self, mm: float) -> float:
        """Convert an amount of rain in millimeters to these units."""
        return mm / _MM_PER_INCH if self is WeatherUnits.IMPERIAL else mm


class WeatherData(NamedTuple):
    """One weather observation.

    A value field is an empty string when the source does not provide it, and
    `timestamp` is None.  The units stay fixed whatever the station logs in, so that
    every client returns the same thing.
    """
    temperature: CsvValue          # °C
    humidity: CsvValue             # %
    solar_radiation: CsvValue      # W/m²
    wind_speed: CsvValue           # km/h
    wind_gust: CsvValue            # km/h
    wind_bearing: CsvValue         # degrees
    rain_since_midnight: CsvValue  # mm, since midnight in the source's own day
    # The weather timestamp: the latest time at which the source stands behind these
    # values.  For CumulusMX that is when it last heard from the station, and for
    # Open-Meteo it is the end of the model interval the values describe.
    timestamp: datetime | None

    def in_units(self, units: WeatherUnits) -> 'WeatherData':
        """Return this observation with temperature, wind speed and rain converted to `units`.

        It rounds temperature and wind to tenths, and rain to hundredths.  A blank field
        stays blank, and the fields that need no conversion pass through untouched.
        """
        return self._replace(
            temperature=self._converted(self.temperature, units.temperature_from_celsius),
            wind_speed=self._converted(self.wind_speed, units.wind_speed_from_kmh),
            wind_gust=self._converted(self.wind_gust, units.wind_speed_from_kmh),
            rain_since_midnight=self._converted(self.rain_since_midnight, units.rain_from_mm,
                                                _RAIN_DECIMALS),
        )

    @staticmethod
    def _converted(value: CsvValue, convert: Callable[[float], float],
                   decimals: int = _CONVERTED_DECIMALS) -> CsvValue:
        """Apply `convert` to a weather field, leaving a blank field blank."""
        if value == '':
            return value
        return round(convert(float(value)), decimals)


# The record used whenever no weather data is available (no source configured,
# or a fetch failed): all fields blank in the CSV.
EMPTY_WEATHER = WeatherData('', '', '', '', '', '', '', None)

# Weather annotates the measurements but must never stall them: a hung server
# would otherwise block the collector thread indefinitely (urlopen's default
# is no timeout at all).
_FETCH_TIMEOUT_S = 10


class WeatherClient(ABC):
    @abstractmethod
    def fetch(self) -> WeatherData:
        """Return the current conditions, in degrees Celsius, km/h and millimeters."""
        pass


class CumulusMXWeatherClient(WeatherClient):
    """Reads a CumulusMX station, in whatever units the station is set to.

    The operator gives the station's address, such as `http://cumulusmx.local:8998/`,
    and the client builds the rest of the URL itself.  The client reads each figure by
    its webtag name, so a template that leaves one out makes every fetch fail and every
    row's weather blank.

    The client posts a template to CumulusMX's text API rather than querying its JSON
    API, because only the text API passes a tag its parameters.  The weather timestamp
    needs `format=Unix`, since the default format of `LastDataReadT` follows the host's
    locale and names no timezone.  See docs-notebook/cumulusmx-json-api.md.

    CumulusMX has unit settings of its own, and serves every figure in them.  The client
    asks the station which units those are on every fetch, through the `tempunitnodeg`,
    `windunit` and `rainunit` webtags, and converts from them.  Asking each time means a
    station whose owner changes its units is read correctly from the next fetch.
    """

    _ENDPOINT_PATH = 'api/tags/process.txt'
    # CumulusMX fills in each <#tag> and returns the rest of the text as it stands, so a
    # template shaped like JSON comes back as JSON.  CumulusMX formats a decimal in its
    # host's locale, so a station in Germany serves "20,5", which float() refuses.
    # `rc=y` is CumulusMX's "remove commas" parameter, which turns the comma back into a
    # period.  The integer tags ignore it, and it is on every value tag anyway so that
    # none can be missed.
    _TEMPLATE = json.dumps({
        'temp': '<#temp rc=y>',
        'hum': '<#hum rc=y>',
        'SolarRad': '<#SolarRad rc=y>',
        'wspeed': '<#wspeed rc=y>',
        'wgust': '<#wgust rc=y>',
        'avgbearing': '<#avgbearing rc=y>',
        'rmidnight': '<#rmidnight rc=y>',
        'tempunitnodeg': '<#tempunitnodeg>',
        'windunit': '<#windunit>',
        'rainunit': '<#rainunit>',
        'LastDataReadT': '<#LastDataReadT format=Unix>',
    })

    # What `LastDataReadT` returns before CumulusMX has read anything from the station.
    # webtags.cs substitutes it for any date at or before its default record date,
    # whatever format was asked for.
    _NO_DATA_YET = '----'

    # The units CumulusMX offers, spelled as `windunit` and `rainunit` return them.  The
    # spellings come from CumulusMX's source, where Cumulus.cs sets `Units.WindText` and
    # `Units.RainText` to exactly one of these, and `tempunitnodeg` returns the letter
    # after the degree sign in "°C" or "°F".  Its main branch confirmed all three on
    # 2026-09-28.
    _KMH_PER_WIND_UNIT: dict[str, float] = {
        'km/h': 1.0,
        'mph': _KMH_PER_MPH,
        'm/s': _KMH_PER_METER_PER_SECOND,
        'kts': _KMH_PER_KNOT,
    }
    _MM_PER_RAIN_UNIT: dict[str, float] = {
        'mm': 1.0,
        'in': _MM_PER_INCH,
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
        return f'{base}{cls._ENDPOINT_PATH}'

    def fetch(self) -> WeatherData:
        request = urllib.request.Request(self._url, data=self._TEMPLATE.encode(),
                                         headers={'Content-Type': 'text/plain'}, method='POST')
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:
            data = json.loads(response.read())
        kmh_per_unit = self._conversion_factor(self._KMH_PER_WIND_UNIT, data.get('windunit'),
                                               'wind speed', 'windunit')
        mm_per_unit = self._conversion_factor(self._MM_PER_RAIN_UNIT, data.get('rainunit'),
                                              'rain', 'rainunit')
        return WeatherData(
            temperature=self._celsius(float(data['temp']), data.get('tempunitnodeg')),
            humidity=data['hum'],
            solar_radiation=data['SolarRad'],
            wind_speed=float(data['wspeed']) * kmh_per_unit,
            wind_gust=float(data['wgust']) * kmh_per_unit,
            wind_bearing=data['avgbearing'],
            rain_since_midnight=float(data['rmidnight']) * mm_per_unit,
            timestamp=self._last_data_read(data['LastDataReadT']),
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

    @staticmethod
    def _conversion_factor(factors: Mapping[str, float], unit: str | None,
                           quantity: str, tag: str) -> float:
        """The factor from `factors` that converts the station's `unit` to ours."""
        try:
            return factors[unit]
        except KeyError:
            offered = ', '.join(factors)
            raise ValueError(
                f'CumulusMX reported its {quantity} unit as {unit!r}, and this program '
                f'expects one of {offered}.  The likely cause is a CumulusMX version without '
                f'the {tag} webtag.  The weather columns stay blank until the station '
                f'reports a unit from that list.') from None

    @classmethod
    def _last_data_read(cls, epoch_seconds: str) -> datetime | None:
        """When CumulusMX last heard from the station, or None when it has not yet."""
        if epoch_seconds == cls._NO_DATA_YET:
            return None
        return datetime.fromtimestamp(int(epoch_seconds), UTC)


class OpenMeteoWeatherClient(WeatherClient):
    """Fetches Open-Meteo's current conditions and its rain since the station's midnight.

    Open-Meteo's figures come from weather models, the past ones included, so rain since
    midnight is a model estimate.  Nothing guarantees that the total only rises, and
    the client records it as it arrives.  See docs-notebook/online-weather-sources.md.
    """

    _BASE = 'https://api.open-meteo.com/v1/forecast'
    _FIELDS = ('temperature_2m,relative_humidity_2m,shortwave_radiation,'
               'wind_speed_10m,wind_gusts_10m,wind_direction_10m')

    def __init__(self, latitude: float, longitude: float, timezone: str) -> None:
        self._zone = ZoneInfo(timezone)
        # Celsius, km/h and mm are Open-Meteo's own defaults.  The query states them
        # anyway, because WeatherData depends on them and a changed default would not
        # announce itself.  `timezone` and `forecast_days=1` make the 15-minute series
        # run from the station's midnight, and `timeformat=unixtime` gives times with
        # no zone to misread.
        self._url = (f'{self._BASE}?latitude={latitude}&longitude={longitude}'
                     f'&current={self._FIELDS}&minutely_15=precipitation'
                     f'&temperature_unit=celsius&wind_speed_unit=kmh&precipitation_unit=mm'
                     f'&timezone={quote(timezone, safe="")}&forecast_days=1'
                     f'&timeformat=unixtime')

    def fetch(self) -> WeatherData:
        with urllib.request.urlopen(self._url, timeout=_FETCH_TIMEOUT_S) as response:
            payload = json.loads(response.read())
        current = payload['current']
        # `current` describes the 15 minutes that end at its time, which is therefore
        # the latest moment the values speak for.
        interval_end = current['time']
        series = payload['minutely_15']
        return WeatherData(
            temperature=current['temperature_2m'],
            humidity=current['relative_humidity_2m'],
            solar_radiation=current['shortwave_radiation'],
            wind_speed=current['wind_speed_10m'],
            wind_gust=current['wind_gusts_10m'],
            wind_bearing=current['wind_direction_10m'],
            rain_since_midnight=self._rain_since_midnight(series['time'], series['precipitation'],
                                                          interval_end),
            timestamp=datetime.fromtimestamp(interval_end, UTC),
        )

    def _rain_since_midnight(self, times: Sequence[int], amounts: Sequence[float | None],
                             interval_end: int) -> CsvValue:
        """Add up the completed 15-minute amounts since the station's midnight.

        Each amount covers the 15 minutes that end at its time.  The one labeled
        midnight therefore fell the day before, and one labeled after `interval_end`
        has not happened yet, which leaves it a forecast.  Both are left out.  A gap in
        the amounts makes the total unknown, so it comes back blank rather than short.
        """
        local_end = datetime.fromtimestamp(interval_end, self._zone)
        midnight = local_end.replace(hour=0, minute=0, second=0).timestamp()
        falling = [amount for time, amount in zip(times, amounts) if midnight < time <= interval_end]
        if None in falling:
            return ''
        return sum(falling, start=0.0)


class NullWeatherClient(WeatherClient):
    def fetch(self) -> WeatherData:
        return EMPTY_WEATHER

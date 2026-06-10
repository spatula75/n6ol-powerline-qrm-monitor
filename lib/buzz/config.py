"""
Configuration dataclasses and TOML loader for the powerline QRM monitor.

BuzzConfig is the top-level config object, composed of four section dataclasses:
AudioConfig, StationConfig, WeatherConfig, and ServerConfig.  Each maps directly
to a [section] in ~/.buzz/config.toml.  BuzzConfig.from_toml() reads the file and
populates the dataclasses; unknown keys are silently ignored so old config files
don't break when new fields are added.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import tomllib

_T = TypeVar('_T')

_MODULE_DIR = Path(__file__).resolve().parent

CONFIG_PATH = Path.home() / '.buzz' / 'config.toml'


@dataclass
class AudioConfig:
    # Sounddevice name of the audio input recording the RF-to-audio converted signal.
    input_device_name: str = 'Line In (Realtek(R) Audio), Windows DirectSound'
    # PortAudio device index for the selected input. Set by configure.py; takes
    # precedence over input_device_name at runtime. None = look up by name.
    device_index: int | None = None
    # Audio sample rate in Hz. Must match what the input device is configured to use.
    sample_rate: int = 16000
    # Length of each audio recording in seconds. Longer = more pulses to average over.
    duration: int = 3
    # Number of recordings averaged together per CSV entry. Higher = less noisy readings.
    measurements_to_take: int = 3
    # Powerline interference pulse rate in Hz: 120 for 60 Hz grid (North America),
    # 100 for 50 Hz grid (Europe and most of the rest of the world).
    pulse_rate: int = 120


@dataclass
class StationConfig:
    # Your amateur radio callsign, used in page titles and plot labels.
    callsign: str = 'N0CALL'
    # IANA timezone name for CSV timestamps and graph labels.
    timezone: str = 'America/Los_Angeles'
    # Local directory where CSV files, plots, and the index page are written.
    path: str = str(Path.home())
    # Receiver noise floor in dBm. Combined with noise_min_snr to set noise_threshold.
    noise_floor: float = -98.0
    # Minimum SNR in dB to count a sample as interference-present in the summary graphs.
    noise_min_snr: float = 12.0
    # dB offset applied to audio amplitude to approximate RF level at the receiver input.
    # Hardware-specific: derived by calibrating against a known signal level.
    audio_rf_conversion_db: float = -32.0
    # Path loss in dB from the powerline to the monitoring location.
    # Used to estimate source strength from the measured level.
    distance_attenuation: float = 29.54
    # ISO 8601 start date for the all-time summary graph.
    summary_start_date_iso: str = '2024-05-15T00:00:00-0700'

    @property
    def noise_threshold(self) -> float:
        """Detection threshold in dBm: the noise floor plus the minimum SNR required for a valid detection.

        A sample must exceed this level to be counted as interference in the summary graphs.
        """
        return self.noise_floor + self.noise_min_snr


@dataclass
class WeatherConfig:
    # Weather data source: 'cumulusmx', 'openmeteo', or 'none'.
    source: str = 'cumulusmx'
    # CumulusMX JSON endpoint (used when source = 'cumulusmx').
    url: str = ''
    # Latitude and longitude for Open-Meteo (used when source = 'openmeteo').
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class ServerConfig:
    # Set to false to disable all uploads and run in local-only mode.
    enabled: bool = True
    # Hostname or IP of the web server that hosts the published output.
    host: str = ''
    username: str = ''                              # SSH username on the web server
    remote_path: str = ''                           # Remote path for uploaded data files
    key_path: str = str(_MODULE_DIR / 'buzz.pem')   # SSH private key for SCP authentication


@dataclass
class BuzzConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    station: StationConfig = field(default_factory=StationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_toml(cls, path: Path | str = CONFIG_PATH) -> 'BuzzConfig':
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls(
            audio=_load_section(data, 'audio', AudioConfig),
            station=_load_section(data, 'station', StationConfig),
            weather=_load_section(data, 'weather', WeatherConfig),
            server=_load_section(data, 'server', ServerConfig),
        )


def _load_section(data: dict[str, Any], key: str, cls: type[_T]) -> _T:
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.get(key, {}).items() if k in known})

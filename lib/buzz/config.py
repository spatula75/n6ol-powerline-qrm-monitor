"""
Configuration dataclasses and TOML loader for the powerline QRM monitor.

BuzzConfig is the top-level config object, composed of five section dataclasses:
AudioConfig, StationConfig, WeatherConfig, ServerConfig, and RecordingConfig.  Each
maps directly to a [section] in ~/.buzz/config.toml.  BuzzConfig.from_toml() reads
the file and populates the dataclasses; unknown keys are silently ignored so old
config files don't break when new fields are added.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar('_T')

CONFIG_PATH = Path.home() / '.buzz' / 'config.toml'


@dataclass
class AudioConfig:
    # Sounddevice name of the audio input recording the RF-to-audio converted signal.
    input_device_name: str = 'Line In (Realtek(R) Audio), Windows DirectSound'
    # PortAudio device index written by configure.py for reference. Not used at
    # runtime — the device is always resolved by input_device_name, which is stable
    # across reboots. Indices change whenever Windows reassigns USB/audio devices.
    device_index: int | None = None
    # Audio sample rate in Hz. Must match what the input device is configured to use.
    sample_rate: int = 16000
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
    distance_attenuation: float = 30.0
    # ISO 8601 start date for the all-time summary graph.
    summary_start_date_iso: str = '2024-01-01T00:00:00+0000'

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
    # Set to true to enable SCP uploads to a web server; false runs in local-only mode.
    enabled: bool = False
    # Hostname or IP of the web server that hosts the published output.
    host: str = ''
    username: str = ''                              # SSH username on the web server
    remote_path: str = ''                           # Remote path for uploaded data files
    key_path: str = str(Path.home() / '.buzz' / 'buzz.pem')   # SSH private key for SCP authentication


@dataclass
class RecordingConfig:
    # Set to true to arm event recording at startup; --enable-recording arms it for
    # one run without editing the config, and the toolbar button toggles it live.
    enabled: bool = False
    # Directory for recorded .wav files.  Empty means <station.path>/recordings.
    # Also where --playback looks when given a bare filename rather than a path.
    directory: str = ''
    # How many of the next events to record before disarming.  0 records every
    # event until recording is switched off by hand.
    max_events: int = 10
    # Minutes between resets of that budget, making max_events a rate rather than a
    # one-off: 1440 gives max_events per day, indefinitely.  The cycle runs from when
    # the budget was last reset, not from when it ran out, so it does not drift.
    # 0 never re-arms: once the budget is spent, recording stays off until armed by hand.
    rearm_reset_minutes: float = 0.0
    # Longest single recording in seconds, measured from the moment of lock; the
    # lead-in and trailer are extra.  0 records the event however long it runs.
    max_seconds: float = 120.0
    # Seconds without a lock before a recording is closed.  This audio is kept, so
    # the value also sets how much trailer every recording ends with.
    stop_after_seconds: float = 10.0
    # How long the pulse train must be held before a recording starts, which is how
    # a night of two-second blips is kept off the disk.  Every second spent waiting
    # is a second of lead-in lost, since the buffer it comes from is a sliding
    # window, so this is capped at what that buffer holds.  0 records every lock.
    min_lock_seconds: float = 0.0

    def directory_path(self, station: StationConfig) -> Path:
        """Resolve `directory` against the station's output path when it is unset."""
        return Path(self.directory) if self.directory else Path(station.path) / 'recordings'


@dataclass
class BuzzConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    station: StationConfig = field(default_factory=StationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    @classmethod
    def from_toml(cls, path: Path | str = CONFIG_PATH) -> 'BuzzConfig':
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls(
            audio=_load_section(data, 'audio', AudioConfig),
            station=_load_section(data, 'station', StationConfig),
            weather=_load_section(data, 'weather', WeatherConfig),
            server=_load_section(data, 'server', ServerConfig),
            recording=_load_section(data, 'recording', RecordingConfig),
        )


def _load_section(data: dict[str, Any], key: str, cls: type[_T]) -> _T:
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.get(key, {}).items() if k in known})

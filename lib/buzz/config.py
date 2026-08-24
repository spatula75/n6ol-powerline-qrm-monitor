"""
Configuration dataclasses and TOML loader for the powerline QRM monitor.

BuzzConfig is the top-level config object, composed of six section dataclasses:
AudioConfig, StationConfig, WeatherConfig, ServerConfig, RecordingConfig, and
RenderConfig. Each maps directly to a [section] in ~/.buzz/config.toml.
BuzzConfig.from_toml() reads the file and populates the dataclasses. Unknown keys
are silently ignored, so old config files don't break when new fields are added.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar('_T')

CONFIG_PATH = Path.home() / '.buzz' / 'config.toml'

# The band of sample rates this program will work in.
#
# The floor is set by what the display and the analysis are looking at: the waterfall
# shows 0-4 kHz, so 8 kHz is exactly twice that and the lowest rate that can carry the
# band at all. Below it the top of the display is above Nyquist and there is nothing
# there to show.
#
# The ceiling is practical rather than theoretical. Nothing in a powerline arc lives
# above a few kHz, so a higher rate buys no signal and costs proportionally more of
# everything. The fixed-size ring buffer also holds 3.2 s at 48 kHz against 9.6 s at
# 16 kHz, so history shrinks as the rate climbs. 48 kHz is the highest rate a file
# from elsewhere is likely to arrive at, and the lowest useful history this can give.
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


def validate_sample_rate(sample_rate: int, source: str, configured_rate: int) -> None:
    """Refuse a sample rate the rest of the program cannot honestly work at.

    This refuses rather than copes, because both directions produce a display and a
    set of numbers that look perfectly plausible and are not. Below the floor, the
    top of the waterfall is above Nyquist and shows an empty band. Far above the
    ceiling, the buffer holds too little history for the analyzer to acquire the way
    it was tuned to. `source` names where the rate came from, since a bad one can
    arrive from the config or from a file somebody sent.

    `configured_rate` is this station's own, which is what the remedy suggests
    resampling to: a file at the rate the rest of the setup already uses is the one
    that needs the least explaining afterwards. It is a suggestion rather than an
    instruction, because anything inside the band will work.
    """
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(
            f'{source} has a sample rate of {sample_rate} Hz, and this program works '
            f'only where {MIN_SAMPLE_RATE} <= sample rate <= {MAX_SAMPLE_RATE} Hz. '
            f'Below {MIN_SAMPLE_RATE} Hz the 4 kHz the display and the analysis look '
            'at is above Nyquist, so there is nothing there to measure. Above '
            f'{MAX_SAMPLE_RATE} Hz the fixed-size buffer holds too little history to '
            'acquire reliably, and a powerline arc has nothing to say up there '
            f'anyway. Consider resampling it to {configured_rate} Hz, the rate this '
            'station is configured to use.')


@dataclass
class AudioConfig:
    # Sounddevice name of the audio input recording the RF-to-audio converted signal.
    # The device is always resolved by this name, never by a stored index: names
    # survive a reboot, and indices change whenever Windows reassigns audio hardware.
    input_device_name: str = 'Line In (Realtek(R) Audio), Windows DirectSound'
    # Audio sample rate in Hz. Must match what the input device is configured to use,
    # and must lie between MIN_SAMPLE_RATE and MAX_SAMPLE_RATE -- see validate_sample_rate.
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
    # Publish a probability summary over the whole data set, not just the last 7 and 30
    # days.  Off by default because a station running more than a few months has usually
    # seen the noise change, and the whole-history average then describes neither the
    # fault nor the repair.
    enable_all_time_summary: bool = False
    # ISO 8601 start date for the all-time summary graph.  Used only when
    # enable_all_time_summary is on.
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
    # Set to true to enable SCP uploads to a web server. False runs in local-only mode.
    enabled: bool = False
    # Hostname or IP of the web server that hosts the published output.
    host: str = ''
    username: str = ''                              # SSH username on the web server
    remote_path: str = ''                           # Remote path for uploaded data files
    key_path: str = str(Path.home() / '.buzz' / 'buzz.pem')   # SSH private key for SCP authentication
    # How data/current.png, the fixed name the web page reads, is published.
    # 'copy' uploads a second copy of the chart each cycle and works on any server.
    # 'symlink' avoids that upload but needs the web server to follow symlinks, which
    # Apache does only with FollowSymLinks and some shared hosts refuse.  Defaulting to
    # 'copy' means a new station works with no server configuration at all; the
    # duplicate upload is one chart per minute, which no link this program targets
    # will notice.
    current_chart: str = 'copy'


@dataclass
class RecordingConfig:
    # Set to true to arm event recording at startup. --enable-recording arms it for
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
    # Longest single recording in seconds, measured from the moment of lock. The
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
    # Minimum SNR in dB before a recording starts, for keeping barely-audible events
    # off the disk.  Recording only: it does not change when the analyzer locks, which
    # happens at ContinuousAnalyzer.LOCK_ACQUIRE_SNR and is not configurable, so
    # anything at or below that is the same as 0.  A signal that starts weak and grows
    # is recorded from the moment it crosses, not skipped.
    min_lock_snr: float = 0.0

    def directory_path(self, station: StationConfig) -> Path:
        """Resolve `directory` against the station's output path when it is unset."""
        return Path(self.directory) if self.directory else Path(station.path) / 'recordings'


@dataclass
class RenderConfig:
    # Where to find ffmpeg, for --render.  Empty searches PATH, which is where a
    # normal install puts it, so most people never set this.  It is here for the
    # installs that don't appear on PATH -- a Windows build unzipped into a folder, or
    # winget's shim directory before the terminal has been restarted.
    #
    # ffmpeg is needed for --render and for nothing else.  A monitor that never
    # renders never looks for it, so leaving this empty costs nothing.
    ffmpeg_path: str = ''


@dataclass
class BuzzConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    station: StationConfig = field(default_factory=StationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

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
            render=_load_section(data, 'render', RenderConfig),
        )


def _load_section(data: dict[str, Any], key: str, cls: type[_T]) -> _T:
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.get(key, {}).items() if k in known})

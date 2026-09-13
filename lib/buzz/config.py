"""
Configuration dataclasses and TOML loader for the powerline QRM monitor.

BuzzConfig is the top-level config object, composed of seven section dataclasses:
AudioConfig, StationConfig, WeatherConfig, ServerConfig, RecordingConfig,
RenderConfig and RtlSdrConfig.  Each maps directly to a [section] in
~/.buzz/config.toml.  BuzzConfig.from_toml() reads the file and populates the
dataclasses.  Unknown keys are silently ignored, so old config files do not break
when new fields are added.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from buzz.constants import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

_T = TypeVar('_T')

CONFIG_PATH = Path.home() / '.buzz' / 'config.toml'


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


# Where the live audio comes from.  These are alternatives rather than additions, so
# a station picks one.  `soundcard` is a radio feeding a sound card, which is what
# this program did before anything else existed.  `rtlsdr` is an RTL-SDR receiver,
# with the settings in the [rtlsdr] section.
SOUNDCARD = 'soundcard'
RTLSDR = 'rtlsdr'


@dataclass
class RtlSdrConfig:
    """Settings for an RTL-SDR receiver.

    The fields are ordered as the setup program shows them, because the menu follows
    the schema and the schema follows this class.  The four an operator sets come
    first, in the order they are set, and the ones nobody should touch come after.
    """

    # Frequency to listen on, in Hz.  The receiver is tuned away from this by
    # tuning_offset_hz and the difference is undone in software, so this is the
    # frequency that is measured rather than the one the hardware sits at.
    #
    # The default puts the whole sampled span inside 80m and clear of the CW DX
    # window.  The device tunes 50 kHz above this, so the 256 kHz span runs from
    # 3.510 to 3.766 MHz.  Powerline noise is generally worse low in HF, which is why
    # the default sits on 80m rather than higher.
    frequency_hz: int = 3_588_000
    # Tuner gain in dB.  Snapped to the nearest step the tuner offers, since it accepts
    # only a fixed set.  Measured on an RTL-SDR Blog V4, the useful range starts
    # around 22.9 dB, because below that the output is the converter's own noise
    # rather than anything from the antenna.
    gain_db: float = 40.2
    # dB added to the measured audio level to get signal level at the receiver input,
    # the same job station.audio_rf_conversion_db does for a sound card.  It lives here
    # rather than there because the figure depends on gain_db above, so the two belong
    # together.
    #
    # Unset means estimate it as the negative of gain_db, which puts a new station
    # within a few dB with no equipment at all.  That is a place to start from and not
    # a substitute for calibrating.  See level_offset_db for how far the estimate
    # drifts.  SNR, lock, phase and grid frequency do not depend on it either way,
    # since the offset cancels in a difference.  Only absolute levels and the S-meter
    # move.
    audio_rf_conversion_db: float | None = None
    # The gain audio_rf_conversion_db was calibrated against, written by the setup
    # program rather than chosen.  Changing gain_db afterwards leaves the offset wrong
    # by roughly the difference, and nothing else would notice, so startup compares
    # the two and says so.  The estimate needs no such check, because it is computed
    # from gain_db every time.
    calibrated_at_gain_db: float | None = None
    # dB of room the gain calibration keeps above the quiet band noise, so that an
    # arc has somewhere to go.  Measured above the level *between* bursts rather than
    # during one, which is what lets the calibration run on a dead band: sizing from
    # an observed peak needs an arc to be present, and nothing arranges that.
    #
    # The value came from measuring, not from theory.  Over 11147 locked minutes at
    # one station the loudest arc reached 30.95 dB above its own noise floor, with
    # 3.7% of minutes past 25 dB and nothing at all past 35.  Those logs were taken
    # through a 4 kHz SSB filter, while clipping happens across the whole 256 kHz,
    # where the crest of a burst runs a decibel or two higher.  32 covers the worst
    # logged arc and that difference.
    #
    # Raising it costs noise-floor accuracy, because the gain it permits is lower and
    # more of the measured floor is then the converter's own.  On the antenna these
    # figures came from, 30 gives 88% of the floor to the antenna and 32 gives 81%,
    # a difference of 0.37 dB in the reported floor.
    #
    # That trade goes this way because clipping is not recoverable and floor error is.
    # Clipping is nonlinear, so it does not merely under-read the arc: it puts
    # products across the whole span that lift the apparent floor in the same capture.
    # Both numbers are then wrong and nothing in the data says so.  A converter-limited
    # floor is wrong by a bounded amount in a known direction.
    arc_headroom_db: float = 32.0
    # Which receiver to use when more than one is plugged in.  Two identical receivers
    # cannot be told apart, since the serial reads 00000001 on both unless somebody
    # reprogrammed it.  Try one, and use the other if the wrong receiver answers.
    device_index: int = 0
    # Rate the receiver samples at, in Hz.  256000 divides by 16 to give exactly 16000
    # Hz of audio, matching what a sound-card station uses.  The hardware cannot
    # produce every rate exactly, so the figure is read back after it is set.
    iq_sample_rate: int = 256_000
    # IQ samples per audio sample.  Must divide iq_sample_rate exactly, so that the
    # audio rate is a whole number of samples per second.
    decimation: int = 16
    # How much of the band to keep, in Hz, on one side of frequency_hz.  4000 matches a
    # typical SSB filter, which is what makes levels comparable with a receiver.
    bandwidth_hz: int = 4_000
    # How far from frequency_hz to tune the hardware, in Hz.  A receiver puts a strong
    # false signal at exactly its own tuning frequency, so this moves that artifact out
    # of the measured band.  Undone in software, so it costs nothing but coverage on
    # one side.
    #
    # Measured on an RTL-SDR Blog V4, it also clears two spurs that ride the tuner: one
    # about 32 dB over the floor at the bottom band edge, and a pair about 24 dB over
    # it at plus and minus 10 kHz.  That was luck rather than design, and it is worth
    # rechecking before this value moves.
    tuning_offset_hz: int = 50_000
    # Which side of frequency_hz to listen to: 'upper' or 'lower'.  Either works for
    # measuring an arc.  A receiver in LSB shows the spectrum reversed, so the two
    # differ in how a waterfall reads rather than in what is measured.
    sideband: str = 'upper'

    @property
    def level_offset_db(self) -> float:
        """The dB offset to apply, measured if there is one and estimated otherwise.

        The estimate is the negative of the tuner gain.  The reasoning is that the gain
        is the only part of the chain that changes, so subtracting it leaves a constant
        belonging to the receiver itself, and assuming that constant is zero gets a new
        station most of the way there.

        It is an estimate rather than an answer, for two reasons.  Everything else in
        the path has a gain of its own, and nothing arranges for it to cancel.  And the
        tuner's own labels are not true dB: measured on an RTL-SDR Blog V4, the full
        range came to 57.5 dB against a nominal 49.6.

        Measured on this hardware, the same unchanging signal reported through this
        estimate moves 5.1 dB across the whole gain range, and 3.0 dB over the part
        anybody would use.  It is good enough to start from and not good enough to
        publish.

        That measurement covers one RTL-SDR Blog V4 on one bench.  The rest of the
        chain summing to near zero is a property of that unit
        rather than of RTL-SDR receivers in general, so another unit could sit
        several dB away.  The tuner gain dominates on any unit, which is why the
        estimate beats zero anywhere, but the residual has been measured exactly
        once.
        """
        if self.audio_rf_conversion_db is not None:
            return self.audio_rf_conversion_db
        return -self.gain_db


@dataclass
class AudioConfig:
    # Where live audio comes from, either 'soundcard' or 'rtlsdr'.
    source: str = SOUNDCARD
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
    rtlsdr: RtlSdrConfig = field(default_factory=RtlSdrConfig)

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
            rtlsdr=_load_section(data, 'rtlsdr', RtlSdrConfig),
        )


def _load_section(data: dict[str, Any], key: str, cls: type[_T]) -> _T:
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.get(key, {}).items() if k in known})

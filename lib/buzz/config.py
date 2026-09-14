"""
Configuration dataclasses and TOML loader for the powerline QRM monitor.

BuzzConfig is the top-level config object, composed of seven section dataclasses:
AudioConfig, StationConfig, WeatherConfig, ServerConfig, RecordingConfig,
RenderConfig and RtlSdrConfig.  Each maps directly to a [section] in
~/.buzz/config.toml.  BuzzConfig.from_toml() reads the file and populates the
dataclasses.  An unknown key is ignored rather than fatal, so a config file does not
break when settings come and go, but _load_section says which key it dropped: silence
there hid a renamed setting reverting to its default.
"""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from buzz.constants import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

logger = logging.getLogger(__name__)

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

# Marks a BuzzConfig field that holds runtime state rather than a configured setting.
#
# Such a field has no place in schema.json or config.example.toml, since nobody sets
# it and writing it to a file would invite somebody to try.  The drift pins that tie
# the dataclasses, the schema and the sample config together skip anything carrying
# it, so adding another needs no edit to those tests.
RUNTIME = {'runtime': True}


def is_runtime(field_info: Any) -> bool:
    """Whether a dataclass field holds runtime state rather than a setting."""
    return bool(field_info.metadata.get('runtime'))


@dataclass
class RtlSdrConfig:
    """Settings for an RTL-SDR receiver.

    The fields are ordered as the setup program shows them, because the menu follows
    the schema and the schema follows this class.  The four an operator sets come
    first, in the order they are set, and the ones nobody should touch come after.
    """

    # Frequency to listen on, in kHz.  The receiver is tuned away from this by
    # tuning_offset_hz and the difference is undone in software, so this is the
    # frequency that is measured rather than the one the hardware sits at.
    #
    # The default puts the whole sampled span inside 80m and clear of the CW DX
    # window.  The device tunes 50 kHz above this, so the 256 kHz span runs from
    # 3.510 to 3.766 MHz.  Powerline noise is generally worse low in HF, which is why
    # the default sits on 80m rather than higher.
    #
    # In kHz because that is how an operator says a frequency, and typing three
    # zeroes on the end of every one is a way to get a band wrong by a factor of ten.
    # Everything below this line stays in Hz, since the hardware and the arithmetic
    # both work there; frequency_hz converts once, at the boundary.
    frequency_khz: float = 3588.0
    # Tuner gain in dB.  The monitor snaps this to the nearest step the tuner offers,
    # since the tuner accepts only a fixed set.  Measured on an RTL-SDR Blog V4, the
    # useful range starts around 20.7 dB, because below that the output is the
    # converter's own noise rather than anything from the antenna.
    #
    # 22.9 is what the automatic calibration measured on the broadband antenna this was
    # developed against, so the shipped figure is one the tool arrived at rather than a
    # guess.
    #
    # Every antenna differs, so run the calibration rather than trusting this.
    gain_db: float = 22.9
    # dB added to the measured audio level to get signal level at the receiver input,
    # the same job station.audio_rf_conversion_db does for a sound card.  It lives here
    # rather than there because the figure depends on gain_db above, so the two belong
    # together.
    #
    # Named differently from the sound card's on purpose.  The two are different
    # quantities: that one describes a radio and its wiring, this one is mostly the
    # negative of the tuner gain and moves whenever the gain does.  Sharing a name
    # across two sections read as one setting stored twice, which is what somebody
    # took it for.
    #
    # Unset means estimate it as the negative of gain_db, which puts a new station
    # within a few dB with no equipment at all.  That is a place to start from and not
    # a substitute for calibrating.  See level_offset_db for how far the estimate
    # drifts.  SNR, lock, phase and grid frequency do not depend on it either way,
    # since the offset cancels in a difference.  Only absolute levels and the S-meter
    # move.
    calibrated_offset_db: float | None = None
    # The gain calibrated_offset_db was calibrated against, written by the setup
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
    # How much of the band to keep, in kHz, on one side of the listening frequency.
    # 4 kHz matches a typical SSB filter, which is what makes levels comparable with a
    # receiver.
    bandwidth_khz: float = 4.0
    # How far from the listening frequency to tune the hardware, in kHz.  A receiver
    # puts a strong false signal at exactly its own tuning frequency, so this moves
    # that artifact out of the measured band.  Undone in software, so it costs nothing
    # but coverage on one side.
    #
    # Measured on an RTL-SDR Blog V4, it also clears two spurs that ride the tuner: one
    # about 32 dB over the floor at the bottom band edge, and a pair about 24 dB over
    # it at plus and minus 10 kHz.  That was luck rather than design, so recheck it
    # before this value moves.
    tuning_offset_khz: float = 50.0
    # Which side of the listening frequency to listen to: 'upper' or 'lower'.  Either works for
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
        if self.calibrated_offset_db is not None:
            return self.calibrated_offset_db
        return -self.gain_db

    @property
    def frequency_hz(self) -> int:
        """The listening frequency in Hz, which is what the receiver is set in."""
        return self._as_hz(self.frequency_khz)

    @property
    def bandwidth_hz(self) -> int:
        """The kept bandwidth in Hz, which is what the filter is designed in."""
        return self._as_hz(self.bandwidth_khz)

    @property
    def tuning_offset_hz(self) -> int:
        """The tuning offset in Hz, which is what the mixer is stepped in."""
        return self._as_hz(self.tuning_offset_khz)

    @staticmethod
    def _as_hz(khz: float) -> int:
        """kHz to whole Hz.

        Rounded rather than truncated, so a figure typed to the nearest hundred Hz
        arrives exactly: 3588.1 kHz is 3588100 Hz and not 3588099.  Neither the tuner
        nor the arithmetic downstream has any use for a fraction of a Hz.
        """
        return round(khz * 1000)


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
    # Publish a chart of the estimated grid frequency over the current day.  Off by
    # default.
    enable_frequency_chart: bool = False

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
    # Second, matching the schema, because the setup program walks the sections in
    # that order and a receiver is configured immediately after the source that
    # selects it.  tests/test_setup_schema.py pins the two together.
    rtlsdr: RtlSdrConfig = field(default_factory=RtlSdrConfig)
    station: StationConfig = field(default_factory=StationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    # Set at runtime rather than from the file, and read only through
    # level_offset_db below.  Playback uses it to adopt the figure a recording was
    # made with, or the one --audio-rf-conversion-db supplies, neither of which
    # describes this station's own hardware.
    #
    # RUNTIME marks it as not a setting, which is what keeps it out of schema.json
    # and config.example.toml.  The drift pins tying those three together read the
    # marker rather than a list of exceptions, so the next one costs nothing.
    level_offset_override_db: float | None = field(default=None, metadata=RUNTIME)

    @property
    def level_offset_db(self) -> float:
        """The dB added to an audio level to get a signal level at the receiver input.

        One question with one answer, resolved here rather than stored.  Two sections
        carry a figure because the two mean different things: a sound card's is a
        property of the wiring and the radio, and a receiver's depends on the tuner
        gain that sits beside it.  Only one can apply, and which one is decided by
        [audio] source.

        Everything that converts a level reads this, so the two cannot be set
        independently and have the program follow the wrong one.  Before it existed,
        startup copied the receiver's figure over the sound card's, and a config file
        then held two settings of the same name with only one in use and nothing
        saying which.  That is what this exists to make impossible.

        A playback override beats both, because it is the only figure anybody
        deliberately supplied for the file being replayed.
        """
        if self.level_offset_override_db is not None:
            return self.level_offset_override_db
        if self.audio.source == RTLSDR:
            return self.rtlsdr.level_offset_db
        return self.station.audio_rf_conversion_db

    @classmethod
    def from_toml(cls, path: Path | str = CONFIG_PATH) -> 'BuzzConfig':
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls(
            audio=_load_section(data, 'audio', AudioConfig),
            rtlsdr=_load_section(data, 'rtlsdr', RtlSdrConfig),
            station=_load_section(data, 'station', StationConfig),
            weather=_load_section(data, 'weather', WeatherConfig),
            server=_load_section(data, 'server', ServerConfig),
            recording=_load_section(data, 'recording', RecordingConfig),
            render=_load_section(data, 'render', RenderConfig),
        )


def _load_section(data: dict[str, Any], key: str, cls: type[_T]) -> _T:
    """Build one section, keeping only the keys it declares and reporting the rest.

    An unknown key is still ignored rather than fatal, so a file written by a newer
    build, or one carrying a setting since removed, still starts.  What changed is
    that it no longer happens in silence.

    Silence was costing more than it saved.  A key renamed during development left a
    receiver's calibration behind without a word: the figure reverted to the default,
    the menu went on describing an estimate as an estimate, and nothing anywhere said
    a line had been dropped.  A typo does exactly the same thing, which is the case
    this keeps catching after the renaming stops.
    """
    known = set(cls.__dataclass_fields__)
    section = data.get(key, {})
    for unknown in sorted(set(section) - known):
        logger.warning(
            '[%s] %s is not a setting this program knows, so it was ignored and the '
            'default used instead.  Check the spelling against config.example.toml.',
            key, unknown)
    return cls(**{k: v for k, v in section.items() if k in known})

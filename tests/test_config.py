"""Tests for configuration dataclasses, TOML loader, and derived properties."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from buzz.config import (
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, AudioConfig, BuzzConfig, RecordingConfig,
    ServerConfig, StationConfig, WeatherConfig, _load_section, validate_sample_rate,
)

_EXAMPLE = Path(__file__).resolve().parent.parent / 'config.example.toml'


class TestAudioConfigDefaults:
    def test_sample_rate(self):
        assert AudioConfig().sample_rate == 16000

    def test_pulse_rate(self):
        assert AudioConfig().pulse_rate == 120

    def test_device_index_default_none(self):
        assert AudioConfig().device_index is None


class TestStationConfigNoiseThreshold:
    def test_noise_threshold_default(self):
        s = StationConfig()
        assert s.noise_threshold == pytest.approx(-86.0)

    def test_noise_threshold_is_floor_plus_snr(self):
        s = StationConfig(noise_floor=-90.0, noise_min_snr=10.0)
        assert s.noise_threshold == pytest.approx(-80.0)

    def test_noise_threshold_updates_with_floor(self):
        s = StationConfig(noise_floor=-100.0, noise_min_snr=12.0)
        assert s.noise_threshold == pytest.approx(-88.0)


class TestWeatherConfigDefaults:
    def test_source_default(self):
        assert WeatherConfig().source == 'cumulusmx'

    def test_lat_lon_default_none(self):
        w = WeatherConfig()
        assert w.latitude is None
        assert w.longitude is None


class TestLoadSection:
    def test_known_keys_are_applied(self):
        data = {'audio': {'sample_rate': 44100, 'pulse_rate': 100}}
        cfg = _load_section(data, 'audio', AudioConfig)
        assert cfg.sample_rate == 44100
        assert cfg.pulse_rate == 100

    def test_unknown_keys_are_silently_ignored(self):
        data = {'audio': {'sample_rate': 44100, 'completely_unknown': 'surprise'}}
        cfg = _load_section(data, 'audio', AudioConfig)
        assert cfg.sample_rate == 44100

    def test_missing_section_returns_all_defaults(self):
        cfg = _load_section({}, 'audio', AudioConfig)
        assert cfg.sample_rate == 16000
        assert cfg.pulse_rate == 120

    def test_partial_section_leaves_remaining_at_defaults(self):
        data = {'station': {'callsign': 'W1AW'}}
        cfg = _load_section(data, 'station', StationConfig)
        assert cfg.callsign == 'W1AW'
        assert cfg.timezone == 'America/Los_Angeles'


class TestBuzzConfigDefaults:
    def test_default_instance_has_all_subsections(self):
        cfg = BuzzConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.station, StationConfig)
        assert isinstance(cfg.weather, WeatherConfig)
        assert isinstance(cfg.server, ServerConfig)


class TestBuzzConfigFromToml:
    def test_reads_audio_section(self, tmp_path):
        toml = tmp_path / 'config.toml'
        toml.write_bytes(b'[audio]\nsample_rate = 44100\n')
        cfg = BuzzConfig.from_toml(toml)
        assert cfg.audio.sample_rate == 44100

    def test_defaults_for_missing_sections(self, tmp_path):
        toml = tmp_path / 'config.toml'
        toml.write_bytes(b'')
        cfg = BuzzConfig.from_toml(toml)
        assert cfg.audio.sample_rate == 16000
        assert cfg.station.callsign == 'N0CALL'

    def test_all_sections_loaded(self, tmp_path):
        toml = tmp_path / 'config.toml'
        toml.write_bytes(b'''
[audio]
sample_rate = 44100
pulse_rate = 100

[station]
callsign = "W1AW"
timezone = "America/New_York"

[weather]
source = "openmeteo"
latitude = 41.7
longitude = -72.7

[server]
enabled = false
host = "example.com"
''')
        cfg = BuzzConfig.from_toml(toml)
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.pulse_rate == 100
        assert cfg.station.callsign == 'W1AW'
        assert cfg.station.timezone == 'America/New_York'
        assert cfg.weather.source == 'openmeteo'
        assert cfg.weather.latitude == pytest.approx(41.7)
        assert cfg.server.enabled is False
        assert cfg.server.host == 'example.com'

    def test_unknown_keys_ignored(self, tmp_path):
        toml = tmp_path / 'config.toml'
        toml.write_bytes(b'[audio]\nextra_unknown_field = "surprise"\n')
        cfg = BuzzConfig.from_toml(toml)
        assert cfg.audio.sample_rate == 16000


class TestExampleConfigMatchesTheDataclasses:
    """config.example.toml documents every option, in the right section, and no others.

    The example file is the only documentation of what may go in a config, and an
    option that never reaches it is one nobody can find.  Checking it against the
    dataclasses here means adding a field without documenting it fails the suite
    rather than going unnoticed until somebody asks why the option does nothing.
    """

    def _documented(self) -> dict[str, set[str]]:
        """Keys the example file documents, by section, commented-out ones included."""
        keys: dict[str, set[str]] = {}
        section = None
        for line in _EXAMPLE.read_text(encoding='utf-8').splitlines():
            text = line.strip().lstrip('#').strip()
            header = re.fullmatch(r'\[(\w+)\]', text)
            if header:
                section = header.group(1)
                keys.setdefault(section, set())
            elif section and (match := re.match(r'([a-z_]+)\s*=', text)):
                keys[section].add(match.group(1))
        return keys

    def _defined(self) -> dict[str, set[str]]:
        return {f.name: set(getattr(BuzzConfig(), f.name).__dataclass_fields__)
                for f in fields(BuzzConfig())}

    def test_every_section_is_documented(self):
        assert set(self._documented()) == set(self._defined())

    def test_every_option_is_documented(self):
        assert self._documented() == self._defined()

    def test_uncommented_values_are_the_defaults(self):
        """The example file says the values it shows are the defaults, so they must be.

        Only the settings that are truly site-specific are exempt - a hostname or
        a home directory cannot have a meaningful default, and the example gives a
        plausible one to edit.  Everything else drifts silently otherwise: a default
        changed in code leaves the example quietly documenting the old value.
        """
        placeholders = {('station', 'path'), ('weather', 'url'), ('server', 'host'),
                        ('server', 'username'), ('server', 'remote_path'),
                        ('server', 'key_path')}
        example, defaults = BuzzConfig.from_toml(_EXAMPLE), BuzzConfig()
        differing = {
            (section.name, option)
            for section in fields(BuzzConfig())
            for option in getattr(defaults, section.name).__dataclass_fields__
            if getattr(getattr(example, section.name), option)
            != getattr(getattr(defaults, section.name), option)
        }
        assert differing == placeholders


class TestRecordingConfigDefaults:
    def test_disabled_by_default(self):
        assert RecordingConfig().enabled is False

    def test_records_a_batch_of_events_by_default(self):
        assert RecordingConfig().max_events == 10

    def test_caps_recording_length_by_default(self):
        assert RecordingConfig().max_seconds == 120.0

    def test_has_a_silence_timeout(self):
        assert RecordingConfig().stop_after_seconds > 0


class TestRecordingDirectory:
    def test_defaults_under_the_station_path(self):
        station = StationConfig(path='/data')
        assert RecordingConfig().directory_path(station) == Path('/data/recordings')

    def test_explicit_directory_wins(self):
        station = StationConfig(path='/data')
        cfg = RecordingConfig(directory='/elsewhere/wavs')
        assert cfg.directory_path(station) == Path('/elsewhere/wavs')


class TestRecordingSectionLoading:
    def _load(self, tmp_path, body: bytes) -> BuzzConfig:
        toml = tmp_path / 'config.toml'
        toml.write_bytes(body)
        return BuzzConfig.from_toml(toml)

    def test_section_is_read(self, tmp_path):
        cfg = self._load(tmp_path, b'[recording]\nenabled = true\nmax_events = 5\n')
        assert (cfg.recording.enabled, cfg.recording.max_events) == (True, 5)

    def test_missing_section_uses_defaults(self, tmp_path):
        cfg = self._load(tmp_path, b'[audio]\nsample_rate = 8000\n')
        assert cfg.recording.enabled is False


class TestSampleRateAdmission:
    """The band the rest of the program can honestly work in, and the refusal outside it.

    Both bounds do work rather than merely looking tidy.  The floor is what makes the
    waterfall's bin count independent of the rate: below 8 kHz the Nyquist clamp in
    spectrum_geometry() starts biting and the window stops being 128 bins wide, which
    is the property buzz.render leans on when it declines to pad the frame.  The
    ceiling is the ring buffer, which is sized in seconds and so costs more memory as
    the rate climbs.
    """

    def test_the_rate_this_program_records_at_is_admitted(self):
        validate_sample_rate(16000, 'event.wav', 16000)

    def test_both_bounds_are_inclusive(self):
        """Stated as a test because the message quotes the bounds as a closed interval,
        and a message that disagrees with the code is worse than no message."""
        validate_sample_rate(MIN_SAMPLE_RATE, 'floor.wav', 16000)
        validate_sample_rate(MAX_SAMPLE_RATE, 'ceiling.wav', 16000)

    def test_the_rates_a_foreign_file_arrives_at_are_admitted(self):
        for rate in (11025, 22050, 44100):
            validate_sample_rate(rate, f'{rate}.wav', 16000)

    def test_below_the_floor_is_refused(self):
        with pytest.raises(ValueError):
            validate_sample_rate(MIN_SAMPLE_RATE - 1, 'slow.wav', 16000)

    def test_above_the_ceiling_is_refused(self):
        with pytest.raises(ValueError):
            validate_sample_rate(MAX_SAMPLE_RATE + 1, 'fast.wav', 16000)

    def test_the_refusal_names_the_file_and_its_rate(self):
        """Whoever hits this is holding a file somebody sent them and has no idea what
        is in it; the message is the only thing that will tell them."""
        with pytest.raises(ValueError, match=r'from-a-ham\.wav.*96000 Hz'):
            validate_sample_rate(96000, 'from-a-ham.wav', 16000)

    def test_the_refusal_states_the_condition_that_failed(self):
        with pytest.raises(ValueError,
                           match=f'{MIN_SAMPLE_RATE} <= sample rate <= {MAX_SAMPLE_RATE}'):
            validate_sample_rate(96000, 'from-a-ham.wav', 16000)

    def test_the_remedy_suggests_this_stations_own_rate(self):
        """Not a hard-coded 16000: a station configured for 48 kHz should be told to
        resample to 48 kHz, or the advice sends it to a rate nothing else here uses."""
        with pytest.raises(ValueError, match='resampling it to 48000 Hz'):
            validate_sample_rate(4000, 'slow.wav', 48000)

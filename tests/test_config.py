"""Tests for configuration dataclasses, TOML loader, and derived properties."""

import pytest

from buzz.config import (
    AudioConfig, BuzzConfig, ServerConfig, StationConfig, WeatherConfig, _load_section,
)


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

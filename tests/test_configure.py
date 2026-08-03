"""Tests for configure.py: _section_dict filtering and main() device-save flow."""

import tomllib
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

import configure
from buzz.config import AudioConfig, RecordingConfig, ServerConfig, StationConfig, WeatherConfig
from configure import _section_dict, main


class TestSectionDict:
    def test_includes_set_values(self):
        cfg = AudioConfig(sample_rate=44100, pulse_rate=100)
        d = _section_dict(cfg)
        assert d['sample_rate'] == 44100
        assert d['pulse_rate'] == 100

    def test_excludes_none_values(self):
        cfg = WeatherConfig(latitude=None, longitude=None)
        d = _section_dict(cfg)
        assert 'latitude' not in d
        assert 'longitude' not in d

    def test_includes_non_none_optional_fields(self):
        cfg = WeatherConfig(source='openmeteo', latitude=37.8, longitude=-122.4)
        d = _section_dict(cfg)
        assert d['latitude'] == pytest.approx(37.8)
        assert d['longitude'] == pytest.approx(-122.4)

    def test_includes_false_boolean(self):
        cfg = ServerConfig(enabled=False)
        d = _section_dict(cfg)
        assert d['enabled'] is False

    def test_includes_zero_numeric(self):
        """Zero is a real setting, not an absent one: max_events = 0 means "record
        every event". Filtering it the way None is filtered would silently restore
        the default of 10."""
        cfg = RecordingConfig(max_events=0)
        d = _section_dict(cfg)
        assert d['max_events'] == 0

    def test_a_section_with_no_none_fields_is_unfiltered(self):
        """StationConfig has no Optional fields, so nothing here should be dropped -
        the complement of test_excludes_none_values, which checks the field that is."""
        station = StationConfig()
        assert _section_dict(station) == asdict(station)


class TestMainNoDeviceSelected:
    def test_no_config_written_when_none_returned(self, tmp_path, capsys):
        config_path = tmp_path / 'config.toml'
        with patch('configure.select_device', return_value=None), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        assert not config_path.exists()

    def test_prints_no_device_message(self, tmp_path, capsys):
        config_path = tmp_path / 'config.toml'
        with patch('configure.select_device', return_value=None), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        assert 'No device selected' in capsys.readouterr().out


class TestMainDeviceSelected:
    def test_config_file_created(self, tmp_path):
        config_path = tmp_path / 'config.toml'
        device = {'name': 'Test Mic', 'hostapi': 0}
        hostapis = [{'name': 'Windows DirectSound'}]
        with patch('configure.select_device', return_value=2), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        assert config_path.exists()

    def test_the_index_is_not_written_to_the_config(self, tmp_path):
        """The chosen index is used to look the device up here and then discarded.
        Writing it would create a second, staler answer to "which device": indices
        move when Windows reassigns audio hardware, and the name does not."""
        config_path = tmp_path / 'config.toml'
        device = {'name': 'Test Mic', 'hostapi': 0}
        hostapis = [{'name': 'Windows DirectSound'}]
        with patch('configure.select_device', return_value=5), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
        assert 'device_index' not in data['audio']

    def test_device_name_saved_to_toml(self, tmp_path):
        config_path = tmp_path / 'config.toml'
        device = {'name': 'USB Audio', 'hostapi': 0}
        hostapis = [{'name': 'WASAPI'}]
        with patch('configure.select_device', return_value=3), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
        assert data['audio']['input_device_name'] == 'USB Audio, WASAPI'

    def test_prints_confirmation(self, tmp_path, capsys):
        config_path = tmp_path / 'config.toml'
        device = {'name': 'Test Mic', 'hostapi': 0}
        hostapis = [{'name': 'DirectSound'}]
        with patch('configure.select_device', return_value=2), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path):
            main()
        output = capsys.readouterr().out
        assert 'Device set to' in output

    def test_loads_existing_config_when_present(self, tmp_path):
        from buzz.config import BuzzConfig
        config_path = tmp_path / 'config.toml'
        config_path.write_bytes(b'[station]\ncallsign = "W6TST"\n')
        device = {'name': 'Mic', 'hostapi': 0}
        hostapis = [{'name': 'DirectSound'}]
        expected_cfg = BuzzConfig()
        expected_cfg.station.callsign = 'W6TST'
        # from_toml's default param is bound at definition time, so patch the method itself
        with patch('configure.select_device', return_value=1), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path), \
             patch.object(BuzzConfig, 'from_toml', return_value=expected_cfg):
            main()
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
        assert data['station']['callsign'] == 'W6TST'

    def test_recording_and_render_settings_survive_a_save(self, tmp_path):
        """BuzzConfig grew two sections after this flow was written, and main() was
        never updated to save them: every run of configure.py silently reverted a
        customised event budget or ffmpeg_path to its default, with nothing to say so.
        Loading a full six-section config and saving it back has to round-trip all of
        it, not just the four sections this bug's fix touched.
        """
        from buzz.config import BuzzConfig
        config_path = tmp_path / 'config.toml'
        config_path.write_bytes(b'')  # only its existence matters; from_toml is patched below
        device = {'name': 'Mic', 'hostapi': 0}
        hostapis = [{'name': 'DirectSound'}]
        loaded_cfg = BuzzConfig()
        loaded_cfg.recording.max_events = 42
        loaded_cfg.render.ffmpeg_path = 'C:/custom/ffmpeg'
        # from_toml's default param is bound at definition time, so patch the method
        # itself rather than relying on patching CONFIG_PATH to redirect the load too.
        with patch('configure.select_device', return_value=1), \
             patch('configure.sd.query_devices', return_value=device), \
             patch('configure.sd.query_hostapis', return_value=hostapis), \
             patch('configure.CONFIG_PATH', config_path), \
             patch.object(BuzzConfig, 'from_toml', return_value=loaded_cfg):
            main()
        with open(config_path, 'rb') as f:
            data = tomllib.load(f)
        assert data['recording']['max_events'] == 42
        assert data['render']['ffmpeg_path'] == 'C:/custom/ffmpeg'

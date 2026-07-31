"""Tests for buzz.main: configure_logging(), weather client factory, playback wiring,
and headless wait."""
import logging
import time
import wave
from unittest.mock import MagicMock, patch

import buzz.main as main_module
import numpy as np
import pytest
from buzz import wavmeta
from buzz.config import BuzzConfig
from buzz.main import (
    _start_collector, _start_playback, _wait_until_interrupted, configure_logging,
    make_weather_client, open_playback_pipeline,
)
from buzz.weather import CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient


@pytest.fixture(autouse=True)
def restore_logging():
    """Restore global logging state after each test so configure_logging() side-effects
    don't leak into other test modules (e.g. breaking caplog-based sampler tests)."""
    root = logging.getLogger()
    buzz = logging.getLogger('buzz')
    root_level = root.level
    buzz_level, buzz_handlers, buzz_propagate = buzz.level, buzz.handlers[:], buzz.propagate
    yield
    root.setLevel(root_level)
    buzz.setLevel(buzz_level)
    buzz.handlers[:] = buzz_handlers
    buzz.propagate = buzz_propagate


class TestConfigureLogging:
    def test_runs_without_error(self):
        configure_logging()

    def test_buzz_logger_level_is_info(self):
        configure_logging()
        assert logging.getLogger('buzz').level == logging.INFO

    def test_buzz_logger_has_console_handler(self):
        configure_logging()
        logger = logging.getLogger('buzz')
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_root_logger_silenced(self):
        configure_logging()
        assert logging.getLogger().level == logging.CRITICAL

    def test_buzz_logger_does_not_propagate(self):
        configure_logging()
        assert logging.getLogger('buzz').propagate is False


class TestModuleConstants:
    def test_root_package_is_buzz(self):
        assert main_module.ROOT_PACKAGE == 'buzz'


class TestMakeWeatherClient:
    def _config(self, source: str) -> BuzzConfig:
        cfg = BuzzConfig()
        cfg.weather.source = source
        cfg.weather.url = 'http://weather.local/realtime.json'
        cfg.weather.latitude = 37.8
        cfg.weather.longitude = -122.4
        return cfg

    def test_openmeteo(self):
        client = make_weather_client(self._config('openmeteo'))
        assert isinstance(client, OpenMeteoWeatherClient)

    def test_cumulusmx(self):
        client = make_weather_client(self._config('cumulusmx'))
        assert isinstance(client, CumulusMXWeatherClient)

    def test_none(self):
        client = make_weather_client(self._config('none'))
        assert isinstance(client, NullWeatherClient)

    def test_unknown_source_returns_null_client(self):
        client = make_weather_client(self._config('wunderground'))
        assert isinstance(client, NullWeatherClient)

    def test_unknown_source_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz'):
            make_weather_client(self._config('wunderground'))
        assert 'Unknown weather source' in caplog.text

    def test_none_source_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz'):
            make_weather_client(self._config('none'))
        assert caplog.text == ''

    def test_openmeteo_without_coordinates_returns_null_client(self):
        cfg = self._config('openmeteo')
        cfg.weather.latitude = None
        client = make_weather_client(cfg)
        assert isinstance(client, NullWeatherClient)

    def test_openmeteo_without_coordinates_logs_warning(self, caplog):
        cfg = self._config('openmeteo')
        cfg.weather.longitude = None
        with caplog.at_level(logging.WARNING, logger='buzz'):
            make_weather_client(cfg)
        assert 'latitude/longitude' in caplog.text

    def test_cumulusmx_without_url_returns_null_client(self):
        cfg = self._config('cumulusmx')
        cfg.weather.url = ''
        client = make_weather_client(cfg)
        assert isinstance(client, NullWeatherClient)

    def test_cumulusmx_without_url_logs_warning(self, caplog):
        cfg = self._config('cumulusmx')
        cfg.weather.url = ''
        with caplog.at_level(logging.WARNING, logger='buzz'):
            make_weather_client(cfg)
        assert 'url is not set' in caplog.text


class TestWaitUntilInterrupted:
    def test_closes_pipeline_on_keyboard_interrupt(self):
        pipeline = MagicMock()
        analyzer = MagicMock()
        with patch('buzz.main.threading.Event') as mock_event:
            mock_event.return_value.wait.side_effect = KeyboardInterrupt
            _wait_until_interrupted(pipeline, analyzer)
        pipeline.close.assert_called_once()

    def test_stops_analyzer_on_keyboard_interrupt(self):
        pipeline = MagicMock()
        analyzer = MagicMock()
        with patch('buzz.main.threading.Event') as mock_event:
            mock_event.return_value.wait.side_effect = KeyboardInterrupt
            _wait_until_interrupted(pipeline, analyzer)
        analyzer.stop.assert_called_once()

    def test_analyzer_stopped_before_pipeline_closed(self):
        """Mirrors MainWindow.closeEvent()'s order: the analyzer thread must be
        told to stop before its audio pipeline is closed out from under it."""
        calls = []
        pipeline = MagicMock()
        pipeline.close.side_effect = lambda: calls.append('pipeline.close')
        analyzer = MagicMock()
        analyzer.stop.side_effect = lambda: calls.append('analyzer.stop')
        with patch('buzz.main.threading.Event') as mock_event:
            mock_event.return_value.wait.side_effect = KeyboardInterrupt
            _wait_until_interrupted(pipeline, analyzer)
        assert calls == ['analyzer.stop', 'pipeline.close']

    def test_recorder_stopped_while_its_audio_source_is_still_open(self):
        """A recording in progress has to be closed before the pipeline feeding it."""
        calls = []
        pipeline = MagicMock()
        pipeline.close.side_effect = lambda: calls.append('pipeline.close')
        analyzer = MagicMock()
        analyzer.stop.side_effect = lambda: calls.append('analyzer.stop')
        recorder = MagicMock()
        recorder.stop.side_effect = lambda: calls.append('recorder.stop')
        with patch('buzz.main.threading.Event') as mock_event:
            mock_event.return_value.wait.side_effect = KeyboardInterrupt
            _wait_until_interrupted(pipeline, analyzer, recorder)
        assert calls == ['analyzer.stop', 'recorder.stop', 'pipeline.close']


class TestPlaybackWritesNothing:
    """Replaying a recording must not write anything durable, or record.

    Every one of these is a property of main()'s wiring rather than of any single
    component: the recorder and the collector are simply never built on the playback
    path, so there is nothing that could be triggered into life later.
    """

    def _run_main(self, argv, tmp_path):
        with patch('sys.argv', ['buzz', '--headless', *argv]), \
             patch('buzz.main.CONFIG_PATH', tmp_path / 'no-such-config.toml'), \
             patch('buzz.main.configure_logging'), \
             patch('buzz.main.open_playback_pipeline') as playback, \
             patch('buzz.main.AudioSampler') as sampler, \
             patch('buzz.main.ContinuousAnalyzer'), \
             patch('buzz.main.EventRecorder') as recorder, \
             patch('buzz.main._start_collector') as collector, \
             patch('buzz.main._wait_until_interrupted'):
            main_module.main()
        return playback, sampler, recorder, collector

    def test_playback_builds_no_recorder(self, tmp_path):
        _, _, recorder, _ = self._run_main(['--playback', 'event.wav'], tmp_path)
        recorder.assert_not_called()

    def test_playback_starts_no_collector(self, tmp_path):
        _, _, _, collector = self._run_main(['--playback', 'event.wav'], tmp_path)
        collector.assert_not_called()

    def test_playback_opens_no_audio_device(self, tmp_path):
        _, sampler, _, _ = self._run_main(['--playback', 'event.wav'], tmp_path)
        sampler.assert_not_called()

    def test_enable_recording_does_not_override_playback(self, tmp_path):
        _, _, recorder, _ = self._run_main(
            ['--playback', 'event.wav', '--enable-recording'], tmp_path)
        recorder.assert_not_called()

    def test_enable_recording_during_playback_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._run_main(['--playback', 'event.wav', '--enable-recording'], tmp_path)
        assert 'ignored during playback' in caplog.text

    def test_live_run_builds_a_recorder(self, tmp_path):
        _, _, recorder, _ = self._run_main([], tmp_path)
        recorder.assert_called_once()

    def test_live_run_starts_the_collector(self, tmp_path):
        _, _, _, collector = self._run_main([], tmp_path)
        collector.assert_called_once()

    def test_live_run_reads_no_playback_file(self, tmp_path):
        playback, _, _, _ = self._run_main([], tmp_path)
        playback.assert_not_called()


class TestPlaybackStartsWithTheDisplay:
    def _write_wav(self, path, sample_rate=16000, n=1024):
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(np.zeros(n, dtype='<i2').tobytes())
        return path

    def test_opening_a_file_does_not_start_it(self, tmp_path):
        """Audio started at open time plays before the window exists, and breaks up
        while widget construction holds the GIL away from the feeder."""
        self._write_wav(tmp_path / 'event.wav')
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav') as pipeline:
            time.sleep(0.05)
            assert pipeline.total_samples == 0

    def test_headless_starts_playback(self, tmp_path):
        pipeline = MagicMock()
        _start_playback(pipeline, 'event.wav')
        pipeline.start.assert_called_once()

    def test_live_audio_has_nothing_to_start(self, tmp_path):
        pipeline = MagicMock()
        _start_playback(pipeline, None)
        pipeline.start.assert_not_called()


class TestStartCollector:
    def _start(self, cfg):
        """Run _start_collector with everything it builds stubbed out."""
        with patch('buzz.main.CsvStore'), patch('buzz.main.Plotter'), \
             patch('buzz.main.Publisher') as publisher, \
             patch('buzz.main.Collector') as collector, \
             patch('buzz.main.threading.Thread') as thread:
            _start_collector(cfg, MagicMock())
        return publisher, collector, thread

    def test_collection_runs_on_a_daemon_thread(self):
        _, _, thread = self._start(BuzzConfig())
        assert thread.call_args.kwargs['daemon'] is True

    def test_thread_is_started(self):
        _, _, thread = self._start(BuzzConfig())
        thread.return_value.start.assert_called_once()

    def test_no_publisher_when_uploads_are_disabled(self):
        cfg = BuzzConfig()
        cfg.server.enabled = False
        _, collector, _ = self._start(cfg)
        assert collector.call_args.args[-1] is None

    def test_publisher_when_uploads_are_enabled(self):
        cfg = BuzzConfig()
        cfg.server.enabled = True
        publisher, collector, _ = self._start(cfg)
        assert collector.call_args.args[-1] is publisher.return_value


class TestOpenPlaybackPipeline:
    def _write_wav(self, path, sample_rate=16000, n=1024):
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(np.zeros(n, dtype='<i2').tobytes())
        return path

    def test_bare_filename_is_found_in_the_recording_directory(self, tmp_path):
        self._write_wav(tmp_path / 'event.wav')
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav') as pipeline:
            assert pipeline.path == tmp_path / 'event.wav'

    def test_full_path_is_used_as_given(self, tmp_path):
        path = self._write_wav(tmp_path / 'event.wav')
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path / 'somewhere-else')
        with open_playback_pipeline(cfg, str(path)) as pipeline:
            assert pipeline.path == path

    def test_config_sample_rate_follows_the_file(self, tmp_path):
        self._write_wav(tmp_path / 'event.wav', sample_rate=8000)
        cfg = BuzzConfig()
        cfg.audio.sample_rate = 16000
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav'):
            assert cfg.audio.sample_rate == 8000

    def test_mismatched_sample_rate_warns(self, tmp_path, caplog):
        self._write_wav(tmp_path / 'event.wav', sample_rate=8000)
        cfg = BuzzConfig()
        cfg.audio.sample_rate = 16000
        cfg.recording.directory = str(tmp_path)
        with caplog.at_level(logging.WARNING, logger='buzz'):
            with open_playback_pipeline(cfg, 'event.wav'):
                pass
        assert 'sample rate 8000' in caplog.text

    def test_missing_file_exits_with_a_message(self, tmp_path):
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path)
        with pytest.raises(SystemExit, match='Cannot play back'):
            open_playback_pipeline(cfg, 'nope.wav')

    def test_unplayable_file_exits_with_a_message(self, tmp_path):
        (tmp_path / 'event.wav').write_bytes(b'not a wav file at all')
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path)
        with pytest.raises(SystemExit, match='Cannot play back'):
            open_playback_pipeline(cfg, 'event.wav')

    def test_wrong_sample_width_exits_with_a_message(self, tmp_path):
        with wave.open(str(tmp_path / 'event.wav'), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x01')
        cfg = BuzzConfig()
        cfg.recording.directory = str(tmp_path)
        with pytest.raises(SystemExit, match='16-bit'):
            open_playback_pipeline(cfg, 'event.wav')

    def test_matching_sample_rate_does_not_warn(self, tmp_path, caplog):
        self._write_wav(tmp_path / 'event.wav', sample_rate=16000)
        cfg = BuzzConfig()
        cfg.audio.sample_rate = 16000
        cfg.recording.directory = str(tmp_path)
        with caplog.at_level(logging.WARNING, logger='buzz'):
            with open_playback_pipeline(cfg, 'event.wav'):
                pass
        # This untagged file does warn about its missing metadata; the sample rate,
        # which comes from the format header and matches, is not what it warns about.
        assert 'sample rate' not in caplog.text


class TestPlaybackAdoptsRecordedSettings:
    """A recording measures the same wherever it is replayed: the settings that
    decide what the numbers mean travel with the file, not with the machine."""

    def _write_tagged(self, tmp_path, **settings):
        path = tmp_path / 'event.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(1024, dtype='<i2').tobytes())
        if settings:
            wavmeta.append_metadata(path, {'ICMT': wavmeta.format_settings(settings)})
        return path

    def _play(self, tmp_path, cfg, **settings):
        self._write_tagged(tmp_path, **settings)
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav'):
            pass
        return cfg

    def test_pulse_rate_comes_from_the_file(self, tmp_path):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 120
        assert self._play(tmp_path, cfg, pulse_rate=100).audio.pulse_rate == 100

    def test_level_calibration_comes_from_the_file(self, tmp_path):
        cfg = BuzzConfig()
        cfg.station.audio_rf_conversion_db = -32.0
        cfg = self._play(tmp_path, cfg, audio_rf_conversion_db=-18.5)
        assert cfg.station.audio_rf_conversion_db == pytest.approx(-18.5)

    def test_mismatched_pulse_rate_warns(self, tmp_path, caplog):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 120
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, cfg, pulse_rate=100)
        assert 'pulse rate 100' in caplog.text

    def test_matching_settings_do_not_warn(self, tmp_path, caplog):
        cfg = BuzzConfig()
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, cfg,
                       pulse_rate=cfg.audio.pulse_rate,
                       audio_rf_conversion_db=cfg.station.audio_rf_conversion_db)
        assert caplog.text == ''

    def test_untagged_file_keeps_the_local_config(self, tmp_path):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 120
        assert self._play(tmp_path, cfg).audio.pulse_rate == 120

    def test_unparsable_setting_keeps_the_local_config(self, tmp_path):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 120
        assert self._play(tmp_path, cfg, pulse_rate='ninety').audio.pulse_rate == 120

    def test_untagged_file_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig())
        assert 'does not record its pulse rate or level calibration' in caplog.text

    def test_warning_states_what_is_being_assumed(self, tmp_path, caplog):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 100
        cfg.station.audio_rf_conversion_db = -25.0
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, cfg)
        assert '100 Hz' in caplog.text and '-25.0 dB' in caplog.text

    def test_partially_tagged_file_warns_about_the_missing_setting(self, tmp_path, caplog):
        cfg = BuzzConfig()
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, cfg, pulse_rate=cfg.audio.pulse_rate)
        assert 'does not record its level calibration' in caplog.text

    def test_unreadable_file_still_plays(self, tmp_path, caplog):
        """A .wav from anywhere else is playable; it just cannot be trusted."""
        cfg = BuzzConfig()
        with caplog.at_level(logging.WARNING, logger='buzz'):
            played = self._play(tmp_path, cfg, pulse_rate='ninety')
        assert played.audio.pulse_rate == BuzzConfig().audio.pulse_rate

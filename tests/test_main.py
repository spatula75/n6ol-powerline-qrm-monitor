"""Tests for buzz.main: configure_logging(), weather client factory, playback wiring,
and headless wait."""
import argparse
import logging
import sys
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import buzz.main as main_module
import numpy as np
import pytest
from buzz import ffmpeg as ffmpeg_module
from buzz import loudness as loudness_module
from buzz import main as main_module
from buzz import sdr as sdr_module
from buzz import wavmeta
from buzz.analyzer import ContinuousAnalyzer
from buzz.collector import Collector
from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore
from buzz.ffmpeg import find_ffmpeg
from buzz.loudness import resolve_gain
from buzz.main import (
    _start_collector, _start_playback, _wait_until_interrupted, build_recording,
    check_playback_source, configure_logging, make_weather_client, open_live_source,
    open_playback_pipeline,
)
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler, RingBufferPipeline
from buzz.sdr import open_device
from buzz.weather import CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient
from tests.patching import patch_in


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
             patch_in(main_module, configure_logging), \
             patch_in(main_module, check_playback_source), \
             patch_in(main_module, open_playback_pipeline) as playback, \
             patch_in(main_module, AudioSampler) as sampler, \
             patch_in(main_module, ContinuousAnalyzer), \
             patch_in(main_module, build_recording) as recorder, \
             patch_in(main_module, _start_collector) as collector, \
             patch_in(main_module, _wait_until_interrupted):
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
        with patch_in(main_module, CsvStore), patch_in(main_module, Plotter), \
             patch_in(main_module, Publisher) as publisher, \
             patch_in(main_module, Collector) as collector, \
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
        # Long enough to get past the length check, so that the 8-bit sample width is
        # what this file is refused for.
        with wave.open(str(tmp_path / 'event.wav'), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(16000)
            wav.writeframes(b'\x01' * RingBufferPipeline.CHUNK_SIZE)
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

    def _play(self, tmp_path, cfg, rf_conversion_db=None, **settings):
        self._write_tagged(tmp_path, **settings)
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav',
                                    rf_conversion_db=rf_conversion_db):
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
        assert cfg.level_offset_db == pytest.approx(-18.5)
        assert cfg.station.audio_rf_conversion_db == -32.0, (
            "a recording's own figure must not overwrite this station's setting")

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

    def test_untagged_file_warns_about_both_settings(self, tmp_path, caplog):
        """Two warnings rather than one: a wrong pulse rate means nothing locks, a
        wrong calibration means everything locks and every level is wrong. Different
        consequences, different remedies."""
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig())
        assert 'does not record its pulse rate' in caplog.text
        assert 'does not record its level calibration' in caplog.text

    def test_warning_states_what_is_being_assumed(self, tmp_path, caplog):
        cfg = BuzzConfig()
        cfg.audio.pulse_rate = 100
        cfg.station.audio_rf_conversion_db = -25.0
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, cfg)
        assert '100 pps' in caplog.text and '-25.0 dB' in caplog.text

    def test_the_calibration_warning_names_its_remedy(self, tmp_path, caplog):
        """Telling somebody a reading may be wrong is only useful alongside how to
        put it right."""
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig())
        assert '--audio-rf-conversion-db' in caplog.text

    def test_a_supplied_calibration_silences_that_warning(self, tmp_path, caplog):
        """Advising the flag to somebody who just passed it would be noise."""
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig(), rf_conversion_db=-28.5)
        assert 'does not record its level calibration' not in caplog.text
        assert 'does not record its pulse rate' in caplog.text

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


class TestGainArgument:
    """--playback-gain takes a number of dB or the word "auto"."""

    def test_a_number_is_a_number(self):
        assert main_module._gain_argument('12') == 12.0
        assert main_module._gain_argument('-6.5') == -6.5

    def test_auto_is_recognised_however_it_is_typed(self):
        for spelling in ('auto', 'AUTO', ' Auto '):
            assert main_module._gain_argument(spelling) == main_module.AUTO_GAIN

    def test_anything_else_explains_both_forms(self):
        """argparse prints this straight to the operator, so it has to name what is
        acceptable rather than just rejecting what was typed."""
        with pytest.raises(argparse.ArgumentTypeError, match='neither a number'):
            main_module._gain_argument('loud')

    def test_zero_is_a_number_and_not_a_missing_value(self):
        """The default is None so that "said nothing" can be told from "said zero" --
        which is how --render knows whether to measure."""
        assert main_module._gain_argument('0') == 0.0


class TestRenderOutputCheck:

    def test_an_existing_output_is_refused(self, tmp_path):
        output = tmp_path / 'demo.mp4'
        output.write_bytes(b'')
        with pytest.raises(RuntimeError, match='already exists'):
            main_module._check_render_output(output)

    def test_the_refusal_says_why_and_what_to_do(self, tmp_path):
        """A render costs as long as the recording it replays, so silently destroying
        one would be expensive; the operator needs to know that is deliberate."""
        output = tmp_path / 'demo.mp4'
        output.write_bytes(b'')
        with pytest.raises(RuntimeError, match='never overwrite'):
            main_module._check_render_output(output)

    def test_a_free_path_passes_silently(self, tmp_path):
        main_module._check_render_output(tmp_path / 'not-there.mp4')


class TestDescribeDuration:
    """How long a render will take, said so a person can act on it."""

    def test_short_renders_are_seconds(self):
        assert main_module._describe_duration(16.6) == '17 s'
        assert main_module._describe_duration(45.0) == '45 s'

    def test_long_renders_read_better_in_minutes(self):
        """"130 s" is arithmetic; "2 min 10 s" is a decision about whether to wait."""
        assert main_module._describe_duration(130.0) == '2 min 10 s'

    def test_a_whole_number_of_minutes_drops_the_seconds(self):
        assert main_module._describe_duration(180.0) == '3 min'

    def test_the_boundary_does_not_produce_nonsense(self):
        assert main_module._describe_duration(119.0) == '119 s'
        assert main_module._describe_duration(120.0) == '2 min'


class TestTheEntryPointLogsAtAll:
    """main.py is the one module that must not name its logger from __name__.

    Run as `python -m buzz.main` its __name__ is '__main__', which sits outside the
    `buzz` hierarchy that configure_logging() attaches the console handler to - so
    every message from this file went nowhere in the invocation the README documents.
    """

    def test_the_logger_is_inside_the_configured_hierarchy(self):
        assert main_module.logger.name == f'{main_module.ROOT_PACKAGE}.main', (
            f'main.py logs to {main_module.logger.name!r}. If that came from __name__ '
            'it will be \'__main__\' when run as a module, and nothing it logs will '
            'be printed.')

    def test_a_main_logger_would_indeed_have_gone_nowhere(self, capsys):
        """The trap itself, so the reason the name is hard-coded stays visible."""
        main_module.configure_logging()
        logging.getLogger('__main__').warning('this must not appear')
        logging.getLogger(f'{main_module.ROOT_PACKAGE}.main').warning('this must appear')
        captured = capsys.readouterr()
        assert 'this must not appear' not in (captured.out + captured.err)
        assert 'this must appear' in (captured.out + captured.err)


class TestSuppliedCalibration:
    """--audio-rf-conversion-db, for a .wav that arrived from another operator.

    The pulse rate and the calibration live in metadata only this program writes, so a
    file from elsewhere is analyzed against this station's figures. Whether it locks
    and what the burst looks like survive that; the dBm and S-unit readings do not.
    This is how somebody who knows the sending station's calibration can supply it.
    """

    def _play(self, tmp_path, cfg, rf_conversion_db=None, **settings):
        path = tmp_path / 'event.wav'
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        if settings:
            wavmeta.append_metadata(path, {'ICMT': wavmeta.format_settings(settings)}, {})
        cfg.recording.directory = str(tmp_path)
        with open_playback_pipeline(cfg, 'event.wav',
                                    rf_conversion_db=rf_conversion_db):
            pass
        return cfg

    def test_it_is_used_when_the_file_says_nothing(self, tmp_path):
        cfg = self._play(tmp_path, BuzzConfig(), rf_conversion_db=-28.5)
        assert cfg.level_offset_db == -28.5

    def test_the_config_stands_when_nothing_is_supplied(self, tmp_path):
        cfg = self._play(tmp_path, BuzzConfig())
        assert cfg.level_offset_db == BuzzConfig().station.audio_rf_conversion_db

    def test_it_overrides_a_figure_the_recording_carries(self, tmp_path):
        """An explicit flag is the only value anybody deliberately supplied, so it
        wins -- but the recording's own is normally the right one, being the receiver
        that made it, so the override is worth saying out loud."""
        cfg = self._play(tmp_path, BuzzConfig(), rf_conversion_db=-28.5,
                         audio_rf_conversion_db=-32.0)
        assert cfg.level_offset_db == -28.5

    def test_overriding_the_recording_says_so(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig(), rf_conversion_db=-28.5,
                       audio_rf_conversion_db=-32.0)
        assert 'records a calibration of -32.0' in caplog.text
        assert '-28.5' in caplog.text

    def test_agreeing_with_the_recording_is_not_an_override(self, tmp_path, caplog):
        """Supplying the figure the file already carries is not a disagreement, so it
        should not be reported as one."""
        with caplog.at_level(logging.WARNING, logger='buzz'):
            self._play(tmp_path, BuzzConfig(), rf_conversion_db=-32.0,
                       audio_rf_conversion_db=-32.0)
        assert 'records a calibration of' not in caplog.text


class TestResolveGain:
    """Which gain a run ends up with, and when measuring is worth the wait.

    The distinction the branching exists for: a render is a file somebody else will
    watch, so it is measured by default; watching a replay is the operator listening
    live with the volume control to hand, so it is not.  An explicit figure -- zero
    included -- beats both, which is why the flag defaults to None rather than 0.0.
    """

    def _args(self, playback_gain=None, render=None) -> argparse.Namespace:
        return argparse.Namespace(playback_gain=playback_gain, render=render)

    def test_watching_without_a_figure_applies_none(self):
        assert main_module._resolve_gain(
            self._args(), BuzzConfig(), Path('event.wav')) == 0.0

    def test_a_figure_is_used_as_given(self):
        assert main_module._resolve_gain(
            self._args(playback_gain=12.5), BuzzConfig(), Path('event.wav')) == 12.5

    def test_an_explicit_zero_overrides_the_render_default(self):
        """The case that needs None to exist at all: "--playback-gain 0 --render" has
        to mean "leave it alone", not "you said nothing, so measure it"."""
        with patch_in(loudness_module, resolve_gain) as measured:
            gain = main_module._resolve_gain(
                self._args(playback_gain=0.0, render='out.mp4'),
                BuzzConfig(), Path('event.wav'))
        assert gain == 0.0
        assert not measured.called, (
            'A gain was given explicitly, so nothing should have been measured -- the '
            'probe reads the whole recording and the operator did not ask for it.')

    def test_rendering_without_a_figure_measures(self):
        config = BuzzConfig()
        with patch_in(ffmpeg_module, find_ffmpeg, return_value='/usr/bin/ffmpeg'), \
                patch_in(loudness_module, resolve_gain, return_value=19.0) as measured:
            gain = main_module._resolve_gain(
                self._args(render='out.mp4'), config, Path('event.wav'))
        assert gain == 19.0
        assert measured.call_args.args == (Path('event.wav'), '/usr/bin/ffmpeg')

    def test_auto_is_honoured_without_a_render(self):
        """--playback-gain auto is allowed on its own; it just is not the default."""
        with patch_in(ffmpeg_module, find_ffmpeg, return_value='/usr/bin/ffmpeg'), \
                patch_in(loudness_module, resolve_gain, return_value=16.4):
            gain = main_module._resolve_gain(
                self._args(playback_gain=main_module.AUTO_GAIN),
                BuzzConfig(), Path('event.wav'))
        assert gain == 16.4

    def test_the_configured_ffmpeg_path_is_offered_to_the_search(self):
        config = BuzzConfig()
        config.render.ffmpeg_path = 'C:/ffmpeg/bin'
        with patch_in(ffmpeg_module, find_ffmpeg, return_value='C:/ffmpeg/bin/ffmpeg.exe') as found, \
                patch_in(loudness_module, resolve_gain, return_value=1.0):
            main_module._resolve_gain(self._args(render='out.mp4'), config,
                                      Path('event.wav'))
        assert found.call_args.args == ('C:/ffmpeg/bin',)

    def test_an_unset_path_searches_only_the_path(self):
        """Empty means "look on PATH", and find_ffmpeg takes None for that -- passing
        the empty string through would have it check a directory named ''."""
        with patch_in(ffmpeg_module, find_ffmpeg, return_value='/usr/bin/ffmpeg') as found, \
                patch_in(loudness_module, resolve_gain, return_value=1.0):
            main_module._resolve_gain(self._args(render='out.mp4'), BuzzConfig(),
                                      Path('event.wav'))
        assert found.call_args.args == (None,)


class TestCheckPlaybackSource:
    """The header-only refusal that runs before anything expensive.

    Separate from open_playback_pipeline so that main() can run it first: the loudness
    probe reads the whole recording, and being turned away after waiting for that is a
    poor way to find out the file was never going to work.
    """

    def _wav(self, path, sample_rate) -> Path:
        # A whole chunk, because a file shorter than one is refused for that reason
        # instead, and these tests are about the sample rate.  A real recording is
        # thousands of chunks; one is the smallest thing that is not a special case.
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b'\x00\x00' * RingBufferPipeline.CHUNK_SIZE)
        return path

    def test_a_rate_in_the_band_passes(self, tmp_path):
        main_module.check_playback_source(
            self._wav(tmp_path / 'ok.wav', 16000), BuzzConfig())

    def test_a_rate_outside_the_band_exits(self, tmp_path):
        """SystemExit rather than a traceback: being sent a 96 kHz file is an ordinary
        thing to happen, not a bug in the monitor."""
        with pytest.raises(SystemExit, match='96000 Hz'):
            main_module.check_playback_source(
                self._wav(tmp_path / 'fast.wav', 96000), BuzzConfig())

    def test_the_exit_message_suggests_this_stations_rate(self, tmp_path):
        config = BuzzConfig()
        config.audio.sample_rate = 48000
        with pytest.raises(SystemExit, match='resampling it to 48000 Hz'):
            main_module.check_playback_source(
                self._wav(tmp_path / 'slow.wav', 4000), config)

    def test_a_file_that_is_not_a_wav_exits_naming_it(self, tmp_path):
        """Reached before the pipeline is built, so this is the first thing to notice a
        mistyped or corrupt file and has to report it as well as the pipeline would."""
        broken = tmp_path / 'not-audio.wav'
        broken.write_bytes(b'this is not a RIFF file')
        with pytest.raises(SystemExit, match='Cannot play back'):
            main_module.check_playback_source(broken, BuzzConfig())

    def test_a_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main_module.check_playback_source(tmp_path / 'gone.wav', BuzzConfig())


class TestTheAudioSourceHasToBeOneThisProgramKnows:
    """_load_section copies whatever the TOML holds, with no check against the schema.

    The schema names the two values and the setup program enforces them, but [rtlsdr]
    has no setup screen yet, so that section reaches the file by hand and a neighboring
    typo in [audio] source reaches it the same way.  A branch that fell through to the
    sound card would then open the device named in input_device_name and log a day of
    whatever that input hears, which looks exactly like a quiet band.
    """

    @pytest.mark.parametrize('typo', ['sdr', 'RTL-SDR', 'rtl_sdr', 'RTLSDR', ''])
    def test_a_misspelled_source_is_refused_rather_than_assumed(self, typo):
        config = BuzzConfig()
        config.audio.source = typo

        with pytest.raises(RuntimeError, match='must be'):
            open_live_source(config)

    def test_the_message_names_both_values_that_work(self):
        config = BuzzConfig()
        config.audio.source = 'rtl-sdr'

        with pytest.raises(RuntimeError) as caught:
            open_live_source(config)

        message = str(caught.value)
        assert 'soundcard' in message and 'rtlsdr' in message, (
            'The message refuses the value without saying what would be accepted.  '
            'Whoever hit this is editing the file by hand and cannot see the schema.  '
            f'It said: {message!r}')

    def test_a_sound_card_source_still_opens_the_sound_card(self):
        """The guard has to let the ordinary case through, or it would read as working
        while refusing every station that never touched the setting.
        """
        config = BuzzConfig()
        with patch.object(main_module, 'AudioSampler') as sampler:
            assert open_live_source(config) is sampler.return_value.pipeline


class TestAReceiverThatWillNotOpenPrintsItsReason:
    """buzz.sdr composes messages for whoever is standing at the radio: which driver to
    install, what else is holding the device, which setting is wrong.

    They were then raised through a call site that caught nothing, so all of that
    arrived as a traceback with the explanation buried in its last line.  The playback
    branch four lines above has always logged and exited instead.
    """

    def run_main_and_capture_the_exit(self, failure, capsys):
        """Drive main() past the point where it opens the live source.

        The output is read off stderr rather than through caplog, because
        configure_logging() clears propagate on the buzz logger, so caplog's handler on
        the root logger never sees these records.  stderr is also where the operator
        reads them, which makes it the honest thing to assert on.
        """
        args = ['buzz', '--headless']
        with patch.object(main_module, 'open_live_source', side_effect=failure), \
                patch.object(sys, 'argv', args):
            with pytest.raises(SystemExit) as exited:
                main_module.main()
        return exited.value.code, capsys.readouterr().err

    def test_a_receiver_failure_is_printed_and_exits_two(self, capsys):
        message = ('Receiver 0 was found but no driver is bound to it.  Run Zadig as '
                   'administrator.')
        code, err = self.run_main_and_capture_the_exit(RuntimeError(message), capsys)

        assert code == 2, f'exited {code} rather than 2, the code playback already uses'
        assert 'Zadig' in err, (
            f'The receiver message did not reach the operator: {err!r}.  It names the '
            'one thing that fixes the commonest Windows failure, and a traceback puts '
            'it where nobody reads it.')
        assert 'Traceback' not in err

    def test_an_impossible_config_is_printed_and_exits_two(self, capsys):
        """IqToAudio._validate refuses an [rtlsdr] section with ValueError, and its
        wording is aimed at the same reader as the receiver messages.
        """
        message = ('An IQ rate of 2400000 Hz decimated by 16 gives 150000 Hz of audio.  '
                   'Set [rtlsdr] decimation between 50 and 300.')
        code, err = self.run_main_and_capture_the_exit(ValueError(message), capsys)

        assert code == 2
        assert 'decimation' in err, (
            f'The config message did not reach the operator: {err!r}.  ValueError has '
            'to be caught alongside RuntimeError, because that is how _validate refuses '
            'a section nobody can use.')


class TestOpeningAReceiverAsTheLiveSource:
    """The wiring between the three pieces the SDR path is built from.

    RtlSdrSource holds the hardware, IqToAudio holds the arithmetic, and
    RtlSdrPipeline joins them to the ring buffer.  open_live_source is the only place
    that knows how they fit together, and what it settles afterwards decides what every
    later measurement means: the audio rate everything downstream counts seconds by,
    and the dB offset every level is reported against.
    """

    def open_with_a_fake_receiver(self, config, device=None):
        """Run the real open_live_source with only the USB device replaced.

        Everything above the device is the production code, so the converter, the
        pipeline and the rate bookkeeping are all the real ones.
        """
        from tests.test_sdr_source import FakeDevice
        with patch_in(sdr_module, open_device, return_value=device or FakeDevice()):
            return open_live_source(config)

    def rtlsdr_config(self, **overrides):
        config = BuzzConfig()
        config.audio.source = 'rtlsdr'
        for name, value in overrides.items():
            setattr(config.rtlsdr, name, value)
        return config

    def test_the_audio_rate_is_taken_from_the_hardware_not_the_config(self):
        """A receiver cannot produce every rate exactly, and everything downstream
        divides samples by this figure to count seconds.  The shipped default asks for
        256 kHz and decimates by 16.
        """
        config = self.rtlsdr_config()
        pipeline = self.open_with_a_fake_receiver(config)

        assert config.audio.sample_rate == 16_000, (
            f'[audio] sample_rate was left at {config.audio.sample_rate} rather than '
            'the 16000 Hz the receiver settings produce.  The recorder writes it into '
            'every .wav header and the analyzer counts seconds by it.')
        assert pipeline.capacity_samples > 0

    def test_raw_iq_is_kept_only_when_the_setting_asks_for_it(self):
        """The setting has to reach the pipeline, or an operator who turned IQ
        recording on gets an event with no IQ file and nothing saying why.

        It is gated because the buffer is not small: several seconds of raw IQ is
        4.7 MB at the default rate and 44 MB at the highest the hardware takes, none
        of it touched by a station that will never record IQ.
        """
        off = self.open_with_a_fake_receiver(self.rtlsdr_config())
        assert off.iq_buffer is None

        config = self.rtlsdr_config()
        config.recording.record_iq = True
        on = self.open_with_a_fake_receiver(config)
        assert on.iq_buffer is not None
        assert on.iq_buffer.dtype.itemsize == 1, 'an RTL-SDR delivers unsigned bytes'

    def test_a_section_that_cannot_work_is_refused_before_anything_starts(self):
        """2.4 MS/s decimated by 16 gives 150 kHz of audio.  Nothing used to check it,
        so the monitor ran with a ring buffer holding one second instead of 9.6 and
        wrote .wav files at a rate --playback then refused.
        """
        config = self.rtlsdr_config(iq_sample_rate=2_400_000)

        with pytest.raises(ValueError, match='150000 Hz of audio'):
            self.open_with_a_fake_receiver(config)

    def test_the_receiver_brings_its_own_level_calibration(self):
        """The sound card's dB offset has nothing to do with a tuner's, so everything
        that converts a level reads BuzzConfig.level_offset_db, which picks by source.
        """
        config = self.rtlsdr_config(calibrated_offset_db=-38.5)
        self.open_with_a_fake_receiver(config)

        assert config.level_offset_db == -38.5

    def test_the_sound_cards_own_figure_is_left_alone(self):
        """Startup used to copy the receiver's figure over it, which left a config
        object holding the same setting twice and a file whose [station] value no
        longer described anything.  Nothing writes to it now.
        """
        config = self.rtlsdr_config(calibrated_offset_db=-38.5)
        config.station.audio_rf_conversion_db = -32.0
        self.open_with_a_fake_receiver(config)

        assert config.station.audio_rf_conversion_db == -32.0
        assert config.level_offset_db == -38.5

    def test_an_uncalibrated_receiver_is_estimated_from_the_gain_and_says_so(self, caplog):
        """The estimate is good enough to start from and not good enough to publish,
        which the operator has no way to know from the numbers themselves.
        """
        config = self.rtlsdr_config(gain_db=40.2, calibrated_offset_db=None)

        with caplog.at_level(logging.WARNING, logger='buzz.main'):
            self.open_with_a_fake_receiver(config)

        assert config.level_offset_db == pytest.approx(-40.2)
        assert any('not been calibrated' in m for m in caplog.messages), (
            f'Nothing warned that the levels are estimated: {caplog.messages}.  They '
            'are a few dB out and move when the gain does, and an S-meter reading gives '
            'no sign of it.')


class TestThePyrtlsdrImportFailureIsExplained:
    """pyrtlsdr resolves rtlsdr_set_dithering as it imports, so a librtlsdr that
    predates that symbol fails at the import rather than at the first call.

    That is the case the local import exists to isolate, and it reached the operator as
    a bare ModuleNotFoundError with none of the explanation this module worked out.
    """

    def test_a_missing_library_names_what_to_install(self):
        import builtins
        real_import = builtins.__import__

        def refuse_rtlsdr(name, *args, **kwargs):
            if name == 'rtlsdr':
                raise ImportError("undefined symbol: rtlsdr_set_dithering")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', refuse_rtlsdr):
            with pytest.raises(RuntimeError) as caught:
                open_device(0)

        message = str(caught.value)
        assert 'pyrtlsdr[lib]' in message, (
            'The message does not name the package to install.  Whoever hits this sees '
            f'an import failure and cannot tell which library failed.  Got: {message!r}')
        assert 'rtlsdr_set_dithering' in message, (
            'The underlying error was dropped, so a mismatched librtlsdr looks the same '
            'as one that was never installed.  The fix differs between the two.')
        assert 'soundcard' in message

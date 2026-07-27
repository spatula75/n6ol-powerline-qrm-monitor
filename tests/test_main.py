"""Tests for buzz.main: configure_logging(), weather client factory, and headless wait."""
import logging
from unittest.mock import MagicMock, patch

import pytest

import buzz.main as main_module
from buzz.config import BuzzConfig
from buzz.main import _wait_until_interrupted, configure_logging, make_weather_client
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


class TestWaitUntilInterrupted:
    def test_closes_sampler_on_keyboard_interrupt(self):
        sampler = MagicMock()
        with patch('buzz.main.threading.Event') as mock_event:
            mock_event.return_value.wait.side_effect = KeyboardInterrupt
            _wait_until_interrupted(sampler)
        sampler.close.assert_called_once()

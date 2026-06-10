"""Tests for buzz.main: configure_logging() and module-level constants."""
import logging

import pytest

import buzz.main as main_module
from buzz.main import configure_logging


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

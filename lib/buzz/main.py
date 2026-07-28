"""
Entry point for the powerline QRM monitor.

Loads configuration, wires up the audio pipeline, collector, weather client,
CSV store, plotter, and publisher, then either launches the Qt waterfall
display (default) or runs headlessly (--headless).  --top keeps the waterfall
window always on top of other windows.

In GUI mode the Qt event loop runs on the main thread; the collector runs
on a daemon thread.  Closing the window (or ^C) stops the analyzer, then
the audio pipeline, and lets the daemon thread exit with the process.

In headless mode the main thread blocks until ^C, then stops the analyzer
and the pipeline in that same order.
"""

import argparse
import faulthandler
import logging
import logging.config
import signal
import sys
import threading

from buzz.analyzer import ContinuousAnalyzer
from buzz.collector import Collector
from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import (
    CumulusMXWeatherClient,
    NullWeatherClient,
    OpenMeteoWeatherClient,
    WeatherClient,
)

faulthandler.enable()

ROOT_PACKAGE = 'buzz'
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'loggers': {
            ROOT_PACKAGE: {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'CRITICAL',
            'handlers': [],
        },
    }
    logging.config.dictConfig(logging_config)


def make_weather_client(config: BuzzConfig) -> WeatherClient:
    """Build the configured weather client, validating its settings up front.

    A misconfigured source degrades to NullWeatherClient with a single startup
    warning, rather than failing (and logging) on every collection cycle.
    """
    weather_config = config.weather
    if weather_config.source == 'openmeteo':
        if weather_config.latitude is None or weather_config.longitude is None:
            logger.warning('Weather source is openmeteo but latitude/longitude are not set; '
                           'weather data disabled.')
            return NullWeatherClient()
        return OpenMeteoWeatherClient(weather_config.latitude, weather_config.longitude)
    if weather_config.source == 'cumulusmx':
        if not weather_config.url:
            logger.warning('Weather source is cumulusmx but url is not set; '
                           'weather data disabled.')
            return NullWeatherClient()
        return CumulusMXWeatherClient(weather_config.url)
    if weather_config.source != 'none':
        logger.warning(
            "Unknown weather source %r — expected 'cumulusmx', 'openmeteo', or 'none'; "
            'weather data disabled.', weather_config.source)
    return NullWeatherClient()


def _wait_until_interrupted(sampler: AudioSampler, analyzer: ContinuousAnalyzer) -> None:
    """Headless main loop: block until ^C, then stop the analyzer and the audio pipeline.

    Stops the analyzer first, mirroring MainWindow.closeEvent() — otherwise the
    analyzer thread's in-flight tick can end up calling into an already-closed
    audio stream during shutdown.
    """
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        sampler.close()


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description='N6OL Powerline QRM Monitor')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI waterfall display')
    parser.add_argument('--top', action='store_true',
                        help='Keep the waterfall window always on top of other windows')
    args = parser.parse_args()

    configure_logging()

    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    sampler = AudioSampler(config)

    analyzer = ContinuousAnalyzer(sampler.pipeline, config)
    analyzer.start()

    weather = make_weather_client(config)

    store = CsvStore(config)
    plotter = Plotter(config, store)
    publisher = Publisher(config) if config.server.enabled else None

    collector = Collector(config, analyzer, weather, store, plotter, publisher)
    collector_thread = threading.Thread(
        target=collector.collection_loop, daemon=True, name='collector',
    )
    collector_thread.start()

    if args.headless:
        _wait_until_interrupted(sampler, analyzer)
        return

    try:
        from PySide6.QtCore import QTimer  # noqa: I001
        from PySide6.QtWidgets import QApplication
        from buzz.waterfall import MainWindow
    except ImportError:
        logger.warning(
            'PySide6 not installed — falling back to headless mode. '
            'Install PySide6 or run with --headless to suppress this warning.'
        )
        _wait_until_interrupted(sampler, analyzer)
        return

    app = QApplication(sys.argv)
    window = MainWindow(sampler.pipeline, analyzer, config, always_on_top=args.top)
    window.show()

    # Allow Ctrl+C to close the window cleanly from the console
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # QTimer keeps the Python interpreter ticking so SIGINT can be delivered
    sigint_keepalive = QTimer()
    sigint_keepalive.timeout.connect(lambda: None)
    sigint_keepalive.start(200)

    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    main()

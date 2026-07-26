"""
Entry point for the powerline QRM monitor.

Loads configuration, wires up the audio pipeline, collector, weather client,
CSV store, plotter, and publisher, then either launches the Qt waterfall
display (default) or runs headlessly (--headless).

In GUI mode the Qt event loop runs on the main thread; the collector runs
on a daemon thread.  Closing the window (or ^C) stops the audio pipeline
and lets the daemon thread exit with the process.

In headless mode the main thread blocks until ^C, then stops the pipeline.
"""

import argparse
import logging
import logging.config
import signal
import sys
import threading

from buzz.collector import Collector
from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient

ROOT_PACKAGE = 'buzz'


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


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description='N6OL Powerline QRM Monitor')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI waterfall display')
    args = parser.parse_args()

    configure_logging()

    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    sampler = AudioSampler(config)

    wc = config.weather
    if wc.source == 'openmeteo':
        weather = OpenMeteoWeatherClient(wc.latitude, wc.longitude)
    elif wc.source == 'cumulusmx':
        weather = CumulusMXWeatherClient(wc.url)
    else:
        weather = NullWeatherClient()

    store = CsvStore(config)
    plotter = Plotter(config, store)
    publisher = Publisher(config) if config.server.enabled else None

    collector = Collector(config, sampler, weather, store, plotter, publisher)
    collector_thread = threading.Thread(
        target=collector.collection_loop, daemon=True, name='collector',
    )
    collector_thread.start()

    if args.headless:
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            sampler.close()
        return

    try:
        from PySide6.QtCore import QTimer  # noqa: I001
        from PySide6.QtWidgets import QApplication
        from buzz.waterfall import MainWindow
    except ImportError:
        logging.getLogger(ROOT_PACKAGE).warning(
            'PySide6 not installed — falling back to headless mode. '
            'Install PySide6 or run with --headless to suppress this warning.'
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            sampler.close()
        return

    app = QApplication(sys.argv)
    window = MainWindow(sampler.pipeline, config)
    window.show()

    # Allow Ctrl+C to close the window cleanly from the console
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # QTimer keeps the Python interpreter ticking so SIGINT can be delivered
    pulse = QTimer()
    pulse.timeout.connect(lambda: None)
    pulse.start(200)

    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    main()

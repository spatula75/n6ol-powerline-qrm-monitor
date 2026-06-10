"""
Entry point for the powerline QRM monitor.

Loads configuration from ~/.buzz/config.toml (or uses defaults if the file is
absent), wires up the sampler, weather client, CSV store, plotter, and publisher,
then runs the collection loop indefinitely.
"""

import logging
import logging.config

from buzz.collector import Collector
from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient

ROOT_PACKAGE = 'buzz'  # getting this from __name__ can be problematic when running this file directly

def configure_logging():
    logging_config = {
        "version": 1,
        # This automatically disables or overrides existing third-party loggers
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            # Dynamically configure ONLY the root package to use the console handler
            ROOT_PACKAGE: {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,  # Prevents logs from going up to the silent root
            },
        },
        "root": {
            # Set the root logger to CRITICAL to ignore all other libraries
            "level": "CRITICAL",
            "handlers": [],
        },
    }

    logging.config.dictConfig(logging_config)

if __name__ == '__main__':  # pragma: no cover
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
    Collector(config, sampler, weather, store, plotter, publisher).collection_loop()

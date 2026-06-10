"""
Entry point for the powerline QRM monitor.

Loads configuration from ~/.buzz/config.toml (or uses defaults if the file is
absent), wires up the sampler, weather client, CSV store, plotter, and publisher,
then runs the collection loop indefinitely.
"""

from buzz.collector import Collector
from buzz.config import BuzzConfig, CONFIG_PATH
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import CumulusMXWeatherClient, NullWeatherClient, OpenMeteoWeatherClient

if __name__ == '__main__':
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
    publisher = Publisher(config)
    Collector(config, sampler, weather, store, plotter, publisher).collection_loop()

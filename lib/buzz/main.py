from buzz.collector import Collector
from buzz.config import BuzzConfig, CONFIG_PATH
from buzz.csv_store import CsvStore
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.sampler import AudioSampler
from buzz.weather import CumulusMXWeatherClient

if __name__ == '__main__':
    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    sampler = AudioSampler(config)
    weather = CumulusMXWeatherClient(config.weather_url)
    store = CsvStore(config)
    plotter = Plotter(config, store)
    publisher = Publisher(config)
    Collector(config, sampler, weather, store, plotter, publisher).collection_loop()

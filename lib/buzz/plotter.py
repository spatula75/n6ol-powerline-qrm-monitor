import gc
from datetime import datetime, timedelta
from functools import wraps
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from numpy import cumsum

from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore

_GRAPH_W = 1600
_GRAPH_H = 640
_SUMMARY_H = 540

# Axes box margins in pixels for the daily graph.
_M_LEFT   = 88   # room for y-axis label + tick labels
_M_RIGHT  = 24
_M_TOP    = 43
_M_BOTTOM = 66   # room for x-axis label + tick labels


def _force_post_gc(func):
    # Workaround for https://github.com/matplotlib/matplotlib/issues/27713
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper


class Plotter:
    def __init__(self, config: BuzzConfig, store: CsvStore):
        self._config = config
        self._store = store

    def _smooth(self, data: list, points: int):
        ret = cumsum(data, dtype=float)
        ret[points:] = ret[points:] - ret[:-points]
        return ret[points - 1:] / points

    @_force_post_gc
    def generate_graph_from_csv(self, input_filename: str, output_filename: str, smooth=0):
        with open(input_filename, 'r') as f:
            timestamps, signals, noises, snrs = [], [], [], []
            for line in f:
                timestamp, snr, signal, noise, _ = line.split(',', 4)
                try:
                    timestamps.append(datetime.fromisoformat(timestamp).astimezone(ZoneInfo(self._config.timezone)))
                    signals.append(float(signal))
                    noises.append(float(noise))
                    snrs.append(float(snr))
                except ValueError:
                    continue

        if smooth:
            if len(timestamps) <= smooth:
                return
            signals = self._smooth(signals, smooth)
            noises = self._smooth(noises, smooth)
            timestamps = timestamps[smooth - 1:]
            title = (f'Powerline Noise vs Noise Floor ({smooth} point moving avg), '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({self._config.timezone} Timezone)')
        else:
            title = (f'Powerline Noise vs Noise Floor, '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({self._config.timezone} Timezone)')

        signals_adjusted = [
            val + self._config.distance_attenuation
            if ((val > self._config.noise_threshold and snrs[i] > self._config.noise_min_snr)
                or val > self._config.noise_threshold + 0.5 * self._config.noise_min_snr)
            else val
            for i, val in enumerate(signals)
        ]

        plt.rcParams['timezone'] = self._config.timezone
        px = 1 / plt.rcParams['figure.dpi']
        figure, axes = plt.subplots(figsize=(_GRAPH_W * px, _GRAPH_H * px))
        figure.subplots_adjust(left=_M_LEFT/_GRAPH_W, right=1 - _M_RIGHT/_GRAPH_W,
                               top=1 - _M_TOP/_GRAPH_H, bottom=_M_BOTTOM/_GRAPH_H)
        plt.title(title)
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        noise_twin = axes.twinx()

        min_y = min(min(signals), min(noises), min(signals_adjusted),
                    -48 + self._config.audio_rf_conversion_db) * 1.33
        max_y = max(max(signals), max(noises), max(signals_adjusted),
                    -48 + self._config.audio_rf_conversion_db) / 1.33

        plot_signal, = axes.plot(timestamps, signals, 'r-', label=f'{self._config.pulse_rate}pps dBm')
        plot_noise, = noise_twin.plot(timestamps, noises, 'g-', label='Noise Floor dBm')

        axes.set_xlim(timestamps[0], timestamps[-1])
        axes.set_ylim(min_y, max_y)
        noise_twin.set_ylim(min_y, max_y)
        axes.set_xlabel('Time')
        axes.set_ylabel('dBm')
        noise_twin.get_yaxis().set_ticks([])

        axes.yaxis.label.set_color(plot_signal.get_color())
        noise_twin.yaxis.label.set_color(plot_noise.get_color())
        tick_kwargs = dict(size=4, width=1.5)
        axes.tick_params(axis='y', colors=plot_signal.get_color(), **tick_kwargs)
        noise_twin.tick_params(axis='y', colors=plot_noise.get_color(), **tick_kwargs)

        plot_s9 = axes.axhline(y=-73, color='tan', linestyle='dashed', label='S9 (-73dBm) signal strength')
        plot_threshold = axes.axhline(y=self._config.noise_threshold, color='gray', linestyle='dashed',
                                      label=f'{self._config.noise_threshold} dBm threshold')
        plot_floor = axes.axhline(y=self._config.noise_floor, color='gray',
                                  label=f'{self._config.noise_floor} dBm typical noise floor')

        axes.legend(loc='lower left', handles=[plot_signal, plot_noise, plot_s9, plot_threshold, plot_floor])
        plt.savefig(output_filename, pil_kwargs={'optimize': True})
        plt.close()

    @_force_post_gc
    def generate_summary_graph(self, output_filename: str, start_date: datetime):
        zone = ZoneInfo(self._config.timezone)
        end_date = datetime.now(zone)
        time_to_snr = self._store.read_range_to_time_dict(start_date, end_date)

        px = 1 / plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(_GRAPH_W * px, _SUMMARY_H * px))
        plt.rcParams['timezone'] = self._config.timezone

        run_time = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=zone)
        all_datetimes = [run_time + timedelta(minutes=15 * i) for i in range(4 * 24)]

        vals = [time_to_snr.get(dt.time(), 0) for dt in all_datetimes]
        max_val = max(vals)
        if max_val == 0:
            return
        normalized_vals = [int(100 * (val / max_val)) for val in vals]

        red_fade = 0xfe - 0x87
        green_fade = 0xfe - 0xce
        blue_fade = 0xfe - 0xeb

        colors = [
            'firebrick' if val == 100
            else 'indianred' if val > 92
            else 'lightcoral' if val > 85
            else (f'#{int(0x87 + (85 - val) / 85 * red_fade):x}'
                  f'{int(0xce + (85 - val) / 85 * green_fade):x}'
                  f'{int(0xeb + (85 - val) / 85 * blue_fade):x}')
            for val in normalized_vals
        ]

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.set_xlim(all_datetimes[0] - timedelta(minutes=10), all_datetimes[-1] + timedelta(minutes=10))
        ax.set_xlabel(f'Time ({self._config.timezone} zone)')
        ax.set_ylabel('Normalized Probability')
        ax.legend(title='Legend', handles=[
            mpatches.Patch(color='skyblue', label='<  85%'),
            mpatches.Patch(color='lightcoral', label='>  85%'),
            mpatches.Patch(color='indianred', label='>  92%'),
            mpatches.Patch(color='firebrick', label='= 100%'),
        ])
        ax.bar(all_datetimes, normalized_vals, width=timedelta(minutes=13), color=colors)
        plt.title(f'Time of Day vs Normalized Probability of {self._config.pulse_rate}pps Interference\n'
                  f'15-minute increments from {start_date.strftime("%Y-%m-%d %H:%M")} '
                  f'to {end_date.strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout(pad=1.1)
        plt.savefig(output_filename)
        plt.close()

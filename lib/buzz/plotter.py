"""
Plot generation for daily noise traces and time-of-day probability summaries.

Plotter.generate_graph_from_csv() renders a daily signal-vs-noise-floor line chart
from a CSV file.  Plotter.generate_summary_graph() renders a bar chart showing the
normalised probability of interference at each 15-minute interval of the day,
aggregated across a configurable date range.

All output is saved as PNG.  The _force_post_gc decorator works around a matplotlib
memory-leak bug that causes handles to accumulate across repeated savefig calls.
"""

import gc  # noqa: I001
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')  # must precede submodule imports; forces non-interactive backend
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from buzz.config import BuzzConfig
from buzz.csv_store import CsvStore

_GRAPH_W = 1600
_GRAPH_H = 640
_SUMMARY_H = 540

# S9 signal strength per IARU recommendation: −73 dBm into 50 Ω.
# Drawn as a reference line on the daily graph so it's easy to gauge signal severity.
_S9_DBM = -73

# Axes box margins in pixels for the daily graph.
_M_LEFT   = 88   # room for y-axis label + tick labels
_M_RIGHT  = 24
_M_TOP    = 43
_M_BOTTOM = 66   # room for x-axis label + tick labels


def _bar_color(val: int) -> str:
    """Map a normalized 0–100 bar value to a matplotlib color string.

    100 → firebrick, >92 → indianred, >85 → lightcoral, ≤85 → gradient
    from skyblue (#87ceeb) at val=85 fading to near-white (#fefefe) at val=0.
    """
    if val == 100:
        return 'firebrick'
    if val > 92:
        return 'indianred'
    if val > 85:
        return 'lightcoral'
    fade = (85 - val) / 85
    r = int(0x87 + fade * (0xfe - 0x87))
    g = int(0xce + fade * (0xfe - 0xce))
    b = int(0xeb + fade * (0xfe - 0xeb))
    return f'#{r:02x}{g:02x}{b:02x}'


def _force_post_gc(func):
    # Workaround for https://github.com/matplotlib/matplotlib/issues/27713
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper


class Plotter:
    def __init__(self, config: BuzzConfig, store: CsvStore) -> None:
        self._config = config
        self._store = store

    def _smooth(self, data: list[float], points: int) -> np.ndarray:
        ret = np.cumsum(data, dtype=float)
        ret[points:] = ret[points:] - ret[:-points]
        return ret[points - 1:] / points

    @_force_post_gc
    def generate_graph_from_csv(self, input_filename: Path | str, output_filename: Path | str, smooth: int = 0) -> None:
        """Render a daily noise trace and save it as a PNG.

        Reads signal and noise floor values from input_filename and plots them on a
        shared y-axis (dBm) against time-of-day.  Reference lines are drawn for S9,
        the detection threshold, and the typical noise floor.

        If smooth > 0, a simple moving average of that many points is applied before
        plotting.  Returns early without writing output if the file has too few rows
        for the requested smoothing window.
        """
        station = self._config.station
        audio = self._config.audio
        rows = self._store.read_rows(input_filename)
        if not rows:
            return
        timestamps    = [r.timestamp for r in rows]
        snrs          = [r.snr for r in rows]
        signals       = [r.signal for r in rows]
        noises        = [r.noise for r in rows]
        lock_statuses = [r.lock_status for r in rows]

        if smooth:
            if len(timestamps) <= smooth:
                return
            signals = self._smooth(signals, smooth)
            noises = self._smooth(noises, smooth)
            timestamps    = timestamps[smooth - 1:]
            lock_statuses = lock_statuses[smooth - 1:]
            title = (f'Powerline Noise vs Noise Floor ({smooth} point moving avg), '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({station.timezone} Timezone)')
        else:
            title = (f'Powerline Noise vs Noise Floor, '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({station.timezone} Timezone)')

        # For qualifying detections, estimate the source power by adding back the measured
        # path loss.  Not plotted as a separate line, but included in the y-axis upper bound
        # so the scale accommodates the estimated source-power level alongside the measured values.
        # (Adjusted values are always >= the originals, so they have no effect on min_y.)
        source_power_estimate = [
            val + station.distance_attenuation
            if ((val > station.noise_threshold and snrs[i] > station.noise_min_snr)
                or val > station.noise_threshold + 0.5 * station.noise_min_snr)
            else val
            for i, val in enumerate(signals)
        ]

        plt.rcParams['timezone'] = station.timezone
        px = 1 / plt.rcParams['figure.dpi']
        figure, axes = plt.subplots(figsize=(_GRAPH_W * px, _GRAPH_H * px))
        figure.subplots_adjust(left=_M_LEFT/_GRAPH_W, right=1 - _M_RIGHT/_GRAPH_W,
                               top=1 - _M_TOP/_GRAPH_H, bottom=_M_BOTTOM/_GRAPH_H)
        plt.title(title)
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        noise_twin = axes.twinx()

        # 1.33 is a margin factor: since dBm values are negative, multiplying the most-negative
        # value by 1.33 pushes the lower axis edge further down, while dividing the
        # least-negative value by 1.33 pulls the upper edge down — keeping reference lines
        # (noise floor, threshold, S9) away from the plot borders.
        min_y = min(min(signals), min(noises), -48 + station.audio_rf_conversion_db) * 1.33
        max_y = max(max(signals), max(noises), max(source_power_estimate),
                    -48 + station.audio_rf_conversion_db) / 1.33

        # Red: NaN where unlocked so the green noise-floor line shows through cleanly.
        # Green is on noise_twin (rendered on top) and is always continuous.
        signals_red = np.where(
            [s != 'none' for s in lock_statuses],
            np.array(signals, dtype=float),
            np.nan,
        )
        plot_signal, = axes.plot(timestamps, signals_red, 'r-', label=f'{audio.pulse_rate}pps dBm')
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

        plot_s9 = axes.axhline(y=_S9_DBM, color='tan', linestyle='dashed', label=f'S9 ({_S9_DBM} dBm) signal strength')
        plot_threshold = axes.axhline(y=station.noise_threshold, color='gray', linestyle='dashed',
                                      label=f'{station.noise_threshold} dBm threshold')
        plot_floor = axes.axhline(y=station.noise_floor, color='gray',
                                  label=f'{station.noise_floor} dBm typical noise floor')

        axes.legend(loc='lower left', handles=[plot_signal, plot_noise, plot_s9, plot_threshold, plot_floor])
        plt.savefig(output_filename, pil_kwargs={'optimize': True})
        plt.close()

    @_force_post_gc
    def generate_summary_graph(self, output_filename: Path | str, start_date: datetime) -> None:
        """Render a time-of-day interference probability bar chart and save it as a PNG.

        Aggregates scores from all CSV files between start_date and now, buckets them
        into 15-minute intervals, normalises to the peak bucket (= 100%), and colours
        each bar by intensity.  Returns early without writing output if there is no
        data in the date range.
        """
        station = self._config.station
        audio = self._config.audio
        zone = ZoneInfo(station.timezone)
        end_date = datetime.now(zone)
        time_to_snr = self._store.read_range_to_time_dict(start_date, end_date)

        run_time = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=zone)
        all_datetimes = [run_time + timedelta(minutes=15 * i) for i in range(4 * 24)]

        # Decide whether there is anything to draw BEFORE creating the figure, so
        # the no-data early return can't leak an unclosed figure.
        vals = [time_to_snr.get(dt.time(), 0) for dt in all_datetimes]
        max_val = max(vals)
        if max_val == 0:
            return
        normalized_vals = [int(100 * (val / max_val)) for val in vals]

        colors = [_bar_color(val) for val in normalized_vals]

        plt.rcParams['timezone'] = station.timezone
        px = 1 / plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(_GRAPH_W * px, _SUMMARY_H * px))

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.set_xlim(all_datetimes[0] - timedelta(minutes=10), all_datetimes[-1] + timedelta(minutes=10))
        ax.set_xlabel(f'Time ({station.timezone} zone)')
        ax.set_ylabel('Normalized Probability')
        ax.legend(title='Legend', handles=[
            mpatches.Patch(color='skyblue', label='<  85%'),
            mpatches.Patch(color='lightcoral', label='>  85%'),
            mpatches.Patch(color='indianred', label='>  92%'),
            mpatches.Patch(color='firebrick', label='= 100%'),
        ])
        ax.bar(all_datetimes, normalized_vals, width=timedelta(minutes=13), color=colors)
        plt.title(f'Time of Day vs Normalized Probability of {audio.pulse_rate}pps Interference\n'
                  f'15-minute increments from {start_date.strftime("%Y-%m-%d %H:%M")} '
                  f'to {end_date.strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout(pad=1.1)
        plt.savefig(output_filename)
        plt.close()

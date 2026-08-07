"""
Plot generation for daily noise traces and time-of-day probability summaries.

Plotter.generate_graph_from_csv() renders a daily signal-vs-noise-floor line chart
from a CSV file. Plotter.generate_summary_graph() renders a bar chart showing the
normalized probability of interference at each 15-minute interval of the day,
aggregated across a configurable date range.

All output is saved as PNG. The _gc_guarded decorator does two things. It forces a
gc.collect() after each render. That works around a matplotlib memory-leak bug that
causes handles to accumulate across repeated savefig calls. It also disables the
cyclic GC for the duration of the render itself, working around a PySide6/shiboken
crash - see the decorator's docstring for the full story.
"""

import gc  # noqa: I001
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')  # must precede submodule imports; forces non-interactive backend
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from buzz.config import BuzzConfig
from buzz.constants import S9_DBM
from buzz.csv_store import BUCKET_MINUTES, CsvStore

# One day's values for a single trace. A plain list as read from the CSV, and a NumPy
# array once _smooth() has run over it, so anything holding a series has to accept both.
Series = list[float] | np.ndarray

# _gc_guarded wraps plot methods without changing their signatures, so it is generic
# over the function it decorates. Callable[..., Any] would compile but erase the
# argument types of every method it is applied to.
_P = ParamSpec('_P')
_R = TypeVar('_R')

_GRAPH_W = 1600
_GRAPH_H = 640
_SUMMARY_H = 540

# The daily graph's y-axis always includes the dBm level corresponding to this
# audio level (−48 dBFS + audio_rf_conversion_db), so a quiet or flat trace still
# gets a sensible scale instead of the axis collapsing around a few dB of noise.
_AXIS_ANCHOR_DBFS = -48

# Visual gap between summary bars: each bucket-wide bar is drawn this many
# minutes narrower than the bucket so adjacent bars don't touch.
_BAR_GAP_MINUTES = 2

# Axes box margins in pixels for the daily graph.
_M_LEFT   = 88   # room for y-axis label + tick labels
_M_RIGHT  = 24
_M_TOP    = 43
_M_BOTTOM = 66   # room for x-axis label + tick labels

# Summary-bar intensity thresholds (normalized 0-100) and their colors.
# Shared by _bar_color and the summary graph's legend so the two can't drift apart.
_PCT_MAX      = 100
_PCT_HIGH     = 92
_PCT_ELEVATED = 85
_COLOR_MAX      = 'firebrick'
_COLOR_HIGH     = 'indianred'
_COLOR_ELEVATED = 'lightcoral'


def _bar_color(val: int) -> str:
    """Map a normalized 0-100 bar value to a matplotlib color string.

    Values above _PCT_ELEVATED use the fixed threshold colors. At or below it, this
    uses a gradient from skyblue (#87ceeb) at the threshold, fading to near-white
    (#fefefe) at val=0.
    """
    if val == _PCT_MAX:
        return _COLOR_MAX
    if val > _PCT_HIGH:
        return _COLOR_HIGH
    if val > _PCT_ELEVATED:
        return _COLOR_ELEVATED
    fade = (_PCT_ELEVATED - val) / _PCT_ELEVATED
    r = int(0x87 + fade * (0xfe - 0x87))
    g = int(0xce + fade * (0xfe - 0xce))
    b = int(0xeb + fade * (0xfe - 0xeb))
    return f'#{r:02x}{g:02x}{b:02x}'


def _smooth(data: list[float], points: int) -> np.ndarray:
    """Simple moving average of the given window length. Output is shorter by points-1.

    This uses the cumulative-sum trick: the difference between cumsum values `points`
    apart is the sum of each sliding window, computed in O(n) instead of O(n·points).
    """
    ret = np.cumsum(data, dtype=float)
    ret[points:] = ret[points:] - ret[:-points]
    return ret[points - 1:] / points


def _gc_guarded(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # ------------------------------------------------------------------------
    # WHY THIS DECORATOR DISABLES THE GC DURING THE CALL - READ BEFORE REMOVING
    #
    # This has hit a reproducible crash in production: a Windows access violation
    # (0xC0000005) that took the whole process down. faulthandler caught it, and
    # the traceback showed the collector thread mid-gc.collect(), inside
    # matplotlib's Artist.set() -> cbook.normalize_kwargs(), which had called into
    # shibokensupport/signature/loader.py - that is PySide6/shiboken internals, not
    # matplotlib's.
    #
    # The mechanism: importing PySide6 anywhere in the process (this app does, for
    # the GUI waterfall window) makes shiboken globally replace builtins.__import__
    # for every thread, not just the Qt thread. That hook supports a Qt-for-Python
    # feature (__feature__ snake_case/true_property) this codebase never uses, but
    # it installs unconditionally regardless. When matplotlib's kwarg-normalization
    # internals do an import in the course of rendering, on this collector thread,
    # which has nothing to do with Qt, the import gets routed through that hook.
    # Qt for Python has a documented history of reference-counting bugs in this
    # exact module (see PYSIDE-2660, "Crash on deallocating None triggered via
    # Shiboken", fixed for that specific repro, but the crash here is a new one on
    # a newer Python/PySide6 combination). The crash happened when the cyclic GC
    # ran *while* that hook's C-level bookkeeping was mid-flight, and walked a
    # corrupted object graph.
    #
    # A threading.Lock cannot fix this: shiboken's internals don't know this lock
    # exists and have no reason to respect it. The PySIDE-2660 repro also crashed
    # with no second thread involved at all, so this is not purely a race that
    # could be serialized away. The one thing that actually protects every thread,
    # this program's and Qt's, is disabling the GC itself for the narrow window
    # where matplotlib is exercising this code path, since gc.disable() is a
    # single interpreter-wide switch with authority over all of them.
    # ------------------------------------------------------------------------
    # The gc.collect() afterward is unrelated: a workaround for a separate
    # matplotlib memory leak (https://github.com/matplotlib/matplotlib/issues/27713)
    # that causes handles to accumulate across repeated savefig calls. It runs
    # after re-enabling GC, once the render is past the risky window.
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            result = func(*args, **kwargs)
        finally:
            if was_enabled:
                gc.enable()
        gc.collect()
        return result
    return wrapper


@dataclass(frozen=True)
class _DailySeries:
    """One day's data for the daily chart, already smoothed if requested."""
    timestamps: list[datetime]
    signals: Series
    noises: Series
    title: str


class Plotter:
    def __init__(self, config: BuzzConfig, store: CsvStore) -> None:
        self._config = config
        self._store = store

    def _load_daily_series(self, input_filename: Path | str, smooth: int) -> '_DailySeries | None':
        """Read one day's CSV rows and, if requested, apply a moving average.

        Returns None when there is nothing to plot: an empty file, or fewer rows
        than the requested smoothing window.
        """
        station = self._config.station
        rows = self._store.read_rows(input_filename)
        if not rows:
            return None
        timestamps = [r.timestamp for r in rows]
        signals    = [r.signal for r in rows]
        noises     = [r.noise for r in rows]

        if smooth:
            if len(timestamps) <= smooth:
                return None
            signals = _smooth(signals, smooth)
            noises = _smooth(noises, smooth)
            timestamps    = timestamps[smooth - 1:]
            title = (f'Powerline Noise vs Noise Floor ({smooth} point moving avg), '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({station.timezone} Timezone)')
        else:
            title = (f'Powerline Noise vs Noise Floor, '
                     f'{timestamps[0].strftime("%Y-%m-%d")} ({station.timezone} Timezone)')

        return _DailySeries(timestamps=timestamps, signals=signals, noises=noises,
                            title=title)

    def _daily_chart_y_bounds(self, series: '_DailySeries') -> tuple[float, float]:
        """Y-axis (dBm) bounds that fit the signal and noise traces.

        This always includes the configured audio-level anchor, so a quiet or flat
        trace still gets a sensible scale.

        1.33 is a margin factor. Since dBm values are negative, multiplying the
        most-negative value by 1.33 pushes the lower axis edge further down, while
        dividing the least-negative value by 1.33 pushes the upper edge further up.
        Both moves go away from the data, which is what keeps reference lines
        (noise floor, threshold, S9) off the plot borders.
        """
        station = self._config.station
        anchor = _AXIS_ANCHOR_DBFS + station.audio_rf_conversion_db
        min_y = min(min(series.signals), min(series.noises), anchor) * 1.33
        max_y = max(max(series.signals), max(series.noises), anchor) / 1.33
        return min_y, max_y

    def _draw_reference_lines(self, axes: Axes) -> list[Line2D]:
        """Draw the S9, detection-threshold, and typical-noise-floor reference
        lines on the daily chart, and return their handles for the legend.
        """
        station = self._config.station
        plot_s9 = axes.axhline(y=S9_DBM, color='tan', linestyle='dashed',
                               label=f'S9 ({S9_DBM:g} dBm) signal strength')
        plot_threshold = axes.axhline(y=station.noise_threshold, color='gray', linestyle='dashed',
                                      label=f'{station.noise_threshold} dBm threshold')
        plot_floor = axes.axhline(y=station.noise_floor, color='gray',
                                  label=f'{station.noise_floor} dBm typical noise floor')
        return [plot_s9, plot_threshold, plot_floor]

    def _style_dual_axes(self, axes: Axes, noise_twin: Axes,
                         plot_signal: Line2D, plot_noise: Line2D) -> None:
        """Label the shared x/y axes, and color each y-axis to match its trace.

        This also hides the twin axis's own tick labels, since it shares its scale
        with the primary axis.
        """
        axes.set_xlabel('Time')
        axes.set_ylabel('dBm')
        noise_twin.get_yaxis().set_ticks([])

        axes.yaxis.label.set_color(plot_signal.get_color())
        noise_twin.yaxis.label.set_color(plot_noise.get_color())
        tick_kwargs = dict(size=4, width=1.5)
        axes.tick_params(axis='y', colors=plot_signal.get_color(), **tick_kwargs)
        noise_twin.tick_params(axis='y', colors=plot_noise.get_color(), **tick_kwargs)

    @_gc_guarded
    def generate_graph_from_csv(self, input_filename: Path | str, output_filename: Path | str, smooth: int = 0) -> None:
        """Render a daily noise trace and save it as a PNG.

        This reads signal and noise floor values from input_filename and plots them
        on a shared y-axis (dBm) against time-of-day. Reference lines are drawn for
        S9, the detection threshold, and the typical noise floor.

        If smooth > 0, a simple moving average of that many points is applied before
        plotting. This returns early without writing output if the file has too few
        rows for the requested smoothing window.
        """
        station = self._config.station
        audio = self._config.audio

        series = self._load_daily_series(input_filename, smooth)
        if series is None:
            return

        # rc_context scopes the timezone setting (used by matplotlib's date
        # machinery) to this render instead of mutating global state.
        with matplotlib.rc_context({'timezone': station.timezone}):
            px = 1 / plt.rcParams['figure.dpi']   # inches per pixel; figsize is in inches
            figure, axes = plt.subplots(figsize=(_GRAPH_W * px, _GRAPH_H * px))
            figure.subplots_adjust(left=_M_LEFT/_GRAPH_W, right=1 - _M_RIGHT/_GRAPH_W,
                                   top=1 - _M_TOP/_GRAPH_H, bottom=_M_BOTTOM/_GRAPH_H)
            axes.set_title(series.title)
            axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            noise_twin = axes.twinx()
            min_y, max_y = self._daily_chart_y_bounds(series)

            # Both lines plot every row's value continuously, with no NaN gaps: when
            # unlocked, Collector._run_collection() already writes signal == noise for
            # that row, so red and green coincide exactly during unlocked stretches.
            # zorder makes green paint on top there, so an unlocked stretch reads as a
            # single clean green trace instead of a gap. NaN-masking red instead used
            # to fragment it into dozens of disconnected dashes whenever lock flickered
            # on and off for a minute or two. That was worse than useless once smoothed,
            # since the moving average blended real signal readings with unlocked rows'
            # noise-floor stand-in before the mask was even applied.
            plot_signal, = axes.plot(series.timestamps, series.signals, 'r-',
                                     label=f'{audio.pulse_rate}pps dBm', zorder=2)
            plot_noise, = noise_twin.plot(series.timestamps, series.noises, 'g-',
                                          label='Noise Floor dBm', zorder=3)

            axes.set_xlim(series.timestamps[0], series.timestamps[-1])
            axes.set_ylim(min_y, max_y)
            noise_twin.set_ylim(min_y, max_y)
            self._style_dual_axes(axes, noise_twin, plot_signal, plot_noise)

            reference_lines = self._draw_reference_lines(axes)
            axes.legend(loc='lower left', handles=[plot_signal, plot_noise, *reference_lines])
            figure.savefig(output_filename, pil_kwargs={'optimize': True})
            plt.close(figure)

    def _summary_bar_data(self, start_date: datetime, end_date: datetime
                          ) -> tuple[list[datetime], list[int], list[str]] | None:
        """Aggregate scores into 15-minute buckets across the date range.

        This normalizes to the peak bucket (= 100%), and picks each bar's color by
        intensity.

        Deciding whether there is anything to draw happens here, before the caller
        creates a figure, so a no-data result can't leak an unclosed figure.

        Returns (slot_datetimes, normalized_scores, colors), or None if there is no
        data anywhere in the date range.
        """
        time_to_score = self._store.read_range_scores(start_date, end_date)

        midnight = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        slots_per_day = 24 * 60 // BUCKET_MINUTES   # 96
        slot_datetimes = [midnight + timedelta(minutes=BUCKET_MINUTES * i)
                          for i in range(slots_per_day)]

        slot_scores = [time_to_score.get(dt.time(), 0) for dt in slot_datetimes]
        max_score = max(slot_scores)
        if max_score == 0:
            return None
        normalized_scores = [int(100 * (score / max_score)) for score in slot_scores]
        colors = [_bar_color(score) for score in normalized_scores]
        return slot_datetimes, normalized_scores, colors

    @_gc_guarded
    def generate_summary_graph(self, output_filename: Path | str, start_date: datetime) -> None:
        """Render a time-of-day interference probability bar chart and save it as a PNG.

        This aggregates scores from all CSV files between start_date and now, buckets
        them into 15-minute intervals, normalizes to the peak bucket (= 100%), and
        colors each bar by intensity. It returns early without writing output if
        there is no data in the date range.
        """
        station = self._config.station
        audio = self._config.audio
        zone = ZoneInfo(station.timezone)
        end_date = datetime.now(zone)

        bar_data = self._summary_bar_data(start_date, end_date)
        if bar_data is None:
            return
        slot_datetimes, normalized_scores, colors = bar_data

        # rc_context scopes the timezone setting (used by matplotlib's date
        # machinery) to this render instead of mutating global state.
        with matplotlib.rc_context({'timezone': station.timezone}):
            px = 1 / plt.rcParams['figure.dpi']   # inches per pixel; figsize is in inches
            figure, axes = plt.subplots(figsize=(_GRAPH_W * px, _SUMMARY_H * px))

            axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            # Pad the x-axis so the first and last bars aren't clipped by the axes box
            axes.set_xlim(slot_datetimes[0] - timedelta(minutes=10),
                          slot_datetimes[-1] + timedelta(minutes=10))
            axes.set_xlabel(f'Time ({station.timezone} zone)')
            axes.set_ylabel('Normalized Probability')
            axes.legend(title='Legend', handles=[
                mpatches.Patch(color='skyblue', label=f'<  {_PCT_ELEVATED}%'),
                mpatches.Patch(color=_COLOR_ELEVATED, label=f'>  {_PCT_ELEVATED}%'),
                mpatches.Patch(color=_COLOR_HIGH, label=f'>  {_PCT_HIGH}%'),
                mpatches.Patch(color=_COLOR_MAX, label=f'= {_PCT_MAX}%'),
            ])
            axes.bar(slot_datetimes, normalized_scores,
                     width=timedelta(minutes=BUCKET_MINUTES - _BAR_GAP_MINUTES), color=colors)
            axes.set_title(f'Time of Day vs Normalized Probability of {audio.pulse_rate}pps Interference\n'
                           f'{BUCKET_MINUTES}-minute increments from {start_date.strftime("%Y-%m-%d %H:%M")} '
                           f'to {end_date.strftime("%Y-%m-%d %H:%M")}')
            figure.tight_layout(pad=1.1)
            figure.savefig(output_filename)
            plt.close(figure)

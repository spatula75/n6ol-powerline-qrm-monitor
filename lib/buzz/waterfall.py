"""
Waterfall display, S-band meter panel, and main application window.

WaterfallWidget renders a scrolling FFT spectrogram driven by a QTimer.
MeterPanelWidget draws a pair of vertical bar-graph S-meters (noise floor
and signal level) updated by polling the ContinuousAnalyzer result slot.
MainWindow composes both widgets side-by-side and handles clean shutdown.

These three classes carry `# pragma: no cover`: they need a live Qt display to
exercise, which the test suite doesn't have.  Everything else in this module —
the colour math, percentile logic, and meter-aggregation helpers the widgets
call into — is plain functions with no Qt dependency, and is unit tested in
test_waterfall_math.py.  Keeping the exclusion at the class level rather than
omitting the whole file from coverage.run means that testable code actually
counts, and a new untested helper added outside these three classes will still
trip the coverage gate.
"""

from collections import deque
from collections.abc import Sequence
from math import ceil

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from buzz.analyzer import AnalysisResult, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.dsp import SILENCE_DBFS
from buzz.sampler import AudioPipeline
from buzz.scope import SCOPE_H, ScopeWidget

# ---------------------------------------------------------------------------
# Waterfall constants
# ---------------------------------------------------------------------------

_CHUNK = AudioPipeline.CHUNK_SIZE           # 512 samples
_MAX_HZ = 4000                              # top of the displayed band
_FREQ_LABEL_HZ = 500                        # frequency-axis tick spacing
_PIXELS_PER_BIN = 5                         # horizontal scale (128 bins → 640 px at 16 kHz)
_N_ROWS = 48                                # history rows (~4.8 s at 100 ms/frame)
_PIXELS_PER_ROW = 2                         # vertical scale; _WINDOW_H is derived from this
_UPDATE_MS = 100
# Initial guess for the floor-to-ceiling span (8 S-units), used only until the
# colour scale has real data to auto-range from — see _update_color_range().
_DB_RANGE = 48.0
# FFT magnitude that corresponds to 0 dBFS: a full-scale (amplitude 32768)
# sinusoid centred on a bin peaks at 32768 × _CHUNK/4.  The /4 is two factors
# of 1/2: the Hann window attenuates the average sample to half (its coherent
# gain, sum(w)/N = 1/2), and a real sinusoid's FFT splits its magnitude
# equally between the +f and −f bins.
_DB_REF = 20 * np.log10(32768.0 * _CHUNK / 4)
# Maps a broadband time-domain level onto the per-bin level the display reads.
#
# _DB_REF references a *tone* (coherent gain, sum(w)/N = 1/2).  Broadband noise
# obeys the window's *noise power* gain instead: a signal of RMS sigma gives
# E|X|^2 = sigma^2 * sum(w^2) = sigma^2 * 3N/8.  So the per-bin reading for noise is
#     20log10(sigma) + 10log10(3N/8) - 20log10(32768 * N/4)
# and undoing the tone reference to recover a per-bin anchor from a known broadband
# floor needs 20log10(N/4) - 10log10(3N/8) = 10log10(N/6).  Using 10log10(3N/8)
# here instead applies the window's ENBW (1.5 bins) in the wrong direction and
# lands the anchor 20log10(1.5) = 3.5 dB low.
_DB_FFT_NOISE_CORR = float(10 * np.log10(_CHUNK / 6))  # ≈ 19.3 dB
_AXIS_H = 24                                # pixels reserved for frequency axis / header

# Periodic (not symmetric) Hann: spectral analysis wants w[n] = 0.5(1-cos(2*pi*n/N)),
# which is what np.hanning(N+1)[:-1] gives.  np.hanning(N) alone divides by N-1 and
# is the symmetric variant intended for filter design.
_HANN = np.hanning(_CHUNK + 1)[:-1].astype(np.float32)
# How much of each FFT frame is re-analysed in the next one.  Hann tapers to zero at
# both edges, so without overlap an impulse's contribution depends entirely on where
# it happens to land: a third of arrival positions are attenuated by more than 12 dB,
# on a display whose whole subject is a 120 pps impulse train.
#
# The overlap has to satisfy COLA in *power*, because _mean_spectrum_db averages |X|².
# That is stricter than the familiar 50% overlap-add rule, which is the COLA condition
# for amplitude: at 50% the summed w² still ripples by 3.01 dB with impulse position
# (over the two overlapping frames it is 0.5(1+cos²), not a constant).  Hann² has
# harmonic content up to twice the fundamental, so it takes 75% to flatten — measured
# ripple there is exactly 0.00 dB.
_FFT_OVERLAP = 0.75
# Samples the FFT window advances between consecutive frames.
_FFT_ADVANCE_SAMPLES = round(_CHUNK * (1 - _FFT_OVERLAP))   # 128

# ---------------------------------------------------------------------------
# Auto-ranging colour scale
# ---------------------------------------------------------------------------
#
# A fixed dB range calibrated once against the station config goes stale the
# moment receiver gain, band conditions, or the sound card setup drift from
# whatever they were when it was set — and then the whole picture floods into
# one colour with no contrast, because nothing in the live signal maps near
# the black end any more.  So instead the floor and ceiling are read off the
# spectrum history itself, continuously, and drift with actual conditions.

# "Quiet background": low enough to sit below the typical level, but a
# percentile rather than the bare minimum, so one unusually silent bin can't
# drag the whole floor down.
_COLOR_FLOOR_PERCENTILE = 10
# "Generally loud": the top 2% of what's currently on screen.
_COLOR_CEILING_PERCENTILE = 98
# Fraction of the colour scale reserved above the ceiling percentile.  The
# ceiling percentile is mapped to (1 - _COLOR_HEADROOM) rather than 1.0, so a
# transient louder than anything in the recent window still has room to render
# visibly hotter than the routine "loud" colour, instead of every merely-strong
# reading clipping to solid red and looking identical to a real spike.  At the
# colormap's cutover points (see build_colormap), 1 - 0.10 = 0.90 lands on
# orange rather than red, which is exactly the point.
_COLOR_HEADROOM = 0.10
# Smallest the ceiling-minus-floor span is allowed to compress to, in dB — this
# is what keeps the display from reading "hot" when the band is genuinely quiet.
#
# Auto-ranging has a failure mode when there's nothing going on: floor and
# ceiling both track down together, and whatever residual scatter is left in
# the noise gets stretched across the *entire* colour range, painting routine
# statistical wobble as yellow/orange "activity".  Measured directly: folding
# pure Gaussian noise through _mean_spectrum_db with no signal at all still
# produces about 5.4 dB of p10-to-p98 spread, consistently, regardless of
# level — that's just the estimator's own variance from averaging a handful of
# overlapped FFT frames per row, nothing environmental. A span floor has to
# clear that by a healthy margin or it does nothing.
#
# Expressed in S-units (6 dB each, the IARU convention already used for the
# meter panel below) rather than a bare dB number, so "how quiet can the
# display claim to be" stays in the same units as everything else here and is
# easy to retune.  4 S-units puts the natural noise-only spread at roughly a
# fifth of the total range once headroom is applied — comfortably confined to
# the cool end of the colormap instead of dominating it.
_MIN_DYNAMIC_RANGE_S_UNITS = 4
_MIN_DYNAMIC_RANGE_DB = _MIN_DYNAMIC_RANGE_S_UNITS * 6.0   # 24 dB
# EMA weight applied each tick (_UPDATE_MS = 100 ms) to the raw percentiles.
# Without this, the ceiling can start moving after just a couple of ticks of
# louder content — the top-2% tail is only ~123 values out of the window's 6144,
# and each new row contributes 128 of them — which would make the picture's
# contrast visibly shift within a few hundred milliseconds of any brief burst.
# 0.05 gives the EMA itself a settling time of a couple of seconds, once the raw
# percentile it's tracking is free to move.
#
# Note how tight that has become: at the current _N_ROWS the ceiling's 2% slice is
# about *one row* of bins, so a single loud row can carry the whole raw percentile.
# The history was shortened for layout reasons when the scope panel was added, and
# this smoothing is now the only thing standing between one loud row and a visible
# contrast shift.  Lengthening the history would relax that; lowering the alpha
# would too, at the cost of tracking speed.
#
# On startup that takes longer than the EMA alone suggests: history_db starts
# full of the seed value (see floor_seed below), and the 10th-percentile floor
# in particular stays pinned to that stale seed until more than 90% of the rows
# have been replaced with real data — about 4.3 s at one row per tick.  Verified
# against live audio (at the 100-row history this display used before the scope
# was added, where the same arithmetic gave ~10 s): the floor sat exactly on its
# seed value for the first ~9 s, then converged within 0.6 dB of an
# independently measured ground truth a few seconds later.  Shortening the
# history scales that lag down proportionally; it has not been re-measured since.
# Either way the one-time startup lag is a fine trade for a display that
# self-corrects for the rest of the session.
_COLOR_RANGE_EMA_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Meter constants
# ---------------------------------------------------------------------------

# S-unit thresholds in dBm (IARU HF standard: S9 = −73 dBm, 6 dB/unit below)
_S_LEVELS_DBM = (-121, -115, -109, -103, -97, -91, -85, -79, -73, -63, -53, -43, -33)
_S_LABELS     = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', '+10', '+20', '+30', '+40')
_N_SEGS       = len(_S_LEVELS_DBM)          # 13

# (lit R,G,B), (dim R,G,B) per segment index 0=S1 … 12=S9+40
_SEG_LIT = (
    *[(0, 200, 0)] * 6,                     # S1–S6 green
    *[(220, 200, 0)] * 3,                   # S7–S9 yellow
    *[(210, 0, 0)] * 4,                     # S9+10 … +40 red
)
_SEG_DIM = (
    *[(0, 50, 0)] * 6,
    *[(60, 55, 0)] * 3,
    *[(60, 0, 0)] * 4,
)

_SEG_H    = 13   # segment bar height in pixels; gaps are computed from _SEGS_H below
_BAR_W    = 22   # width of each meter bar
_LABEL_W  = 30   # width of shared S-unit label column
_PAD      = 3    # outer and inner horizontal padding
# Panel width: PAD + BAR + PAD + LABEL + PAD + BAR + PAD
_PANEL_W  = _PAD + _BAR_W + _PAD + _LABEL_W + _PAD + _BAR_W + _PAD  # 86 px

# Height of the waterfall widget alone.  The scope sits above it at the same height,
# so the two panels read as a matched pair.
_WATERFALL_H = _N_ROWS * _PIXELS_PER_ROW + _AXIS_H  # 120 px
# Breathing room between panels, used on both axes: vertically between the scope and
# the waterfall, and horizontally between that stack and the meter panel.  One
# constant for both so the layout stays visually consistent rather than having the
# meters crammed against the displays.
_PANEL_GAP   = 8
# Full window height, which is also the meter panel's height: the meters run the
# whole side of the window rather than aligning with either panel, so the 13
# S-segments get the full travel.
#
# This height is not free to drift.  MeterPanelWidget lays out _N_SEGS segments of
# _SEG_H each within _WINDOW_H - 34 pixels, so base_gap = (_WINDOW_H - 203) // 12:
# every pixel of gap between segments costs 12 px of window.  Below 203 the gap goes
# negative and the segments overlap.  At 248 the gap is 3 px, which reads as a
# bar-graph rather than either a solid bar or a scattered row of ticks.
_WINDOW_H    = SCOPE_H + _PANEL_GAP + _WATERFALL_H  # 248 px

# Segment area geometry (within the panel widget)
_CORR_H      = _SEG_H // 4                 # 3 px — phase-correction indicator height
_CORR_TOP    = _AXIS_H + 2                 # top of correction strip (just below header)
_SEGS_TOP    = _CORR_TOP + _CORR_H + 1    # S-meter bars start here
_SEGS_BOTTOM = _WINDOW_H - 4              # 4 px bottom margin
_SEGS_H      = _SEGS_BOTTOM - _SEGS_TOP   # ≈ 190 px for 13 segments

_METER_UPDATE_MS = 200                    # meter poll cadence (matches analyzer LOCKED tick)
_SMOOTH_N        = 5                      # recent results averaged for meter display


def _n_segments_lit(dbm: float) -> int:
    """Return number of segments that should be illuminated for a given dBm reading."""
    return sum(1 for level in _S_LEVELS_DBM if dbm >= level)


def _correction_offset(correction: int, half_w: int, max_corr: int) -> tuple[int, int]:
    """Map a ±max_corr phase correction to (tick width, left offset from bar center).

    0 -> a one-pixel dot at center; otherwise a bar growing left or right,
    scaled so ±max_corr reaches the bar edge.
    """
    if correction == 0:
        return 1, 0
    px = max(1, round(abs(correction) * half_w / max_corr))
    return px, (-px if correction < 0 else 0)


def _aggregate_meter_history(history: Sequence[AnalysisResult]) -> tuple[float, float, bool]:
    """Average recent results into the (nf_dbm, sig_dbm, locked) triple the meters draw.

    Noise averages over every result.  Signal averages only over locked results, so
    unlocked readings (where signal == noise per AnalysisResult.unlocked) don't drag
    the level down mid-window during an intermittent signal; with no locked results
    the signal reading falls back to the noise floor.  An empty history reads as
    silence on both meters.
    """
    if not history:
        return SILENCE_DBFS, SILENCE_DBFS, False
    nf_dbm = sum(r.noise_dbm for r in history) / len(history)
    locked_results = [r for r in history if r.locked]
    if not locked_results:
        return nf_dbm, nf_dbm, False
    return nf_dbm, sum(r.signal_dbm for r in locked_results) / len(locked_results), True


def _mean_spectrum_db(samples: np.ndarray, display_bins: int, db_min: float) -> np.ndarray | None:
    """Mean FFT power in dB across overlapped frames of samples, or None if
    there isn't a single whole frame.

    Averaging every frame captured since the previous display row — instead of
    FFT'ing only the newest one — means no audio goes unrendered: a transient
    shorter than the frame interval still contributes to its row instead of
    falling in the ~68 ms gap a single-frame row would leave.

    Frames advance by _FFT_ADVANCE_SAMPLES and are aligned so the last one ends on
    the newest sample.  Power (|X|^2) is averaged rather than magnitude: Welch
    averaging is defined on power, and averaging magnitude biases the result
    10log10(4/pi) = 1.05 dB low for Gaussian noise while converging more slowly.
    """
    if len(samples) < _CHUNK:
        return None
    n_frames = (len(samples) - _CHUNK) // _FFT_ADVANCE_SAMPLES + 1
    start = len(samples) - ((n_frames - 1) * _FFT_ADVANCE_SAMPLES + _CHUNK)
    frames = np.lib.stride_tricks.sliding_window_view(samples[start:], _CHUNK)[::_FFT_ADVANCE_SAMPLES]
    windowed = frames.astype(np.float32) * _HANN
    spectrum = np.fft.rfft(windowed, axis=1)[:, :display_bins]
    power = (spectrum.real ** 2 + spectrum.imag ** 2).mean(axis=0)
    # log10(0) bins are replaced by db_min via the where(); suppress the -inf warning.
    # 10*log10(power) is the same scale as 20*log10(magnitude), so _DB_REF still applies.
    with np.errstate(divide='ignore'):
        return np.where(power > 0, 10 * np.log10(power) - _DB_REF, db_min)


def _spectrum_percentiles(history_db: np.ndarray, floor_percentile: float,
                          ceiling_percentile: float) -> tuple[float, float]:
    """Raw (floor, ceiling) percentiles of the visible spectrum history, in dB.

    "Raw" meaning unsmoothed — this reads straight off whatever is currently in
    history_db.  The caller is expected to blend these into a slower-moving
    estimate (see _COLOR_RANGE_EMA_ALPHA) rather than render with them directly,
    since one new row of history can move a high percentile like the ceiling
    noticeably on its own.
    """
    return (float(np.percentile(history_db, floor_percentile)),
            float(np.percentile(history_db, ceiling_percentile)))


def _color_scale_range(floor: float, ceiling: float, headroom: float) -> float:
    """dB span such that `ceiling` maps to (1 - headroom) rather than 1.0.

    Callers use the result as: t = clip((value - floor) / range, 0, 1), then index
    the colour map with t.  Reserving `headroom` above the ceiling is what leaves
    room for a spike louder than anything in the recent window to still read as
    hotter than the routine "loud" colour — see _COLOR_HEADROOM.

    The span is floored at _MIN_DYNAMIC_RANGE_DB so a genuinely quiet window can't
    collapse the range down to whatever residual noise-estimator scatter is left in
    the signal — see that constant's comment for the measurement behind the number.
    """
    span = max(ceiling - floor, _MIN_DYNAMIC_RANGE_DB)
    return span / (1.0 - headroom)


def build_colormap() -> np.ndarray:
    """Return a 256×3 uint8 RGB lookup table: black → blue → cyan → yellow → red."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            s = t * 4
            lut[i] = [0, 0, int(s * 180)]
        elif t < 0.5:
            s = (t - 0.25) * 4
            lut[i] = [0, int(s * 255), int(180 + s * 75)]
        elif t < 0.75:
            s = (t - 0.5) * 4
            lut[i] = [int(s * 255), 255, int((1 - s) * 255)]
        else:
            s = (t - 0.75) * 4
            lut[i] = [255, int((1 - s) * 255), 0]
    return lut


_COLORMAP = build_colormap()


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class WaterfallWidget(QWidget):  # pragma: no cover -- requires a live Qt display
    """Scrolling FFT spectrogram, updated by a QTimer at ~10 fps.

    Each frame renders the averaged FFT of all audio captured since the previous
    frame, so the display covers the full timeline rather than sampling it.
    """

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        # Bin geometry follows the configured sample rate: each FFT bin spans
        # sample_rate/_CHUNK Hz, and we display 0–_MAX_HZ (clamped to the rfft
        # output length for sample rates below 2×_MAX_HZ).
        sample_rate = config.audio.sample_rate
        self._hz_per_bin = sample_rate / _CHUNK
        self._display_bins = min(_MAX_HZ * _CHUNK // sample_rate, _CHUNK // 2)
        self._label_interval = max(1, round(_FREQ_LABEL_HZ / self._hz_per_bin))
        # First guess for the colour floor, from the station's configured
        # calibration.  It only has to hold up for the first couple of seconds —
        # _update_color_range() replaces it with the real, live level once actual
        # data starts arriving.
        floor_seed = (config.station.noise_floor
                      - config.station.audio_rf_conversion_db
                      - _DB_FFT_NOISE_CORR)
        # Audio pulled per display row, rounded up to whole chunks so consecutive
        # rows overlap slightly (4 chunks = 128 ms per 100 ms row at 16 kHz) rather
        # than leaving gaps.  _mean_spectrum_db then splits this into 50%-overlapped
        # FFT frames (7 frames of 512 for 2048 samples).
        self._frame_chunks = ceil(_UPDATE_MS / 1000 * sample_rate / _CHUNK)
        self._last_total_samples = 0
        self._history_db = np.full((_N_ROWS, self._display_bins), floor_seed, dtype=np.float32)
        # Smoothed colour floor/ceiling that paintEvent renders with; see
        # _update_color_range(), which is what actually keeps these current.
        self._color_floor = floor_seed
        self._color_ceiling = floor_seed + _DB_RANGE
        self._color_range = _color_scale_range(
            self._color_floor, self._color_ceiling, _COLOR_HEADROOM)
        self.setFixedSize(self._display_bins * _PIXELS_PER_BIN, _WATERFALL_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_UPDATE_MS)

    def _tick(self) -> None:
        # If the stream has stalled, freeze the display rather than scrolling
        # duplicated rows of stale audio (which would fabricate repeated time).
        total = self._pipeline.total_samples
        if total == self._last_total_samples:
            return
        self._last_total_samples = total

        samples = self._pipeline.get_snapshot(self._frame_chunks * _CHUNK)
        db = _mean_spectrum_db(samples, self._display_bins, self._color_floor)
        if db is None:
            return
        self._history_db[1:] = self._history_db[:-1]
        self._history_db[0] = db
        self._update_color_range()
        self.update()

    def _update_color_range(self) -> None:
        """Blend this tick's floor/ceiling percentiles into the smoothed values
        paintEvent renders with.  See _COLOR_RANGE_EMA_ALPHA for why the raw
        per-tick percentiles aren't used directly.
        """
        floor, ceiling = _spectrum_percentiles(
            self._history_db, _COLOR_FLOOR_PERCENTILE, _COLOR_CEILING_PERCENTILE)
        a = _COLOR_RANGE_EMA_ALPHA
        self._color_floor   += a * (floor - self._color_floor)
        self._color_ceiling += a * (ceiling - self._color_ceiling)
        self._color_range = _color_scale_range(
            self._color_floor, self._color_ceiling, _COLOR_HEADROOM)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        w = self.width()

        # Frequency axis
        painter.fillRect(0, 0, w, _AXIS_H, QColor(30, 30, 30))
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Monospace', 8))
        bin_px = w / self._display_bins
        # Stop before _display_bins: a tick at that bin sits at x == width, drawing
        # its label off the right edge of the widget.
        for b in range(0, self._display_bins, self._label_interval):
            x = int(b * bin_px)
            painter.drawLine(x, _AXIS_H - 5, x, _AXIS_H)
            painter.drawText(x + 2, _AXIS_H - 6, f'{int(b * self._hz_per_bin)} Hz')

        # Waterfall — each row is exactly _PIXELS_PER_ROW pixels tall.
        # Map dB-above-floor onto the 0–255 colormap index range: _color_floor
        # renders as the colormap's cold end, _color_floor + _color_range as hot.
        # Both are auto-ranged from live data — see _update_color_range().
        norm = np.clip(
            (self._history_db - self._color_floor) / self._color_range * 255, 0, 255,
        ).astype(np.uint8)
        used_h = _N_ROWS * _PIXELS_PER_ROW
        rgb_rows = np.ascontiguousarray(_COLORMAP[norm].repeat(_PIXELS_PER_ROW, axis=0))
        img = QImage(rgb_rows.data, self._display_bins, used_h,
                     self._display_bins * 3, QImage.Format.Format_RGB888)
        # Scale horizontally (bins → _PIXELS_PER_BIN px each); used_h is already exact so no vertical scaling.
        scaled = img.scaled(w, used_h,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.FastTransformation)
        painter.drawImage(0, _AXIS_H, scaled)

    def stop(self) -> None:
        self._timer.stop()


class MeterPanelWidget(QWidget):  # pragma: no cover -- requires a live Qt display
    """Pair of vertical S-band bar-graph meters: noise floor (left) and signal (right).

    _tick() polls the analyzer and reduces the recent history to the displayed
    (nf_dbm, sig_dbm, locked) values; paintEvent() only draws them.
    """

    def __init__(self, analyzer: ContinuousAnalyzer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._analyzer = analyzer
        self._history: deque[AnalysisResult] = deque(maxlen=_SMOOTH_N)
        self._nf_dbm, self._sig_dbm, self._locked = _aggregate_meter_history(())
        self.setFixedSize(_PANEL_W, _WINDOW_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_METER_UPDATE_MS)

    def _tick(self) -> None:
        result = self._analyzer.latest_result()
        if result is not None:
            self._history.append(result)
        self._nf_dbm, self._sig_dbm, self._locked = _aggregate_meter_history(self._history)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), QColor(20, 20, 20))

        # Column x-coordinates
        nf_x  = _PAD
        lbl_x = _PAD + _BAR_W + _PAD
        sig_x = _PAD + _BAR_W + _PAD + _LABEL_W + _PAD

        # Header row (matches waterfall axis bar)
        painter.fillRect(0, 0, self.width(), _AXIS_H, QColor(30, 30, 30))
        painter.setFont(QFont('Monospace', 7, QFont.Weight.Bold))
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(nf_x, 0, _BAR_W, _AXIS_H,
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, 'NF')
        painter.drawText(sig_x, 0, _BAR_W, _AXIS_H,
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, 'SIG')

        nf_lit  = _n_segments_lit(self._nf_dbm)
        sig_lit = _n_segments_lit(self._sig_dbm)
        locked  = self._locked

        # Phase-correction indicators — above each bar, independent per NF/SIG since
        # the noise and signal phases are searched (and can drift) independently —
        # see ContinuousAnalyzer._phase_search().  Range: ±PHASE_SEARCH_RADIUS
        # (10 samples) each.  0 → one-pixel dot at center.
        grey       = QColor(160, 160, 160)
        half_w     = _BAR_W // 2
        max_corr   = ContinuousAnalyzer.PHASE_SEARCH_RADIUS
        line_y     = _CORR_TOP + _CORR_H // 2
        noise_px, noise_offset   = _correction_offset(
            self._analyzer.latest_noise_correction(), half_w, max_corr)
        signal_px, signal_offset = _correction_offset(
            self._analyzer.latest_signal_correction(), half_w, max_corr)
        painter.fillRect(nf_x + half_w + noise_offset, line_y, noise_px, 1, grey)
        painter.fillRect(sig_x + half_w + signal_offset, line_y, signal_px, 1, grey)

        # Integer layout: anchor each bar from the bottom so the leftover pixels
        # from the integer gap division fall above the top bar, not below S1.
        base_gap = (_SEGS_H - _N_SEGS * _SEG_H) // (_N_SEGS - 1)

        painter.setFont(QFont('Monospace', 7))

        for i in range(_N_SEGS):
            # i=0 → S1 (bottom), i=12 → S9+40 (top)
            y = _SEGS_BOTTOM - (i + 1) * _SEG_H - i * base_gap

            lit_rgb = _SEG_LIT[i]
            dim_rgb = _SEG_DIM[i]

            # Noise bar
            r, g, b = lit_rgb if i < nf_lit else dim_rgb
            painter.fillRect(nf_x, y, _BAR_W, _SEG_H, QColor(r, g, b))

            # Signal bar (dimmer when unlocked)
            if locked:
                r, g, b = lit_rgb if i < sig_lit else dim_rgb
            else:
                # Show last known reading at 50% brightness when not locked
                base = lit_rgb if i < sig_lit else dim_rgb
                r, g, b = base[0] // 2, base[1] // 2, base[2] // 2
            painter.fillRect(sig_x, y, _BAR_W, _SEG_H, QColor(r, g, b))

            # S-unit label between bars (every other label to avoid crowding)
            if i % 2 == 0 or i >= 9:
                painter.setPen(QColor(160, 160, 160))
                painter.drawText(lbl_x, y, _LABEL_W, _SEG_H,
                                 Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                                 _S_LABELS[i])

    def stop(self) -> None:
        self._timer.stop()


class MainWindow(QMainWindow):  # pragma: no cover -- requires a live Qt display
    """Top-level window; closing it shuts down the audio pipeline and analyzer.

    Layout is a scope over a waterfall on the left, with the S-meter panel running
    the full height on the right:

        +-------------------------+--------+
        |  ScopeWidget            |        |
        |  (SCOPE_H)              |        |
        +----- _PANEL_GAP --------+ Meters |
        |  WaterfallWidget        |        |
        |  (_WATERFALL_H)         |        |
        +-------------------------+--------+
    """

    def __init__(self, pipeline: AudioPipeline, analyzer: ContinuousAnalyzer,
                 config: BuzzConfig, always_on_top: bool = False) -> None:
        super().__init__()
        self.setWindowTitle('N6OL Powerline QRM Monitor')
        if always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._pipeline = pipeline
        self._analyzer = analyzer

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_PANEL_GAP)   # horizontal pad before the meter column

        self._waterfall = WaterfallWidget(pipeline, config)
        # The scope takes its width from the waterfall rather than recomputing the
        # bin geometry, so the two panels cannot drift out of alignment if the
        # sample rate or displayed bandwidth ever changes.
        self._scope  = ScopeWidget(pipeline, analyzer, config, self._waterfall.width())
        self._meters = MeterPanelWidget(analyzer)

        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(_PANEL_GAP)
        stack.addWidget(self._scope)
        stack.addWidget(self._waterfall)

        layout.addLayout(stack)
        layout.addWidget(self._meters)

        self.setCentralWidget(container)
        self.setFixedSize(
            self._waterfall.width() + _PANEL_GAP + self._meters.width(), _WINDOW_H)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """A toggles the scope between raw-persistence and averaged-envelope modes."""
        if event.key() == Qt.Key.Key_A:
            self._scope.toggle_mode()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._scope.stop()
        self._waterfall.stop()
        self._meters.stop()
        self._analyzer.stop()
        self._pipeline.close()
        event.accept()

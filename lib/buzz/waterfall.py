"""
Waterfall display, S-band meter panel, and main application window.

WaterfallWidget renders a scrolling FFT spectrogram driven by a QTimer.
MeterPanelWidget draws a pair of vertical bar-graph S-meters (noise floor
and signal level) updated by polling the ContinuousAnalyzer result slot.
MainWindow composes both widgets side-by-side and handles clean shutdown.
"""

from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from buzz.analyzer import AnalysisResult, ContinuousAnalyzer
from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline

# ---------------------------------------------------------------------------
# Waterfall constants
# ---------------------------------------------------------------------------

_CHUNK = AudioPipeline.CHUNK_SIZE           # 512 samples
_SAMPLE_RATE = 16000
_MAX_HZ = 4000
_DISPLAY_BINS = _MAX_HZ * _CHUNK // _SAMPLE_RATE   # 128 bins = 0–4000 Hz
_FREQ_LABEL_INTERVAL = 16                   # tick every 16 bins = every 500 Hz
_N_ROWS = 100                               # history rows (~10 s at 100 ms/frame)
_PIXELS_PER_ROW = 2                         # must divide (window_height - _AXIS_H) evenly
_UPDATE_MS = 100
_DB_RANGE = 48.0                            # colour scale dynamic range in dB (8 S-units)
_DB_REF = 20 * np.log10(32768.0 * _CHUNK / 4)  # 0 dBFS for a full-scale Hann-windowed sinusoid
# Broadband noise spreads across CHUNK/2 bins; each bin sits ~23 dB below the
# time-domain power that the level meter measures.  Subtract this so the noise
# floor anchor applies to per-bin energy, not total broadband power.
_DB_FFT_NOISE_CORR = float(10 * np.log10(_CHUNK * 3 / 8))  # ≈ 22.8 dB for Hann
_AXIS_H = 24                                # pixels reserved for frequency axis / header

_HANN = np.hanning(_CHUNK).astype(np.float32)

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

_SEG_H    = 13   # segment bar height in pixels
_SEG_GAP  = 2    # gap between segments
_BAR_W    = 22   # width of each meter bar
_LABEL_W  = 30   # width of shared S-unit label column
_PAD      = 3    # outer and inner horizontal padding
# Panel width: PAD + BAR + PAD + LABEL + PAD + BAR + PAD
_PANEL_W  = _PAD + _BAR_W + _PAD + _LABEL_W + _PAD + _BAR_W + _PAD  # 86 px

_WATERFALL_W = _DISPLAY_BINS * 5            # 640 px (5 px/bin, no scaling)
_WINDOW_W    = _WATERFALL_W + _PANEL_W
_WINDOW_H    = _N_ROWS * _PIXELS_PER_ROW + _AXIS_H  # 224 px

# Segment area geometry (within the panel widget)
_CORR_H      = _SEG_H // 4                 # 3 px — phase-correction indicator height
_CORR_TOP    = _AXIS_H + 2                 # top of correction strip (just below header)
_SEGS_TOP    = _CORR_TOP + _CORR_H + 1    # S-meter bars start here
_SEGS_BOTTOM = _WINDOW_H - 4
_SEGS_H      = _SEGS_BOTTOM - _SEGS_TOP   # ≈ 190 px for 13 segments


def _n_lit(dbm: float) -> int:
    """Return number of segments that should be illuminated for a given dBm reading."""
    return sum(1 for level in _S_LEVELS_DBM if dbm >= level)


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

class WaterfallWidget(QWidget):
    """Scrolling FFT spectrogram, updated by a QTimer at ~10 fps."""

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._db_min = (config.station.noise_floor
                        - config.station.audio_rf_conversion_db
                        - _DB_FFT_NOISE_CORR)
        self._waterfall = np.full((_N_ROWS, _DISPLAY_BINS), self._db_min, dtype=np.float32)
        self.setFixedSize(_WATERFALL_W, _WINDOW_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_UPDATE_MS)

    def _tick(self) -> None:
        chunk = self._pipeline.latest_chunk()
        if chunk is None:
            return
        windowed = chunk.astype(np.float32) * _HANN
        spectrum = np.abs(np.fft.rfft(windowed, n=_CHUNK))[:_DISPLAY_BINS]
        db = np.where(spectrum > 0, 20 * np.log10(spectrum) - _DB_REF, self._db_min)
        self._waterfall[1:] = self._waterfall[:-1]
        self._waterfall[0] = db.astype(np.float32)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        w = self.width()

        # Frequency axis
        painter.fillRect(0, 0, w, _AXIS_H, QColor(30, 30, 30))
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Monospace', 8))
        hz_per_bin = _SAMPLE_RATE / _CHUNK
        bin_px = w / _DISPLAY_BINS
        for b in range(0, _DISPLAY_BINS + 1, _FREQ_LABEL_INTERVAL):
            x = int(b * bin_px)
            painter.drawLine(x, _AXIS_H - 5, x, _AXIS_H)
            painter.drawText(x + 2, _AXIS_H - 6, f'{int(b * hz_per_bin)} Hz')

        # Waterfall — each row is exactly _PIXELS_PER_ROW pixels tall
        norm = np.clip(
            (self._waterfall - self._db_min) / _DB_RANGE * 255, 0, 255,
        ).astype(np.uint8)
        used_h = _N_ROWS * _PIXELS_PER_ROW
        rgb_rows = np.ascontiguousarray(_COLORMAP[norm].repeat(_PIXELS_PER_ROW, axis=0))
        img = QImage(rgb_rows.data, _DISPLAY_BINS, used_h,
                     _DISPLAY_BINS * 3, QImage.Format.Format_RGB888)
        # Scale horizontally (128 bins → 640 px); used_h is already exact so no vertical scaling.
        scaled = img.scaled(w, used_h,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.FastTransformation)
        painter.drawImage(0, _AXIS_H, scaled)

    def stop(self) -> None:
        self._timer.stop()


class MeterPanelWidget(QWidget):
    """Pair of vertical S-band bar-graph meters: noise floor (left) and signal (right)."""

    _METER_UPDATE_MS = 200

    _SMOOTH_N = 5   # number of recent results to average for display

    def __init__(self, analyzer: ContinuousAnalyzer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._analyzer = analyzer
        self._history: deque[AnalysisResult] = deque(maxlen=self._SMOOTH_N)
        self.setFixedSize(_PANEL_W, _WINDOW_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._METER_UPDATE_MS)

    def _tick(self) -> None:
        result = self._analyzer.latest_result()
        if result is not None:
            self._history.append(result)
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

        if self._history:
            n          = len(self._history)
            nf_dbm     = sum(r.noise_dbm for r in self._history) / n
            # Average signal only over locked results so unlocked readings
            # (where signal_dbm == noise_dbm) don't drag the level down
            # mid-window during an intermittent signal.
            locked_rs  = [r for r in self._history if r.locked]
            if locked_rs:
                sig_dbm = sum(r.signal_dbm for r in locked_rs) / len(locked_rs)
                locked  = True
            else:
                sig_dbm = nf_dbm
                locked  = False
        else:
            nf_dbm  = -128.0
            sig_dbm = -128.0
            locked  = False

        nf_lit  = _n_lit(nf_dbm)
        sig_lit = _n_lit(sig_dbm)

        # Phase-correction indicator — above both bars.
        # Range: ±PHASE_SEARCH_RADIUS (10 samples).  0 → one-pixel dot at center.
        correction = self._analyzer.latest_correction()
        grey       = QColor(160, 160, 160)
        half_w     = _BAR_W // 2
        max_corr   = ContinuousAnalyzer.PHASE_SEARCH_RADIUS
        line_y     = _CORR_TOP + _CORR_H // 2
        if correction == 0:
            px, left_offset = 1, 0
        else:
            px          = max(1, round(abs(correction) * half_w / max_corr))
            left_offset = -px if correction < 0 else 0
        for bar_x in (nf_x, sig_x):
            painter.fillRect(bar_x + half_w + left_offset, line_y, px, 1, grey)

        # Integer layout: anchor each bar from the bottom so the 1 px remainder
        # from 25 px of total gap / 12 spaces falls above the top bar, not below S1.
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
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 _S_LABELS[i])

    def stop(self) -> None:
        self._timer.stop()


class MainWindow(QMainWindow):
    """Top-level window; closing it shuts down the audio pipeline and analyzer."""

    def __init__(self, pipeline: AudioPipeline, analyzer: ContinuousAnalyzer,
                 config: BuzzConfig) -> None:
        super().__init__()
        self.setWindowTitle('N6OL QRM Monitor')
        self._pipeline = pipeline
        self._analyzer = analyzer

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._waterfall = WaterfallWidget(pipeline, config)
        self._meters    = MeterPanelWidget(analyzer)

        layout.addWidget(self._waterfall)
        layout.addWidget(self._meters)

        self.setCentralWidget(container)
        self.setFixedSize(_WINDOW_W, _WINDOW_H)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._waterfall.stop()
        self._meters.stop()
        self._analyzer.stop()
        self._pipeline.close()
        event.accept()

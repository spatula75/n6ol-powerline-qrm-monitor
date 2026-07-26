"""
Waterfall display and main application window for the QRM monitor.

WaterfallWidget renders a scrolling FFT spectrogram driven by a QTimer.
Each tick it grabs the latest chunk from the AudioPipeline ring buffer,
applies a Hann window, computes the magnitude spectrum, maps it to colour
via a blue-to-red lookup table, and prepends a new row to the rolling 2-D
array that is painted as a QImage scaled to the widget size.

MainWindow wraps the widget and handles clean shutdown: closing the window
stops the update timer, closes the audio pipeline, and lets the daemon
collector thread exit with the process.
"""

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QMainWindow, QWidget

from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline

_CHUNK = AudioPipeline.CHUNK_SIZE           # 512 samples
_SAMPLE_RATE = 16000
_MAX_HZ = 4000
_DISPLAY_BINS = _MAX_HZ * _CHUNK // _SAMPLE_RATE   # 128 bins = 0–4000 Hz
_FREQ_LABEL_INTERVAL = 16                   # tick every 16 bins = every 500 Hz
_N_ROWS = 100                               # history rows (~10 s at 100 ms/frame)
_PIXELS_PER_ROW = 2                         # must divide (window_height - _AXIS_H) evenly
_UPDATE_MS = 100
_DB_RANGE = 48.0                            # dynamic range of the colour scale in dB (8 S-units)
_DB_REF = 20 * np.log10(32768.0 * _CHUNK / 4)  # peak magnitude of a full-scale Hann-windowed sinusoid
# Broadband noise spreads across CHUNK/2 bins; each bin sits ~23 dB below the
# time-domain power that the level meter measures.  Subtract this so the noise
# floor anchor applies to per-bin energy, not total broadband power.
_DB_FFT_NOISE_CORR = float(10 * np.log10(_CHUNK * 3 / 8))  # ≈ 22.8 dB for Hann
_AXIS_H = 24                                # pixels reserved for frequency axis

_HANN = np.hanning(_CHUNK).astype(np.float32)


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


class WaterfallWidget(QWidget):
    """Scrolling FFT spectrogram widget, updated by a QTimer at ~10 fps."""

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        # Cold end of the colour scale: convert the configured noise floor from dBm
        # to the FFT dBFS scale by adding the station calibration offset.
        self._db_min = config.station.noise_floor - config.station.audio_rf_conversion_db - _DB_FFT_NOISE_CORR
        self._db_max = self._db_min + _DB_RANGE
        self._waterfall = np.full((_N_ROWS, _DISPLAY_BINS), self._db_min, dtype=np.float32)
        self.setMinimumSize(640, 300)

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

        # Waterfall image — each row is exactly _PIXELS_PER_ROW pixels tall.
        norm = np.clip(
            (self._waterfall - self._db_min) / _DB_RANGE * 255, 0, 255,
        ).astype(np.uint8)
        used_h = _N_ROWS * _PIXELS_PER_ROW
        rgb_rows = np.ascontiguousarray(_COLORMAP[norm].repeat(_PIXELS_PER_ROW, axis=0))
        img = QImage(rgb_rows.data, _DISPLAY_BINS, used_h,
                     _DISPLAY_BINS * 3, QImage.Format.Format_RGB888)
        scaled = img.scaled(w, used_h,
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.FastTransformation)
        painter.drawImage(0, _AXIS_H, scaled)

    def stop(self) -> None:
        self._timer.stop()


class MainWindow(QMainWindow):
    """Top-level window; stopping it shuts down the audio pipeline cleanly."""

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig) -> None:
        super().__init__()
        self.setWindowTitle('N6OL QRM Monitor — Waterfall')
        self._pipeline = pipeline
        self._widget = WaterfallWidget(pipeline, config)
        self.setCentralWidget(self._widget)
        self.setFixedSize(640, _N_ROWS * _PIXELS_PER_ROW + _AXIS_H)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._widget.stop()
        self._pipeline.close()
        event.accept()

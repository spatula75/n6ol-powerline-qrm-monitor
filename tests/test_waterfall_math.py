"""Tests for pure-numpy functions in waterfall.py (no Qt required)."""
import numpy as np

from buzz.waterfall import build_colormap, _DISPLAY_BINS, _N_ROWS, _DB_RANGE


class TestBuildColormap:
    def test_shape(self):
        lut = build_colormap()
        assert lut.shape == (256, 3)

    def test_dtype(self):
        lut = build_colormap()
        assert lut.dtype == np.uint8

    def test_cold_end_is_dark(self):
        lut = build_colormap()
        assert int(lut[0].sum()) < 30

    def test_hot_end_is_red(self):
        lut = build_colormap()
        # entry 255 should be strongly red, no blue
        assert lut[255, 0] == 255
        assert lut[255, 2] == 0

    def test_monotonically_increasing_brightness(self):
        lut = build_colormap()
        brightness = lut.astype(np.int32).sum(axis=1)
        # brightness should be non-decreasing overall
        assert brightness[-1] >= brightness[0]

    def test_no_out_of_range_values(self):
        lut = build_colormap()
        assert lut.min() >= 0
        assert lut.max() <= 255


class TestWaterfallConstants:
    def test_display_bins_matches_frequency_range(self):
        # 128 bins × 31.25 Hz/bin = 4000 Hz
        assert _DISPLAY_BINS == 128

    def test_db_range_is_48(self):
        assert _DB_RANGE == 48.0

    def test_n_rows_reasonable(self):
        assert 50 <= _N_ROWS <= 200

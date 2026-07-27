"""Tests for pure-numpy functions in waterfall.py (no Qt required)."""
import numpy as np

from buzz.analyzer import AnalysisResult
from buzz.dsp import SILENCE_DBFS
from buzz.waterfall import (
    build_colormap, _aggregate_meter_history, _correction_offset,
    _CHUNK, _MAX_HZ, _N_ROWS, _DB_RANGE,
)


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


class TestCorrectionOffset:
    def test_zero_correction_is_one_pixel_dot_centered(self):
        assert _correction_offset(0, half_w=11, max_corr=10) == (1, 0)

    def test_positive_correction_grows_right_from_center(self):
        px, offset = _correction_offset(5, half_w=11, max_corr=10)
        assert px > 1
        assert offset == 0

    def test_negative_correction_grows_left_from_center(self):
        px, offset = _correction_offset(-5, half_w=11, max_corr=10)
        assert px > 1
        assert offset == -px

    def test_max_correction_reaches_bar_edge(self):
        px, offset = _correction_offset(10, half_w=11, max_corr=10)
        assert px == 11

    def test_negative_and_positive_same_magnitude_give_same_width(self):
        px_pos, _ = _correction_offset(4, half_w=11, max_corr=10)
        px_neg, _ = _correction_offset(-4, half_w=11, max_corr=10)
        assert px_pos == px_neg

    def test_small_nonzero_correction_still_at_least_one_pixel(self):
        px, _ = _correction_offset(1, half_w=11, max_corr=10)
        assert px >= 1


def _locked(signal_dbm: float, noise_dbm: float) -> AnalysisResult:
    return AnalysisResult(signal_dbm=signal_dbm, noise_dbm=noise_dbm,
                          snr=signal_dbm - noise_dbm, locked=True)


class TestAggregateMeterHistory:
    def test_empty_history_reads_as_silence(self):
        assert _aggregate_meter_history([]) == (SILENCE_DBFS, SILENCE_DBFS, False)

    def test_all_locked_averages_both_channels(self):
        history = [_locked(-70.0, -90.0), _locked(-80.0, -100.0)]
        nf_dbm, sig_dbm, locked = _aggregate_meter_history(history)
        assert nf_dbm == -95.0
        assert sig_dbm == -75.0
        assert locked is True

    def test_signal_averages_only_locked_results(self):
        history = [_locked(-70.0, -90.0), AnalysisResult.unlocked(-90.0)]
        nf_dbm, sig_dbm, locked = _aggregate_meter_history(history)
        assert sig_dbm == -70.0   # unlocked reading excluded from the signal average
        assert locked is True

    def test_noise_averages_all_results(self):
        history = [_locked(-70.0, -90.0), AnalysisResult.unlocked(-94.0)]
        nf_dbm, _, _ = _aggregate_meter_history(history)
        assert nf_dbm == -92.0

    def test_all_unlocked_signal_falls_back_to_noise(self):
        history = [AnalysisResult.unlocked(-90.0), AnalysisResult.unlocked(-92.0)]
        nf_dbm, sig_dbm, locked = _aggregate_meter_history(history)
        assert sig_dbm == nf_dbm == -91.0
        assert locked is False


class TestWaterfallConstants:
    def test_display_bins_formula_at_16k(self):
        # At 16 kHz: 128 bins × 31.25 Hz/bin = 4000 Hz
        assert _MAX_HZ * _CHUNK // 16000 == 128

    def test_db_range_is_48(self):
        assert _DB_RANGE == 48.0

    def test_n_rows_reasonable(self):
        assert 50 <= _N_ROWS <= 200

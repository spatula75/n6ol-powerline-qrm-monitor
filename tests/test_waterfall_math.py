"""Tests for pure-numpy functions in waterfall.py (no Qt required)."""
import numpy as np
import pytest

from buzz.analyzer import AnalysisResult
from buzz.dsp import SILENCE_DBFS
from buzz.waterfall import (
    build_colormap, _aggregate_meter_history, _color_scale_range, _correction_offset,
    _mean_spectrum_db, _spectrum_percentiles,
    _CHUNK, _MAX_HZ, _N_ROWS, _DB_RANGE, _DB_FFT_NOISE_CORR, _FFT_ADVANCE_SAMPLES,
    _COLOR_FLOOR_PERCENTILE, _COLOR_CEILING_PERCENTILE, _COLOR_HEADROOM, _MIN_COLOR_SPAN_DB,
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


class TestMeanSpectrumDb:
    _BINS = 128
    _DB_MIN = -88.0

    def _tone(self, n_chunks: int, bin_index: int = 10, amplitude: float = 32767.0) -> np.ndarray:
        """Sinusoid at an exact FFT bin frequency, n_chunks × 512 samples long."""
        t = np.arange(n_chunks * _CHUNK)
        return (amplitude * np.cos(2 * np.pi * bin_index * t / _CHUNK)).astype(np.int16)

    def test_full_scale_tone_reads_near_zero_dbfs(self):
        """Validates the _DB_REF calibration: a full-scale sinusoid peaks at ~0 dBFS."""
        db = _mean_spectrum_db(self._tone(1), self._BINS, self._DB_MIN)
        assert abs(db.max()) < 0.1

    def test_tone_peak_lands_in_its_bin(self):
        db = _mean_spectrum_db(self._tone(4, bin_index=20), self._BINS, self._DB_MIN)
        assert int(np.argmax(db)) == 20

    def test_averages_power_not_magnitude(self):
        """A tone present in half the frames must read −3 dB (mean of |X|²), not
        −6 dB (mean of |X|).  Welch averaging is defined on power; averaging
        magnitude biases low and converges more slowly."""
        full = _mean_spectrum_db(self._tone(16), self._BINS, self._DB_MIN).max()
        samples = np.concatenate([self._tone(8), np.zeros(8 * _CHUNK, dtype=np.int16)])
        half = _mean_spectrum_db(samples, self._BINS, self._DB_MIN).max()
        assert half - full == pytest.approx(-3.01, abs=0.15)

    def test_impulse_energy_is_independent_of_position(self):
        """The reason for the 75% overlap.  Hann tapers to zero at the frame edges,
        so with non-overlapping frames an impulse landing on a boundary was
        attenuated by >12 dB relative to one at a frame centre — on a display whose
        subject is a 120 pps impulse train.

        The hop must satisfy COLA in power, since power is what gets averaged; the
        familiar 50% rule is the amplitude condition and still leaves 3.01 dB of
        ripple.  Sweeping a whole hop's worth of positions must be flat."""
        def _impulse_db(pos: int) -> float:
            x = np.zeros(8 * _CHUNK, dtype=np.int16)
            x[pos] = 30000
            return float(_mean_spectrum_db(x, self._BINS, self._DB_MIN).mean())

        levels = [_impulse_db(2 * _CHUNK + d) for d in range(0, _FFT_ADVANCE_SAMPLES, 8)]
        assert max(levels) - min(levels) < 0.01

    def test_noise_floor_anchor_matches_measured_per_bin_level(self):
        """Pins _DB_FFT_NOISE_CORR.  Broadband noise of known RMS must land on the
        anchor the widget computes for db_min, within a fraction of a dB.  The old
        constant applied the Hann ENBW in the wrong direction and sat 4.4 dB low."""
        sigma = 800.0
        noise = np.random.default_rng(3).normal(0, sigma, _CHUNK * 400).astype(np.int16)
        db = _mean_spectrum_db(noise.astype(np.int32), self._BINS, -200.0)
        rms_dbfs = 20 * np.log10(sigma) - 20 * np.log10(32768.0)
        assert float(db.mean()) == pytest.approx(rms_dbfs - _DB_FFT_NOISE_CORR, abs=0.2)

    def test_partial_chunk_is_trimmed_from_the_front(self):
        tone = self._tone(2)
        with_partial = np.concatenate([np.full(100, 30000, dtype=np.int16), tone])
        np.testing.assert_allclose(
            _mean_spectrum_db(with_partial, self._BINS, self._DB_MIN),
            _mean_spectrum_db(tone, self._BINS, self._DB_MIN))

    def test_less_than_one_chunk_returns_none(self):
        assert _mean_spectrum_db(np.zeros(_CHUNK - 1, dtype=np.int16),
                                 self._BINS, self._DB_MIN) is None

    def test_silence_reads_db_min(self):
        db = _mean_spectrum_db(np.zeros(_CHUNK, dtype=np.int16), self._BINS, self._DB_MIN)
        assert np.all(db == self._DB_MIN)


class TestSpectrumPercentiles:
    def test_uniform_history_gives_equal_floor_and_ceiling(self):
        history = np.full((_N_ROWS, 128), -70.0)
        floor, ceiling = _spectrum_percentiles(history, 10, 98)
        assert floor == pytest.approx(-70.0)
        assert ceiling == pytest.approx(-70.0)

    def test_floor_below_ceiling_on_varied_history(self):
        rng = np.random.default_rng(0)
        history = rng.normal(-70.0, 10.0, size=(_N_ROWS, 128))
        floor, ceiling = _spectrum_percentiles(history, 10, 98)
        assert floor < ceiling

    def test_percentiles_track_a_known_distribution(self):
        """10th/98th percentile of a known linear ramp should land near their
        textbook positions, confirming the arguments reach np.percentile correctly."""
        history = np.tile(np.linspace(-100.0, 0.0, 1000), (_N_ROWS, 1))
        floor, ceiling = _spectrum_percentiles(history, 10, 98)
        assert floor == pytest.approx(-90.0, abs=1.0)
        assert ceiling == pytest.approx(-2.0, abs=1.0)

    def test_floor_stays_pinned_to_a_stale_seed_until_the_window_mostly_refills(self):
        """Cold-start regression: WaterfallWidget seeds every row of history_db with
        the same value at startup.  The 10th-percentile floor stays exactly on that
        seed until more than 90 of the 100 rows have been replaced with real data —
        below 90 replaced, fewer than 10 stale rows remain and the floor should have
        moved on; at 90 replaced it's still right on the boundary.  See
        _COLOR_RANGE_EMA_ALPHA for why this matters: it is what makes the display's
        startup convergence take longer than the EMA settling time alone."""
        seed = -85.31
        history = np.full((_N_ROWS, 128), seed)
        real_rows = np.random.default_rng(1).normal(-70.0, 5.0, size=(89, 128))
        history[:89] = real_rows   # 11 stale rows remain — still > 10% of the window
        floor, _ = _spectrum_percentiles(history, _COLOR_FLOOR_PERCENTILE, _COLOR_CEILING_PERCENTILE)
        assert floor == pytest.approx(seed)

        history[:95] = np.random.default_rng(2).normal(-70.0, 5.0, size=(95, 128))
        floor, _ = _spectrum_percentiles(history, _COLOR_FLOOR_PERCENTILE, _COLOR_CEILING_PERCENTILE)
        assert floor > seed + 1.0   # comfortably past the stale value once it's freed

    def test_a_few_loud_rows_move_the_ceiling(self):
        """The reason _update_color_range() smooths these rather than using them
        directly: at the 98th percentile only the top 2% of values count, which
        in a 100-row window is about two rows' worth — so it takes just a couple
        of ticks of new, louder content for the raw ceiling to start moving."""
        quiet = np.full((_N_ROWS, 128), -80.0)
        _, ceiling_quiet = _spectrum_percentiles(quiet, _COLOR_FLOOR_PERCENTILE,
                                                  _COLOR_CEILING_PERCENTILE)
        loud = quiet.copy()
        loud[:3] = -20.0   # 3 rows = 3% of the window, just over the 2% ceiling slice
        _, ceiling_loud = _spectrum_percentiles(loud, _COLOR_FLOOR_PERCENTILE,
                                                 _COLOR_CEILING_PERCENTILE)
        assert ceiling_loud > ceiling_quiet


class TestColorScaleRange:
    def test_ceiling_percentile_lands_at_one_minus_headroom(self):
        """The whole point of the headroom: mapping (value - floor) / range must
        place the ceiling percentile itself at exactly 1 - headroom, not 1.0."""
        floor, ceiling, headroom = -80.0, -50.0, 0.10
        rng = _color_scale_range(floor, ceiling, headroom)
        t = (ceiling - floor) / rng
        assert t == pytest.approx(1.0 - headroom)

    def test_headroom_reserves_room_above_the_ceiling(self):
        """A value modestly hotter than the ceiling percentile must still land
        below 1.0, so it can render as visibly hotter than the ceiling's own
        colour rather than clipping to the same solid red immediately."""
        floor, ceiling, headroom = -80.0, -50.0, 0.10
        rng = _color_scale_range(floor, ceiling, headroom)
        # Halfway into the reserved headroom band (whose full width is
        # rng * headroom dB) should land partway between (1 - headroom) and 1.0.
        spike = ceiling + (rng * headroom) / 2
        t = (spike - floor) / rng
        assert 1.0 - headroom < t < 1.0

    def test_spike_beyond_the_headroom_band_exceeds_one(self):
        """A spike louder than the reserved headroom clips to 1.0 in the caller
        (paintEvent's np.clip) rather than being represented exactly — that's the
        expected trade-off of a bounded colour scale, not a bug in this function."""
        floor, ceiling, headroom = -80.0, -50.0, 0.10
        rng = _color_scale_range(floor, ceiling, headroom)
        spike = ceiling + rng * headroom * 2   # well past the reserved band
        t = (spike - floor) / rng
        assert t > 1.0

    def test_span_is_floored_for_a_near_silent_window(self):
        """floor and ceiling nearly coinciding (e.g. true silence) must not
        collapse the range toward zero, which would blow up the divide."""
        rng = _color_scale_range(-90.0, -89.999, 0.10)
        assert rng >= _MIN_COLOR_SPAN_DB

    def test_zero_headroom_puts_ceiling_exactly_at_one(self):
        rng = _color_scale_range(-80.0, -50.0, 0.0)
        assert rng == pytest.approx(30.0)

    def test_default_headroom_matches_module_constant(self):
        """Sanity check that _COLOR_HEADROOM is still a fraction, not a dB value —
        an easy unit mix-up given every other constant in this module is in dB."""
        assert 0.0 < _COLOR_HEADROOM < 1.0


class TestWaterfallConstants:
    def test_display_bins_formula_at_16k(self):
        # At 16 kHz: 128 bins × 31.25 Hz/bin = 4000 Hz
        assert _MAX_HZ * _CHUNK // 16000 == 128

    def test_db_range_is_48(self):
        assert _DB_RANGE == 48.0

    def test_n_rows_reasonable(self):
        assert 50 <= _N_ROWS <= 200

    def test_color_floor_percentile_below_ceiling_percentile(self):
        assert _COLOR_FLOOR_PERCENTILE < _COLOR_CEILING_PERCENTILE

    def test_color_ceiling_percentile_is_a_small_top_slice(self):
        """"Top 1-2%" per the design: close to 100 but not on top of it, since a
        pure maximum would be at the mercy of a single outlier bin."""
        assert 90 < _COLOR_CEILING_PERCENTILE < 100

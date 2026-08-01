"""
Tests for the pure display/formatting functions in device_setup.
Hardware-dependent functions (_probe, _enumerate_input_devices, select_device)
are not tested here - they require real PortAudio devices.
"""

import pytest
from buzz.device_setup import _amplitude_bar, _reason_bar, _BAR_WIDTH, _FILL, _EMPTY


class TestAmplitudeBar:

    def test_silence_is_all_empty(self):
        assert _amplitude_bar(0.0) == _EMPTY * _BAR_WIDTH

    def test_full_scale_is_all_filled(self):
        assert _amplitude_bar(32768.0) == _FILL * _BAR_WIDTH

    def test_length_always_bar_width(self):
        for amp in [0, 1, 100, 1000, 10000, 32768, 1_000_000]:
            assert len(_amplitude_bar(float(amp))) == _BAR_WIDTH

    def test_monotonically_non_decreasing(self):
        amps = [0, 1, 10, 100, 1000, 5000, 10000, 32768]
        fills = [_amplitude_bar(float(a)).count(_FILL) for a in amps]
        assert fills == sorted(fills)

    def test_only_fill_and_empty_chars(self):
        for amp in [0, 500, 5000, 32768]:
            assert all(c in (_FILL, _EMPTY) for c in _amplitude_bar(float(amp)))

    def test_fill_chars_always_precede_empty(self):
        for amp in [0, 500, 5000, 32768]:
            bar = _amplitude_bar(float(amp))
            n_fill = bar.count(_FILL)
            assert bar == _FILL * n_fill + _EMPTY * (_BAR_WIDTH - n_fill)

    @pytest.mark.parametrize('amp,min_bars,max_bars', [
        (100,    3,  8),   # weak signal: a few bars
        (3000,   9, 14),   # moderate signal: mid-range
        (30000, 14, 19),   # strong signal: nearly full
    ])
    def test_approximate_logarithmic_placement(self, amp, min_bars, max_bars):
        n = _amplitude_bar(float(amp)).count(_FILL)
        assert min_bars <= n <= max_bars


class TestReasonBar:

    def test_length_always_bar_width(self):
        for text in ['needs 44100 Hz', 'could not open', 'x' * 100, '']:
            assert len(_reason_bar(text)) == _BAR_WIDTH

    def test_short_text_is_centered(self):
        result = _reason_bar('hi')
        stripped = result.strip()
        assert stripped == 'hi'
        left_pad = len(result) - len(result.lstrip())
        right_pad = len(result) - len(result.rstrip())
        assert abs(left_pad - right_pad) <= 1

    def test_exact_width_text_unchanged(self):
        text = 'x' * _BAR_WIDTH
        assert _reason_bar(text) == text

    def test_overlong_text_truncated_to_bar_width(self):
        text = 'x' * (_BAR_WIDTH + 10)
        result = _reason_bar(text)
        assert len(result) == _BAR_WIDTH
        assert result == 'x' * _BAR_WIDTH

    @pytest.mark.parametrize('hz', [8000, 44100, 48000, 96000, 192000])
    def test_common_sample_rate_reasons_fit(self, hz):
        assert len(_reason_bar(f'needs {hz} Hz')) == _BAR_WIDTH

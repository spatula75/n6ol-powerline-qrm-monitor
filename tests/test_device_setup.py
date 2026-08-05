"""
Tests for the pure display/formatting functions in device_setup.
Hardware-dependent functions (_probe, enumerate_input_devices, select_device)
are not tested here - they require real PortAudio devices.
"""

import pytest
from buzz.constants import FULL_SCALE_COUNTS
from buzz.device_setup import (DeviceInfo, _amplitude_bar, _build_selection_prompt,
                               _reason_bar, _BAR_WIDTH, _FILL, _EMPTY, current_device)


class TestAmplitudeBar:

    def test_silence_is_all_empty(self):
        assert _amplitude_bar(0.0) == _EMPTY * _BAR_WIDTH

    def test_full_scale_is_all_filled(self):
        assert _amplitude_bar(FULL_SCALE_COUNTS) == _FILL * _BAR_WIDTH

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
        """Asserting the text survives, not that the bar is _BAR_WIDTH long.

        _reason_bar pads or truncates unconditionally, so a length check passes
        whatever it was handed and this test could not have failed.  It matters now:
        the bar narrowed from 19 columns to 15 when its width was derived from
        DB_PER_S_UNIT, and 'needs 192000 Hz' is exactly 15 - the longest reason the
        program produces has no headroom left, and truncation would silently print
        'needs 192000 H'.
        """
        reason = f'needs {hz} Hz'
        assert _reason_bar(reason).strip() == reason, (
            f'The rate a device wants no longer fits the {_BAR_WIDTH}-column level '
            f'bar, so it is truncated to something misleading. Either shorten the '
            f'wording or widen the bar - but the width is derived from '
            f'DB_PER_S_UNIT, so widening it changes what a segment means.')


class TestCurrentDevice:
    """Which entry the table marks as the one already configured.

    Matched on the device name rather than a stored PortAudio index, because an
    index is only true until Windows next reassigns audio hardware - the same
    reason the running program resolves by name at every startup.
    """

    def _devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(real_index=7, name='Line In, WASAPI', display_name='Line In',
                       selectable=True, amplitude=100.0, bar='', display_index=1),
            DeviceInfo(real_index=3, name='USB Audio, DirectSound', display_name='USB Audio',
                       selectable=True, amplitude=50.0, bar='', display_index=2),
        ]

    def test_it_finds_the_configured_device_by_name(self):
        found = current_device(self._devices(), 'USB Audio, DirectSound')
        assert found is not None and found.real_index == 3

    def test_the_index_it_reports_is_the_live_one(self):
        """The point of matching by name: whatever index the device sits at now is
        the answer, and no stale number from a config file can override it."""
        found = current_device(self._devices(), 'Line In, WASAPI')
        assert found.real_index == 7

    def test_a_device_that_is_no_longer_present_matches_nothing(self):
        """Unplugged between runs. Marking nothing as current is honest; marking
        whatever now sits at the old index would point at the wrong hardware."""
        assert current_device(self._devices(), 'Some Other Card, MME') is None

    def test_no_configured_name_matches_nothing(self):
        assert current_device(self._devices(), None) is None

    def test_an_empty_configured_name_matches_nothing(self):
        """A fresh config has never had a device chosen. The list deliberately holds a
        device whose own name came back empty - a driver reporting nothing is rare but
        real - so that the guard is what makes this pass. Without a nameless device
        present the assertion holds either way and tests nothing."""
        devices = self._devices()
        devices.append(DeviceInfo(real_index=9, name='', display_name='', selectable=True,
                                  amplitude=0.0, bar='', display_index=3))
        assert current_device(devices, '') is None

    def test_matching_is_exact_not_partial(self):
        """'Line In' is a prefix of a real entry. Accepting prefixes would mark the
        wrong device whenever two cards share the start of their names."""
        assert current_device(self._devices(), 'Line In') is None


class TestSelectionPrompt:
    def _selectable(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(real_index=7, name='Line In, WASAPI', display_name='Line In',
                       selectable=True, amplitude=100.0, bar='', display_index=1),
            DeviceInfo(real_index=3, name='USB Audio, DirectSound', display_name='USB Audio',
                       selectable=True, amplitude=50.0, bar='', display_index=2),
        ]

    def test_it_offers_every_selectable_number(self):
        prompt, valid = _build_selection_prompt(self._selectable(), None)
        assert valid == {1, 2}
        assert '1, 2' in prompt

    def test_it_offers_to_keep_the_current_device(self):
        devices = self._selectable()
        prompt, _ = _build_selection_prompt(devices, devices[1])
        assert 'Enter to keep current [2]' in prompt

    def test_it_does_not_offer_to_keep_what_is_not_there(self):
        """With no current device - first run, or one that has been unplugged -
        pressing Enter has nothing to mean, so the prompt must not suggest it."""
        prompt, _ = _build_selection_prompt(self._selectable(), None)
        assert 'keep current' not in prompt

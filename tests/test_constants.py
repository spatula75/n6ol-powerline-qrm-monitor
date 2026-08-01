"""Pins the values in buzz.constants, and the cross-module derivations that lean on
them, so a change to a shared constant shows up here rather than as a silent drift in
whichever consumer nobody happened to be looking at.

waterfall.py's own S-meter ladder is pinned in test_waterfall_math.py, next to the
rest of that module's tests; this file is for the constants themselves and for
consumers, like device_setup.py's bar width, that live elsewhere.
"""
from math import log10

import pytest

from buzz.constants import DB_PER_S_UNIT, FULL_SCALE_COUNTS, S9_DBM
from buzz.device_setup import _BAR_WIDTH


class TestConstants:
    def test_full_scale_is_sixteen_bit(self):
        assert FULL_SCALE_COUNTS == 32768.0

    def test_s9_is_the_iaru_reference(self):
        assert S9_DBM == -73.0

    def test_db_per_s_unit_is_six(self):
        """Not a design choice this program made -- it is what "S-unit" means."""
        assert DB_PER_S_UNIT == 6.0


class TestDeviceSetupBarWidth:
    """device_setup.py's level bar derives its width from the shared constants rather
    than a literal, which is the fix for the bug that motivated this file: the bar
    used to hard-code 19 and come out at 4.75 dB/segment while every S-meter in the
    program stayed at 6.
    """

    def test_each_segment_is_close_to_a_full_s_unit(self):
        """_BAR_WIDTH has to be a whole number of characters, so this can round to the
        nearest segment count but cannot land on DB_PER_S_UNIT exactly; the gap this
        allows is small enough that a reader would never notice it by eye."""
        full_range_db = 20 * log10(FULL_SCALE_COUNTS)
        assert full_range_db / _BAR_WIDTH == pytest.approx(DB_PER_S_UNIT, abs=0.1)

    def test_matches_the_fifteen_segments_this_produces_today(self):
        """Redundant with the ratio test above -- which is what actually guards the
        relationship -- but a bare 15 here is what a reader checks the derivation
        against without doing the arithmetic themselves."""
        assert _BAR_WIDTH == 15

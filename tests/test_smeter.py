"""Tests for buzz.setup.smeter: the S-meter reading string and the ASCII bar."""

import pytest

from buzz.constants import DB_PER_S_UNIT, S9_DBM
from buzz.setup.smeter import S1_DBM, S9P60_DBM, SCALE_ROW, TENS_ROW, dbm_to_s_string, s_meter_bar

_BAR_WIDTH = len(SCALE_ROW) - 2  # SCALE_ROW is '[' + one char per bar position + ']'


class TestDbmToSString:
    def test_s9_at_the_reference_point(self):
        assert dbm_to_s_string(S9_DBM) == 'S9'

    def test_one_s_unit_below_s9_is_s8(self):
        assert dbm_to_s_string(S9_DBM - DB_PER_S_UNIT) == 'S8'

    def test_eight_s_units_below_s9_is_s1(self):
        assert dbm_to_s_string(S9_DBM - 8 * DB_PER_S_UNIT) == 'S1'

    def test_never_reports_below_s1(self):
        """Well below S1, the arithmetic wants a negative S-number.  A ham's dial
        does not go below S1, so this floors there instead."""
        assert dbm_to_s_string(S9_DBM - 20 * DB_PER_S_UNIT) == 'S1'

    @pytest.mark.parametrize('over_s9,expected', [
        (0.0, 'S9'), (9.9, 'S9'),
        (10.0, 'S9+10'), (19.9, 'S9+10'),
        (20.0, 'S9+20'), (29.9, 'S9+20'),
        (30.0, 'S9+30'), (39.9, 'S9+30'),
        (40.0, 'S9+40'), (49.9, 'S9+40'),
        (50.0, 'S9+60'), (100.0, 'S9+60'),
    ])
    def test_above_s9_steps_every_ten_db(self, over_s9, expected):
        """IARU's convention changes above S9: 10 dB a step, not DB_PER_S_UNIT's 6."""
        assert dbm_to_s_string(S9_DBM + over_s9) == expected


class TestSMeterBar:
    def test_length_always_bar_width(self):
        for dbm in [S1_DBM - 100, S1_DBM, S9_DBM, S9P60_DBM, S9P60_DBM + 100]:
            assert len(s_meter_bar(dbm)) == _BAR_WIDTH

    def test_only_fill_and_empty_chars(self):
        for dbm in [S1_DBM - 100, S1_DBM, S9_DBM, S9P60_DBM]:
            assert all(c in '█░' for c in s_meter_bar(dbm))

    def test_fill_chars_always_precede_empty(self):
        for dbm in [S1_DBM - 100, S1_DBM, S9_DBM, S9P60_DBM]:
            bar = s_meter_bar(dbm)
            n_fill = bar.count('█')
            assert bar == '█' * n_fill + '░' * (_BAR_WIDTH - n_fill)

    def test_well_below_s1_is_all_empty(self):
        assert s_meter_bar(S1_DBM - 100) == '░' * _BAR_WIDTH

    def test_s1_lights_exactly_one_position(self):
        assert s_meter_bar(S1_DBM).count('█') == 1

    def test_s9_lights_exactly_nine_positions(self):
        """S1-S9 is the first _S1_TO_S9_CHARS positions, one per S-unit - see the
        module's own layout comment."""
        assert s_meter_bar(S9_DBM).count('█') == 9

    def test_s9p60_lights_every_position(self):
        assert s_meter_bar(S9P60_DBM) == '█' * _BAR_WIDTH

    def test_above_s9p60_still_fills_completely(self):
        """The fraction above S9+60 would exceed 1.0 unclamped, which would
        overfill the bar past its own width."""
        assert s_meter_bar(S9P60_DBM + 50) == '█' * _BAR_WIDTH

    def test_monotonically_non_decreasing(self):
        dbms = [S1_DBM - 50, S1_DBM, S9_DBM - 20, S9_DBM, S9_DBM + 20, S9P60_DBM]
        fills = [s_meter_bar(dbm).count('█') for dbm in dbms]
        assert fills == sorted(fills)


class TestScaleRows:
    def test_scale_row_and_tens_row_are_the_same_width(self):
        assert len(SCALE_ROW) == len(TENS_ROW)

    def test_scale_row_is_bracketed(self):
        assert SCALE_ROW.startswith('[') and SCALE_ROW.endswith(']')

    def test_scale_row_marks_the_low_and_high_ends(self):
        assert SCALE_ROW[1] == '1'   # S1, the leftmost position
        assert SCALE_ROW[-2] == '0'  # the +60 mark, the rightmost position

    def test_tens_row_marks_the_plus_sixty_column(self):
        """The tens digit of '+60' sits above the units '0' SCALE_ROW puts at the
        rightmost position - together the two rows spell out '60'."""
        assert TENS_ROW[-2] == '6'

"""Choosing a playback gain from an R128 measurement, without running ffmpeg.

The arithmetic and the parsing are the parts worth pinning down, and neither needs a
binary: real ebur128 output is pasted in as fixtures, so these run anywhere.
"""

from unittest.mock import patch

import pytest

from buzz.ffmpeg import FfmpegError
from buzz.loudness import (CEILING_DBTP, TARGET_LUFS, Loudness, auto_gain_db, measure,
                           resolve_gain)

# Real output, from one of the recordings this was developed against.
SUMMARY = """\
[Parsed_ebur128_0 @ 0000029748d02dc0] Summary:

  Integrated loudness:
    I:         -45.3 LUFS
    Threshold: -56.3 LUFS

  Loudness range:
    LRA:        13.7 LU
    Threshold: -70.0 LUFS
    LRA low:   -56.2 LUFS
    LRA high:  -42.5 LUFS

  True peak:
    Peak:      -21.5 dBFS
"""

SILENT = SUMMARY.replace('-45.3 LUFS', '-70.0 LUFS').replace('-21.5 dBFS', '0.2 dBFS')


def measured(**overrides) -> Loudness:
    values = {'integrated_lufs': -42.0, 'true_peak_dbtp': -30.0,
              'loudness_range_lu': 5.0, 'is_effectively_silent': False}
    return Loudness(**{**values, **overrides})


class TestParsingTheMeter:

    def test_it_reads_all_three_values(self):
        with patch('buzz.loudness.run', return_value=SUMMARY):
            loudness = measure('event.wav', 'ffmpeg')
        assert loudness.true_peak_dbtp == -21.5
        assert loudness.loudness_range_lu == 13.7

    def test_the_dual_mono_correction_is_applied(self):
        """R128 counts a mono file sent to both speakers as 3.01 LU louder, which is
        what a player does with the mono track in the rendered .mp4.  Without this
        every render comes out that much hot."""
        with patch('buzz.loudness.run', return_value=SUMMARY):
            assert measure('event.wav', 'ffmpeg').integrated_lufs == pytest.approx(-42.29)

    def test_it_does_not_confuse_lra_with_the_lines_beneath_it(self):
        """"LRA low" and "LRA high" sit directly under "LRA", and "Threshold" appears
        twice, so the patterns are anchored to the line and the unit."""
        with patch('buzz.loudness.run', return_value=SUMMARY):
            assert measure('event.wav', 'ffmpeg').loudness_range_lu == 13.7

    def test_a_summary_that_never_arrived_says_so(self):
        with patch('buzz.loudness.run', return_value='Unknown filter ebur128'):
            with pytest.raises(FfmpegError, match='printed no integrated value'):
                measure('event.wav', 'ffmpeg')

    def test_the_failure_points_at_the_patterns_to_update(self, ):
        """If ffmpeg ever changes the summary format, the person reading this failure
        needs to know it is a parsing problem and where the parsing lives."""
        with patch('buzz.loudness.run', return_value='Summary: nothing familiar'):
            with pytest.raises(FfmpegError, match='buzz.loudness'):
                measure('event.wav', 'ffmpeg')


class TestSilence:
    """A file with nothing above the absolute gate has no measurable loudness."""

    def test_the_meters_floor_is_recognised(self):
        """ebur128 reports exactly -70.0 to mean "nothing here"."""
        with patch('buzz.loudness.run', return_value=SILENT):
            assert measure('empty.wav', 'ffmpeg').is_effectively_silent

    def test_silence_is_judged_before_the_dual_mono_correction(self):
        """The bug this pins: adding 3.01 first lifts the -70.0 sentinel to -66.99, so
        the test never fires and a zero-length recording gets a real gain computed for
        it.  Observed on an actual 0-second file, which came out at -2.2 dB."""
        with patch('buzz.loudness.run', return_value=SILENT):
            loudness = measure('empty.wav', 'ffmpeg')
        assert loudness.integrated_lufs > -70.0     # corrected, above the sentinel
        assert loudness.is_effectively_silent       # and still known to be silent

    def test_no_gain_is_applied_to_silence(self):
        assert auto_gain_db(measured(is_effectively_silent=True)) == 0.0

    def test_silence_does_not_get_attenuated_by_the_ceiling(self):
        """A near-empty file measured +0.2 dBTP, which the ceiling alone would answer
        with -2.2 dB of attenuation - arithmetically right and meaningless."""
        assert auto_gain_db(measured(is_effectively_silent=True,
                                     true_peak_dbtp=0.2)) == 0.0


class TestAutoGain:
    """gain = min(target - integrated, ceiling - true peak)."""

    def test_the_loudness_target_binds_when_there_is_peak_headroom(self):
        assert auto_gain_db(measured(integrated_lufs=-42.0,
                                     true_peak_dbtp=-34.8)) == pytest.approx(19.0)

    def test_the_ceiling_binds_when_the_peaks_are_close_to_full_scale(self):
        """Impulsive content has a large crest factor, so a recording whose bursts sit
        high above its noise floor reaches the ceiling before the loudness target."""
        assert auto_gain_db(measured(integrated_lufs=-42.0,
                                     true_peak_dbtp=-10.0)) == pytest.approx(8.0)

    def test_it_never_pushes_true_peak_past_the_ceiling(self):
        for peak in (-40.0, -20.0, -10.0, -3.0, -1.0):
            gain = auto_gain_db(measured(true_peak_dbtp=peak))
            assert peak + gain <= CEILING_DBTP + 1e-9

    def test_it_never_overshoots_the_loudness_target(self):
        for integrated in (-60.0, -45.0, -30.0, -20.0):
            gain = auto_gain_db(measured(integrated_lufs=integrated))
            assert integrated + gain <= TARGET_LUFS + 1e-9

    def test_a_loud_recording_is_turned_down(self):
        """Nothing says the gain is positive: a recording already above the target
        should come down to it."""
        assert auto_gain_db(measured(integrated_lufs=-10.0, true_peak_dbtp=-30.0)) < 0

    def test_the_targets_are_the_broadcast_ones(self):
        assert (TARGET_LUFS, CEILING_DBTP) == (-23.0, -2.0)


class TestResolveGain:

    def test_it_reports_which_constraint_decided(self, caplog):
        """The one thing an operator checks when a render comes out louder or quieter
        than expected."""
        with patch('buzz.loudness.run', return_value=SUMMARY):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                resolve_gain('event.wav', 'ffmpeg')
        assert 'the loudness target' in caplog.text

    def test_silence_does_not_claim_a_constraint_it_did_not_use(self, caplog):
        """Reporting a gain of zero as "set by the true-peak ceiling" would be
        inventing a reason."""
        with patch('buzz.loudness.run', return_value=SILENT):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                resolve_gain('empty.wav', 'ffmpeg')
        assert 'set by' not in caplog.text
        assert 'no measurable loudness' in caplog.text

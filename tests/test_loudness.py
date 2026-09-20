"""Choosing a playback gain from an R128 measurement, without running ffmpeg.

The arithmetic and the parsing are the parts worth pinning down, and neither needs a
binary: real ebur128 output is pasted in as fixtures, so these run anywhere.
"""

import math

import pytest

from buzz import loudness as loudness_module
from buzz.ffmpeg import FfmpegError
from buzz.loudness import (CEILING_DBTP, TARGET_LUFS, Loudness, auto_gain_db, measure,
                           resolve_gain, run)
from tests.patching import patch_in

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

# Real output from a 0-second file.  Everything the meter has to say about a file
# with no samples in it: the gate value for the loudness, and -inf for the peak.
SILENT = """\
[Parsed_ebur128_0 @ 000001f5d3af0840] Summary:

  Integrated loudness:
    I:         -70.0 LUFS
    Threshold:   0.0 LUFS

  Loudness range:
    LRA:         0.0 LU
    Threshold:   0.0 LUFS
    LRA low:     0.0 LUFS
    LRA high:    0.0 LUFS

  True peak:
    Peak:       -inf dBFS
"""

# Real output from 15 s of a -75 dBFS sine, which is quiet and far from empty.  The
# meter says exactly what it says about the empty file above, apart from the peak.
UNDER_THE_GATE = SILENT.replace('-inf dBFS', '-89.1 dBFS')

# Real output from both passes over a 20 s event recorded at this station on
# 2026-09-19, which sits under the gate at a true peak of -65.8 dBFS.  The second is
# the same file lifted 63.8 dB, which is what puts that peak on the -2.0 dBTP ceiling.
# Subtracting the lift from the second reading recovers -77.29 LUFS for the original.
QUIET_EVENT = SILENT.replace('-inf dBFS', '-65.8 dBFS')

QUIET_EVENT_LIFTED = """\
[Parsed_ebur128_1 @ 00000234c8861e00] Summary:

  Integrated loudness:
    I:         -16.5 LUFS
    Threshold: -26.5 LUFS

  Loudness range:
    LRA:         0.4 LU
    Threshold: -36.5 LUFS
    LRA low:   -16.7 LUFS
    LRA high:  -16.3 LUFS

  True peak:
    Peak:       -2.0 dBFS
"""


def measured(**overrides) -> Loudness:
    values = {'integrated_lufs': -42.0, 'true_peak_dbtp': -30.0,
              'loudness_range_lu': 5.0, 'has_integrated_reading': True}
    return Loudness(**{**values, **overrides})


class TestParsingTheMeter:

    def test_it_reads_all_three_values(self):
        with patch_in(loudness_module, run, return_value=SUMMARY):
            loudness = measure('event.wav', 'ffmpeg')
        assert loudness.true_peak_dbtp == -21.5
        assert loudness.loudness_range_lu == 13.7

    def test_the_dual_mono_correction_is_applied(self):
        """R128 counts a mono file sent to both speakers as 3.01 LU louder, which is
        what a player does with the mono track in the rendered .mp4.  Without this
        every render comes out that much hot."""
        with patch_in(loudness_module, run, return_value=SUMMARY):
            assert measure('event.wav', 'ffmpeg').integrated_lufs == pytest.approx(-42.29)

    def test_it_does_not_confuse_lra_with_the_lines_beneath_it(self):
        """"LRA low" and "LRA high" sit directly under "LRA", and "Threshold" appears
        twice, so the patterns are anchored to the line and the unit."""
        with patch_in(loudness_module, run, return_value=SUMMARY):
            assert measure('event.wav', 'ffmpeg').loudness_range_lu == 13.7

    def test_a_summary_that_never_arrived_says_so(self):
        with patch_in(loudness_module, run, return_value='Unknown filter ebur128'):
            with pytest.raises(FfmpegError, match='printed no integrated value'):
                measure('event.wav', 'ffmpeg')

    def test_the_failure_points_at_the_patterns_to_update(self, ):
        """If ffmpeg ever changes the summary format, the person reading this failure
        needs to know it is a parsing problem and where the parsing lives."""
        with patch_in(loudness_module, run, return_value='Summary: nothing familiar'):
            with pytest.raises(FfmpegError, match='buzz.loudness'):
                measure('event.wav', 'ffmpeg')


class TestTheMetersGate:
    """A reading at the gate is the meter saying it has no answer, not a measurement."""

    def test_the_gate_value_is_not_taken_for_a_measurement(self):
        with patch_in(loudness_module, run, return_value=SILENT):
            assert not measure('empty.wav', 'ffmpeg').has_integrated_reading

    def test_a_real_measurement_is_taken_for_one(self):
        """The companion to the test above, so that neither can pass by always
        answering the same way."""
        with patch_in(loudness_module, run, return_value=SUMMARY):
            assert measure('event.wav', 'ffmpeg').has_integrated_reading

    def test_the_gate_is_judged_before_the_dual_mono_correction(self):
        """The bug this pins: adding 3.01 first lifts the gate value to -66.99, which
        sits above the gate, so the check never fires and a zero-length recording gets
        a real gain computed for it.  A real 0-second file did exactly that, and came
        out at -2.2 dB."""
        with patch_in(loudness_module, run, return_value=SILENT):
            loudness = measure('empty.wav', 'ffmpeg')
        assert loudness.integrated_lufs > -70.0         # corrected, above the gate
        assert not loudness.has_integrated_reading      # and still known to be no answer

    def test_a_quiet_file_and_an_empty_one_are_told_apart_by_the_peak(self):
        """Why the peak decides this and the loudness cannot.  Both fixtures here are
        real ffmpeg output and their integrated lines are identical: a -75 dBFS sine
        reads -70.0 LUFS exactly as 0 seconds of nothing does.  Any sentinel taken from
        the integrated figure therefore has to call one of the two wrong."""
        with patch_in(loudness_module, run, return_value=SILENT):
            empty = measure('empty.wav', 'ffmpeg')
        with patch_in(loudness_module, run, return_value=UNDER_THE_GATE):
            quiet = measure('quiet.wav', 'ffmpeg')
        assert empty.integrated_lufs == quiet.integrated_lufs
        assert empty.is_silent
        assert not quiet.is_silent

    def test_no_gain_is_applied_to_an_empty_file(self):
        assert auto_gain_db(measured(true_peak_dbtp=-math.inf)) == 0.0

    def test_a_file_with_a_real_peak_is_not_treated_as_empty(self):
        """A 0-second file whose peak the meter reported as +0.2 dBTP used to be
        answered with -2.2 dB of attenuation, which is arithmetically right and
        meaningless.  The peak decides emptiness now, so a file reporting a real peak
        has content whatever its loudness reading says."""
        assert auto_gain_db(measured(has_integrated_reading=False,
                                     true_peak_dbtp=0.2)) == pytest.approx(-2.2)

    def test_a_quiet_file_is_lifted_to_the_ceiling(self):
        """The case this exists for: a recording under the meter's gate is audible
        content with no loudness reading, so the ceiling sets the gain alone.  The
        peak came from ffmpeg on a 15 s -75 dBFS sine, which wants +87.1 dB."""
        gain = auto_gain_db(measured(has_integrated_reading=False,
                                     true_peak_dbtp=-89.1))
        assert gain == pytest.approx(87.1)
        assert -89.1 + gain == pytest.approx(CEILING_DBTP)

    def test_the_target_is_not_used_when_there_is_no_reading(self):
        """integrated_lufs holds the corrected gate value in this state, and taking a
        gain from it would answer +43.99 dB on a file the meter never measured."""
        gain = auto_gain_db(measured(has_integrated_reading=False,
                                     integrated_lufs=-66.99, true_peak_dbtp=-30.0))
        assert gain == pytest.approx(28.0)


class TestRecoveringAQuietFile:
    """A file under the gate is measured again, lifted, rather than guessed at.

    This is the ordinary path at this station rather than an unusual one: a real
    20 s event peaks at -65.8 dBFS, which the meter refuses to measure.
    """

    def test_the_loudness_is_recovered_from_the_lifted_pass(self):
        """Loudness is a ratio, so the lift comes straight back off the reading.  The
        second pass reads -13.49 LUFS corrected, the lift was 63.8 dB, and the file is
        therefore -77.29 LUFS.  ffmpeg confirms it at probe gains of 20, 30, 40, 50
        and 60 dB, every one of which recovers -77.29."""
        with patch_in(loudness_module, run,
                      side_effect=[QUIET_EVENT, QUIET_EVENT_LIFTED]):
            loudness = measure('quiet.wav', 'ffmpeg')
        assert loudness.has_integrated_reading
        assert loudness.integrated_lufs == pytest.approx(-77.29)

    def test_the_peak_comes_from_the_unlifted_pass(self):
        """The lifted pass reports -2.0 dBFS, which describes a file that does not
        exist.  Taking it would answer a gain of exactly zero for every quiet file."""
        with patch_in(loudness_module, run,
                      side_effect=[QUIET_EVENT, QUIET_EVENT_LIFTED]):
            loudness = measure('quiet.wav', 'ffmpeg')
        assert loudness.true_peak_dbtp == -65.8

    def test_the_loudness_range_comes_from_the_lifted_pass(self):
        """A range is the same either side of a constant gain, and the unlifted pass
        reports 0.0 LU because it measured nothing."""
        with patch_in(loudness_module, run,
                      side_effect=[QUIET_EVENT, QUIET_EVENT_LIFTED]):
            assert measure('quiet.wav', 'ffmpeg').loudness_range_lu == 0.4

    def test_the_target_binds_the_recovered_reading(self):
        """The point of recovering it.  Answering from the peak alone would give
        +63.8 dB, which leaves the file at -13.49 LUFS, 9.5 dB over the target."""
        with patch_in(loudness_module, run,
                      side_effect=[QUIET_EVENT, QUIET_EVENT_LIFTED]):
            loudness = measure('quiet.wav', 'ffmpeg')
        gain = auto_gain_db(loudness)
        assert gain == pytest.approx(54.29)
        assert loudness.integrated_lufs + gain == pytest.approx(TARGET_LUFS)

    def test_a_measurable_file_is_only_measured_once(self):
        """The second pass costs another ffmpeg run, so it has to stay off the path
        every ordinary recording takes."""
        with patch_in(loudness_module, run, return_value=SUMMARY) as meter:
            measure('event.wav', 'ffmpeg')
        assert meter.call_count == 1

    def test_an_empty_file_is_not_lifted(self):
        """Its peak is -inf, so the lift would be infinite and the second pass would
        measure nothing twice."""
        with patch_in(loudness_module, run, return_value=SILENT) as meter:
            assert measure('empty.wav', 'ffmpeg').is_silent
        assert meter.call_count == 1

    def test_a_file_with_nothing_to_measure_even_lifted_keeps_its_peak(self):
        """A few isolated samples in a long silence reach the ceiling and still give
        the meter no complete block.  That one falls back to the peak rather than
        looping or inventing a reading."""
        with patch_in(loudness_module, run,
                      side_effect=[UNDER_THE_GATE, UNDER_THE_GATE]):
            loudness = measure('sparse.wav', 'ffmpeg')
        assert not loudness.has_integrated_reading
        assert loudness.true_peak_dbtp == -89.1
        assert auto_gain_db(loudness) == pytest.approx(87.1)


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
        with patch_in(loudness_module, run, return_value=SUMMARY):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                resolve_gain('event.wav', 'ffmpeg')
        assert 'the loudness target' in caplog.text

    def test_an_empty_file_does_not_claim_a_constraint_it_did_not_use(self, caplog):
        """Reporting a gain of zero as "set by the true-peak ceiling" would be
        inventing a reason."""
        with patch_in(loudness_module, run, return_value=SILENT):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                gain = resolve_gain('empty.wav', 'ffmpeg')
        assert gain == 0.0
        assert 'set by' not in caplog.text
        assert 'holds no signal at all' in caplog.text

    def test_a_file_with_nothing_measurable_does_not_quote_the_gate(self, caplog):
        """The figure it would print is -67.0 LUFS, which is the meter's gate plus the
        dual-mono correction rather than anything measured."""
        with patch_in(loudness_module, run,
                      side_effect=[UNDER_THE_GATE, UNDER_THE_GATE]):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                gain = resolve_gain('sparse.wav', 'ffmpeg')
        assert gain == pytest.approx(87.1)
        assert '-67.0 LUFS' not in caplog.text
        assert 'set by' not in caplog.text
        assert 'found nothing to measure' in caplog.text

    def test_a_recovered_quiet_file_reports_like_any_other(self, caplog):
        """Once measure() has recovered a reading there is nothing special left to
        say, so the operator gets the usual line naming the constraint."""
        with patch_in(loudness_module, run,
                      side_effect=[QUIET_EVENT, QUIET_EVENT_LIFTED]):
            with caplog.at_level('INFO', logger='buzz.loudness'):
                gain = resolve_gain('quiet.wav', 'ffmpeg')
        assert gain == pytest.approx(54.29)
        assert 'the loudness target' in caplog.text
        assert '-77.3 LUFS' in caplog.text

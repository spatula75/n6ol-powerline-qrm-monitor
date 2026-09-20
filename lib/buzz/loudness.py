"""How loud a recording is, and how much gain would make it comfortable to listen to.

Recorded events sit around -45 LUFS: audible, but far below a normal listening
level. That is deliberate rather than accidental. The calibration process sets the
audio levels low on purpose, and impulsive content needs the headroom, since a 2.5-6 ms
burst runs 20-odd dB above the integrated level it contributes to. Measured on real
recordings: -45 LUFS integrated against a -21.5 dBFS true peak.

Good for measuring and awkward for showing somebody, so a render works out the gain for
itself rather than making the operator guess a figure and try again.

The measurement comes from ffmpeg's `ebur128` meter, which reports EBU R128 integrated
loudness together with true peak. The gain is then applied as a plain `volume=`, and
deliberately not by ffmpeg's `loudnorm` filter - see measure() for why the meter is
not loudnorm either, and auto_gain_db() for why the application is not.
"""

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from buzz.ffmpeg import FfmpegError, run

logger = logging.getLogger(__name__)

# EBU R128's broadcast reference. Chosen because it is a standard rather than a taste:
# a demo normalized to it sits at the same loudness as anything else made to R128.
TARGET_LUFS = -23.0
# Ceiling for true peak, in dBTP. 2 dB below full scale leaves room for the
# overshoot that inter-sample peaks and lossy encoding both produce, so nothing
# clips on playback even though the samples themselves never exceed it.
CEILING_DBTP = -2.0
# This is the meter's own floor, and what it prints when it has no answer.  BS.1770
# discards every block under an absolute gate of -70 LUFS, so the mean of the blocks
# that survive cannot come out below it, and a file with nothing above the gate
# reports exactly this figure or -inf.
#
# That means no threshold under -70 can ever be reached, and a threshold there cannot
# tell an empty file from a quiet one.  Measured through ffmpeg on 2026-09-19: sine
# tones at -60, -75, -85 and -95 dBFS all report -70.0 LUFS, and so does 15 s of
# digital silence.  The true peak separates those two, which is what Loudness.is_silent
# reads, and measure() recovers the loudness of the quiet one with a second pass.
_METER_GATE_LUFS = -70.0
# 10*log10(2): a mono signal sent to both speakers measures this much louder than the
# same signal as one channel of a stereo pair. R128 says to account for it.
_DUAL_MONO_LU = 3.01


@dataclass(frozen=True)
class Loudness:
    """What R128 says about a recording."""

    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float
    # This is false where the meter printed its own gate instead of a measurement, so
    # integrated_lufs holds no answer and nothing may be computed from it.  It comes
    # from the raw reading, before the dual-mono correction: adding 3.01 first would
    # lift the gate value clear of the comparison, and a zero-length recording would
    # get a real gain computed for it.  See _METER_GATE_LUFS.
    has_integrated_reading: bool = True

    @property
    def is_silent(self) -> bool:
        """Whether every sample in the file is zero.

        The integrated figure cannot answer this, because it reads -70.0 LUFS for an
        empty file and for a merely quiet one alike.  A true peak of -inf is the meter
        saying that no sample rose above zero, so the peak is the one reading that
        separates the two.  A capture that stopped before it wrote audio is the case
        this catches, and it came out of an actual 0-second file that was given -2.2 dB
        of gain.
        """
        return self.true_peak_dbtp == -math.inf


def measure(path: Path | str, ffmpeg: str) -> Loudness:
    """Measure `path` with ffmpeg's R128 meter.

    This uses `ebur128` rather than `loudnorm`, although loudnorm prints JSON and
    would be less work to read. ebur128 is ffmpeg's implementation of the ITU-R
    BS.1770 standard meter. loudnorm's analysis is tuned for its own normalization
    and does not agree with it on this material.  Measured across six real
    recordings:

        duration   LRA    ebur128   loudnorm     diff
          10.0 s  13.7 LU   -45.3     -38.28    +7.02
          16.6 s   5.1 LU   -45.0     -45.12    -0.12
          39.6 s   2.3 LU   -42.6     -42.68    -0.08
         122.2 s   3.3 LU   -40.9     -40.94    -0.04
         129.6 s   3.8 LU   -42.4     -42.56    -0.16

    They agree to within 0.16 LU everywhere except the file with a wide loudness
    range, and it is not a matter of duration: a shorter file agrees to 0.12.  It
    tracks LRA, and the reason is the relative gate.

    BS.1770 measures integrated loudness in two passes: discard blocks below an
    absolute gate at -70 LUFS, then discard blocks more than 10 LU below the mean of
    whatever survived.  When the content sits in a narrow band, every block falls the
    same side of that second gate in both implementations and they agree exactly.
    When the range is 13.7 LU, mostly quiet noise floor with brief loud bursts,
    which is precisely what a QRM recording is, a large fraction of blocks sit near
    the threshold.  A small difference in the ungated mean then moves the gate,
    excludes more quiet blocks, raises the mean, and moves the gate again.  Small
    implementation differences get amplified rather than averaged away.

    The direction confirms it: on that file loudnorm gated at -54.44 against ebur128's
    -56.3, keeping less of the quiet material and reporting a louder average, while
    both reported an identical true peak.  So the disagreement is entirely about
    gating and not about measuring loudness.

    This content is therefore unusually good at exposing the difference, which makes
    using the standard's own meter the right call rather than a fussy one.  A target
    expressed as a broadcast standard should be measured by the standard's meter.

    The whole file is measured, including the lead-in and the trailer.  What is being
    normalized is the viewer's experience of the video, not the event in isolation,
    and R128's own relative gate already discards the quiet passages, so the number
    reflects the part worth hearing without anyone having to trim to it.

    This measures a file under the meter's gate twice.  The first pass comes back with
    no integrated reading, so the second lifts the file to the true-peak ceiling and
    asks again, and the gain applied is subtracted from the answer.  Loudness is a
    ratio, so lifting a file by a constant moves its integrated figure by that same
    constant: measured on a real event at probe gains of 20, 30, 40, 50 and 60 dB, the
    recovered figure came out -77.29 LUFS at every one of them.  The station this was
    written for records well under the gate, at a true peak of -65.8 dBFS, so this is
    the ordinary path rather than an unusual one.

    This is fast enough not to matter, at over 200x real time, and the second pass
    only runs where the first found nothing.
    """
    first_pass = _run_meter(path, ffmpeg)
    if first_pass.has_integrated_reading or first_pass.is_silent:
        return first_pass
    return _measure_by_lifting(path, ffmpeg, first_pass)


def _run_meter(path: Path | str, ffmpeg: str, pre_gain_db: float = 0.0) -> Loudness:
    """One pass of ebur128, over the file lifted by `pre_gain_db` first.

    The lift goes in the filter chain rather than into a temporary file, so the second
    pass costs one more ffmpeg run and no disk.  `volume` works in floating point here
    whatever the input sample format, so a lift that takes the peak to the ceiling
    cannot clip on the way to the meter.  The same measurement taken from an
    amplified file on disk agreed with this one to the penny.
    """
    lift = f'volume={pre_gain_db}dB,' if pre_gain_db else ''
    output = run([
        ffmpeg, '-hide_banner', '-nostats', '-i', str(path),
        '-af', f'{lift}ebur128=peak=true', '-f', 'null', '-',
    ])
    return _parse(output, path)


def _measure_by_lifting(path: Path | str, ffmpeg: str, first_pass: Loudness) -> Loudness:
    """Measure a file the meter would not measure, by lifting it over the gate.

    The lift takes the true peak to the ceiling, which is the loudest this program
    would ever make the file.  A file that still reads nothing at that point holds too
    little for R128 to describe at all, so it comes back as it arrived, with no
    integrated reading, and auto_gain_db answers it from the peak.

    This keeps the peak from the first pass.  The lifted pass reports the lifted peak,
    which describes a file that does not exist.  The loudness range comes from the
    lifted pass instead.  A range is the same either side of a constant gain, and the
    first pass reports 0.0 LU for a file it could not measure.
    """
    lift = CEILING_DBTP - first_pass.true_peak_dbtp
    lifted = _run_meter(path, ffmpeg, lift)
    if not lifted.has_integrated_reading:
        return first_pass
    return Loudness(
        integrated_lufs=lifted.integrated_lufs - lift,
        true_peak_dbtp=first_pass.true_peak_dbtp,
        loudness_range_lu=lifted.loudness_range_lu,
        has_integrated_reading=True,
    )


# ebur128 prints a labeled summary; these pick the three values out of it. Anchored
# to the line start and to the unit, because "Threshold:" appears twice and "LRA low"
# and "LRA high" sit directly beneath "LRA".
_INTEGRATED = re.compile(r'^\s+I:\s+(-?[\d.]+|-inf)\s+LUFS\s*$', re.MULTILINE)
_TRUE_PEAK = re.compile(r'^\s+Peak:\s+(-?[\d.]+|-inf)\s+dBFS\s*$', re.MULTILINE)
_RANGE = re.compile(r'^\s+LRA:\s+(-?[\d.]+)\s+LU\s*$', re.MULTILINE)


def _parse(output: str, path: Path | str) -> Loudness:
    """Read the three numbers out of ebur128's summary.

    ebur128 has no JSON mode, so this is a text parse. It parses a labeled, stable
    summary rather than arbitrary output, and it fails loudly with everything
    ffmpeg said if the format ever moves.
    """
    found = {}
    for name, pattern in (('integrated', _INTEGRATED), ('true peak', _TRUE_PEAK),
                          ('loudness range', _RANGE)):
        match = pattern.search(output)
        if match is None:
            raise FfmpegError(
                f'Could not measure the loudness of {path}: ebur128 printed no '
                f'{name} value. Either the filter did not run to completion, or its '
                'summary format has changed between ffmpeg versions -- in which case '
                'the patterns in buzz.loudness need updating. What ffmpeg said '
                'follows:\n\n' + output[-1500:])
        found[name] = float(match.group(1))

    # R128 counts a mono file played through both speakers as 3.01 LU louder than the
    # same signal as one channel of a stereo pair, and that is what a player does with
    # the mono track in the rendered .mp4. loudnorm has a dual_mono option; the
    # standard meter does not, so the correction is applied here where it can be seen.
    integrated = found['integrated']
    return Loudness(
        integrated_lufs=integrated + _DUAL_MONO_LU,
        true_peak_dbtp=found['true peak'],
        loudness_range_lu=found['loudness range'],
        has_integrated_reading=integrated > _METER_GATE_LUFS,
    )


def auto_gain_db(loudness: Loudness, target_lufs: float = TARGET_LUFS,
                 ceiling_dbtp: float = CEILING_DBTP) -> float:
    """Gain that reaches the loudness target without letting true peak past the ceiling.

    Whichever constraint binds first wins:

        gain = min(target - integrated, ceiling - true_peak)

    Measured across three real events the target bound every time, giving +19.0, +19.3
    and +16.4 dB.  On one of them, though, the ceiling permitted only +19.5, so it came
    within 0.2 dB of binding.  The constraint is not theoretical: a recording whose
    bursts sit higher above its noise floor will reach the ceiling first and end up
    quieter than the target, which is the right way round.

    A single number, applied as `volume=`, so the waveform is multiplied by a constant
    and nothing else happens to it. loudnorm can apply its own normalization and is
    not asked to: it switches to a dynamic mode that varies gain over time, which was
    observed on these very recordings even at a loudness range well inside its
    threshold.  Time-varying gain on a 2.5-6 ms burst train compresses the envelope
    that carries the severity of the interference, which is the one thing a recording
    of it exists to preserve.  A limiter is refused for the same reason: attack and
    release exist to reshape transients.

    Two kinds of recording offer no target to aim at, and they take different answers.
    An empty file gets no gain, because multiplying zero by anything leaves zero and
    the operator needs to hear that the capture holds nothing.  A file that measure()
    could not read even lifted to the ceiling gets the ceiling constraint on its own,
    which is the most this program would apply to it in any case.

    A quiet file is not one of those two.  measure() recovers its loudness with a
    second pass, so the file arrives here with a reading like any other and the target
    binds it in the ordinary way.  Taking the ceiling for it instead overshot the
    target by 9.5 dB on a real event, because that content is noise-like and has almost
    no crest factor for the ceiling to leave room in.
    """
    if loudness.is_silent:
        logger.warning(
            'This recording holds no signal at all, so no automatic gain is applied.  '
            'Every sample is zero, which happens when a capture stopped before it '
            'wrote any audio.  Pass an explicit --playback-gain to amplify it anyway.')
        return 0.0
    to_ceiling = ceiling_dbtp - loudness.true_peak_dbtp
    if not loudness.has_integrated_reading:
        # This is reachable only with a finite peak, because a silent file has
        # already returned above, so the gain here cannot be infinite.
        logger.warning(
            'The meter found nothing to measure in this recording, even lifted to '
            'the %.1f dBTP ceiling, so the true-peak ceiling sets the gain alone.  '
            'It holds a few isolated samples and silence, or it is shorter than one '
            'measurement block.  Pass an explicit --playback-gain to choose another '
            'figure.',
            ceiling_dbtp)
        return to_ceiling
    to_target = target_lufs - loudness.integrated_lufs
    return min(to_target, to_ceiling)


def resolve_gain(path: Path | str, ffmpeg: str) -> float:
    """Measure `path` and report the gain chosen, with the reasoning in the log.

    This is logged at info because it is the one thing an operator will want to
    check when a render comes out louder or quieter than expected. It also matters
    because the probe takes a couple of seconds during which nothing else appears
    to be happening.
    """
    loudness = measure(path, ffmpeg)
    gain = auto_gain_db(loudness)
    if loudness.is_silent or not loudness.has_integrated_reading:
        # auto_gain_db has already explained both of these at warning level.  The line
        # below would misreport either one: it names a constraint that a gain of zero
        # did not use, and it quotes an integrated figure that is the meter's gate
        # rather than a measurement.
        return gain
    binding = ('the true-peak ceiling'
               if CEILING_DBTP - loudness.true_peak_dbtp < TARGET_LUFS - loudness.integrated_lufs
               else 'the loudness target')
    logger.info('Measured %.1f LUFS, true peak %.1f dBTP, range %.1f LU; '
                'applying %+.1f dB, set by %s.',
                loudness.integrated_lufs, loudness.true_peak_dbtp,
                loudness.loudness_range_lu, gain, binding)
    return gain

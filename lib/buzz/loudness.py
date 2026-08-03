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
# BS.1770's absolute gate. A file with nothing above it has no measurable loudness,
# and ebur128 reports exactly this figure (or -inf) to say so - it is the meter's way
# of reporting silence rather than a threshold this program chose on its own.
#
# Tested against the *raw* reading, before the dual-mono correction below. Adding 3.01
# first would lift the sentinel to -66.99 and the test would never fire, which is
# precisely the bug a zero-length recording exposed.
_SILENCE_FLOOR_LUFS = -70.0
# 10*log10(2): a mono signal sent to both speakers measures this much louder than the
# same signal as one channel of a stereo pair. R128 says to account for it.
_DUAL_MONO_LU = 3.01


@dataclass(frozen=True)
class Loudness:
    """What R128 says about a recording."""

    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float
    # Set from what the meter reported before any correction was applied; see
    # _SILENCE_FLOOR_LUFS.
    is_effectively_silent: bool = False


def measure(path: Path | str, ffmpeg: str) -> Loudness:
    """Measure `path` with ffmpeg's R128 meter.

    This uses `ebur128` rather than `loudnorm`, although loudnorm prints JSON and
    would be less work to read. ebur128 is ffmpeg's implementation of the ITU-R
    BS.1770 standard meter. loudnorm's analysis is tuned for its own normalization
    and does not agree with it on this material. Measured across six real
    recordings:

        duration   LRA    ebur128   loudnorm     diff
          10.0 s  13.7 LU   -45.3     -38.28    +7.02
          16.6 s   5.1 LU   -45.0     -45.12    -0.12
          39.6 s   2.3 LU   -42.6     -42.68    -0.08
         122.2 s   3.3 LU   -40.9     -40.94    -0.04
         129.6 s   3.8 LU   -42.4     -42.56    -0.16

    They agree to within 0.16 LU everywhere except the file with a wide loudness
    range, and it is not a matter of duration: a shorter file agrees to 0.12. It
    tracks LRA, and the reason is the relative gate.

    BS.1770 measures integrated loudness in two passes: discard blocks below an
    absolute gate at -70 LUFS, then discard blocks more than 10 LU below the mean of
    whatever survived. When the content sits in a narrow band, every block falls the
    same side of that second gate in both implementations and they agree exactly.
    When the range is 13.7 LU, mostly quiet noise floor with brief loud bursts,
    which is precisely what a QRM recording is, a large fraction of blocks sit near
    the threshold. A small difference in the ungated mean then moves the gate,
    excludes more quiet blocks, raises the mean, and moves the gate again. Small
    implementation differences get amplified rather than averaged away.

    The direction confirms it: on that file loudnorm gated at -54.44 against ebur128's
    -56.3, keeping less of the quiet material and reporting a louder average, while
    both reported an identical true peak. So the disagreement is entirely about
    gating and not about measuring loudness.

    This content is therefore unusually good at exposing the difference, which makes
    using the standard's own meter the right call rather than a fussy one. A target
    expressed as a broadcast standard should be measured by the standard's meter.

    The whole file is measured, including the lead-in and the trailer. What is being
    normalized is the viewer's experience of the video, not the event in isolation,
    and R128's own relative gate already discards the quiet passages, so the number
    reflects the part worth hearing without anyone having to trim to it.

    Fast enough not to matter: over 200x real time.
    """
    output = run([
        ffmpeg, '-hide_banner', '-nostats', '-i', str(path),
        '-af', 'ebur128=peak=true', '-f', 'null', '-',
    ])
    return _parse(output, path)


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
    return Loudness(
        integrated_lufs=found['integrated'] + _DUAL_MONO_LU,
        true_peak_dbtp=found['true peak'],
        loudness_range_lu=found['loudness range'],
        is_effectively_silent=found['integrated'] <= _SILENCE_FLOOR_LUFS,
    )


def auto_gain_db(loudness: Loudness, target_lufs: float = TARGET_LUFS,
                 ceiling_dbtp: float = CEILING_DBTP) -> float:
    """Gain that reaches the loudness target without letting true peak past the ceiling.

    Whichever constraint binds first wins:

        gain = min(target - integrated, ceiling - true_peak)

    Measured across three real events the target bound every time, giving +19.0, +19.3
    and +16.4 dB. On one of them, though, the ceiling permitted only +19.5, so it came
    within 0.2 dB of binding. The constraint is not theoretical: a recording whose
    bursts sit higher above its noise floor will reach the ceiling first and end up
    quieter than the target, which is the right way round.

    A single number, applied as `volume=`, so the waveform is multiplied by a constant
    and nothing else happens to it. loudnorm can apply its own normalization and is
    not asked to: it switches to a dynamic mode that varies gain over time, which was
    observed on these very recordings even at a loudness range well inside its
    threshold. Time-varying gain on a 2.5-6 ms burst train compresses the envelope
    that carries the severity of the interference, which is the one thing a recording
    of it exists to preserve. A limiter is refused for the same reason: attack and
    release exist to reshape transients.
    """
    if loudness.is_effectively_silent:
        logger.warning(
            'The meter found nothing above its %.0f LUFS absolute gate, so this '
            'recording has no measurable loudness and no automatic gain is applied. '
            'It is silent, or short enough to hold no complete measurement block; '
            'amplifying it would raise only its noise floor. Pass an explicit '
            '--playback-gain if you want one anyway.',
            _SILENCE_FLOOR_LUFS)
        return 0.0
    to_target = target_lufs - loudness.integrated_lufs
    to_ceiling = ceiling_dbtp - loudness.true_peak_dbtp
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
    if loudness.is_effectively_silent:
        # auto_gain_db has already explained itself at warning level; saying which
        # constraint "set" a gain of zero would be inventing a reason it did not use.
        return gain
    binding = ('the true-peak ceiling'
               if CEILING_DBTP - loudness.true_peak_dbtp < TARGET_LUFS - loudness.integrated_lufs
               else 'the loudness target')
    logger.info('Measured %.1f LUFS, true peak %.1f dBTP, range %.1f LU; '
                'applying %+.1f dB, set by %s.',
                loudness.integrated_lufs, loudness.true_peak_dbtp,
                loudness.loudness_range_lu, gain, binding)
    return gain

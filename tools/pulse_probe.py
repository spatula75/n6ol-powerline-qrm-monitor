"""
Diagnostic: measure the real shape of the powerline pulse train from live audio.

Run this when you want to know what the interference actually looks like at your
station, rather than what the analyzer assumes about it.  It reports three things:

  1. How fast the pulse phase is drifting, and therefore the utility line frequency.
  2. How many separate sources (utility phases) are arcing.
  3. How wide the pulse is, which is what PULSE_WIDTH_SAMPLES should be set to.

Usage:
    python tools/pulse_probe.py [--seconds 8] [--quiet]

-------------------------------------------------------------------------------
How it works
-------------------------------------------------------------------------------

The pulse train repeats every `samples_per_pulse` samples (133.33 at 16 kHz /
120 pps).  If you chop the audio into pieces that long and average them on top of
each other, the pulses line up and reinforce while the random noise averages away.
That average is called the *folded profile*, and it is the pulse shape.

The catch is that utility power is never exactly 60 Hz.  A small error means
the pulses arrive slightly early or late each time, so over a few seconds they walk
away from where the fold expects them.  Averaging then blurs the pulse instead of
sharpening it — at the drift rates seen in practice, an 8-second fold can smear a
pulse across 40+ samples.

So we do not guess the drift: we search for it.  Every candidate drift rate gets a
folded profile, and the correct rate is the one that produces the *sharpest* profile,
because that is the only rate at which the pulses land on top of each other.  This
measures the drift and removes it in a single step, and it needs no phase tracking
and no assumption about how many sources are present — every source hangs off the
same grid, so one drift correction de-smears all of them at once.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

import numpy as np
from buzz.config import BuzzConfig
from buzz.dsp import pulse_phase_period
from buzz.sampler import AudioSampler

# Drift rates to consider, in samples of phase slip per second of audio.  A grid
# error of 0.1 Hz slips about 27 samples/s, so +/-40 covers anything short of a
# serious grid disturbance while keeping the search cheap.
DRIFT_SEARCH_LIMIT = 40.0
DRIFT_SEARCH_STEP = 0.1

# A peak must rise this far up the profile's range to count as a separate source,
# and two peaks closer together than this many samples are treated as one.
SOURCE_PEAK_THRESHOLD = 0.30
SOURCE_PEAK_MIN_GAP = 5

# Widest PULSE_WIDTH_SAMPLES the width sweep will consider.
MAX_PULSE_WIDTH = 20


def rectify(audio: np.ndarray) -> np.ndarray:
    """Remove the DC offset and take the absolute value, as the analyzer does.

    The median is used rather than the mean because the pulse train itself would
    drag a mean away from the true offset — see ContinuousAnalyzer._capture.
    """
    samples = audio.astype(np.float64)
    return np.abs(samples - np.median(samples))


def fold_pulses(rectified: np.ndarray, sample_rate: int, pulse_rate: int,
                drift_rate: float, half_width: int) -> tuple[np.ndarray, int]:
    """Average one pulse period of audio around every pulse in the recording.

    drift_rate is the assumed phase slip in samples per second.  Pulse i arrives at
    time i/pulse_rate, by which point the train has slipped drift_rate * i /
    pulse_rate samples; adding that back keeps successive pulses aligned so they
    reinforce instead of blurring.

    Returns (profile, n_pulses).  The profile spans half_width samples either side
    of each pulse position, so index half_width is the nominal pulse centre.
    """
    samples_per_pulse = sample_rate / pulse_rate
    max_pulses = int((len(rectified) - 2 * half_width) / samples_per_pulse) - 1
    if max_pulses < 1:
        return np.zeros(2 * half_width + 1), 0

    pulse_index = np.arange(max_pulses)
    centres = (np.round(pulse_index * samples_per_pulse).astype(np.int64)
               + np.round(drift_rate * pulse_index / pulse_rate).astype(np.int64))
    centres = centres[(centres - half_width >= 0) & (centres + half_width < len(rectified))]
    if len(centres) == 0:
        return np.zeros(2 * half_width + 1), 0

    window = np.arange(-half_width, half_width + 1)
    return rectified[centres[:, None] + window[None, :]].mean(axis=0), len(centres)


def profile_sharpness(profile: np.ndarray) -> float:
    """How peaked a folded profile is: its maximum over its median.

    Used to score candidate drift rates.  A profile folded at the correct rate has
    its pulses stacked on one spot and scores high; a wrong rate spreads the same
    energy over many samples and scores low.  The median is the reference rather
    than the minimum because it is not swayed by a single quiet sample.
    """
    middle = np.median(profile)
    return float(profile.max() / middle) if middle > 0 else 0.0


def estimate_drift_rate(rectified: np.ndarray, sample_rate: int, pulse_rate: int,
                        half_width: int) -> float:
    """Find the phase slip rate that makes the folded profile sharpest.

    Returns samples of slip per second.  Positive means the pulses arrive later than
    a perfect `pulse_rate` train would predict, which means the grid is running slow.
    """
    candidates = np.arange(-DRIFT_SEARCH_LIMIT, DRIFT_SEARCH_LIMIT + DRIFT_SEARCH_STEP,
                           DRIFT_SEARCH_STEP)
    scores = [profile_sharpness(fold_pulses(rectified, sample_rate, pulse_rate, rate, half_width)[0])
              for rate in candidates]
    return float(candidates[int(np.argmax(scores))])


def drift_rate_to_grid_hz(drift_rate: float, sample_rate: int, pulse_rate: int) -> float:
    """Convert a phase slip rate into the utility line frequency it implies, in Hz.

    Note the direction: a *positive* slip means each pulse arrives later than the
    configured rate predicts, so the true pulse rate is *lower* and the grid is
    running slow.  Frequency moves opposite to the slip.

    Writing the true pulse rate as r, aligning the fold to the actual pulses requires
    drift_rate = sample_rate * (pulse_rate - r) / r, which rearranges to
    r = sample_rate * pulse_rate / (sample_rate + drift_rate).  There are two pulses
    per power cycle (one per half-cycle), so halving r gives the line frequency.
    """
    true_pulse_rate = sample_rate * pulse_rate / (sample_rate + drift_rate)
    return true_pulse_rate / 2


def find_source_peaks(profile: np.ndarray) -> list[int]:
    """Return the profile indices of distinct pulse sources, strongest first.

    Each arcing utility phase produces its own pulse train, offset from the others by
    120 degrees of the power cycle.  Within one pulse period that is a third of the
    period — about 44 samples at 16 kHz / 120 pps — so several peaks at roughly that
    spacing means several phases are arcing at once.
    """
    floor = profile.min()
    span = profile.max() - floor
    if span <= 0:
        return []
    threshold = floor + SOURCE_PEAK_THRESHOLD * span

    local_maxima = [i for i in range(1, len(profile) - 1)
                    if profile[i] > threshold
                    and profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]]

    # Collapse runs of adjacent samples that belong to the same peak.
    groups: list[list[int]] = []
    for i in local_maxima:
        if groups and i - groups[-1][-1] <= SOURCE_PEAK_MIN_GAP:
            groups[-1].append(i)
        else:
            groups.append([i])
    peaks = [max(group, key=lambda i: profile[i]) for group in groups]
    return sorted(peaks, key=lambda i: profile[i], reverse=True)


def matched_width_sweep(profile: np.ndarray) -> list[tuple[int, int, float]]:
    """Score each candidate PULSE_WIDTH_SAMPLES, best-scoring width last in ties.

    Returns [(width, start_offset_from_peak, figure_of_merit), ...].

    The score is sum(signal) / sqrt(width), the standard matched-filter result for a
    rectangular window in white noise.  Both halves matter: a wider window collects
    more of the pulse's energy (the sum grows) but also admits more noise (the
    sqrt(width) divisor).  The best width is where those balance, which is the pulse's
    effective duration.

    A tempting but wrong alternative is to score by the peak-to-trough contrast of the
    profile.  That always favours width 1, because the profile has already been
    averaged over hundreds of pulses, so it cannot show the noise reduction that is
    the entire reason to use a wider window on a single measurement.
    """
    baseline = np.percentile(profile, 10)      # robust stand-in for the inter-pulse level
    excess = np.maximum(profile - baseline, 0.0)
    peak = int(profile.argmax())

    results = []
    for width in range(1, MAX_PULSE_WIDTH + 1):
        sums = np.convolve(excess, np.ones(width), mode='valid')
        start = int(sums.argmax())
        results.append((width, start - peak, float(sums[start] / np.sqrt(width))))
    return results


def capture(config: BuzzConfig, seconds: float) -> np.ndarray:
    """Record `seconds` of contiguous audio from the configured input device."""
    sampler = AudioSampler(config)
    try:
        pipeline = sampler.pipeline
        wanted = int(seconds * config.audio.sample_rate)
        align = pulse_phase_period(config.audio.sample_rate, config.audio.pulse_rate)
        deadline = time.monotonic() + seconds + 20
        while time.monotonic() < deadline:
            if pipeline.wait_for_data(wanted + align, timeout=seconds + 15):
                break
        return pipeline.get_snapshot(wanted, align=align)
    finally:
        sampler.close()


def report(rectified: np.ndarray, config: BuzzConfig, show_profile: bool) -> None:
    """Print the full diagnostic for one captured recording."""
    sample_rate = config.audio.sample_rate
    pulse_rate = config.audio.pulse_rate
    samples_per_pulse = sample_rate / pulse_rate
    half_width = int(samples_per_pulse // 2)

    drift_rate = estimate_drift_rate(rectified, sample_rate, pulse_rate, half_width)
    profile, n_pulses = fold_pulses(rectified, sample_rate, pulse_rate, drift_rate, half_width)
    uncorrected, _ = fold_pulses(rectified, sample_rate, pulse_rate, 0.0, half_width)
    duration = len(rectified) / sample_rate

    print('=' * 72)
    print('UTILITY LINE DRIFT')
    print('=' * 72)
    print(f'  phase slip        : {drift_rate:+8.2f} samples/s')
    grid_hz = drift_rate_to_grid_hz(drift_rate, sample_rate, pulse_rate)
    print(f'  pulse-rate error  : {2 * grid_hz - pulse_rate:+8.4f} pps')
    print(f'  grid frequency    : {grid_hz:.4f} Hz'
          f'  ({grid_hz - pulse_rate / 2:+.4f} Hz from nominal)')
    print(f'  smear avoided     : {abs(drift_rate) * duration:.0f} samples over this {duration:.0f} s capture')
    print(f'  sharpness         : {profile_sharpness(profile):.3f} corrected'
          f'  vs {profile_sharpness(uncorrected):.3f} uncorrected')

    peaks = find_source_peaks(profile)
    strongest = int(profile.argmax())
    third_of_period = samples_per_pulse / 3
    print()
    print('=' * 72)
    print('ACTIVE UTILITY PHASES')
    print('=' * 72)
    print(f'  a separate utility phase would sit {third_of_period:.1f} samples away (120 deg of the period)')
    print(f'  distinct peaks    : {len(peaks)}')
    for peak in peaks:
        offset = peak - strongest
        thirds = offset / third_of_period
        tag = ''
        if abs(offset) > SOURCE_PEAK_MIN_GAP and abs(thirds - round(thirds)) < 0.2 and round(thirds):
            tag = f'   <- {round(thirds):+d}/3 period: separate utility phase'
        print(f'    {offset:+6d} samples  {20 * np.log10(profile[peak] / profile[strongest]):+6.2f} dB{tag}')
    if len(peaks) > 1:
        print('  NOTE: with several sources overlapping, the width sweep below measures a')
        print('        blend of them, not one pulse.  Re-run when one source dominates.')

    if show_profile:
        print()
        print('=' * 72)
        print(f'FOLDED PULSE PROFILE  ({n_pulses} pulses averaged)')
        print('=' * 72)
        floor = profile.min()
        span = max(profile.max() - floor, 1e-9)
        for i, value in enumerate(profile):
            bar = '#' * int(round(50 * (value - floor) / span))
            print(f'  {i - strongest:+5d} {value:8.1f} |{bar}' + (' <-PEAK' if i == strongest else ''))

    sweep = matched_width_sweep(profile)
    current = next(fom for width, _, fom in sweep if width == 3)
    best_width, best_offset, best_fom = max(sweep, key=lambda row: row[2])
    print()
    print('=' * 72)
    print('MATCHED WIDTH SWEEP   (score = sum(signal) / sqrt(width))')
    print('=' * 72)
    print('    W   start_off     score    dB vs W=3')
    print('  ' + '-' * 44)
    for width, offset, fom in sweep:
        note = '  <- current' if width == 3 else ('  <- best' if width == best_width else '')
        print(f'  {width:3d}   {offset:+8d}  {fom:9.1f}   {20 * np.log10(fom / current):+6.2f}{note}')
    print()
    print(f'  best PULSE_WIDTH_SAMPLES = {best_width} '
          f'({20 * np.log10(best_fom / current):+.2f} dB vs the current 3, '
          f'window starts {best_offset:+d} from the peak)')
    print(f'  implied pulse duration ~{best_width / sample_rate * 1e6:.0f} us')
    if best_width >= MAX_PULSE_WIDTH:
        print('  WARNING: the score never turned over, so the pulse is not resolved here.')
        print('           That happens when sources overlap enough to fill the period.')
        print('           Treat this as "no answer", not as a recommendation.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('---')[0].strip())
    parser.add_argument('--seconds', type=float, default=8.0,
                        help='length of audio to capture (default 8; the ring buffer holds ~9.6)')
    parser.add_argument('--quiet', action='store_true',
                        help='skip the sample-by-sample profile listing')
    args = parser.parse_args()

    config = BuzzConfig.from_toml()
    print(f'device : {config.audio.input_device_name}')
    print(f'rates  : {config.audio.sample_rate} Hz, {config.audio.pulse_rate} pps '
          f'({config.audio.sample_rate / config.audio.pulse_rate:.3f} samples per pulse)')
    print(f'capturing {args.seconds:.1f} s ...\n')

    audio = capture(config, args.seconds)
    rectified = rectify(audio)
    print(f'captured {len(audio)} samples   '
          f'rectified mean {rectified.mean():.1f} / median {np.median(rectified):.1f} '
          f'/ max {rectified.max():.0f}\n')
    report(rectified, config, show_profile=not args.quiet)


if __name__ == '__main__':
    main()

"""
Pulse-train DSP core shared by the sampler, analyzer, and calibration tools.

Everything here is pure signal processing with no I/O or threading: building
the pps scan kernel, scoring how well a pulse train fits at each phase offset
(FFT convolution), averaging amplitude at the pulse positions (Numba JIT), and
converting amplitudes to dBFS.  analyze_window() ties them together: given one
window of rectified audio it finds the best and worst pulse phases and measures
the amplitude at each — the single analysis step both the continuous analyzer
and the golden-file tests build on.
"""

from dataclasses import dataclass
from math import gcd, log10

import numpy as np
from numba import njit
from numpy import uint32, zeros
from scipy.signal import fftconvolve

# dBFS reference for 16-bit audio: 0 dBFS = full-scale amplitude of 32768 (2^15).
# The factor of 20 (not 10) is because dBFS is defined in terms of amplitude, not power.
DB_REFERENCE = 20 * log10(32768.0)

# Amplitudes of zero (receiver off, muted input, driver returning silence) have no
# logarithm; -128 dBFS is well below the ~-90 dBFS minimum for a 1-LSB 16-bit signal,
# so it is unambiguously a sentinel and never confused with a real reading.
SILENCE_DBFS = -128.0

# Number of consecutive samples that must be elevated for a position to count as a pulse.
# Requiring a sustained signal across several adjacent samples rejects very short transient
# pops (static, clicks, relay bounce) that are unlikely to be part of a repetitive
# powerline-arc pattern.  The value is somewhat arbitrary — 3 samples at 16 kHz is only
# ~188 µs — but it meaningfully reduces false triggers from sub-millisecond impulse noise.
PULSE_WIDTH_SAMPLES = 3


@dataclass(frozen=True)
class WindowAnalysis:
    """Result of analyzing one window of rectified audio for a pulse train.

    Amplitudes are raw mean-absolute values (not dB) so callers can apply their
    own reference/offset; phases are sample offsets within one pulse period.
    """
    signal_amplitude: float
    noise_amplitude: float
    peak_phase: int
    noise_phase: int


def pulse_phase_period(sample_rate: int, pulse_rate: int) -> int:
    """Number of samples after which the pulse grid repeats exactly.

    Pulse positions are round(i * sample_rate / pulse_rate), and samples_per_pulse is
    not generally a whole number of samples (133.333 at 16 kHz / 120 pps).  The grid
    only realigns with the sample clock after sample_rate // gcd(sample_rate,
    pulse_rate) samples — 400 samples, or exactly 3 pulse periods, at the defaults.

    This, not samples_per_pulse, is the correct modulus for a phase.  Two start offsets
    one full period apart sample identical positions; offsets one samples_per_pulse
    apart differ by a sample on two thirds of the pulses.  Being an integer, it also
    keeps the phase arithmetic exact: reducing a float argmax by a fractional modulus
    can land a hair under the true value (4.999999999997698) and truncate to the
    wrong phase.
    """
    return sample_rate // gcd(sample_rate, pulse_rate)


def amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a mean-absolute amplitude to dBFS; SILENCE_DBFS for zero/negative input."""
    return 20 * log10(amplitude) - DB_REFERENCE if amplitude > 0 else SILENCE_DBFS


def amplitude_to_dbm(amplitude: float, offset_db: float) -> float:
    """Convert a mean-absolute amplitude to approximate dBm at the receiver input.

    offset_db is the station's audio_rf_conversion_db calibration value.  Silence
    returns SILENCE_DBFS with no offset applied, so the sentinel reads identically
    regardless of calibration.
    """
    return amplitude_to_dbfs(amplitude) + offset_db if amplitude > 0 else SILENCE_DBFS


@njit(boundscheck=True)  # boundscheck raises IndexError instead of segfaulting on OOB access
def average_pulse_amplitude(mono_amplitude_array: np.ndarray, samples_per_pulse: float,
                            n_pulses: int, start_index: int) -> float:
    """Average the amplitude at each pulse position across the analysis window.

    samples_per_pulse is the spacing between consecutive pulses.  It is a float and
    it is passed in rather than derived from sample_rate / pulse_rate, because the
    true spacing depends on the actual grid frequency: if the grid is off by even
    0.05 pps, sampling at the nominal spacing walks 6.7 samples away from the real
    pulses over a one-second window, which costs about 6 dB.  Callers that have
    measured the drift should pass the corrected spacing — see
    ContinuousAnalyzer._effective_samples_per_pulse().

    n_pulses is the number of pulse positions to average.  Samples PULSE_WIDTH_SAMPLES
    adjacent values per pulse position, then divides by the total sample count.
    Equivalent to correlating with the pulse kernel but without multiplying through
    the ~97% zeros — build_pulse_kernel() places its ones at the same round()ed
    positions, so the two agree sample for sample.

    Internally clamps n_pulses so the last access (start_index + pos + width - 1)
    cannot exceed the array bounds regardless of what the caller passes.  Returns 0.0
    when there is insufficient room for even one pulse group.

    The mean is returned as a float rather than floor-divided to an integer: near the
    noise floor a mean-absolute amplitude of 1 LSB is only ~-90 dBFS, where one integer
    step is 6 dB.  Quantising here would coarsen exactly the weak-signal readings the
    monitor exists to make.
    """
    # Maximum pulses that fit: last group ends at start_index + pos + PULSE_WIDTH_SAMPLES - 1.
    # pos = round((n-1) * spp) <= (n-1)*spp + 0.5, so last index <= start_index + n*spp - spp + 0.5 + width - 1.
    # Solving for n: n <= (len - start_index - width) / spp  (the 0.5 slack is absorbed by spp > 100).
    max_pulses = int((len(mono_amplitude_array) - start_index - PULSE_WIDTH_SAMPLES) // samples_per_pulse)
    n = min(n_pulses, max_pulses)
    if n < 1:
        return 0.0
    total = 0.0
    for i in range(n):
        pos = round(i * samples_per_pulse)
        for j in range(PULSE_WIDTH_SAMPLES):
            total += mono_amplitude_array[start_index + pos + j]
    return total / (PULSE_WIDTH_SAMPLES * n)


def build_pulse_kernel(sample_rate: int, pulse_rate: int,
                       n_pulses: int | None = None) -> np.ndarray:
    """Build the pps scan kernel: ones at each pulse position, zeros between.

    n_pulses controls kernel length; defaults to pulse_rate // 2 (half a second).
    Length is round((n_pulses-1)*samples_per_pulse)+PULSE_WIDTH_SAMPLES, placing the
    last pulse group flush against the end so there are no trailing zeros.

    Pulse positions use round(), matching average_pulse_amplitude() exactly.  This
    matters: samples_per_pulse is 133.333 at 16 kHz / 120 pps, and truncating instead
    would put a third of the positions one sample away from where the amplitude
    averager reads them.  At PULSE_WIDTH_SAMPLES = 3 a one-sample offset costs a third
    of that pulse's energy, so the FFT would be optimising a phase that the subsequent
    measurement never actually samples.

    The kernel is NOT a palindrome (round() is not translation-consistent the way
    truncation happens to be), so calculate_pps_fit_array reverses it to turn
    convolution into correlation rather than relying on symmetry.
    """
    samples_per_pulse = sample_rate / pulse_rate
    scan_pulses = n_pulses if n_pulses is not None else pulse_rate // 2
    last_pos = round((scan_pulses - 1) * samples_per_pulse)
    coefficients = zeros(last_pos + PULSE_WIDTH_SAMPLES, dtype=uint32)
    for i in range(scan_pulses):
        pos = round(i * samples_per_pulse)
        coefficients[pos:pos + PULSE_WIDTH_SAMPLES] = 1
    return coefficients


def calculate_pps_fit_array(mono_amplitude_array: np.ndarray, kernel: np.ndarray,
                            scan_pulses: int) -> np.ndarray:
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  O((N+M) log(N+M)) vs O(N*M) for direct correlation.

    fftconvolve computes convolution, which reads the kernel backwards; reversing the
    kernel first therefore yields true cross-correlation, so fit[n] is exactly the sum
    average_pulse_amplitude() would compute at start_index = n.  (The kernel used to be
    built as a palindrome so the reversal could be skipped, but that only held for rate
    pairs where the truncated pulse positions happened to be symmetric.)

    mode='valid' returns only positions where the kernel fits entirely within the
    signal, avoiding edge artefacts from the convolution.
    """
    raw = fftconvolve(mono_amplitude_array.astype(np.float64),
                      kernel[::-1].astype(np.float64), mode='valid')
    # Each raw score is a sum over the kernel's PULSE_WIDTH_SAMPLES × scan_pulses
    # ones; dividing by that count converts it to a mean amplitude per sampled
    # position — the same scale average_pulse_amplitude reports.
    return raw / (PULSE_WIDTH_SAMPLES * scan_pulses)


def analyze_window(mono_amplitude_array: np.ndarray, sample_rate: int, pulse_rate: int,
                   kernel: np.ndarray, scan_pulses: int) -> WindowAnalysis | None:
    """Find the pulse train in one window of rectified audio and measure it.

    Scores every phase offset with the FFT fit, takes the best-fit phase as the
    signal and the worst-fit phase as the noise reference, then averages the
    amplitude at each across the window.  The minimum-correlation phase is used
    as the noise reference deliberately: sampling at the positions LEAST
    correlated with the pulse train measures everything except the powerline
    interference — atmospheric noise, man-made QRM, receiver thermal noise — so
    the caller can judge whether powerline noise is actually the dominant problem.

    Returns None when the window is too short to fit even one pulse train after
    the phase offsets are applied.
    """
    fit = calculate_pps_fit_array(mono_amplitude_array, kernel, scan_pulses)
    if fit.size == 0:   # window shorter than the kernel — no valid positions
        return None
    samples_per_pulse = sample_rate / pulse_rate
    # Reduce modulo the exact repeat period, in integer arithmetic — see
    # pulse_phase_period().  A perfectly periodic signal makes every position one
    # period apart an equal-best fit, so which one argmax returns is decided by FFT
    # round-off; only an exact modulus maps them all back to the same phase.
    phase_period = pulse_phase_period(sample_rate, pulse_rate)
    peak_phase = int(fit.argmax()) % phase_period
    noise_phase = int(fit.argmin()) % phase_period

    # Start the analysis window after whichever phase offset is larger, so both
    # the peak and noise trains fit entirely within the recording.
    analysis_start = max(peak_phase, noise_phase)
    n_pulses = int((len(mono_amplitude_array) - analysis_start) // samples_per_pulse)
    if n_pulses < 1:
        return None

    # Nominal spacing: this is the acquisition search, run before any drift estimate
    # exists.  Once locked, the analyzer re-measures with the corrected spacing.
    signal_amplitude = float(average_pulse_amplitude(
        mono_amplitude_array, samples_per_pulse, n_pulses, peak_phase))
    noise_amplitude = float(average_pulse_amplitude(
        mono_amplitude_array, samples_per_pulse, n_pulses, noise_phase))
    return WindowAnalysis(signal_amplitude, noise_amplitude, peak_phase, noise_phase)

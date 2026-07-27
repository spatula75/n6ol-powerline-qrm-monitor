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
from math import log10

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


def amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a mean-absolute amplitude to dBFS; SILENCE_DBFS for zero/negative input."""
    return 20 * log10(amplitude) - DB_REFERENCE if amplitude > 0 else SILENCE_DBFS


@njit(boundscheck=True)  # boundscheck raises IndexError instead of segfaulting on OOB access
def average_pulse_amplitude(mono_amplitude_array: np.ndarray, sample_rate: int,
                            pulse_rate: int, analysis_size: int, start_index: int) -> int:
    """Average the amplitude at each pulse position across the analysis window.

    Samples PULSE_WIDTH_SAMPLES adjacent values per pulse position, then divides by the
    total sample count.  Equivalent to correlating with the pulse kernel but without
    multiplying through the ~97% zeros.

    Internally clamps analysis_size so the last access (start_index + pos + width - 1)
    cannot exceed the array bounds regardless of what the caller passes.  Returns 0 when
    there is insufficient room for even one pulse group.
    """
    samples_per_pulse = sample_rate / pulse_rate
    # Maximum pulses that fit: last group ends at start_index + pos + PULSE_WIDTH_SAMPLES - 1.
    # pos = round((n-1) * spp) <= (n-1)*spp + 0.5, so last index <= start_index + n*spp - spp + 0.5 + width - 1.
    # Solving for n: n <= (len - start_index - width) / spp  (the 0.5 slack is absorbed by spp > 100).
    max_size = int((len(mono_amplitude_array) - start_index - PULSE_WIDTH_SAMPLES) // samples_per_pulse)
    n = min(analysis_size, max_size)
    if n < 1:
        return 0
    total = 0
    for i in range(n):
        pos = round(i * samples_per_pulse)
        for j in range(PULSE_WIDTH_SAMPLES):
            total += mono_amplitude_array[start_index + pos + j]
    return total // (PULSE_WIDTH_SAMPLES * n)


def build_pulse_kernel(sample_rate: int, pulse_rate: int,
                       n_pulses: int | None = None) -> np.ndarray:
    """Build the symmetric pps scan kernel.

    n_pulses controls kernel length; defaults to pulse_rate // 2 (half a second).
    Length is int((n_pulses-1)*samples_per_pulse)+3, placing the last pulse group
    flush against the end so the kernel is an exact palindrome.  A palindrome kernel
    means fftconvolve (convolution) == cross-correlation.
    """
    samples_per_pulse = sample_rate / pulse_rate
    scan_pulses = n_pulses if n_pulses is not None else pulse_rate // 2
    last_pos = int((scan_pulses - 1) * samples_per_pulse)
    coefficients = zeros(last_pos + PULSE_WIDTH_SAMPLES, dtype=uint32)
    for i in range(scan_pulses):
        pos = int(i * samples_per_pulse)
        coefficients[pos:pos + PULSE_WIDTH_SAMPLES] = 1
    return coefficients


def calculate_pps_fit_array(mono_amplitude_array: np.ndarray, kernel: np.ndarray,
                            scan_pulses: int) -> np.ndarray:
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  Uses fftconvolve with a symmetric kernel so convolution ==
    correlation; O((N+M) log(N+M)) vs O(N*M) for direct correlation.

    mode='valid' returns only positions where the kernel fits entirely within the
    signal, avoiding edge artefacts from the convolution.
    """
    raw = fftconvolve(mono_amplitude_array.astype(np.float64), kernel, mode='valid')
    return np.rint(raw).astype(np.int64) // (PULSE_WIDTH_SAMPLES * scan_pulses)


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
    peak_phase = int(fit.argmax() % samples_per_pulse)
    noise_phase = int(fit.argmin() % samples_per_pulse)

    # Start the analysis window after whichever phase offset is larger, so both
    # the peak and noise trains fit entirely within the recording.
    analysis_start = max(peak_phase, noise_phase)
    analysis_size = int((len(mono_amplitude_array) - analysis_start) // samples_per_pulse)
    if analysis_size < 1:
        return None

    signal_amplitude = float(average_pulse_amplitude(
        mono_amplitude_array, sample_rate, pulse_rate, analysis_size, peak_phase))
    noise_amplitude = float(average_pulse_amplitude(
        mono_amplitude_array, sample_rate, pulse_rate, analysis_size, noise_phase))
    return WindowAnalysis(signal_amplitude, noise_amplitude, peak_phase, noise_phase)

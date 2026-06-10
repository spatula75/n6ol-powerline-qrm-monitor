"""
Audio sampling and pulse-train analysis for powerline interference detection.

Records a short audio clip from the configured input device, then uses FFT-based
convolution to find the phase and amplitude of the periodic pulse train produced
by arcing powerline hardware.  Returns signal level, noise floor, and SNR in dBFS
so the caller can convert to dBm using the station's calibration offset.
"""

import logging
from math import log10

import numpy as np
import sounddevice as sd
from numba import njit
from numpy import uint32, zeros
from scipy.signal import fftconvolve

from buzz.config import BuzzConfig

logger = logging.getLogger(__name__)

# dBFS reference for 16-bit audio: 0 dBFS = full-scale amplitude of 32768 (2^15).
# The factor of 20 (not 10) is because dBFS is defined in terms of amplitude, not power.
_DB_REFERENCE = 20 * log10(32768.0)

# Number of consecutive samples that must be elevated for a position to count as a pulse.
# Requiring a sustained signal across several adjacent samples rejects very short transient
# pops (static, clicks, relay bounce) that are unlikely to be part of a repetitive
# powerline-arc pattern.  The value is somewhat arbitrary — 3 samples at 16 kHz is only
# ~188 µs — but it meaningfully reduces false triggers from sub-millisecond impulse noise.
_PULSE_WIDTH_SAMPLES = 3


class AudioSampler:
    def __init__(self, config: BuzzConfig) -> None:
        """Initialise the sampler and resolve the PortAudio device to record from.

        If config.audio.device_index is set it takes precedence over
        input_device_name; a warning is printed if the name no longer matches
        (e.g. after a USB reconnect).  Also pre-builds the pulse-train kernel so
        it isn't reconstructed on every sample.
        """
        self._config = config
        audio = config.audio
        if audio.device_index is not None:
            device = sd.query_devices(audio.device_index)
            hostapis = sd.query_hostapis()
            current_name = f"{device['name']}, {hostapis[device['hostapi']]['name']}"
            if current_name != audio.input_device_name:
                logger.warning(
                    'Audio device at index %d is now "%s" but was configured as "%s"; '
                    'the device may have been reconnected or renamed. '
                    'Continuing with index %d — re-run configure.py if recording fails.',
                    audio.device_index, current_name, audio.input_device_name, audio.device_index,
                )
            self._device_index = audio.device_index
        else:
            device = sd.query_devices(audio.input_device_name, 'input')
            self._device_index = device['index']
        self._kernel = _build_pulse_kernel(audio.sample_rate, audio.pulse_rate)
        self._scan_pulses = audio.pulse_rate // 2

    def take_sample(self) -> tuple[float, float, float]:
        """Record one audio sample and return (snr_db, signal_dbfs, noise_dbfs).

        Blocks for config.audio.duration seconds while recording.  Signal and noise
        levels are mean-absolute-amplitude dBFS (not RMS dBFS) — see inline comments
        for the reasoning behind that choice.
        """
        audio = self._config.audio
        recording = sd.rec(
            int(audio.duration * audio.sample_rate),
            samplerate=audio.sample_rate,
            channels=1,
            blocking=True,
            dtype='int16',
            device=self._device_index,
        )
        # Powerline arcing is highly impulsive: the arc fires only near the voltage peaks of
        # the 60 Hz AC cycle, so its duty cycle is typically a few percent or less.  Using
        # broadband RMS would spread that burst energy across the entire cycle and understate
        # the peak interference amplitude by a factor of √(duty_cycle) — easily 10–20 dB for
        # a 1–5 % duty cycle arc.  Mean absolute amplitude (|sample| averaged over many cycles
        # at the known pulse-phase positions) captures the arc's characteristic impulse level
        # directly and gives a more honest picture of how severe the interference actually is
        # during the moments it occurs.  Both signal and noise are measured the same way, so
        # the SNR and absolute levels are self-consistent; the station calibration offset must
        # be derived with this same MAV-based code.
        mono_amplitude_array = np.abs(recording[:, 0].astype(np.int32))

        output = _calculate_pps_fit_array(mono_amplitude_array, self._kernel, self._scan_pulses)

        peak_offset_index = output.argmax()
        # The minimum-correlation phase is used as the noise reference deliberately: by sampling
        # at the phase positions LEAST correlated with the 120 pps pulse train we measure
        # everything except the powerline interference — atmospheric noise, man-made QRM, solar
        # noise, receiver thermal noise, etc.  This gives a stable, interference-free baseline
        # so the graph can answer the practical question: is the powerline noise actually the
        # dominant problem, or are other HF noise sources equally loud?  There is little value
        # in chasing a powerline arc if atmospheric or solar noise is just as strong.
        noise_offset_index = output.argmin()

        # samples_per_pulse: the pulse period expressed in samples (non-integer at 120 pps / 16 kHz)
        samples_per_pulse = audio.sample_rate / audio.pulse_rate
        peak_phase = int(peak_offset_index % samples_per_pulse)
        noise_phase = int(noise_offset_index % samples_per_pulse)

        # Start the analysis window after whichever phase offset is larger, so both
        # the peak and noise trains fit entirely within the recording
        analysis_start = max(peak_phase, noise_phase)
        analysis_size = int((len(mono_amplitude_array) - analysis_start) // samples_per_pulse)

        avg_peak = _average_pulse_amplitude(
            mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, peak_phase)
        avg_noise = _average_pulse_amplitude(
            mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, noise_phase)

        # log10(0) is undefined; guard against an all-zero recording (e.g. receiver
        # powered off, muted input, or driver returning silence).  -128 dBFS is well
        # below the ~-90 dBFS minimum for a 1-LSB 16-bit signal, so it is unambiguously
        # a sentinel value and will never be confused with a real reading.
        db_peak = 20 * log10(avg_peak) if avg_peak > 0 else -128
        db_pulse_normalized = db_peak - _DB_REFERENCE
        db_background = 20 * log10(avg_noise) if avg_noise > 0 else -128
        db_background_normalized = db_background - _DB_REFERENCE
        snr = db_pulse_normalized - db_background_normalized

        return round(snr, 2), db_pulse_normalized, db_background_normalized


@njit  # Numba JIT-compiles this tight inner loop; avoids Python per-element overhead (~10× faster)
def _average_pulse_amplitude(mono_amplitude_array: np.ndarray, sample_rate: int,
                             pulse_rate: int, analysis_size: int, start_index: int) -> int:
    """Average the amplitude at each pulse position across the analysis window.

    Samples _PULSE_WIDTH_SAMPLES adjacent values per pulse position, then divides by the
    total sample count.  Equivalent to correlating with the pulse kernel but without
    multiplying through the ~97% zeros.
    """
    samples_per_pulse = sample_rate / pulse_rate
    total = 0
    for i in range(analysis_size):
        pos = int(i * samples_per_pulse)
        for j in range(_PULSE_WIDTH_SAMPLES):
            total += mono_amplitude_array[start_index + pos + j]
    return total // (_PULSE_WIDTH_SAMPLES * analysis_size)


def _build_pulse_kernel(sample_rate: int, pulse_rate: int) -> np.ndarray:
    """Build the symmetric pps scan kernel covering half a second of pulses.

    Length is int((scan_pulses-1)*samples_per_pulse)+3, placing the last pulse
    group flush against the end so the kernel is an exact palindrome.  A palindrome
    kernel means fftconvolve (convolution) == cross-correlation.
    """
    samples_per_pulse = sample_rate / pulse_rate
    scan_pulses = pulse_rate // 2  # half a second worth of pulses
    last_pos = int((scan_pulses - 1) * samples_per_pulse)
    coefficients = zeros(last_pos + _PULSE_WIDTH_SAMPLES, dtype=uint32)
    for i in range(scan_pulses):
        pos = int(i * samples_per_pulse)
        coefficients[pos:pos + _PULSE_WIDTH_SAMPLES] = 1
    return coefficients


def _calculate_pps_fit_array(mono_amplitude_array: np.ndarray, kernel: np.ndarray, scan_pulses: int) -> np.ndarray:
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  Uses fftconvolve with a symmetric kernel so convolution ==
    correlation; O((N+M) log(N+M)) vs O(N*M) for direct correlation.

    mode='valid' returns only positions where the kernel fits entirely within the
    signal, avoiding edge artefacts from the convolution.
    """
    raw = fftconvolve(mono_amplitude_array.astype(np.float64), kernel, mode='valid')
    return np.rint(raw).astype(np.int64) // (_PULSE_WIDTH_SAMPLES * scan_pulses)

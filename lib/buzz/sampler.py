"""
Audio sampling and pulse-train analysis for powerline interference detection.

Records a short audio clip from the configured input device, then uses FFT-based
convolution to find the phase and amplitude of the periodic pulse train produced
by arcing powerline hardware.  Returns signal level, noise floor, and SNR in dBFS
so the caller can convert to dBm using the station's calibration offset.
"""

from math import log10

import numpy as np
import sounddevice as sd
from numba import njit
from numpy import uint32, zeros
from scipy.signal import fftconvolve

from buzz.config import BuzzConfig

# dBFS reference for 16-bit audio: 0 dBFS = full-scale amplitude of 32768 (2^15)
_DB_REFERENCE = 20 * log10(32768.0)


class AudioSampler:
    def __init__(self, config: BuzzConfig):
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
                print(f'Warning: device {audio.device_index} is now "{current_name}", '
                      f'expected "{audio.input_device_name}". Using index anyway.')
            self._device_index = audio.device_index
        else:
            device = sd.query_devices(audio.input_device_name, 'input')
            self._device_index = device['index']
        self._kernel = _build_pulse_kernel(audio.sample_rate, audio.pulse_rate)
        self._scan_pulses = audio.pulse_rate // 2

    def take_sample(self) -> tuple[float, float, float]:
        """Record one audio sample and return (snr_db, signal_dbfs, noise_dbfs).

        Blocks for config.audio.duration seconds while recording.
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
        mono_amplitude_array = np.abs(recording[:, 0].astype(np.int32))

        output = _calculate_pps_fit_array(mono_amplitude_array, self._kernel, self._scan_pulses)

        peak_offset_index = output.argmax()
        noise_offset_index = output.argmin()

        # samples_per_pulse is a fractional period — the spacing between pulses in samples
        samples_per_pulse = audio.sample_rate / audio.pulse_rate
        peak_phase = int(peak_offset_index % samples_per_pulse)
        noise_phase = int(noise_offset_index % samples_per_pulse)

        # Start the analysis window after whichever phase offset is larger, so both
        # the peak and noise trains fit entirely within the recording
        analysis_start = max(peak_phase, noise_phase)
        analysis_size = int((len(mono_amplitude_array) - analysis_start) // samples_per_pulse)

        avg_peak = _average_pulse_amplitude(mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, peak_phase)
        avg_noise = _average_pulse_amplitude(mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, noise_phase)

        db_peak = 20 * log10(avg_peak) if avg_peak > 0 else -128
        db_pulse_normalized = db_peak - _DB_REFERENCE
        db_background = 20 * log10(avg_noise) if avg_noise > 0 else -128
        db_background_normalized = db_background - _DB_REFERENCE
        snr = db_pulse_normalized - db_background_normalized

        return round(snr, 2), db_pulse_normalized, db_background_normalized


@njit
def _average_pulse_amplitude(mono_amplitude_array, sample_rate, pulse_rate, analysis_size, start_index):
    """Average the amplitude at each pulse position across the analysis window.

    Samples three adjacent values per pulse position (pulse spans ~3 samples at
    typical sample rates) then divides by the total count.  Equivalent to
    correlating with the pulse kernel but without multiplying through the ~97% zeros.
    """
    samples_per_pulse = sample_rate / pulse_rate
    total = 0
    for i in range(analysis_size):
        pos = int(i * samples_per_pulse)
        total += mono_amplitude_array[start_index + pos]
        total += mono_amplitude_array[start_index + pos + 1]
        total += mono_amplitude_array[start_index + pos + 2]
    return total // (3 * analysis_size)


def _build_pulse_kernel(sample_rate: int, pulse_rate: int) -> np.ndarray:
    """Build the symmetric pps scan kernel covering half a second of pulses.

    Length is int((scan_pulses-1)*samples_per_pulse)+3, placing the last pulse
    group flush against the end so the kernel is an exact palindrome.  A palindrome
    kernel means fftconvolve (convolution) == cross-correlation.
    """
    samples_per_pulse = sample_rate / pulse_rate
    scan_pulses = pulse_rate // 2  # half a second worth of pulses
    last_pos = int((scan_pulses - 1) * samples_per_pulse)
    coefficients = zeros(last_pos + 3, dtype=uint32)
    for i in range(scan_pulses):
        pos = int(i * samples_per_pulse)
        coefficients[pos:pos + 3] = 1
    return coefficients


def _calculate_pps_fit_array(mono_amplitude_array, kernel, scan_pulses):
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  Uses fftconvolve with a symmetric kernel so convolution ==
    correlation; O((N+M) log(N+M)) vs O(N*M) for direct correlation.

    mode='valid' returns only positions where the kernel fits entirely within the
    signal, avoiding edge artefacts from the convolution.
    """
    raw = fftconvolve(mono_amplitude_array.astype(np.float64), kernel, mode='valid')
    return np.rint(raw).astype(np.int64) // (3 * scan_pulses)

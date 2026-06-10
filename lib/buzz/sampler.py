from math import log10

import numpy as np
import sounddevice as sd
from numba import njit
from numpy import uint32, zeros
from scipy.signal import fftconvolve

from buzz.config import BuzzConfig

_DB_REFERENCE = 20 * log10(32768.0)


class AudioSampler:
    def __init__(self, config: BuzzConfig):
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

    @property
    def sample_data(self) -> tuple[float, float, float]:
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
        min_offset_index = output.argmin()

        peak_sample_frequency = audio.sample_rate / audio.pulse_rate
        first_peak_index = int(peak_offset_index % peak_sample_frequency)
        first_noise_index = int(min_offset_index % peak_sample_frequency)

        latest_start = max(first_peak_index, first_noise_index)
        analysis_size = int((len(mono_amplitude_array) - latest_start) // peak_sample_frequency)

        avg_peak = _sum_pulse_train(mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, first_peak_index)
        avg_noise = _sum_pulse_train(mono_amplitude_array, audio.sample_rate, audio.pulse_rate, analysis_size, first_noise_index)

        db_peak = 20 * log10(avg_peak) if avg_peak > 0 else -128
        db_pulse_normalized = db_peak - _DB_REFERENCE
        db_background = 20 * log10(avg_noise) if avg_noise > 0 else -128
        db_background_normalized = db_background - _DB_REFERENCE
        snr = db_pulse_normalized - db_background_normalized

        return round(snr, 2), db_pulse_normalized, db_background_normalized


@njit
def _sum_pulse_train(mono_amplitude_array, sample_rate, pulse_rate, analysis_size, start_index):
    """Sum samples directly at the pulse positions — equivalent to correlating with the 0/1
    coefficient array but without the cost of multiplying through ~97% zeros."""
    peak_sample_frequency = sample_rate / pulse_rate
    total = 0
    for i in range(analysis_size):
        pos = int(i * peak_sample_frequency)
        total += mono_amplitude_array[start_index + pos]
        total += mono_amplitude_array[start_index + pos + 1]
        total += mono_amplitude_array[start_index + pos + 2]
    return total // (3 * analysis_size)


def _build_pulse_kernel(sample_rate: int, pulse_rate: int) -> np.ndarray:
    """Build the symmetric pps scan kernel covering half a second of pulses.

    Length is int((scan_pulses-1)*pf)+3, placing the last pulse group flush against
    the end so the kernel is an exact palindrome.  A palindrome kernel means
    fftconvolve (convolution) == cross-correlation.
    """
    pf = sample_rate / pulse_rate
    scan_pulses = pulse_rate // 2  # half a second worth of pulses
    last_pos = int((scan_pulses - 1) * pf)
    coefficients = zeros(last_pos + 3, dtype=uint32)
    for i in range(scan_pulses):
        pos = int(i * pf)
        coefficients[pos] = 1
        coefficients[pos + 1] = 1
        coefficients[pos + 2] = 1
    return coefficients


def _calculate_pps_fit_array(mono_amplitude_array, kernel, scan_pulses):
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  Uses fftconvolve with a symmetric kernel so convolution ==
    correlation; O((N+M) log(N+M)) vs O(N*M) for direct correlation.
    """
    raw = fftconvolve(mono_amplitude_array.astype(np.float64), kernel, mode='valid')
    return np.rint(raw).astype(np.int64) // (3 * scan_pulses)

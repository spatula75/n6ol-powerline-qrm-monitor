from math import pow, log10

import numpy as np
import sounddevice as sd
from numba import njit
from numpy import abs, uint32, zeros, array
from scipy.signal import fftconvolve

from buzz.config import BuzzConfig


class AudioSampler:
    def __init__(self, config: BuzzConfig):
        self._config = config
        device = sd.query_devices(config.input_device_name, 'input')
        self._device_index = device['index']

    @property
    def sample_data(self) -> tuple[float, float, float]:
        recording = sd.rec(
            int(self._config.duration * self._config.sample_rate),
            samplerate=self._config.sample_rate,
            channels=1,
            blocking=True,
            dtype='int16',
            device=self._device_index,
        )
        mono_amplitude_array = array([abs(channel[0]) for channel in recording])

        output = _calculate_pps_fit_array(mono_amplitude_array, self._config.sample_rate, self._config.pulse_rate)

        peak_offset_index = output.argmax()
        min_offset_index = output.argmin()

        peak_sample_frequency = self._config.sample_rate / self._config.pulse_rate
        peak_repeat_count = int(peak_offset_index / peak_sample_frequency)
        min_repeat_count = int(min_offset_index / peak_sample_frequency)
        first_peak_index = int(peak_offset_index - (peak_repeat_count * peak_sample_frequency))
        first_noise_index = int(min_offset_index - (min_repeat_count * peak_sample_frequency))

        latest_start = max(first_peak_index, first_noise_index)
        analysis_size = int((len(mono_amplitude_array) - latest_start) // peak_sample_frequency)

        avg_peak = _sum_pulse_train(mono_amplitude_array, self._config.sample_rate, self._config.pulse_rate, analysis_size, first_peak_index)
        avg_noise = _sum_pulse_train(mono_amplitude_array, self._config.sample_rate, self._config.pulse_rate, analysis_size, first_noise_index)

        db_reference = 20 * log10(pow(2, 16) / 2)
        db_peak = 20 * log10(avg_peak) if avg_peak > 0 else -128
        db_pulse_normalized = db_peak - db_reference
        db_background = 20 * log10(avg_noise) if avg_noise > 0 else -128
        db_background_normalized = db_background - db_reference
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


def _calculate_pps_fit_array(mono_amplitude_array, sample_rate, pulse_rate):
    """Return a score at each sample position for how well a pulse train starting
    there fits the data.  Uses fftconvolve with a symmetric kernel so convolution ==
    correlation; O((N+M) log(N+M)) vs O(N*M) for direct correlation.
    """
    kernel = _build_pulse_kernel(sample_rate, pulse_rate)
    scan_pulses = pulse_rate // 2
    raw = fftconvolve(mono_amplitude_array.astype(np.float64), kernel, mode='valid')
    return np.rint(raw).astype(np.int64) // (3 * scan_pulses)

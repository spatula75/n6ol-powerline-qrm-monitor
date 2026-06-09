import numpy as np
import scipy.io.wavfile as wav
import os


class NoiseGenerator:
    """
    Code to generate a pure form of the utility powerline noise for use as an example for comparison to noise
    received over the air.
    """

    SAMPLE_RATE = 16000
    BITS_PER_SAMPLE = 16
    AMPLITUDE_PERCENTAGE = 0.95  # Percentage of maximum amplitude at BITS_PER_SAMPLE the waveform should be
    AMPLITUDE = pow(2, BITS_PER_SAMPLE - 1) * AMPLITUDE_PERCENTAGE
    THRESHOLD_PERCENTAGE = 0.98  # At what percentage of the amplitude peak should we consider the voltage high enough to arc?
    DURATION = 10  # How long of a sound should be produced, in seconds
    POWER_HZ = 60
    OVERLAY_HZ = POWER_HZ * 4

    _ARRAY_LENGTH = SAMPLE_RATE * DURATION
    _THRESHOLD_AMPLITUDE = AMPLITUDE * THRESHOLD_PERCENTAGE

    def generate(self):
        sample_range = np.arange(NoiseGenerator.SAMPLE_RATE * NoiseGenerator.DURATION)
        sample_data = (NoiseGenerator.AMPLITUDE * np.sin(2 * np.pi * NoiseGenerator.POWER_HZ  * sample_range
                                                        / NoiseGenerator.SAMPLE_RATE))

        # Apply threshold
        mask = np.abs(sample_data) < NoiseGenerator._THRESHOLD_AMPLITUDE
        sample_data[mask] = 0

        overlay_data = np.sin(2 * np.pi * NoiseGenerator.OVERLAY_HZ * sample_range
                              / NoiseGenerator.SAMPLE_RATE)

        sample_data = (sample_data * overlay_data).astype(np.int16)

        return sample_data.astype(np.int16)

    def save(self, sample_data: np.ndarray, filename: str):
        wav.write(filename, NoiseGenerator.SAMPLE_RATE, sample_data)

if __name__ == '__main__':
    ng = NoiseGenerator()
    noise_data = ng.generate()
    ng.save(noise_data, r'C:\Users\passp\sample_noise.wav')

    print(",".join([str(i) for i in NoiseGenerator().generate().tolist()]))

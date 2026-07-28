"""
Run this script to regenerate the golden test fixtures in tests/resources/.

    python tests/resources/generate_goldens.py

Golden files pin the exact numeric output of the DSP pipeline against known
synthetic input.  Re-run and commit when the algorithm changes intentionally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'lib'))

import numpy as np

from buzz.dsp import amplitude_to_dbfs, analyze_window, build_pulse_kernel, calculate_pps_fit_array

SAMPLE_RATE = 16000
PULSE_RATE = 120
DURATION = 3
N = SAMPLE_RATE * DURATION          # 48000
PULSE_PHASE = 5
PULSE_AMPLITUDE = 20000
NOISE_LO, NOISE_HI = 50, 150
SCAN_PULSES = PULSE_RATE // 2

RESOURCES = Path(__file__).resolve().parent


def make_synthetic_audio() -> np.ndarray:
    """Return a (N, 1) int16 array: low noise with a 120pps pulse train at PULSE_PHASE."""
    rng = np.random.default_rng(42)
    audio = rng.integers(NOISE_LO, NOISE_HI, size=(N, 1), dtype=np.int16)
    spp = SAMPLE_RATE / PULSE_RATE
    for i in range(int(N / spp)):
        pos = PULSE_PHASE + round(i * spp)
        if pos + 3 < N:
            audio[pos, 0] = PULSE_AMPLITUDE
            audio[pos + 1, 0] = PULSE_AMPLITUDE
            audio[pos + 2, 0] = PULSE_AMPLITUDE
    return audio


def main() -> None:
    audio = make_synthetic_audio()
    np.save(RESOURCES / 'synthetic_audio.npy', audio)
    print(f'synthetic_audio.npy  shape={audio.shape}  dtype={audio.dtype}')

    mono = np.abs(audio[:, 0].astype(np.int32))
    kernel = build_pulse_kernel(SAMPLE_RATE, PULSE_RATE)
    fit = calculate_pps_fit_array(mono, kernel, SCAN_PULSES)
    np.save(RESOURCES / 'fit_array_golden.npy', fit)
    print(f'fit_array_golden.npy  shape={fit.shape}  peak_idx={fit.argmax()}')

    window = analyze_window(mono, SAMPLE_RATE, PULSE_RATE, kernel, SCAN_PULSES)
    assert window is not None
    db_signal = amplitude_to_dbfs(window.signal_amplitude)
    db_noise = amplitude_to_dbfs(window.noise_amplitude)
    snr = round(db_signal - db_noise, 2)

    golden = np.array([snr, db_signal, db_noise])
    np.save(RESOURCES / 'analysis_golden.npy', golden)
    print(f'analysis_golden.npy  snr={snr:.2f}  signal={db_signal:.2f}  noise={db_noise:.2f}')


if __name__ == '__main__':
    main()

"""Continuous pulse-train analysis publishing signal/noise levels for real-time display.

ContinuousAnalyzer runs on a daemon thread and maintains a three-state machine:

  SEARCHING   — no valid phase pair (initial startup only).  Runs the full FFT
                fit on 1 s of audio every SEARCH_INTERVAL seconds until a pulse
                train is found.

  LOCKED      — the 120 pps phase is known and the signal is present.  Each
                tick (~200 ms) it calls _average_pulse_amplitude at the stored
                peak and noise phases — O(scan_pulses) ≈ 60 operations.
                Every REFINE_INTERVAL seconds it re-runs the full FFT fit to
                correct slow mains-frequency drift.

  SIGNAL_LOST — phase pair known but signal absent.  Three-tier re-acquisition:
                Tier 1 (200 ms): _noise_check() samples live noise at _noise_phase
                  and tries _peak_phase for instant re-acquisition.
                Tier 2 (1 s): _phase_search() scans ±PHASE_SEARCH_RADIUS samples
                  around _peak_phase using Numba amplitude averaging — ~40× cheaper
                  than an FFT, handles slow mains-frequency drift.
                Tier 3 (30 s): _full_analysis() FFT fallback for large drift or
                  extended absence; updates both phases if the signal is found.

Results are deposited in a lock-protected slot; the Qt UI polls it on each
paint tick without blocking.
"""

import threading
import time
from dataclasses import dataclass
from math import log10

import numpy as np

from buzz.config import BuzzConfig
from buzz.sampler import (
    AudioPipeline,
    _average_pulse_amplitude,
    _build_pulse_kernel,
    _calculate_pps_fit_array,
)

_DB_REFERENCE = 20 * log10(32768.0)


@dataclass(frozen=True)
class AnalysisResult:
    signal_dbm: float
    noise_dbm: float
    snr: float
    locked: bool


class ContinuousAnalyzer:
    """Background analysis thread; call start() once, then poll latest_result()."""

    LOCK_ACQUIRE_SNR    = 6.0   # dB — minimum SNR to enter LOCKED
    LOCK_LOSE_SNR       = 2.0   # dB — SNR below which consecutive failures are counted
    LOSE_LOCK_COUNT     = 3     # consecutive _quick_check failures before SIGNAL_LOST
    LOCKED_INTERVAL     = 0.2   # s  — fast tick while LOCKED or SIGNAL_LOST
    SEARCH_INTERVAL     = 1.0   # s  — narrow phase-search interval while SIGNAL_LOST
    REFINE_INTERVAL     = 10.0  # s  — full FFT phase-refinement interval while LOCKED
    SIGNAL_LOST_REFINE  = 30.0  # s  — full FFT fallback interval while SIGNAL_LOST
    PHASE_SEARCH_RADIUS = 10    # samples either side of stored peak to scan in SIGNAL_LOST

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig) -> None:
        self._pipeline    = pipeline
        audio             = config.audio
        self._sample_rate = audio.sample_rate
        self._pulse_rate  = audio.pulse_rate
        self._offset      = config.station.audio_rf_conversion_db
        self._n_samples   = audio.sample_rate          # 1 s window
        self._scan_pulses = audio.pulse_rate // 2
        self._spp         = audio.sample_rate / audio.pulse_rate
        self._kernel      = _build_pulse_kernel(audio.sample_rate, audio.pulse_rate)

        self._state       = 'SEARCHING'
        self._peak_phase  = 0
        self._noise_phase = 0

        # True once we have acquired at least one lock; kept True even after the
        # signal disappears so SIGNAL_LOST can reuse the stored phases.
        self._phases_valid: bool = False
        self._consecutive_low_snr: int = 0

        self._result: AnalysisResult | None = None
        self._result_lock = threading.Lock()
        self._stop        = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True, name='analyzer')

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest_result(self) -> AnalysisResult | None:
        with self._result_lock:
            return self._result

    # ----------------------------------------------------------------- private

    def _publish(self, result: AnalysisResult) -> None:
        with self._result_lock:
            self._result = result

    def _to_dbm(self, amplitude: float) -> float:
        return (20 * log10(amplitude) - _DB_REFERENCE + self._offset
                if amplitude > 0 else -128.0)

    def _sample_phases(self, abs_data: np.ndarray) -> tuple[float, float, float] | None:
        """Sample signal and noise amplitudes at the stored phases.

        Returns (sig_dbm, noise_dbm, snr) or None if the buffer is too short.
        """
        start = max(self._peak_phase, self._noise_phase)
        size  = int((len(abs_data) - start) // self._spp)
        if size < 1:
            return None
        sig_amp   = float(_average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, size, self._peak_phase))
        noise_amp = float(_average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, size, self._noise_phase))
        sig_dbm   = self._to_dbm(sig_amp)
        noise_dbm = self._to_dbm(noise_amp)
        return sig_dbm, noise_dbm, sig_dbm - noise_dbm

    def _run(self) -> None:  # pragma: no cover
        last_narrow = 0.0   # last _phase_search call while SIGNAL_LOST
        last_fft    = 0.0   # last _full_analysis call (SEARCHING or SIGNAL_LOST fallback)
        last_refine = 0.0   # last _full_analysis call while LOCKED
        while not self._stop.is_set():
            now = time.monotonic()
            if self._state == 'LOCKED':
                if now - last_refine >= self.REFINE_INTERVAL:
                    self._full_analysis()
                    last_refine = time.monotonic()
                else:
                    self._quick_check()
                self._stop.wait(self.LOCKED_INTERVAL)
            elif self._state == 'SIGNAL_LOST':
                # Tier 1 (200 ms): live noise + exact-phase re-acquisition attempt
                self._noise_check()
                if self._state == 'LOCKED':
                    last_refine = time.monotonic()
                else:
                    now = time.monotonic()
                    # Tier 2 (1 s): cheap narrow amplitude scan ± PHASE_SEARCH_RADIUS
                    if now - last_narrow >= self.SEARCH_INTERVAL:
                        self._phase_search()
                        last_narrow = now
                    if self._state == 'LOCKED':
                        last_refine = time.monotonic()
                    # Tier 3 (30 s): full FFT fallback for large phase drift / long absence
                    elif now - last_fft >= self.SIGNAL_LOST_REFINE:
                        self._full_analysis()
                        last_fft = time.monotonic()
                        if self._state == 'LOCKED':
                            last_refine = time.monotonic()
                self._stop.wait(self.LOCKED_INTERVAL)
            else:  # SEARCHING — no valid phases yet
                self._full_analysis()
                last_fft = time.monotonic()
                if self._state == 'LOCKED':
                    last_refine = time.monotonic()
                self._stop.wait(self.SEARCH_INTERVAL)

    def _full_analysis(self) -> None:
        """FFT fit over 1 s of audio; establishes or refreshes the locked phase pair.

        When called from SIGNAL_LOST and no lock is found, nothing is published —
        _noise_check() is already providing live noise results on each tick.
        """
        if not self._pipeline.wait_for_data(self._n_samples, timeout=2.0):
            return
        snapshot = self._pipeline.get_snapshot(self._n_samples)
        abs_data = np.abs(snapshot.astype(np.int32))

        fit       = _calculate_pps_fit_array(abs_data, self._kernel, self._scan_pulses)
        peak_idx  = int(fit.argmax())
        noise_idx = int(fit.argmin())

        peak_phase  = int(peak_idx  % self._spp)
        noise_phase = int(noise_idx % self._spp)
        start       = max(peak_phase, noise_phase)
        size        = int((len(abs_data) - start) // self._spp)
        if size < 1:
            return

        sig_amp   = float(_average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, size, peak_phase))
        noise_amp = float(_average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, size, noise_phase))

        sig_dbm   = self._to_dbm(sig_amp)
        noise_dbm = self._to_dbm(noise_amp)
        snr       = sig_dbm - noise_dbm

        if snr >= self.LOCK_ACQUIRE_SNR:
            self._peak_phase          = peak_phase
            self._noise_phase         = noise_phase
            self._state               = 'LOCKED'
            self._consecutive_low_snr = 0
            self._phases_valid        = True
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
        elif not self._phases_valid:
            # SEARCHING: no stored phases to fall back on — publish what the FFT found
            self._publish(AnalysisResult(
                signal_dbm=noise_dbm, noise_dbm=noise_dbm, snr=0.0, locked=False,
            ))
        # else: SIGNAL_LOST with valid phases — _noise_check() handles publishing

    def _phase_search(self) -> bool:
        """Narrow amplitude scan ± PHASE_SEARCH_RADIUS around the stored peak phase.

        ~40× cheaper than _full_analysis() — evaluates _average_pulse_amplitude at
        2*PHASE_SEARCH_RADIUS+1 candidate phases (Numba JIT) rather than running an
        FFT convolution over the full audio window.  If the best candidate passes
        LOCK_ACQUIRE_SNR, _peak_phase is updated to the refined position and the
        machine transitions to LOCKED.  Returns True if re-locked, False otherwise.
        Does not publish on failure; _noise_check() already published the noise result
        on this tick.
        """
        if not self._pipeline.wait_for_data(self._n_samples, timeout=2.0):
            return False
        snapshot = self._pipeline.get_snapshot(self._n_samples)
        abs_data = np.abs(snapshot.astype(np.int32))
        spp      = self._spp
        spp_int  = int(spp)

        best_amp   = -1.0
        best_phase = self._peak_phase
        for offset in range(-self.PHASE_SEARCH_RADIUS, self.PHASE_SEARCH_RADIUS + 1):
            candidate = (self._peak_phase + offset) % spp_int
            start     = max(candidate, self._noise_phase)
            size      = int((len(abs_data) - start) // spp)
            if size < 1:
                continue
            amp = float(_average_pulse_amplitude(
                abs_data, self._sample_rate, self._pulse_rate, size, candidate))
            if amp > best_amp:
                best_amp   = amp
                best_phase = candidate

        start = max(best_phase, self._noise_phase)
        size  = int((len(abs_data) - start) // spp)
        if size < 1:
            return False
        noise_amp = float(_average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, size, self._noise_phase))

        sig_dbm   = self._to_dbm(best_amp)
        noise_dbm = self._to_dbm(noise_amp)

        if sig_dbm - noise_dbm >= self.LOCK_ACQUIRE_SNR:
            self._peak_phase          = best_phase
            self._consecutive_low_snr = 0
            self._state               = 'LOCKED'
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm,
                snr=sig_dbm - noise_dbm, locked=True,
            ))
            return True
        return False

    def _noise_check(self) -> None:
        """Live noise + fast signal re-acquisition at stored phases (SIGNAL_LOST only).

        Samples the noise floor at _noise_phase every tick so the NF meter stays
        current.  Also samples _peak_phase; if SNR is high enough, transitions
        directly back to LOCKED without needing a full FFT fit.
        """
        if not self._pipeline.wait_for_data(self._n_samples, timeout=2.0):
            return
        snapshot = self._pipeline.get_snapshot(self._n_samples)
        abs_data = np.abs(snapshot.astype(np.int32))

        measured = self._sample_phases(abs_data)
        if measured is None:
            return
        sig_dbm, noise_dbm, snr = measured

        if snr >= self.LOCK_ACQUIRE_SNR:
            self._consecutive_low_snr = 0
            self._state = 'LOCKED'
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
        else:
            self._publish(AnalysisResult(
                signal_dbm=noise_dbm, noise_dbm=noise_dbm, snr=0.0, locked=False,
            ))

    def _quick_check(self) -> None:
        """Cheap amplitude check at stored phases; debounces lock loss (LOCKED only)."""
        if not self._pipeline.wait_for_data(self._n_samples, timeout=2.0):
            return
        snapshot = self._pipeline.get_snapshot(self._n_samples)
        abs_data = np.abs(snapshot.astype(np.int32))

        measured = self._sample_phases(abs_data)
        if measured is None:
            return
        sig_dbm, noise_dbm, snr = measured

        if snr < self.LOCK_LOSE_SNR:
            self._consecutive_low_snr += 1
            if self._consecutive_low_snr >= self.LOSE_LOCK_COUNT:
                self._state               = 'SIGNAL_LOST'
                self._consecutive_low_snr = 0
                self._publish(AnalysisResult(
                    signal_dbm=noise_dbm, noise_dbm=noise_dbm, snr=0.0, locked=False,
                ))
            # else: hold current result during debounce window — don't publish
        else:
            self._consecutive_low_snr = 0
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))

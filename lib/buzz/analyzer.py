"""Continuous pulse-train analysis publishing signal/noise levels for real-time display.

ContinuousAnalyzer runs on a daemon thread and maintains a three-state machine:

  SEARCHING   — no valid phase pair (initial startup only).  Runs the full FFT
                fit on 1 s of audio every SEARCH_INTERVAL seconds until a pulse
                train is found.

  LOCKED      — the 120 pps phase is known and the signal is present.  Each
                tick (~200 ms) it calls average_pulse_amplitude at the stored
                peak and noise phases — O(scan_pulses) ≈ 60 operations.
                Every REFINE_INTERVAL seconds it runs _phase_search() to
                correct slow mains-frequency drift.  A full FFT isn't needed
                here: any drift too large for the narrow scan would already
                cause _quick_check() to lose lock first.

  SIGNAL_LOST — phase pair known but signal absent.  Four-tier re-acquisition:
                Tier 1 (200 ms): _noise_check() samples live noise at _noise_phase
                  and tries _peak_phase for instant re-acquisition.
                Tier 2 (1 s): _phase_search() scans ±PHASE_SEARCH_RADIUS samples
                  around _peak_phase using Numba amplitude averaging — ~40× cheaper
                  than an FFT, handles slow mains-frequency drift.
                Tier 3a (5 s): _fast_scan() runs a short-kernel FFT (FAST_SCAN_PULSES
                  pulses, FAST_SCAN_SAMPLES audio) as a cheap candidate detector; ~6×
                  cheaper than the full FFT, skips Tier 3b when nothing is present.
                Tier 3b (on Tier-3a hit, or every SIGNAL_LOST_REFINE as safety net):
                  _full_analysis() with the full kernel to confirm and refresh phases.

State transitions are centralised: each tier method measures, publishes, and
returns the state it believes the machine should be in; the per-state tick
methods pass that through _transition(), which owns all transition bookkeeping
(state change, debounce-counter reset, phase validation, refine timestamp).

Results are deposited in a lock-protected slot; the Qt UI polls it on each
paint tick without blocking.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from math import gcd, log10

import numpy as np

from buzz.config import BuzzConfig
from buzz.dsp import (
    amplitude_to_dbm,
    analyze_window,
    average_pulse_amplitude,
    build_pulse_kernel,
    calculate_pps_fit_array,
)
from buzz.sampler import AudioPipeline


class AnalyzerState(StrEnum):
    SEARCHING   = 'SEARCHING'
    LOCKED      = 'LOCKED'
    SIGNAL_LOST = 'SIGNAL_LOST'


@dataclass(frozen=True)
class AnalysisResult:
    signal_dbm: float
    noise_dbm: float
    snr: float
    locked: bool

    @classmethod
    def unlocked(cls, noise_dbm: float) -> 'AnalysisResult':
        """Result for a tick with no pulse-train lock.

        By convention an unlocked result has signal_dbm == noise_dbm and snr == 0:
        the plotter and the meter panel both rely on the pair coinciding so unlocked
        stretches render as a single continuous noise trace rather than a gap.
        """
        return cls(signal_dbm=noise_dbm, noise_dbm=noise_dbm, snr=0.0, locked=False)


class ContinuousAnalyzer:
    """Background analysis thread; call start() once, then poll latest_result()."""

    LOCK_ACQUIRE_SNR    = 6.0   # dB — minimum SNR to enter LOCKED
    LOCK_LOSE_SNR       = 2.0   # dB — SNR below which consecutive failures are counted
    LOSE_LOCK_COUNT     = 3     # consecutive _quick_check failures before SIGNAL_LOST
    FAST_TICK_INTERVAL  = 0.2   # s  — tick cadence in LOCKED and SIGNAL_LOST
    SEARCH_INTERVAL     = 1.0   # s  — SEARCHING tick cadence; also the Tier-2 narrow-scan cadence in SIGNAL_LOST
    REFINE_INTERVAL     = 2.0   # s  — phase-search refinement interval while LOCKED
    SIGNAL_LOST_REFINE  = 120.0 # s  — unconditional full-FFT safety net in SIGNAL_LOST
    PHASE_SEARCH_RADIUS = 10    # samples either side of stored peak to scan in SIGNAL_LOST
    FAST_SCAN_PULSES    = 15    # pulses in the Tier-3a screening kernel (~1/4 of full)
    FAST_SCAN_SAMPLES   = 4000  # audio window for Tier-3a (~0.25 s at 16 kHz)
    FAST_SCAN_INTERVAL  = 5.0   # s  — Tier-3a cadence in SIGNAL_LOST
    FAST_SCAN_SNR       = 4.0   # dB — Tier-3a hit threshold; triggers Tier-3b full FFT
    CAPTURE_TIMEOUT     = 2.0   # s  — max wait for the pipeline to supply a window

    def __init__(self, pipeline: AudioPipeline, config: BuzzConfig) -> None:
        self._pipeline          = pipeline
        audio                   = config.audio
        self._sample_rate       = audio.sample_rate
        self._pulse_rate        = audio.pulse_rate
        self._offset_db         = config.station.audio_rf_conversion_db
        self._window_samples    = audio.sample_rate    # 1 s analysis window
        self._scan_pulses       = audio.pulse_rate // 2   # half a second of pulses (full kernel)
        self._samples_per_pulse = audio.sample_rate / audio.pulse_rate
        self._kernel            = build_pulse_kernel(audio.sample_rate, audio.pulse_rate)
        self._fast_kernel       = build_pulse_kernel(
            audio.sample_rate, audio.pulse_rate, n_pulses=self.FAST_SCAN_PULSES)
        # Snapshot alignment: the smallest whole-sample interval that is an exact
        # number of pulse periods (400 samples = 3 periods at 16 kHz / 120 pps).
        # Windows ending on multiples of this share a phase origin, so phases
        # learned in one snapshot stay valid in every later one.
        self._phase_align = audio.sample_rate // gcd(audio.sample_rate, audio.pulse_rate)

        self._state       = AnalyzerState.SEARCHING
        self._peak_phase  = 0
        self._noise_phase = 0

        # True once we have acquired at least one lock; kept True even after the
        # signal disappears so SIGNAL_LOST can reuse the stored phases.
        self._phases_valid: bool = False
        self._consecutive_low_snr: int = 0

        # Cadence timestamps (monotonic); owned by the tick methods and _transition.
        self._last_refine      = 0.0  # last _phase_search while LOCKED (or lock acquisition)
        self._last_narrow_scan = 0.0  # last _phase_search while SIGNAL_LOST
        self._last_full_fft    = 0.0  # last _full_analysis (SEARCHING or SIGNAL_LOST backstop)
        self._last_fast_scan   = 0.0  # last _fast_scan while SIGNAL_LOST

        self._latest_result: AnalysisResult | None = None
        # Drained by the collector once per minute.  At the fastest publish cadence
        # (one per FAST_TICK_INTERVAL) a minute produces ~300 results; 600 gives a
        # late collection cycle a full extra minute before results are lost.
        self._result_buffer: deque[AnalysisResult] = deque(maxlen=600)
        self._latest_signal_correction: int = 0
        self._latest_noise_correction: int = 0
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
            return self._latest_result

    def latest_signal_correction(self) -> int:
        with self._result_lock:
            return self._latest_signal_correction

    def latest_noise_correction(self) -> int:
        with self._result_lock:
            return self._latest_noise_correction

    def drain_results(self) -> list[AnalysisResult]:
        """Return all results published since the last drain (oldest first) and clear them.

        Draining rather than copying keeps successive collector cycles averaging
        disjoint sets of results — a non-draining read would re-average the tail
        of the previous minute into every row.
        """
        with self._result_lock:
            results = list(self._result_buffer)
            self._result_buffer.clear()
            return results

    # ----------------------------------------------------- state machine core

    def _transition(self, new_state: AnalyzerState) -> None:
        """Apply a state change proposed by a tier method.

        The single place transition bookkeeping happens: entering any new state
        resets the lock-loss debounce counter; entering LOCKED additionally marks
        the stored phases valid and stamps the refine timer so a fresh lock isn't
        immediately re-refined.  A proposal matching the current state is a no-op.
        """
        if new_state == self._state:
            return
        self._consecutive_low_snr = 0
        if new_state == AnalyzerState.LOCKED:
            self._phases_valid = True
            self._last_refine  = time.monotonic()
        self._state = new_state

    def _run(self) -> None:  # pragma: no cover
        while not self._stop.is_set():
            if self._state == AnalyzerState.LOCKED:
                interval = self._locked_tick()
            elif self._state == AnalyzerState.SIGNAL_LOST:
                interval = self._signal_lost_tick()
            else:
                interval = self._searching_tick()
            self._stop.wait(interval)

    def _searching_tick(self) -> float:
        self._transition(self._full_analysis())
        self._last_full_fft = time.monotonic()
        return self.SEARCH_INTERVAL

    def _locked_tick(self) -> float:
        if time.monotonic() - self._last_refine >= self.REFINE_INTERVAL:
            self._transition(self._phase_search())
            self._last_refine = time.monotonic()
        else:
            self._transition(self._quick_check())
        return self.FAST_TICK_INTERVAL

    def _signal_lost_tick(self) -> float:
        # Tier 1 (200 ms): live noise + exact-phase re-acquisition attempt
        self._transition(self._noise_check())
        # Tier 2 (1 s): cheap narrow amplitude scan ± PHASE_SEARCH_RADIUS
        if (self._state != AnalyzerState.LOCKED
                and time.monotonic() - self._last_narrow_scan >= self.SEARCH_INTERVAL):
            self._transition(self._phase_search())
            self._last_narrow_scan = time.monotonic()
        # Tier 3a (5 s): short-kernel FFT screens for a candidate;
        # Tier 3b: full FFT on a hit, or every SIGNAL_LOST_REFINE as backstop
        if (self._state != AnalyzerState.LOCKED
                and time.monotonic() - self._last_fast_scan >= self.FAST_SCAN_INTERVAL):
            triggered = self._fast_scan()
            self._last_fast_scan = time.monotonic()
            if triggered or time.monotonic() - self._last_full_fft >= self.SIGNAL_LOST_REFINE:
                self._transition(self._full_analysis())
                self._last_full_fft = time.monotonic()
        return self.FAST_TICK_INTERVAL

    # ------------------------------------------------------------ tier methods
    #
    # Each measures one window, publishes any result, and returns the state the
    # machine should be in.  None of them mutates _state directly — that is
    # _transition()'s job.

    def _publish(self, result: AnalysisResult) -> None:
        with self._result_lock:
            self._latest_result = result
            self._result_buffer.append(result)

    def _to_dbm(self, amplitude: float) -> float:
        return amplitude_to_dbm(amplitude, self._offset_db)

    def _capture(self, n_samples: int) -> np.ndarray | None:
        """Wait for n_samples of phase-aligned audio and return it rectified (absolute int32).

        Snapshots are aligned to _phase_align so every window starts at the same
        offset within the pulse period.  Without this the window origin moves with
        the ring buffer tail between ticks, and phases stored from one snapshot
        point at the wrong samples in the next — silently breaking _quick_check,
        _noise_check, and _phase_search.

        Returns None when the pipeline cannot supply the data within
        CAPTURE_TIMEOUT — the caller should leave the state machine unchanged.
        """
        if not self._pipeline.wait_for_data(n_samples + self._phase_align,
                                            timeout=self.CAPTURE_TIMEOUT):
            return None
        snapshot = self._pipeline.get_snapshot(n_samples, align=self._phase_align)
        return np.abs(snapshot.astype(np.int32))

    def _sample_phases(self, abs_data: np.ndarray,
                       peak_phase: int | None = None,
                       noise_phase: int | None = None) -> tuple[float, float, float] | None:
        """Sample signal and noise amplitudes at the given phases (stored ones by default).

        Returns (sig_dbm, noise_dbm, snr) or None if the buffer is too short.
        """
        peak     = self._peak_phase if peak_phase is None else peak_phase
        noise    = self._noise_phase if noise_phase is None else noise_phase
        # Start after the larger of the two phase offsets so both pulse trains
        # fit entirely inside the window and are averaged over the same pulses.
        start    = max(peak, noise)
        n_pulses = int((len(abs_data) - start) // self._samples_per_pulse)
        if n_pulses < 1:
            return None
        sig_amp   = float(average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, n_pulses, peak))
        noise_amp = float(average_pulse_amplitude(
            abs_data, self._sample_rate, self._pulse_rate, n_pulses, noise))
        sig_dbm   = self._to_dbm(sig_amp)
        noise_dbm = self._to_dbm(noise_amp)
        return sig_dbm, noise_dbm, sig_dbm - noise_dbm

    def _fast_scan(self) -> bool:
        """Short-kernel FFT screening: cheaply detect whether a signal candidate exists.

        Uses FAST_SCAN_PULSES and FAST_SCAN_SAMPLES (~0.25 s) so the FFT is ~6× cheaper
        than the full analysis.  Returns True when the best-phase fit score exceeds the
        worst by FAST_SCAN_SNR dB — a weak hint that something is worth confirming.
        Does not publish or propose a state; only gates whether Tier 3b runs.
        """
        abs_data = self._capture(self.FAST_SCAN_SAMPLES)
        if abs_data is None:
            return False
        fit = calculate_pps_fit_array(abs_data, self._fast_kernel, self.FAST_SCAN_PULSES)
        if len(fit) < 2:
            return False
        peak   = float(fit.max())
        trough = float(fit.min())
        if trough <= 0:
            # Quantised fit scores can floor to zero on very quiet audio; the dB
            # ratio is undefined then, so treat any positive peak as a candidate.
            return peak > 0
        return 20 * log10(peak / trough) >= self.FAST_SCAN_SNR

    def _full_analysis(self) -> AnalyzerState:
        """FFT fit over 1 s of audio; establishes or refreshes the locked phase pair.

        Returns LOCKED when a pulse train passes LOCK_ACQUIRE_SNR, otherwise the
        current state.  When called from SIGNAL_LOST and no lock is found, nothing
        is published — _noise_check() is already providing live noise results on
        each tick.
        """
        abs_data = self._capture(self._window_samples)
        if abs_data is None:
            return self._state

        window = analyze_window(abs_data, self._sample_rate, self._pulse_rate,
                                self._kernel, self._scan_pulses)
        if window is None:
            return self._state

        sig_dbm   = self._to_dbm(window.signal_amplitude)
        noise_dbm = self._to_dbm(window.noise_amplitude)
        snr       = sig_dbm - noise_dbm

        if snr >= self.LOCK_ACQUIRE_SNR:
            self._peak_phase  = window.peak_phase
            self._noise_phase = window.noise_phase
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
            return AnalyzerState.LOCKED
        if not self._phases_valid:
            # SEARCHING: no stored phases to fall back on — publish what the FFT found
            self._publish(AnalysisResult.unlocked(noise_dbm))
        # else: SIGNAL_LOST with valid phases — _noise_check() handles publishing
        return self._state

    def _scan_phase(self, abs_data: np.ndarray, center: int, anchor: int,
                    minimize: bool) -> tuple[int, int]:
        """Scan ± PHASE_SEARCH_RADIUS around center for the best pulse amplitude.

        anchor is the other pulse train's phase; the analysis window starts at
        max(candidate, anchor) so both trains fit within the same audio.  minimize
        selects the quietest candidate (noise scan) instead of the loudest (signal
        scan).  Returns (best_phase, best_offset); center with offset 0 if no
        candidate fits.
        """
        spp_int     = int(self._samples_per_pulse)
        best_amp    = float('inf') if minimize else -1.0
        best_phase  = center
        best_offset = 0
        for offset in range(-self.PHASE_SEARCH_RADIUS, self.PHASE_SEARCH_RADIUS + 1):
            candidate = (center + offset) % spp_int
            start     = max(candidate, anchor)
            n_pulses  = int((len(abs_data) - start) // self._samples_per_pulse)
            if n_pulses < 1:
                continue
            amp = float(average_pulse_amplitude(
                abs_data, self._sample_rate, self._pulse_rate, n_pulses, candidate))
            if (amp < best_amp) if minimize else (amp > best_amp):
                best_amp    = amp
                best_phase  = candidate
                best_offset = offset
        return best_phase, best_offset

    def _phase_search(self) -> AnalyzerState:
        """Narrow amplitude scan ± PHASE_SEARCH_RADIUS around each stored phase.

        ~40× cheaper than _full_analysis() — evaluates average_pulse_amplitude at
        2*PHASE_SEARCH_RADIUS+1 candidates per phase (Numba JIT) rather than running
        an FFT over the full audio window.

        Signal and noise phases are searched independently.  The tempting shortcut of
        shifting _noise_phase by the same delta as _peak_phase is wrong in the general
        case: the quiet inter-pulse window is wherever no arc source happens to land,
        and with multiple overlapping sources that window can drift at a completely
        different rate — or disappear and reappear elsewhere — independent of any one
        source's phase.  Searching both independently costs one extra pass of 21
        Numba amplitude averages and correctly handles all source configurations.

        Each scan's winning offset is recorded separately (_latest_signal_correction for
        signal, _latest_noise_correction for noise) on every call, win or lose, so the
        UI can display how much each phase actually moved rather than assuming they
        track together.

        If the best signal candidate passes LOCK_ACQUIRE_SNR both phases are updated
        and LOCKED is returned; otherwise the current state.  Does not publish on
        failure; _noise_check() already published the noise result on this tick.
        """
        abs_data = self._capture(self._window_samples)
        if abs_data is None:
            return self._state

        # Signal scan: loudest candidate near _peak_phase.
        best_phase, best_offset = self._scan_phase(
            abs_data, self._peak_phase, anchor=self._noise_phase, minimize=False)
        with self._result_lock:
            self._latest_signal_correction = best_offset

        # Noise scan: independently, the quietest candidate near _noise_phase.
        best_noise_phase, best_noise_offset = self._scan_phase(
            abs_data, self._noise_phase, anchor=best_phase, minimize=True)
        with self._result_lock:
            self._latest_noise_correction = best_noise_offset

        # Re-measure both winners over a consistent window before the SNR decision.
        measured = self._sample_phases(abs_data, best_phase, best_noise_phase)
        if measured is None:
            return self._state
        sig_dbm, noise_dbm, snr = measured

        if snr >= self.LOCK_ACQUIRE_SNR:
            self._peak_phase  = best_phase
            self._noise_phase = best_noise_phase
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
            return AnalyzerState.LOCKED
        return self._state

    def _noise_check(self) -> AnalyzerState:
        """Live noise + fast signal re-acquisition at stored phases (SIGNAL_LOST only).

        Samples the noise floor at _noise_phase every tick so the NF meter stays
        current.  Also samples _peak_phase; if SNR is high enough, returns LOCKED
        without needing a full FFT fit.
        """
        abs_data = self._capture(self._window_samples)
        if abs_data is None:
            return self._state

        measured = self._sample_phases(abs_data)
        if measured is None:
            return self._state
        sig_dbm, noise_dbm, snr = measured

        if snr >= self.LOCK_ACQUIRE_SNR:
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
            return AnalyzerState.LOCKED
        self._publish(AnalysisResult.unlocked(noise_dbm))
        return self._state

    def _quick_check(self) -> AnalyzerState:
        """Cheap amplitude check at stored phases; debounces lock loss (LOCKED only)."""
        abs_data = self._capture(self._window_samples)
        if abs_data is None:
            return self._state

        measured = self._sample_phases(abs_data)
        if measured is None:
            return self._state
        sig_dbm, noise_dbm, snr = measured

        if snr < self.LOCK_LOSE_SNR:
            self._consecutive_low_snr += 1
            if self._consecutive_low_snr >= self.LOSE_LOCK_COUNT:
                self._publish(AnalysisResult.unlocked(noise_dbm))
                return AnalyzerState.SIGNAL_LOST
            # else: hold current result during debounce window — don't publish
            return self._state
        self._consecutive_low_snr = 0
        self._publish(AnalysisResult(
            signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
        ))
        return self._state

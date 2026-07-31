"""Continuous pulse-train analysis publishing signal/noise levels for real-time display.

ContinuousAnalyzer runs on a daemon thread and maintains a three-state machine:

  SEARCHING   — no valid phase pair.  Runs the full FFT fit on 1 s of audio every
                SEARCH_INTERVAL seconds until a pulse train is found.  Entered at
                startup, and again from SIGNAL_LOST once the stored phases have
                aged past PHASE_HOLD_TIMEOUT and are no longer worth extrapolating.

  LOCKED      — the 120 pps phase is known and the signal is present.  Each
                tick (~200 ms) it calls average_pulse_amplitude at the
                *predicted* peak and noise phases — O(scan_pulses) ≈ 60
                operations.  Every REFINE_INTERVAL seconds it runs
                _phase_search() to re-measure the phase and refine the drift
                estimate.  A full FFT isn't needed here: any drift too large
                for the narrow scan would already cause _quick_check() to lose
                lock first.

  SIGNAL_LOST — phase pair known but signal absent.  Four-tier re-acquisition:
                Tier 1 (200 ms): _noise_check() samples live noise at _noise_phase
                  and tries _peak_phase for instant re-acquisition.
                Tier 2 (1 s): _phase_search() scans ±PHASE_SEARCH_RADIUS samples
                  around _peak_phase using Numba amplitude averaging — ~40× cheaper
                  than an FFT, handles slow grid-frequency drift.
                Tier 3a (5 s): _fast_scan() runs a short-kernel FFT (FAST_SCAN_PULSES
                  pulses, FAST_SCAN_SAMPLES audio) as a cheap candidate detector; ~6×
                  cheaper than the full FFT, skips Tier 3b when nothing is present.
                Tier 3b (on Tier-3a hit, or every SIGNAL_LOST_REFINE as safety net):
                  _full_analysis() with the full kernel to confirm and refresh phases.

Tracking utility power drift
----------------------------
Utility power is never exactly 60 Hz, and a grid error of only 0.02 Hz walks the
pulse phase about 5 samples every second — in either direction, depending on
whether the grid is running fast or slow.  With pulses only PULSE_WIDTH_SAMPLES
wide, three samples of error is enough to sample the gap between pulses instead
of the pulses themselves.

So the analyzer does not just store where the pulse train was; it also estimates
how fast that position is moving.  Each refine appends the measured phase to a
short history, and the rate is re-fitted as the least-squares slope of phase
against time through those points (_record_phase_measurement, fit_drift_rate).
That one number then fixes two separate problems:

  * The phase goes stale between refines.  Projecting it forward on every tick
    keeps each measurement sampling the pulses rather than the gaps
    (_predicted_phases).

  * The sampling grid walks away from the pulses *within* a single analysis
    window, because the nominal spacing assumes an exact 60 Hz grid.  Spacing the
    samples at the measured rate instead fixes this
    (_effective_samples_per_pulse).  This is the larger of the two effects, and no
    amount of phase correction touches it.

Untracked, both together bias every published level downward by 5-7 dB at
ordinary drift rates.  Tracked, the residual is about a dB, which is the floor
set by phases being whole numbers of samples: the prediction is always up to half
a sample off.  Going below that would need sub-sample interpolation of the audio.

Fitting a line through several measurements, rather than dividing one prediction
error by one refine interval, is what keeps the *rate* itself away from that same
quantisation floor: the per-measurement errors are zero-mean and the fit spans a
baseline several times longer than a single interval.  Against synthetic audio at
known drift rates this holds the rate to under 0.01 samples/s where the previous
estimator sat at 0.38 — the difference between a synchronised display creeping
about a division a minute and one that genuinely stands still.

The estimate starts at zero and needs DRIFT_FIT_MIN_POINTS measurements before it
produces anything at all, so levels read low for the first couple of seconds after
a cold acquisition and reach full accuracy once the history fills.

State transitions are centralised: each tier method measures, publishes, and
returns the state it believes the machine should be in; the per-state tick
methods pass that through _transition(), which owns all transition bookkeeping
(state change, debounce-counter reset, phase validation, refine timestamp).

Results are deposited in a lock-protected slot; the Qt UI polls it on each
paint tick without blocking.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import log10

import numpy as np

from buzz.config import BuzzConfig
from buzz.dsp import (
    amplitude_to_dbm,
    analyze_window,
    average_pulse_amplitude,
    build_pulse_kernel,
    calculate_pps_fit_array,
    pulse_phase_period,
)
from buzz.sampler import RingBufferPipeline

logger = logging.getLogger(__name__)


class AnalyzerState(StrEnum):
    SEARCHING   = 'SEARCHING'
    LOCKED      = 'LOCKED'
    SIGNAL_LOST = 'SIGNAL_LOST'


class TriggerSync(StrEnum):
    """How much a display should trust the phase returned by trigger_phase().

    This is a coarser view of the state machine than AnalyzerState, because a
    synchronised display cares about a different question than the analyzer does:
    not "what am I doing" but "is the phase I am about to draw with real".

      LOCK — measured from live audio; the pulse train is present and tracked.
      HOLD — no signal right now, but a valid phase pair and drift rate survive
             from before it faded, so the phase is being extrapolated forward.
             Still worth syncing to: it is what lets a returning signal build up
             on a persistence display before the analyzer formally re-locks.
             Bounded — the analyzer abandons the phases and returns to SEARCHING
             after PHASE_HOLD_TIMEOUT, so HOLD cannot outlive its own accuracy.
      FREE — no phase pair at all: never locked, or locked once and since given up
             on.  The sweep is free-running.
    """
    LOCK = 'LOCK'
    HOLD = 'HOLD'
    FREE = 'FREE'


def fit_drift_rate(history: Sequence[tuple[float, float]],
                   min_points: int) -> float | None:
    """Least-squares slope of phase against time, in samples of phase per second.

    `min_points` is required rather than defaulted: the only correct value lives in
    ContinuousAnalyzer.DRIFT_FIT_MIN_POINTS, and a default here would silently go on
    testing the old number if that constant were ever retuned.

    `history` is (audio-clock seconds, unwrapped phase) pairs.  Under a steady grid
    error those points lie on a straight line whose slope *is* the drift rate, so
    fitting the line recovers the rate directly.

    Why a fit rather than the obvious rise-over-run between the last two points:
    phases are measured to the nearest whole sample, so every point carries up to
    half a sample of quantisation error.  Dividing one such error by one short
    refine interval puts it straight into the answer — at a 0.6 s interval that is
    0.5 / 0.6 = 0.83 samples/s of error with nothing to average against.  Fitting a
    line through many points beats that twice over: the individual errors are
    zero-mean so they partly cancel, and the outermost points span a far longer
    baseline, over which the same vertical uncertainty implies a much smaller slope
    uncertainty.  Measured against synthetic audio at known drift rates, this took
    the steady-state error from 0.38 samples/s to under 0.01 — and settled *faster*
    after a step change, so it costs no responsiveness.

    "Least squares" is the line minimising the sum of squared vertical distances to
    the points; squaring keeps errors above and below the line from cancelling and
    admits an exact closed-form answer rather than a search.  The expression below
    is that closed form: how much time and phase vary together, over how much time
    varies alone.

    Returns None when there are too few points to define a slope, or when every
    measurement landed at the same instant (a stalled audio clock), leaving the
    caller's existing estimate untouched.
    """
    n = len(history)
    if n < min_points:
        return None
    times  = np.fromiter((t for t, _ in history), dtype=np.float64, count=n)
    phases = np.fromiter((p for _, p in history), dtype=np.float64, count=n)
    centred_times = times - times.mean()
    spread = float(centred_times @ centred_times)
    if spread <= 0.0:
        return None
    return float(centred_times @ (phases - phases.mean()) / spread)


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
    # dB — SNR below which consecutive failures are counted.  Both the FFT fit and the
    # narrow scan take a max and a min over many noisy candidates, and that selection
    # alone yields ~1.2 dB of apparent SNR (measured up to 1.9 dB) on pure Gaussian
    # noise with no pulse train present.  Holding lock at 2.0 dB therefore demanded
    # almost no real signal; 3.0 keeps a genuine margin above the selection floor.
    LOCK_LOSE_SNR       = 3.0
    LOSE_LOCK_COUNT     = 3     # consecutive _quick_check failures before SIGNAL_LOST
    FAST_TICK_INTERVAL  = 0.2   # s  — tick cadence in LOCKED and SIGNAL_LOST
    SEARCH_INTERVAL     = 1.0   # s  — SEARCHING tick cadence; also the Tier-2 narrow-scan cadence in SIGNAL_LOST
    # How often to re-measure the phase and refine the drift estimate.  Between
    # refines the phase is predicted forward rather than held constant, so this no
    # longer has to outpace the drift by itself — but it still sets how quickly the
    # drift estimate can follow a changing grid, and every refine is a fresh
    # correction against accumulated prediction error.
    REFINE_INTERVAL     = 0.6   # s  — phase-search refinement interval while LOCKED
    SIGNAL_LOST_REFINE  = 120.0 # s  — unconditional full-FFT safety net in SIGNAL_LOST
    # Age, in seconds of captured audio, past which the stored phase pair is abandoned
    # and the machine falls back to SEARCHING.
    #
    # SIGNAL_LOST exists to re-acquire cheaply from a known phase, but that phase is
    # only useful while it is still close enough for the narrow scan to find:
    # _phase_search looks +/-PHASE_SEARCH_RADIUS (10 samples) around the prediction, and
    # at ordinary drift the extrapolation is already outside that window well before a
    # minute has passed.  Beyond it the cheap tiers cannot succeed by construction, so
    # holding the state costs a re-acquisition path and buys nothing.
    #
    # Returning to SEARCHING also keeps the displays honest: trigger_phase() reports
    # HOLD purely on the strength of a valid phase pair, so without this it would go on
    # claiming a synchronised sweep for as long as the signal stayed away — overnight,
    # if the arc quits at dusk.  One state machine, one answer.
    #
    # Measured against the audio clock rather than the wall clock, for the same reason
    # _predicted_phases is: a stalled sound device means no audio elapsed, therefore no
    # drift, therefore phases that are still exactly as good as when they were taken.
    PHASE_HOLD_TIMEOUT  = 60.0  # s  — SIGNAL_LOST -> SEARCHING once phases go this stale
    PHASE_SEARCH_RADIUS = 10    # samples either side of stored peak to scan in SIGNAL_LOST
    # A challenger phase must beat the incumbent's amplitude by this ratio
    # (~0.4 dB) before the stored phase moves.  Without the deadband, noise
    # flips the winner between adjacent offsets — and the noise scan, which
    # picks the QUIETEST of 21 noisy measurements, moves almost every refine —
    # so the phases and their correction indicators dance even when nothing
    # real changed.
    PHASE_MOVE_MARGIN   = 1.05
    FAST_SCAN_PULSES    = 15    # pulses in the Tier-3a screening kernel (~1/4 of full)
    FAST_SCAN_SAMPLES   = 4000  # audio window for Tier-3a (~0.25 s at 16 kHz)
    FAST_SCAN_INTERVAL  = 5.0   # s  — Tier-3a cadence in SIGNAL_LOST
    # dB — Tier-3a hit threshold; triggers Tier-3b full FFT.  This statistic is a
    # peak/trough ratio over a short fit array, so it has a large noise-only floor:
    # measured on pure Gaussian noise it averages 6.4 dB and reaches 7.6 dB.  The
    # previous 4.0 sat below that floor, so the gate fired on every call and Tier 3a
    # was pure overhead — a short FFT that never once saved the full one.  (It was
    # tuned against audio carrying a DC pedestal, which compressed the ratio; real
    # zero-mean audio never looked like that.)  8.0 clears the noise floor while
    # still admitting signals at LOCK_ACQUIRE_SNR, which measure 8.8 dB and up.
    FAST_SCAN_SNR       = 8.0
    CAPTURE_TIMEOUT     = 2.0   # s  — max wait for the pipeline to supply a window
    # EMA weight for the input DC estimate.  At the ~5 Hz capture cadence 0.02 is a
    # ~10 s time constant: slow enough to average out per-window estimation noise,
    # fast enough to follow sound-card thermal drift and receiver AGC baseline wander.
    DC_EMA_ALPHA        = 0.02

    # --- Drift tracking ----------------------------------------------------
    #
    # Utility power is never exactly 60 Hz, so the pulse train slowly walks through the
    # sample grid.  Rather than only correcting that walk after the fact, the
    # analyzer estimates how fast it is happening and projects the phase forward
    # between measurements — see _predicted_phases().
    #
    # How many recent phase measurements the rate is fitted over — see
    # fit_drift_rate().  At a 0.6 s REFINE_INTERVAL, 10 points span about 5.5 s of
    # baseline.  More points resolve the rate more finely but track a changing grid
    # more slowly; measured against synthetic audio, 5, 10 and 20 all landed within
    # 0.01 samples/s in steady state, while post-step settling went 2.4 s, 6 s and
    # 9 s respectively.  10 keeps settling at least as quick as the estimator this
    # replaced while leaving a single bad measurement outvoted nine to one.
    DRIFT_FIT_POINTS = 10
    # Fewest points that define a slope worth trusting.  Two points would fit exactly
    # and reproduce the old rise-over-run behaviour, quantisation error and all.
    DRIFT_FIT_MIN_POINTS = 3
    # Largest gap between consecutive measurements the history can be trusted across.
    # Unwrapping in _record_phase_measurement folds each step to the shortest
    # equivalent move, which is only the true move while the phase has travelled less
    # than half a period since the last measurement.  At MAX_PHASE_DRIFT_RATE that
    # takes 3.3 s, so 2.0 s leaves margin.  A longer gap — a signal outage, a stalled
    # device — means the points are no longer one continuous track, and continuing to
    # fit across it would invent a slope from an unknowable number of wraps.
    MAX_PHASE_HISTORY_GAP = 2.0       # s
    # Hard limit on the estimated rate, in samples of phase per second.  60 samples/s
    # corresponds to a grid error of about 0.22 Hz — far outside anything the power
    # system does in normal operation.  This is a safety rail: if the scan ever
    # latches onto the wrong peak, the clamp stops a bad error from compounding into
    # a runaway prediction.
    MAX_PHASE_DRIFT_RATE = 60.0       # samples/s

    def __init__(self, pipeline: RingBufferPipeline, config: BuzzConfig) -> None:
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
        # learned in one snapshot stay valid in every later one.  It doubles as the
        # modulus for phase arithmetic — the same quantity analyze_window reduces by.
        self._phase_align = pulse_phase_period(audio.sample_rate, audio.pulse_rate)

        self._state       = AnalyzerState.SEARCHING
        # Phase of the pulse train and of the quiet gap between pulses, in samples
        # from the start of a snapshot.  Kept as floats so that fractional drift
        # accumulates instead of being rounded away between refines; they are
        # rounded only at the moment they are used to index audio.
        self._peak_phase  = 0.0
        self._noise_phase = 0.0
        # How fast those phases are walking, in samples of phase per second of audio.
        # Positive means the pulses are arriving later than a perfect pulse_rate
        # train would predict, i.e. the grid is running slightly slow.  Zero until a
        # lock has been held long enough to measure it.
        self._phase_drift_rate = 0.0
        # When the phases above were last measured, on the audio clock (see
        # _audio_clock_seconds).  Predictions are made relative to this instant, so
        # it must be stamped whenever the phases are.
        self._phases_measured_at = 0.0
        # Recent (audio-clock second, unwrapped phase) pairs that the drift rate is
        # fitted through.  "Unwrapped" meaning the phase is accumulated across the
        # modulus rather than folded back into [0, _phase_align) — a line cannot be
        # fitted through a sawtooth.  Reset whenever the track is broken; see
        # MAX_PHASE_HISTORY_GAP and _transition().
        self._phase_history: deque[tuple[float, float]] = deque(maxlen=self.DRIFT_FIT_POINTS)
        self._unwrapped_phase = 0.0
        self._previous_measured_phase: int | None = None

        # EMA of the input DC offset, subtracted before rectification in _capture.
        # None until the first window seeds it, so startup converges immediately
        # instead of ramping up from zero.  Only ever touched on the analysis thread.
        self._dc: float | None = None

        # Whether _peak_phase/_noise_phase describe where the train actually is.
        # Set on the first lock and kept through SIGNAL_LOST, so the cheap
        # re-acquisition tiers can reuse the stored phases — but cleared again on any
        # return to SEARCHING, which is what PHASE_HOLD_TIMEOUT triggers once the
        # phases have aged out.  Not monotonic: it goes back to False.
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
        self._reset_requested = threading.Event()
        self._state_listeners: list[Callable[[AnalyzerState], None]] = []

        self._thread = threading.Thread(target=self._run, daemon=True, name='analyzer')

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def state(self) -> AnalyzerState:
        """The state machine's current state.

        Mostly of interest as the starting point for a listener registered before
        start() — add_state_listener() reports every change after that, but not the
        state the machine is already in.

        Not lock-protected: _state is only ever assigned on the analyzer thread, and
        rebinding an attribute is atomic, so a reader gets one state or the other and
        never a torn value.
        """
        return self._state

    def reset(self) -> None:
        """Forget everything learned about the signal and start acquiring again.

        For replaying a recording from the top: the analyzer that has just watched
        the event through knows where the pulse train is and how fast it is drifting,
        so without this the second pass opens already locked.  That is not what
        happened live, and watching the acquisition is usually the point of replaying
        an event at all.

        Split in two, because the two halves have different deadlines.  The tracker
        state — drift rate, fitted history, DC estimate, tier timers — belongs to the
        analysis thread and cannot be swapped out from under a tick already using it,
        so it is left for _apply_reset() to do when that thread next comes round.

        What the displays read cannot wait that long.  A tick can sit in
        wait_for_data for a second at a time, and a second is long enough for the
        replayed audio to refill the buffer and re-lock, so the operator would see
        the lock indicator never drop and conclude the restart did nothing.  The
        display-visible fields are therefore cleared here and now, under the lock
        trigger_phase() reads them through: the scope drops to FREE and the meters to
        silence the instant the button is pressed.
        """
        self._reset_requested.set()
        with self._result_lock:
            # A single store of an immutable value, like every other write to this
            # field; the analyzer thread setting it back to True would only mean it
            # had genuinely re-locked, and _apply_reset clears it again regardless.
            self._phases_valid = False
            self._latest_result = None
            self._result_buffer.clear()

    def add_state_listener(self, listener: Callable[[AnalyzerState], None]) -> None:
        """Call `listener(new_state)` whenever the state machine changes state.

        Lock is an event, not a level.  A consumer that polls this class instead has
        to infer the event by watching for the level to differ from last time, at
        whatever rate it happens to poll — which means a brief lock that comes and
        goes between two polls is invisible to it, and the intermittent signals this
        monitor exists to catch are exactly the ones that behave that way.

        Listeners run on the analyzer thread, inside the transition, so they must
        return promptly and must not call back into the analyzer.  Recording audio to
        disk from one would stall analysis; the event recorder's listener therefore
        only sets a flag, and its own thread does the work.

        Register before start().  The list is not synchronised, and everything is
        wired up before the analyzer thread exists.
        """
        self._state_listeners.append(listener)

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

        Leaving LOCKED drops the fitted phase history.  The signal is gone, so the
        next measurement will arrive after an unknown gap during which the phase may
        have wrapped any number of times — unwrapping across that would fabricate a
        slope.  The rate estimate itself survives; see _reset_phase_history().

        Entering SEARCHING additionally invalidates the phase pair.  SEARCHING means
        "we know nothing about where the train is" — that is true at startup and it is
        true again once PHASE_HOLD_TIMEOUT has expired, and both callers want the same
        consequences: a cold FFT search, a drift estimate reset on the next lock (see
        _full_analysis), and trigger_phase() reporting FREE rather than a HOLD it can
        no longer justify.

        Being the one place state changes happen, this is also where they are
        published to listeners (see add_state_listener) — after the bookkeeping, so a
        listener that reads the analyzer sees a consistent view.
        """
        if new_state == self._state:
            return
        self._consecutive_low_snr = 0
        if new_state == AnalyzerState.LOCKED:
            self._phases_valid = True
            self._last_refine  = time.monotonic()
        else:
            self._reset_phase_history()
            if new_state == AnalyzerState.SEARCHING:
                self._phases_valid = False
        self._state = new_state
        self._publish_state(new_state)

    def _publish_state(self, state: AnalyzerState) -> None:
        """Notify listeners of a state change, isolating each from the others.

        A listener raising must not abort the transition or kill the analyzer
        thread: analysis is the primary job, and something as peripheral as a
        recorder or a display losing one notification is not worth stopping it for.
        """
        for listener in self._state_listeners:
            try:
                listener(state)
            except Exception:
                logger.exception('Analyzer state listener failed — continuing.')

    def _apply_reset(self) -> None:
        """Return to the state a freshly-built analyzer is in (analysis thread only).

        Everything measured from the audio goes; everything derived from the config
        stays, since the kernels and geometry describe the signal being looked for
        rather than anything learned about it.
        """
        self._reset_requested.clear()
        self._peak_phase = self._noise_phase = 0.0
        self._phase_drift_rate = 0.0
        self._phases_measured_at = 0.0
        self._unwrapped_phase = 0.0
        self._reset_phase_history()
        self._dc = None
        self._phases_valid = False
        self._consecutive_low_snr = 0
        self._last_refine = self._last_narrow_scan = 0.0
        self._last_full_fft = self._last_fast_scan = 0.0
        with self._result_lock:
            self._latest_result = None
            self._result_buffer.clear()
            self._latest_signal_correction = self._latest_noise_correction = 0
        # Straight to SEARCHING, telling listeners only if that is a change: the reset
        # clears what the machine knows either way, but a machine that was already
        # searching has not changed state and has nothing to announce.
        previous, self._state = self._state, AnalyzerState.SEARCHING
        if previous != AnalyzerState.SEARCHING:
            self._publish_state(AnalyzerState.SEARCHING)
        logger.info('Analyzer reset — acquiring from scratch.')

    def _run(self) -> None:  # pragma: no cover
        # Guards against a transient failure (a numerical edge case, a hiccup from
        # the audio device) silently killing this daemon thread.  Without this, an
        # uncaught exception here stops all analysis for the rest of the process
        # with no indication anything is wrong — the GUI keeps showing the last
        # published result frozen forever, and the collector keeps writing minutes
        # of stale-looking data.  Mirrors Collector.collection_loop()'s same
        # catch-log-retry approach to the same class of failure.
        while not self._stop.is_set():
            try:
                if self._reset_requested.is_set():
                    self._apply_reset()
                if self._state == AnalyzerState.LOCKED:
                    interval = self._locked_tick()
                elif self._state == AnalyzerState.SIGNAL_LOST:
                    interval = self._signal_lost_tick()
                else:
                    interval = self._searching_tick()
            except Exception:
                logger.exception(
                    'Analyzer tick failed — likely a transient audio-device or '
                    'numerical error; retrying rather than stopping analysis.'
                )
                interval = self.FAST_TICK_INTERVAL
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
        # Give up on the stored phases once they are too stale for the cheap tiers to
        # use, and drop back to a cold search.  See PHASE_HOLD_TIMEOUT.
        if (self._state != AnalyzerState.LOCKED
                and self._audio_clock_seconds() - self._phases_measured_at
                > self.PHASE_HOLD_TIMEOUT):
            self._transition(AnalyzerState.SEARCHING)
        return self.FAST_TICK_INTERVAL

    # ------------------------------------------------------------ tier methods
    #
    # Each measures one window, publishes any result, and returns the state the
    # machine should be in.  None of them mutates _state directly — that is
    # _transition()'s job.

    # ------------------------------------------------------------ drift tracking

    def _effective_samples_per_pulse(self) -> float:
        """Actual spacing between consecutive pulses, corrected for measured drift.

        The nominal spacing (sample_rate / pulse_rate) assumes the grid sits exactly
        at pulse_rate.  When it does not, the sampling grid does not merely start in
        the wrong place — it walks away from the real pulses *within* a single
        analysis window.  At 0.05 pps of error the two diverge by 6.7 samples over
        one second, which with 3-sample pulses costs about 6 dB.

        That loss is invisible to phase tracking, because it is a spacing error
        rather than an offset: no matter where the window starts, pulses at one end
        of it line up only if pulses at the other end do not.  Correcting it needs
        the drift rate, which is exactly what the tracker measures.

        A drift of `rate` samples per second is spread across pulse_rate pulses in
        that second, so each successive pulse sits rate / pulse_rate samples further
        along than the nominal spacing predicts.
        """
        return self._samples_per_pulse + self._phase_drift_rate / self._pulse_rate

    def _audio_clock_seconds(self) -> float:
        """Seconds of audio captured since the stream started.

        Drift is measured against the audio timeline, not the wall clock.  The two
        agree while capture is healthy, but if the sound device stalls, wall time
        keeps running while no audio arrives — and the pulse train cannot drift
        through samples that were never recorded.  Predicting against wall time
        would then extrapolate straight past the real position.  Counting captured
        samples instead makes a stalled stream simply freeze the prediction, which
        is the correct behaviour.
        """
        return self._pipeline.total_samples / self._sample_rate

    def _shortest_phase_move(self, difference: float) -> float:
        """Fold a raw difference between two phases into the shortest equivalent move.

        Phases wrap around at _phase_align, so going from 395 to 5 is a move of +10
        samples forward, not -390 backwards.  Taking the raw subtraction would make
        every wrap look like an enormous jump and would throw the drift estimate
        badly off, so differences are always folded into +/- half a period.
        """
        half_period = self._phase_align / 2
        return (difference + half_period) % self._phase_align - half_period

    def _predicted_phases(self, at_time: float | None = None) -> tuple[int, int]:
        """Where the signal and noise phases should be at `at_time` (default: now).

        This is the point of tracking a drift rate at all.  The pulse train walks
        through the sample grid at _phase_drift_rate samples per second, so a phase
        measured even half a second ago is already stale — and with pulses only
        PULSE_WIDTH_SAMPLES wide, being three samples late means sampling the gap
        between pulses instead of the pulses, which reads as pure noise.  Projecting
        the last measurement forward keeps every tick sampling the right place
        instead of drifting off between refines.

        Both phases are shifted by the same amount because both are anchored to the
        same power cycle: the noise phase marks a gap *between* pulses, so when the
        pulses walk the gap walks with them.  (They are still searched separately in
        _phase_search, because which gap is quietest can change for reasons that have
        nothing to do with drift — see that method's docstring.)

        `at_time` is a reading of the audio clock; it defaults to right now.  Drift
        can be either sign — a grid running fast walks the phase one way, a grid
        running slow walks it the other — so `shift` is signed, and the modulo below
        wraps correctly for negative values because Python's % always returns a
        non-negative result.
        """
        now = self._audio_clock_seconds() if at_time is None else at_time
        return (self._project_phase(self._peak_phase, self._phase_drift_rate,
                                    self._phases_measured_at, now),
                self._project_phase(self._noise_phase, self._phase_drift_rate,
                                    self._phases_measured_at, now))

    def _project_phase(self, phase: float, drift_rate: float,
                       measured_at: float, at_time: float) -> int:
        """Walk one phase forward from when it was measured to `at_time`.

        Split out from _predicted_phases so trigger_phase() can apply the identical
        projection to a snapshot of the state it took under the lock, rather than
        re-deriving the arithmetic and risking the two drifting apart.
        """
        return int(round(phase + drift_rate * (at_time - measured_at))) % self._phase_align

    def _reset_phase_history(self) -> None:
        """Drop the fitted history, keeping the current rate estimate.

        Called when the measurements stop being one continuous track — a signal
        outage, a cold start, a stalled device.  The *rate* deliberately survives:
        the grid went on drifting while the signal was gone, so the last estimate is
        a far better starting guess than zero (cold start zeroes it separately, in
        _full_analysis, because there is no history there to be right about).
        """
        self._phase_history.clear()
        self._previous_measured_phase = None

    def _record_phase_measurement(self, peak_phase: int, noise_phase: int,
                                  measured_at: float) -> None:
        """Store a freshly measured phase pair and re-fit the drift rate through it.

        The rate is the least-squares slope of the recent (time, phase) history — see
        fit_drift_rate() for what that buys over the rise-over-run this replaced.

        Two things have to be true for a line fit to mean anything, and both are
        handled here rather than in the fit:

        * The phases must be *unwrapped*.  Measurements come back folded into
          [0, _phase_align), so a steadily drifting train produces a sawtooth, and
          the slope of a sawtooth is not the drift rate.  Each step is folded to its
          shortest equivalent move and accumulated instead, turning the sawtooth back
          into the straight line it physically is.

        * The points must be one continuous track.  Unwrapping infers the move from
          the shortest path, which is only right while less than half a period has
          passed; across a gap the phase may have wrapped an unknowable number of
          times.  A gap beyond MAX_PHASE_HISTORY_GAP therefore restarts the history
          rather than fitting a fabricated slope through it.

        `measured_at` is a reading of the audio clock, not the wall clock.
        """
        if (self._previous_measured_phase is None
                or measured_at - self._phases_measured_at > self.MAX_PHASE_HISTORY_GAP):
            self._reset_phase_history()
            self._unwrapped_phase = float(peak_phase)
        else:
            self._unwrapped_phase += self._shortest_phase_move(
                peak_phase - self._previous_measured_phase)
        self._previous_measured_phase = peak_phase
        self._phase_history.append((measured_at, self._unwrapped_phase))

        fitted = fit_drift_rate(self._phase_history, self.DRIFT_FIT_MIN_POINTS)
        # Locked as a group.  These are read together by trigger_phase() on the Qt
        # thread, which projects the phase forward using the interval since
        # _phases_measured_at — so a read landing between the phase write and the
        # timestamp write would extrapolate a fresh phase across a stale interval and
        # put the trigger in the wrong place.  Writing them atomically costs nothing
        # here (a few times a second) and removes the tear entirely.
        with self._result_lock:
            if fitted is not None:
                # The clamp is a safety rail, not part of the estimate: if the scan
                # ever latches onto the wrong peak, it stops one bad fit compounding
                # into a runaway prediction.
                self._phase_drift_rate = max(-self.MAX_PHASE_DRIFT_RATE,
                                             min(self.MAX_PHASE_DRIFT_RATE, fitted))
            self._peak_phase = float(peak_phase)
            self._noise_phase = float(noise_phase)
            self._phases_measured_at = measured_at

    def phase_drift_rate(self) -> float:
        """Current estimate of the pulse-train drift, in samples of phase per second.

        Locked because this is written on the analyzer thread and read from
        others (the collector, when logging grid frequency to the CSV).
        """
        with self._result_lock:
            return self._phase_drift_rate

    def grid_frequency_hz(self) -> float:
        """Utility line frequency implied by the measured drift, in Hz.

        Note the direction.  A *positive* drift means each pulse turns up later than
        the configured rate predicts, so the true pulse rate is *lower* than
        configured and the grid is running slow.  Frequency therefore moves opposite
        to the drift.

        Writing the true rate as r, matching the sampling grid to the actual pulses
        requires drift = sample_rate * (pulse_rate - r) / r, which rearranges to
        r = sample_rate * pulse_rate / (sample_rate + drift).  Arcing happens twice
        per power cycle — once per half-cycle, at each voltage peak — so halving
        that converts pulses per second into cycles per second.
        """
        with self._result_lock:
            drift_rate = self._phase_drift_rate
        true_pulse_rate = self._sample_rate * self._pulse_rate / (self._sample_rate + drift_rate)
        return true_pulse_rate / 2

    def trigger_phase(self) -> tuple[int, TriggerSync]:
        """Where the pulse train is right now, for a display to synchronise its sweep to.

        Returns (phase in samples within pulse_phase_period, how much to trust it).
        Safe to call from the Qt thread: the four pieces of tracker state are read as
        one atomic group and projected forward outside the lock, so the analyzer
        thread is never blocked for longer than four attribute reads.

        The phase is the *predicted* one, not the last measured one — the same
        projection the analyzer uses for its own sampling (see _predicted_phases).
        A display syncing to a stale measurement would show the trace creeping across
        the screen at the grid's drift rate, which is precisely what this view exists
        to cancel out.

        FREE returns phase 0 rather than a meaningless number.  Note that this still
        yields a *stationary* trace rather than a sliding one, because the caller reads
        phase-aligned snapshots: with no lock the sweep simply starts at an arbitrary
        point in the pulse period instead of a chosen one.  For a display that is the
        right failure mode — an un-synced trace that holds still is readable, whereas
        one scrolling at an arbitrary rate is not, and the returned TriggerSync is what
        tells the operator not to trust its horizontal position.
        """
        with self._result_lock:
            # _state and _phases_valid are written on the analyzer thread outside this
            # lock, but both are single stores of an immutable value, so the worst case
            # is reading the previous one — never a torn one.  Taking them here anyway
            # keeps the whole snapshot to one acquisition.
            state       = self._state
            valid       = self._phases_valid
            peak        = self._peak_phase
            drift_rate  = self._phase_drift_rate
            measured_at = self._phases_measured_at
        if not valid:
            return 0, TriggerSync.FREE
        phase = self._project_phase(peak, drift_rate, measured_at, self._audio_clock_seconds())
        return phase, (TriggerSync.LOCK if state == AnalyzerState.LOCKED else TriggerSync.HOLD)

    # ------------------------------------------------------------ tier helpers

    def _publish(self, result: AnalysisResult) -> None:
        with self._result_lock:
            self._latest_result = result
            self._result_buffer.append(result)

    def _to_dbm(self, amplitude: float) -> float:
        return amplitude_to_dbm(amplitude, self._offset_db)

    def _capture(self, n_samples: int) -> np.ndarray | None:
        """Wait for n_samples of phase-aligned audio and return it DC-corrected and rectified.

        Snapshots are aligned to _phase_align so every window starts at the same
        offset within the pulse period.  Without this the window origin moves with
        the ring buffer tail between ticks, and phases stored from one snapshot
        point at the wrong samples in the next — silently breaking _quick_check,
        _noise_check, and _phase_search.

        The DC offset is removed before rectification.  The receiver is in LSB, so its
        audio is a frequency-translated slice of RF: bipolar and zero-mean by
        construction, and any constant is the sound card's own offset.  abs(x + d)
        adds that offset to the pulse positions and the inter-pulse positions alike,
        which compresses the ratio between them and costs SNR — worst at the weak-signal
        end.

        The estimator is the median, not the mean.  The mean is not robust to the very
        thing being measured: a 120 pps train of 20000-count pulses pulls the window
        mean to ~450 counts when the true offset is zero, so subtracting it would
        inject an error that grows with signal strength.  The pulses occupy ~2% of
        samples, which the median ignores entirely.  It costs ~270 us per window
        against ~1185 us for one FFT fit.

        The estimate is additionally EMA-smoothed (see DC_EMA_ALPHA) rather than taken
        fresh per window, so per-window estimation noise isn't injected into every
        reading while genuine drift is still followed.

        Returns None when the pipeline cannot supply the data within
        CAPTURE_TIMEOUT — the caller should leave the state machine unchanged.
        """
        if not self._pipeline.wait_for_data(n_samples + self._phase_align,
                                            timeout=self.CAPTURE_TIMEOUT):
            return None
        snapshot = self._pipeline.get_snapshot(n_samples, align=self._phase_align).astype(np.float32)
        window_dc = float(np.median(snapshot))
        self._dc = (window_dc if self._dc is None
                    else self._dc + self.DC_EMA_ALPHA * (window_dc - self._dc))
        return np.abs(snapshot - self._dc)

    def _sample_phases(self, abs_data: np.ndarray,
                       peak_phase: int | None = None,
                       noise_phase: int | None = None) -> tuple[float, float, float] | None:
        """Sample signal and noise amplitudes at the given phases.

        With no phases supplied it uses the *predicted* ones — where drift says the
        pulse train should be right now — rather than wherever it was last measured.
        That is what stops the reading sagging between refines; see
        _predicted_phases().

        Returns (sig_dbm, noise_dbm, snr) or None if the buffer is too short.
        """
        predicted_peak, predicted_noise = self._predicted_phases()
        peak     = predicted_peak if peak_phase is None else peak_phase
        noise    = predicted_noise if noise_phase is None else noise_phase
        spacing  = self._effective_samples_per_pulse()
        # Start after the larger of the two phase offsets so both pulse trains
        # fit entirely inside the window and are averaged over the same pulses.
        start    = max(peak, noise)
        n_pulses = int((len(abs_data) - start) // spacing)
        if n_pulses < 1:
            return None
        sig_amp   = float(average_pulse_amplitude(abs_data, spacing, n_pulses, peak))
        noise_amp = float(average_pulse_amplitude(abs_data, spacing, n_pulses, noise))
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
            # Only reachable on a digitally silent input, where every fit score is
            # exactly zero and the dB ratio is undefined.  Treat any positive peak
            # as a candidate rather than dividing by zero.
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
            if not self._phases_valid:
                # Cold start: no history, so there is no drift estimate to keep.
                # (After a signal outage the previous estimate IS kept — the grid
                # went on drifting while our signal was gone, and the old rate is a
                # better starting guess than zero.)  Locked for the same reason as
                # the write in _record_phase_measurement().
                with self._result_lock:
                    self._phase_drift_rate = 0.0
            # A full FFT search is an absolute re-measurement, not a tracking step:
            # it can land anywhere in the period, including on a different peak than
            # the history was following.  Unwrapping that against the previous point
            # would read the jump as drift, so the track restarts here.
            self._reset_phase_history()
            self._record_phase_measurement(window.peak_phase, window.noise_phase,
                                           measured_at=self._audio_clock_seconds())
            self._publish(AnalysisResult(
                signal_dbm=sig_dbm, noise_dbm=noise_dbm, snr=snr, locked=True,
            ))
            return AnalyzerState.LOCKED
        if not self._phases_valid:
            # SEARCHING: no stored phases to fall back on — publish what the FFT found
            self._publish(AnalysisResult.unlocked(noise_dbm))
        # else: SIGNAL_LOST with valid phases — _noise_check() handles publishing
        return self._state

    def _is_new_best(self, amplitude: float, best_amp: float | None, minimize: bool) -> bool:
        """Whether `amplitude` beats the best candidate found so far in a
        _scan_phase pass: the quietest one when hunting for the noise gap, the
        loudest one when hunting for the signal.
        """
        if best_amp is None:
            return True
        return amplitude < best_amp if minimize else amplitude > best_amp

    def _challenger_beats_incumbent(self, best_amp: float, incumbent_amp: float,
                                    minimize: bool) -> bool:
        """Whether the scan's winning candidate cleared PHASE_MOVE_MARGIN over the
        incumbent (offset 0) — see _scan_phase's docstring for why this deadband
        exists.
        """
        if minimize:
            return best_amp * self.PHASE_MOVE_MARGIN < incumbent_amp
        return best_amp > incumbent_amp * self.PHASE_MOVE_MARGIN

    def _scan_phase(self, abs_data: np.ndarray, center: int, anchor: int,
                    minimize: bool) -> tuple[int, int]:
        """Scan ± PHASE_SEARCH_RADIUS around center for the best pulse amplitude.

        anchor is the other pulse train's phase; the analysis window starts at
        max(candidate, anchor) so both trains fit within the same audio.  minimize
        selects the quietest candidate (noise scan) instead of the loudest (signal
        scan).

        The incumbent (offset 0) keeps its place unless a challenger beats it by
        PHASE_MOVE_MARGIN — genuine drift shows large amplitude differences (one
        sample of offset already costs ~1/3 of a pulse's overlap), so the deadband
        only suppresses noise-driven flicker, not real movement.

        Returns (best_phase, best_offset); center with offset 0 if no candidate
        fits or no challenger clears the margin.
        """
        best_amp      = None
        best_phase    = center
        best_offset   = 0
        incumbent_amp = None
        spacing       = self._effective_samples_per_pulse()
        for offset in range(-self.PHASE_SEARCH_RADIUS, self.PHASE_SEARCH_RADIUS + 1):
            # Wrap on the exact repeat period, not samples_per_pulse: offsets one
            # period apart sample identical positions, so this is the only modulus
            # under which a wrapped candidate really is the same phase.
            candidate = (center + offset) % self._phase_align
            start     = max(candidate, anchor)
            n_pulses  = int((len(abs_data) - start) // spacing)
            if n_pulses < 1:
                continue
            amp = float(average_pulse_amplitude(abs_data, spacing, n_pulses, candidate))
            if offset == 0:
                incumbent_amp = amp
            if self._is_new_best(amp, best_amp, minimize):
                best_amp    = amp
                best_phase  = candidate
                best_offset = offset
        if best_amp is None:
            return center, 0
        if (best_offset != 0 and incumbent_amp is not None
                and not self._challenger_beats_incumbent(best_amp, incumbent_amp, minimize)):
            return center, 0
        return best_phase, best_offset

    def _scan_signal_and_noise_phases(self, abs_data: np.ndarray, predicted_peak: int,
                                      predicted_noise: int) -> tuple[int, int]:
        """Scan for the current signal and noise phases around their predictions,
        recording each scan's winning offset for the UI's correction indicators.
        Returns (signal_phase, noise_phase).  See _phase_search's docstring for why
        the two are searched independently rather than shifted together.
        """
        signal_phase, signal_offset = self._scan_phase(
            abs_data, predicted_peak, anchor=predicted_noise, minimize=False)
        with self._result_lock:
            self._latest_signal_correction = signal_offset

        noise_phase, noise_offset = self._scan_phase(
            abs_data, predicted_noise, anchor=signal_phase, minimize=True)
        with self._result_lock:
            self._latest_noise_correction = noise_offset

        return signal_phase, noise_phase

    def _phase_search_snr_threshold(self) -> float:
        """SNR bar for accepting a _phase_search measurement: LOCK_ACQUIRE_SNR when
        re-acquiring from SIGNAL_LOST, LOCK_LOSE_SNR while already LOCKED.  See
        _phase_search's docstring for why these differ.
        """
        return self.LOCK_LOSE_SNR if self._state == AnalyzerState.LOCKED else self.LOCK_ACQUIRE_SNR

    def _phase_search(self) -> AnalyzerState:
        """Narrow amplitude scan ± PHASE_SEARCH_RADIUS around each predicted phase.

        This is the analyzer's tracking step, and it does two jobs: it finds where
        the pulse train actually is now, and it uses the gap between that and where
        drift predicted it would be to refine the drift estimate itself.  Scanning
        around the *prediction* rather than the last measurement is what keeps the
        search centred — under steady drift the pulse has already moved by the time
        we look again, so a search centred on the old position starts off-target and
        has to spend its radius catching up.

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
        track together.  Note that these are offsets from the predicted phase, so once
        drift is being tracked well they read near zero even while the grid is drifting
        hard — they show the tracker's residual error, not the raw movement.

        The SNR bar for updating the phases depends on how we got here.  Acquiring
        from SIGNAL_LOST demands LOCK_ACQUIRE_SNR — strong evidence before trusting
        a new phase pair.  While already LOCKED, tracking only requires
        LOCK_LOSE_SNR, the same bar _quick_check() holds the lock at; requiring
        the acquisition bar here would create a dead zone (LOCK_LOSE_SNR ≤ snr <
        LOCK_ACQUIRE_SNR) where the lock is held but drift can't be followed,
        guaranteeing eventual loss on weak signals.

        On success both phases are updated and LOCKED is returned; otherwise the
        current state.  Does not publish on failure; _noise_check() already
        published the noise result on this tick.
        """
        abs_data = self._capture(self._window_samples)
        if abs_data is None:
            return self._state

        # Stamp the audio clock after the capture, so it refers to the audio we
        # actually just analysed rather than to whatever had arrived before we waited.
        measured_at = self._audio_clock_seconds()
        predicted_peak, predicted_noise = self._predicted_phases(measured_at)
        signal_phase, noise_phase = self._scan_signal_and_noise_phases(
            abs_data, predicted_peak, predicted_noise)

        # Re-measure both winners over a consistent window before the SNR decision.
        measured = self._sample_phases(abs_data, signal_phase, noise_phase)
        if measured is None:
            return self._state
        sig_dbm, noise_dbm, snr = measured

        if snr >= self._phase_search_snr_threshold():
            # Commit the measurement, and let how far it landed from the prediction
            # sharpen the drift estimate for next time.
            self._record_phase_measurement(
                signal_phase, noise_phase, measured_at=measured_at)
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

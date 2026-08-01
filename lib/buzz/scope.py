"""
Phase-synchronised oscilloscope display for the powerline pulse train.

ScopeWidget draws a repeating sweep of raw audio with the pulse train pinned to
a fixed horizontal position, so a 120 pps arc renders as a standing wave rather
than something sliding across the screen.  A classic CRT phosphor persistence
model gives each sweep a decaying afterglow.

Triggering
----------
This is not a threshold-crossing trigger like a bench scope's.  ContinuousAnalyzer
already tracks where the pulse train sits (_peak_phase) and how fast that position
is walking through the sample grid (_phase_drift_rate), so the trigger is a *phase
lookup* rather than a search for an edge.  That is strictly better for this signal:
it cannot be fooled by a single loud noise spike, and it holds through fades
because it is driven by the drift tracker rather than by any individual pulse.

The sweep shows _SWEEP_PULSES pulse periods - 3 pulse periods, 400 samples, 25 ms
at the defaults - fixed at that pulse count so the timebase reflects the signal at
any sample rate rather than a gcd of it (see sweep_geometry).  The starting offset
is still wrapped to pulse_phase_period(), the modulus the analyzer reduces its own
phases by, which is what makes the mapping from phase to screen position exact
regardless of how that modulus relates to the sweep's own width - the two coincide
at the defaults but not in general; sweep_geometry's docstring has the case where
they differ.

The sweep is shorter than the frame interval, so each frame overlays several real
sweeps (4 at 25 ms per sweep and _UPDATE_MS = 100).  That is both the authentic
multi-trace look of an analogue scope and the reason no audio goes unrendered -
the same argument _mean_spectrum_db makes for averaging every FFT frame per
waterfall row rather than rendering only the newest one.

Persistence carries real information here.  The bright core of the trace is where
successive sweeps agree; the dim halo around it is where they disagree, so the
halo's vertical width is a direct readout of pulse-to-pulse amplitude scatter and
its horizontal width is a readout of timing jitter.

ScopeWidget carries `# pragma: no cover` for the same reason the widgets in
waterfall.py do: it needs a live Qt display.  The sweep extraction, auto-ranging,
rasterisation and colour maths below it are plain functions with no Qt dependency
and are unit tested in test_scope_math.py.
"""

from dataclasses import dataclass
from math import ceil, log10

import numpy as np
from numba import njit
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from buzz.analyzer import ContinuousAnalyzer, TriggerSync
from buzz.config import BuzzConfig
from buzz.constants import FULL_SCALE_COUNTS
from buzz.dsp import pulse_phase_period
from buzz.fonts import display_font
from buzz.sampler import RingBufferPipeline

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Height of the status bar above the trace.  Deliberately equal to waterfall.py's
# _AXIS_H so the two panels' header bars read as one design rather than two; the
# constant is duplicated rather than imported because waterfall.py imports *this*
# module (MainWindow composes the scope), and importing back would be circular.
# test_scope_math.py asserts the two stay equal.
_HEADER_H = 24
# Trace area height.  Divides evenly by _V_DIVISIONS, giving exactly 12 px per
# division so the graticule sits on whole pixels rather than being rounded.
_TRACE_H = 96
# Public because MainWindow (in waterfall.py) sizes the window around it.
SCOPE_H = _HEADER_H + _TRACE_H           # 120 px, matching the waterfall panel

_H_DIVISIONS = 10                        # 2.5 ms/div across a 25 ms sweep
_V_DIVISIONS = 8                         # centre line at division 4

# How far in from the left edge the triggering pulse sits, in divisions.
#
# The trigger is not the pulse's onset, it is its *peak*: peak_phase comes from the
# argmax of average_pulse_amplitude, which scores a PULSE_WIDTH_SAMPLES-wide window,
# so it sits where the pulse is brightest.  The pulse itself begins before that and
# persists after it - the arc reaches us through the receiver's audio passband, which
# rings, spreading a sharp impulse over milliseconds on both sides of its peak.  Give
# the sweep too little lead-in and that leading flank simply falls off the left edge,
# which looks like a pulse jammed against the frame.  1.5 divisions is 60 samples
# (3.75 ms at 16 kHz) of room ahead of the peak.
#
# The half-division is deliberate too: a whole number of divisions would put the
# peak exactly on a graticule line, hiding the leading flank under it and reading as
# pinned to the grid rather than placed on it.  x.5 sits midway between two lines.
#
# Sized against the real burst rather than by eye.  Observed directly on this display:
# the arc is not an impulse but a symmetric broadband burst 2.5-6 ms wide (40-96
# samples), so at its widest it extends ~48 samples = 77 px either side of the peak.
# 1.5 divisions puts the first burst's leading edge 19 px inside the left rail and
# still leaves 40 px past the third burst's tail.  Going to 2.0 divisions would trade
# that for only 8 px at the right, and past ~2.1 the third burst wraps off screen -
# so this is near the middle of the usable range, not an arbitrary choice.
#
# tools/pulse_probe.py can quantify the burst, but note its MAX_PULSE_WIDTH = 20
# ceiling is far below what is actually there; raise it before trusting a width from
# it.  (Its flags are --seconds and --quiet only.)
_PRETRIGGER_DIVISIONS = 1.5

_UPDATE_MS = 100                         # matches the waterfall's frame cadence

# What the trace shows and how long the phosphor remembers, both counted in *pulse
# periods* rather than in samples or milliseconds.
#
# The pulse period is the only length here with any physical meaning: it is the thing
# being looked at.  Counting in it is what makes the display say the same thing at any
# sample rate and at either grid frequency - three cycles across the screen, twenty-four
# cycles of persistence, whether the audio arrives at 8 kHz or 48, and whether the grid
# is 60 Hz (120 pps) or 50 Hz (100 pps).
#
# Time per division then follows the grid, which is the honest answer rather than a
# compromise: 2.50 ms/div at 120 pps and 3.00 ms/div at 100 pps, because a 50 Hz grid
# truly has a longer period.  Forcing both to 2.50 would show 2.5 cycles at 100 pps
# - the same picture stretched, saying less about the waveform.
#
# 24 is chosen so the sweep count comes out whole everywhere: every phase period met in
# practice is 1, 2, 3, 4 or 8 pulse periods, and 24 divides by all of them.  See
# sweep_geometry for what happens at a rate where it does not.
_SWEEP_PULSES = 3
_PHOSPHOR_PULSES = 24

# ---------------------------------------------------------------------------
# Phosphor
# ---------------------------------------------------------------------------

# Intensity deposited by one sweep.  Below 1.0 so that a single sweep renders as
# mid-scale on the colour ramp and only agreement between several sweeps drives the
# trace to its hot white-green core - which is what makes the core/halo distinction
# carry information instead of everything saturating on the first sweep.
_SWEEP_INTENSITY = 0.55
# Fraction of the previous frame's phosphor retained each _UPDATE_MS.  0.72 per
# 100 ms frame decays a trace to ~5% over nine frames, a little under a second -
# long enough to see jitter accumulate as a visible halo, short enough that the
# display still tracks a changing signal rather than smearing its whole history.
_PHOSPHOR_DECAY = 0.72

# EMA weight for averaging mode, applied once per frame to the mean of that frame's
# sweeps.  Two stages of averaging stack: the frame's own sweeps are meaned first
# (4 of them at _UPDATE_MS = 100 and a 25 ms sweep), then folded into the EMA.
#
# For an EMA of weight a over inputs of variance s^2, the steady-state variance is
# s^2 * a/(2-a); pre-averaging k sweeps divides the input variance by k.  So the
# scatter shrinks by sqrt(a/(2-a)/k) = sqrt(0.05/1.95/4) = 0.080, which is 21.9 dB.
# Measured against Gaussian noise: single-sweep |x| scatter 0.6029, averaged 0.0483,
# ratio 21.9 dB - theory and measurement agree.  That is what makes pulse shape
# visible at low SNR.
#
# Kept as an EMA rather than a fixed-N block average so the display keeps tracking a
# changing signal instead of freezing once its bucket fills.
_AVERAGE_ALPHA = 0.05

# Colour ramp for the phosphor, as (intensity, (R, G, B)) stops.  The blue-green
# cast of a P31-type CRT: a near-black face with a faint teal bias, through the
# saturated cyan-green of a normally-driven trace, to a white-hot core where many
# sweeps overlap.
_PHOSPHOR_STOPS = (
    (0.00, (  4,  14,  16)),
    (0.25, (  0,  70,  72)),
    (0.55, (  0, 190, 165)),
    (0.85, (120, 255, 205)),
    (1.00, (225, 255, 240)),
)

# Graticule intensities, on the same 0–1 scale as the phosphor, so the etched
# grid glows like part of the CRT face and any real trace outranks it.
_GRATICULE_LINE = 0.10
_GRATICULE_AXIS = 0.17                   # centre horizontal/vertical rules

# ---------------------------------------------------------------------------
# Auto-ranging vertical scale
# ---------------------------------------------------------------------------
#
# The same problem the waterfall's colour scale has, on the vertical axis: a fixed
# volts-per-division calibrated once against the station's gain goes stale as soon
# as the receiver, band, or sound-card setup drifts, and then the trace is either a
# flat line or permanently slammed into the top and bottom rails.  So the deflection
# is read off the signal itself, continuously.

# The pulses are the point of the display, and they occupy only ~2% of samples (3
# samples out of every 133), so the statistic has to reach well into the upper tail
# to see them at all - a p90 would describe the noise between pulses and scale the
# pulses clean off the screen.  p99.5 sits above the pulse population while still
# being a percentile rather than a bare max, so one sample of static crash can't
# rescale the whole display on its own.
_RANGE_PERCENTILE = 99.5
# Full scale is set above the measured percentile so that routine peaks sit around
# 3 of the 4 available divisions, leaving the top division for a transient truly
# louder than anything recent.  Same intent as the waterfall's _COLOR_HEADROOM.
_RANGE_HEADROOM = 1.30
# EMA weight per frame, matching the waterfall's _COLOR_RANGE_EMA_ALPHA and chosen
# for the same reason: without it a brief burst re-scales the picture within a few
# hundred milliseconds and the trace visibly breathes.  0.05 at 100 ms frames gives
# a settling time of a couple of seconds.
_RANGE_EMA_ALPHA = 0.05
# Smallest full-scale deflection allowed, in raw int16 counts.  This is the vertical
# analogue of _MIN_DYNAMIC_RANGE_DB, and exists for the same failure mode: with a
# truly silent input the percentile collapses toward zero and the auto-range
# would stretch quantisation dither across the entire screen, painting a dead
# channel as a healthy full-amplitude noise trace.  32 counts is about -60 dBFS.
_MIN_FULL_SCALE = 32.0
# Initial guess, used only until the EMA has real data to converge from.
_INITIAL_FULL_SCALE = 2048.0

# Raw sample magnitude at which the input is treated as clipping.  Slightly inside
# the int16 rail, since a converter usually flattens before it reaches the last code.
_CLIP_COUNTS = 32000

# ---------------------------------------------------------------------------
# Status bar colours, by trigger sync state
# ---------------------------------------------------------------------------

_SYNC_STYLE = {
    TriggerSync.LOCK: ('◆ LOCK', (110, 255, 190)),   # live phase measurement
    TriggerSync.HOLD: ('◇ HOLD', (255, 190,  70)),   # extrapolating through a fade
    TriggerSync.FREE: ('○ FREE', (120, 140, 145)),   # nothing to sync to
}


# ---------------------------------------------------------------------------
# Sweep extraction
# ---------------------------------------------------------------------------

def sweep_start_offset(trigger_phase: int, pretrigger: int, sweep_samples: int) -> int:
    """Index within a phase-aligned snapshot where the displayed sweep begins.

    Backing off by `pretrigger` puts the triggering pulse one division in from the
    left edge rather than jammed against it, so its leading edge is visible - the
    same reason a bench scope offers pre-trigger.

    `sweep_samples` names the parameter for the common case, where it is the sweep's
    own width.  ScopeWidget instead passes the phase period: trigger_phase is only
    meaningful modulo that period, since it is what the analyzer itself reduces its
    phases by, so wrapping by anything else would not address the same position in
    the pulse grid.  The two happen to be equal at the default sample rate, but not
    in general - see sweep_geometry's docstring for when the phase period is shorter
    than the sweep.
    """
    return int(trigger_phase - pretrigger) % sweep_samples


@dataclass(frozen=True)
class SweepGeometry:
    """How the trace is cut out of the audio, for one sample rate and grid frequency.

    A pure value, so the arithmetic can be checked exhaustively without a display.
    Getting it wrong does not crash anything: it quietly shows a different number of
    pulses, or superimposes sweeps that were sampled at different sub-sample offsets
    and blurs the trace, neither of which announces itself.
    """

    sample_rate: int
    pulse_rate: int
    # The pulse grid's repeat period.  Sweeps must start a whole number of these apart
    # or they sample the grid at different offsets -- see stride_samples.
    phase_period: int
    sweep_samples: int      # what is drawn: _SWEEP_PULSES pulse periods
    stride_samples: int     # distance between successive sweeps
    n_sweeps: int           # how many are accumulated into the phosphor
    capture_samples: int    # audio to pull per frame to supply them
    pretrigger: int
    ms_per_division: float

    @property
    def phosphor_seconds(self) -> float:
        return self.n_sweeps * self.stride_samples / self.sample_rate


def sweep_geometry(sample_rate: int, pulse_rate: int) -> SweepGeometry:
    """Work out the trace geometry for a sample rate and grid frequency.

    Everything is counted in pulse periods and then converted, which is what keeps the
    display saying the same thing everywhere: _SWEEP_PULSES cycles across the screen and
    _PHOSPHOR_PULSES cycles of persistence, at any rate and either grid frequency.

    The one thing that cannot be chosen freely is the *stride*.  Successive sweeps have
    to begin a whole phase period apart, because that is the only interval after which
    the pulse grid repeats against the sample clock; step by anything else and each
    sweep samples the pulses at a slightly different sub-sample offset, so they no
    longer superimpose and the trace smears.  The phase period is often shorter than
    the sweep (one pulse period at 48 kHz / 120 pps), so the stride is the fewest whole
    phase periods that reach the sweep width -- which means consecutive sweeps can
    overlap in the audio, exactly as a real scope re-triggers on a running signal.
    """
    pulse_period = sample_rate / pulse_rate
    phase_period = pulse_phase_period(sample_rate, pulse_rate)
    # Always a whole number: the phase period is by construction a whole number of
    # pulse periods.
    pulses_per_phase = max(1, round(phase_period / pulse_period))

    stride_phases = ceil(_SWEEP_PULSES / pulses_per_phase)
    stride_samples = stride_phases * phase_period
    stride_pulses = stride_phases * pulses_per_phase
    sweep_samples = round(_SWEEP_PULSES * pulse_period)

    # At every rate and grid frequency met in practice this divides exactly.  It will
    # not at a rate whose phase period is a large number of pulse periods -- 8001 Hz
    # repeats only every 40 -- and there the phosphor simply holds fewer, longer
    # sweeps rather than misbehaving.
    n_sweeps = max(1, round(_PHOSPHOR_PULSES / stride_pulses))
    return SweepGeometry(
        sample_rate=sample_rate,
        pulse_rate=pulse_rate,
        phase_period=phase_period,
        sweep_samples=sweep_samples,
        stride_samples=stride_samples,
        n_sweeps=n_sweeps,
        # A whole phase period of slack at the head, because the trigger offset can
        # push the first sweep that far in before any of it is drawn.
        capture_samples=phase_period + (n_sweeps - 1) * stride_samples + sweep_samples,
        pretrigger=round(sweep_samples * _PRETRIGGER_DIVISIONS / _H_DIVISIONS),
        ms_per_division=sweep_samples / sample_rate * 1000 / _H_DIVISIONS,
    )


def n_complete_sweeps(n_samples: int, start: int, sweep_samples: int,
                      stride: int | None = None) -> int:
    """How many whole sweeps fit in a snapshot after skipping to `start`.

    Sweeps begin every `stride` samples and are `sweep_samples` long; the two differ
    once the trace shows a fixed number of pulse periods while sweeps step by a whole
    phase period.  With stride == sweep_samples this tiles the capture, which is what
    it did when they were necessarily equal.

    Partial sweeps are dropped rather than padded: a half-drawn trace running off
    into a flat line would read as signal that isn't there.
    """
    stride = sweep_samples if stride is None else stride
    if sweep_samples <= 0 or stride <= 0 or start >= n_samples:
        return 0
    available = n_samples - start
    if available < sweep_samples:
        return 0
    return (available - sweep_samples) // stride + 1


def extract_sweeps(samples: np.ndarray, start: int, sweep_samples: int,
                   stride: int | None = None, limit: int | None = None) -> np.ndarray:
    """Stack up to `limit` sweeps of `sweep_samples`, each `stride` apart from `start`.

    `stride` defaults to `sweep_samples`, which tiles the capture with no gaps and is
    what the display did when the two were necessarily equal.  They are separate now:
    the trace shows a fixed number of pulse periods, while successive sweeps must be a
    whole phase period apart or they sample the pulse grid at different sub-sample
    offsets and the traces no longer superimpose.  See sweep_geometry.

    Returns an empty (0, sweep_samples) array when not even one whole sweep fits,
    which every consumer below treats as "nothing to draw this frame".
    """
    stride = sweep_samples if stride is None else stride
    n = n_complete_sweeps(len(samples), start, sweep_samples, stride)
    if limit is not None:
        n = min(n, limit)
    if n < 1:
        return np.empty((0, sweep_samples), dtype=samples.dtype)
    windows = np.lib.stride_tricks.sliding_window_view(samples[start:], sweep_samples)
    return windows[::stride][:n]


# ---------------------------------------------------------------------------
# Auto-ranging
# ---------------------------------------------------------------------------

def auto_range_full_scale(sweeps: np.ndarray, previous: float) -> float:
    """Blend this frame's measured deflection into the smoothed full-scale value.

    "Full scale" is the sample magnitude that reaches the top (or bottom) rail of
    the trace area, i.e. half the screen height.  Returns `previous` unchanged when
    there is nothing to measure, so a stalled frame holds the current scale rather
    than collapsing it to the minimum.

    See _RANGE_PERCENTILE, _RANGE_HEADROOM, _RANGE_EMA_ALPHA and _MIN_FULL_SCALE
    for why each of the four terms is here.
    """
    if sweeps.size == 0:
        return previous
    raw = float(np.percentile(np.abs(sweeps), _RANGE_PERCENTILE)) * _RANGE_HEADROOM
    blended = previous + _RANGE_EMA_ALPHA * (raw - previous)
    return max(blended, _MIN_FULL_SCALE)


def full_scale_dbfs(full_scale: float) -> float:
    """Full-scale deflection in dB relative to digital clipping, for the status bar.

    Reported as the deflection at the top of the graticule rather than as a
    per-division figure, because the trace is linear: divisions are evenly spaced in
    amplitude, not in dB, so a "dB/div" number would not describe anything.  Read
    this as headroom - at -24 dBFS the top of the screen is 24 dB below clipping.

    Deliberately dBFS rather than the dBm the meters and the CSV speak.
    amplitude_to_dbm() converts a *mean-absolute* amplitude, whereas this scale is
    driven by a p99.5 peak (see _RANGE_PERCENTILE).  Pushing a peak through that
    conversion would print a number several dB above what the S-meters show for the
    very same signal, and two readouts on one window that appear to disagree about
    level are worse than no readout at all.

    Goes positive when the auto-range is chasing a signal that is already clipping,
    which is information worth showing rather than clamping away.

    Caller supplies a value from auto_range_full_scale(), which is floored at
    _MIN_FULL_SCALE and so is always strictly positive.
    """
    return 20.0 * log10(full_scale / FULL_SCALE_COUNTS)


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def resample_to_columns(sweep: np.ndarray, n_columns: int) -> np.ndarray:
    """Linearly interpolate a sweep onto n_columns evenly spaced positions.

    The sweep carries fewer samples than the display has columns (400 across 640 px
    at the defaults), so this interpolates up rather than decimating down.  Linear
    interpolation is the right choice for a scope: it is exactly the straight line
    an analogue beam would trace between two sampled points, and unlike a smoothing
    interpolator it cannot invent overshoot that the signal does not contain.
    """
    return np.interp(np.linspace(0, len(sweep) - 1, n_columns),
                     np.arange(len(sweep)), sweep)


def trace_rows(values: np.ndarray, full_scale: float, height: int,
               bipolar: bool = True) -> np.ndarray:
    """Map sample values to row indices, 0 = top of the trace area.

    Bipolar values are centred on the middle row and deflect either way, which is
    how the raw audio is drawn.  Unipolar values (a rectified envelope) run from the
    bottom edge upward instead, because a rectified signal has a hard floor at zero
    and centring it would waste the lower half of the screen on territory it can
    never reach.

    Values beyond full scale are clipped to the rails rather than wrapped, matching
    what a real instrument does when it is over-driven.
    """
    if bipolar:
        half = (height - 1) / 2.0
        rows = half - (values / full_scale) * half
    else:
        rows = (height - 1) * (1.0 - values / full_scale)
    return np.clip(np.rint(rows), 0, height - 1).astype(np.intp)


# Eagerly compiled at import; see the note on average_pulse_amplitude in dsp.py for
# why, and for what boundscheck and cache are doing.
@njit('void(float32[:,:], int64[:], float64)', boundscheck=True, cache=True)
def accumulate_trace(phosphor: np.ndarray, rows: np.ndarray, intensity: float) -> None:
    """Add one connected trace to the phosphor buffer, in place.

    `rows` must hold one more entry than the buffer is wide: each adjacent pair is a
    segment, so N+1 row positions yield N segments and every column gets drawn.
    boundscheck turns a violation of that contract into an IndexError rather than a
    silent misdraw or a crash.

    Each column is filled between its segment's two endpoints rather than having a
    single pixel set.  Without that the trace breaks into disconnected dots wherever
    the signal moves faster than one row per column - which for a noisy pulse is
    most of the screen.

    JIT-compiled, because this is the display's hot path: it runs once per sweep,
    about fifty times a second.  The obvious vectorised alternative - mark the span
    ends in a difference array and cumulative-sum down each column - is what this
    replaced, and it was 40x slower.  Not because the arithmetic was worse, but
    because it allocated a (height+1, width) float32 temporary every call (a quarter
    of a megabyte, fifty times a second) and then integrated all of it, whether a
    given cell needed touching or not.  Measured on a 96x640 buffer with five sweeps
    per frame: 2088 us per frame vectorised, 50 us here, output identical to within
    float32 rounding.  That is 1.8% of a core given back, continuously, on a monitor
    designed to run for months at a time.

    Writing the spans directly is also simply the honest expression of the operation.
    The difference-and-integrate trick only existed to express a loop as array ops.
    """
    width = phosphor.shape[1]
    for x in range(width):
        low = rows[x]
        high = rows[x + 1]
        if low > high:
            low, high = high, low
        for y in range(low, high + 1):
            phosphor[y, x] += intensity


def build_graticule(height: int, width: int) -> np.ndarray:
    """Etched-grid intensity mask, on the same 0-1 scale as the phosphor.

    Composited under the trace with a maximum rather than drawn over it, so the grid
    sits on the CRT face and any real signal outranks it - a graticule painted on
    top of the trace would look like an overlay instead of glass.
    """
    grid = np.zeros((height, width), dtype=np.float32)
    for i in range(1, _H_DIVISIONS):
        grid[:, round(i * width / _H_DIVISIONS)] = _GRATICULE_LINE
    for i in range(1, _V_DIVISIONS):
        grid[round(i * height / _V_DIVISIONS), :] = _GRATICULE_LINE
    # Centre rules brighter.  Both division counts are even, so these sit exactly on
    # an existing grid line and overwrite it rather than adding a neighbouring one.
    grid[:, round(width / 2)] = _GRATICULE_AXIS
    grid[round(height / 2), :] = _GRATICULE_AXIS
    return grid


def build_phosphor_colormap() -> np.ndarray:
    """Return a 256x3 uint8 RGB lookup table interpolated through _PHOSPHOR_STOPS."""
    t = np.linspace(0.0, 1.0, 256)
    positions = np.array([p for p, _ in _PHOSPHOR_STOPS], dtype=np.float64)
    lut = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        values = np.array([c[channel] for _, c in _PHOSPHOR_STOPS], dtype=np.float64)
        lut[:, channel] = np.rint(np.interp(t, positions, values)).astype(np.uint8)
    return lut


_COLORMAP = build_phosphor_colormap()


def update_running_average(average: np.ndarray | None, sweeps: np.ndarray,
                           alpha: float) -> np.ndarray | None:
    """Fold rectified sweeps into an exponentially-weighted mean, for averaging mode.

    Averaging is done on the *rectified* sweep, never on the raw bipolar waveform.
    The arc is not phase-locked to the receiver's BFO, so the audio-frequency carrier
    inside each burst arrives with a random phase every sweep and averaging the raw
    trace would cancel the pulses toward zero - erasing exactly the thing being
    measured.  The envelope has no such phase, so it adds coherently and the noise
    falls as 1/sqrt(N) around it.

    Returns None (leaving the caller's state untouched) when there is nothing to fold in.
    """
    if sweeps.size == 0:
        return average
    rectified = np.abs(sweeps).mean(axis=0)
    if average is None:
        return rectified
    return average + alpha * (rectified - average)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class ScopeWidget(QWidget):  # pragma: no cover -- requires a live Qt display
    """Phase-synchronised sweep of the pulse train, with CRT phosphor persistence.

    Two modes, toggled by toggle_mode():

      RAW - bipolar waveform with persistence.  The default, and what the display
            is for: it shows the signal as it actually arrives, noise and all.
      AVG - rectified envelope, coherently averaged.  Trades the live view for
            sensitivity, resolving pulse shape that a single sweep buries in noise.
    """

    def __init__(self, pipeline: RingBufferPipeline, analyzer: ContinuousAnalyzer,
                 config: BuzzConfig, width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._analyzer = analyzer
        self._sample_rate = config.audio.sample_rate
        # The trace shows a fixed number of pulse periods at any sample rate, while
        # sweeps step by a whole phase period so they superimpose exactly.  The two
        # were necessarily the same length before, which made the sweep width -- and
        # so the time base -- follow a gcd rather than the signal.
        self._geometry = sweep_geometry(config.audio.sample_rate, config.audio.pulse_rate)
        self._sweep_samples = self._geometry.sweep_samples
        self._pretrigger = self._geometry.pretrigger
        self._capture_samples = self._geometry.capture_samples
        self._ms_per_division = self._geometry.ms_per_division

        self._width = width
        self._phosphor = np.zeros((_TRACE_H, width), dtype=np.float32)
        self._graticule = build_graticule(_TRACE_H, width)
        self._average: np.ndarray | None = None

        self._full_scale = _INITIAL_FULL_SCALE
        self._average_full_scale = _INITIAL_FULL_SCALE
        self._sync = TriggerSync.FREE
        self._averaging = False
        self._clipping = False
        self._last_total_samples = 0

        self.setFixedSize(width, SCOPE_H)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_UPDATE_MS)

    def toggle_mode(self) -> None:
        """Switch between raw-persistence and rectified-average views.

        Both accumulators are reset so the incoming mode starts clean rather than
        showing a stale picture built under the other mode's scaling.
        """
        self._averaging = not self._averaging
        self._phosphor[:] = 0.0
        self._average = None

    def _tick(self) -> None:
        # A stalled stream freezes the display rather than redrawing the same audio
        # as though it were new - same guard, and same reasoning, as WaterfallWidget.
        total = self._pipeline.total_samples
        if total == self._last_total_samples:
            return
        self._last_total_samples = total

        raw = self._pipeline.get_snapshot(
            self._capture_samples,
            # Aligned on the phase period, which is the interval the analyzer's phases
            # are reduced by and the only one after which the grid repeats exactly.
            align=self._geometry.phase_period).astype(np.float32)
        # The sound card's DC offset would sit the whole trace off the centre line and,
        # in averaging mode, add a constant pedestal to the rectified envelope.  The
        # median rather than the mean, because the pulses themselves would drag a mean
        # (see ContinuousAnalyzer._capture).  No EMA smoothing is needed at this window
        # size - 2400 samples is a stable estimate on its own, and unlike the analyzer
        # this value is never used for a measurement anyone reports.
        samples = raw - float(np.median(raw))
        self._clipping = bool(np.abs(raw).max() >= _CLIP_COUNTS)

        phase, self._sync = self._analyzer.trigger_phase()
        start = sweep_start_offset(phase, self._pretrigger, self._geometry.phase_period)
        sweeps = extract_sweeps(samples, start, self._sweep_samples,
                                self._geometry.stride_samples, self._geometry.n_sweeps)
        if sweeps.size == 0:
            return

        if self._averaging:
            self._average = update_running_average(self._average, sweeps, _AVERAGE_ALPHA)
            self._average_full_scale = auto_range_full_scale(self._average,
                                                             self._average_full_scale)
        else:
            self._full_scale = auto_range_full_scale(sweeps, self._full_scale)
            self._phosphor *= _PHOSPHOR_DECAY
            for sweep in sweeps:
                self._draw(self._phosphor, sweep, self._full_scale,
                           _SWEEP_INTENSITY, bipolar=True)
            np.clip(self._phosphor, 0.0, 1.0, out=self._phosphor)
        self.update()

    def _draw(self, buffer: np.ndarray, values: np.ndarray, full_scale: float,
              intensity: float, bipolar: bool) -> None:
        """Rasterise one trace into `buffer`.  One extra column of row positions is
        requested so that N+1 positions give N segments - see accumulate_trace."""
        columns = resample_to_columns(values, self._width + 1)
        rows = trace_rows(columns, full_scale, _TRACE_H, bipolar=bipolar)
        accumulate_trace(buffer, rows, intensity)

    def _intensity_field(self) -> np.ndarray:
        """The 0-1 intensity image to colour-map, graticule composited underneath."""
        if self._averaging:
            field = np.zeros((_TRACE_H, self._width), dtype=np.float32)
            if self._average is not None:
                self._draw(field, self._average, self._average_full_scale,
                           1.0, bipolar=False)
        else:
            field = self._phosphor
        return np.clip(np.maximum(field, self._graticule), 0.0, 1.0)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        self._paint_header(painter)

        index = (self._intensity_field() * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(_COLORMAP[index])
        image = QImage(rgb.data, self._width, _TRACE_H,
                       self._width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, _HEADER_H, image)

    def _paint_header(self, painter: QPainter) -> None:
        painter.fillRect(0, 0, self._width, _HEADER_H, QColor(30, 30, 30))
        painter.setFont(display_font(8, bold=True))

        label, (r, g, b) = _SYNC_STYLE[self._sync]
        painter.setPen(QColor(r, g, b))
        painter.drawText(6, 0, 70, _HEADER_H,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

        painter.setFont(display_font(8))
        painter.setPen(QColor(190, 190, 190))
        # Grid frequency is only meaningful once a phase has been measured; with no
        # lock the drift rate is still at its startup zero and would read as a
        # suspiciously exact 60.00 Hz.
        if self._sync is not TriggerSync.FREE:
            painter.drawText(80, 0, 90, _HEADER_H,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f'{self._analyzer.grid_frequency_hz():.2f} Hz')

        scale = self._average_full_scale if self._averaging else self._full_scale
        mode = 'AVG' if self._averaging else 'RAW'
        painter.drawText(0, 0, self._width - 6, _HEADER_H,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f'{mode}   {self._ms_per_division:.2f} ms/div   '
                         f'FS {full_scale_dbfs(scale):+.1f} dBFS')

        if self._clipping:
            painter.setPen(QColor(255, 80, 80))
            painter.drawText(180, 0, 50, _HEADER_H,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 'CLIP')

    def stop(self) -> None:
        self._timer.stop()

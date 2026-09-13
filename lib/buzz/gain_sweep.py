"""Choose a tuner gain by measuring the band, not by asking the operator to guess.

The receiver has one knob that matters and two ways to get it wrong.  Too little gain
and the noise floor is the converter listening to itself, so the reported floor is too
high and every SNR is compressed.  Too much and an arc clips.

Clipping is the worse failure.  It is nonlinear, so it does not merely under-read the
arc: it puts products across the whole span and lifts the apparent floor in the same
capture, leaving both numbers wrong with nothing in the data to say so.  A
converter-limited floor is wrong by a bounded amount in a known direction.  So this
treats headroom as a hard limit and takes whatever floor accuracy is left underneath.

Neither measurement may depend on a signal being present, because nobody can promise
an arc is running when an operator opens the tool.  Both of the ones used here exist
on a dead band:

  * The quiet level is a low percentile of the per-frame RMS, so bursts are stepped
    over and what remains is the level between them.
  * The antenna's share of the floor comes from the shape of the whole sweep rather
    than from any one reading, by separating the part that grows with gain from the
    part that does not.

See docs-notebook/sdr-gain-calibration.md for the measurements these rules came from.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from buzz.sdr import IqBlock

logger = logging.getLogger(__name__)

# Told the sweep how far it has got: the step just starting, how many there are, and
# the gain about to be measured.
ProgressCallback = Callable[[int, int, float], None]


class SweepSource(Protocol):
    """The part of RtlSdrSource a sweep uses.

    Declared rather than imported so that the whole sweep runs against a stand-in with
    no receiver attached, the same reason RtlSdrDevice exists one layer down.
    """

    @property
    def supported_gains_db(self) -> list[float]:
        ...

    @property
    def iq_sample_rate(self) -> int:
        ...

    @property
    def blocks_to_discard_after_gain_change(self) -> int:
        ...

    def set_gain(self, gain_db: float) -> float:
        ...

    def read(self, timeout: float = 1.0) -> IqBlock | None:
        ...

# Percentile of per-frame RMS taken as the quiet level.
#
# Low enough to sit under an arc that fires on a fraction of frames, high enough not
# to ride the single quietest frame in the capture.  A 120 pps train with 6 ms bursts
# occupies about 72% of a 1 ms frame grid at worst, so a quarter is not safe; a tenth
# is.  The value is chosen rather than measured, and the sweep's own interleaving is
# what defends against an arc that runs for a whole pass.
_QUIET_PERCENTILE = 10.0

# Samples per frame for that percentile.
#
# Short against a burst, so a burst spoils few frames, and long enough that the RMS of
# one frame is a stable number: 1024 samples is 4 ms at 256 kHz and averages hundreds
# of independent noise samples.
_FRAME_SAMPLES = 1024

# The share of the noise floor that has to come from the antenna before the gain
# counts as usable.
#
# One half is the knee itself rather than a threshold near it.  The share at gain g is
# A*10^(g/10) / (A*10^(g/10) + C), so asking for one half is asking for
# A*10^(g/10) >= C: the point where the antenna's noise power equals the converter's,
# which is the bend KneeFit exists to find.  The rule this constant expresses is
# therefore "sit at the knee", and any other value is an offset past it: 0.8 is the
# antenna at four times the converter, six dB up, and 0.9 is nine times, nine and a
# half dB up.
#
# Below the knee the station is mostly measuring its own receiver, which is the case
# the mag loop in the notebook showed: 5.0 dB below converter noise at every gain the
# tuner offered.
#
# Sitting exactly on the knee would cost 3.01 dB of floor accuracy, since half the
# reported power would be the converter's.  In practice it costs less, because the
# tuner's steps are coarse and the chosen one sits above the knee rather than on it.
# Against the station's broadband antenna, whose shape the notebook records, the knee
# falls at 30.1 dB and the lowest step at or above it is 32.8 dB, which delivers a
# share of 0.65 and an error of 1.87 dB.  The whole range of bars, on that antenna:
#
#     bar    gain picked    reported floor reads high by
#     0.5       32.8 dB               1.87 dB
#     0.7       33.8 dB               1.54 dB
#     0.8       36.4 dB               0.92 dB
#     0.9       40.2 dB               0.41 dB
#
# A higher bar buys floor accuracy with headroom, and 0.8 would reproduce the 36.4 dB
# that station settled on by hand.  The knee is deliberate anyway: clipping is the
# failure that cannot be recovered from, so the spare 3.6 dB above the reserve is
# worth more than the decibel of floor accuracy it costs, and a lower bar also finds
# an answer on quieter antennas where a higher one finds none.
_ANTENNA_SHARE_FLOOR = 0.5


@dataclass(frozen=True)
class GainMeasurement:
    """What one gain step looked like, combined across every pass of the sweep.

    `quiet_dbfs` is the median across passes, because it decides where the antenna
    takes over and a transient is noise against that question.  `peak_dbfs` and
    `clipped` are the maximum and the total, because the worst case is the only
    interesting one: a median peak would size the headroom for a quiet moment.
    """

    gain_db: float
    quiet_dbfs: float
    peak_dbfs: float
    clipped: int
    passes: int


@dataclass(frozen=True)
class SweepResult:
    """The chosen gain, and enough of the working to argue with it.

    `chosen_db` is None when no gain satisfies both bounds, which happens on an
    antenna too quiet to swamp the converter at any setting the tuner offers.  That is
    a real answer about the station rather than a failure of the sweep, so `reason`
    carries it in words and the measurements are still here to look at.
    """

    chosen_db: float | None
    reason: str
    antenna_share: float
    lowest_usable_db: float | None
    highest_safe_db: float | None
    measurements: tuple[GainMeasurement, ...]

    @property
    def floor_error_db(self) -> float:
        """How much high the reported noise floor reads at the chosen gain.

        The converter's own noise adds to the antenna's, and the sum is what gets
        reported, so this is the whole point of the dominance bound.  An antenna share
        of one half is 3.01 dB, nine tenths is 0.46 dB.
        """
        if self.antenna_share <= 0.0:
            return float('inf')
        return float(-10.0 * np.log10(self.antenna_share))


class BandMeasurement:
    """Turns blocks of complex samples into the two figures a sweep compares.

    Separate from the sweep because it needs no hardware, so the arithmetic can be
    checked against arrays whose answer is known by construction.
    """

    @staticmethod
    def quiet_dbfs(samples: np.ndarray) -> float:
        """The level between bursts, in dB relative to a full-scale sine.

        Takes the RMS of each frame and then a low percentile across frames.  The
        percentile is what steps over an arc: a burst lifts the frames it falls in and
        leaves the rest alone, so the low end of the distribution still describes the
        quiet band underneath it.  A plain mean over the whole capture would fold the
        burst back in, which is the thing this has to avoid.

        The DC offset is removed first, because the monitor never hears it.  A
        receiver puts a strong false signal at exactly its own tuning frequency, which
        is why the station tunes tuning_offset_khz away from what it listens to, and
        the one-sided filter then rejects the spike entirely.  Measuring the raw IQ
        with it still in there sizes the gain against an artifact nothing downstream
        sees: an offset of 0.04 read 7.07 dB high, and worst at low gain, where the
        true noise is smallest and the knee fit most needs it.

        The mean is the right DC estimate here, unlike in the rectified audio the
        analyzer works on.  These samples are complex baseband, where both the noise
        and an arc are zero-mean, so nothing but the offset survives the average.
        ContinuousAnalyzer._capture takes a median instead because its input has been
        rectified, and there a 120 pps train pulls the mean well away from zero.
        """
        usable = len(samples) // _FRAME_SAMPLES * _FRAME_SAMPLES
        if usable == 0:
            return float('-inf')
        frames = samples[:usable].reshape(-1, _FRAME_SAMPLES)
        frames = frames - np.mean(samples[:usable])
        # abs() before the mean, so this is power per frame rather than the mean of a
        # complex number, which for noise is approximately zero whatever its level.
        frame_rms = np.sqrt(np.mean(np.abs(frames) ** 2, axis=1))
        quiet = float(np.percentile(frame_rms, _QUIET_PERCENTILE))
        return BandMeasurement._to_dbfs(quiet)

    @staticmethod
    def peak_dbfs(samples: np.ndarray) -> float:
        """The loudest single sample, in the same units as quiet_dbfs.

        The DC offset stays in, where quiet_dbfs takes it out, and the two differ
        because they answer different questions.  Clipping happens at the converter,
        before any filtering, so an offset does use up headroom and belongs in the
        figure that decides whether an arc has room.  The noise floor is what
        the monitor reports after filtering, and the spike never reaches that.
        """
        if len(samples) == 0:
            return float('-inf')
        return BandMeasurement._to_dbfs(float(np.max(np.abs(samples))))

    @staticmethod
    def _to_dbfs(magnitude: float) -> float:
        """Magnitude to dB, with zero mapped to negative infinity rather than raised.

        Full scale is a magnitude of 1, which is where as_complex() puts a byte at the
        converter's rail, so 0 dBFS is the point at which a sample clips.
        """
        return 20.0 * np.log10(magnitude) if magnitude > 0 else float('-inf')


class KneeFit:
    """Separates the part of the noise floor that grows with gain from the part that
    does not.

    The tuner's gain is applied before the converter digitizes, so the converter's own
    noise is added afterwards and does not grow with the knob.  Measured power at a
    gain g is therefore

        P(g) = A * 10^(g/10) + C

    where A is what the antenna delivers at unity gain and C is the converter's floor.
    Plotted in dB against dB that is flat at low gain, where C dominates, and rises
    with slope 1 at high gain, where A does.  The bend between the two is the knee,
    and where it falls is what says whether this station is listening to the band or
    to its own receiver.

    The useful part is that the expression is linear in A and C even though it is
    curved in g, so the two come out of an ordinary least-squares solve against the
    basis [10^(g/10), 1] with no iteration and nothing to converge.  A reader can
    check it by generating a curve from a known A and C and getting them back, which
    is what TestTheKneeFitRecoversWhatItWasGiven does.
    """

    def __init__(self, gains_db: np.ndarray, powers: np.ndarray) -> None:
        """Fit against linear powers, never dB, since only the linear ones add.

        Weighted by 1/P, which is not a refinement but the difference between an
        answer and a wrong one.  Powers across a sweep span five orders of magnitude,
        so an unweighted solve is dominated by the top of the range: the residual at
        the highest gain is larger than C itself, and C is left almost unconstrained.
        Measured on a synthetic curve with 5% noise, an unweighted fit recovered C 76
        times too large while getting A right, and C is the term the dominance
        decision rests on.

        Dividing both the basis and the target by P makes each residual relative, so a
        5% error costs the same at either end.  It stays one linear solve.
        """
        gains = np.asarray(gains_db, dtype=float)
        measured = np.asarray(powers, dtype=float)
        basis = np.column_stack([10.0 ** (gains / 10.0), np.ones(len(gains))])
        # Guard the division rather than the input: a gain step that read as silence
        # would otherwise take the whole fit with it.
        weights = np.where(measured > 0.0, 1.0 / np.where(measured > 0.0, measured, 1.0), 0.0)
        solution, *_ = np.linalg.lstsq(basis * weights[:, None], measured * weights,
                                       rcond=None)
        # A negative coefficient means noise beat the model rather than that the
        # antenna or the converter delivers negative power.  Clamping keeps every
        # number downstream physical; the share it produces then reads as 0 or 1,
        # which is the honest summary of a fit that could not separate the two.
        self.antenna_at_unity = max(float(solution[0]), 0.0)
        self.converter = max(float(solution[1]), 0.0)

    def antenna_share(self, gain_db: float) -> float:
        """What fraction of the floor at this gain comes from the antenna.

        One means the converter contributes nothing measurable, zero means the
        reading is entirely the receiver listening to itself.
        """
        antenna = self.antenna_at_unity * 10.0 ** (gain_db / 10.0)
        total = antenna + self.converter
        if total <= 0.0:
            return 0.0
        return antenna / total

    def lowest_gain_where_the_antenna_dominates(self, gains_db: list[float]) -> float | None:
        """The smallest offered gain whose antenna share clears the bar, or None.

        Lowest rather than highest on purpose.  Every dB of gain above what the
        antenna needs is a dB of headroom an arc no longer has, so the cheapest gain
        that still measures the band is the one to want.

        The candidates are the gains the tuner actually reported, so the answer is one
        of them by construction.  Nothing computes the knee as a continuous number and
        rounds it: the fitted share is evaluated at each real step and the first one
        that clears the bar is returned, which is the next step at or above the knee.
        A rounded knee could name a gain the hardware does not have.
        """
        for gain in sorted(gains_db):
            if self.antenna_share(gain) >= _ANTENNA_SHARE_FLOOR:
                return gain
        return None


class GainChooser:
    """Turns a set of measurements into one gain, or into a reason there is not one.

    Two bounds, from opposite directions:

      * The **dominance bound** is the lowest gain at which the antenna, rather than
        the converter, is most of what the floor is made of.  Below it the station
        measures its own receiver.
      * The **headroom bound** is the highest gain at which the quiet level still
        leaves `headroom_db` before a sample reaches the rail.  Above it an arc clips.

    The answer is the dominance bound, checked against the headroom bound.  Lowest
    rather than highest, because every dB above what the antenna needs is a dB an arc
    no longer has.

    They can cross.  An antenna quiet enough to need most of the tuner's range to beat
    the converter may need more gain than the headroom allows, and then there is no
    number that satisfies both.  Saying so is the right output: "your antenna is quiet
    enough that the noise floor will be partly mine" is something an operator can act
    on, where a number picked from a rule that failed is not.
    """

    def __init__(self, measurements: tuple[GainMeasurement, ...], headroom_db: float) -> None:
        self._measurements = measurements
        self._headroom_db = headroom_db

    def choose(self) -> SweepResult:
        """Pick a gain, or explain why the two bounds leave nothing."""
        if not self._measurements:
            return SweepResult(None, 'The sweep measured no gains at all.',
                               0.0, None, None, ())
        gains = [m.gain_db for m in self._measurements]
        fit = KneeFit(np.array(gains),
                      np.array([self._power_of(m) for m in self._measurements]))
        lowest_usable = fit.lowest_gain_where_the_antenna_dominates(gains)
        highest_safe = self._highest_gain_with_headroom()

        # Headroom first, because it is measured directly where dominance comes from
        # a fit.  A band loud enough to clip at every gain also gives the fit nothing
        # to separate, so checking dominance first would report a puzzled fit instead
        # of the concrete thing an operator can act on.
        if highest_safe is None:
            return SweepResult(
                None,
                f'Even the lowest gain leaves less than {self._headroom_db:.0f} dB '
                'before clipping, so the band is loud enough that an arc would clip '
                'whatever this is set to.  An attenuator ahead of the receiver is the '
                'fix.',
                fit.antenna_share(min(gains)), lowest_usable, None, self._measurements)
        if lowest_usable is None:
            return SweepResult(
                None,
                'No gain this tuner offers makes the antenna louder than the receiver '
                'itself, so the noise floor would be mostly the receiver at any '
                'setting.  A larger or better matched antenna is the only fix.',
                fit.antenna_share(max(gains)), None, highest_safe, self._measurements)
        if lowest_usable > highest_safe:
            return SweepResult(
                None,
                f'The antenna needs {lowest_usable:.1f} dB before it beats the '
                f'receiver, and clipping allows at most {highest_safe:.1f} dB.  The '
                'two do not overlap, so this station has to accept either a noise '
                'floor that is partly the receiver or an arc that clips.',
                fit.antenna_share(lowest_usable), lowest_usable, highest_safe,
                self._measurements)

        share = fit.antenna_share(lowest_usable)
        return SweepResult(
            lowest_usable,
            f'{lowest_usable:.1f} dB is the lowest gain where the antenna is most of '
            f'the noise floor, and it leaves {self._headroom_db:.0f} dB for an arc.',
            share, lowest_usable, highest_safe, self._measurements)

    def _highest_gain_with_headroom(self) -> float | None:
        """The largest gain whose quiet level still clears the reserve.

        Measured from the level *between* bursts rather than from an observed peak,
        which is what lets this run on a dead band: sizing from a peak needs an arc to
        be present and nothing arranges that.  The reserve stands in for the arc that
        has not arrived.
        """
        safe = [m.gain_db for m in self._measurements
                if m.quiet_dbfs + self._headroom_db <= 0.0]
        return max(safe) if safe else None

    @staticmethod
    def _power_of(measurement: GainMeasurement) -> float:
        """The quiet level as linear power, which is the domain the knee fit adds in."""
        return float(10.0 ** (measurement.quiet_dbfs / 10.0))


class GainSweep:
    """Measures every gain the tuner offers, several times, and picks one.

    The receiver is injected rather than opened here, so the whole class runs against
    a stand-in with no hardware attached.

    **Why it takes several passes rather than one.**  A first attempt swept each gain
    once in sequence and produced neighboring steps disagreeing by 16 dB, which is
    more than the gain between them.  The receiver was not at fault:

        at a fixed 32.8 dB, 20 consecutive seconds:  floor spread 0.9 dB, sd 0.2 dB
        at the same 32.8 dB a few minutes earlier:   floor 7.4 dB higher, 357 clipped

    The floor is steady second to second and unsteady minute to minute, because the
    arc comes and goes.  A sweep takes about a minute, so a single pass measures every
    step in a different world.  Several passes alternating direction put each gain at
    a different point in that drift, and combining across them recovers the shape.

    Five passes at a quarter second per step took about 75 seconds and gave a
    monotonic curve where one pass had given noise.
    """

    # Odd on purpose; see the constructor.  Five rather than three because the figure
    # came from hardware, where five passes at a quarter second per step gave a
    # monotonic curve that one pass had not.
    DEFAULT_PASSES = 5
    DEFAULT_SECONDS_PER_STEP = 0.25

    def __init__(self, source: SweepSource, headroom_db: float, *,
                 passes: int = DEFAULT_PASSES,
                 seconds_per_step: float = DEFAULT_SECONDS_PER_STEP) -> None:
        self._source = source
        self._headroom_db = headroom_db
        # Rounded up to an odd number, because the floor is combined with a median and
        # numpy's median of an even count averages the two middle values rather than
        # picking one.  That is exactly the outlier rejection the median is here for,
        # so an even count quietly gives up the thing the passes were added to buy.
        #
        # Measured against a simulated arc that lifts the band noise for a stretch of
        # the sweep: three, five and seven passes all recovered the arc-free answer in
        # 25 runs out of 25, and two passes recovered it in none of them.
        self._passes = passes + 1 if passes % 2 == 0 else passes
        self._seconds_per_step = seconds_per_step
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the sweep to stop at the next step.  Safe from another thread."""
        self._cancelled = True

    def run(self, on_progress: ProgressCallback | None = None) -> SweepResult:
        """Sweep every gain and return the choice.

        `on_progress` is called with (step, total, gain_db) before each step, so a
        dialog can say where it is.  It runs on this thread and must not block.
        """
        gains = sorted(self._source.supported_gains_db)
        if not gains:
            return SweepResult(None, 'The receiver reported no gain settings.',
                               0.0, None, None, ())
        readings: dict[float, list[tuple[float, float, int]]] = {gain: [] for gain in gains}
        total = self._passes * len(gains)
        step = 0
        for index in range(self._passes):
            # Alternating direction is what stops a slow drift over the sweep from
            # reading as a slope against gain.  Ascending and descending passes put
            # opposite ends of the range at opposite ends of the drift, so combining
            # them cancels what one pass alone would bake in.
            order = gains if index % 2 == 0 else list(reversed(gains))
            for gain in order:
                if self._cancelled:
                    return self._combine(readings, gains)
                if on_progress is not None:
                    on_progress(step, total, gain)
                self._measure_one(gain, readings)
                step += 1
        return self._combine(readings, gains)

    def _measure_one(self, gain_db: float,
                     readings: dict[float, list[tuple[float, float, int]]]) -> None:
        """Set one gain, wait out the stale blocks, and record what follows."""
        actual = self._source.set_gain(gain_db)
        for _ in range(self._source.blocks_to_discard_after_gain_change):
            if self._source.read() is None:
                return
        samples, clipped = self._collect(actual)
        if len(samples) == 0:
            return
        readings.setdefault(actual, []).append(
            (BandMeasurement.quiet_dbfs(samples), BandMeasurement.peak_dbfs(samples), clipped))

    def _collect(self, gain_db: float) -> tuple[np.ndarray, int]:
        """Gather about seconds_per_step of samples at the gain already set."""
        wanted = int(self._seconds_per_step * self._source.iq_sample_rate)
        parts: list[np.ndarray] = []
        clipped = 0
        gathered = 0
        while gathered < wanted:
            block = self._source.read()
            if block is None:
                logger.warning('The receiver stopped delivering blocks at %.1f dB, so '
                               'that gain was measured on less audio than the rest.',
                               gain_db)
                break
            clipped += block.clipped_samples
            samples = block.as_complex()
            parts.append(samples)
            gathered += len(samples)
        return (np.concatenate(parts) if parts else np.empty(0, dtype=np.complex128)), clipped

    def _combine(self, readings: dict[float, list[tuple[float, float, int]]],
                 gains: list[float]) -> SweepResult:
        """Fold the passes together, each quantity the way its question needs.

        The floor takes the median, because it decides where the antenna takes over
        and a transient arc is noise against that.  The peak takes the maximum and the
        clipping the total, because for those the worst case is the only interesting
        one: a median peak would size the headroom for a quiet moment.
        """
        measurements = tuple(
            GainMeasurement(
                gain_db=gain,
                quiet_dbfs=float(np.median([r[0] for r in readings[gain]])),
                peak_dbfs=float(np.max([r[1] for r in readings[gain]])),
                clipped=int(sum(r[2] for r in readings[gain])),
                passes=len(readings[gain]))
            for gain in gains if readings.get(gain))
        return GainChooser(measurements, self._headroom_db).choose()

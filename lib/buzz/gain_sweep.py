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

from buzz.sdr import CLIPPING_WORTH_NOTICING
from buzz.sdr_device import IqBlock

logger = logging.getLogger(__name__)

# Told the sweep how far it has got: the step just starting, how many there are, and
# the gain about to be measured.
ProgressCallback = Callable[[int, int, float], None]


class SweepSource(Protocol):
    """What a sweep needs of whatever it reads from.  `buzz.sdr.SweepReader` is it.

    This is declared rather than imported so that the whole sweep runs against a
    stand-in with no receiver attached, the same reason RtlSdrDevice exists one layer
    down.

    A sweep moves the gain between measurements, which an RTL-SDR refuses while it
    streams, so the only implementation reads synchronously.  A streaming source cannot
    satisfy this and is not meant to: `set_gain` is the member it cannot offer.
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

    def drain(self) -> int:
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

# How long one frame of that percentile covers, in seconds.
#
# A frame has to fit inside the gap between two bursts, which is the whole reason the
# percentile can see the band underneath an arc at all.  A 120 pps train is bursts of
# 2.5 to 6 ms inside an 8.33 ms period, so the gap is 2.3 ms at worst, and a frame of
# 1 ms sits inside it with margin to spare.
#
# It was 1024 samples, which is 4 ms at 256 kHz and therefore the same order as a
# burst, so nearly every frame straddled one and the percentile had no clean frame to
# find.  Measured against a simulated train, error in the reported floor:
#
#     arc                       1 ms frame    4 ms frame
#     120 pps, 4 ms, +10 dB       -0.22 dB      +2.29 dB
#     120 pps, 6 ms, +10 dB       -0.00 dB      +6.72 dB
#     120 pps, 6 ms, +25 dB       +0.01 dB     +21.18 dB
#
# The 4 ms column is also fragile in a way the figures hide: its error depends on how
# the frame length happens to align with the pulse period, so the same setting reads
# correctly at 100 pps and 21 dB high at 120.  A frame that fits in a gap does not
# care about the alignment.
#
# What no frame length fixes is an arc with no gap.  At 7.5 ms of an 8.33 ms period
# every size above reads about 22 dB high, correctly: there is no quiet band to
# measure.  That is the case the documentation covers by saying to calibrate when the
# band is quiet.
_QUIET_FRAME_SECONDS = 0.001

# The fewest samples a frame may hold, whatever the rate works out to.
#
# A short frame estimates its own RMS badly, and the percentile of a wider spread sits
# further below the true floor, so the reading is dragged low.  Measured on clean
# noise, the bias against the real floor: 0.36 dB at 256 samples, 0.74 at 64, 1.07 at
# 32 and 2.35 at 8.  Below about 64 the measurement is mostly describing the noise of
# its own estimator rather than the band.
#
# The bias matters less than it looks, because it is the same at every gain.  A curve
# shifted equally throughout leaves KneeFit's antenna and converter terms scaled
# together and the share between them unchanged, so the floor bound does not move.
# Only the headroom bound sees it, against a reserve of 32 dB.
_MIN_QUIET_FRAME_SAMPLES = 64

# The antenna's share of the reported noise floor to aim for.
#
# The converter's own noise adds to the antenna's and the sum is what gets reported,
# so this share is the whole quantity the lower bound is about.  One half is the knee
# of the curve, where the antenna and the converter contribute equally.
_ANTENNA_SHARE_TARGET = 0.5

# The same figure as the error it costs, which is what anybody weighing it thinks in.
# It is derived rather than written as 3.0, so the two can never disagree.
_FLOOR_ERROR_TARGET_DB = -10.0 * np.log10(_ANTENNA_SHARE_TARGET)


@dataclass(frozen=True)
class _PassReading:
    """What one pass over one gain step measured, before the passes are combined.

    A record rather than a tuple because it carries two counts of different things,
    and `r[2]` against `r[3]` is the kind of distinction that survives review and then
    goes wrong in an edit.
    """

    quiet_dbfs: float
    peak_dbfs: float
    clipped: int
    raw_values: int


# Holds every pass measured so far, keyed by the gain the tuner settled on rather than
# the gain asked for, because a V4 snaps a request to its own nearest step.
_Readings = dict[float, list[_PassReading]]


@dataclass(frozen=True)
class GainMeasurement:
    """What one gain step looked like, combined across every pass of the sweep.

    `quiet_dbfs` is the median across passes, because it decides where the antenna
    takes over and a transient is noise against that question.  `peak_dbfs` and
    `clipped` are the maximum and the total, because the worst case is the only
    interesting one: a median peak would size the headroom for a quiet moment.

    `raw_values` is how many converter outputs `clipped` was counted out of, totalled
    the same way, and it is here so that the count can be read as a share.  Both count
    I and Q separately, which is how IqBlock reports them.
    """

    gain_db: float
    quiet_dbfs: float
    peak_dbfs: float
    clipped: int
    raw_values: int
    passes: int

    @property
    def clipping_worth_noticing(self) -> bool:
        """Whether this gain clipped enough for the clipping to mean anything.

        A share rather than a count, against the figure the monitor already uses to
        decide the same question while it runs.  One value at a rail is not evidence
        about arcs, because the tuner's own DC offset puts the occasional sample there
        at a high gain.  Five passes of a quarter second at 256 kHz collect 640,000
        raw values, so one of them is 1.6 parts per million against a bar of 4, and it
        takes three to disqualify a gain.  Counting a single one as clipping capped
        the headroom bound a step or more low and said nothing about why.

        Three out of 640,000 is a low bar in absolute terms, which is the intent.
        What it excludes is the isolated value, not a burst: an arc loud enough to
        reach the rail does it repeatedly, over the couple of thousand raw values a
        single 4 ms burst covers at this rate.
        """
        return self.clipped > 0 and self.clipped >= self.raw_values * CLIPPING_WORTH_NOTICING


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
    floor_bound_db: float | None
    headroom_bound_db: float | None
    measurements: tuple[GainMeasurement, ...]

    @property
    def floor_error_db(self) -> float:
        """How much high the reported noise floor reads at the chosen gain.

        The converter's own noise adds to the antenna's, and the sum is what gets
        reported, so this is the whole point of the floor bound.  An antenna share of
        one half is 3.01 dB, nine tenths is 0.46 dB.
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
    def frame_samples(sample_rate: int) -> int:
        """Samples per frame at this rate, never fewer than the floor above."""
        return max(round(_QUIET_FRAME_SECONDS * sample_rate), _MIN_QUIET_FRAME_SAMPLES)

    @staticmethod
    def quiet_dbfs(samples: np.ndarray, sample_rate: int) -> float:
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
        frame = BandMeasurement.frame_samples(sample_rate)
        usable = len(samples) // frame * frame
        if usable == 0:
            return float('-inf')
        frames = samples[:usable].reshape(-1, frame)
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
        times too large while getting A right, and C is the term the floor bound
        rests on.

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

    def floor_error_db(self, gain_db: float) -> float:
        """How much high the reported noise floor reads at this gain.

        The converter's own noise adds to the antenna's and the sum is what gets
        reported, so a share of one half reads 3.01 dB high and nine tenths 0.46 dB.
        """
        share = self.antenna_share(gain_db)
        if share <= 0.0:
            return float('inf')
        return float(-10.0 * np.log10(share))

    def gain_nearest_the_floor_target(self, gains_db: list[float]) -> float | None:
        """The offered gain whose reported floor sits closest to the target error.

        Nearest rather than the lowest gain inside a budget, which is what this was
        first.  A budget is a bar, and a bar decides by which side of it a step falls,
        so two steps 2.2 dB apart can sit either side of it and a fit that moves by
        half a decibel between runs moves the answer by a whole step.  A station near
        the bar saw exactly that: the same antenna gave 25.4 dB under one budget and
        20.7 under another, where the step between them was the reasonable answer
        throughout.  Measuring distance to a target instead makes the fit's own wobble
        cost a fraction of a step rather than all of one.

        It also bounds what the rule can spend.  The budget let the floor error run to
        whatever the next step down happened to cost, where the worst a target can
        accept is half the gap between two steps.  Swept over curve shapes with the
        converter between 1 and 100,000 times the antenna at unity gain, the chosen
        error stayed between 2.04 and 3.98 dB against a 3.01 dB target.

        None when even the highest gain leaves the antenna short of the target, which
        is an antenna too quiet for this converter rather than a failure of the sweep.
        Without that guard a fit that separated nothing would report every gain as
        equally far off and the lowest one would win a tie it should not be in.

        The candidates are the gains the tuner reported, so the answer is one of them
        by construction.  Nothing computes the knee as a continuous number and rounds
        it, because a rounded knee could name a gain the hardware does not have.
        """
        ordered = sorted(gains_db)
        if not ordered or self.antenna_share(ordered[-1]) < _ANTENNA_SHARE_TARGET:
            return None
        # Ascending, so min() breaks a tie towards the lower gain.  That is the side to
        # err on, because the decibel it costs the floor is one an arc gets to use.
        return min(ordered, key=lambda gain: abs(
            self.floor_error_db(gain) - _FLOOR_ERROR_TARGET_DB))


class GainChooser:
    """Turns a set of measurements into one gain, or into a reason there is not one.

    Two bounds, from opposite directions:

      * The **floor bound** is the gain whose reported noise floor reads closest to
        _FLOOR_ERROR_TARGET_DB above the truth.  Below it too much of what the station
        reports is its own receiver.
      * The **headroom bound** is the highest gain at which the quiet level still
        leaves `headroom_db` before a sample reaches the rail.  Above it an arc clips.

    The answer is the floor bound, checked against the headroom bound.  It sits at
    the knee rather than comfortably above it, because every dB above what the antenna
    needs is a dB an arc no longer has.

    They can cross.  An antenna quiet enough to need most of the tuner's range to beat
    the converter may need more gain than the headroom allows.

    They are not the same kind of bound, and that decides what happens then.  Headroom
    is hard, because clipping is nonlinear and cannot be undone: a clipped arc reads
    small and lifts the apparent floor in the same capture, so both numbers are wrong
    and nothing says so.  Dominance degrades a decibel at a time, and how far it has
    degraded is a figure this can measure and report.

    So headroom wins and the floor pays, and the answer says what it paid.  Refusing
    outright was the first design and is worse: a station near the crossing got no
    gain at all, and its operator then set one by hand, making this very trade without
    the figures needed to make it.
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
        floor_bound = fit.gain_nearest_the_floor_target(gains)
        headroom_bound = self._highest_gain_with_headroom()

        # Headroom first, because it is measured directly where the floor bound comes
        # from a fit.  A band loud enough to clip at every gain gives the fit nothing
        # to separate, so checking the floor bound first would report a puzzled fit
        # instead of the concrete thing an operator can act on.
        if headroom_bound is None:
            return SweepResult(
                None,
                f'Even the lowest gain leaves less than {self._headroom_db:.0f} dB '
                'before clipping, so the band is loud enough that an arc would clip '
                'whatever this is set to.  Try a higher band, where powerline noise '
                'is weaker, or a frequency further from where the antenna is '
                'resonant.  An attenuator ahead of the receiver is the last resort '
                'and the only one that helps if every band is this loud.',
                fit.antenna_share(min(gains)), floor_bound, None, self._measurements)
        # The two bounds are not the same kind of thing, and the answer follows from
        # that.  Headroom is a hard limit, because clipping is nonlinear and cannot be
        # undone: a clipped arc reads small and lifts the apparent floor in the same
        # capture.  The floor bound is a preference that degrades a decibel at a time,
        # and how far it has degraded is a number this can report.
        #
        # So when they conflict, headroom wins and the floor pays, and the reply says
        # what it paid.  Refusing instead was tried and is worse: a station near the
        # crossing then gets no gain at all, and its operator sets one by hand anyway,
        # making exactly this trade without the figures to make it on.
        if floor_bound is None or floor_bound > headroom_bound:
            return SweepResult(
                headroom_bound,
                f'{headroom_bound:.1f} dB is the most this band allows before an arc '
                f'would clip, which is below what the antenna needs, so the reported '
                f'noise floor will read about '
                f'{fit.floor_error_db(headroom_bound):.1f} dB high.  A larger or '
                f'better matched antenna is what would improve that.  Clipping cannot '
                f'be undone and a floor that reads high can, which is why the gain '
                f'went this way.',
                fit.antenna_share(headroom_bound), floor_bound, headroom_bound,
                self._measurements)

        # Both bounds are named, not just the one that won.  An operator whose arcs
        # clip at the chosen gain has no way to act otherwise: the remedy is to raise
        # [rtlsdr] arc_headroom_db until the headroom bound falls below this one, and
        # that is impossible to judge without knowing where it currently sits.
        return SweepResult(
            floor_bound,
            f'{floor_bound:.1f} dB is the gain whose reported noise floor comes '
            f'closest to the {_FLOOR_ERROR_TARGET_DB:.1f} dB target, reading about '
            f'{fit.floor_error_db(floor_bound):.1f} dB high.  It leaves at least '
            f'{self._headroom_db:.0f} dB for an arc, where clipping alone would have '
            f'allowed up to {headroom_bound:.1f} dB, so the floor is what set this.  If '
            f'arcs still clip at this gain, raise [rtlsdr] arc_headroom_db to bring '
            f'that {headroom_bound:.1f} dB down.',
            fit.antenna_share(floor_bound), floor_bound, headroom_bound,
            self._measurements)

    def _highest_gain_with_headroom(self) -> float | None:
        """The largest gain that leaves room for an arc, by both the model and the
        evidence.

        The model is the reserve above the quiet level, measured between bursts rather
        than from an observed peak, which is what lets this run on a dead band: sizing
        from a peak needs an arc to be present and nothing arranges that.

        The evidence is any clipping the sweep actually saw, above the share at which
        clipping means anything.  A gain that clipped is not a prediction about arcs,
        it is one that happened, so it outranks the reserve and so does every gain
        above it.  The five passes are what make this evidence rather than luck: an
        intermittent arc that fires during any one of them is caught, where a single
        pass would usually miss it.

        Both are needed.  The reserve alone let a station settle one step too high,
        because the sweep ran between bursts and the reserve turned out slightly tight
        for the initiation transient.  Clipping alone would say nothing on a dead
        band, which is most of the time.
        """
        clipping_started_at = min(
            (m.gain_db for m in self._measurements if m.clipping_worth_noticing),
            default=None)
        safe = [m.gain_db for m in self._measurements
                if m.quiet_dbfs + self._headroom_db <= 0.0
                and (clipping_started_at is None or m.gain_db < clipping_started_at)]
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

    # What a step costs beyond the samples it collects: the gain write, the blocks
    # thrown away while the tuner settles, and the USB turnaround on every read.
    #
    # The value came from measuring rather than from theory.  A V4 at 256 kHz swept
    # its 29 gains five times in about 75 seconds, which is 145 steps at 0.52 s each
    # against 0.25 s of samples.  It is here only to estimate how long a sweep will
    # take, so it does not have to be better than about right.
    _STEP_OVERHEAD_SECONDS = 0.27

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

    @property
    def passes(self) -> int:
        """How many times each gain gets measured, after the rounding above."""
        return self._passes

    def estimated_seconds(self, gain_count: int) -> float:
        """About how long a sweep of this many gains will take.

        Derived rather than stated, because the gain count belongs to the tuner.  A
        V4 offers 29 steps and other receivers offer more or fewer, so any fixed
        figure is right for one device and wrong for the rest.
        """
        return (self._passes * gain_count
                * (self._seconds_per_step + self._STEP_OVERHEAD_SECONDS))

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
        readings: _Readings = {gain: [] for gain in gains}
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

    def _measure_one(self, gain_db: float, readings: _Readings) -> None:
        """Set one gain, wait out the stale blocks, and record what follows."""
        actual = self._source.set_gain(gain_db)
        # Two buffers stand between the tuner and this loop, and both hold data from
        # before the change.  drain() empties the source's own queue, which the count
        # below does not cover, and the count then waits out librtlsdr's transfer
        # pool.  Draining first is what makes the count mean what it says.
        self._source.drain()
        for _ in range(self._source.blocks_to_discard_after_gain_change):
            if self._source.read() is None:
                return
        samples, clipped, raw_values = self._collect(actual)
        if len(samples) == 0:
            return
        readings.setdefault(actual, []).append(_PassReading(
            quiet_dbfs=BandMeasurement.quiet_dbfs(samples, self._source.iq_sample_rate),
            peak_dbfs=BandMeasurement.peak_dbfs(samples),
            clipped=clipped, raw_values=raw_values))

    def _collect(self, gain_db: float) -> tuple[np.ndarray, int, int]:
        """Gather about seconds_per_step of samples at the gain already set.

        The raw count comes back beside the clipped one, because a count of values at
        the rail says nothing on its own.  Counting it here rather than deriving it
        from the sample total keeps it true when a receiver stops part way through and
        the loop breaks early.
        """
        wanted = int(self._seconds_per_step * self._source.iq_sample_rate)
        parts: list[np.ndarray] = []
        clipped = 0
        raw_values = 0
        gathered = 0
        while gathered < wanted:
            block = self._source.read()
            if block is None:
                logger.warning('The receiver stopped delivering blocks at %.1f dB, so '
                               'that gain was measured on less audio than the rest.',
                               gain_db)
                break
            clipped += block.clipped_samples
            raw_values += len(block.raw)
            samples = block.as_complex()
            parts.append(samples)
            gathered += len(samples)
        return ((np.concatenate(parts) if parts else np.empty(0, dtype=np.complex128)),
                clipped, raw_values)

    def _combine(self, readings: _Readings, gains: list[float]) -> SweepResult:
        """Fold the passes together, each quantity the way its question needs.

        The floor takes the median, because it decides where the antenna takes over
        and a transient arc is noise against that.  The peak takes the maximum, and
        the clipping and the raw count their totals, because for those the worst case
        is the only interesting one: a median peak would size the headroom for a quiet
        moment.  The two counts are totalled together so that the share they form
        describes the same audio.
        """
        measurements = tuple(
            GainMeasurement(
                gain_db=gain,
                quiet_dbfs=float(np.median([r.quiet_dbfs for r in readings[gain]])),
                peak_dbfs=float(np.max([r.peak_dbfs for r in readings[gain]])),
                clipped=sum(r.clipped for r in readings[gain]),
                raw_values=sum(r.raw_values for r in readings[gain]),
                passes=len(readings[gain]))
            for gain in gains if readings.get(gain))
        return GainChooser(measurements, self._headroom_db).choose()

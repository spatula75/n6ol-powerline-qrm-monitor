"""Tests for the automatic tuner-gain calibration.

The arithmetic is the part worth pinning here, and none of it needs a receiver: the
knee fit is checked by generating a curve from a known antenna and converter and
asking for them back, and the sweep runs against a stand-in that models the same
physics.  See docs-notebook/sdr-gain-calibration.md for where the rules came from.
"""
import numpy as np
import pytest

from buzz.gain_sweep import (
    BandMeasurement,
    GainChooser,
    GainMeasurement,
    GainSweep,
    KneeFit,
    SweepResult,
)
from buzz.sdr import IqBlock

# The 29 steps an RTL-SDR Blog V4 reports, which is what the real sweep walks.
# The receiver rate these captures stand for, which sets the frame length.
IQ_RATE = 256_000

V4_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
            22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
            43.9, 44.5, 48.0, 49.6]


def _measurement(gain_db: float, quiet_dbfs: float, peak_dbfs: float = -10.0,
                 clipped: int = 0) -> GainMeasurement:
    return GainMeasurement(gain_db=gain_db, quiet_dbfs=quiet_dbfs, peak_dbfs=peak_dbfs,
                           clipped=clipped, passes=5)


def _curve(gains, antenna_at_unity, converter):
    """Quiet levels in dBFS for an antenna and converter of known size."""
    power = antenna_at_unity * 10 ** (np.array(gains) / 10) + converter
    return [_measurement(g, float(10 * np.log10(p))) for g, p in zip(gains, power)]


class FakeReceiver:
    """A receiver whose noise is antenna-through-the-gain plus a fixed converter floor.

    The arc is optional and fires on a fraction of blocks, which is what lets a test
    ask the question the whole design turns on: does the answer change when one is
    running?
    """

    def __init__(self, antenna_at_unity: float, converter: float,
                 seed: int = 0, arc_db: float | None = None) -> None:
        self._antenna = antenna_at_unity
        self._converter = converter
        self._rng = np.random.default_rng(seed)
        self._arc_db = arc_db
        # A constant added to every sample, standing in for the converter's own DC
        # offset.  The monitor filters it out, so a sweep must not be moved by it.
        self.dc_offset = 0.0
        self._gain = 0.0
        self.gains_set: list[float] = []
        self.reads = 0
        self.supported_gains_db = list(V4_GAINS)
        self.iq_sample_rate = 256_000
        self.blocks_to_discard_after_gain_change = 16

    def set_gain(self, gain_db: float) -> float:
        self._gain = min(self.supported_gains_db, key=lambda c: abs(c - gain_db))
        self.gains_set.append(self._gain)
        return self._gain

    def read(self, timeout: float = 1.0) -> IqBlock:
        self.reads += 1
        n = 2048
        power = self._antenna * 10 ** (self._gain / 10) + self._converter
        sigma = np.sqrt(power / 2)
        z = (self._rng.normal(0, sigma, n) + 1j * self._rng.normal(0, sigma, n))
        if self._arc_db is not None and self._rng.random() < 0.08:
            z[:200] *= 10 ** (self._arc_db / 20)
        z = z + complex(self.dc_offset, self.dc_offset)
        interleaved = np.stack([z.real, z.imag], axis=-1).ravel()
        raw = np.clip(np.round((interleaved + 1) * 127.5), 0, 255).astype(np.uint8)
        return IqBlock(raw=raw, arrived_at=0.0, index=self.reads)


class TestTheKneeFitRecoversWhatItWasGiven:
    """The fit claims to separate the antenna's contribution from the converter's.
    The way to check that is to build a curve out of a known pair and ask for them
    back, which is the only test here that could tell a correct fit from a plausible
    one.
    """

    @pytest.mark.parametrize('antenna,converter', [
        (1e-2, 1e-4),      # a loud broadband antenna
        (1e-6, 1e-4),      # antenna and converter comparable mid-range
        (3e-9, 1e-4),      # a mag loop, under the converter until high gain
    ])
    def test_it_returns_the_parameters_the_curve_was_built_from(self, antenna, converter):
        power = antenna * 10 ** (np.array(V4_GAINS) / 10) + converter
        fit = KneeFit(np.array(V4_GAINS), power)
        assert fit.antenna_at_unity == pytest.approx(antenna, rel=1e-6)
        assert fit.converter == pytest.approx(converter, rel=1e-6)

    def test_the_relative_weighting_is_what_makes_the_converter_recoverable(self):
        """The reason the fit is weighted by 1/P rather than solved plainly.

        Powers span five orders of magnitude across a sweep, so an unweighted solve is
        dominated by the top of the range and leaves the converter term almost free.
        This reproduces the unweighted answer beside the real one, so the difference
        is visible rather than asserted in a comment.
        """
        antenna, converter = 1e-4, 1e-4
        gains = np.array(V4_GAINS)
        rng = np.random.default_rng(3)
        power = (antenna * 10 ** (gains / 10) + converter) * rng.normal(1.0, 0.05, len(gains))

        basis = np.column_stack([10.0 ** (gains / 10.0), np.ones(len(gains))])
        unweighted, *_ = np.linalg.lstsq(basis, power, rcond=None)

        weighted = KneeFit(gains, power)
        assert weighted.converter == pytest.approx(converter, rel=0.15)
        assert unweighted[1] > converter * 10, (
            'the unweighted solve is supposed to be the bad one here.  If it has become '
            'accurate then the weighting no longer earns its complexity.')

    def test_a_fit_that_cannot_separate_them_reports_a_share_rather_than_a_negative(self):
        """Noise can push a coefficient below zero, which is not a physical answer.
        Clamping turns it into a share of 0 or 1, which is the honest summary.
        """
        fit = KneeFit(np.array([0.0, 10.0, 20.0]), np.array([1.0, 1.0, 1.0]))
        assert fit.antenna_at_unity >= 0.0
        assert fit.converter >= 0.0
        assert 0.0 <= fit.antenna_share(20.0) <= 1.0


class TestTheQuietLevelStepsOverBursts:
    """The measurement has to describe the band between arcs, because an arc may or
    may not be running and the answer must not depend on which.
    """

    def _noise(self, rms, n=64 * 1024, seed=0):
        rng = np.random.default_rng(seed)
        sigma = rms / np.sqrt(2)
        return rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)

    def test_it_measures_the_level_of_plain_noise(self):
        quiet = BandMeasurement.quiet_dbfs(self._noise(0.01), IQ_RATE)
        assert quiet == pytest.approx(20 * np.log10(0.01), abs=0.5)

    def test_an_arc_on_a_tenth_of_the_capture_barely_moves_it(self):
        """The property the whole calibration rests on.  A mean would move by about
        10 dB here; the percentile is what keeps the answer about the quiet band.
        """
        clean = self._noise(0.01)
        with_arc = clean.copy()
        frames = len(with_arc) // 10
        with_arc[:frames] *= 10 ** (30 / 20)
        assert (BandMeasurement.quiet_dbfs(with_arc, IQ_RATE)
                == pytest.approx(BandMeasurement.quiet_dbfs(clean, IQ_RATE), abs=0.5))

    def test_the_peak_does_notice_the_arc(self):
        """The peak exists precisely to see what the quiet level ignores, so the two
        moving together would mean one of them is not doing its job.
        """
        clean = self._noise(0.01)
        with_arc = clean.copy()
        with_arc[:100] *= 10 ** (30 / 20)
        assert BandMeasurement.peak_dbfs(with_arc) > BandMeasurement.peak_dbfs(clean) + 20

    def test_silence_reads_as_negative_infinity_rather_than_raising(self):
        assert (BandMeasurement.quiet_dbfs(np.zeros(4096, dtype=complex), IQ_RATE)
                == float('-inf'))
        assert BandMeasurement.peak_dbfs(np.zeros(4096, dtype=complex)) == float('-inf')

    def test_a_capture_shorter_than_one_frame_reads_as_silence(self):
        assert (BandMeasurement.quiet_dbfs(np.ones(4, dtype=complex), IQ_RATE)
                == float('-inf'))


class TestTheChooserWeighsBothBounds:
    """Dominance from below and headroom from above.  Each of the three ways they can
    fail to leave an answer says something different about the station, so each gets
    its own wording rather than one "calibration failed".
    """

    def test_it_picks_the_lowest_gain_where_the_antenna_dominates(self):
        """Lowest rather than highest: every dB above what the antenna needs is a dB
        an arc no longer has.
        """
        result = GainChooser(tuple(_curve(V4_GAINS, 1e-6, 1e-4)), 32.0).choose()
        assert result.chosen_db is not None
        assert result.antenna_share >= 0.5
        lower = [m.gain_db for m in result.measurements if m.gain_db < result.chosen_db]
        assert lower, 'the chosen gain is the lowest offered, so nothing was ruled out'

    def test_a_quiet_antenna_gets_told_so_rather_than_given_a_number(self):
        """The mag loop case.  A gain picked from a rule that failed is worse than
        being told the antenna is the problem.
        """
        antenna = 1e-14      # far below the converter at every gain the tuner offers
        result = GainChooser(tuple(_curve(V4_GAINS, antenna, 1e-4)), 32.0).choose()
        assert result.chosen_db is None
        assert 'antenna' in result.reason

    def test_a_band_too_loud_for_any_gain_asks_for_an_attenuator(self):
        """A real antenna, loud enough that the quiet level is already inside the
        reserve at the lowest gain the tuner offers.  It still rises with gain, since
        a flat curve would mean no antenna at all and is a different diagnosis.
        """
        loud = tuple(_curve(V4_GAINS, 1e-3, 1e-6))
        result = GainChooser(loud, 32.0).choose()
        assert loud[0].quiet_dbfs + 32.0 > 0.0, 'the fixture is meant to clip at once'
        assert result.chosen_db is None
        assert 'attenuator' in result.reason

    def test_a_loud_band_is_named_before_a_puzzled_fit(self):
        """Both diagnoses can be true at once for a loud band, because a curve with no
        usable headroom also gives the knee fit little to separate.  Headroom is the
        measured one, so it is the one reported.
        """
        loud = tuple(_measurement(g, -5.0) for g in V4_GAINS)
        assert 'attenuator' in GainChooser(loud, 32.0).choose().reason

    def test_bounds_that_cross_are_reported_as_crossing(self):
        """Neither bound is unreachable on its own, and together they leave nothing.
        This is the case the notebook predicted for a quiet antenna on a live band.
        """
        # Dominance needs high gain; headroom allows only low.
        curve = _curve(V4_GAINS, 2e-11, 1e-8)
        raised = tuple(GainMeasurement(m.gain_db, m.quiet_dbfs + 45.0, m.peak_dbfs,
                                       m.clipped, m.passes) for m in curve)
        result = GainChooser(raised, 32.0).choose()
        assert result.chosen_db is None
        assert result.lowest_usable_db is not None and result.highest_safe_db is not None
        assert result.lowest_usable_db > result.highest_safe_db
        assert 'do not overlap' in result.reason

    def test_the_chosen_gain_really_does_leave_the_reserve(self):
        """The bound stated as the property it exists to guarantee, so a change to how
        it is computed has to keep the guarantee rather than merely keep the number.
        """
        result = GainChooser(tuple(_curve(V4_GAINS, 1e-6, 1e-4)), 32.0).choose()
        chosen = next(m for m in result.measurements if m.gain_db == result.chosen_db)
        assert chosen.quiet_dbfs + 32.0 <= 0.0

    def test_a_bigger_reserve_never_picks_a_higher_gain(self):
        """Asking for more headroom cannot buy more gain.  A monotonicity a reader can
        check by eye, and the kind of thing an edit to the bound could break silently.
        """
        curve = tuple(_curve(V4_GAINS, 1e-6, 1e-4))
        chosen = [GainChooser(curve, h).choose().chosen_db for h in (20.0, 26.0, 32.0, 38.0)]
        present = [c for c in chosen if c is not None]
        assert present == sorted(present, reverse=True) or len(set(present)) == 1

    def test_no_measurements_is_reported_rather_than_crashing(self):
        result = GainChooser((), 32.0).choose()
        assert result.chosen_db is None
        assert result.reason


class TestTheSweepAgainstAReceiver:
    """The whole thing, driven against a stand-in that models the same physics."""

    def _sweep(self, receiver, **kwargs):
        return GainSweep(receiver, headroom_db=32.0, passes=3,
                         seconds_per_step=0.02, **kwargs).run()

    def test_the_answer_does_not_depend_on_an_arc_being_present(self):
        """The constraint the entire design exists to satisfy.  Nobody can promise an
        arc is running when an operator opens the tool, so the same antenna has to
        give the same gain either way.
        """
        quiet_band = self._sweep(FakeReceiver(1.2e-6, 2e-8, seed=5))
        arcing = self._sweep(FakeReceiver(1.2e-6, 2e-8, seed=5, arc_db=26.0))
        assert quiet_band.chosen_db == arcing.chosen_db

    def test_it_discards_the_transfer_pool_after_every_gain_change(self):
        """Blocks in flight still carry the old gain.  Measuring without dropping them
        reads the previous step's answer shifted by one, which looks like a plausible
        curve and is wrong.
        """
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=1, seconds_per_step=0.0).run()
        assert receiver.reads >= (receiver.blocks_to_discard_after_gain_change
                                  * len(V4_GAINS))

    def test_passes_alternate_direction(self):
        """A slow drift over the sweep would otherwise read as a slope against gain."""
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=3, seconds_per_step=0.0).run()
        step = len(V4_GAINS)
        first = receiver.gains_set[:step]
        second = receiver.gains_set[step:2 * step]
        third = receiver.gains_set[2 * step:]
        assert first == sorted(first)
        assert second == sorted(second, reverse=True)
        assert third == sorted(third)

    def test_every_gain_is_measured_once_per_pass(self):
        receiver = FakeReceiver(1e-6, 1e-4)
        result = GainSweep(receiver, 32.0, passes=3, seconds_per_step=0.02).run()
        assert {m.passes for m in result.measurements} == {3}
        assert len(result.measurements) == len(V4_GAINS)

    def test_progress_is_reported_for_every_step(self):
        seen = []
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=3, seconds_per_step=0.0).run(
            on_progress=lambda step, total, gain: seen.append((step, total, gain)))
        assert len(seen) == 3 * len(V4_GAINS)
        assert [s for s, _, _ in seen] == list(range(3 * len(V4_GAINS)))
        assert {t for _, t, _ in seen} == {3 * len(V4_GAINS)}

    def test_cancelling_stops_early_and_still_answers(self):
        """A cancelled sweep returns what it has rather than nothing, because a
        partial curve still beats making the operator start again.
        """
        receiver = FakeReceiver(1e-6, 1e-4)
        sweep = GainSweep(receiver, 32.0, passes=5, seconds_per_step=0.0)

        def stop_after_a_few(step, total, gain):
            if step >= 3:
                sweep.cancel()

        result = sweep.run(on_progress=stop_after_a_few)
        assert len(result.measurements) < len(V4_GAINS) * 5
        assert result.reason

    def test_a_receiver_with_no_gains_is_reported(self):
        receiver = FakeReceiver(1e-6, 1e-4)
        receiver.supported_gains_db = []
        assert self._sweep(receiver).chosen_db is None


class TestAReceiverThatGoesAwayMidSweep:
    """A sweep runs for a minute and a half with somebody standing at the radio, so a
    cable pulled out part way through is an ordinary event rather than an exotic one.
    """

    class _DyingReceiver(FakeReceiver):
        def __init__(self, dies_after: int, **kwargs):
            super().__init__(1e-6, 1e-4, **kwargs)
            self._dies_after = dies_after

        def read(self, timeout: float = 1.0):
            if self.reads >= self._dies_after:
                self.reads += 1
                return None
            return super().read(timeout)

    def test_a_receiver_that_stops_during_the_discard_is_not_measured(self):
        """Every block after it went quiet would carry an unknown gain, so recording
        one would put a wrong point on the curve rather than leave a gap.
        """
        receiver = self._DyingReceiver(dies_after=0)
        result = GainSweep(receiver, 32.0, passes=1, seconds_per_step=0.02).run()
        assert result.measurements == ()
        assert result.chosen_db is None

    def test_a_receiver_that_stops_mid_capture_keeps_what_it_had(self):
        """A short capture still measures the right gain, so it is kept and the log
        says it was short.  Discarding it would throw away a usable point.
        """
        receiver = self._DyingReceiver(dies_after=40)
        result = GainSweep(receiver, 32.0, passes=1, seconds_per_step=0.05).run()
        assert result.measurements, 'the gains measured before it died were discarded'

    def test_a_gain_step_that_yields_no_samples_is_skipped_rather_than_recorded(self):
        """np.median of nothing is a NaN, and a NaN in the curve poisons the fit
        silently rather than failing.
        """
        receiver = self._DyingReceiver(dies_after=17)
        result = GainSweep(receiver, 32.0, passes=1, seconds_per_step=0.02).run()
        assert all(np.isfinite(m.quiet_dbfs) for m in result.measurements)


class TestTheDegenerateAnswers:
    """Values that are correct and unusual.  Each one used to be a division or a
    logarithm that would have raised on real data from a quiet station.
    """

    def test_an_empty_capture_has_no_peak_rather_than_raising(self):
        assert BandMeasurement.peak_dbfs(np.empty(0, dtype=complex)) == float('-inf')

    def test_a_floor_made_entirely_of_converter_noise_reports_infinite_error(self):
        """Rather than a logarithm of zero.  It is the honest number: none of the
        reading is the band, so the error against the band is unbounded.
        """
        result = SweepResult(None, 'x', antenna_share=0.0, lowest_usable_db=None,
                             highest_safe_db=None, measurements=())
        assert result.floor_error_db == float('inf')

    @pytest.mark.parametrize('share,expected_db', [(1.0, 0.0), (0.5, 3.01), (0.9, 0.46)])
    def test_the_floor_error_is_what_the_share_implies(self, share, expected_db):
        """The figures quoted in the docstring, checked rather than asserted in prose."""
        result = SweepResult(None, 'x', share, None, None, ())
        assert result.floor_error_db == pytest.approx(expected_db, abs=0.01)

    def test_a_silent_sweep_gives_a_share_of_zero_rather_than_dividing_by_it(self):
        fit = KneeFit(np.array([0.0, 10.0, 20.0]), np.array([0.0, 0.0, 0.0]))
        assert fit.antenna_share(20.0) == 0.0


class TestTheAnswerIsAlwaysAGainTheTunerHas:
    """A gain the hardware cannot be set to would be written into the config, snapped
    by the driver at the next startup, and the calibration would then be wrong by the
    difference with nothing saying so.
    """

    def test_the_chosen_gain_is_one_the_receiver_offered(self):
        for antenna, converter in ((1.2e-6, 2e-8), (1e-6, 1e-4), (2e-8, 1e-4)):
            for seed in range(4):
                receiver = FakeReceiver(antenna, converter, seed=seed)
                result = GainSweep(receiver, 32.0, passes=3,
                                   seconds_per_step=0.02).run()
                if result.chosen_db is not None:
                    assert result.chosen_db in V4_GAINS, (
                        f'{result.chosen_db} dB is not a step this tuner offers')

    def test_both_bounds_are_offered_gains_too(self):
        """Not just the answer.  The two bounds are reported to the operator when they
        cross, so a figure the hardware never had would be quoted in the reason.
        """
        result = GainChooser(tuple(_curve(V4_GAINS, 1e-6, 1e-4)), 32.0).choose()
        for bound in (result.lowest_usable_db, result.highest_safe_db):
            assert bound is None or bound in V4_GAINS

    def test_the_knee_is_not_rounded_into_an_answer(self):
        """The continuous knee generally falls between two steps.  The chooser has to
        return the step at or above it rather than the knee itself, which is a number
        the tuner cannot be set to.
        """
        antenna, converter = 1e-6, 1e-4
        knee_db = 10 * np.log10(converter / antenna)
        assert knee_db not in V4_GAINS, 'pick a fixture whose knee misses every step'
        result = GainChooser(tuple(_curve(V4_GAINS, antenna, converter)), 32.0).choose()
        assert result.chosen_db in V4_GAINS
        assert result.chosen_db >= knee_db
        below = [g for g in V4_GAINS if g < result.chosen_db]
        assert not below or max(below) < knee_db, (
            'a lower offered step also cleared the knee, so this is not the lowest')

    def test_what_the_sweep_records_is_what_the_device_accepted(self):
        """The property that makes all of the above true.  Recording the request
        rather than the reply would label the curve with gains the hardware never had.
        """
        receiver = FakeReceiver(1e-6, 1e-4)
        result = GainSweep(receiver, 32.0, passes=1, seconds_per_step=0.02).run()
        assert {m.gain_db for m in result.measurements} <= set(receiver.gains_set)


class TestOnlyTheReportedGainsAreSwept:
    """The tuner's own list is the whole domain.  Nothing interpolates between steps,
    and nothing probes outside them, because a gain the hardware does not have cannot
    be measured and a value between two steps is just the nearer of the two.
    """

    def test_every_requested_gain_is_one_the_receiver_reported(self):
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=5, seconds_per_step=0.0).run()
        assert set(receiver.gains_set) == set(V4_GAINS)

    def test_each_reported_gain_is_visited_once_per_pass_and_no_more(self):
        """Not more, because extra visits cost the operator time for nothing; not
        fewer, because a gap in the curve is a gap in the fit.
        """
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=5, seconds_per_step=0.0).run()
        assert [receiver.gains_set.count(g) for g in V4_GAINS] == [5] * len(V4_GAINS)

    def test_a_shorter_list_is_swept_in_full_and_nothing_is_invented(self):
        """A different receiver reports a different list, and three steps is still a
        sweep rather than a special case.
        """
        receiver = FakeReceiver(1e-6, 1e-4)
        receiver.supported_gains_db = [0.0, 20.0, 40.0]
        result = GainSweep(receiver, 32.0, passes=2, seconds_per_step=0.02).run()
        assert set(receiver.gains_set) == {0.0, 20.0, 40.0}
        assert [m.gain_db for m in result.measurements] == [0.0, 20.0, 40.0]

    def test_the_step_count_is_what_the_progress_callback_promises(self):
        """The dialog shows "step N of M" from these, so a total that did not match
        the work would leave the bar stuck short of the end or run past it.
        """
        seen = []
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=3, seconds_per_step=0.0).run(
            on_progress=lambda step, total, gain: seen.append((step, total)))
        assert len(seen) == 3 * len(V4_GAINS)
        assert {total for _, total in seen} == {3 * len(V4_GAINS)}


class TestTheReceiverDcOffsetDoesNotReachTheFloor:
    """A receiver puts a strong false signal at its own tuning frequency.  The station
    tunes away from it and the one-sided filter rejects it, so the monitor never hears
    it, and a sweep that measured raw IQ would size the gain against an artifact.

    It bit worst where it mattered most: at low gain, where the true noise is smallest
    and the knee fit needs the bottom of the curve to estimate the converter term.
    """

    def _noise(self, rms=0.02, n=1 << 16, seed=0):
        rng = np.random.default_rng(seed)
        sigma = rms / np.sqrt(2)
        return rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)

    @pytest.mark.parametrize('offset', [0.004, 0.016, 0.04, 0.2])
    def test_the_quiet_level_ignores_it(self, offset):
        """Before the fix, an offset of 0.04 read 7.07 dB high."""
        noise = self._noise()
        assert (BandMeasurement.quiet_dbfs(noise + complex(offset, offset), IQ_RATE)
                == pytest.approx(BandMeasurement.quiet_dbfs(noise, IQ_RATE), abs=0.01))

    def test_the_peak_still_sees_it(self, offset=0.2):
        """The two measurements differ on purpose.  Clipping happens at the converter
        before any filtering, so an offset really does use up headroom and belongs in
        the figure that decides whether an arc has room.
        """
        noise = self._noise()
        assert (BandMeasurement.peak_dbfs(noise + complex(offset, offset))
                > BandMeasurement.peak_dbfs(noise) + 6)

    def test_a_swept_curve_is_unchanged_by_a_constant_offset(self):
        """The property that matters: the gain chosen must not depend on the receiver's
        DC offset, since that is a property of the hardware and not of the band.
        """
        without = GainSweep(FakeReceiver(1e-6, 1e-4, seed=2), 32.0, passes=3,
                            seconds_per_step=0.02).run()
        biased = FakeReceiver(1e-6, 1e-4, seed=2)
        biased.dc_offset = 0.03
        with_offset = GainSweep(biased, 32.0, passes=3, seconds_per_step=0.02).run()
        assert without.chosen_db == with_offset.chosen_db

    def test_the_mean_is_the_right_estimate_for_complex_baseband(self):
        """Unlike the analyzer, which takes a median because its input is rectified
        and a 120 pps train pulls the mean away from zero.  Here both the noise and an
        arc are zero-mean, so only the offset survives the average.
        """
        noise = self._noise()
        arc = noise.copy()
        arc[:2000] *= 30
        assert abs(np.mean(arc)) < 0.002, 'an arc should not look like a DC offset'


class TestThePassCountStaysOdd:
    """The floor is combined with a median, and numpy's median of an even count
    averages the two middle values rather than picking one.  That gives up exactly the
    outlier rejection the passes were added to buy.

    Measured against a simulated arc that lifts the band noise for a stretch of the
    sweep: three, five and seven passes each recovered the arc-free answer 25 times out
    of 25, and two passes recovered it in none of them.
    """

    @pytest.mark.parametrize('asked,used', [(1, 1), (2, 3), (3, 3), (4, 5), (5, 5),
                                            (6, 7)])
    def test_an_even_request_is_rounded_up(self, asked, used):
        receiver = FakeReceiver(1e-6, 1e-4)
        GainSweep(receiver, 32.0, passes=asked, seconds_per_step=0.0).run()
        assert len(receiver.gains_set) == used * len(V4_GAINS)

    def test_the_shipped_default_is_already_odd(self):
        assert GainSweep.DEFAULT_PASSES % 2 == 1

    def test_the_median_of_an_even_count_is_why(self):
        """Stated as the numpy behavior it rests on, so the reason survives even if
        somebody later decides the rounding is unnecessary.

        An arc that runs through half the sweep contaminates half the readings, which
        is the case that separates the two.  With three the clean majority wins
        outright; with four the answer is dragged halfway to the arc.
        """
        clean, arcing = 10.0, 24.0
        assert np.median([clean, clean, arcing]) == clean
        assert np.median([clean, clean, arcing, arcing]) == (clean + arcing) / 2


class TestTheFloorSurvivesARunningArc:
    """The percentile can only see the band underneath an arc if a frame fits inside
    the gap between two bursts.  At 4 ms it did not: a frame was the same order as a
    burst, so nearly every one straddled a burst and the reported floor read up to
    21 dB high, which dragged the knee down and gave a different gain every run.
    """

    FLOOR_RMS = 0.01

    def _band(self, seed, burst_ms=None, arc_db=None, pulse_rate=120, n=1 << 19):
        rng = np.random.default_rng(seed)
        sigma = self.FLOOR_RMS / np.sqrt(2)
        samples = rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)
        if arc_db is None:
            return samples
        period = IQ_RATE / pulse_rate
        width = int(burst_ms / 1000 * IQ_RATE)
        for index in range(int(n / period) + 1):
            start = int(index * period)
            samples[start:start + width] *= 10 ** (arc_db / 20)
        return samples

    def _error(self, **kwargs):
        """How far the reported floor sits from the floor that is really there."""
        measured = BandMeasurement.quiet_dbfs(self._band(**kwargs), IQ_RATE)
        return measured - 20 * np.log10(self.FLOOR_RMS)

    @pytest.mark.parametrize('burst_ms,arc_db', [(4.0, 10.0), (6.0, 10.0),
                                                 (6.0, 25.0), (2.5, 30.0)])
    def test_a_120_pps_arc_barely_moves_it(self, burst_ms, arc_db):
        """At 4 ms frames these read +2.3, +6.7, +21.2 and +1.0 dB high."""
        assert abs(self._error(seed=0, burst_ms=burst_ms, arc_db=arc_db)) < 0.6

    def test_a_100_pps_arc_barely_moves_it(self):
        """The other grid.  A frame that fits in a gap does not care how the frame
        length happens to align with the pulse period, which is what made the old
        setting read correctly at 100 pps and 21 dB high at 120.
        """
        assert abs(self._error(seed=0, burst_ms=6.0, arc_db=25.0,
                               pulse_rate=100)) < 0.6

    def test_a_clean_band_still_reads_true(self):
        """The cost of the shorter frame, which is a systematic pull downward from the
        percentile of a noisier per-frame estimate.
        """
        assert -0.6 < self._error(seed=1) < 0.0

    def test_an_arc_with_no_gap_is_not_rejected_and_should_not_be(self):
        """7.5 ms of an 8.33 ms period leaves no quiet band to measure, so reporting
        the arc is the honest answer.  No frame length fixes this, which is why the
        documentation says to calibrate when the band is quiet.
        """
        assert self._error(seed=0, burst_ms=7.5, arc_db=25.0) > 15.0

    def test_the_frame_is_a_millisecond_at_any_rate(self):
        for rate in (250_000, 256_000, 1_024_000):
            assert BandMeasurement.frame_samples(rate) / rate == pytest.approx(
                0.001, rel=0.01)

    def test_it_never_shrinks_to_measuring_its_own_estimator(self):
        """Below about 64 samples the reading describes the spread of the estimate
        rather than the band: measured bias is 0.74 dB at 64, 1.07 at 32, 2.35 at 8.
        """
        assert BandMeasurement.frame_samples(8_000) == 64
        assert BandMeasurement.frame_samples(1) == 64

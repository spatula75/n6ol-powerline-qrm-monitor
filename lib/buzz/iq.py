"""Convert a software defined radio's IQ stream into the mono audio this program expects.

An SDR hands back complex baseband: pairs of numbers describing a whole block of
radio spectrum centered on whatever the device is tuned to.  Everything downstream
of buzz.sampler expects something quite different, a single real audio channel
carrying one narrow slice of that spectrum, which is what a radio in SSB
produces.  This module converts the first into the second, and it is deliberately
the only place that knows about both.

This produces audio rather than an envelope.  Handing the analyzer np.abs(iq) would
be easy and wrong.  ContinuousAnalyzer._capture subtracts the median of its window as
a DC offset, which works because SSB audio swings either side of zero.  The median is
then the sound card's offset and nothing else.

An envelope is never negative, so its median is the noise floor instead.  Subtracting
that and rectifying again would fold the noise floor onto itself and compress the
pulse-to-gap ratio the whole measurement depends on.  Real audio keeps the assumption
true, so the analyzer, the recorder, playback, rendering and the display all work with
no changes at all.

The chain has four steps:

1. Mix, which slides the spectrum sideways until the frequency of interest sits at
   zero.
2. Filter to the wanted bandwidth, keeping one side of zero and rejecting the other.
3. Decimate, keeping one sample in `decimation` and discarding the rest.
4. Take the real part and scale to int16.

Steps 2 and 3 are one operation in practice.  Throwing away 15 of every 16 samples
drops the sample rate by 16, which drops the Nyquist frequency by 16 as well.
Anything above the new Nyquist does not vanish when the samples go.  It folds back
down into the band that remains, arriving at a mirrored frequency where nothing can
tell it apart from signal that was always there.

So the filter has to remove that content first, and the filter is the real work while
the discarding is bookkeeping.  It matters more here than in most applications,
because a powerline arc is broadband by nature.  Folded energy from an arc reads as a
stronger arc rather than as an artifact.

Read this module alongside buzz.sampler.  Nothing here touches a device, a thread
or a file.  The hardware lives in the SDR source, which feeds blocks through a
converter and appends the result to the ring buffer.  Keeping this part clear of all
that is what lets it be tested exhaustively with no receiver attached.
"""

from math import ceil, gcd

import numpy as np
from scipy.signal import firwin, upfirdn

from buzz.constants import FULL_SCALE_COUNTS, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

# Which side of the tuned frequency to keep.  A radio in USB hears the spectrum
# above the dial and one in LSB hears the spectrum below it, with the audio running
# backwards against frequency.  Either one works for measuring an arc, since a
# broadband burst has no sideband of its own.  The setting exists so that a file can
# record which one was used.
UPPER = 'upper'
LOWER = 'lower'

# How far down the filter pushes everything outside the wanted band, in dB.
#
# Chosen against what the receiver can represent rather than picked for roundness.
# The receiver's 8-bit converter spans about 45 dB between its own noise and full
# scale, measured on the hardware.  At 70 dB of rejection, anything that folds in
# during decimation arrives at least 25 dB below the quietest level the converter
# can express, so it cannot affect a reading.  Going deeper would only buy more
# filter taps.
_STOPBAND_DB = 70.0

# Kaiser fitted these four numbers to measurements of real filters, so they carry no
# meaning on their own and cannot be derived from anything else here.  They are named
# rather than left bare so that the two expressions below read as the formulas they
# are.  Writing A for the stopband depth in dB:
#
#     shape parameter   beta = BETA_SCALE * (A - BETA_OFFSET)
#     length in taps    N    = (A - LENGTH_OFFSET) / (LENGTH_SCALE * skirt)
#
# where `skirt` is the skirt width in radians per sample.  The beta expression holds
# only for A above 50 dB, and Kaiser gives a different one below that.  Nothing in
# the code would notice the boundary being crossed, so a test pins _STOPBAND_DB above
# it - see test_the_stopband_depth_stays_inside_kaisers_formula.
_KAISER_BETA_SCALE = 0.1102
_KAISER_BETA_OFFSET = 8.7
_KAISER_LENGTH_SCALE = 2.285
_KAISER_LENGTH_OFFSET = 8.0

# The skirt is the stretch of frequency over which a filter changes from passing
# signal to rejecting it.  This constant sets how wide that stretch is, as a fraction
# of the bandwidth being kept.  A narrower skirt gives a sharper band edge and costs
# more taps.
#
# The value came from measuring, not from theory.  Filters were built at tap counts
# from 257 to 1201, and each one was tested against the two jobs this filter has.
#
# The first job is to reject the bands that fold onto the measurement during
# decimation.  Every filter tested managed better than 82 dB there, which is ample.
# The second job is to reject the unwanted sideband.  The leakiest filter tested
# shifted the level by 0.14 dB, which is also ample.
#
# So neither job decides the tap count.  What is left is the band edge, and a skirt
# half as wide as the band gives an edge much like an ordinary SSB filter's.  That is
# the value used here.  It builds a 561-tap filter and costs 3.0% of one CPU core.
# Halving the skirt would need 1121 taps and 5.3%.  Both timings cover the whole
# conversion, not the filter alone.
_SKIRT_FRACTION = 0.5

# int16's negative rail sits one step further from zero than its positive one, so
# the two limits are not symmetric.  Writing the positive one as FULL_SCALE_COUNTS
# minus one keeps the pair tied to the shared constant rather than restating it.
_INT16_MAX = FULL_SCALE_COUNTS - 1
_INT16_MIN = -FULL_SCALE_COUNTS


class IqToAudio:
    """Turns blocks of IQ into blocks of int16 audio, keeping its place between calls.

    Feed it whatever the device hands over, in whatever sizes, and it returns the
    audio that has become available.  It holds the state that makes a stream of
    separate blocks behave as one continuous signal, and that is the part which has
    to be right.  Both the mixing sinusoid and the filter carry from one block to the
    next.  Restarting either at a block boundary would put a step in the audio at the
    block rate.  A step repeating at a steady rate is precisely what this program
    exists to detect, so it would be found and believed.  The tests assert that
    feeding a signal in blocks gives the same samples as feeding it all at once.

    The device is tuned deliberately away from the frequency of interest, and
    `tuning_offset_hz` says how far.  An SDR puts a strong false signal at exactly
    the frequency it is tuned to, an artifact of the tuner leaking into its own
    mixer.  Measured on an RTL-SDR Blog V4 it stood 37 dB above the surrounding
    noise.  Tuning to one side and mixing back by the same amount moves that
    artifact well outside the measured band, into the filter's stopband.  The operator
    never has to think about it.
    """

    def __init__(self, iq_sample_rate: int, decimation: int, bandwidth_hz: int,
                 tuning_offset_hz: int, sideband: str = UPPER) -> None:
        self._validate(iq_sample_rate, decimation, bandwidth_hz, sideband)
        self._iq_sample_rate = iq_sample_rate
        self._decimation = decimation
        self._audio_sample_rate = self.audio_sample_rate_for(iq_sample_rate, decimation)
        self._n_taps = self.filter_length(iq_sample_rate, bandwidth_hz, decimation)
        self._taps = self.one_sided_filter(iq_sample_rate, bandwidth_hz,
                                  self._n_taps, sideband)
        # Where the first usable result sits in upfirdn's output.  See filter_length
        # for why this division comes out exact.
        self._first_usable = (self._n_taps - 1) // decimation

        self._tuning_offset_hz = tuning_offset_hz
        # The mixing sinusoid repeats exactly after this many samples, so counting
        # samples modulo it lets the mix run for months without drifting.  Two
        # positions this far apart sit at the same point in the sinusoid's cycle, so
        # reducing the count by it loses nothing.  buzz.dsp.pulse_phase_period rests
        # on the same gcd argument, applied to the pulse grid instead.
        self._mix_period = (iq_sample_rate // gcd(abs(tuning_offset_hz), iq_sample_rate)
                            if tuning_offset_hz else 1)

        self._pending = np.empty(0, dtype=np.complex128)
        self._sample_index = 0
        self._saturated = 0

    @property
    def processing_gain_bits(self) -> float:
        """Bits of noise resolution added by the IQ filter before int16 output.

        The root sum square of the taps is the gain applied to uncorrelated sample
        noise.  Expressing its reduction in bits lets the pipeline combine the
        receiver's delivered resolution with the converter that the scope sees.
        """
        return -float(np.log2(np.sqrt(np.sum(np.abs(self._taps) ** 2))))

    @staticmethod
    def _validate(iq_sample_rate: int, decimation: int, bandwidth_hz: int,
                  sideband: str) -> None:
        """Refuse a combination of settings that cannot produce honest audio.

        Each of these would otherwise produce output rather than an error, which is
        the worse failure.  A wrong audio rate mislabels every later measurement, and
        a bandwidth wider than the audio can hold folds the top of the band back onto
        the bottom.

        The checks run in this order because each needs the ones before it to have
        passed.  The audio rate cannot be worked out until the decimation is known to
        be positive and to divide.  The bandwidth limit is then a fraction of that
        rate.
        """
        if decimation < 1:
            raise ValueError(
                f'Decimation is {decimation}, and it must be 1 or more.  It is the '
                'number of IQ samples kept per audio sample.  A value below 1 '
                'describes nothing.  Set [rtlsdr] decimation to a whole number.')
        if iq_sample_rate % decimation:
            raise ValueError(
                f'An IQ rate of {iq_sample_rate} Hz does not divide evenly by a '
                f'decimation of {decimation}.  The audio rate would not be a whole '
                'number of samples per second.  Every sample position downstream is '
                'counted in whole samples.  Choose a rate that divides, such as '
                f'{decimation * (iq_sample_rate // decimation)} Hz.')

        audio_rate = IqToAudio.audio_sample_rate_for(iq_sample_rate, decimation)
        if not MIN_SAMPLE_RATE <= audio_rate <= MAX_SAMPLE_RATE:
            lowest, highest = IqToAudio.decimation_bounds_for(iq_sample_rate)
            raise ValueError(
                f'An IQ rate of {iq_sample_rate} Hz decimated by {decimation} gives '
                f'{audio_rate} Hz of audio.  This program works only between '
                f'{MIN_SAMPLE_RATE} Hz and {MAX_SAMPLE_RATE} Hz.  Below the floor '
                'the 4 kHz the display and the analysis look at is above Nyquist.  '
                'Above the ceiling the fixed-size buffer holds too little history to '
                f'acquire reliably.  Set [rtlsdr] decimation between {lowest} and '
                f'{highest}, choosing one that divides {iq_sample_rate} exactly.')

        widest = IqToAudio.widest_bandwidth_for(audio_rate)
        if not 0 < bandwidth_hz <= widest:
            raise ValueError(
                f'A bandwidth of {bandwidth_hz} Hz does not fit in audio sampled at '
                f'{audio_rate} Hz, which carries {widest} Hz at most.  The filter '
                'skirt reaches past the band edge, so the limit sits below half the '
                'audio rate.  Anything above it folds back onto the measurement.  '
                f'Lower [rtlsdr] bandwidth_hz to {widest} or less, or lower the '
                'decimation to raise the audio rate.')

        if sideband not in (UPPER, LOWER):
            raise ValueError(
                f'The sideband is {sideband!r}, and it must be {UPPER!r} or '
                f'{LOWER!r}.  It selects which side of the tuned frequency the '
                'receiver hears.  Set [rtlsdr] sideband to one of those two values.')

    @staticmethod
    def widest_bandwidth_for(audio_rate: int) -> int:
        """The widest band that can be kept at `audio_rate` without an alias folding in.

        Half the audio rate is the obvious answer and it is too generous, because the
        filter does not stop at the band edge.  Its skirt runs another
        `bandwidth_hz * _SKIRT_FRACTION / 2` past that edge before the stopband
        begins.  Whatever is still passing at half the audio rate therefore folds
        straight back onto the measurement.

        The figures come from measuring, at 256 kHz decimated by 16.  Peak response in
        the first alias band ran -71.6 dB at a 6400 Hz bandwidth and -24.5 dB at
        7000 Hz.  At the 8000 Hz that half the audio rate allowed it reached -6.1 dB,
        which is near enough to no rejection at all.  The tap count also falls as the
        bandwidth grows, from 561 to 289, so the leak widens faster than the band does.

        The condition is `bandwidth * (1 + _SKIRT_FRACTION / 2) <= audio_rate / 2`,
        and this is that rearranged.  It rounds down to a whole Hz, so the answer is
        always legal rather than one Hz over.
        """
        return int(audio_rate / 2 / (1 + _SKIRT_FRACTION / 2))

    @staticmethod
    def decimation_bounds_for(iq_sample_rate: int) -> tuple[int, int]:
        """The lowest and highest decimation whose audio rate falls inside the band.

        For the remedy in _validate, which names a range rather than a single value.
        It has to, because not every IQ rate has a divisor inside the band at all.
        Both ends round inwards, so any decimation between them gives a legal rate and
        the caller only has to find one that divides.
        """
        lowest = ceil(iq_sample_rate / MAX_SAMPLE_RATE)
        highest = iq_sample_rate // MIN_SAMPLE_RATE
        return lowest, highest

    @staticmethod
    def audio_sample_rate_for(iq_sample_rate: int, decimation: int) -> int:
        """The audio rate produced by a given IQ rate and decimation.

        Three places need this answer, and any two of them disagreeing would be hard to
        notice.  The converter sizes its filter with it.  The pipeline reports it to
        everything downstream.  The recorder writes it into a file's header.  So it is
        worked out in one place rather than three.
        """
        return iq_sample_rate // decimation

    @staticmethod
    def filter_length(iq_sample_rate: int, bandwidth_hz: int, decimation: int) -> int:
        """Number of filter taps needed, rounded up until the decimation lines up with it.

        Two separate requirements decide this, and satisfying only the binding one would
        leave the other to break quietly later.

        The first is the filter's own.  A Kaiser-windowed filter needs more taps for a
        narrower skirt and for deeper rejection, and Kaiser's standard design formula
        says how many.  That is `estimate` below.

        The second belongs to how the decimation is done.  scipy's upfirdn computes the
        filtered result and then keeps every `decimation`-th one, counting from the very
        first.  The earliest results are not usable, because the filter was still
        reaching back past the start of the data for samples that had not arrived.  The
        first usable result sits at index `n_taps - 1`, so the two grids agree only when
        `n_taps - 1` divides exactly by `decimation`.  Round up until it does.  Where
        they disagree, every output comes from a fraction of a sample away from where it
        belongs, and nothing about the output shows it.

        The third requirement is that the answer comes out odd, because a symmetric
        filter of even length delays the signal by half a sample.  group_delay_samples
        reports a whole number of IQ samples, and it cannot report a half.  An even
        length puts two exactly equal peaks in the impulse response, straddling the
        true delay, so which one np.argmax returns decides the answer.

        An even count only arises when `decimation` is odd, since the expression below
        is one more than a multiple of it.  Adding `decimation` again fixes the parity
        and leaves `n_taps - 1` divisible, because both terms keep that property.
        Taking the lcm of the two spacings the way panel_width does would round much
        further for the same effect.

        Of the 137 rate and decimation pairs this program admits, 8 reach the bump,
        and each pays between 1.1% and 4.8% more taps.  Nothing else changes, because
        the shipped default is odd already.  Those figures are pinned by
        test_the_parity_fix_costs_what_the_docstring_says.
        """
        skirt_hz = bandwidth_hz * _SKIRT_FRACTION
        # Kaiser's length estimate: deeper rejection and a narrower skirt each cost taps,
        # in proportion.  The skirt is converted to radians per sample because that is
        # the unit his formula is written in.
        skirt_radians = 2 * np.pi * skirt_hz / iq_sample_rate
        estimate = int(ceil((_STOPBAND_DB - _KAISER_LENGTH_OFFSET)
                            / (_KAISER_LENGTH_SCALE * skirt_radians)))
        n_taps = decimation * int(ceil((estimate - 1) / decimation)) + 1
        return n_taps if n_taps % 2 else n_taps + decimation

    @staticmethod
    def one_sided_filter(iq_sample_rate: int, bandwidth_hz: int, n_taps: int,
                         sideband: str) -> np.ndarray:
        """Build a filter that keeps `bandwidth_hz` on one side of zero and rejects the other.

        This is the step that makes the output equivalent to a radio in SSB rather
        than one hearing both sidebands at once.  An ordinary lowpass looks as though
        it would do the same job and does not.

        An ordinary filter has real coefficients, and a real-coefficient filter cannot
        tell a frequency from its negative.  Whatever it does to one, it does equally to
        the other.  In an IQ stream those two are different radio frequencies, one above
        the tuned frequency and one below, so a lowpass keeps a slice on each side.  The
        real part taken in step 4 then adds the two together.  The result looks
        perfectly reasonable, the analyzer locks onto it happily, every level reads about
        3.5 dB high, and "4 kHz" quietly means 8 kHz of radio spectrum.  Measured on a
        recorded arc, the two ways of doing it differed by 3.78 dB on signal and 3.19 dB
        on noise.

        A filter with complex coefficients has no such symmetry and can keep one side
        alone.  Building one takes two steps.  Start with an ordinary lowpass half the
        width we want, which passes from minus half-bandwidth to plus half-bandwidth.
        Then multiply its coefficients by a complex sinusoid at half the bandwidth.

        That slides the whole response up by the same amount, so it passes from zero to
        the full bandwidth instead.  Multiplying by a sinusoid to move a response
        sideways is the same trick as the mix in step 1, applied to the filter rather
        than to the signal.  Negating the shift selects the lower sideband.

        The window is Kaiser, whose single parameter trades skirt width against rejection
        depth.  The expression for that parameter is Kaiser's own, and it holds for any
        rejection deeper than 50 dB.
        """
        beta = _KAISER_BETA_SCALE * (_STOPBAND_DB - _KAISER_BETA_OFFSET)
        half_width = firwin(n_taps, bandwidth_hz / 2, fs=iq_sample_rate,
                            window=('kaiser', beta))
        direction = 1 if sideband == UPPER else -1
        shift = direction * 2j * np.pi * (bandwidth_hz / 2) / iq_sample_rate
        return half_width * np.exp(shift * np.arange(n_taps))

    @property
    def audio_sample_rate(self) -> int:
        """Sample rate of the audio this converter produces, in Hz."""
        return self._audio_sample_rate

    @property
    def n_taps(self) -> int:
        """Number of taps in the filter this converter built.

        The tap count sets the CPU cost and the sharpness of the band edge.  Reading
        it back from a converter beats recomputing it from the settings and hoping the
        two agree.
        """
        return self._n_taps

    @property
    def group_delay_samples(self) -> int:
        """How far the filter shifts the signal, counted in IQ samples.

        A symmetric filter answers about the middle of its window rather than its
        start.  Audio sample m is computed from IQ samples m*decimation through
        m*decimation + n_taps - 1, and what it describes is the sample at the center
        of that span.  This is how far along the center sits.

        Anything matching an audio position back to an IQ position needs it:

            iq_index = audio_index * decimation + group_delay_samples

        Leaving the term out puts the IQ half a filter earlier than the audio it is
        meant to match.  At the default settings that is 280 samples, or 1.09 ms,
        which is a fifth of a 6 ms arc burst.  The error is constant, so nothing
        about the result looks wrong: no drift, no jitter, and no artifact to notice.

        This lives here because this class owns both numbers the mapping needs.  A
        caller that recomputed it would have to duplicate the tap count and the
        decimation, and then stay in step with them.
        """
        return (self._n_taps - 1) // 2

    @property
    def saturated_samples(self) -> int:
        """How many output samples have been clipped to fit int16 since the last reset.

        Saturation here means the receiver gain is too high for what the antenna is
        hearing.  A clipped arc measures smaller than it truly is, so the events it
        spoils are the loud ones that matter most.  Nothing about the audio looks
        wrong afterwards, so a caller that never reads this learns nothing from the
        failure.  SdrPipeline._warn_about_clipping is the one that does.
        """
        return self._saturated

    def reset(self) -> None:
        """Forget the part-converted block and the position in the mixing sinusoid.

        For starting a fresh stream through a converter that has already run, which
        happens when the device is reopened.  Without it, the first block of the new
        stream would be filtered together with the tail of the old one.  It would also
        be mixed from wherever in the sinusoid the previous stream stopped.
        """
        self._pending = np.empty(0, dtype=np.complex128)
        self._sample_index = 0
        self._saturated = 0

    def convert(self, iq: np.ndarray) -> np.ndarray:
        """Convert one block of IQ samples into however much audio is ready.

        The returned block holds roughly `len(iq) / decimation` samples, and it can
        be empty.  It falls short whenever the filter still needs samples that have
        not arrived.  That is always true of the first call, and it can be true of any
        call carrying very little data.  Those samples are not lost, because a later
        call holds and uses them.  That is why the total over a run comes out right
        even though no single call does.

        This reads as the four steps in the module docstring, in order.
        """
        mixed = self._mixed_to_baseband(iq)
        one_sideband = self._filtered_and_decimated(mixed)
        return self._as_int16(one_sideband.real)

    def _mixed_to_baseband(self, iq: np.ndarray) -> np.ndarray:
        """Slide the spectrum sideways so the frequency of interest sits at zero.

        Multiplying by a complex sinusoid at some frequency shifts everything in the
        signal by that frequency, which is what tuning a radio does.  The device
        is tuned `tuning_offset_hz` above what we want to hear, so what we want sits
        that far below zero, and shifting up by the same amount brings it back.

        The sinusoid has to continue smoothly from where the previous block left it,
        so the sample counter advances by the length of every block.  It is reduced
        modulo the sinusoid's exact repeat length, which keeps it small and exact
        however long the receiver runs.  A counter growing without limit would
        eventually lose precision in the multiplication below and put a slow error
        into the tuning.
        """
        if not np.iscomplexobj(iq):
            raise TypeError(
                f'The IQ block has dtype {iq.dtype}, and it must be complex.  Real '
                'samples carry no phase.  Without phase there is no way to tell a '
                'frequency above the tuned frequency from one below it.  Pass the '
                'samples as the receiver produced them.  Do not take the real part '
                'first.')
        positions = self._sample_index + np.arange(len(iq))
        self._sample_index = (self._sample_index + len(iq)) % self._mix_period
        turns = self._tuning_offset_hz * positions / self._iq_sample_rate
        return np.asarray(iq, dtype=np.complex128) * np.exp(2j * np.pi * turns)

    def _filtered_and_decimated(self, mixed: np.ndarray) -> np.ndarray:
        """Add a block to the backlog, then filter and decimate whatever it completes.

        upfirdn does the filtering and the decimating at once, in the order that
        costs least.  Filtering every sample and then discarding fifteen in every
        sixteen would spend most of the work on results nobody keeps, so upfirdn
        computes only the ones that survive.  Measured on 8192-sample blocks, that is
        1.8% of a CPU core against 4.9% for filtering everything first.  The saving
        is smaller than the sixteen-to-one arithmetic suggests, because upfirdn pays
        a fixed cost per call that a larger block would spread further.

        The backlog exists because the filter reaches backwards.  Every output needs
        `n_taps` consecutive inputs, so the last few samples of each block belong to
        outputs that cannot be computed until the next block arrives.  They stay in
        `_pending` and are read again next time, and only the samples no future
        output can need are dropped.

        Taking the block as an argument, rather than letting the caller append it,
        leaves `_pending` with one owner.  Adding to it and dropping from it are two
        halves of the same decision about what is still needed, so they belong in one
        place.

        The backlog is built locally and stored only once the filtering has returned.
        SdrPipeline._feed catches every exception and goes on to the next block.  A
        fault that repeats would otherwise add 16384 samples per block to a backlog
        nothing ever drops, which is 256 KB a block for as long as the receiver runs.
        Storing afterwards costs nothing and bounds the failure at one block.
        """
        pending = np.concatenate([self._pending, mixed])
        usable = len(pending) - self._n_taps
        if usable < 0:
            self._pending = pending
            return np.empty(0, dtype=np.complex128)
        n_out = usable // self._decimation + 1
        filtered = upfirdn(self._taps, pending, up=1, down=self._decimation)
        self._pending = pending[n_out * self._decimation:]
        return filtered[self._first_usable:self._first_usable + n_out]

    def _as_int16(self, audio: np.ndarray) -> np.ndarray:
        """Scale the audio to int16 counts, clipping anything that will not fit.

        A receiver reports its samples between -1 and 1, where 1 is as loud as the
        hardware can represent.  The rest of this program works in the counts a 16-bit
        sound card produces.  Scaling against FULL_SCALE_COUNTS matches the two, and
        that is the same reference every dB figure in the program uses.

        Clipping is possible even when the receiver itself is not overloading, because
        the filter can leave a peak slightly above where the input sat.  Without the
        clip a sample past the rail wraps around, so 40000 counts arrives as -25536.
        That reads as an enormous impulse, and an impulse is the very thing this
        program hunts, so it would be believed.
        """
        counts = audio * FULL_SCALE_COUNTS
        self._saturated += int(np.count_nonzero((counts > _INT16_MAX) | (counts < _INT16_MIN)))
        return np.clip(counts, _INT16_MIN, _INT16_MAX).astype(np.int16)

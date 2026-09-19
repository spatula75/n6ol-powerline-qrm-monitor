"""Tests for buzz.iq, the IQ-to-audio conversion.

Every test here is aimed at one specific way the chain could be wrong, and each one
was chosen because that particular mistake produces audio that looks entirely
reasonable.  A lost filter state, a folded alias, a doubled sideband and a rectified
envelope all sound plausible and analyze plausibly.  None of them announces itself.

Where a test asserts a number, that number comes from how the test signal was built
rather than from the chain's own output, so the chain cannot agree with itself.
"""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.signal import freqz, lfilter

from buzz import iq as iq_module
from buzz.constants import FULL_SCALE_COUNTS, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE
from buzz.dsp import amplitude_to_dbfs
from buzz.iq import LOWER, UPPER, IqToAudio

IQ_RATE = 256_000
DECIMATION = 16
BANDWIDTH = 4_000
OFFSET = 50_000
AUDIO_RATE = IQ_RATE // DECIMATION


def converter(**overrides):
    settings = dict(iq_sample_rate=IQ_RATE, decimation=DECIMATION,
                    bandwidth_hz=BANDWIDTH, tuning_offset_hz=OFFSET)
    settings.update(overrides)
    return IqToAudio(**settings)


def tone(hz_from_dial, n_samples, amplitude=0.1, rate=IQ_RATE, offset=OFFSET):
    """IQ carrying one steady tone `hz_from_dial` away from the frequency of interest.

    The receiver is tuned `offset` above the frequency of interest, so a signal that
    far from the dial sits at (hz_from_dial - offset) in the samples themselves.
    Positive means above the dial, which is the upper sideband.
    """
    n = np.arange(n_samples)
    return amplitude * np.exp(2j * np.pi * (hz_from_dial - offset) * n / rate)


def noise(n_samples, seed=0, amplitude=0.05):
    rng = np.random.default_rng(seed)
    return amplitude * (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples))


def peak_bin_db(audio, rate=AUDIO_RATE):
    """Level of the strongest frequency in a block of audio, in dB, and where it is."""
    window = np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(audio * window)) / len(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    peak = int(np.argmax(spectrum))
    return 20 * np.log10(spectrum[peak] + 1e-20), freqs[peak]


class TestTheDerivedGeometry:
    """The numbers the converter computes before it sees any samples."""

    def test_the_audio_rate_is_the_iq_rate_over_the_decimation(self):
        assert IqToAudio.audio_sample_rate_for(256_000, 16) == 16_000
        assert IqToAudio.audio_sample_rate_for(240_000, 15) == 16_000

    def test_the_filter_length_lets_the_decimation_divide_it_exactly(self):
        """upfirdn keeps every Nth result counting from the first, while the first
        usable result sits at index n_taps - 1.  The two grids coincide only when the
        decimation divides n_taps - 1, and where they do not, every output is taken a
        fraction of a sample from where it belongs with nothing in the audio to show
        it.  Swept over every combination that could plausibly be configured.
        """
        for rate in (8_000, 48_000, 240_000, 250_000, 256_000, 1_024_000):
            for decimation in range(1, 33):
                if rate % decimation:
                    continue
                n_taps = IqToAudio.filter_length(rate, 3_000, decimation)
                assert (n_taps - 1) % decimation == 0, (
                    f'{n_taps} taps at a decimation of {decimation} leaves a '
                    f'remainder of {(n_taps - 1) % decimation}.  upfirdn would then '
                    'take every output a fraction of a sample from where the filter '
                    'puts it.  Nothing in the audio reveals that.')

    def test_the_filter_length_is_odd_so_the_delay_is_a_whole_sample(self):
        for decimation in (2, 4, 8, 16, 32):
            assert IqToAudio.filter_length(IQ_RATE, BANDWIDTH, decimation) % 2 == 1

    def test_a_converter_reports_the_geometry_it_actually_uses(self):
        """The pipeline reads the audio rate to tell everything downstream what the
        samples mean, so a converter that reported one rate and produced another
        would mislabel every measurement taken after it.  The last assertion is the
        one with teeth: it compares the reported rate against how much audio a known
        length of IQ really produced.
        """
        c = converter()
        assert c.audio_sample_rate == IqToAudio.audio_sample_rate_for(IQ_RATE, DECIMATION)
        assert c.n_taps == IqToAudio.filter_length(IQ_RATE, BANDWIDTH, DECIMATION)

        one_second = c.convert(noise(IQ_RATE))
        assert len(one_second) == pytest.approx(c.audio_sample_rate, rel=0.01), (
            f'One second of IQ produced {len(one_second)} audio samples, while the '
            f'converter reports a rate of {c.audio_sample_rate} Hz.  Everything '
            'downstream counts seconds by dividing samples by that figure.')

    def test_a_narrower_skirt_would_need_more_taps(self):
        """Guards the direction of the Kaiser estimate.  A sign slip in it would still
        return a plausible tap count, just one that gets worse as the requirement gets
        harder.
        """
        assert IqToAudio.filter_length(IQ_RATE, 2_000, 16) > IqToAudio.filter_length(IQ_RATE, 8_000, 16)


class TestSettingsThatCannotProduceHonestAudio:
    """Each of these would otherwise produce output rather than an error."""

    def test_a_rate_that_does_not_divide_is_refused(self):
        """250000 over 15 is 16666.67, so the audio rate would not be a whole number
        of samples per second, and every sample position downstream is counted in
        whole samples.  Note that 250000 over 16 divides exactly, at 15625 Hz, so the
        pairing matters rather than the rate alone.
        """
        with pytest.raises(ValueError, match='does not divide evenly'):
            converter(iq_sample_rate=250_000, decimation=15)

    def test_a_decimation_below_one_is_refused(self):
        with pytest.raises(ValueError, match='must be 1 or more'):
            converter(decimation=0)

    def test_a_bandwidth_wider_than_the_audio_can_hold_is_refused(self):
        """At 16 kHz audio the band tops out at 8 kHz.  Anything wider folds back onto
        the measurement and cannot be separated from it afterward.
        """
        with pytest.raises(ValueError, match='does not fit in audio'):
            converter(bandwidth_hz=9_000)

    def test_an_unknown_sideband_is_refused(self):
        with pytest.raises(ValueError, match='must be'):
            converter(sideband='sideways')

    def test_real_samples_are_refused_rather_than_silently_mistuned(self):
        """Real samples cannot distinguish a frequency above the dial from one below,
        so the sideband selection would be meaningless.  It would still return audio.
        """
        with pytest.raises(TypeError, match='must be complex'):
            converter().convert(np.zeros(8192, dtype=np.float64))


class TestStreamingMatchesOnePass:
    """The state that makes a run of separate blocks behave as one signal.

    Restarting the filter or the mixing sinusoid at a block boundary puts a step in
    the audio at the block rate.  This program hunts for steps at a steady rate, so
    that fault would be detected, reported and believed.
    """

    @pytest.mark.parametrize('block', [DECIMATION, 100, 512, 4096, 8192, 30_000])
    def test_any_block_size_gives_the_same_samples_as_one_pass(self, block):
        # 50,000 samples is about 3,000 audio samples out, far more than the filter
        # needs to settle.  The smallest block here divides that into 3,125 calls,
        # which is enough to exercise every boundary case several times over without
        # the suite paying for a quarter of a million of them.
        signal = noise(50_000) + tone(1_500, 50_000)

        whole = converter().convert(signal)

        streamed = converter()
        pieces = [streamed.convert(signal[i:i + block])
                  for i in range(0, signal.size, block)]
        joined = np.concatenate(pieces)

        assert joined.size == whole.size, (
            f'Feeding {signal.size} samples in blocks of {block} produced '
            f'{joined.size} audio samples where one pass produced {whole.size}.  '
            'Samples are being dropped or repeated at the block boundaries.')
        assert np.array_equal(joined, whole), (
            f'Blocks of {block} gave different audio from one pass.  The filter state '
            'or the mixing sinusoid is not carrying across calls, which puts a step '
            'in the audio once per block.')

    def test_reset_returns_the_converter_to_a_fresh_one(self):
        signal = noise(100_000)
        fresh = converter().convert(signal)

        used = converter()
        used.convert(noise(50_000, seed=99))
        used.reset()

        assert np.array_equal(used.convert(signal), fresh)

    def test_the_mixing_sinusoid_does_not_drift_over_a_long_run(self):
        """The sample counter is reduced modulo the sinusoid's exact repeat length so
        that it stays small and stays exact.  A counter growing without limit would
        lose precision in the multiplication and mistune the receiver by a little
        more as the day went on, which no short test would notice.  This compares the
        tone's frequency early in a long run against its frequency late in the same
        run, rather than against a figure of its own.
        """
        c = converter()
        block, blocks = 8192, 60
        n = np.arange(block)
        pieces = []
        for repeat in range(blocks):
            start = repeat * block
            pieces.append(c.convert(
                0.1 * np.exp(2j * np.pi * (1_500 - OFFSET) * (start + n) / IQ_RATE)))
        audio = np.concatenate(pieces)

        quarter = len(audio) // 4
        _, early = peak_bin_db(audio[:quarter])
        _, late = peak_bin_db(audio[-quarter:])
        assert early == pytest.approx(late, abs=AUDIO_RATE / quarter), (
            f'The tone reads {early:.1f} Hz early in the run and {late:.1f} Hz late '
            'in it.  The mixing sinusoid is drifting, so the modulus the sample '
            'counter is reduced by does not match its true repeat length.')


class TestTheOutputMatchesADirectConvolution:
    """The whole chain, checked against nested loops written from the definition.

    The streaming tests above compare the converter against itself, one block size
    against another, so an error that shifts every path equally would pass them all.
    A systematic off-by-one in the `_pending` arithmetic is exactly that kind of
    error: every block size would agree, and every one of them would be wrong.

    This reference shares no code with the converter.  It does not use upfirdn, it
    does not keep a pending buffer, and it indexes the convolution straight from
    `y[m] = sum over k of h[k] * x[m*D + T - 1 - k]`.  Agreement between the two is
    therefore evidence rather than a restatement.

    The settings are small so the nested loops stay quick.  32 kHz is the lowest IQ
    rate that gives a legal audio rate at this decimation, since _validate refuses
    anything under MIN_SAMPLE_RATE, and a lower rate would build a shorter filter and
    a faster loop than any real setting could reach.
    """

    RATE, DECIM, BAND, OFFSET = 32_000, 4, 1_000, 500

    def reference(self, iq):
        """The chain again, in loops, with nothing borrowed from the converter."""
        n_taps = IqToAudio.filter_length(self.RATE, self.BAND, self.DECIM)
        taps = IqToAudio.one_sided_filter(self.RATE, self.BAND, n_taps, UPPER)
        mixed = iq * np.exp(2j * np.pi * self.OFFSET * np.arange(len(iq)) / self.RATE)

        out = []
        m = 0
        while m * self.DECIM + n_taps <= len(mixed):
            total = 0j
            for k in range(n_taps):
                total += taps[k] * mixed[m * self.DECIM + n_taps - 1 - k]
            out.append(total)
            m += 1
        counts = np.array(out).real * FULL_SCALE_COUNTS
        return np.clip(counts, -FULL_SCALE_COUNTS, FULL_SCALE_COUNTS - 1).astype(np.int16)

    @pytest.mark.parametrize('block', [37, 64, 400])
    def test_every_sample_matches_the_definition(self, block):
        iq = noise(1_200, amplitude=0.3) + tone(200, 1_200, rate=self.RATE,
                                                 offset=self.OFFSET)
        c = IqToAudio(self.RATE, self.DECIM, self.BAND, self.OFFSET)
        actual = np.concatenate([c.convert(iq[i:i + block])
                                 for i in range(0, len(iq), block)])
        expected = self.reference(iq)

        assert len(actual) == len(expected), (
            f'The converter produced {len(actual)} samples where a direct convolution '
            f'produces {len(expected)}.  One of them counts the outputs the filter has '
            'enough history for differently, which is the _pending arithmetic.')
        assert np.array_equal(actual, expected), (
            f'The converter disagrees with a direct convolution at block size {block}.  '
            'Check _first_usable and the n_out expression, since a constant offset in '
            'either shifts every output without changing how many there are.')


class TestTheOutputIsAudioAndNotAnEnvelope:
    """ContinuousAnalyzer._capture subtracts the median of its window as a DC offset.

    That holds only while the audio swings either side of zero.  An envelope never
    goes negative, so its median is the noise floor, and subtracting that before
    rectifying would fold the floor onto itself and flatten the pulse-to-gap ratio
    the whole measurement depends on.  The assumption lives in a different module
    from the code that has to honor it, which is exactly why it gets a test.
    """

    def test_the_audio_swings_either_side_of_zero(self):
        audio = converter().convert(noise(200_000))
        assert audio.min() < 0 < audio.max()

    def test_the_median_sits_at_zero_rather_than_at_the_noise_floor(self):
        audio = converter().convert(noise(200_000))
        floor = np.median(np.abs(audio))
        assert abs(np.median(audio)) < floor / 4, (
            f'The median of the audio is {np.median(audio):.1f} counts against a '
            f'noise floor of {floor:.1f}.  The analyzer treats the median as a DC '
            'offset, so a median near the floor means it is being handed something '
            'envelope-shaped rather than audio.')


class TestOnlyOneSidebandSurvives:
    """A real-coefficient lowpass cannot tell a frequency from its negative, so it
    keeps a slice on each side of the dial and the real part adds the two together.
    Measured on a recorded arc, that reads 3.78 dB high on signal and 3.19 dB on
    noise, and it silently doubles what the bandwidth setting means.
    """

    def test_a_tone_on_the_wanted_side_comes_through(self):
        audio = converter(sideband=UPPER).convert(tone(1_500, 200_000))
        level, where = peak_bin_db(audio)
        assert where == pytest.approx(1_500, abs=100)

    def test_a_tone_on_the_other_side_is_rejected(self):
        """The same tone, the same distance from the dial, on the far side of it.  A
        plain lowpass would pass both at equal strength.
        """
        wanted = converter(sideband=UPPER).convert(tone(1_500, 200_000))
        unwanted = converter(sideband=UPPER).convert(tone(-1_500, 200_000))

        kept, _ = peak_bin_db(wanted)
        leaked, _ = peak_bin_db(unwanted)
        assert kept - leaked > 40, (
            f'A tone 1500 Hz below the dial came through only {kept - leaked:.1f} dB '
            f'down on one 1500 Hz above it.  The filter is not one-sided, so both '
            'sidebands are being summed and every level reads about 3.5 dB high.')

    def test_choosing_the_lower_sideband_swaps_which_one_survives(self):
        below = converter(sideband=LOWER).convert(tone(-1_500, 200_000))
        above = converter(sideband=LOWER).convert(tone(1_500, 200_000))
        assert peak_bin_db(below)[0] - peak_bin_db(above)[0] > 40


class TestNothingFoldsIntoTheMeasurement:
    """Decimation drops the Nyquist frequency along with the sample rate, and content
    above the new limit folds down into the band that is kept.  It arrives at a
    mirrored frequency where nothing distinguishes it from signal that was always
    there.  A powerline arc is broadband, so folded arc energy would read as a
    stronger arc rather than as an artifact.
    """

    @pytest.mark.parametrize('offset_from_dial', [
        AUDIO_RATE - 1_500,          # folds onto 1500 Hz through the first repeat
        AUDIO_RATE + 1_500,
        2 * AUDIO_RATE - 1_500,
        2 * AUDIO_RATE + 1_500,
        3 * AUDIO_RATE + 1_500,
    ])
    def test_a_tone_that_would_fold_onto_the_band_is_rejected(self, offset_from_dial):
        inband = converter().convert(tone(1_500, 200_000))
        aliasing = converter().convert(tone(offset_from_dial, 200_000))

        kept, _ = peak_bin_db(inband)
        folded, _ = peak_bin_db(aliasing)
        assert kept - folded > 60, (
            f'A tone {offset_from_dial} Hz from the dial arrived only '
            f'{kept - folded:.1f} dB down on an in-band one.  It folds onto '
            f'{offset_from_dial % AUDIO_RATE} Hz during decimation, where nothing can '
            'separate it from signal that belongs there.')


class TestTheDcSpikeIsRejected:
    """An SDR puts a strong false signal at exactly the frequency it is tuned to, from
    the tuner leaking into its own mixer.  Measured at 37 dB above the surrounding
    noise on an RTL-SDR Blog V4.  Tuning to one side of the frequency of interest is
    what moves it out of the measured band, so the offset has a job to do.
    """

    def test_a_constant_in_the_iq_does_not_reach_the_audio(self):
        """Leakage at the tuned frequency is a constant in the samples, because the
        tuned frequency is zero in the receiver's own frame.
        """
        spike = np.full(200_000, 0.5 + 0.0j)
        audio = converter().convert(spike + noise(200_000, amplitude=0.001))
        just_noise = converter().convert(noise(200_000, amplitude=0.001))

        assert amplitude_to_dbfs(float(np.mean(np.abs(audio)))) < \
               amplitude_to_dbfs(float(np.mean(np.abs(just_noise)))) + 3.0, (
            'A constant five times the noise amplitude lifted the audio level.  The '
            'tuning offset is not moving the receiver away from the band being '
            'measured, so the spike is being measured along with the signal.')

    def test_no_offset_lets_the_spike_straight_through(self):
        """The counterpart, so the test above cannot pass for the wrong reason.  With
        the receiver tuned directly at the frequency of interest, the spike sits in
        the middle of the passband and the measurement is spoiled.
        """
        spike = np.full(200_000, 0.5 + 0.0j)
        audio = converter(tuning_offset_hz=0).convert(spike + noise(200_000, amplitude=0.001))
        assert float(np.mean(np.abs(audio))) > 1_000


class TestScalingToInt16:

    def test_a_full_scale_input_reaches_full_scale_counts(self):
        """The conversion is what ties a receiver's -1 to 1 range to the counts every
        dB figure in the program is measured against.
        """
        audio = converter().convert(tone(1_500, 200_000, amplitude=1.0))
        assert 0.7 < np.abs(audio).max() / 32_768 <= 1.0

    def test_an_overdriven_tone_is_clipped_rather_than_wrapped(self):
        """Without the clip, a sample past the rail wraps from a large positive value
        to a large negative one: 40000 counts arrives as -25536.  That reads as an
        enormous impulse, which is the shape of the signal this program hunts, so it
        would be believed rather than questioned.

        A wrap shows up as a sign change where the waveform should still be positive,
        so counting sign changes separates the two.  A clean tone crosses zero twice
        per cycle and clipping does not add crossings, so the expected count comes
        from the tone's own frequency rather than from anything the converter did.
        """
        c = converter()
        audio = c.convert(tone(1_500, 200_000, amplitude=4.0))

        assert c.saturated_samples > 0, 'a tone at four times full scale must saturate'
        expected = 2 * 1_500 * len(audio) / AUDIO_RATE
        crossings = int(np.count_nonzero(np.diff(np.sign(audio)) != 0))
        assert crossings == pytest.approx(expected, rel=0.02), (
            f'The output crosses zero {crossings} times where a 1500 Hz tone over '
            f'{len(audio)} samples should cross {expected:.0f} times.  The extra '
            'crossings are peaks wrapping to the opposite sign instead of clipping.')
        at_the_rail = np.count_nonzero(np.abs(audio) == 32_767)
        assert at_the_rail > len(audio) / 4, (
            f'Only {at_the_rail} of {len(audio)} samples sit at the rail.  A tone at '
            'four times full scale should spend most of its cycle clamped there.')

    def test_a_quiet_input_saturates_nothing(self):
        c = converter()
        c.convert(noise(200_000, amplitude=0.01))
        assert c.saturated_samples == 0

    def test_reset_clears_the_saturation_count(self):
        c = converter()
        c.convert(tone(1_500, 100_000, amplitude=4.0))
        c.reset()
        assert c.saturated_samples == 0


class TestTheFilterItself:
    """Properties of the coefficients, checked without running any samples through."""

    def test_the_upper_and_lower_filters_are_mirror_images(self):
        n_taps = IqToAudio.filter_length(IQ_RATE, BANDWIDTH, DECIMATION)
        upper = IqToAudio.one_sided_filter(IQ_RATE, BANDWIDTH, n_taps, UPPER)
        lower = IqToAudio.one_sided_filter(IQ_RATE, BANDWIDTH, n_taps, LOWER)
        assert np.allclose(upper, np.conjugate(lower))

    def test_the_filter_reaches_the_stopband_depth_it_declares(self):
        """_STOPBAND_DB is a promise, and three separate pieces have to be right for
        the filter to keep it: Kaiser's shape parameter, Kaiser's length estimate, and
        the rounding that aligns the taps with the decimation.  A wrong constant in
        any of them still builds a filter, just a shallower one, and a shallower one
        lets folded energy into the measurement.  Measuring the built filter checks
        all three at once, where asserting on the constants would check none of them.
        """
        n_taps = IqToAudio.filter_length(IQ_RATE, BANDWIDTH, DECIMATION)
        taps = IqToAudio.one_sided_filter(IQ_RATE, BANDWIDTH, n_taps, UPPER)

        w, h = freqz(taps, worN=65536, fs=IQ_RATE, whole=True)
        w = np.where(w > IQ_RATE / 2, w - IQ_RATE, w)
        order = np.argsort(w)
        w, response = w[order], 20 * np.log10(np.abs(h[order]) + 1e-15)
        response -= response[np.argmin(np.abs(w - BANDWIDTH / 2))]

        # Clear of both skirts, which run about half a bandwidth either side of the
        # passband edges at 0 and BANDWIDTH.
        stopband = (w > BANDWIDTH + BANDWIDTH / 2) | (w < -BANDWIDTH / 2)
        worst = response[stopband].max()
        assert worst <= -iq_module._STOPBAND_DB, (
            f'The filter only reaches {worst:.1f} dB in its stopband where it '
            f'declares {iq_module._STOPBAND_DB:.0f} dB.  Content that folds in during '
            'decimation would then arrive louder than the design allows.')

    def test_the_stopband_depth_stays_inside_kaisers_formula(self):
        """Kaiser's expression for the shape parameter holds only above 50 dB, and he
        gives a different one below that.  Nothing in the code checks which side of
        the boundary it is on, so lowering _STOPBAND_DB past 50 would go on using an
        expression that no longer applies and quietly build a filter that misses its
        own stopband.  This test is the only thing coupling the two.
        """
        assert iq_module._STOPBAND_DB > 50, (
            f'_STOPBAND_DB is {iq_module._STOPBAND_DB} dB, at or below the 50 dB floor '
            "of Kaiser's expression for beta.  Below that floor his other expression "
            'applies, and IqToAudio.one_sided_filter does not implement it.')

    def test_the_reported_group_delay_is_where_an_impulse_actually_comes_out(self):
        """group_delay_samples is the term that maps an audio position back to an IQ
        position, and the IQ recorder will depend on it.  Asserting it equals
        (n_taps - 1) // 2 would only restate the code, so this feeds an impulse
        through the filter the converter built and measures where it emerges.

        Getting this wrong offsets an IQ recording against the audio it is meant to
        match, by a constant, with no drift or artifact to give it away.
        """
        c = converter()
        taps = IqToAudio.one_sided_filter(IQ_RATE, BANDWIDTH, c.n_taps, UPPER)

        impulse = np.zeros(4 * c.n_taps, dtype=np.complex128)
        impulse[c.n_taps] = 1.0
        response = lfilter(taps, 1.0, impulse)
        arrived = int(np.argmax(np.abs(response)))

        assert arrived - c.n_taps == c.group_delay_samples, (
            f'An impulse at sample {c.n_taps} emerged at {arrived}, a delay of '
            f'{arrived - c.n_taps} samples, while group_delay_samples reports '
            f'{c.group_delay_samples}.  Any IQ pulled with that mapping would sit '
            f'{abs(arrived - c.n_taps - c.group_delay_samples)} samples away from '
            'the audio it is supposed to match.')

    def test_the_filter_passes_its_own_band_at_roughly_unity(self):
        """A filter with a large gain or loss of its own would shift every level the
        program reports, without changing anything else about the audio.
        """
        n_taps = IqToAudio.filter_length(IQ_RATE, BANDWIDTH, DECIMATION)
        taps = IqToAudio.one_sided_filter(IQ_RATE, BANDWIDTH, n_taps, UPPER)
        middle = np.sum(taps * np.exp(-2j * np.pi * (BANDWIDTH / 2)
                                      * np.arange(n_taps) / IQ_RATE))
        assert abs(middle) == pytest.approx(1.0, abs=0.05)


class TestTheBandwidthLimitAccountsForTheFilterSkirt:
    """Half the audio rate looks like the limit and is not.

    one_sided_filter's transition runs another `bandwidth * _SKIRT_FRACTION / 2` past
    the band edge before the stopband starts.  At a bandwidth of exactly half the audio
    rate that transition sits on top of the first alias band, so the top of the
    measured band folds back onto the bottom at nearly full amplitude.  A broadband arc
    has energy right there, and folded energy reads as a stronger arc.
    """

    def alias_rejection_db(self, bandwidth):
        """Peak response inside the first alias band, relative to the passband.

        Decimating by D folds everything near multiples of the audio rate onto the
        measurement.  The first such band starts at `audio_rate - bandwidth`, so this
        measures the worst the filter lets through between there and the audio rate,
        which is what decimation would bring back.
        """
        n_taps = IqToAudio.filter_length(IQ_RATE, bandwidth, DECIMATION)
        taps = IqToAudio.one_sided_filter(IQ_RATE, bandwidth, n_taps, UPPER)
        freqs, response = freqz(taps, worN=1 << 16, fs=IQ_RATE, whole=True)
        magnitude = np.abs(response)
        passband = magnitude[(freqs > 100) & (freqs < bandwidth * 0.9)].max()
        folds = (freqs >= AUDIO_RATE - bandwidth) & (freqs <= AUDIO_RATE)
        return 20 * np.log10(magnitude[folds].max() / passband)

    def test_the_widest_allowed_bandwidth_still_rejects_its_aliases(self):
        """The limit is set so the skirt finishes before the first alias band starts,
        which should leave rejection no worse than the filter's own stopband.
        """
        widest = IqToAudio.widest_bandwidth_for(AUDIO_RATE)
        rejection = self.alias_rejection_db(widest)

        assert rejection <= -iq_module._STOPBAND_DB, (
            f'At the widest allowed bandwidth of {widest} Hz, the first alias band is '
            f'rejected by only {-rejection:.1f} dB against a stopband of '
            f'{iq_module._STOPBAND_DB} dB.  Decimation folds that band onto the '
            'measurement, so the limit is too generous for the skirt the filter has.')

    def test_half_the_audio_rate_would_barely_be_rejected_at_all(self):
        """The figure the old limit allowed, kept as a test so the reason for the
        tighter one cannot be forgotten and quietly relaxed.

        This is the measurement that motivated the change rather than a property of
        the code, so it asserts the direction and a loose bound instead of the exact
        -6.1 dB it happens to produce.
        """
        rejection = self.alias_rejection_db(AUDIO_RATE // 2)

        assert rejection > -20, (
            f'A bandwidth of half the audio rate is now rejected by {-rejection:.1f} dB '
            'in the first alias band.  It used to be about 6 dB, near enough to none, '
            'which is why widest_bandwidth_for exists.  If the filter got this much '
            'sharper, the limit can be reconsidered.')

    def test_the_widest_bandwidth_is_accepted_and_one_hertz_more_is_not(self):
        widest = IqToAudio.widest_bandwidth_for(AUDIO_RATE)
        converter(bandwidth_hz=widest)

        with pytest.raises(ValueError, match=f'carries {widest} Hz at most'):
            converter(bandwidth_hz=widest + 1)

    def test_the_limit_is_derived_from_the_skirt_rather_than_written_down(self):
        """A drift pin.  _SKIRT_FRACTION is measured and could be re-measured, and the
        limit has to follow it rather than stay at whatever it evaluates to today.
        """
        expected = int(AUDIO_RATE / 2 / (1 + iq_module._SKIRT_FRACTION / 2))

        assert IqToAudio.widest_bandwidth_for(AUDIO_RATE) == expected, (
            'widest_bandwidth_for no longer follows _SKIRT_FRACTION.  The limit exists '
            'to keep the whole skirt below half the audio rate.  The skirt width is '
            'what _SKIRT_FRACTION measures, so the two have to move together.')


class TestTheAudioRateHasToLandInTheBandTheProgramWorksIn:
    """The SDR path is a second way into the sample-rate band, and it was unguarded.

    config.validate_sample_rate refuses a rate that arrived from the config or from a
    .wav.  Nothing refused one the receiver settings produced, and schema.json bounds
    neither iq_sample_rate nor decimation above 1, so an ordinary setting reached rates
    far outside it.
    """

    def test_the_canonical_receiver_rate_at_the_default_decimation_is_refused(self):
        """2.4 MS/s is the rate every RTL-SDR guide names, and 16 is the shipped
        decimation, so this pair is the easiest one in the world to type.  It gives
        150 kHz of audio: a ring buffer holding one second instead of 9.6, and every
        recorded .wav at a rate --playback then refuses.
        """
        with pytest.raises(ValueError, match='150000 Hz of audio'):
            IqToAudio(2_400_000, 16, 4_000, 50_000)

    def test_a_rate_below_the_floor_is_refused(self):
        """Below MIN_SAMPLE_RATE the 4 kHz band the analysis looks at is above Nyquist,
        which is the case validate_sample_rate's docstring exists to describe.
        """
        with pytest.raises(ValueError, match='4000 Hz of audio'):
            IqToAudio(256_000, 64, 2_000, 50_000)

    @pytest.mark.parametrize('decimation, audio_rate',
                             [(32, MIN_SAMPLE_RATE), (16, 16_000), (8, 32_000)])
    def test_rates_inside_the_band_are_accepted(self, decimation, audio_rate):
        assert IqToAudio(256_000, decimation, 2_000,
                         50_000).audio_sample_rate == audio_rate

    def test_both_ends_of_the_band_itself_are_legal(self):
        """The boundaries are inclusive, so a receiver landing exactly on either one
        works rather than failing by a single hertz.
        """
        for rate in (MIN_SAMPLE_RATE, MAX_SAMPLE_RATE):
            assert IqToAudio(rate * 4, 4, 2_000, 500).audio_sample_rate == rate

    def test_the_remedy_names_a_decimation_range_that_actually_works(self):
        """The message tells the operator to pick a decimation between two figures.
        Both ends have to be honest, or it sends them to another failure.
        """
        lowest, highest = IqToAudio.decimation_bounds_for(2_400_000)

        for decimation in (lowest, highest):
            rate = IqToAudio.audio_sample_rate_for(2_400_000, decimation)
            assert MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE, (
                f'decimation_bounds_for suggested {decimation}, which gives {rate} Hz '
                f'of audio, outside the {MIN_SAMPLE_RATE} to {MAX_SAMPLE_RATE} Hz band '
                'the same message says to stay inside.')
        assert IqToAudio.audio_sample_rate_for(2_400_000, lowest - 1) > MAX_SAMPLE_RATE
        assert IqToAudio.audio_sample_rate_for(2_400_000, highest + 1) < MIN_SAMPLE_RATE


# Every (IQ rate, decimation) pair an operator can reach: the rates an RTL-SDR
# actually supports, crossed with the decimations that divide them and leave the audio
# rate inside the band.  Built here rather than written out, so adding a rate or
# moving the band moves what the sweep below covers.
_SUPPORTED_IQ_RATES = (225_001, 250_000, 256_000, 300_000, 900_001, 1_024_000,
                       1_200_000, 1_400_000, 1_800_000, 1_920_000, 2_048_000,
                       2_400_000, 2_560_000)


def legal_settings():
    """Each (iq_sample_rate, decimation) pair IqToAudio._validate would accept."""
    return [(rate, decimation)
            for rate in _SUPPORTED_IQ_RATES
            for decimation in range(1, 400)
            if rate % decimation == 0
            and MIN_SAMPLE_RATE <= rate // decimation <= MAX_SAMPLE_RATE]


class TestTheFilterLengthSatisfiesBothConstraintsAtOnce:
    """filter_length answers to two requirements, and only one of them used to bind.

    `n_taps - 1` has to divide by `decimation`, or upfirdn's output grid sits a
    fraction of a sample away from where the samples belong.  And `n_taps` has to be
    odd, or the symmetric filter delays by half a sample and group_delay_samples,
    which returns whole IQ samples, cannot say so.

    The second held by accident while every decimation in use was even.  A sweep is
    cheap here, so it covers the domain rather than the default.
    """

    def test_every_legal_setting_gives_an_odd_tap_count_on_the_decimation_grid(self):
        wrong = []
        for rate, decimation in legal_settings():
            n_taps = IqToAudio.filter_length(rate, BANDWIDTH, decimation)
            if n_taps % 2 == 0:
                wrong.append(f'{rate}/{decimation}: {n_taps} taps is even, so the '
                             'filter delays by half a sample')
            if (n_taps - 1) % decimation:
                wrong.append(f'{rate}/{decimation}: {n_taps} taps leaves n_taps - 1 '
                             f'off the decimation grid by {(n_taps - 1) % decimation}')

        assert not wrong, (
            f'{len(wrong)} of {len(legal_settings())} legal settings break one of the '
            'two constraints filter_length has to satisfy together.  The fix differs '
            'by case, so each is named:\n  ' + '\n  '.join(wrong[:10]))

    def test_the_parity_fix_costs_what_the_docstring_says(self):
        """filter_length quotes two counts and a cost, and a reader has no way to check
        them without rebuilding the sweep.  This is that sweep.

        The counts depend on _SUPPORTED_IQ_RATES above, so a rate added there moves the
        figures in the docstring too, and this is what says so.
        """
        legal = legal_settings()
        bumped = []
        for rate, decimation in legal:
            # The count before the parity bump, which is what the docstring compares to.
            skirt = 2 * np.pi * BANDWIDTH * iq_module._SKIRT_FRACTION / rate
            estimate = int(np.ceil((iq_module._STOPBAND_DB - iq_module._KAISER_LENGTH_OFFSET)
                                   / (iq_module._KAISER_LENGTH_SCALE * skirt)))
            before = decimation * int(np.ceil((estimate - 1) / decimation)) + 1
            after = IqToAudio.filter_length(rate, BANDWIDTH, decimation)
            if after != before:
                bumped.append((rate, decimation, before, after))

        assert (len(legal), len(bumped)) == (137, 8), (
            f'filter_length says 8 of 137 legal settings reach the parity bump.  The '
            f'sweep now finds {len(bumped)} of {len(legal)}.  Update the docstring, or '
            'check whether a rate was added to _SUPPORTED_IQ_RATES here.')
        costs = [(after - before) / before for _, _, before, after in bumped]
        assert 0.011 <= min(costs) and max(costs) <= 0.048, (
            f'The bump now costs between {min(costs):.1%} and {max(costs):.1%} more '
            'taps.  filter_length quotes 1.1% to 4.8% as the whole price of the fix, '
            'so one of the two is now wrong.')

    def test_an_odd_decimation_is_what_would_have_broken_it(self):
        """Names the case the sweep protects, so a later reader can see why the bump
        is there without rerunning the sweep in their head.

        1.2 MS/s decimated by 75 gives 16 kHz of audio, which is an ordinary thing to
        want.  Before the parity fix it built 2626 taps.
        """
        n_taps = IqToAudio.filter_length(1_200_000, BANDWIDTH, 75)

        assert n_taps % 2 == 1 and (n_taps - 1) % 75 == 0
        assert n_taps == 2_701, (
            f'1.2 MS/s decimated by 75 now builds {n_taps} taps rather than 2701.  '
            'The underlying Kaiser estimate was 2626, an even number, and the parity '
            'fix adds one decimation to it.  If the estimate moved, check that the '
            'result is still the first odd count on the decimation grid above it.')

    def test_the_shipped_default_pays_nothing_for_the_parity_fix(self):
        """The bump only fires at an odd decimation, so the setting every station
        starts from builds exactly the filter it did before.
        """
        assert IqToAudio.filter_length(IQ_RATE, BANDWIDTH, DECIMATION) == 561

    @pytest.mark.parametrize('rate, decimation', [(IQ_RATE, DECIMATION),
                                                  (1_200_000, 75),
                                                  (256_000, 25)])
    def test_an_impulse_emerges_exactly_where_the_group_delay_says(self, rate, decimation):
        """The same check the default already had, run at the two settings whose tap
        count came out even before the fix.

        The impulse half of this cannot stand alone at an even tap count, because the
        response then has two peaks of equal magnitude straddling the true delay and
        np.argmax picks between them by tie-break.  So the reported delay is also
        measured a second way, by folding the filter about it.  A linear-phase filter
        has a symmetric magnitude, so the two halves line up to floating-point noise
        when the fold sits on the true center and not otherwise.  Measured here, an odd
        count mismatches by about 1e-18 and an even one by about 1e-5.
        """
        c = IqToAudio(rate, decimation, BANDWIDTH, 500)
        taps = IqToAudio.one_sided_filter(rate, BANDWIDTH, c.n_taps, UPPER)

        impulse = np.zeros(4 * c.n_taps, dtype=np.complex128)
        impulse[c.n_taps] = 1.0
        response = np.abs(lfilter(taps, 1.0, impulse))
        arrived = int(np.argmax(response))

        assert arrived - c.n_taps == c.group_delay_samples, (
            f'At {rate} Hz decimated by {decimation}, an impulse emerged {arrived - c.n_taps} '
            f'samples late while group_delay_samples reports {c.group_delay_samples}.  '
            'IQ pulled with that mapping would sit beside the audio it should match.')

        delay, magnitude = c.group_delay_samples, np.abs(taps)
        reach = min(delay, c.n_taps - 1 - delay)
        mismatch = np.abs(magnitude[delay - reach:delay][::-1]
                          - magnitude[delay + 1:delay + 1 + reach]).max()
        assert mismatch < 1e-12, (
            f'Folding the {c.n_taps}-tap filter about the reported delay of {delay} '
            f'leaves the two halves {mismatch:.2e} apart, so that index is not the '
            'center of symmetry.  An even tap count puts the center on a half sample, '
            'which group_delay_samples cannot report.')


class TestAFailedConversionDoesNotGrowTheBacklogForever:
    """SdrPipeline._feed catches every exception and goes on to the next block.

    A fault that repeats every block would therefore retry forever, and if the backlog
    grew before the filtering rather than after it, each retry would add another
    block's samples to a buffer nothing ever drops.  At 16384 complex samples a block
    that is 256 KB per attempt, for as long as the receiver runs.
    """

    def test_a_raising_filter_leaves_the_backlog_where_it_was(self):
        c = converter()
        c.convert(noise(IQ_RATE // 10))
        settled = len(c._pending)

        with patch('buzz.iq.upfirdn', side_effect=MemoryError('no room')):
            for _ in range(20):
                with pytest.raises(MemoryError):
                    c.convert(noise(IQ_RATE // 10))

        assert len(c._pending) == settled, (
            f'After 20 failed conversions the backlog holds {len(c._pending)} samples '
            f'rather than the {settled} it held before.  _pending is being added to '
            'before the filtering rather than after, so every retry costs memory that '
            'is never given back.')

    def test_the_converter_still_works_after_a_failure(self):
        """The bound is worth nothing if holding the backlog back corrupts the stream.
        A block lost to an exception is lost, and the next one has to carry on from
        the samples that were already there.
        """
        c = converter()
        c.convert(noise(IQ_RATE // 10))

        with patch('buzz.iq.upfirdn', side_effect=MemoryError('no room')):
            with pytest.raises(MemoryError):
                c.convert(noise(IQ_RATE // 10))

        assert len(c.convert(noise(IQ_RATE // 10))) > 0

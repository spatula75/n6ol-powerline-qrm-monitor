"""Tests for buzz.sdr.SdrPipeline, the thread joining capture to the ring buffer.

No receiver and no threads: the feeder's body is driven by calling _consume directly,
which is where all the behavior lives.  The thread itself only decides when to call
it, and a test that started it would be waiting on timeouts to prove nothing.
"""

import logging

from unittest.mock import patch

import numpy as np
import pytest

from buzz import sdr as sdr_module
from buzz.iq import IqToAudio
from buzz.sampler import buffer_chunks
from buzz.sdr import IqBlock, SdrPipeline
from buzz.sdr_device import RTL_SDR_FORMAT, DeviceProfile
from buzz.sdrplay_device import SDRPLAY_FORMAT

IQ_RATE, DECIMATION, BANDWIDTH, OFFSET = 256_000, 16, 4_000, 50_000
BLOCK = 16_384


class StubSource:
    """Stands in for SdrSource.  The pipeline reads from it, closes it, and asks it
    about the receiver clock.
    """

    def __init__(self):
        self.started = False
        self.closed = False
        self.blocks = []
        self.clock_drift_seconds = 0.0
        # The clipping report is a rate rather than a count, so it needs the rate.
        self.iq_sample_rate = 256_000
        # What the raw IQ buffer sizes its chunks by, when one is being kept.
        self.block_samples = BLOCK
        # Twelve, which is neither receiver's real answer, so a test that this
        # reaches the pipeline cannot pass by matching a real device by accident.
        self.effective_bits = 12
        # Two and a bit, which is neither receiver's real answer, for the same reason.
        self.scope_floor_steps = 2.25
        self.profile = DeviceProfile('stub', 'rtlsdr', RTL_SDR_FORMAT, 0, 0, False)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def read(self, timeout=0.5):
        return self.blocks.pop(0) if self.blocks else None


class FakeClock:
    """A clock a test can wind forward, so a minute of monitoring costs no time.

    The health report only looks once every _HEALTH_INTERVAL_SECONDS.  Waiting for the
    real clock would put that minute into the suite for every case here.
    """

    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def pipeline(clock=None, keep_iq=False):
    converter = IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH, OFFSET)
    return (SdrPipeline(StubSource(), converter, clock=clock or FakeClock(),
                           keep_iq=keep_iq),
            converter)


def raw_to_complex(raw):
    """The conversion under test, reached the way the pipeline reaches it."""
    return IqBlock(raw=np.asarray(raw, dtype=np.uint8), fmt=RTL_SDR_FORMAT,
                   arrived_at=0.0, index=0).as_complex()


def block(n_samples=BLOCK, seed=0, clipped=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(40, 215, size=n_samples * 2, dtype=np.uint8)
    raw[:clipped] = 255
    return IqBlock(raw=raw, fmt=RTL_SDR_FORMAT, arrived_at=0.0, index=1)


class TestTheRawConversion:

    def test_a_byte_of_zero_is_the_negative_rail(self):
        assert raw_to_complex(np.array([0, 0], dtype=np.uint8))[0] == -1 - 1j

    def test_a_byte_of_255_is_the_positive_rail(self):
        assert raw_to_complex(np.array([255, 255], dtype=np.uint8))[0] == 1 + 1j

    def test_bytes_interleave_as_i_then_q(self):
        """The device sends I then Q.  Reading them the other way round would put every
        signal on the wrong side of the dial, which is the one thing the quadrature is
        there to tell us.
        """
        got = raw_to_complex(np.array([255, 0], dtype=np.uint8))[0]
        assert got.real == 1.0 and got.imag == -1.0

    def test_the_midpoint_sits_near_zero(self):
        """An 8-bit converter has no exact zero, since 0 to 255 has no middle.  128
        gives the smallest non-negative value, which is what the analyzer's DC estimate
        has to cope with.
        """
        got = raw_to_complex(np.array([128, 128], dtype=np.uint8))[0]
        assert 0 < got.real < 0.01


class TestFeedingTheRingBuffer:

    def test_converted_audio_reaches_the_buffer(self):
        p, _ = pipeline()
        p._consume(block())
        assert p.total_samples > 0

    def test_audio_is_appended_in_whole_chunks(self):
        """get_snapshot decides how many chunks to read by dividing by CHUNK_SIZE, so a
        shorter chunk makes it return less audio than asked for, with no error.
        Measured on the real buffer: 256-sample chunks against a request for 4000
        samples returns 2064.
        """
        p, _ = pipeline()
        for i in range(4):
            p._consume(block(seed=i))
        assert p.total_samples % p.CHUNK_SIZE == 0, (
            f'{p.total_samples} samples reached the buffer, which is not a whole '
            f'number of {p.CHUNK_SIZE}-sample chunks.  get_snapshot would then hand '
            'the analyzer a window shorter than the one it requested.')

    def test_every_sample_is_accounted_for_across_blocks(self):
        """A converted block does not divide evenly into chunks, so what is left over
        has to wait rather than be dropped or padded.

        Rather than bounding the shortfall, this accounts for all of it.  Two blocks of
        IQ are worth a fixed number of audio samples, and every one of them has to be
        in the buffer, waiting as a remainder, or spent on the filter warming up.  A
        bound would pass while samples quietly went missing inside it.
        """
        p, converter = pipeline()
        p._consume(block(seed=1))
        first = p.total_samples
        p._consume(block(seed=2))

        ideal = 2 * BLOCK // DECIMATION
        warm_up = (converter.n_taps - 1) // DECIMATION
        assert p.total_samples > first
        assert p.total_samples + len(p._leftover) + warm_up == ideal, (
            f'{p.total_samples} samples reached the buffer, {len(p._leftover)} are '
            f'waiting and {warm_up} went to the filter warming up.  That comes to '
            f'{p.total_samples + len(p._leftover) + warm_up} rather than the {ideal} '
            'two blocks of IQ are worth.  Audio is being lost between them.')

    def test_a_snapshot_comes_back_the_length_it_was_asked_for(self):
        """The property the chunking exists to protect, checked through the buffer's
        own reader rather than by counting appends.
        """
        p, _ = pipeline()
        for i in range(12):
            p._consume(block(seed=i))
        assert len(p.get_snapshot(4_000, align=400)) == 4_000

    def test_the_buffer_is_sized_for_the_audio_rate_not_the_iq_rate(self):
        """RingBufferPipeline sizes itself in seconds, so the rate it is handed decides
        how much history the analyzer can reach back through.

        Handing it the IQ rate would make the buffer sixteen times shorter in time
        while looking perfectly healthy, because every count downstream is in samples.
        This reads capacity_samples off the pipeline rather than the rate off the
        converter, since only the pipeline can be wrong about it.
        """
        p, converter = pipeline()
        expected = buffer_chunks(converter.audio_sample_rate,
                                 p.CHUNK_SIZE) * p.CHUNK_SIZE

        assert p.capacity_samples == expected, (
            f'The buffer holds {p.capacity_samples} samples where {expected} covers '
            f'the same seconds at {converter.audio_sample_rate} Hz.  At the IQ rate it '
            f'would hold {buffer_chunks(IQ_RATE, p.CHUNK_SIZE) * p.CHUNK_SIZE}, which '
            'is the same history divided by the decimation.')


class TestCountingClippedSamples:

    def test_clean_audio_counts_nothing(self):
        p, _ = pipeline()
        p._consume(block())
        assert p.clipped_samples == 0

    def test_clipped_raw_values_are_counted(self):
        p, _ = pipeline()
        p._consume(block(clipped=40))
        assert p.clipped_samples == 40

    def test_the_count_accumulates_across_blocks(self):
        p, _ = pipeline()
        p._consume(block(seed=1, clipped=10))
        p._consume(block(seed=2, clipped=5))
        assert p.clipped_samples == 15


class TestStartingAndStopping:

    def test_starting_starts_the_capture_underneath(self):
        p, _ = pipeline()
        p.start()
        try:
            assert p.source.started
        finally:
            p.close()

    def test_closing_closes_the_capture_underneath(self):
        p, _ = pipeline()
        p.start()
        p.close()
        assert p.source.closed

    def test_closing_without_starting_does_not_raise(self):
        """Shutdown after a failed setup must not replace a clean exit with a
        traceback.  join() on an unstarted thread raises.
        """
        p, _ = pipeline()
        p.close()
        assert p.source.closed


def test_the_pipeline_is_a_ring_buffer_like_every_other_source():
    """Consumers read this exactly as they read the sound card and playback, and must
    not be able to tell which they have.
    """
    from buzz.sampler import RingBufferPipeline
    p, _ = pipeline()
    assert isinstance(p, RingBufferPipeline)
    for name in ('get_snapshot', 'read_from', 'wait_for_data', 'total_samples',
                 'capacity_samples', 'clear'):
        assert hasattr(p, name), f'{name} is missing, so a consumer would break on it'


class TestWhatThePipelineSaysAboutItsReceiver:
    """The scope holds a pipeline, not a device, so the pipeline has to answer.

    This was written after the property was put on SweepReader by mistake, where
    nothing asks.  The scope then fell through to the base class, which answers
    sixteen, and every receiver silently got a sound card's magnification limit.
    ScopeWidget carries a coverage pragma, so nothing else would have found it.
    """

    def test_the_bit_depth_reaches_the_pipeline(self):
        p, converter = pipeline()
        expected = min(16.0, 12 + converter.processing_gain_bits)
        assert p.effective_bits == pytest.approx(expected)

    def test_the_multiple_the_floor_is_worth_reaches_the_pipeline_too(self):
        """The step size and how many steps are two different facts, and the second
        one belongs to the receiver rather than to the conversion.

        The filter changes the size of a step, which effective_bits already carries.
        Adding the filter to this as well would count it twice.
        """
        p, _ = pipeline()
        assert p.scope_floor_steps == 2.25

    def test_the_floor_follows_from_it(self):
        """End to end, in the unit the scope works in: a coarser receiver is allowed
        less magnification, and the arithmetic in between is scope.minimum_full_scale.
        """
        from buzz.scope import _FLOOR_STEPS, minimum_full_scale
        p, converter = pipeline()
        expected_bits = min(16.0, 12 + converter.processing_gain_bits)
        assert minimum_full_scale(p.effective_bits, _FLOOR_STEPS) == pytest.approx(
            minimum_full_scale(expected_bits, _FLOOR_STEPS))


class TestTheHealthCountersReachTheLog:
    """Four counters recorded a quiet failure and nothing read any of them.

    A receiver whose gain is too high clips every loud arc, measures it smaller than it
    is, and writes that figure to the CSV with nothing to mark it.  Lost samples do the
    same to the grid frequency.  Both look exactly like a quiet band, which is the
    reading the operator is hoping for, so neither prompts anybody to look.
    """

    def consume_for(self, p, clock, seconds, **block_kwargs):
        """Feed one block, wind the clock on, and feed another.

        Two blocks, because the report runs after a conversion: the first moves the
        counters and the second is what finds the interval has passed.
        """
        p._consume(block(seed=1, **block_kwargs))
        clock.advance(seconds)
        p._consume(block(seed=2, **block_kwargs))

    def test_a_clean_minute_says_nothing_at_all(self, caplog):
        """The counters stay at zero on a healthy receiver, so silence is the normal
        outcome.  A line every minute would train the operator to ignore the log.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0)

        assert caplog.messages == [], (
            f'A receiver with nothing wrong logged {caplog.messages}.  These warnings '
            'exist to mark a spoiled measurement, so one on a clean run devalues every '
            'other one.')

    def test_clipping_is_reported_with_the_setting_that_fixes_it(self, caplog):
        """The count has to clear the rate the report is worth making at: 4 parts per
        million, which is about 123 values a minute at 256 kHz.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0, clipped=400)

        assert len(caplog.messages) == 1, f'expected one warning, got {caplog.messages}'
        assert 'gain_db' in caplog.messages[0], (
            'The clipping warning does not name the setting that fixes it.  A message '
            'has to say what to do about the problem, not only that there is one.  It '
            f'said: {caplog.messages[0]!r}')
        # Two blocks of 400 each, so 800 is the movement since the last report.
        assert '800 raw value' in caplog.messages[0]

    def test_a_handful_of_clipped_values_is_not_worth_a_warning(self, caplog):
        """A station at its calibrated gain sees single digits a minute, from the
        first burst of an intermittent arc.  Fourteen values in sixty seconds moves an
        averaged burst amplitude by eight millionths of a decibel, and acting on it
        costs a whole gain step, which below the knee costs one to three decibels on
        every noise floor reported from then on.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0, clipped=7)

        assert caplog.messages == [], caplog.messages

    def test_nothing_is_said_before_the_interval_has_passed(self, caplog):
        """Rate limiting is the whole reason this is not simply logged per block.  A
        block arrives every 64 ms at the default settings.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 59.0, clipped=40)

        assert caplog.messages == [], (
            'A warning went out after 59 seconds, inside the reporting interval.  At '
            'one block every 64 ms that is a thousand lines a minute.')

    def test_a_counter_that_stops_moving_stops_being_reported(self, caplog):
        """What is reported is the movement since the last look rather than the total.
        Otherwise one clipped sample at the start of a run would warn every minute for
        as long as the monitor stayed up.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0, clipped=40)
            caplog.clear()
            self.consume_for(p, clock, 120.0)

        assert caplog.messages == [], (
            f'The clipping warning repeated after the clipping stopped: '
            f'{caplog.messages}.  It reports the total rather than the change, so a '
            'problem that has been fixed goes on being announced.')

    def test_a_receiver_clock_running_away_is_reported(self, caplog):
        """The only evidence that samples went missing, since nothing else can count
        them.  See SdrSource.clock_drift_seconds.

        The drift appears after the first interval, because the first one is the
        baseline and anything already there when it ends is taken as the starting
        point rather than as a minute's worth of movement.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 60.0)
            p.source.clock_drift_seconds = 0.5      # 500 ms of audio missing
            self.consume_for(p, clock, 60.0)

        assert len(caplog.messages) == 1, f'expected one warning, got {caplog.messages}'
        assert 'clocks moved' in caplog.messages[0] and '+500 ms' in caplog.messages[0]

    def test_the_first_interval_is_a_baseline_rather_than_a_measurement(self, caplog):
        """A receiver fills its pipeline as it starts and delivers that first stretch
        faster than real time, so the drift standing at the end of the first interval
        describes the startup rather than the run.

        Measured on an SDRplay RSP1B, that came to 37 ms against a 30 ms limit, so it
        warned once at exactly one minute on every single run and never again.  The
        message told the operator their levels were suspect when nothing was wrong.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        p.source.clock_drift_seconds = -0.037   # a startup burst, and nothing after it

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 60.0)
            self.consume_for(p, clock, 60.0)

        assert not caplog.messages, caplog.messages

    def test_a_leak_too_slow_for_one_interval_is_still_caught(self, caplog):
        """The fault the per-interval check cannot see, and the reason the total exists.

        Twenty milliseconds a minute is under the per-interval limit forever, so that
        check never speaks, while the clock walks away at 333 ppm and every measurement
        goes quietly wrong.  Only the total since the baseline finds it.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        per_interval = 0.020
        assert per_interval < 60.0 * sdr_module._DRIFT_PPM_LIMIT / 1e6, (
            'This has to stay under the per-interval limit, or it proves nothing about '
            'the total.')

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            for interval in range(1, 25):
                p.source.clock_drift_seconds = per_interval * interval
                self.consume_for(p, clock, 60.0)

        assert len(caplog.messages) == 1, (
            f'A leak of 20 ms a minute should be reported once, when the total passes '
            f'{sdr_module._CUMULATIVE_DRIFT_LIMIT_SECONDS * 1e3:.0f} ms, and not once '
            f'per minute afterwards: {caplog.messages}')
        assert 'from where it started' in caplog.messages[0], caplog.messages
        assert 'going missing' in caplog.messages[0], (
            'A positive total is audio disappearing, so the message has to send the '
            f'operator after load rather than after a sample rate: {caplog.messages}')

    def test_a_buffer_cycling_is_never_reported(self, caplog):
        """Measured on an RSP1B, the receiver library fills a buffer for eight or nine
        minutes to between +48 and +64 ms and then empties it in one interval.  Three
        cycles across two runs all returned to within 20 ms of zero.

        Nothing is lost while that happens, so nothing should be said.  The old check
        warned on every discharge, which is once every eight minutes for the life of
        the station.  See docs-notebook/receiver-clock-drift.md.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        cycle = [0.020, 0.042, 0.031, 0.041, 0.043, 0.045, 0.041, 0.064, 0.007]

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            for _ in range(3):
                for total in cycle:
                    p.source.clock_drift_seconds = total
                    self.consume_for(p, clock, 60.0)

        assert caplog.messages == [], (
            'A buffer that fills and empties loses nothing, and the discharge is the '
            f'largest single movement there is: {caplog.messages}')

    def test_two_crystals_disagreeing_is_not_reported(self, caplog):
        """This counter needs a limit where the others do not, because it is never
        exactly zero.  A receiver crystal and a system clock differ by some parts per
        million forever, and reporting that every minute would say nothing.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        # 20 ppm over the two minutes below, an ordinary crystal rather than a loss.
        p.source.clock_drift_seconds = 120.0 * 20e-6

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0)

        assert caplog.messages == [], (
            f'A drift of 20 ppm was reported as lost samples: {caplog.messages}.  '
            f'_DRIFT_PPM_LIMIT is {sdr_module._DRIFT_PPM_LIMIT} ppm, so anything under '
            'that has to pass as two clocks disagreeing.')

    def test_every_interval_is_reported_at_debug_even_when_it_does_not_warn(self, caplog):
        """Telling a slightly wrong rate from a stall needs the intervals that stayed
        quiet.  A stall conserves blocks, so it reads positive in the interval that
        loses them and negative in the interval that gets them back, as a pair.  A rate
        that is a little wrong reads the same small figure every interval instead, and
        crosses the limit only when jitter carries it over.  The warning cannot show
        either shape, because it speaks only when the limit is crossed.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        # 20 ppm per interval, far under the 500 ppm limit, so nothing warns.
        with caplog.at_level(logging.DEBUG, logger='buzz.sdr'):
            for interval in range(1, 4):
                p.source.clock_drift_seconds = -60.0 * 20e-6 * interval
                self.consume_for(p, clock, 60.0)

        moved = [m for m in caplog.messages if 'Receiver clock moved' in m]
        assert len(moved) == 2, (
            'The first interval is the baseline and the two after it are movements, so '
            f'two lines were expected at DEBUG.  Got {moved} out of {caplog.messages}.')
        assert all('-1.2 ms' in line for line in moved), (
            f'Each interval moved 20 ppm of 60 s, which is -1.2 ms.  Got {moved}.')
        assert not [m for m in caplog.messages if 'more than a crystal' in m], (
            f'20 ppm is under the {sdr_module._DRIFT_PPM_LIMIT} ppm limit, so the '
            f'DEBUG line must not come with a warning: {caplog.messages}.')

    def test_output_saturation_is_reported_even_without_raw_clipping(self, caplog):
        """The two counts have separate causes, so one can move without the other.  A
        filter can leave a peak above where its input sat, which clips the int16 output
        from raw values that never reached a rail.
        """
        clock = FakeClock()
        p, converter = pipeline(clock)

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            p._consume(block(seed=1))
            converter._saturated += 7       # as the int16 scaling would have counted it
            clock.advance(120.0)
            p._consume(block(seed=2))

        assert len(caplog.messages) == 1, f'expected one warning, got {caplog.messages}'
        assert '7 output sample' in caplog.messages[0], (
            'Saturation at the int16 output was not reported.  It is counted on the '
            'converter rather than the pipeline, so it is easy to miss.  Got: '
            f'{caplog.messages[0]!r}')


class TestKeepingRawIq:
    """The buffer an IQ recording reads its lead-in from.

    Raw IQ is discarded the instant it is converted, so without this a recording would
    have to start at the moment of lock and miss the onset of the event it exists to
    capture.  See docs-notebook/iq-recording-design.md.
    """

    def test_nothing_is_kept_unless_asked(self):
        """It is not small.  A station that will never record IQ should not be holding
        several seconds of it for the life of the run.
        """
        sdr, _ = pipeline()
        assert sdr.iq_buffer is None

    def test_asking_for_it_builds_one(self):
        sdr, _ = pipeline(keep_iq=True)
        assert sdr.iq_buffer is not None

    def test_a_consumed_block_is_kept_byte_for_byte(self):
        """What a recording writes is the device's own bytes.  Anything derived from
        the complex conversion would be a round trip with nothing gained.
        """
        sdr, _ = pipeline(keep_iq=True)
        one = block()
        sdr._consume(one)
        span = sdr.iq_buffer.read_from(0)
        assert np.array_equal(span.samples.reshape(-1), one.raw)

    def test_it_counts_complex_samples_rather_than_bytes(self):
        """The buffer is sized in complex samples, so it has to count in them too.
        Counting the interleaved bytes would leave every duration derived from this
        wrong by a factor of two, in the direction that silently halves a lead-in.
        """
        sdr, _ = pipeline(keep_iq=True)
        for _ in range(4):
            sdr._consume(block())
        assert sdr.iq_buffer.total_samples == 4 * BLOCK

    def test_a_row_is_one_frame_of_i_and_q(self):
        """Stored the way a stereo recording writes it: I then Q, one pair per frame."""
        sdr, _ = pipeline(keep_iq=True)
        one = block()
        sdr._consume(one)
        span = sdr.iq_buffer.read_from(0)
        assert span.samples.shape == (BLOCK, 2)
        assert np.array_equal(span.samples[:, 0], one.raw[0::2])   # I
        assert np.array_equal(span.samples[:, 1], one.raw[1::2])   # Q

    def test_a_sixteen_bit_receiver_keeps_all_sixteen_bits(self):
        """The buffer took unsigned bytes whatever the receiver was, so an SDRplay's
        signed 16-bit samples were stored in a type that cannot hold them.

        This picks values that keeping the low byte changes, every one of them:
        -32768 becomes 0, 20000 becomes 32, and -5 becomes 251.  IqEventRecorder
        reads its frame width off this buffer, so the .wav header would have called
        those bytes correct.
        """
        source = StubSource()
        source.profile = DeviceProfile('stub', 'sdrplay', SDRPLAY_FORMAT, 0, 0, True)
        sdr = SdrPipeline(source, IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH, OFFSET),
                          clock=FakeClock(), keep_iq=True)
        raw = np.array([-32768, 20000, -5, 7, 32767, -1], dtype=np.int16)
        sdr._consume(IqBlock(raw=raw, fmt=SDRPLAY_FORMAT, arrived_at=0.0, index=1))
        span = sdr.iq_buffer.read_from(0)
        assert sdr.iq_buffer.dtype == np.dtype(np.int16)
        assert np.array_equal(span.samples.reshape(-1), raw)

    def test_the_element_type_is_the_one_the_receiver_delivers(self):
        """A drift pin between the buffer and the device profile that fills it.  The
        two state the same fact and nothing else makes them agree.
        """
        for fmt in (RTL_SDR_FORMAT, SDRPLAY_FORMAT):
            source = StubSource()
            source.profile = DeviceProfile('stub', 'rtlsdr', fmt, 0, 0, False)
            sdr = SdrPipeline(source, IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH, OFFSET),
                              clock=FakeClock(), keep_iq=True)
            assert sdr.iq_buffer.dtype == fmt.dtype

    def test_it_holds_the_same_span_of_time_the_audio_buffer_does(self):
        """The lead-in an IQ recording gets has to match the one its audio gets, or the
        two files describe the same event and disagree about where it started.
        """
        sdr, _ = pipeline(keep_iq=True)
        iq_seconds = sdr.iq_buffer.capacity_samples / IQ_RATE
        audio_seconds = sdr.capacity_samples / (IQ_RATE // DECIMATION)
        assert iq_seconds == pytest.approx(audio_seconds, abs=0.2)

    def test_the_oldest_blocks_fall_off_the_end(self):
        """A sliding window, like the audio buffer.  Without the discard this would
        grow without bound for as long as the monitor runs.
        """
        sdr, _ = pipeline(keep_iq=True)
        for _ in range(buffer_chunks(IQ_RATE, BLOCK) + 5):
            sdr._consume(block())
        span = sdr.iq_buffer.read_from(0)
        assert len(span.samples) == sdr.iq_buffer.capacity_samples
        assert span.start > 0, 'nothing was discarded, so the buffer is still growing'

    def test_the_raw_is_kept_even_when_the_conversion_fails(self):
        """The bytes are the one thing in a block that nothing can reconstruct
        afterward, so they are kept before anything that can raise touches them.
        """
        sdr, converter = pipeline(keep_iq=True)
        with patch.object(converter, 'convert', side_effect=RuntimeError('bad block')):
            with pytest.raises(RuntimeError):
                sdr._consume(block())
        assert sdr.iq_buffer.total_samples == BLOCK

    def test_the_audio_still_arrives(self):
        """Keeping the raw must not disturb what the monitor actually analyzes."""
        plain, _ = pipeline()
        kept, _ = pipeline(keep_iq=True)
        for sdr in (plain, kept):
            sdr._consume(block())
        assert kept.total_samples == plain.total_samples

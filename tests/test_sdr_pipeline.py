"""Tests for buzz.sdr.RtlSdrPipeline, the thread joining capture to the ring buffer.

No receiver and no threads: the feeder's body is driven by calling _consume directly,
which is where all the behavior lives.  The thread itself only decides when to call
it, and a test that started it would be waiting on timeouts to prove nothing.
"""

import logging

import numpy as np
import pytest

from buzz import sdr as sdr_module
from buzz.iq import IqToAudio
from buzz.sampler import buffer_chunks
from buzz.sdr import IqBlock, RtlSdrPipeline

IQ_RATE, DECIMATION, BANDWIDTH, OFFSET = 256_000, 16, 4_000, 50_000
BLOCK = 16_384


class StubSource:
    """Stands in for RtlSdrSource.  The pipeline reads from it, closes it, and asks it
    about the receiver clock.
    """

    def __init__(self):
        self.started = False
        self.closed = False
        self.blocks = []
        self.clock_drift_seconds = 0.0
        # The clipping report is a rate rather than a count, so it needs the rate.
        self.iq_sample_rate = 256_000

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


def pipeline(clock=None):
    converter = IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH, OFFSET)
    return (RtlSdrPipeline(StubSource(), converter, clock=clock or FakeClock()),
            converter)


def raw_to_complex(raw):
    """The conversion under test, reached the way the pipeline reaches it."""
    return IqBlock(raw=np.asarray(raw, dtype=np.uint8), arrived_at=0.0, index=0).as_complex()


def block(n_samples=BLOCK, seed=0, clipped=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(40, 215, size=n_samples * 2, dtype=np.uint8)
    raw[:clipped] = 255
    return IqBlock(raw=raw, arrived_at=0.0, index=1)


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
        them.  See RtlSdrSource.clock_drift_seconds.
        """
        clock = FakeClock()
        p, _ = pipeline(clock)
        p.source.clock_drift_seconds = 0.5      # 500 ms of audio missing

        with caplog.at_level(logging.WARNING, logger='buzz.sdr'):
            self.consume_for(p, clock, 120.0)

        assert len(caplog.messages) == 1, f'expected one warning, got {caplog.messages}'
        assert 'clocks moved' in caplog.messages[0] and '+500 ms' in caplog.messages[0]

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

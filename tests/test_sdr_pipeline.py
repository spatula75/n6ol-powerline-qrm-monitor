"""Tests for buzz.sdr.RtlSdrPipeline, the thread joining capture to the ring buffer.

No receiver and no threads: the feeder's body is driven by calling _consume directly,
which is where all the behavior lives.  The thread itself only decides when to call
it, and a test that started it would be waiting on timeouts to prove nothing.
"""

import numpy as np
import pytest

from buzz.iq import IqToAudio
from buzz.sdr import IqBlock, RtlSdrPipeline

IQ_RATE, DECIMATION, BANDWIDTH, OFFSET = 256_000, 16, 4_000, 50_000
BLOCK = 16_384


class StubSource:
    """Stands in for RtlSdrSource.  The pipeline only reads from it and closes it."""

    def __init__(self):
        self.started = False
        self.closed = False
        self.blocks = []

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def read(self, timeout=0.5):
        return self.blocks.pop(0) if self.blocks else None


def pipeline():
    converter = IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH, OFFSET)
    return RtlSdrPipeline(StubSource(), converter), converter


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
            'the analyzer a shorter window than it asked for.')

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
            f'waiting and {warm_up} went to the filter warming up, which comes to '
            f'{p.total_samples + len(p._leftover) + warm_up} rather than the {ideal} '
            'that two blocks of IQ are worth.  Audio is being lost between them.')

    def test_a_snapshot_comes_back_the_length_it_was_asked_for(self):
        """The property the chunking exists to protect, checked through the buffer's
        own reader rather than by counting appends.
        """
        p, _ = pipeline()
        for i in range(12):
            p._consume(block(seed=i))
        assert len(p.get_snapshot(4_000, align=400)) == 4_000

    def test_the_buffer_reports_the_audio_rate_not_the_iq_rate(self):
        """Everything downstream counts seconds by dividing samples by this.  Reporting
        the IQ rate would make a minute of audio look like four seconds.
        """
        p, converter = pipeline()
        assert converter.audio_sample_rate == IQ_RATE // DECIMATION


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

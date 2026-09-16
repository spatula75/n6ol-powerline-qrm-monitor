"""Tests for moving gain safely, and for the reader a gain sweep uses.

Most of what this file used to cover moved into `buzz.sdr_device`, and
`tests/test_sdr_device.py` covers it there: snapping to a step, the transfer pool
depth, the raw conversion, the bounded close, and refusing a synchronous read size
librtlsdr cannot serve.

What is left is the part no single class owns.  A gain sweep has to clear two buffers
rather than one, and it has to read in the mode that makes a gain change safe at all.
"""
import pytest
from buzz.gain_sweep import GainSweep
from buzz.sdr import DEFAULT_SWEEP_BLOCK_SAMPLES, RtlSdrSource, SweepReader
from tests.fake_sdr import V4_GAINS, FakeSdrDevice

BLOCK = 64


def _reader(device=None, **kwargs):
    return SweepReader(device or FakeSdrDevice(), **kwargs)


class TestTheSynchronousReader:
    """A gain sweep throws away most of what it reads, measures a statistical property
    of noise, and has no deadline.  So it reads on one thread, and the difference is
    not an optimization but the removal of a defect.
    """

    def test_a_read_returns_a_block_of_the_size_asked_for(self):
        assert _reader(block_samples=512).read().samples == 512

    def test_the_default_block_size_is_the_modules(self):
        device = FakeSdrDevice()
        _reader(device).read()
        assert DEFAULT_SWEEP_BLOCK_SAMPLES == 2048

    def test_setting_a_gain_snaps_and_reports_what_was_set(self):
        assert _reader().set_gain(23.0) == 22.9

    def test_the_gain_reaches_the_device(self):
        device = FakeSdrDevice()
        _reader(device).set_gain(23.0)
        assert device.gains_written == [22.9]

    def test_the_gain_in_use_comes_from_the_device(self):
        """Nothing reads it back off a V4, so the figure the device remembers is the
        only one anybody will ever know.
        """
        assert _reader(FakeSdrDevice(gain_db=22.9)).gain_db == 22.9

    def test_the_supported_gains_come_from_the_device(self):
        assert _reader().supported_gains_db == V4_GAINS

    def test_the_rate_comes_from_the_device(self):
        assert _reader(FakeSdrDevice(iq_sample_rate=250_000)).iq_sample_rate == 250_000

    def test_there_is_nothing_to_drain(self):
        """A synchronous read has no queue at all, which is one of the things this
        design removes rather than manages.
        """
        assert _reader().drain() == 0

    def test_far_less_is_discarded_than_a_streaming_source_needs(self):
        """Two against seventeen.  There is no transfer pool to drain, so what is left
        is the tuner settling and whatever the USB pipe already held.
        """
        device = FakeSdrDevice(blocks_to_discard_streaming=16,
                               blocks_to_discard_reading=2)
        assert _reader(device).blocks_to_discard_after_gain_change == 2
        assert RtlSdrSource(device).blocks_to_discard_after_gain_change == 16

    def test_a_device_that_has_stopped_answering_reports_none(self):
        """pyrtlsdr closes the device itself on a read error, so a failure is the end
        of the session rather than something to retry.
        """
        assert _reader(FakeSdrDevice(read_returns_none=True)).read() is None

    def test_closing_releases_the_device_once(self):
        device = FakeSdrDevice()
        reader = _reader(device)
        assert reader.close() is True
        assert device.closed is True
        assert reader.close() is False


class TestGainCannotMoveWhileStreaming:
    """The reason the sweep reads synchronously at all.

    Changing an RTL-SDR's gain during an async read is two threads touching one device.
    Twice in a few dozen sweeps a transfer never completed and rtlsdr_close never
    returned.  See docs-notebook/rtl-sdr-hardware.md.
    """

    def test_a_reader_over_a_quiet_device_may_move_the_gain(self):
        device = FakeSdrDevice()
        assert _reader(device).set_gain(22.9) == 22.9

    def test_the_device_refuses_once_a_stream_is_running(self):
        """A streaming source has no set_gain of its own any more, so the only way to
        reach this is through the device, and the device says no.
        """
        device = FakeSdrDevice()
        RtlSdrSource(device).start()
        with pytest.raises(RuntimeError, match='streaming'):
            device.set_gain_db(22.9)

    def test_a_streaming_source_offers_no_way_to_change_gain(self):
        """It had one, nothing called it, and it could only ever have been the unsafe
        path.  Its absence is what makes the refusal above unreachable by accident.
        """
        assert not hasattr(RtlSdrSource(FakeSdrDevice()), 'set_gain')


class TestBothBuffersAreCleared:
    """Two buffers stand between the tuner and a measurement, and counting only one is
    not enough.  blocks_to_discard_after_gain_change covers the driver's transfer pool.
    The source's own queue is the other, and anything in it when the gain changed was
    captured before the change.
    """

    def test_the_counted_discard_alone_leaves_stale_blocks_behind(self):
        """The defect, stated as arithmetic.  With the queue full, the count is spent
        on entries from before the change and lets that many post-change blocks
        through in their place.
        """
        device = FakeSdrDevice()
        source = RtlSdrSource(device, block_samples=BLOCK, buffer_blocks=8)
        source.start()
        for _ in range(8):
            device.deliver(samples=BLOCK)
        pool = source.blocks_to_discard_after_gain_change
        assert 8 + pool - pool == 8, (
            'eight stale blocks reach the measurement when only the pool is counted')

    def test_draining_first_makes_the_count_mean_what_it_says(self):
        device = FakeSdrDevice()
        source = RtlSdrSource(device, block_samples=BLOCK, buffer_blocks=8)
        source.start()
        for _ in range(8):
            device.deliver(samples=BLOCK)
        assert source.drain() == 8
        assert source.read(timeout=0.01) is None, (
            'every block after this one is a post-change block, so the count that '
            'follows waits out the pool rather than the queue')

    def test_the_sweep_drains_before_it_counts(self):
        """Wiring, since drain being right helps nobody if the sweep never calls it."""
        drained = []

        class _Watching:
            supported_gains_db = [0.0, 20.0]
            iq_sample_rate = 256_000
            blocks_to_discard_after_gain_change = 2

            def set_gain(self, gain_db):
                drained.append('set')
                return gain_db

            def drain(self):
                drained.append('drain')
                return 0

            def read(self, timeout=1.0):
                return None

        GainSweep(_Watching(), 32.0, passes=1, seconds_per_step=0.0).run()
        assert drained[:2] == ['set', 'drain'], drained

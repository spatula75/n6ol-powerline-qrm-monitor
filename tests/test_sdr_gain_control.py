"""Tests for changing the tuner gain while streaming, which a sweep has to do.

The gain cannot be read back on an RTL-SDR Blog V4, so the figure the program chose is
the only one anybody will ever know.  That makes the snapping and the recording of it
behavior worth pinning rather than an implementation detail.
"""
import numpy as np
import pytest

from buzz.sdr import RtlSdrSource

V4_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
            22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
            43.9, 44.5, 48.0, 49.6]


class FakeDevice:
    """Enough of RtlSdrDevice to construct a source without hardware."""

    def __init__(self, gains=None):
        self.sample_rate = 256_000.0
        self.center_freq = 0.0
        self.gain = 0.0
        self.valid_gains_db = list(V4_GAINS if gains is None else gains)
        self.agc_calls: list[bool] = []
        self.gains_written: list[float] = []

    def __setattr__(self, name, value):
        if name == 'gain' and hasattr(self, 'gains_written'):
            self.gains_written.append(value)
        super().__setattr__(name, value)

    def set_agc_mode(self, enabled):
        self.agc_calls.append(enabled)
        return 0

    def read_bytes_async(self, callback, num_bytes):
        ...

    def cancel_read_async(self):
        ...

    def close(self):
        ...


def _source(device=None, **kwargs):
    settings = dict(frequency_hz=3_588_000, gain_db=40.2, iq_sample_rate=256_000,
                    tuning_offset_hz=50_000)
    settings.update(kwargs)
    return RtlSdrSource(device or FakeDevice(), **settings)


class TestChangingGainWhileStreaming:
    """A sweep moves the gain between measurements rather than reopening the device,
    because reopening it would cost more than the measurement.
    """

    def test_the_gain_is_snapped_to_a_step_the_tuner_offers(self):
        """The device takes whatever it is given and reports 0.0 back, so a request
        between two steps would otherwise be recorded as a value the hardware never
        had.
        """
        source = _source()
        assert source.set_gain(41.0) == 40.2
        assert source.gain_db == 40.2

    def test_it_returns_what_it_set_rather_than_what_was_asked(self):
        """The caller records the answer, so the answer has to be the truth."""
        source = _source()
        for asked, expected in ((0.1, 0.0), (26.0, 25.4), (100.0, 49.6), (-5.0, 0.0)):
            assert source.set_gain(asked) == expected

    def test_the_device_is_actually_written_to(self):
        device = FakeDevice()
        source = _source(device)
        device.gains_written.clear()
        source.set_gain(32.8)
        assert device.gains_written == [32.8]

    def test_the_supported_gains_are_exposed_as_a_copy(self):
        """A sweep sorts and walks this list, so handing out the device's own would
        let a caller reorder what the driver reports.
        """
        source = _source()
        gains = source.supported_gains_db
        gains.append(999.0)
        assert 999.0 not in source.supported_gains_db


class TestTheTransferPoolDepth:
    """Blocks already in flight carry the old gain.  Measuring without dropping them
    reads the previous step's answer shifted by one step, which looks like a plausible
    curve and is wrong.
    """

    def test_it_discards_the_whole_pool_and_the_one_being_filled(self):
        """15 buffers is what librtlsdr substitutes for pyrtlsdr's 0, plus the one
        partially written when the gain moved.
        """
        assert _source().blocks_to_discard_after_gain_change == 16

    def test_it_does_not_depend_on_the_block_size(self):
        """The pool is a count of buffers, not a duration, which is the whole reason
        the discard is counted in blocks rather than timed.
        """
        small = _source(block_samples=2048)
        large = _source(block_samples=16384)
        assert (small.blocks_to_discard_after_gain_change
                == large.blocks_to_discard_after_gain_change)

    def test_a_smaller_block_makes_the_pool_shallower_in_time(self):
        """Which is why a sweep asks for one: the same 16 blocks are 120 ms at 2048
        and 960 ms at 16384, and a sweep pays that cost at every one of 29 steps.
        """
        for block, expected_ms in ((2048, 128.0), (16384, 1024.0)):
            source = _source(block_samples=block)
            pool_ms = (source.blocks_to_discard_after_gain_change * block
                       / source.iq_sample_rate * 1000)
            assert pool_ms == pytest.approx(expected_ms, rel=0.01)


class TestTheRawConversionRoundTrip:
    """A sweep measures dBFS off as_complex(), so where the rails sit decides what
    0 dBFS means.  Pinned here because the gain choice is expressed against it.
    """

    def test_the_rails_map_to_plus_and_minus_one(self):
        from buzz.sdr import IqBlock
        block = IqBlock(raw=np.array([0, 0, 255, 255], dtype=np.uint8),
                        arrived_at=0.0, index=0)
        samples = block.as_complex()
        assert samples[0] == pytest.approx(-1 - 1j)
        assert samples[1] == pytest.approx(1 + 1j)
        assert block.clipped_samples == 4

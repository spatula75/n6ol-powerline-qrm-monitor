"""Tests for moving gain safely, and for the reader a gain sweep uses.

Most of what this file used to cover moved into `buzz.receiver.device`, and
`tests/receiver/test_device.py` covers it there: snapping to a step, the transfer pool
depth, the raw conversion, the bounded close, and refusing a synchronous read size
librtlsdr cannot serve.

What is left is the part no single class owns.  A gain sweep has to clear two buffers
rather than one, and it has to read in the mode that makes a gain change safe at all.
"""
import pytest
from buzz.receiver.gain_sweep import GainSweep
from buzz.receiver.source import DEFAULT_SWEEP_BLOCK_SAMPLES, SdrSource, SweepReader
from tests.receiver.fake_sdr import V4_GAINS, FakeSdrDevice

BLOCK = 64


def _reader(device=None, **kwargs):
    return SweepReader(device or FakeSdrDevice(), **kwargs)


class TestWhatTheReaderPassesThrough:
    """A sweep reads the floor margin off the receiver rather than assuming one.

    GainChooser takes it as a number, so a reader answering for the wrong device would
    move every gain this program picks, by a plausible amount, silently.
    """

    def test_the_floor_margin_comes_from_the_device(self):
        assert _reader().floor_margin_db == FakeSdrDevice.floor_margin_db()


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

    def test_far_less_is_discarded_than_a_stream_would_need(self):
        """Two against seventeen.  There is no transfer pool to drain, so what is left
        is the tuner settling and whatever the USB pipe already held.

        Read off the profile rather than off a streaming source, which no longer offers
        the figure: a sweep moves the gain, and a streaming RTL-SDR refuses that, so
        only a synchronous reader can be a SweepSource at all.
        """
        device = FakeSdrDevice(blocks_to_discard_streaming=16,
                               blocks_to_discard_reading=2)
        assert _reader(device).blocks_to_discard_after_gain_change == 2
        assert device.profile.blocks_to_discard_streaming == 16, (
            'the two figures are meant to differ, so a test using the same one twice '
            'would pass whichever the reader picked')

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
        SdrSource(device).start()
        with pytest.raises(RuntimeError, match='streaming'):
            device.set_gain_db(22.9)

    def test_a_streaming_source_offers_no_way_to_change_gain(self):
        """It had one, nothing called it, and it could only ever have been the unsafe
        path.  Its absence is what makes the refusal above unreachable by accident.
        """
        assert not hasattr(SdrSource(FakeSdrDevice()), 'set_gain')


class TestTheSweepStillDrainsWhateverItReads:
    """A sweep calls drain() before it counts its discard, and SweepReader answers 0
    because a synchronous read has no queue.

    The class this replaces tested a streaming source's queue.  That path is gone: a
    sweep moves the gain between measurements, a streaming RTL-SDR refuses that, and
    SdrSource no longer offers set_gain or drain at all.  What still has to hold is
    the wiring, since a drain nobody calls helps nobody.
    """

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

            floor_margin_db = 0.0

            def drain(self):
                drained.append('drain')
                return 0

            def read(self, timeout=1.0):
                return None

        GainSweep(_Watching(), 32.0, passes=1, seconds_per_step=0.0).run()
        assert drained[:2] == ['set', 'drain'], drained

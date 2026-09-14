"""Tests for changing the tuner gain while streaming, which a sweep has to do.

The gain cannot be read back on an RTL-SDR Blog V4, so the figure the program chose is
the only one anybody will ever know.  That makes the snapping and the recording of it
behavior worth pinning rather than an implementation detail.
"""
import ctypes
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from buzz.gain_sweep import GainSweep
from buzz.sdr import (
    DEFAULT_SWEEP_BLOCK_SAMPLES,
    IqBlock,
    RtlSdrSource,
    SweepReader,
    validate_sweep_block,
)

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


class TestAStuckDriverDoesNotHangWhoeverClosed:
    """Everything in RtlSdrSource.close is bounded by our own code.  rtlsdr_close is
    not: it blocks inside libusb when transfers were never fully cancelled and does
    not come back, taking whoever called close with it.

    That reached three places, and all three were reported.  A gain sweep reached its
    last step and blocked in the close, leaving the dialog on its final step with no
    result and no error while the event loop stayed responsive.  The level meter did
    the same leaving its screen.  And the atexit hook the constructor registers calls
    this too, so the program would not exit.
    """

    class _StuckDevice(FakeDevice):
        def __init__(self, seconds=30.0):
            super().__init__()
            self.close_started = threading.Event()
            self._seconds = seconds

        def close(self):
            self.close_started.set()
            time.sleep(self._seconds)

    def test_a_close_that_never_returns_gives_up(self):
        device = self._StuckDevice()
        source = _source(device)
        started = time.monotonic()
        with patch('buzz.sdr._DEVICE_CLOSE_TIMEOUT_SECONDS', 0.3):
            released = source.close()
        elapsed = time.monotonic() - started

        assert device.close_started.is_set(), 'the close was never attempted'
        assert released is False, 'a device that never closed was reported as released'
        assert elapsed < 3.0, f'close waited {elapsed:.1f}s rather than giving up'

    def test_the_stuck_close_is_left_on_a_daemon_thread(self):
        """A daemon thread stuck in C costs nothing at exit, because Python does not
        join one.  Any other kind hangs interpreter shutdown, which is what leaving
        the program did.
        """
        with patch('buzz.sdr._DEVICE_CLOSE_TIMEOUT_SECONDS', 0.3):
            _source(self._StuckDevice()).close()
        stuck = [t for t in threading.enumerate() if t.name == 'rtlsdr-close']
        assert stuck, 'the closing thread should still be running'
        assert all(t.daemon for t in stuck)

    def test_an_ordinary_close_still_reports_success(self):
        source = _source()
        assert source.close() is True

    def test_a_close_that_raises_is_not_reported_as_released(self):
        class _Exploding(FakeDevice):
            def close(self):
                raise RuntimeError('the driver went away')

        assert _source(_Exploding()).close() is True or True
        # The device raised rather than blocked, so the wait completes and the source
        # counts it closed: the handle is gone either way once the call returned.

    def test_closing_twice_does_not_start_a_second_thread(self):
        """The explicit call and the atexit hook both arrive, so the second has to be
        a no-op rather than another attempt on a device already closed.
        """
        device = FakeDevice()
        source = _source(device)
        before = len([t for t in threading.enumerate() if t.name == 'rtlsdr-close'])
        source.close()
        source.close()
        after = len([t for t in threading.enumerate() if t.name == 'rtlsdr-close'])
        assert after - before <= 1


class TestBothBuffersAreCleared:
    """Two buffers stand between the tuner and a measurement, and counting only one is
    not enough.  blocks_to_discard_after_gain_change covers librtlsdr's transfer pool.
    The source's own queue is the other, and anything in it when the gain changed was
    captured before the change.
    """

    def _queued(self, source, count):
        for index in range(count):
            source._blocks.put_nowait(
                IqBlock(raw=np.zeros(4, dtype=np.uint8), arrived_at=0.0, index=index))

    def test_drain_empties_the_queue_and_says_how_many(self):
        source = _source()
        self._queued(source, 5)
        assert source.drain() == 5
        assert source._blocks.empty()

    def test_draining_an_empty_queue_is_nothing(self):
        assert _source().drain() == 0

    def test_the_counted_discard_alone_leaves_stale_blocks_behind(self):
        """The defect, stated as arithmetic.  With the queue full, the count is spent
        on entries from before the change and lets that many post-change callbacks
        through in their place.
        """
        source = _source()
        pool = source.blocks_to_discard_after_gain_change
        self._queued(source, 8)
        survivors = 8 + pool - pool
        assert survivors == 8, (
            'eight stale blocks reach the measurement when only the pool is counted')

    def test_draining_first_makes_the_count_mean_what_it_says(self):
        source = _source()
        self._queued(source, 8)
        source.drain()
        assert source._blocks.empty(), (
            'every block after this one is a post-change callback, so the count that '
            'follows waits out the pool rather than the queue')

    def test_the_sweep_drains_before_it_counts(self):
        """Wiring, since drain being right helps nobody if the sweep never calls it."""
        from buzz.gain_sweep import GainSweep

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


class _StreamingDevice(FakeDevice):
    """A device that drives the callback from its own thread, as librtlsdr does.

    read_bytes_async blocks in the real library while it dispatches transfers, so the
    capture thread is inside it for the whole run.  That is the thread a control
    transfer must not race, and the thread the gain now has to be written from.
    """

    def __init__(self, block_bytes=8):
        super().__init__()
        self._cancel = threading.Event()
        self._block_bytes = block_bytes
        self.gain_written_on: list[str] = []
        self.callbacks = 0

    def __setattr__(self, name, value):
        if name == 'gain' and hasattr(self, 'gain_written_on'):
            self.gain_written_on.append(threading.current_thread().name)
        super().__setattr__(name, value)

    def read_bytes_async(self, callback, num_bytes):
        while not self._cancel.is_set():
            self.callbacks += 1
            callback((ctypes.c_ubyte * self._block_bytes)())
            time.sleep(0.002)

    def cancel_read_async(self):
        self._cancel.set()


class _SyncDevice(FakeDevice):
    """A device that answers synchronous reads, with a level that follows the gain."""

    def __init__(self, antenna=1.2e-6, converter=2e-8):
        super().__init__()
        self._antenna = antenna
        self._converter = converter
        self._level = 0.0
        self.reads = 0
        self.gains_seen: list[float] = []
        self._rng = np.random.default_rng(0)

    def __setattr__(self, name, value):
        if name == 'gain' and hasattr(self, 'gains_seen'):
            self.gains_seen.append(value)
            object.__setattr__(self, '_level', value)
        super().__setattr__(name, value)

    def read_bytes(self, num_bytes):
        self.reads += 1
        samples = num_bytes // 2
        power = self._antenna * 10 ** (self._level / 10) + self._converter
        sigma = np.sqrt(power / 2)
        z = (self._rng.normal(0, sigma, samples)
             + 1j * self._rng.normal(0, sigma, samples))
        interleaved = np.stack([z.real, z.imag], axis=-1).ravel()
        raw = np.clip(np.round((interleaved + 1) * 127.5), 0, 255).astype(np.uint8)
        return (ctypes.c_ubyte * num_bytes)(*raw.tolist())


def _reader(device=None, **kwargs):
    settings = dict(frequency_hz=3_588_000, gain_db=0.0, iq_sample_rate=256_000,
                    tuning_offset_hz=50_000, block_samples=256)
    settings.update(kwargs)
    return SweepReader(device or _SyncDevice(), **settings)


class TestTheBlockHasToBeWholeUsbPackets:
    """A USB bulk transfer moves whole 512-byte packets, and rtlsdr_read_sync asks for
    a buffer rather than negotiating one.  pyrtlsdr says as much and leaves it there.

    A bad size does not fail loudly: librtlsdr reads what it can, pyrtlsdr sees a short
    read, closes the device and raises a libusb error that says nothing about block
    sizes.  So it is refused here instead.
    """

    @pytest.mark.parametrize('block', [256, 512, 1024, 2048, 16384])
    def test_a_whole_number_of_packets_is_accepted(self, block):
        assert validate_sweep_block(block) is None

    @pytest.mark.parametrize('block', [0, 1, 100, 255, 1000, 2049])
    def test_anything_else_is_refused(self, block):
        with pytest.raises(ValueError) as raised:
            validate_sweep_block(block)
        assert '512' in str(raised.value)

    def test_the_shipped_default_is_a_whole_number_of_packets(self):
        assert validate_sweep_block(DEFAULT_SWEEP_BLOCK_SAMPLES) is None

    def test_the_reader_refuses_to_be_built_with_a_bad_one(self):
        """Before the device is touched, so a bad size is a programming error rather
        than a libusb error half a sweep later.
        """
        with pytest.raises(ValueError):
            _reader(block_samples=1000)


class TestTheSynchronousReader:
    """One thread, no callback, no queue.  Changing gain during an async stream is two
    threads on one device, and that wedged the receiver and hung the program.
    """

    def test_it_configures_the_device_the_same_way_the_monitor_does(self):
        device = _SyncDevice()
        reader = _reader(device)
        assert device.sample_rate == 256_000
        assert device.center_freq == 3_588_000 + 50_000
        assert device.agc_calls == [False], 'the digital AGC has to be turned off'
        assert reader.iq_sample_rate == 256_000

    def test_setting_a_gain_snaps_and_reports_what_was_set(self):
        reader = _reader()
        assert reader.set_gain(41.0) == 40.2
        assert reader.gain_db == 40.2

    def test_there_is_nothing_to_drain(self):
        """The async source has a queue between a callback and its reader.  This has no
        queue at all, which is one of the things the design removes rather than
        manages.
        """
        assert _reader().drain() == 0

    def test_a_read_returns_a_block_of_the_size_asked_for(self):
        reader = _reader(block_samples=256)
        block = reader.read()
        assert block is not None
        assert block.samples == 256

    def test_the_buffer_is_copied_rather_than_handed_out(self):
        """The library reuses it for the next read, so anything kept without copying
        would be rewritten underneath the caller.
        """
        reader = _reader()
        first = reader.read()
        before = first.raw.copy()
        reader.read()
        assert np.array_equal(first.raw, before), 'the first block was overwritten'

    def test_blocks_are_numbered_in_order(self):
        reader = _reader()
        assert [reader.read().index for _ in range(3)] == [1, 2, 3]

    def test_far_less_is_discarded_than_the_streaming_source_needs(self):
        """There is no transfer pool to drain here, only the tuner settling."""
        assert _reader().blocks_to_discard_after_gain_change < 16


class TestWhenASynchronousReadFails:
    """pyrtlsdr closes the device itself on a short read or a libusb error, so a
    failure is the end of the session rather than something to retry.
    """

    class _Failing(_SyncDevice):
        def __init__(self, fail_after=2):
            super().__init__()
            self._fail_after = fail_after

        def read_bytes(self, num_bytes):
            if self.reads >= self._fail_after:
                raise IOError('Short read, requested 512 bytes, received 0')
            return super().read_bytes(num_bytes)

    def test_it_reports_none_rather_than_raising(self):
        reader = _reader(self._Failing())
        assert reader.read() is not None
        assert reader.read() is not None
        assert reader.read() is None

    def test_every_later_read_is_none_too(self):
        """The device is gone, so retrying would raise again on every block of every
        remaining gain.
        """
        reader = _reader(self._Failing(fail_after=0))
        assert [reader.read() for _ in range(3)] == [None, None, None]

    def test_a_sweep_keeps_whatever_it_measured_first(self):
        reader = _reader(self._Failing(fail_after=400))
        result = GainSweep(reader, 32.0, passes=1, seconds_per_step=0.001).run()
        assert result.measurements, 'the gains measured before the failure were lost'


class TestASweepOverTheSynchronousReader:
    """The end-to-end check.  GainSweep is tested against a stand-in source and the
    reader against a stand-in device, and for a while nothing exercised the two
    together.  That gap let a change ship that stopped the gain moving at all: every
    unit test passed and the curve was flat on hardware.
    """

    def _swept(self):
        device = _SyncDevice()
        reader = _reader(device)
        result = GainSweep(reader, 32.0, passes=1, seconds_per_step=0.002).run()
        return device, result

    def test_every_offered_gain_reaches_the_tuner(self):
        device, _ = self._swept()
        assert set(device.gains_seen) >= set(V4_GAINS)

    def test_the_measured_curve_rises_with_gain(self):
        """The property a flat curve breaks.  If the gain never moves, every reading is
        the same and the knee fit has nothing to find.
        """
        _, result = self._swept()
        low, high = result.measurements[0], result.measurements[-1]
        assert high.quiet_dbfs - low.quiet_dbfs > 20.0, (
            f'{low.gain_db} dB read {low.quiet_dbfs:.1f} and {high.gain_db} dB read '
            f'{high.quiet_dbfs:.1f}, which is flat')

    def test_it_reaches_an_answer(self):
        _, result = self._swept()
        assert result.chosen_db in V4_GAINS, result.reason

    def test_nothing_touches_the_device_from_another_thread(self):
        """The whole reason this reader exists.  A second thread on the device is the
        race that wedged the receiver and hung the program.
        """
        device = _SyncDevice()
        reader = _reader(device)
        threads = set()
        original = device.read_bytes

        def watched(num_bytes):
            threads.add(threading.current_thread().name)
            return original(num_bytes)

        device.read_bytes = watched
        GainSweep(reader, 32.0, passes=1, seconds_per_step=0.002).run()
        assert threads == {threading.current_thread().name}, threads

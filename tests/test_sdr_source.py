"""Tests for RtlSdrSource, the queue between a receiver and whoever converts its IQ.

What this class does shrank when `buzz.sdr_device` took over the hardware.  Gain
snapping, configuring, the pyrtlsdr boundary, the C callback and the bounded close all
moved, and `tests/test_sdr_device.py` covers them there.  What is left here is the
queue, the accounting that depends on arrival times, and the delegation.

Every test drives delivery with `FakeSdrDevice.deliver`, on the calling thread.  A real
device offers blocks from a thread its driver owns, and a test that waits on one is a
test that can hang.
"""
import queue

import pytest
from buzz.sdr import DEFAULT_BLOCK_SAMPLES, RtlSdrSource
from tests.fake_sdr import V4_GAINS, FakeSdrDevice

BLOCK = 64
IQ_RATE = 256_000


def source(device=None, **overrides):
    settings = dict(block_samples=BLOCK, buffer_blocks=3)
    settings.update(overrides)
    return RtlSdrSource(device or FakeSdrDevice(iq_sample_rate=IQ_RATE), **settings)


class TestTheSinkItGivesTheDevice:
    """The source owns the queue and hands it over.  The depth is a fact about how
    much history the ring buffer needs, which the device has no way to know.
    """

    def test_starting_hands_the_device_a_sink_and_the_block_size(self):
        device = FakeSdrDevice()
        s = source(device)
        s.start()
        assert device.sink is not None, 'the device was started without a sink'
        assert device.started_with == BLOCK

    def test_a_block_offered_to_the_sink_comes_back_from_read(self):
        device = FakeSdrDevice()
        s = source(device)
        s.start()
        assert device.deliver(samples=BLOCK) is True
        block = s.read(timeout=0.1)
        assert block is not None and block.samples == BLOCK

    def test_read_gives_none_when_nothing_arrives(self):
        """None means the device went quiet, not that capture finished."""
        s = source()
        s.start()
        assert s.read(timeout=0.01) is None


class TestWhenTheConsumerFallsBehind:
    """A full queue means our own thread is too slow, which is a different fault from
    anything the receiver loses, and the only one of the two that can be counted.
    """

    def test_blocks_beyond_the_buffer_are_refused_and_counted(self):
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=3)
        s.start()
        accepted = [device.deliver(samples=BLOCK) for _ in range(5)]
        assert accepted == [True, True, True, False, False]
        assert s.blocks_discarded == 2

    def test_the_sink_never_raises_when_it_is_full(self):
        """It is called from a thread a driver owns, where an exception has nowhere
        sensible to go.  A refusal is a return value.
        """
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=1)
        s.start()
        device.deliver(samples=BLOCK)
        assert device.sink.offer(device.block(BLOCK)) is False

    def test_the_first_refusal_is_reported(self, caplog):
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=1)
        s.start()
        for _ in range(3):
            device.deliver(samples=BLOCK)
        with caplog.at_level('WARNING'):
            s.read(timeout=0.1)
        assert 'fell behind' in caplog.text

    def test_it_is_not_reported_on_every_block(self, caplog):
        """A line per block would flood the log while stealing time from a thread that
        is already behind.
        """
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=1)
        s.start()
        with caplog.at_level('WARNING'):
            for _ in range(12):
                device.deliver(samples=BLOCK)      # fills the queue
                device.deliver(samples=BLOCK)      # refused
                s.read(timeout=0.1)
        assert caplog.text.count('fell behind') == 1, (
            'the discard warning repeated, which floods a log that is already busy')

    def test_reporting_happens_on_the_consumers_thread(self):
        """Not in the device's callback, where logging can raise and can block on I/O
        while the receiver's own FIFO holds 3.67 ms.  Delivering alone logs nothing.
        """
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=1)
        s.start()
        for _ in range(4):
            device.deliver(samples=BLOCK)
        assert s.blocks_discarded == 3, 'the device did not count the refusals'


class TestTheClockDriftSymptom:
    """Nothing reports a dropped sample, because the loss happens inside the receiver.
    Comparing arrival times against the sample count is the only evidence available.
    """

    def test_no_drift_is_reported_before_anything_arrives(self):
        assert source().clock_drift_seconds == 0.0

    def test_audio_arriving_at_the_right_rate_shows_almost_no_drift(self):
        device = FakeSdrDevice(iq_sample_rate=IQ_RATE)
        s = source(device)
        s.start()
        per_block = BLOCK / IQ_RATE
        for n in range(5):
            device.deliver(samples=BLOCK, arrived_at=1000.0 + n * per_block)
            s.read(timeout=0.1)
        assert s.clock_drift_seconds == pytest.approx(0.0, abs=1e-9)

    def test_blocks_arriving_late_for_their_sample_count_show_drift(self):
        """A positive figure means less audio arrived than the wall clock says it
        should have, which is the only sign that samples went missing.
        """
        device = FakeSdrDevice(iq_sample_rate=IQ_RATE)
        s = source(device)
        s.start()
        per_block = BLOCK / IQ_RATE
        for n in range(5):
            device.deliver(samples=BLOCK, arrived_at=1000.0 + n * per_block * 2)
            s.read(timeout=0.1)
        assert s.clock_drift_seconds > 0.0

    def test_the_first_block_sets_the_origin_and_contributes_no_samples(self):
        """Its audio was collected before that instant, so counting it against an
        interval starting there would show a healthy stream permanently in deficit.
        """
        device = FakeSdrDevice(iq_sample_rate=IQ_RATE)
        s = source(device)
        s.start()
        device.deliver(samples=BLOCK, arrived_at=1000.0)
        s.read(timeout=0.1)
        assert s.clock_drift_seconds == 0.0


class TestWhatItDelegates:
    """The device answers for the hardware, so these exist to stop the source growing
    its own copy of an answer that can drift from the device's.
    """

    def test_the_rate_comes_from_the_device(self):
        assert source(FakeSdrDevice(iq_sample_rate=250_000)).iq_sample_rate == 250_000

    def test_the_gain_ladder_comes_from_the_device(self):
        assert source().supported_gains_db == V4_GAINS

    def test_the_gain_comes_from_the_device(self):
        assert source(FakeSdrDevice(gain_db=22.9)).gain_db == 22.9

    def test_the_tuned_frequency_comes_from_the_device(self):
        assert source(FakeSdrDevice(tuned_hz=3_638_000)).tuned_hz == 3_638_000

    def test_the_discard_count_is_the_streaming_one(self):
        """A streaming source drains a transfer pool, where a synchronous read has
        none.  Taking the reading figure here would discard too few.
        """
        device = FakeSdrDevice(blocks_to_discard_streaming=16,
                               blocks_to_discard_reading=2)
        assert source(device).blocks_to_discard_after_gain_change == 16

    def test_the_block_size_is_the_one_it_was_given(self):
        assert source(block_samples=2048).block_samples == 2048

    def test_the_default_block_size_is_the_modules(self):
        assert RtlSdrSource(FakeSdrDevice()).block_samples == DEFAULT_BLOCK_SAMPLES


class TestDraining:
    """Two buffers sit between the tuner and a caller, and counting only one is not
    enough.  This queue is the one the source can see.
    """

    def test_it_empties_the_queue_and_says_how_many(self):
        device = FakeSdrDevice()
        s = source(device)
        s.start()
        for _ in range(3):
            device.deliver(samples=BLOCK)
        assert s.drain() == 3
        assert s.read(timeout=0.01) is None

    def test_draining_an_empty_queue_is_nothing(self):
        assert source().drain() == 0


class TestClosing:
    def test_it_closes_the_device(self):
        device = FakeSdrDevice()
        assert source(device).close() is True
        assert device.closed is True

    def test_a_second_close_does_not_touch_the_device_again(self):
        """The explicit call happens during shutdown and the device's atexit hook
        fires afterwards regardless, so the second one has to be a no-op.
        """
        device = FakeSdrDevice()
        s = source(device)
        assert s.close() is True
        assert s.close() is False


def test_the_queue_is_bounded_so_a_slow_consumer_cannot_exhaust_memory():
    """A deep queue would hide a slow consumer for a while and then fail anyway, where
    a shallow one reports the problem at once.
    """
    s = source(buffer_blocks=3)
    assert isinstance(s._blocks, queue.Queue)
    assert s._blocks.maxsize == 3


class TestTheLevelMeterStream:
    """SdrLevelStream builds its own source and drains it on a thread of its own.

    A meter wants the newest reading rather than a gapless one, so it does not reach
    through the monitor's pipeline.  Nothing covered its constructor before, because
    every test either patched the class or bypassed __init__ with __new__.
    """

    def _stream(self, device=None):
        from buzz.iq import IqToAudio
        from buzz.sdr import SdrLevelStream
        device = device or FakeSdrDevice(iq_sample_rate=IQ_RATE)
        source = RtlSdrSource(device, block_samples=BLOCK, buffer_blocks=2)
        converter = IqToAudio(IQ_RATE, 16, 6_400, 50_000)
        return SdrLevelStream(source, converter, -40.2), device

    def test_it_starts_the_source_it_was_given(self):
        stream, device = self._stream()
        try:
            assert device.sink is not None, 'the meter never started capture'
        finally:
            stream.close()

    def test_closing_closes_the_device(self):
        stream, device = self._stream()
        stream.close()
        assert device.closed is True, (
            'the meter left the receiver held, which stops the monitor opening it'
        )

    def test_closing_twice_is_safe(self):
        """The screen closes it on the way out and the teardown does it again."""
        stream, _ = self._stream()
        stream.close()
        stream.close()

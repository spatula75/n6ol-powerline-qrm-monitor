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
from buzz.sdr import (_what_a_loss_means, _what_a_sustained_drift_means,
                      _DISCARD_LOG_EVERY, DEFAULT_BLOCK_SAMPLES, RtlSdrSource)
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

    def test_a_sustained_problem_keeps_being_reported(self, caplog):
        """The rate limit is a distance rather than a multiple, because this counter is
        not read at every value.

        The device counts refusals on its own thread, and a consumer that has fallen
        behind meets several between two reads, so a test for `discarded % 100 == 0`
        steps over almost every multiple.  Refusals arriving three at a time run 3, 6,
        9 and up to 99, 102, and the next multiple of a hundred they meet is 300.  A
        station discarding blocks all night reported it about a third as often as the
        constant says, at intervals of 300 rather than 100.
        """
        device = FakeSdrDevice()
        s = source(device, buffer_blocks=1)
        s.start()
        refusals_per_read = 3
        reads = 105
        with caplog.at_level('WARNING'):
            for _ in range(reads):
                device.deliver(samples=BLOCK)      # fills the queue
                for _ in range(refusals_per_read):
                    device.deliver(samples=BLOCK)  # refused, so the count moves in threes
                s.read(timeout=0.1)

        discarded = s.blocks_discarded
        said = caplog.text.count('fell behind')
        # One at the start, then one per _DISCARD_LOG_EVERY after it.  The modulus
        # version manages two over this run: the first, and the one at 300.
        assert said >= discarded // _DISCARD_LOG_EVERY, (
            f'{discarded} blocks were discarded and the warning was given {said} '
            f'time(s), where roughly one per {_DISCARD_LOG_EVERY} was intended.  A '
            f'count that arrives in bursts steps over the exact multiples.')

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


class TestWhatTheSourcePassesThrough:
    """A source answers for its device rather than making callers reach past it.

    A passthrough that returns the wrong device's answer is invisible, because the
    figure is a plausible number either way and nothing else in the program reads it
    twice.  This one was first written onto SweepReader by mistake, where nothing asks
    it, and the scope silently got a sound card's answer instead.
    """

    def test_the_bit_depth_comes_from_the_device(self):
        reader = source(FakeSdrDevice(iq_sample_rate=IQ_RATE))
        assert reader.effective_bits == FakeSdrDevice.effective_bits() == 12


class TestWhatTheDriftWarningSays:
    """One interval short of audio and a clock that has walked away are different
    faults, and the wording has to send an operator to different places.

    It said "samples were probably lost, check what else is taking the CPU" for both
    signs once, which is wrong for a negative figure and sent somebody hunting a busy
    machine that had eleven idle cores.  It then said a stall held the receiver up,
    which a direct measurement of the callback ruled out: over nine paired minutes on
    an RSP1B the worst backlog stayed between 71.6 and 87.6 ms while the drift swung
    from -56.4 to +22.3 ms.  See docs-notebook/receiver-clock-drift.md.
    """

    def test_a_loss_names_the_measurements_it_spoils(self):
        message = _what_a_loss_means()
        assert 'samples were lost' in message
        assert 'suspect' in message, (
            'Audio that went missing does spoil the minute it went missing from, and '
            'an operator needs to know which rows to distrust.')
        assert 'CPU' in message

    def test_a_sustained_positive_drift_is_a_steady_loss(self):
        """Audio can only go missing in the direction that leaves the interval holding
        more time than audio, so a positive total is samples disappearing.
        """
        message = _what_a_sustained_drift_means(0.4)
        assert 'going missing' in message
        assert 'CPU' in message

    def test_a_sustained_negative_drift_is_a_rate_that_is_wrong(self):
        """A negative total cannot be a loss, so the receiver is producing more audio
        than the rate it was configured at accounts for.  That is the one fault the
        per-interval check cannot see, because a buffer cycling looks the same for one
        minute at a time.
        """
        message = _what_a_sustained_drift_means(-0.4)
        assert 'faster than the rate' in message
        assert 'CPU' not in message, (
            'A machine that is too busy cannot make a receiver produce extra audio, so '
            'sending an operator to look at load would waste their time.')

    def test_the_two_directions_do_not_share_wording(self):
        assert _what_a_sustained_drift_means(0.4) != _what_a_sustained_drift_means(-0.4)



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

    def test_the_block_size_is_the_one_it_was_given(self):
        assert source(block_samples=2048).block_samples == 2048

    def test_the_default_block_size_is_the_modules(self):
        assert RtlSdrSource(FakeSdrDevice()).block_samples == DEFAULT_BLOCK_SAMPLES


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

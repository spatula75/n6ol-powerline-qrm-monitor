"""Tests for buzz.sdr.RtlSdrSource, the raw IQ capture.

Every test here runs with no receiver attached.  The device is injected, so a stand-in
supplies the handful of members the source touches, and the capture thread is driven
by calling the callback directly rather than by waiting on hardware.

The stand-in is a boundary, which is the kind of thing to mock.  What is not
mocked anywhere is the source's own behavior: the queue, the counters, the snapping
and the discarding are all the real code.
"""

import queue
import threading

import numpy as np
import pytest

from buzz.sdr import DEFAULT_BLOCK_SAMPLES, IqBlock, RtlSdrSource

# Measured from the hardware, so the stand-in offers what the real tuner offers.
V4_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
            22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
            43.9, 44.5, 48.0, 49.6]

IQ_RATE = 256_000
BLOCK = 1_024


class FakeDevice:
    """Stands in for pyrtlsdr's RtlSdr, recording what was asked of it.

    Deliberately not a Mock.  The source reads `sample_rate` back after setting it,
    and a device that cannot do that faithfully would hide the one behavior that
    readback exists to catch.
    """

    def __init__(self, actual_rate: float | None = None):
        self._actual_rate = actual_rate
        self.sample_rate = 0.0
        self.center_freq = 0
        self.gain = None
        self.valid_gains_db = list(V4_GAINS)
        self.agc_calls: list[bool] = []
        self.read_block_bytes: int | None = None
        self.cancelled = False
        self.closed = False
        self._callback = None
        self._reading = threading.Event()

    def __setattr__(self, name, value):
        # The real device rounds the rate to what its divider can produce.
        if name == 'sample_rate' and getattr(self, '_actual_rate', None) is not None:
            value = self._actual_rate
        object.__setattr__(self, name, value)

    def set_agc_mode(self, enabled):
        self.agc_calls.append(enabled)
        return 0

    def read_bytes_async(self, callback, num_bytes):
        self._callback = callback
        self.read_block_bytes = num_bytes
        self._reading.set()
        while not self.cancelled:            # stands in for librtlsdr's blocking loop
            threading.Event().wait(0.01)

    def cancel_read_async(self):
        self.cancelled = True

    def close(self):
        self.closed = True

    def deliver(self, raw):
        """Push one block through the source's callback, as librtlsdr would."""
        self._callback(raw)


def source(device=None, **overrides):
    settings = dict(frequency_hz=7_074_000, gain_db=40.2, iq_sample_rate=IQ_RATE,
                    tuning_offset_hz=50_000, block_samples=BLOCK, buffer_blocks=3)
    settings.update(overrides)
    return RtlSdrSource(device or FakeDevice(), **settings)


def block_of(n_samples=BLOCK, value=128):
    return np.full(n_samples * 2, value, dtype=np.uint8)


def iq_block(raw):
    """An IqBlock wrapped round some raw bytes.  The arrival time and the index play
    no part in reading the bytes, so they are whatever is convenient.
    """
    return IqBlock(raw=np.asarray(raw, dtype=np.uint8), arrived_at=0.0, index=0)


class TestGainSnapping:
    """The tuner accepts only the values in valid_gains_db, and on a V4 the gain
    cannot be read back at all: measured, the getter returns 0.0 whatever is set.
    So the value we choose is the only one anybody will ever know, which is why it is
    chosen here rather than left to the driver.
    """

    @pytest.mark.parametrize('asked, expected', [
        (40.2, 40.2),      # already a supported step
        (40.0, 40.2),      # just below one
        (41.0, 40.2),      # between 40.2 and 42.1, nearer the lower
        (41.5, 42.1),      # between them, nearer the upper
        (-5.0, 0.0),       # below the range
        (99.0, 49.6),      # above the range
    ])
    def test_a_request_snaps_to_the_nearest_step_the_tuner_offers(self, asked, expected):
        assert RtlSdrSource._nearest_supported_gain(asked, V4_GAINS) == expected

    def test_the_source_reports_the_snapped_value_not_the_request(self):
        """What gets written into a recording's metadata.  Reporting the request would
        put a figure in the file that the tuner never used.
        """
        s = source(gain_db=41.0)
        assert s.gain_db == 40.2

    def test_the_snapped_value_is_what_reaches_the_device(self):
        device = FakeDevice()
        source(device, gain_db=41.0)
        assert device.gain == 40.2


class TestConfiguringTheDevice:

    def test_the_digital_agc_is_turned_off_explicitly(self):
        """The RTL2832U has an AGC of its own, separate from the tuner's manual gain.
        An AGC riding on the impulses would compress exactly what this program
        measures while leaving the noise floor looking healthy.
        """
        device = FakeDevice()
        source(device)
        assert device.agc_calls == [False]

    def test_the_device_tunes_away_from_the_frequency_of_interest(self):
        """A receiver puts a strong false signal at its own tuning frequency, measured
        at 37 dB over the surrounding noise.  Tuning to one side is what keeps it out
        of the measured band, and buzz.iq mixes back by the same amount.
        """
        device = FakeDevice()
        s = source(device, frequency_hz=7_074_000, tuning_offset_hz=50_000)
        assert s.tuned_hz == 7_124_000
        assert device.center_freq == 7_124_000

    def test_the_sample_rate_is_read_back_rather_than_assumed(self):
        """The device derives its rate from a 28.8 MHz divider and cannot hit every
        request.  Measured on this hardware: 250000 comes back as 250000.000414.
        Everything downstream counts seconds by dividing samples by this figure.
        """
        device = FakeDevice(actual_rate=250_000.000414)
        s = source(device, iq_sample_rate=250_000)
        assert s.iq_sample_rate == 250_000

    def test_a_rate_the_hardware_rounds_is_reported_and_adopted(self, caplog):
        device = FakeDevice(actual_rate=255_999.0)
        with caplog.at_level('WARNING'):
            s = source(device, iq_sample_rate=IQ_RATE)
        assert s.iq_sample_rate == 255_999
        assert 'ppm' in caplog.text


class TestTakingBlocksFromTheDevice:

    def test_a_delivered_block_can_be_read_back(self):
        device = FakeDevice()
        s = source(device)
        s._on_block(block_of(value=200))

        got = s.read(timeout=0.1)
        assert got is not None
        assert got.samples == BLOCK
        assert np.all(got.raw == 200)

    def test_the_block_is_copied_out_of_the_buffer_the_device_reuses(self):
        """pyrtlsdr hands over a view of a buffer librtlsdr overwrites for the next
        transfer.  Keeping it without copying would leave the consumer reading samples
        that had been rewritten underneath it, which would look like corrupted audio
        rather than like a bug here.
        """
        device = FakeDevice()
        s = source(device)
        buffer = block_of(value=10)
        s._on_block(buffer)

        buffer[:] = 99                       # as the next transfer would
        got = s.read(timeout=0.1)
        assert np.all(got.raw == 10), (
            'The block changed when the device buffer was overwritten, so it was '
            'stored by reference rather than copied.')

    def test_read_returns_none_when_nothing_arrives(self):
        assert source().read(timeout=0.01) is None

    def test_blocks_are_numbered_so_a_gap_is_visible(self):
        s = source()
        for _ in range(3):
            s._on_block(block_of())
        assert [s.read(timeout=0.1).index for _ in range(3)] == [1, 2, 3]


class TestWhenTheConsumerFallsBehind:
    """A full queue means our own thread is too slow, which is a different fault from
    anything the receiver loses, and the only one of the two that can be counted.
    """

    def test_blocks_beyond_the_buffer_are_discarded_and_counted(self):
        s = source(buffer_blocks=3)
        for _ in range(5):
            s._on_block(block_of())

        assert s.blocks_discarded == 2
        assert s._blocks.qsize() == 3

    def test_the_numbering_reveals_which_blocks_went_missing(self):
        """The count says how many were lost.  The indices say where, which is what
        distinguishes a burst from a steady deficit.
        """
        s = source(buffer_blocks=2)
        for _ in range(4):
            s._on_block(block_of())
        kept = [s.read(timeout=0.1).index for _ in range(2)]

        s._on_block(block_of())
        kept.append(s.read(timeout=0.1).index)
        assert kept == [1, 2, 5], (
            f'Kept blocks {kept}.  Blocks 3 and 4 were refused while the queue was '
            'full, and the gap in the numbering is what says so.')

    def test_a_discard_is_reported_but_not_on_every_block(self, caplog):
        """If it happens once it will happen continuously, so a line per block would
        flood the log while stealing time from the thread that is already behind.
        """
        s = source(buffer_blocks=1)
        with caplog.at_level('WARNING'):
            for _ in range(50):
                s._on_block(block_of())

        assert s.blocks_discarded == 49
        assert len(caplog.records) == 1

    def test_discarded_blocks_do_not_count_toward_the_clock(self):
        """clock_drift_seconds compares elapsed time against audio delivered.  Counting
        a block that was thrown away would hide the very shortfall it exists to show.

        Three blocks are kept and two of those are counted, because the first
        establishes the time origin and contributes no samples to the interval.
        """
        s = source(buffer_blocks=3)
        for _ in range(6):
            s._on_block(block_of())
        assert s._samples_delivered == 2 * BLOCK


class TestTheClockDriftSymptom:
    """Nothing reports a dropped sample, because the loss happens inside the receiver.
    Comparing arrival times against the sample count is the only evidence available.
    """

    def test_no_drift_is_reported_before_anything_arrives(self):
        assert source().clock_drift_seconds == 0.0

    def test_audio_arriving_at_the_right_rate_shows_almost_no_drift(self, monkeypatch):
        s = source()
        clock = [1000.0]
        monkeypatch.setattr('buzz.sdr.monotonic', lambda: clock[0])
        for _ in range(10):
            s._on_block(block_of())
            clock[0] += BLOCK / IQ_RATE      # exactly one block's worth of time
            s.read(timeout=0.1)
        assert abs(s.clock_drift_seconds) < BLOCK / IQ_RATE

    def test_time_passing_without_audio_shows_up_as_drift(self, monkeypatch):
        """What a lost run of samples looks like from here: the wall clock moved and
        the sample count did not.
        """
        s = source()
        clock = [1000.0]
        monkeypatch.setattr('buzz.sdr.monotonic', lambda: clock[0])
        s._on_block(block_of())
        s.read(timeout=0.1)
        clock[0] += 1.0                      # a second passes, one block arrives
        s._on_block(block_of())

        assert s.clock_drift_seconds == pytest.approx(1.0 - BLOCK / IQ_RATE, abs=1e-6)


class TestCountingClippedSamples:
    """Bytes 0 and 255 are the converter's rails, since pyrtlsdr maps a byte with
    (byte / 127.5) - 1.  An arc that clips reads smaller than it is, so the events
    spoiled are the loud ones that matter most.
    """

    def test_nothing_at_the_rails_counts_nothing(self):
        assert iq_block(np.full(100, 128, dtype=np.uint8)).clipped_samples == 0

    def test_both_rails_count(self):
        raw = np.array([0, 128, 255, 200, 0], dtype=np.uint8)
        assert iq_block(raw).clipped_samples == 3

    def test_values_just_inside_the_rails_do_not_count(self):
        """1 and 254 are the loudest values that are not clipped.  Counting them would
        report clipping on a signal that merely came close.
        """
        assert iq_block(np.array([1, 254], dtype=np.uint8)).clipped_samples == 0

    def test_i_and_q_are_counted_separately(self):
        """One sample with both halves at the rail counts twice, which the docstring
        says and which a caller converting to a percentage needs to know.
        """
        assert iq_block(np.array([0, 255], dtype=np.uint8)).clipped_samples == 2


class TestStartingAndStopping:

    def test_starting_asks_the_device_for_blocks_of_the_configured_size(self):
        device = FakeDevice()
        s = source(device, block_samples=4_096)
        s.start()
        assert device._reading.wait(timeout=2.0)
        assert device.read_block_bytes == 4_096 * 2, (
            'The device was asked for a different block size than configured.  The '
            'block size sets how long the draining thread has per callback.')
        s.close()

    def test_closing_cancels_the_read_and_closes_the_device(self):
        device = FakeDevice()
        s = source(device)
        s.start()
        assert device._reading.wait(timeout=2.0)
        s.close()

        assert device.cancelled
        assert device.closed
        assert not s._thread.is_alive()

    def test_closing_survives_a_device_that_fails_on_the_way_down(self):
        """Shutdown must not raise, and a failure in one step must not skip the next.

        The hazard is that cancel_read_async failing would take close() down with it
        and leave the receiver streaming into nothing, which is the state the atexit
        hook exists to avoid.  Both steps are attempted even when the first throws.
        """
        device = FakeDevice()
        attempts = []

        def boom():
            attempts.append('called')
            raise OSError('the receiver was unplugged')

        s = source(device)
        device.cancel_read_async = boom
        device.close = boom
        s.close()

        assert len(attempts) == 2, (
            f'{len(attempts)} of the two shutdown steps reached the device.  '
            'cancel_read_async and close are both meant to be attempted even when '
            'the other raises, or a receiver that failed on the way out is left '
            'streaming with nothing collecting from it.')

        s.close()
        assert len(attempts) == 2, (
            'A second close called into the device again.  close is meant to be a '
            'no-op after the first, including when the first one failed partway.')


class TestTheBlockItself:

    def test_a_block_reports_its_sample_count_from_its_bytes(self):
        assert IqBlock(raw=np.zeros(2_048, dtype=np.uint8), arrived_at=0.0,
                       index=1).samples == 1_024

    def test_the_default_block_is_sized_for_the_deadline_it_sets(self):
        """The block duration is the budget the draining thread has per callback.
        Measured: a callback averaging under the block duration keeps up forever,
        while one over it loses the stream and never recovers.  At 256 kHz the default
        gives 64 ms against roughly 2 ms of conversion work.
        """
        budget_ms = DEFAULT_BLOCK_SAMPLES / IQ_RATE * 1000
        assert budget_ms == pytest.approx(64.0, abs=0.1)


def test_the_queue_is_bounded_so_a_slow_consumer_cannot_exhaust_memory():
    """An unbounded queue would turn a slow consumer into a memory leak, and the
    symptom would arrive hours later as an unexplained death rather than as a warning.
    """
    s = source(buffer_blocks=4)
    assert isinstance(s._blocks, queue.Queue)
    assert s._blocks.maxsize == 4


class TestTheBoundaryWithPyrtlsdr:
    """open_device is the only place this module touches pyrtlsdr, and the import
    sits inside it on purpose.  A station using a sound card must never load the
    library, because pyrtlsdr resolves rtlsdr_set_dithering at import time and a
    mismatched librtlsdr therefore fails the import rather than the first call.
    """

    def test_importing_buzz_sdr_does_not_import_pyrtlsdr(self):
        """Run in a fresh interpreter, since any other test may have imported it.

        Without the subprocess this would pass or fail depending on what ran before
        it, which is the kind of test that gets deleted after it flakes once.
        """
        import subprocess
        import sys
        code = ('import sys; import buzz.sdr; '
                "print('rtlsdr' in sys.modules or 'rtlsdr.rtlsdr' in sys.modules)")
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, cwd='lib')
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == 'False', (
            'Importing buzz.sdr pulled pyrtlsdr in with it, so a sound-card station '
            'would load a library it never uses, and one that can fail at import.')

    def test_a_failure_with_no_driver_bound_points_at_zadig(self):
        """libusb calls this "Entity not found", which reads like a missing library
        and sends people to advice about replacing librtlsdr.dll.  That advice is
        wrong here and breaks a working install, so the message has to get in first.
        """
        from buzz.sdr import _why_the_receiver_would_not_open
        failure = OSError(-5, 'LIBUSB_ERROR_NOT_FOUND')
        message = _why_the_receiver_would_not_open(0, failure)

        assert 'Zadig' in message
        assert 'WinUSB' in message
        assert 'librtlsdr.dll' in message, (
            'The message does not warn against replacing the DLL, which is the first '
            'advice somebody finds when they search the libusb wording.')

    def test_any_other_failure_names_the_causes_worth_checking(self):
        """Every other cause is one where taking the device would be wrong, so the
        message says what to close rather than offering to force anything.
        """
        from buzz.sdr import _why_the_receiver_would_not_open
        message = _why_the_receiver_would_not_open(0, OSError(-3, 'LIBUSB_ERROR_ACCESS'))

        assert 'using it' in message
        assert 'plugged in' in message
        assert 'Zadig' not in message, (
            'A busy device is not a driver problem, and sending somebody to Zadig '
            'would have them replace a driver that already works.')

    def test_open_device_wraps_a_failure_rather_than_leaking_libusb_wording(self, monkeypatch):
        import sys
        import types

        from buzz.sdr import open_device

        def explode(index):
            raise OSError(-5, 'LIBUSB_ERROR_NOT_FOUND')

        monkeypatch.setitem(sys.modules, 'rtlsdr',
                            types.SimpleNamespace(RtlSdr=explode))
        with pytest.raises(RuntimeError, match='Zadig'):
            open_device(0)

    def test_open_device_asks_pyrtlsdr_for_the_requested_index(self, monkeypatch):
        import sys
        import types

        from buzz.sdr import open_device
        asked = []
        monkeypatch.setitem(sys.modules, 'rtlsdr', types.SimpleNamespace(
            RtlSdr=lambda index: asked.append(index) or 'the device'))

        assert open_device(2) == 'the device'
        assert asked == [2]


def test_a_failure_receiving_a_block_does_not_escape_into_the_c_callback(caplog):
    """The callback is invoked from C, where an exception has nowhere sensible to go.

    Anything that raises here must be logged and swallowed, or a transient fault
    inside one block would take down capture with no message anybody could act on.
    """
    s = source()
    with caplog.at_level('ERROR'):
        s._on_block(object())            # not something as_array can make sense of

    assert 'Receiving a block' in caplog.text
    assert s.read(timeout=0.01) is None


class TestClosingIsGuaranteedAndRepeatable:
    """A receiver that is never closed keeps streaming and stays claimed, so the next
    process cannot have it.  Measured on real hardware: cancel then close reopened in
    0.66 s on the first attempt, while cancel with no close was still refused after
    16 s of retries.  See librtlsdr issue 116.
    """

    def test_an_exit_hook_is_registered_so_a_skipped_shutdown_still_closes(self, monkeypatch):
        """The explicit close is the one that normally runs.  The hook covers the paths
        that skip it, such as an unhandled exception on another thread.

        atexit offers no public way to read its registry, so registration is observed
        by watching the call rather than by inspecting a result.
        """
        registered = []
        monkeypatch.setattr('buzz.sdr.atexit.register', registered.append)

        s = source()
        assert s.close in registered, (
            'Nothing was registered with atexit, so a shutdown path that skipped the '
            'explicit close would leave the receiver streaming and still claimed.')

    def test_closing_twice_does_not_touch_the_device_twice(self):
        """The explicit call and the atexit hook both fire during a normal shutdown,
        so the second must be a no-op rather than a second attempt at a closed device.
        """
        calls = []
        device = FakeDevice()
        device.close = lambda: calls.append('close')
        device.cancel_read_async = lambda: calls.append('cancel')

        s = source(device)
        s.close()
        s.close()
        s.close()

        assert calls == ['cancel', 'close'], (
            f'The device saw {calls}.  close() is not idempotent, so the atexit hook '
            'would operate on a device that was already shut down.')

    def test_the_hook_is_removed_once_the_device_is_closed(self, monkeypatch):
        """Leaving a registered hook would keep the source alive until interpreter
        exit, holding the device object with it.
        """
        removed = []
        monkeypatch.setattr('buzz.sdr.atexit.unregister', removed.append)

        s = source()
        assert removed == []
        s.close()
        assert removed == [s.close], (
            'close() left its atexit hook in place, so the interpreter holds a '
            'reference to a source whose device is already shut down.')

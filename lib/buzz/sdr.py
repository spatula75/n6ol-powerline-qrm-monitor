"""Raw IQ capture from an RTL-SDR receiver, and nothing else.

This module holds the hardware.  It opens the device, configures it, and hands out
blocks of raw bytes exactly as they arrived.  It has no opinion about what happens
next.  The conversion to audio lives in buzz.iq, and the ring buffer everything else
reads lives in buzz.sampler.  Keeping the three apart is what lets the conversion be
tested exhaustively with no receiver attached.  It also leaves somewhere for IQ
recording to tap in later, without disturbing either neighbor.

Why the callback does almost nothing
------------------------------------
The receiver's own FIFO holds 1880 bytes, which is 940 complex samples, so at 256 kHz
it overflows 3.67 ms after collection stops.  That is far shorter than a Windows
scheduler quantum, and nothing anywhere reports the loss, because it happens in
hardware upstream of every piece of software.  A sound card can report an overflow
because the driver owns the buffer that overflowed.  Here nobody owns it.

What saves the arrangement is librtlsdr's pool of USB transfers, which the host
controller fills by DMA without our thread being scheduled at all.  Measured on this
hardware, that pool holds about 960 ms at our block size, so the 3.67 ms deadline is
met by the USB stack rather than by Python.

The deadline that does fall to us is softer and different.  Each callback carries
`block_samples` worth of audio, so the callback must average less than that duration
or the pool drains and never recovers.  Measurement showed that a callback stalled
60 ms against a 64 ms block stayed clean, while one stalled 200 ms lost 67% of the
stream and kept losing it.  The pool absorbs bursts, not sustained slowness.

So the callback copies its block, timestamps it, and returns.  Every other piece of
work, including converting to complex and counting clipped samples, belongs to
whatever thread drains this class.

Why close() is also registered with atexit
------------------------------------------
A receiver that is never closed keeps running.  It goes on streaming with nothing
collecting the samples, which wastes power and warms the tuner for no purpose.  See
https://github.com/librtlsdr/librtlsdr/issues/116

Be careful what that does and does not cost, because the obvious guess is wrong.
Measured on this hardware, across a process that exits without closing, the next
process opened the device in 0.72 s on its first attempt.  Closing first made no
difference, at 0.75 s and also on the first attempt.  The operating system reclaims the USB handle
when a process ends, so a skipped close does not strand the device for anybody else.

The hook is therefore ordinary resource hygiene rather than a fix for a measured
failure.  It costs nothing, the explicit call is the one that normally runs, and the
hook covers paths that skip it such as an unhandled exception on another thread.
close() is idempotent because during an orderly shutdown both will fire.

Nothing saves a device from a hard kill, and nothing can.
"""

import atexit
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import numpy as np

from buzz.sampler import LevelStream, RingBufferPipeline

if TYPE_CHECKING:
    from buzz.iq import IqToAudio

logger = logging.getLogger(__name__)

# The raw byte values that mean the converter hit its rail.  pyrtlsdr turns a byte
# into a sample with (byte / 127.5) - 1, so 0 becomes -1.0 and 255 becomes +1.0 and
# nothing else reaches either.  Testing the bytes rather than the converted samples
# keeps this an exact integer comparison instead of a float one.
_RAW_MIN = 0
_RAW_MAX = 255

# Half the span between the rails, which is what pyrtlsdr's (byte / 127.5) - 1
# divides by.  Derived rather than written as 127.5, so the conversion and the rails
# above cannot drift apart.  TestTheRawConversion pins both ends of the mapping.
_RAW_HALF_SPAN = (_RAW_MAX - _RAW_MIN) / 2

# Bytes per complex sample, one for I and one for Q.
_BYTES_PER_SAMPLE = 2

# How many transfers librtlsdr keeps in flight.
#
# pyrtlsdr passes DEFAULT_ASYNC_BUF_NUMBER = 0 straight to rtlsdr_read_async and
# librtlsdr substitutes 15, so the pool holds buf_num * block_samples / sample_rate.
# At the default block and 256 kHz that is 960 ms, which matched a hand measurement to
# the digit.  It is a library constant rather than a parameter, so this restates it and
# TestTheTransferPoolDepth pins the arithmetic that depends on it.
_TRANSFER_POOL_BLOCKS = 15

# Samples per callback, which is the deadline the draining thread has to beat.
#
# 16384 samples is 64 ms at 256 kHz, against roughly 2 ms of conversion work, so the
# margin is about thirty to one.  Smaller blocks tighten the deadline and buy
# nothing.  Larger ones delay the first samples and coarsen the drop check, which
# compares arrival times against the sample count.
DEFAULT_BLOCK_SAMPLES = 16_384

# How many blocks may wait for the draining thread before the oldest is refused.
#
# The queue is deliberately small, because a deep one would hide a slow consumer for
# a while and then fail anyway, where a shallow one reports the problem at once.
#
# It also keeps the two kinds of loss apart.  A refusal here is our own thread
# running late, which is countable and fixable.  A loss inside the receiver is
# neither.
DEFAULT_BUFFER_BLOCKS = 8

# How often to repeat the discard warning, counted in discarded blocks.  The first is
# always reported and then every hundredth, so a sustained problem stays visible
# without costing a log line per block.
_DISCARD_LOG_EVERY = 100

# How long to wait for a thread to finish during shutdown, in seconds.
#
# The two threads stop by different means.  The feeder notices its stop flag when its
# next read times out, which is within 0.5 s.  The capture thread returns from
# read_bytes_async once cancel_read_async has taken effect.  Five seconds is several
# times either, so a thread still running afterwards is stuck rather than slow, and
# RtlSdrSource.close treats it that way.
_THREAD_JOIN_TIMEOUT_SECONDS = 5.0

# How long a draining thread waits for a block before it rechecks its stop flag.
# It sets how quickly close() returns on a receiver that has gone quiet, so it is
# chosen short against the join above rather than against the block rate.
_FEED_READ_TIMEOUT_SECONDS = 0.5

# How long to wait for rtlsdr_close before giving up on the driver.
#
# Short, because nothing useful happens after it.  A close that has not returned by
# now is inside libusb and is not coming back, and the operating system releases the
# handle when the process ends regardless.  See
# RtlSdrSource._shut_the_device_without_waiting_for_ever.
_DEVICE_CLOSE_TIMEOUT_SECONDS = 3.0


# How often the pipeline looks at its own health counters, in seconds.
#
# A minute matches the collector's own cadence, so a warning reaches the log beside
# the CSV row it spoiled.  Looking more often would find nothing sooner, because the
# counters only move when a block arrives.
_HEALTH_INTERVAL_SECONDS = 60.0

# How far the receiver clock may run from the system clock, in parts per million,
# before the difference means lost samples rather than two crystals disagreeing.
#
# The figure is chosen rather than measured.  RTL-SDR crystals are specified in the
# tens of parts per million, so 500 leaves room for a poor one and still catches a
# loss, which runs to thousands.  See RtlSdrSource.clock_drift_seconds.
_DRIFT_PPM_LIMIT = 500


class RtlSdrDevice(Protocol):
    """The part of pyrtlsdr's RtlSdr this module uses.

    It is declared so that a test can supply something else.  Everything here runs
    against a plain object with these members, which is what lets the whole class be
    exercised with no receiver attached.
    """

    sample_rate: float
    center_freq: float
    gain: float
    valid_gains_db: list[float]

    def set_agc_mode(self, enabled: bool) -> int:
        ...

    def read_bytes_async(self, callback: object, num_bytes: int) -> None:
        ...

    def cancel_read_async(self) -> None:
        ...

    def close(self) -> None:
        ...


def open_device(index: int = 0) -> RtlSdrDevice:
    """Open the receiver at `index`.

    The import sits inside the function so that a station using a sound card never
    loads pyrtlsdr.  pyrtlsdr looks up rtlsdr_set_dithering at import time, so a
    mismatched librtlsdr makes the import itself fail rather than the first call.
    buzz.render takes the same approach with ffmpeg for the same reason.

    That failure is caught and reworded here, so both ways of not reaching a receiver
    leave by the same door.  Everything this raises is a RuntimeError carrying a
    message for the operator, which is what lets main.py print one rather than a
    traceback.

    This reports a failure with its likely causes named, because libusb's own wording
    sends people the wrong way.  "Entity not found" reads like a missing library and means
    no driver is bound to the device, which on Windows is what Zadig exists to fix.
    Searching that phrase finds advice about replacing librtlsdr.dll, which is both
    wrong and destructive here.

    Nothing tries to take the device from whoever already has it.  A device cannot be
    closed without the handle that opened it, and every cause of a failure here is a
    case where taking it would be wrong.  The monitor may be running already, another
    program may be using the receiver deliberately, or there may be no receiver.  A previous run that died
    without closing is not among them, because the operating system reclaims the
    handle when a process ends.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError as exc:
        raise RuntimeError(
            f'The pyrtlsdr library would not load ({exc}), and [audio] source is set '
            'to rtlsdr.  Either it is not installed, or its bundled librtlsdr is too '
            'old to carry the symbol it looks up as it imports.  Run pip install '
            '"pyrtlsdr[lib]", or set [audio] source back to soundcard.') from exc
    try:
        return RtlSdr(index)
    except Exception as exc:
        raise RuntimeError(_why_the_receiver_would_not_open(index, exc)) from exc


def close_device(device: RtlSdrDevice) -> bool:
    """Release a receiver, and stop waiting if the driver never comes back.

    Paired with open_device, and for the same kind of reason: librtlsdr needs a
    wrapper here that the rest of the program should not have to know about.

    rtlsdr_close blocks inside libusb when transfers were never fully cancelled, and
    does not return at all.  Whoever called it blocks too, and that reached everything
    that touches a receiver.  A gain sweep reached its last step and stopped there,
    showing the final step with no result and no error while the event loop stayed
    perfectly responsive, because the block was in a worker thread.  The level meter
    did it leaving its screen.  The gain picker did it after reading one list.  And
    leaving the program took five minutes, because asyncio waits
    THREAD_JOIN_TIMEOUT seconds for its default executor before giving up.

    A daemon thread costs nothing at exit, since Python does not join one.  The device
    stays held until the process ends, which is what happened anyway.  What changes is
    that the program keeps working, and says the receiver is still held rather than
    leaving somebody to meet LIBUSB_ERROR_ACCESS on the next run and read it as a
    permissions problem.

    Returns whether the device actually closed.
    """
    finished = threading.Event()

    def shut() -> None:
        try:
            device.close()
        except Exception:
            logger.debug('Closing the receiver failed.', exc_info=True)
        finally:
            finished.set()

    threading.Thread(target=shut, daemon=True, name='rtlsdr-close').start()
    if finished.wait(_DEVICE_CLOSE_TIMEOUT_SECONDS):
        return True
    logger.warning(
        'The receiver did not close within %.0f seconds and was left to the operating '
        'system.  The driver can block inside libusb and never return, so waiting '
        'longer would only hang this program.', _DEVICE_CLOSE_TIMEOUT_SECONDS)
    return False


def _why_the_receiver_would_not_open(index: int, exc: Exception) -> str:
    """Turn a libusb failure into something that names what to try.

    Split out from open_device so the wording can be read and tested without a
    receiver, and so the two likely causes stay side by side where they can be compared.
    """
    if getattr(exc, 'errno', None) == -5:        # LIBUSB_ERROR_NOT_FOUND
        return (f'Receiver {index} was found but no driver is bound to it ({exc}).  On '
                'Windows this means Zadig has not been run for this device on this USB '
                'port.  Run Zadig as administrator, tick Options then List All Devices, '
                'select "Bulk-In, Interface (Interface 0)", and install WinUSB.  Do not '
                'replace librtlsdr.dll, which is the usual advice and is wrong here.')
    return (f'Receiver {index} could not be opened ({exc}).  Either something else is '
            'using it, such as another copy of this monitor or an SDR application, or '
            'no receiver is plugged in.  Close whatever holds it, or check the cable, '
            'and start again.')


@dataclass(frozen=True)
class IqBlock:
    """One block of raw IQ, as the device delivered it.

    `raw` is interleaved unsigned bytes, I then Q, already copied out of the buffer
    librtlsdr reuses.  `arrived_at` is the monotonic clock when the callback ran.
    `index` counts blocks the device produced rather than blocks that survived, so a
    gap in the sequence tells a consumer that something was refused.
    """

    raw: np.ndarray
    arrived_at: float
    index: int

    @property
    def samples(self) -> int:
        """How many complex samples this block carries."""
        return len(self.raw) // _BYTES_PER_SAMPLE

    @property
    def clipped_samples(self) -> int:
        """How many raw values in this block sat at the converter's rail.

        Counts I and Q separately, so one sample with both at the rail counts twice.
        The figure is a symptom rather than a measurement, and what it means is that
        the receiver gain is set too high for what the antenna is hearing.

        A clipped arc reads smaller than it truly is, so the events it spoils are the
        loud ones that matter most, and nothing else about the audio looks wrong.
        Measured on this hardware, at maximum gain on a quiet band, the peak already
        reached 0.35 of full scale and left 9 dB for an impulse.

        Counted on the raw bytes rather than after the conversion.  pyrtlsdr maps a
        byte to a sample with (byte / 127.5) - 1, so the rails are exactly 0 and 255
        and the test is an integer comparison.  Done after conversion it would be a
        float comparison against 1.0, which is the same question asked less precisely.
        """
        return int(np.count_nonzero((self.raw == _RAW_MIN) | (self.raw == _RAW_MAX)))

    def as_complex(self) -> np.ndarray:
        """Turn the raw bytes into the complex samples the conversion expects.

        The arithmetic is pyrtlsdr's own, reproduced here rather than called, because
        read_bytes_async is what this module uses and packed_bytes_to_iq is a method on
        a device object the conversion thread has no business touching.

        Viewing a float64 array as complex128 pairs consecutive values into real and
        imaginary parts, which is exactly the interleaving the device produces.  A byte
        of 0 becomes -1.0 and a byte of 255 becomes +1.0, which is what makes the
        clipped count above equivalent to asking whether a sample reached the rail.
        """
        return self.raw.astype(np.float64).view(np.complex128) / _RAW_HALF_SPAN - (1 + 1j)


class RtlSdrSource:
    """Pulls raw IQ off the receiver and queues it for somebody else to convert.

    Call start(), then read() until it returns None, then close().  The caller
    injects the device rather than this opening one, so a test can pass a stand-in.  See open_device
    for the real one.
    """

    def __init__(self, device: RtlSdrDevice, *, frequency_hz: int, gain_db: float,
                 iq_sample_rate: int, tuning_offset_hz: int,
                 block_samples: int = DEFAULT_BLOCK_SAMPLES,
                 buffer_blocks: int = DEFAULT_BUFFER_BLOCKS) -> None:
        self._device = device
        self._block_samples = block_samples
        self._blocks: queue.Queue[IqBlock] = queue.Queue(maxsize=buffer_blocks)

        self._produced = 0
        self._discarded = 0
        self._samples_delivered = 0
        self._first_arrival: float | None = None
        self._last_arrival: float | None = None

        self._stopping = threading.Event()
        self._closed = False
        self._released = False
        self._thread = threading.Thread(target=self._run, daemon=True, name='rtlsdr')

        self._tuned_hz = frequency_hz + tuning_offset_hz
        self._configure(gain_db, iq_sample_rate)
        # A receiver that is never closed keeps streaming with nothing collecting from
        # it.  See https://github.com/librtlsdr/librtlsdr/issues/116
        #
        # It does not strand the device for the next process, which the module
        # docstring measures, so this is hygiene rather than repair.
        #
        # The hook is registered after configuring, so a device that failed to
        # configure is not left with one pointing at a half-built object.
        atexit.register(self.close)

    def _configure(self, gain_db: float, iq_sample_rate: int) -> None:
        """Set the rate, the tuning and the gain, and turn both gain controls off.

        Order matters less than completeness here, but two of these are easy to leave
        out and neither announces itself.

        The RTL2832U has a digital AGC of its own, separate from the tuner's manual
        gain, and it is off by default only by convention.  An AGC riding on the
        impulses would compress exactly what this program measures while leaving the
        noise floor looking healthy, so it is disabled explicitly.

        This reads the rate back, because the device derives it from a 28.8 MHz
        divider and cannot hit every request.  Measured on this hardware, 256000 comes back
        exactly, where 250000 comes back as 250000.000414.
        """
        self._device.sample_rate = iq_sample_rate
        actual = float(self._device.sample_rate)
        self._iq_sample_rate = int(round(actual))
        if self._iq_sample_rate != iq_sample_rate:
            logger.warning(
                'Asked the receiver for %d Hz and got %.6f Hz.  Everything downstream '
                'will treat the audio as %d Hz.  A rate the hardware cannot produce '
                'exactly is normal, and the difference here is %.1f ppm.',
                iq_sample_rate, actual, self._iq_sample_rate,
                abs(actual - iq_sample_rate) / iq_sample_rate * 1e6)

        self._device.center_freq = self._tuned_hz
        self._device.set_agc_mode(False)

        self._gain_db = self._nearest_supported_gain(gain_db, list(self._device.valid_gains_db))
        self._device.gain = self._gain_db
        if self._gain_db != gain_db:
            logger.info('Receiver gain %.1f dB is not one the tuner offers, so %.1f dB '
                        'was used instead.', gain_db, self._gain_db)

    @staticmethod
    def _nearest_supported_gain(gain_db: float, supported: list[float]) -> float:
        """The value from `supported` closest to `gain_db`.

        Snapped here rather than left to the driver, because on an RTL-SDR Blog V4 the
        gain cannot be read back.  Measured on this hardware, the setter works and the
        level moves by 57.5 dB over the full range, while the getter returns 0.0 at
        every setting.  So the only figure we can ever know is the one we chose, and
        choosing it ourselves is the only way to record it accurately in a file's
        metadata.
        """
        return min(supported, key=lambda candidate: abs(candidate - gain_db))

    # ------------------------------------------------------------------ public

    @property
    def block_samples(self) -> int:
        """Complex samples per callback, which is how long one block lasts.

        Public because the block duration is the deadline every consumer of this
        class works against, and because it sets the time constant of anything
        smoothing across blocks.
        """
        return self._block_samples

    @property
    def iq_sample_rate(self) -> int:
        """The rate the device is actually running at, rounded to whole samples."""
        return self._iq_sample_rate

    @property
    def supported_gains_db(self) -> list[float]:
        """Every tuner gain this device offers, in the order it reports them."""
        return list(self._device.valid_gains_db)

    @property
    def blocks_to_discard_after_gain_change(self) -> int:
        """Blocks that may predate a gain change, and so have to be thrown away.

        Up to _TRANSFER_POOL_BLOCKS buffers are filled or in flight when the gain
        moves, plus the one being written at that moment, so discarding this many
        makes every later block provably post-change.  Counting blocks rather than
        waiting a duration keeps it independent of scheduler jitter and of the sample
        rate being what was asked for.

        Measuring without the discard reads the previous step's answer shifted by one
        step, which looks like a plausible curve and is wrong.
        """
        return _TRANSFER_POOL_BLOCKS + 1

    def set_gain(self, gain_db: float) -> float:
        """Move the tuner gain while streaming, and return the value actually set.

        The request is snapped to a step the tuner offers, the same way the
        constructor snaps it, because a V4 cannot report its own gain and the figure
        we chose is the only one anybody will ever know.

        Blocks already in the transfer pool still carry the old gain.  A caller
        measuring the result has to drop blocks_to_discard_after_gain_change of them
        first, which is why that number is public.

        This writes the tuner from the calling thread, which is not free of risk and
        is the least bad of the options measured so far.

        Setting a gain is a pair of synchronous USB control transfers, and while
        capture runs the capture thread is inside rtlsdr_read_async driving libusb's
        event loop on the same device.  Two threads touching one device is a race, and
        twice in a few dozen sweeps it ended with a transfer that never completed and
        an rtlsdr_close that never returned.  close_device bounds that rather than
        preventing it.

        Moving the write into the callback was tried and is worse.  libusb's
        synchronous API completes a transfer by pumping the event loop itself, so
        calling it from inside a callback that libusb_handle_events is already running
        re-enters event handling, which libusb does not allow.  On real hardware the
        write simply failed, the gain never moved, and a sweep of a flat curve reached
        no answer at all.  In a stand-in with no USB underneath it, the same code
        passed every test.

        The remaining option is to stop the async read around each change and restart
        it, which removes the concurrency outright at the cost of a cancel and a pool
        refill at every one of 145 steps.  cancel_read_async is itself implicated in
        the hang, so that trade has not been taken without measuring it.
        """
        self._gain_db = self._nearest_supported_gain(gain_db, self.supported_gains_db)
        self._device.gain = self._gain_db
        return self._gain_db

    @property
    def gain_db(self) -> float:
        """The gain that was set, which is the only figure we can know.

        Nothing reads it back from the device.  See _nearest_supported_gain.
        """
        return self._gain_db

    @property
    def tuned_hz(self) -> int:
        """Where the device is tuned, which is deliberately not the frequency of interest.

        An SDR puts a strong false signal at exactly its tuning frequency, from the
        tuner leaking into its own mixer.  It measured 37 dB above the surrounding
        noise on this hardware.  Tuning to one side and mixing back in buzz.iq moves
        it out of the measured band.
        """
        return self._tuned_hz

    @property
    def blocks_discarded(self) -> int:
        """Blocks refused because the draining thread had not kept up.

        This counts something different from anything the receiver lost, and the two
        stay apart on purpose.  A count here means our own consumer is too slow, which
        is a bug with a fix.  A loss inside the receiver cannot be counted at all.
        """
        return self._discarded

    @property
    def clock_drift_seconds(self) -> float:
        """Elapsed time minus the audio the device delivered for it.

        The only available evidence that samples went missing, since nothing reports a
        drop.  A positive figure means less audio arrived than the wall clock says it
        should have.

        Read it as a symptom rather than a measurement.  The receiver's crystal and the
        system clock differ by some parts per million that nobody here has measured,
        so the two separate slowly even when nothing is wrong.  Milliseconds over a
        few seconds mean lost samples.  Microseconds mean clocks.
        """
        if self._first_arrival is None or self._last_arrival is None:
            return 0.0
        elapsed = self._last_arrival - self._first_arrival
        return elapsed - self._samples_delivered / self._iq_sample_rate

    def start(self) -> None:
        """Begin capture, on a thread of its own.

        read_bytes_async does not return until the read is cancelled, so it cannot run
        on the caller's thread.
        """
        self._thread.start()

    def read(self, timeout: float = 1.0) -> IqBlock | None:
        """Take the next block, or None if none arrived within `timeout`.

        None means the device has gone quiet rather than that capture has finished.
        A caller looping on this should check whether it is still meant to be running
        rather than treat None as the end.
        """
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Throw away every block already queued, and say how many that was.

        There are two buffers between the tuner and a caller, and counting only one
        of them is not enough.  blocks_to_discard_after_gain_change covers librtlsdr's
        transfer pool, which is the buffering nobody here can see.  This queue is the
        other one, and anything sitting in it when the gain changes was captured
        before the change.

        Without this, the counted discard spends itself on stale queue entries first
        and lets that many true post-change callbacks through in their place, so
        up to buffer_blocks blocks of the previous gain reach the measurement.  It is
        worst at the first step of a sweep, where the queue has been filling since
        start() with nobody reading.
        """
        dropped = 0
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return dropped
            dropped += 1

    def close(self) -> bool:
        """Stop capture and release the device.

        Cancelling makes read_bytes_async return, which lets the capture thread end,
        and the device is closed after that.  Closing is what leaves the receiver usable
        by the next process; see the module docstring for the measurements.

        Safe to call more than once, and it will be.  The explicit call happens during
        an orderly shutdown, and the atexit hook fires afterwards regardless, so the
        second one has to be a no-op rather than a second attempt at a closed device.

        Returns whether the device was actually released.  False means it is still
        held, and the consequence falls on whatever opens a receiver next: libusb
        refuses with LIBUSB_ERROR_ACCESS, which reads as a permissions problem and is
        not one.  A caller about to reopen the device has to be able to say so, rather
        than leave somebody to work it out from that.
        """
        if self._closed:
            return self._released
        self._closed = True
        atexit.unregister(self.close)
        self._stopping.set()
        try:
            self._device.cancel_read_async()
        except Exception:
            logger.debug('Cancelling the receiver read failed during shutdown.',
                         exc_info=True)
        # join() raises on a thread that was never started, which happens whenever
        # setup failed between construction and start().  Shutdown must not raise.
        if self._thread.is_alive():
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        # A join that timed out leaves the capture thread inside librtlsdr's own read.
        # Closing now would free the handle it is reading through, which is a crash in
        # C rather than an exception here.  The module docstring measures what skipping
        # the close costs, and the answer is nothing.
        if self._thread.is_alive():
            logger.warning(
                'The receiver capture thread did not stop within %.0f seconds, so the '
                'device was left open.  Closing it now would free a handle that thread '
                'is still reading through.  The operating system releases it when this '
                'process ends.', _THREAD_JOIN_TIMEOUT_SECONDS)
            return False
        self._released = close_device(self._device)
        return self._released

    # ----------------------------------------------------------------- private

    def _run(self) -> None:  # pragma: no cover -- thread body; _on_block is tested directly
        """Capture thread body.  read_bytes_async blocks here until cancelled."""
        try:
            self._device.read_bytes_async(self._on_block, self._block_samples * _BYTES_PER_SAMPLE)
        except Exception:
            if not self._stopping.is_set():
                logger.exception(
                    'The receiver stopped delivering samples.  Capture has ended and '
                    'will not restart on its own, because the device cannot be '
                    'reopened promptly.  Restart the monitor to try again.')

    def _on_block(self, buffer: object, _context: object = None) -> None:
        """Take one block from the device, on librtlsdr's own thread.

        This does almost nothing, because the pool it drains holds about 960 ms and
        the only way to exhaust that is to spend longer here, on average, than the
        block's own duration.  See the module docstring.

        The copy is not optional, because pyrtlsdr hands over a view of a buffer
        librtlsdr reuses for the next transfer.  Anything kept without copying would
        be rewritten underneath the consumer.

        Nothing here is allowed to raise, because this is called from C, where an
        exception has nowhere sensible to go.
        """
        try:
            arrived = monotonic()
            raw = np.ctypeslib.as_array(buffer).astype(np.uint8, copy=True)
            self._produced += 1
            block = IqBlock(raw=raw, arrived_at=arrived, index=self._produced)
            try:
                self._blocks.put_nowait(block)
            except queue.Full:
                self._discarded += 1
                self._report_discard()
                return
            # The first block establishes the origin and contributes no samples.
            # Its audio was collected before this instant, so counting it against an
            # interval that starts here would show a healthy stream permanently one
            # block in deficit.
            if self._first_arrival is None:
                self._first_arrival = arrived
            else:
                self._samples_delivered += block.samples
            self._last_arrival = arrived
        except Exception:
            logger.exception('Receiving a block of samples failed.  Capture continues.')

    def _report_discard(self) -> None:
        """Say that a block was refused, without saying it on every block.

        This gets reported because it means something different from a loss inside the
        receiver, and rate-limited because it will not happen once.  A line per block
        would flood the log while stealing time from the thread that is already
        behind.
        """
        if self._discarded == 1 or self._discarded % _DISCARD_LOG_EVERY == 0:
            logger.warning(
                'Discarded %d block(s) of receiver samples because the conversion '
                'thread fell behind.  The audio now has gaps in it, so levels and '
                'grid frequency from this period are both suspect.  Something else on '
                'this machine is probably taking the CPU.', self._discarded)


class RtlSdrPipeline(RingBufferPipeline):
    """Feeds the shared ring buffer from a receiver, converting on the way.

    This is the third of the three pieces.  RtlSdrSource holds the hardware,
    IqToAudio holds the arithmetic, and this owns the thread that carries blocks from
    one to the other.  Everything downstream reads this exactly as it reads the sound card, and
    cannot tell which it has.

    This thread is the one with a deadline, because it must average less than a
    block's own duration.  That is 64 ms at the default settings against roughly 2 ms
    of work.  Run longer than it and the pool inside librtlsdr drains, and samples are
    lost where nothing can count them.  So the loop does the least it can, which is to
    convert, count and append.  See the module docstring.
    """

    def __init__(self, source: RtlSdrSource, converter: 'IqToAudio', *,
                 clock: Callable[[], float] = monotonic) -> None:
        super().__init__(converter.audio_sample_rate)
        self._source = source
        self._converter = converter
        self._clipped = 0
        self._leftover = np.empty(0, dtype=np.int16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, daemon=True, name='sdr-feeder')
        # The clock is injected so a test can drive a minute of monitoring in no time
        # at all.  Waiting for the real one would put _HEALTH_INTERVAL_SECONDS into
        # the suite for every case.
        self._clock = clock
        self._health_checked_at = clock()
        self._clipped_reported = 0
        self._saturated_reported = 0
        self._drift_reported = 0.0

    @property
    def clipped_samples(self) -> int:
        """Raw values that reached the converter's rail since capture began.

        A count above zero means the receiver gain is too high for what the antenna is
        hearing, and that loud events are being measured smaller than they are.
        """
        return self._clipped

    @property
    def source(self) -> RtlSdrSource:
        """The capture this is draining, for anything that wants its counters."""
        return self._source

    def start(self) -> None:
        """Begin capture and begin draining it."""
        self._source.start()
        self._thread.start()

    def close(self) -> None:
        """Stop capture, let the feeder finish, and release the device.

        The source is closed first so no further blocks arrive.  The feeder then sees
        its read time out, notices the stop, and ends.
        """
        self._stop.set()
        self._source.close()
        if self._thread.is_alive():
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def _feed(self) -> None:  # pragma: no cover -- thread body; _consume is tested directly
        """Drain the source until told to stop."""
        while not self._stop.is_set():
            try:
                block = self._source.read(timeout=_FEED_READ_TIMEOUT_SECONDS)
                if block is not None:
                    self._consume(block)
            except Exception:
                logger.exception(
                    'Converting a block of receiver samples failed.  That block is '
                    'lost and capture continues, so the audio has a gap in it.')

    def _consume(self, block: IqBlock) -> None:
        """Convert one block and hand the audio to the ring buffer.

        Counting the clipping here rather than in the callback is deliberate.  The raw
        bytes are sitting in the block either way, and the callback has a deadline
        this does not.
        """
        self._clipped += block.clipped_samples
        self._append_in_chunks(self._converter.convert(block.as_complex()))
        self._report_health()

    def _report_health(self) -> None:
        """Say once a minute whether the receiver is still delivering honest samples.

        Each failure this reports is silent in the data it spoils.  A receiver whose
        gain is too high clips every loud arc, measures it smaller than it is, and
        writes that figure to the CSV with nothing to mark it.  Lost samples do the
        same to the grid frequency.  Both read as a quiet band, which is the answer
        the operator is hoping for, so neither prompts anybody to look.

        A log line is the smallest thing that makes the counters visible.  The
        counters stay public so that a CSV column or a light on the display can read
        them later instead.

        The check runs from _consume rather than from the feeder loop, so a test can
        drive it directly.  A stream that stops entirely therefore stops reporting,
        which is correct.  _run already says the receiver went quiet, and a second
        voice for one failure would only add noise.
        """
        now = self._clock()
        elapsed = now - self._health_checked_at
        if elapsed < _HEALTH_INTERVAL_SECONDS:
            return
        self._health_checked_at = now
        self._warn_about_clipping(elapsed)
        self._warn_about_drift(elapsed)

    def _warn_about_clipping(self, elapsed: float) -> None:
        """Report any sample that hit a rail, at the receiver or at the int16 output.

        The two are counted apart because they have separate causes.  A raw value at
        the converter's rail means the antenna is louder than the tuner gain allows.
        A clipped output sample can happen without that, because the filter can leave
        a peak slightly above where its input sat.

        Any movement at all is reported.  A threshold would need a figure nobody has
        measured, and on a quiet band the honest count is zero, so a single clipped
        sample is already news.
        """
        clipped = max(0, self._clipped - self._clipped_reported)
        saturated = max(0, self._converter.saturated_samples - self._saturated_reported)
        self._clipped_reported = self._clipped
        self._saturated_reported = self._converter.saturated_samples
        if not clipped and not saturated:
            return
        logger.warning(
            'The receiver clipped %d raw value(s) in the last %.0f seconds, and the '
            'conversion clipped %d output sample(s).  Loud events are measured smaller '
            'than they are.  Lower [rtlsdr] gain_db by one step.',
            clipped, elapsed, saturated)

    def _warn_about_drift(self, elapsed: float) -> None:
        """Report a receiver clock that has run away from the system clock.

        This one needs a limit where the counters above do not, because the figure is
        never exactly zero.  Two crystals always disagree by some parts per million,
        so "any movement" would report every minute of a healthy run.  What is
        measured here is the change since the last report rather than the total, so a
        steady offset settles instead of accumulating into a warning.
        """
        drift = self._source.clock_drift_seconds
        moved = drift - self._drift_reported
        self._drift_reported = drift
        if abs(moved) <= elapsed * _DRIFT_PPM_LIMIT / 1e6:
            return
        logger.warning(
            'The receiver and system clocks moved %+.0f ms apart over the last %.0f '
            'seconds, which is more than a crystal explains.  Samples were probably '
            'lost, so levels and grid frequency from this period are suspect.  Check '
            'what else on this machine is taking the CPU.', moved * 1e3, elapsed)

    def _append_in_chunks(self, audio: np.ndarray) -> None:
        """Hand the audio over in pieces of exactly CHUNK_SIZE, holding any remainder.

        get_snapshot works out how many chunks to read by dividing the sample count
        by CHUNK_SIZE, so a shorter chunk makes it return
        less audio than it was asked for, with no error anywhere.  Measured on the real buffer,
        appending 256-sample chunks to one asked for 4000 samples returns 2064.  The
        analyzer would then be handed a window shorter than the one it sized its
        arithmetic for.

        A converted block does not divide evenly into chunks, and the first one is
        short while the filter fills, so what does not make a whole chunk waits here
        for the next block.
        """
        pending = np.concatenate([self._leftover, audio])
        whole = len(pending) // self.CHUNK_SIZE * self.CHUNK_SIZE
        for start in range(0, whole, self.CHUNK_SIZE):
            self._append(pending[start:start + self.CHUNK_SIZE])
        self._leftover = pending[whole:]


class SdrLevelStream(LevelStream):
    """A live level in dBm, fed by a receiver, for the setup program's meter.

    This owns a thread rather than reaching through RtlSdrPipeline, because a meter
    wants the newest reading rather than a history.  The ring buffer would only add
    its own latency to a number somebody is watching while they turn a knob.

    Everything that turns a block into a reading stays in LevelStream, and none of it
    is overridden here.  That is the point of the split: an operator calibrating
    against this meter has to be calibrating against the figure the monitor itself
    would report.
    """

    def __init__(self, source: RtlSdrSource, converter: 'IqToAudio',
                 offset_db: float) -> None:
        # One IQ block converts to one audio block, so they last the same time and
        # either one gives the smoothing its time constant.
        block_seconds = source.block_samples / source.iq_sample_rate
        super().__init__(offset_db, converter.audio_sample_rate,
                         round(block_seconds * converter.audio_sample_rate))
        self._source = source
        self._converter = converter
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name='sdr-level')
        self._source.start()
        self._thread.start()

    def _drain(self) -> None:  # pragma: no cover -- thread body; _consume is tested directly
        """Read from the receiver until told to stop."""
        while not self._stopping.is_set():
            try:
                block = self._source.read(timeout=_FEED_READ_TIMEOUT_SECONDS)
                if block is not None:
                    self._consume(block)
            except Exception:
                logger.exception(
                    'Converting a block for the level meter failed.  The reading is '
                    'now stale, and the meter keeps running.')

    def _consume(self, block: IqBlock) -> None:
        """Convert one block and fold it into the reading.

        An empty result is normal rather than an error: the filter needs samples it
        has not been given yet, which is always true of the first call.  Passing an
        empty block on would take the median of nothing and poison the reading with
        a NaN that never clears.
        """
        audio = self._converter.convert(block.as_complex())
        if len(audio):
            self._on_block(audio)

    def _stop(self) -> None:
        self._stopping.set()
        self._source.close()
        if self._thread.is_alive():
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

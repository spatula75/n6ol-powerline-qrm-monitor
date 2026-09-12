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
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import numpy as np

from buzz.sampler import RingBufferPipeline

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
    from rtlsdr import RtlSdr
    try:
        return RtlSdr(index)
    except Exception as exc:
        raise RuntimeError(_why_the_receiver_would_not_open(index, exc)) from exc


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
    def iq_sample_rate(self) -> int:
        """The rate the device is actually running at, rounded to whole samples."""
        return self._iq_sample_rate

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

    def close(self) -> None:
        """Stop capture and release the device.

        Cancelling makes read_bytes_async return, which lets the capture thread end,
        and the device is closed after that.  Closing is what leaves the receiver usable
        by the next process; see the module docstring for the measurements.

        Safe to call more than once, and it will be.  The explicit call happens during
        an orderly shutdown, and the atexit hook fires afterwards regardless, so the
        second one has to be a no-op rather than a second attempt at a closed device.
        """
        if self._closed:
            return
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
            self._thread.join(timeout=5.0)
        try:
            self._device.close()
        except Exception:
            logger.debug('Closing the receiver failed during shutdown.', exc_info=True)

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

    def __init__(self, source: RtlSdrSource, converter: 'IqToAudio') -> None:
        super().__init__(converter.audio_sample_rate)
        self._source = source
        self._converter = converter
        self._clipped = 0
        self._leftover = np.empty(0, dtype=np.int16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, daemon=True, name='sdr-feeder')

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
            self._thread.join(timeout=5.0)

    def _feed(self) -> None:  # pragma: no cover -- thread body; _consume is tested directly
        """Drain the source until told to stop."""
        while not self._stop.is_set():
            try:
                block = self._source.read(timeout=0.5)
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

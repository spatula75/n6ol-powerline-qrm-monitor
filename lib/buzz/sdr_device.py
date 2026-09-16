"""The contract for talking to SDR hardware, and the RTL-SDR implementation of it.

This module owns every operation performed against a receiver: opening it, configuring
it, moving its gain, streaming from it, reading one block synchronously, and closing
it.  Nothing above it imports a driver library, so adding a second kind of receiver
means writing another `SdrDevice` rather than editing the code that acquires IQ.

The split from `buzz.sdr` is by subject.  This module reaches the hardware, and
`buzz.sdr` turns what arrives into IQ, audio and measurements.

Three things shape the interface, and each came from the hardware rather than from
taste.

**A device delivers by callback, and the callback must not raise.**  Both librtlsdr
and the SDRplay API call into the program from a thread they own, hand over a buffer
they are about to reuse, and expect it back immediately.  An exception there has
nowhere to go.  So the callback lives here and does the least it can: copy, wrap,
offer, count.  The queue belongs to whoever is consuming, and arrives as a `BlockSink`,
which keeps the depth and the overflow policy with the component that knows what the
ring buffer needs.  `BlockSink.offer` returns a bool rather than raising for the same
reason.

**A device's samples are its own.**  An RTL-SDR gives 8-bit unsigned pairs, and a
14-bit converter will give 16-bit signed ones.  `SampleFormat` carries what it takes to
read either, so `IqBlock` interprets rather than assumes, and a recording can keep the
device's own bytes instead of this program's reading of them.

**Some devices cannot be retuned while they stream.**  Changing an RTL-SDR's gain
during an async read wedged the receiver twice in a few dozen sweeps, and
`docs-notebook/rtl-sdr-hardware.md` records what was tried.  That is a fact about one
driver rather than about SDRs, so `DeviceProfile` states it and each device enforces
its own answer.  The SDRplay API has `sdrplay_api_Update` for exactly this, so another
device may answer differently.
"""


import atexit
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, Self

import numpy as np

logger = logging.getLogger(__name__)

# Values per complex sample, one for I and one for Q.  A property of IQ rather than of
# any device, which is why it is not in SampleFormat: a 16-bit receiver still sends two
# values per sample, it just spends four bytes doing it.
#
# Public because anything counting raw values needs it and must not reach for
# bytes_per_frame instead.  The two are equal only on an 8-bit device, so that
# substitution is correct today and silently doubles on a wider one.
VALUES_PER_FRAME = 2

# How long to wait for a capture thread to leave the driver's read during shutdown.
#
# The thread returns from read_bytes_async once cancel_read_async has taken effect.
# Five seconds is several times that, so a thread still running afterwards is stuck
# rather than slow, and close treats it that way.
_THREAD_JOIN_TIMEOUT_SECONDS = 5.0

# How long to wait for a driver's close to return before giving up on it.
#
# Short, because nothing useful happens after it.  A close that has not returned by now
# is inside the driver and is not coming back, and the operating system releases the
# handle when the process ends.  See RtlSdrDevice.close.
_DEVICE_CLOSE_TIMEOUT_SECONDS = 3.0

# How many transfers librtlsdr keeps in flight.
#
# pyrtlsdr passes DEFAULT_ASYNC_BUF_NUMBER = 0 straight to rtlsdr_read_async and
# librtlsdr substitutes 15, so the pool holds buf_num * block_samples / sample_rate.
# At the default block and 256 kHz that is 960 ms, which matched a hand measurement to
# the digit.  It is a library constant rather than a parameter, so this restates it.
_TRANSFER_POOL_BLOCKS = 15

# What rtlsdr_read_sync wants a read to be a whole number of.  pyrtlsdr admits this in
# a FIXME without enforcing it, and a bad size does not fail loudly: librtlsdr reads
# what it can, pyrtlsdr sees a short read, closes the device, and raises a libusb error
# that says nothing about sizes.
_USB_PACKET_BYTES = 512


@dataclass(frozen=True)
class SampleFormat:
    """How to read one device's raw samples.

    `dtype` and `bytes_per_frame` say how the bytes are laid out, where a frame is one
    complex sample, I then Q.  A sample converts with `raw / half_span - zero_offset`.
    That reaches exactly -1.0 and +1.0 at the rails, whatever the converter's width.

    That expression is pyrtlsdr's own, and it is deliberately not the tidier
    `(raw - midpoint) / half_span`.  The two are the same algebraically, but floating
    point rounds them differently, so the results can disagree in the last bit.
    Golden files downstream pin these samples, so this program keeps the expression it
    has always used.

    The rails are kept as raw values rather than derived, because clipping is counted
    on the raw array.  That keeps it an exact integer comparison instead of a float one
    against 1.0, which is the same question asked less precisely.
    """

    dtype: np.dtype
    bytes_per_frame: int
    half_span: float
    zero_offset: complex
    rail_low: float
    rail_high: float


# pyrtlsdr turns a byte into a sample with (byte / 127.5) - 1, so 0 becomes -1.0 and
# 255 becomes +1.0 and nothing else reaches either.  Restated here rather than called,
# because read_bytes_async is what this module uses and packed_bytes_to_iq is a method
# on a device object the conversion thread has no business touching.
RTL_SDR_FORMAT = SampleFormat(
    dtype=np.dtype(np.uint8),
    bytes_per_frame=2,
    half_span=127.5,
    zero_offset=1 + 1j,
    rail_low=0,
    rail_high=255,
)


@dataclass(frozen=True)
class DeviceProfile:
    """What a device is, as the code above it needs to know.

    A device builds this once it is configured, rather than declaring it as a class
    constant, because a device may answer differently depending on how it was set up.

    There are two discard counts, because the right one depends on how the caller
    reads rather than on the device alone.  Both say how many blocks may predate a
    gain change and so have to be thrown away.  `blocks_to_discard_streaming` covers
    the driver's
    transfer pool, which is full of samples captured before the change.
    `blocks_to_discard_reading` covers a synchronous read, which has no pool at all,
    so only the tuner settling and whatever the USB pipe already held remain.

    Counting blocks rather than waiting a duration keeps both independent of scheduler
    jitter and of the sample rate being what was asked for.  Measuring without the
    discard reads the previous step's answer shifted by one step, which looks like a
    plausible curve and is wrong.

    `gain_changes_while_streaming` says whether the gain may move while the device is
    streaming.  An RTL-SDR answers no, and the hazard behind that answer is measured in
    `docs-notebook/rtl-sdr-hardware.md`.
    """

    name: str
    sample_format: SampleFormat
    blocks_to_discard_streaming: int
    blocks_to_discard_reading: int
    gain_changes_while_streaming: bool


@dataclass(frozen=True)
class IqBlock:
    """One block of raw IQ, as the device delivered it.

    `raw` is the device's own samples, already copied out of the buffer the driver
    reuses, and `fmt` says how to read them.  `arrived_at` is the monotonic clock when
    the callback ran, taken there because that is the only place it is accurate.
    `index` counts blocks the device produced rather than blocks that survived, so a
    gap in the sequence tells a consumer that something was refused.
    """

    raw: np.ndarray
    fmt: SampleFormat
    arrived_at: float
    index: int

    @property
    def samples(self) -> int:
        """How many complex samples this block carries.

        This counts values rather than bytes, because `raw` is a typed array and its
        length is already an element count.  Dividing by bytes_per_frame would be right
        for an 8-bit device by coincidence and wrong for every wider one.
        """
        return len(self.raw) // VALUES_PER_FRAME

    @property
    def clipped_samples(self) -> int:
        """How many raw values in this block sat at the converter's rail.

        Counts I and Q separately, so one sample with both at the rail counts twice.
        The figure is a symptom rather than a measurement, and what it means is that
        the receiver gain is set too high for what the antenna is hearing.

        A clipped arc reads smaller than it truly is, so the events it spoils are the
        loud ones that matter most, and nothing else about the audio looks wrong.
        Measured on an RTL-SDR Blog V4, at maximum gain on a quiet band, the peak
        already reached 0.35 of full scale and left 9 dB for an impulse.
        """
        at_rail = (self.raw == self.fmt.rail_low) | (self.raw == self.fmt.rail_high)
        return int(np.count_nonzero(at_rail))

    def as_complex(self) -> np.ndarray:
        """Return the samples as complex128, with a rail at exactly -1.0 or +1.0.

        A float64 array viewed as complex128 takes consecutive values as the real and
        imaginary parts, and that is how a receiver interleaves I and Q.  The divisor
        and the offset both come from `fmt`.  So a signed 16-bit device and an unsigned
        8-bit one reach the same rails, and `clipped_samples` above means the same
        thing for either.
        """
        paired = self.raw.astype(np.float64).view(np.complex128)
        return paired / self.fmt.half_span - self.fmt.zero_offset


class BlockSink(Protocol):
    """Where a streaming device puts the blocks it has copied.

    The consumer supplies the sink rather than the device making one.  That keeps the
    queue depth, and what happens when it fills, with the component that knows how
    much history the ring buffer needs.
    """

    def offer(self, block: IqBlock) -> bool:
        """Take a block, or return False when there is no room.

        Never raises, because this is called from a driver's own thread where an
        exception has nowhere sensible to go.  A device counts what was refused and
        leaves the reporting to whoever reads that count on an ordinary thread.
        """
        ...


class SdrDevice(ABC):
    """One receiver, and every operation this program performs against it.

    There are two reading modes, and the difference is deliberate rather than
    historical.  `start_stream` misses nothing, which is what the monitor needs.
    `read_block` misses the samples between one call and the next.  That costs a gain
    sweep nothing and would ruin the monitor, and in exchange it runs on the caller's
    thread with no second thread to race.  A device whose gain cannot move during a
    stream says so in its profile, and a gain sweep over such a device has to use
    `read_block`.
    """

    @property
    @abstractmethod
    def profile(self) -> DeviceProfile:
        """What this device is.  Valid once the device is configured."""

    @property
    @abstractmethod
    def iq_sample_rate(self) -> int:
        """The rate the device settled on, rounded to whole samples."""

    @property
    @abstractmethod
    def tuned_hz(self) -> int:
        """Where the device is tuned, which is not the frequency of interest.

        A receiver puts a strong false signal at exactly its tuning frequency, from the
        tuner leaking into its own mixer.  `buzz.iq` tunes to one side and mixes back,
        so that false signal falls outside the measured band.
        """

    @property
    @abstractmethod
    def gain_db(self) -> float:
        """The gain in use."""

    @property
    @abstractmethod
    def supported_gains_db(self) -> list[float]:
        """Every gain this device offers, in the order it reports them."""

    @property
    @abstractmethod
    def blocks_refused(self) -> int:
        """Blocks a sink had no room for.

        The device counts these, because this is where a refusal happens, and the
        consumer reports them, so that no logging runs on a driver's callback thread.
        """

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """Whether a stream is running right now."""

    @abstractmethod
    def set_gain_db(self, gain_db: float) -> float:
        """Move the gain, and return the value actually set.

        A device that snaps a request to a step it offers returns the step.  A device
        whose profile refuses gain changes while streaming raises rather than letting a
        caller find out through a receiver that has stopped answering.
        """

    @abstractmethod
    def start_stream(self, sink: BlockSink, block_samples: int) -> None:
        """Begin delivering blocks of `block_samples` to `sink`.

        The device runs this on a thread of its own, because a driver's read does not
        return until it is cancelled.
        """

    @abstractmethod
    def stop_stream(self) -> bool:
        """Stop delivering, and say whether the delivering thread actually finished.

        False means the thread is still inside the driver, which is what decides
        whether the handle may be released.
        """

    @abstractmethod
    def read_block(self, block_samples: int) -> IqBlock | None:
        """Read one block on the calling thread, or None once the device has stopped.

        This is not valid while a stream is running.
        """

    @abstractmethod
    def close(self) -> bool:
        """Release the device, and say whether it actually closed."""


class RtlSdrHandle(Protocol):
    """The part of pyrtlsdr's RtlSdr that RtlSdrDevice uses.

    This is declared so a test can supply something else, which is what lets the shim
    be exercised with no receiver attached.
    """

    sample_rate: float
    center_freq: float
    gain: float
    valid_gains_db: list[float]

    def set_agc_mode(self, enabled: bool) -> int:
        ...

    def read_bytes(self, num_bytes: int) -> object:
        ...

    def read_bytes_async(self, callback: object, num_bytes: int) -> None:
        ...

    def cancel_read_async(self) -> None:
        ...

    def close(self) -> None:
        ...


class RtlSdrDevice(SdrDevice):
    """An RTL-SDR, reached through pyrtlsdr.

    **Get one from `RtlSdrDevice.open`.**  The constructor takes a handle that is
    already open, so that a test can inject one and exercise every path here with no
    receiver attached.  Python cannot stop anybody calling it directly, so this says
    instead which of the two is the contract.  `open` is the one that turns a driver
    failure into a message an operator can act on, and it is the only place the
    pyrtlsdr object is created.
    """

    @classmethod
    def open(cls, index: int = 0, *, tuned_hz: int, gain_db: float,
             iq_sample_rate: int) -> Self:
        """Open the receiver at `index`, configure it, and return it ready to read.

        This is a classmethod rather than a static one, because an alternative
        constructor needs `cls` to be inheritable.  A subclass then gets its own type
        back and can reword the failure below.  A hardcoded class name would hand every
        subclass an RtlSdrDevice and this module's diagnostics instead.

        The import sits inside this method so that a station using a sound card never
        loads pyrtlsdr.  pyrtlsdr looks up rtlsdr_set_dithering as it imports, so a
        mismatched librtlsdr fails the import rather than the first call.  buzz.render
        takes the same approach with ffmpeg for the same reason.

        Everything raised here is a RuntimeError carrying a message for the operator,
        which is what lets main.py print one rather than a traceback.

        Nothing tries to take the receiver from whoever already holds it.  A device
        cannot be closed without the handle that opened it, and every cause of a
        failure here is a case where taking it would be wrong.
        """
        try:
            from rtlsdr import RtlSdr
        except ImportError as exc:
            raise RuntimeError(
                f'The pyrtlsdr library would not load ({exc}), and [audio] source is '
                'set to rtlsdr.  Either it is not installed, or its bundled librtlsdr '
                'is too old to carry the symbol it looks up as it imports.  Run pip '
                'install "pyrtlsdr[lib]", or set [audio] source back to soundcard.'
            ) from exc
        try:
            handle = RtlSdr(index)
        except Exception as exc:
            raise RuntimeError(
                cls._why_the_receiver_would_not_open(index, exc)) from exc
        return cls(handle, tuned_hz=tuned_hz, gain_db=gain_db,
                   iq_sample_rate=iq_sample_rate)

    def __init__(self, handle: RtlSdrHandle, *, tuned_hz: int, gain_db: float,
                 iq_sample_rate: int) -> None:
        self._handle = handle
        self._tuned_hz = tuned_hz
        self._blocks_refused = 0
        self._produced = 0
        self._closed = False
        self._released = False
        self._sink: BlockSink | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

        self._iq_sample_rate, self._gain_db = self._configure(gain_db, iq_sample_rate)
        self._profile = DeviceProfile(
            name='RTL-SDR',
            sample_format=RTL_SDR_FORMAT,
            # Up to _TRANSFER_POOL_BLOCKS buffers are filled or in flight when the gain
            # moves, plus the one being written at that moment, so discarding this many
            # makes every later block provably post-change.
            blocks_to_discard_streaming=_TRANSFER_POOL_BLOCKS + 1,
            # Two, where streaming needs seventeen.  rtlsdr_read_sync takes what the
            # device has now, so there is no pool to drain.  One block covers the tuner
            # settling and the pipe, and this is deliberately one more than that.
            blocks_to_discard_reading=2,
            # Measured, not assumed.  See docs-notebook/rtl-sdr-hardware.md.
            gain_changes_while_streaming=False,
        )
        # A receiver that is never closed keeps streaming with nothing collecting from
        # it.  See https://github.com/librtlsdr/librtlsdr/issues/116
        #
        # Registered after configuring, so a device that failed to configure is not
        # left with a hook pointing at a half-built object.
        atexit.register(self.close)

    # ------------------------------------------------------------------ public

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def iq_sample_rate(self) -> int:
        return self._iq_sample_rate

    @property
    def tuned_hz(self) -> int:
        return self._tuned_hz

    @property
    def gain_db(self) -> float:
        """The gain that was set, which is the only figure we can know.

        Nothing reads it back.  Measured on an RTL-SDR Blog V4, the setter works and
        the level moves by 57.5 dB over the full range, while the getter returns 0.0 at
        every setting.
        """
        return self._gain_db

    @property
    def supported_gains_db(self) -> list[float]:
        return list(self._handle.valid_gains_db)

    @property
    def blocks_refused(self) -> int:
        return self._blocks_refused

    @property
    def is_streaming(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_gain_db(self, gain_db: float) -> float:
        """Snap the request to a step the tuner offers, set it, and return the step.

        This snaps the request rather than leaving it to the driver, because a V4
        cannot report its own gain.  The figure we chose is then the only one anybody
        will ever know, and it is what a recording's metadata has to carry.

        It refuses while streaming.  Setting a gain is a pair of synchronous USB control
        transfers, and the capture thread is inside rtlsdr_read_async driving libusb's
        event loop on the same device.  Two threads touching one device ended twice in
        a few dozen sweeps with a transfer that never completed and an rtlsdr_close that
        never returned.  See docs-notebook/rtl-sdr-hardware.md, which also records why
        writing the gain from inside the callback is worse rather than better.
        """
        if self.is_streaming:
            raise RuntimeError(
                'The receiver gain cannot move while it is streaming, because the gain '
                'is a USB control transfer and the capture thread already has the '
                'device.  Doing it anyway has left a receiver that never answers again.  '
                'Read with read_block instead, which uses no second thread.')
        self._gain_db = self.nearest_supported_gain(gain_db, self.supported_gains_db)
        self._handle.gain = self._gain_db
        return self._gain_db

    def start_stream(self, sink: BlockSink, block_samples: int) -> None:
        """Begin capture on a thread of its own.

        read_bytes_async does not return until the read is cancelled, so it cannot run
        on the caller's thread.
        """
        self._sink = sink
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, args=(block_samples,), daemon=True, name='rtlsdr')
        self._thread.start()

    def stop_stream(self) -> bool:
        """Cancel the async read and wait for the capture thread to leave it.

        Returns whether the thread actually stopped.  A thread that has not stopped is
        still inside the driver's own read, which is what decides whether the handle
        may be closed.
        """
        self._stopping.set()
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._thread = None
            return True
        try:
            self._handle.cancel_read_async()
        except Exception:
            logger.debug('Cancelling the receiver read failed during shutdown.',
                         exc_info=True)
        thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            return False
        self._thread = None
        return True

    def read_block(self, block_samples: int) -> IqBlock | None:
        """One block, read on the calling thread, or None once the device has stopped.

        rtlsdr_read_sync completes its bulk transfer before returning, so there is no
        outstanding transfer for rtlsdr_close to wait on and no second thread to race.
        What it gives up is continuity, since samples between one read and the next are
        missed.

        pyrtlsdr closes the device itself on a short read or a libusb error, so a
        failure here is the end of the session rather than something to retry.  This
        reports it once and returns None from then on.
        """
        self.validate_sync_block(block_samples)
        if self._closed:
            return None
        try:
            buffer = self._handle.read_bytes(
                block_samples * self._profile.sample_format.bytes_per_frame)
        except Exception:
            self._closed = True
            logger.warning(
                'Reading from the receiver failed, and the library closes the device '
                'on any read error, so this read cannot continue.  Whatever was '
                'measured before this point is still used.', exc_info=True)
            return None
        return self._block_from(buffer)

    def close(self) -> bool:
        """Release the receiver, and stop waiting if the driver never comes back.

        rtlsdr_close blocks inside libusb when transfers were never fully cancelled,
        and then it never returns.  So this runs the close on a daemon thread, which
        costs nothing at exit because Python does not join one.  The receiver stays
        held until the process ends, which is where a blocked close left it anyway.
        What that buys is a program that carries on and reports the receiver as still
        held, instead of leaving somebody to meet LIBUSB_ERROR_ACCESS on the next run
        and read it as a permissions problem.

        This is safe to call twice, and it will be called twice.  Shutdown calls it
        explicitly, and the atexit hook fires afterwards regardless, so the second
        call has to do nothing rather than attempt a device that is already closed.

        Returns whether the device was actually released.  False means it is still
        held, and the consequence falls on whatever opens a receiver next: libusb
        refuses with LIBUSB_ERROR_ACCESS, which reads as a permissions problem and is
        not one.
        """
        if self._closed:
            return self._released
        self._closed = True
        atexit.unregister(self.close)
        # A join that timed out leaves the capture thread inside librtlsdr's own read.
        # Closing now would free the handle it is reading through, which is a crash in
        # C rather than an exception here.  The cost of skipping the close is measured
        # in docs-notebook/rtl-sdr-hardware.md, and the answer is nothing.
        if not self.stop_stream():
            logger.warning(
                'The receiver capture thread did not stop within %.0f seconds, so the '
                'device was left open.  Closing it now would free a handle that thread '
                'is still reading through.  The operating system releases it when this '
                'process ends.', _THREAD_JOIN_TIMEOUT_SECONDS)
            return False
        self._released = self._close_handle()
        return self._released

    # ------------------------------------------------------------------ static

    @staticmethod
    def _why_the_receiver_would_not_open(index: int, exc: Exception) -> str:
        """Turn a libusb failure into something that names what to try.

        This sits apart from `open` so the wording can be read and tested without a
        receiver, and so the two likely causes stay side by side where they can be
        compared.  libusb's own wording sends people the wrong way: "Entity not found"
        reads like a missing library and means no driver is bound to the device.
        """
        if getattr(exc, 'errno', None) == -5:        # LIBUSB_ERROR_NOT_FOUND
            return (
                f'Receiver {index} was found but no driver is bound to it ({exc}).  On '
                'Windows this means Zadig has not been run for this device on this USB '
                'port.  Run Zadig as administrator, tick Options then List All Devices, '
                'select "Bulk-In, Interface (Interface 0)", and install WinUSB.  Do not '
                'replace librtlsdr.dll, which is the usual advice and is wrong here.')
        return (
            f'Receiver {index} could not be opened ({exc}).  Either something else is '
            'using it, such as another copy of this monitor or an SDR application, or '
            'no receiver is plugged in.  Close whatever holds it, or check the cable, '
            'and start again.')

    @staticmethod
    def nearest_supported_gain(gain_db: float, supported: list[float]) -> float:
        """The value from `supported` closest to `gain_db`."""
        return min(supported, key=lambda candidate: abs(candidate - gain_db))

    @staticmethod
    def validate_sync_block(block_samples: int) -> None:
        """Refuse a synchronous read size librtlsdr cannot serve exactly.

        rtlsdr_read_sync wants a whole number of 512-byte USB packets.  A bad size does
        not fail loudly, so this refuses one before the device is touched.
        """
        wanted = _USB_PACKET_BYTES // RTL_SDR_FORMAT.bytes_per_frame
        if block_samples <= 0 or block_samples % wanted:
            raise ValueError(
                f'A synchronous block of {block_samples} samples is '
                f'{block_samples * RTL_SDR_FORMAT.bytes_per_frame} bytes, and the '
                f'receiver reads whole {_USB_PACKET_BYTES}-byte USB packets.  Use a '
                f'multiple of {wanted} samples.')

    # ----------------------------------------------------------------- private

    def _configure(self, gain_db: float, iq_sample_rate: int) -> tuple[int, float]:
        """Set the rate, the tuning and the gain, and turn the digital AGC off.

        The RTL2832U has a digital AGC of its own, separate from the tuner's manual
        gain, and it is off by default only by convention.  An AGC riding on the
        impulses would compress exactly what this program measures while leaving the
        noise floor looking healthy, so it is disabled explicitly.

        This reads the rate back, because the device derives it from a 28.8 MHz divider
        and cannot hit every request.  Measured on this hardware, 256000 comes back
        exactly, where 250000 comes back as 250000.000414.
        """
        self._handle.sample_rate = iq_sample_rate
        actual = float(self._handle.sample_rate)
        settled = int(round(actual))
        if settled != iq_sample_rate:
            logger.warning(
                'Asked the receiver for %d Hz and got %.6f Hz.  Everything downstream '
                'will treat the audio as %d Hz.  A rate the hardware cannot produce '
                'exactly is normal, and the difference here is %.1f ppm.',
                iq_sample_rate, actual, settled,
                abs(actual - iq_sample_rate) / iq_sample_rate * 1e6)

        self._handle.center_freq = self._tuned_hz
        self._handle.set_agc_mode(False)

        gain = self.nearest_supported_gain(gain_db, list(self._handle.valid_gains_db))
        self._handle.gain = gain
        if gain != gain_db:
            logger.info('Receiver gain %.1f dB is not one the tuner offers, so %.1f dB '
                        'was used instead.', gain_db, gain)
        return settled, gain

    def _block_from(self, buffer: object) -> IqBlock:
        """Copy a driver buffer into a block, because the driver reuses it."""
        raw = np.ctypeslib.as_array(buffer).astype(
            self._profile.sample_format.dtype, copy=True)
        self._produced += 1
        return IqBlock(raw=raw, fmt=self._profile.sample_format,
                       arrived_at=monotonic(), index=self._produced)

    def _run(self, block_samples: int) -> None:  # pragma: no cover -- thread body
        """Capture thread body.  read_bytes_async blocks here until cancelled."""
        try:
            self._handle.read_bytes_async(
                self._on_block,
                block_samples * self._profile.sample_format.bytes_per_frame)
        except Exception:
            if not self._stopping.is_set():
                logger.exception(
                    'The receiver stopped delivering samples.  Capture has ended and '
                    'will not restart on its own, because the device cannot be '
                    'reopened promptly.  Restart the monitor to try again.')

    def _on_block(self, buffer: object, _context: object = None) -> None:
        """Take one block from the device, on librtlsdr's own thread.

        Nothing here is allowed to raise, because this is called from C where an
        exception has nowhere sensible to go.  It does the least it can: copy the
        buffer the driver is about to reuse, wrap it, offer it, and count a refusal.
        The reporting belongs to whoever reads blocks_refused on an ordinary thread,
        because logging can raise and can block on I/O, and the receiver's own FIFO
        holds 3.67 ms at 256 kHz.
        """
        try:
            block = self._block_from(buffer)
            sink = self._sink
            if sink is None or not sink.offer(block):
                self._blocks_refused += 1
        except Exception:  # pragma: no cover -- the last resort in a C callback
            self._blocks_refused += 1

    def _close_handle(self) -> bool:
        """Call the driver's close on a thread, and give up on it after a timeout."""
        finished = threading.Event()

        def shut() -> None:
            try:
                self._handle.close()
            except Exception:
                logger.debug('Closing the receiver failed.', exc_info=True)
            finally:
                finished.set()

        threading.Thread(target=shut, daemon=True, name='rtlsdr-close').start()
        if finished.wait(timeout=_DEVICE_CLOSE_TIMEOUT_SECONDS):
            return True
        logger.warning(
            'The receiver did not close within %.0f seconds and was left to the '
            'operating system.  The driver can block inside libusb and never return, '
            'so waiting longer would only hang this program.',
            _DEVICE_CLOSE_TIMEOUT_SECONDS)
        return False

"""An SDRplay RSP1A or RSP1B, behind the `SdrDevice` contract.

`buzz.sdr_device` states what a receiver has to do and implements it for an RTL-SDR.
This module does the same job for an SDRplay RSP, over the generated ctypes bindings in
`buzz.sdrplay_api`.  Nothing above either module knows which one it holds.

Four things differ from an RTL-SDR, and each one shapes the code below.

**The samples arrive in two arrays.**  An RTL-SDR interleaves I and Q in one buffer,
which is also how a `.wav` frame is laid out, so `SampleFormat` assumes interleaved.
The SDRplay hands over `short *xi` and `short *xq`, so `_on_stream` interleaves while
it copies.  The copy has to happen anyway, because the library reuses both buffers.

**The gain is two knobs.**  `gRdB` is baseband gain reduction from 20 to 59 dB, and
`LNAstate` picks a front-end reduction out of a table that depends on the band.  Both
reduce gain, so a larger number is less gain.  `_knobs_for` turns one figure from the
ladder back into the pair, and `supported_gains_db` is the ladder.

**The library never reads on the caller's thread.**  There is no synchronous call at
all, so `read_block` runs a stream of its own and takes one block from it.  A gain
sweep sees the same interface either way, and `buzz.sdr.SweepReader` needs no change.

**The hardware says when the gain moved.**  Every block carries `grChanged`, which is
set on the block where a gain change took effect.  So this device drops stale blocks by
reading that flag rather than by counting, which is what an RTL-SDR has to do.

**This has never run on Linux.**  The bindings are correct there by construction, and
`docs-notebook/todo.md` says what that rests on and what nobody has checked.
"""


import atexit
import ctypes
import logging
import queue
import threading
from pathlib import Path
from time import monotonic
from typing import Protocol, Self

import numpy as np

from buzz import sdrplay_api as api
from buzz.sdr_device import VALUES_PER_FRAME, BlockSink, DeviceProfile, IqBlock, SampleFormat, SdrDevice

logger = logging.getLogger(__name__)

# The library file, under every name it goes by.  Windows installs `sdrplay_api.dll`.
# Linux installs `libsdrplay_api.so.3` and makes `libsdrplay_api.so` a symlink beside
# it, which a runtime-only install may not have, so both spellings are tried.
LIBRARY_NAMES = ('sdrplay_api.dll', 'libsdrplay_api.so', 'libsdrplay_api.so.3')

# Where an installer puts the library, for the case where it is not on the search path.
# Windows puts it under Program Files with the API version in the directory name, and
# the Linux installer puts it in /usr/local/lib.  The loader tries these after the
# plain names, so a library it can already find always wins.
KNOWN_LIBRARY_DIRECTORIES = (
    Path(r'C:\Program Files\SDRplay\API\x64'),
    Path('/usr/local/lib'),
    Path('/usr/lib'),
)

# How the RSP's samples are read.  The library delivers signed 16-bit values, so the
# rails are the ends of that range.
#
# Whether a real RSP reaches them has not been checked on hardware.  The converter is
# narrower than 16 bits and the library decimates for us, which adds bits back, so the
# decimated stream may or may not reach the ends of the container.  `clipped_samples`
# is what a gain sweep uses to reject a gain, so a rail that never occurs would let the
# sweep pick a gain that clips.  The overload event below is the hardware's own answer
# to the same question and does not depend on this figure.  See `docs-notebook`.
SDRPLAY_FORMAT = SampleFormat(
    dtype=np.dtype(np.int16),
    bytes_per_frame=VALUES_PER_FRAME * 2,
    half_span=32768.0,
    zero_offset=0 + 0j,
    rail_low=-32768,
    rail_high=32767,
)

# LNA gain reduction in dB, by LNAstate, for an RSP1A or an RSP1B in its lowest band.
# Copied from the gain reduction tables in section 5 of
# SDRplay_API_Specification_v3.15.pdf, where both receivers state this same row.
#
# The table is not the authority at run time.  `set_gain_db` compares what this
# predicts against the `gainVals.curr` the library fills in, and logs once when the two
# disagree, so a table that goes stale reports itself rather than quietly shifting
# every measurement by a fixed amount.
HF_LNA_GAIN_REDUCTION_DB = (0, 6, 12, 18, 37, 42, 61)

# Where that row stops, by hwVer.  Both receivers move to a ten-entry row with
# different figures above the edge, and an RSP1B changes over 10 MHz lower than an
# RSP1A does.  This program measures powerline QRM on HF, so a tuning above the edge is
# refused rather than served from a table nobody here can check.
HF_LNA_MAX_HZ = {
    api.SDRPLAY_RSP1B_ID: 50_000_000,
    api.SDRPLAY_RSP1A_ID: 60_000_000,
}

# The baseband gain reduction range, in dB.  The minimum is the API's own
# NORMAL_MIN_GR, which is the setting this device uses: EXTENDED_MIN_GR opens up 0 to
# 19 dB as well and SDRplay document it as usable only on some bands.
MIN_GAIN_REDUCTION_DB = int(api.sdrplay_api_MinGainReductionT.sdrplay_api_NORMAL_MIN_GR)
MAX_GAIN_REDUCTION_DB = api.MAX_BB_GR

# The converter's lowest rate, and what the library will decimate by.  An RSP cannot
# sample at the rates this program wants, so it samples fast and decimates, and
# `_rate_plan` picks the pair.  Both figures come from the API specification.
MIN_ADC_RATE_HZ = 2_000_000
DECIMATION_FACTORS = (1, 2, 4, 8, 16, 32)

# How many blocks the synchronous reader will hold before it starts refusing.
#
# Two, because a synchronous reader wants the newest samples rather than a backlog, and
# one block in hand while the next arrives is what keeps `read_block` from waiting a
# whole block every time.  A gain sweep throws away most of what it reads anyway.
_SYNC_QUEUE_BLOCKS = 2

# How long `read_block` waits for the internal stream to produce one.
#
# A block is `block_samples / iq_sample_rate` seconds, which is 8 ms for the sweep's
# default at 256 kHz, so five seconds is several hundred times the expected wait.  A
# read that takes this long means the library has stopped delivering.
_SYNC_READ_TIMEOUT_SECONDS = 5.0

# How long to wait for the library's own uninit and close during shutdown.  Short,
# because nothing useful happens after it, and the same reasoning as
# `RtlSdrDevice.close`: a call that has not returned by now is inside the library.
_DEVICE_CLOSE_TIMEOUT_SECONDS = 3.0

# Blocks for a consumer to throw away after a gain change, on top of what this device
# has already dropped.
#
# This device drops every block up to and including the one the library marks with
# `grChanged`, so by the time a consumer sees anything the change has provably taken
# effect.  One is a backstop rather than a measurement, and it is here because the
# alternative reading is that `grChanged` behaves differently from how the
# specification describes it, which nobody here has checked against hardware.
_BLOCKS_TO_DISCARD_AFTER_GAIN_CHANGE = 1


class SdrplayApi(Protocol):
    """The SDRplay API calls this device makes, with every failure already raised.

    The library returns an error code from each call and this layer turns one into an
    exception, so the device below reads as a procedure rather than as a check after
    every line.  It is declared as a Protocol so that a test can supply something else,
    which is what lets the device be exercised with no receiver attached.

    The names here are the API's own with the prefix removed, so that anybody holding
    the specification can follow which call each one makes.
    """

    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def api_version(self) -> float:
        ...

    def lock(self) -> None:
        ...

    def unlock(self) -> None:
        ...

    def devices(self) -> list[api.sdrplay_api_DeviceT]:
        ...

    def select(self, device: api.sdrplay_api_DeviceT) -> None:
        ...

    def release(self, device: api.sdrplay_api_DeviceT) -> None:
        ...

    def device_params(self, handle: int) -> api.sdrplay_api_DeviceParamsT:
        ...

    def init(self, handle: int, callbacks: api.sdrplay_api_CallbackFnsT) -> None:
        ...

    def uninit(self, handle: int) -> None:
        ...

    def update(self, handle: int, tuner: int, reason: int) -> None:
        ...


class SdrplayLibrary:
    """`SdrplayApi` over the real shared library.

    **Get one from `SdrplayLibrary.load`.**  The constructor takes a `ctypes.CDLL` that
    is already open, which is what lets a test drive this layer itself rather than only
    the device above it.

    Every method raises a RuntimeError carrying the library's own error string, so that
    `buzz.main` can print a message rather than a traceback.
    """

    @classmethod
    def load(cls, api_path: Path | str | None = None) -> Self:
        """Find the shared library, load it, and declare every call it exports.

        The search order is the setting first, then each plain name, then the places an
        installer puts it.  The setting comes first so that an operator with two copies
        installed can say which one to use, and the plain names come before the known
        directories so that a library already on the search path wins over a guess
        about where it lives.
        """
        for candidate in cls._candidates(api_path):
            try:
                library = ctypes.CDLL(str(candidate))
            except OSError:
                continue
            logger.debug('Loaded the SDRplay API from %s.', candidate)
            return cls(library)
        raise RuntimeError(cls._why_the_library_would_not_load(api_path))

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        for symbol, restype, argtypes in api.FUNCTIONS:
            try:
                function = getattr(library, symbol)
            except AttributeError as exc:
                raise RuntimeError(
                    f'The SDRplay library does not export {symbol}.  This program '
                    f'needs API version {api.API_VERSION}, and an older one is the '
                    f'likely cause.  Install the current API from '
                    f'https://sdrplay.com/hardware-api/.') from exc
            function.restype = restype
            function.argtypes = argtypes

    # ------------------------------------------------------------------ public

    def open(self) -> None:
        self._check('sdrplay_api_Open', self._library.sdrplay_api_Open())

    def close(self) -> None:
        self._check('sdrplay_api_Close', self._library.sdrplay_api_Close())

    def api_version(self) -> float:
        version = ctypes.c_float()
        self._check('sdrplay_api_ApiVersion',
                    self._library.sdrplay_api_ApiVersion(ctypes.byref(version)))
        return float(version.value)

    def lock(self) -> None:
        self._check('sdrplay_api_LockDeviceApi',
                    self._library.sdrplay_api_LockDeviceApi())

    def unlock(self) -> None:
        self._check('sdrplay_api_UnlockDeviceApi',
                    self._library.sdrplay_api_UnlockDeviceApi())

    def devices(self) -> list[api.sdrplay_api_DeviceT]:
        """Every receiver the service can see, as the library filled them in.

        This sizes the array by the API's own maximum rather than by a guess, and the
        count comes back in `found`, so a receiver plugged in later simply shows up on
        the next call.
        """
        found = ctypes.c_uint(0)
        devices = (api.sdrplay_api_DeviceT * api.SDRPLAY_MAX_DEVICES)()
        self._check('sdrplay_api_GetDevices', self._library.sdrplay_api_GetDevices(
            devices, ctypes.byref(found), api.SDRPLAY_MAX_DEVICES))
        return [devices[index] for index in range(found.value)]

    def select(self, device: api.sdrplay_api_DeviceT) -> None:
        self._check('sdrplay_api_SelectDevice',
                    self._library.sdrplay_api_SelectDevice(ctypes.byref(device)))

    def release(self, device: api.sdrplay_api_DeviceT) -> None:
        self._check('sdrplay_api_ReleaseDevice',
                    self._library.sdrplay_api_ReleaseDevice(ctypes.byref(device)))

    def device_params(self, handle: int) -> api.sdrplay_api_DeviceParamsT:
        """The parameter block for a selected device.

        The library owns this memory and hands back a pointer into it, so this program
        changes a setting by writing through the object that comes back.  Copying it
        would change nothing on the device.
        """
        params = ctypes.POINTER(api.sdrplay_api_DeviceParamsT)()
        self._check('sdrplay_api_GetDeviceParams',
                    self._library.sdrplay_api_GetDeviceParams(
                        handle, ctypes.byref(params)))
        return params.contents

    def init(self, handle: int, callbacks: api.sdrplay_api_CallbackFnsT) -> None:
        self._check('sdrplay_api_Init', self._library.sdrplay_api_Init(
            handle, ctypes.byref(callbacks), None))

    def uninit(self, handle: int) -> None:
        self._check('sdrplay_api_Uninit', self._library.sdrplay_api_Uninit(handle))

    def update(self, handle: int, tuner: int, reason: int) -> None:
        self._check('sdrplay_api_Update', self._library.sdrplay_api_Update(
            handle, tuner, reason, api.sdrplay_api_ReasonForUpdateExtension1T
            .sdrplay_api_Update_Ext1_None))

    # ----------------------------------------------------------------- private

    def _check(self, name: str, code: int) -> None:
        """Turn a non-zero return into an exception carrying the library's own words."""
        if code == api.sdrplay_api_ErrT.sdrplay_api_Success:
            return
        raise RuntimeError(
            f'{name} failed: {self._error_string(code)}.  '
            f'{_what_an_api_failure_usually_means(code)}')

    def _error_string(self, code: int) -> str:
        """The library's text for an error code, or the code when it has none."""
        try:
            text = self._library.sdrplay_api_GetErrorString(code)
        except Exception:  # pragma: no cover -- a library too old to have the call
            return f'error {code}'
        return text.decode('utf-8', 'replace') if text else f'error {code}'

    @staticmethod
    def _candidates(api_path: Path | str | None) -> list[Path]:
        """Every place to try, in order, with no duplicates."""
        found: list[Path] = []
        if api_path:
            found.append(Path(api_path))
        found += [Path(name) for name in LIBRARY_NAMES]
        found += [directory / name
                  for directory in KNOWN_LIBRARY_DIRECTORIES
                  for name in LIBRARY_NAMES]
        seen: set[str] = set()
        return [path for path in found
                if str(path) not in seen and not seen.add(str(path))]

    @staticmethod
    def _why_the_library_would_not_load(api_path: Path | str | None) -> str:
        """Name the missing library, where it comes from, and the setting that finds it."""
        if api_path:
            return (
                f'The SDRplay API library was not at {api_path}, which is where '
                f'[sdrplay] api_path points.  Correct that setting, or remove it and '
                f'let this program search the usual places.')
        return (
            'The SDRplay API library was not found.  Install the SDRplay Hardware API '
            'from https://sdrplay.com/hardware-api/, listed under "Other" on that '
            'site.  SDRconnect on its own is not enough, because it speaks to the '
            'receiver directly and installs neither the API nor its service.  Set '
            '[sdrplay] api_path if it is installed somewhere unusual.')


class SdrplayDevice(SdrDevice):
    """An SDRplay RSP1A or RSP1B, reached through the SDRplay Hardware API.

    **Get one from `SdrplayDevice.open`.**  The constructor takes a library and a
    device the caller has already selected, so that a test can inject both and exercise
    every path here with no receiver attached.  `open` is the one that turns a library
    failure into a message an operator can act on.

    This admits the RSP1A and the RSP1B and nothing else.  Every other RSP needs a
    different LNA table, and two of them embed extra parameter structs this device
    never writes, so admitting one would claim support nobody here can test.
    """

    @classmethod
    def open(cls, index: int = 0, *, tuned_hz: int, gain_db: float,
             iq_sample_rate: int, api_path: Path | str | None = None) -> Self:
        """Open the receiver at `index`, configure it, and return it ready to read.

        This is a classmethod rather than a static one, for the reason
        `RtlSdrDevice.open` gives: a subclass then gets its own type back.

        The API is locked around finding and selecting a receiver, because the service
        is shared with every other program on the machine and the window between
        listing and selecting is where two of them collide.
        """
        library = SdrplayLibrary.load(api_path)
        library.open()
        try:
            device = cls._select(library, index)
        except BaseException:
            # Nothing owns the API session yet, so it has to be given back here.  A
            # session left open keeps the service holding the receiver, and the next
            # run meets a device that is present and cannot be selected.
            _quietly(library.close)
            raise
        try:
            return cls(library, device, tuned_hz=tuned_hz, gain_db=gain_db,
                       iq_sample_rate=iq_sample_rate)
        except BaseException:
            _quietly(lambda: library.release(device))
            _quietly(library.close)
            raise

    def __init__(self, library: SdrplayApi, device: api.sdrplay_api_DeviceT, *,
                 tuned_hz: int, gain_db: float, iq_sample_rate: int) -> None:
        self._library = library
        self._device = device
        self._tuned_hz = tuned_hz
        self._blocks_refused = 0
        self._produced = 0
        self._overloads = 0
        self._closed = False
        self._released = False
        self._initialized = False
        self._sink: BlockSink | None = None
        self._sync_sink: _SyncBlocks | None = None
        self._gain_table_checked = False
        # Blocks to drop before anything is offered, because a gain change is in
        # flight.  The block the library marks with `grChanged` clears it.
        self._awaiting_gain_change = False
        # Partly filled block, as arrays in arrival order, with their total length.
        # Only the stream callback touches either.
        self._pending: list[np.ndarray] = []
        self._pending_values = 0
        self._block_values = 0
        self._lock = threading.Lock()

        self._check_band()
        self._params = library.device_params(int(device.dev))
        self._iq_sample_rate = self._configure(iq_sample_rate)
        self._gain_db = self._write_gain(self.nearest_supported_gain(
            gain_db, self.supported_gains_db))
        self._profile = DeviceProfile(
            name=f'SDRplay {"RSP1B" if device.hwVer == api.SDRPLAY_RSP1B_ID else "RSP1A"}',
            sample_format=SDRPLAY_FORMAT,
            blocks_to_discard_streaming=_BLOCKS_TO_DISCARD_AFTER_GAIN_CHANGE,
            blocks_to_discard_reading=_BLOCKS_TO_DISCARD_AFTER_GAIN_CHANGE,
            # The API has sdrplay_api_Update for exactly this, so a gain change during
            # a stream is the supported order rather than the hazard it is on an
            # RTL-SDR.
            gain_changes_while_streaming=True,
        )
        # A session left open keeps the service holding the receiver.  This registers
        # after configuring, so a device that failed to configure leaves no hook
        # pointing at a half-built object.
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
        """The gain in use, as the receiver reports it.

        This is the hardware's own figure rather than the one that was asked for, which
        is the difference from an RTL-SDR: a V4 cannot report its gain at all, so
        `RtlSdrDevice` can only return what it wrote.
        """
        return self._gain_db

    @property
    def supported_gains_db(self) -> list[float]:
        """Every gain this receiver offers on HF, from the most to the least.

        Each figure is the negative of a total gain reduction, so the list runs from
        -20 dB down to -120 dB in one-dB steps.  Two knobs reach most of those totals
        more than one way, and `_knobs_for` picks which pair to use.
        """
        return [float(-total) for total in self._gain_reduction_ladder()]

    @property
    def blocks_refused(self) -> int:
        return self._blocks_refused

    @property
    def overloads(self) -> int:
        """How many times the receiver has reported its front end overloading.

        The library raises this as an event, so it is the hardware's own answer to the
        question `clipped_samples` asks by looking at the samples.  It does not depend
        on where the rails of the decimated stream sit, which nobody here has measured.
        """
        return self._overloads

    @property
    def is_streaming(self) -> bool:
        """Whether a caller's stream is running.

        A synchronous read runs a stream of its own and this stays False through it,
        because what this answers is whether `start_stream` has been called and not
        whether the library is delivering.
        """
        return self._sink is not None

    def set_gain_db(self, gain_db: float) -> float:
        """Move the gain, and return the figure the receiver reports afterwards.

        A request is snapped to the ladder, split back into a gain reduction and an
        LNA state, and written through `sdrplay_api_Update`.  That is the API's
        supported way to change gain while the receiver streams, so this does not
        refuse the way an RTL-SDR has to.

        The receiver fills in `gainVals.curr` and that figure is what comes back, so
        the answer is measured rather than assumed.
        """
        if self._closed:
            raise RuntimeError(
                'The receiver gain cannot move because the device is closed.  Open it '
                'again with SdrplayDevice.open.')
        wanted = self.nearest_supported_gain(gain_db, self.supported_gains_db)
        # Armed only when the library is running, because only then does an update go
        # out and only then will a block come back marked grChanged.  Arming it while
        # the receiver is idle would drop every block of the stream that follows, with
        # nothing ever arriving to clear it.
        self._awaiting_gain_change = self._initialized
        self._gain_db = self._write_gain(wanted)
        return self._gain_db

    def start_stream(self, sink: BlockSink, block_samples: int) -> None:
        """Begin delivering blocks of `block_samples` to `sink`.

        The library runs its own thread, so there is none to start here.  What this
        does is tell the callback where to put what arrives.
        """
        if self._closed:
            raise RuntimeError(
                'The receiver cannot start streaming because it is closed.  Open it '
                'again with SdrplayDevice.open.')
        if self._sink is not None:
            raise RuntimeError(
                'The receiver is already streaming, and one device cannot serve two '
                'sinks.  Call stop_stream first.')
        if self._sync_sink is not None:
            raise RuntimeError(
                'The receiver is serving synchronous reads, and one device cannot do '
                'both at once.  Finish with read_block first.')
        self._sink = sink
        self._begin(block_samples)

    def stop_stream(self) -> bool:
        """Stop delivering, and say whether the library actually stopped.

        The library's own thread is what `sdrplay_api_Uninit` waits for, so a False
        here means that call did not return rather than that a thread of ours is stuck.
        """
        if self._sink is None:
            return True
        self._sink = None
        return self._end()

    def read_block(self, block_samples: int) -> IqBlock | None:
        """One block, or None once the receiver has stopped answering.

        There is no synchronous call in this API, so the first read starts a stream
        that feeds a short queue, and every read takes one block from it.  The stream
        stays up until `close`, because starting and stopping one per block would take
        far longer than the block it produced.

        What this gives up is the same thing `RtlSdrDevice.read_block` gives up.  The
        queue holds two blocks, so anything older is dropped rather than queued, and a
        caller reading slowly misses samples instead of falling behind.
        """
        if self._closed:
            return None
        if self._sink is not None:
            raise RuntimeError(
                'The receiver cannot serve a synchronous read while it is streaming.  '
                'Call stop_stream first.')
        if self._sync_sink is None:
            self._sync_sink = _SyncBlocks(_SYNC_QUEUE_BLOCKS)
            self._begin(block_samples)
        block = self._sync_sink.take(_SYNC_READ_TIMEOUT_SECONDS)
        if block is None:
            logger.warning(
                'The receiver produced no samples within %.0f seconds.  Capture has '
                'stopped and will not restart on its own.  Restart the monitor to try '
                'again.', _SYNC_READ_TIMEOUT_SECONDS)
        return block

    def close(self) -> bool:
        """Release the receiver and the API session, and say whether it let go.

        This is safe to call twice, and it will be called twice, for the reason
        `RtlSdrDevice.close` gives: shutdown calls it and the atexit hook fires after.

        Each library call runs on a thread with a timeout, the same as the RTL-SDR
        close, because a call that has not returned is inside the library and the
        operating system releases the device when the process ends.
        """
        if self._closed:
            return self._released
        self._closed = True
        atexit.unregister(self.close)
        self._sink = None
        self._sync_sink = None
        released = self._end()
        released = _bounded(lambda: self._library.release(self._device),
                            'releasing the receiver') and released
        released = _bounded(self._library.close, 'closing the API session') and released
        self._released = released
        return released

    # ------------------------------------------------------------------ static

    @staticmethod
    def nearest_supported_gain(gain_db: float, supported: list[float]) -> float:
        """The value from `supported` closest to `gain_db`."""
        return min(supported, key=lambda candidate: abs(candidate - gain_db))

    @staticmethod
    def _gain_reduction_ladder() -> list[int]:
        """Every total gain reduction the two knobs reach, in dB, lowest first.

        Each total is an LNA reduction from the table plus a baseband reduction from 20
        to 59, and the ranges overlap, so the distinct totals come out as one unbroken
        run from 20 to 120.
        """
        return sorted({lna + baseband
                       for lna in HF_LNA_GAIN_REDUCTION_DB
                       for baseband in range(MIN_GAIN_REDUCTION_DB,
                                             MAX_GAIN_REDUCTION_DB + 1)})

    @staticmethod
    def _knobs_for(gain_db: float) -> tuple[int, int]:
        """Split one gain into the LNA state and the baseband reduction to write.

        Most totals are reachable more than one way, and this takes the least LNA
        reduction that leaves a baseband figure in range.  That is the quietest of the
        choices, because reduction taken at the front end costs noise figure where
        reduction taken at baseband does not.  The receiver reports an overload event
        when that choice is wrong for the signal present.
        """
        total = int(round(-gain_db))
        for lna_state, lna_reduction in enumerate(HF_LNA_GAIN_REDUCTION_DB):
            baseband = total - lna_reduction
            if MIN_GAIN_REDUCTION_DB <= baseband <= MAX_GAIN_REDUCTION_DB:
                return lna_state, baseband
        raise ValueError(
            f'A gain of {gain_db:.0f} dB needs a total reduction of {total} dB, and '
            f'this receiver reaches {SdrplayDevice._gain_reduction_ladder()[0]} '
            f'through {SdrplayDevice._gain_reduction_ladder()[-1]} dB.  Pick a gain '
            f'from supported_gains_db.')

    @staticmethod
    def _rate_plan(iq_sample_rate: int) -> tuple[float, int]:
        """The converter rate and decimation factor that give `iq_sample_rate`.

        An RSP cannot sample as slowly as this program reads, so it samples fast and
        the library decimates.  This takes the smallest factor that lifts the converter
        to its own minimum, because a higher converter rate spends USB bandwidth and
        buys nothing once the band is already narrower than the result.
        """
        for factor in DECIMATION_FACTORS:
            if iq_sample_rate * factor >= MIN_ADC_RATE_HZ:
                return float(iq_sample_rate * factor), factor
        raise ValueError(
            f'A sample rate of {iq_sample_rate} Hz is too low for this receiver.  The '
            f'converter runs no slower than {MIN_ADC_RATE_HZ} Hz and decimates by at '
            f'most {DECIMATION_FACTORS[-1]}.  Use at least '
            f'{MIN_ADC_RATE_HZ // DECIMATION_FACTORS[-1]} Hz.')

    @staticmethod
    def _bandwidth_for(iq_sample_rate: int) -> int:
        """The narrowest filter that passes the whole band this rate carries.

        A filter narrower than the sample rate would roll off inside the spectrum the
        monitor displays, and the widest one lets in signals that alias back.
        """
        widths = sorted(width for width in api.sdrplay_api_Bw_MHzT
                        if width > api.sdrplay_api_Bw_MHzT.sdrplay_api_BW_Undefined)
        wanted_khz = iq_sample_rate / 1000
        return next((int(width) for width in widths if width >= wanted_khz),
                    int(widths[-1]))

    # ----------------------------------------------------------------- private

    @classmethod
    def _select(cls, library: SdrplayApi, index: int) -> api.sdrplay_api_DeviceT:
        """Find the receiver at `index`, select it, and hand it back.

        This holds the lock across listing and selecting, because the API service is
        shared with every other program on the machine.  Releasing it between the two
        is what lets another program take the receiver this one just listed.
        """
        library.lock()
        try:
            found = library.devices()
            supported = [device for device in found if device.hwVer in HF_LNA_MAX_HZ]
            if len(supported) <= index:
                raise RuntimeError(cls._why_there_was_no_receiver(index, found))
            device = supported[index]
            library.select(device)
            return device
        finally:
            _quietly(library.unlock)

    @staticmethod
    def _why_there_was_no_receiver(index: int,
                                   found: list[api.sdrplay_api_DeviceT]) -> str:
        """Say what was seen, which is the difference between none and the wrong one."""
        if not found:
            return (
                f'No SDRplay receiver was found for index {index}.  Either none is '
                f'plugged in, or the SDRplayAPIService is not running.  Check the '
                f'cable, and start the service if it is stopped.')
        kinds = ', '.join(sorted({f'hwVer {device.hwVer}' for device in found}))
        return (
            f'No RSP1A or RSP1B was found for index {index}.  {len(found)} SDRplay '
            f'receiver(s) are present ({kinds}), and this program supports the RSP1A '
            f'and the RSP1B only.  Set [audio] source back to rtlsdr or soundcard.')

    def _check_band(self) -> None:
        """Refuse a tuning the vendored LNA table does not cover."""
        edge = HF_LNA_MAX_HZ[self._device.hwVer]
        if self._tuned_hz >= edge:
            raise ValueError(
                f'This receiver is tuned to {self._tuned_hz / 1e6:.3f} MHz, and this '
                f'program supports it below {edge / 1e6:.0f} MHz.  The gain table '
                f'changes above that edge and nobody has tested the higher one.  '
                f'Powerline QRM is an HF measurement, so tune below the edge.')

    def _configure(self, iq_sample_rate: int) -> int:
        """Set the rate, the tuning, the filter, and turn the AGC off.

        The RSP has an AGC of its own and the API turns it on by default.  An AGC
        riding on the impulses would compress exactly what this program measures while
        leaving the noise floor looking healthy, so it is disabled explicitly, the same
        as the RTL-SDR's digital AGC is.

        This reads nothing back, because the library applies these settings as it
        initializes and reports a failure from `sdrplay_api_Init` instead.
        """
        fs_hz, decimation = self._rate_plan(iq_sample_rate)
        channel = self._params.rxChannelA
        self._params.devParams.contents.fsFreq.fsHz = fs_hz
        channel.contents.ctrlParams.decimation.enable = 1 if decimation > 1 else 0
        channel.contents.ctrlParams.decimation.decimationFactor = decimation
        channel.contents.ctrlParams.agc.enable = (
            api.sdrplay_api_AgcControlT.sdrplay_api_AGC_DISABLE)
        channel.contents.tunerParams.rfFreq.rfHz = float(self._tuned_hz)
        channel.contents.tunerParams.bwType = self._bandwidth_for(iq_sample_rate)
        channel.contents.tunerParams.ifType = (
            api.sdrplay_api_If_kHzT.sdrplay_api_IF_Zero)
        channel.contents.tunerParams.gain.minGr = (
            api.sdrplay_api_MinGainReductionT.sdrplay_api_NORMAL_MIN_GR)
        logger.debug('Converter at %.0f Hz, decimating by %d for %d Hz.',
                     fs_hz, decimation, iq_sample_rate)
        return iq_sample_rate

    def _write_gain(self, gain_db: float) -> float:
        """Write one gain, and return what the receiver says it ended up at.

        Before `sdrplay_api_Init` this simply sets the values, because nothing is
        running to update.  Afterwards they go through `sdrplay_api_Update`, which is
        what makes a gain change during a stream the supported order.
        """
        lna_state, baseband = self._knobs_for(gain_db)
        gain = self._params.rxChannelA.contents.tunerParams.gain
        gain.gRdB = baseband
        gain.LNAstate = lna_state
        if self._initialized:
            self._library.update(
                int(self._device.dev), int(self._device.tuner),
                api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Tuner_Gr)
        return self._reported_gain(gain_db, lna_state, baseband)

    def _reported_gain(self, wanted_db: float, lna_state: int,
                       baseband: int) -> float:
        """The receiver's own figure, and a warning when the table disagrees with it.

        `gainVals.curr` is an output parameter, so this is what the hardware says its
        gain is rather than what the vendored table predicts.  The two are compared
        once per session: a table that has gone stale then reports itself instead of
        shifting every measurement by a fixed amount with nothing to notice.

        The receiver fills the figure in as it applies the change, so a zero means the
        change has not taken effect yet and the predicted figure is the better answer.
        """
        gain = self._params.rxChannelA.contents.tunerParams.gain
        reported = float(gain.gainVals.curr)
        if not reported:
            return wanted_db
        if not self._gain_table_checked:
            self._gain_table_checked = True
            if round(reported) != round(wanted_db):
                logger.warning(
                    'The receiver reports %.1f dB of gain at LNA state %d and %d dB '
                    'of baseband reduction.  The gain table in buzz.sdrplay_device '
                    'predicts %.1f dB.  The hardware figure is the one in use.  The '
                    'table may be out of date for this receiver or this band.',
                    reported, lna_state, baseband, wanted_db)
        return reported

    def _begin(self, block_samples: int) -> None:
        """Hand the library its callbacks and start it delivering.

        This device keeps the callback objects rather than passing them and forgetting
        them, because ctypes keeps no reference of its own.  A collected callback
        leaves the library calling into freed memory, which is a crash in C rather
        than an exception here.
        """
        self._block_values = block_samples * VALUES_PER_FRAME
        self._pending = []
        self._pending_values = 0
        if self._initialized:
            # No caller reaches this today, and what keeps it that way sits in three
            # other methods: start_stream refuses when either sink is set, stop_stream
            # clears the flag through _end, and read_block calls here only while it has
            # no sink of its own.  A fourth caller that missed one of those would reach
            # `sdrplay_api_Init` on a running library, which the API refuses as an
            # already-open device, and leaves the stream delivering to callbacks this
            # method has already replaced.  The guard costs less than that failure.
            return  # pragma: no cover -- unreachable while the three guards above hold
        self._stream_callback = api.sdrplay_api_StreamCallback_t(self._on_stream)
        self._event_callback = api.sdrplay_api_EventCallback_t(self._on_event)
        self._callbacks = api.sdrplay_api_CallbackFnsT(
            StreamACbFn=self._stream_callback,
            StreamBCbFn=api.sdrplay_api_StreamCallback_t(),
            EventCbFn=self._event_callback)
        self._library.init(int(self._device.dev), self._callbacks)
        self._initialized = True

    def _end(self) -> bool:
        """Stop the library delivering, and say whether its uninit returned."""
        if not self._initialized:
            return True
        self._initialized = False
        return _bounded(lambda: self._library.uninit(int(self._device.dev)),
                        'stopping the receiver')

    def _on_stream(self, xi: object, xq: object, params: object, num_samples: int,
                   _reset: int, _context: object) -> None:
        """Take one delivery from the library, on the library's own thread.

        Nothing here is allowed to raise, because this is called from C where an
        exception has nowhere sensible to go.  The library chooses how many samples to
        deliver and it is not the block size anybody asked for, so this accumulates and
        emits whole blocks.

        A delivery marked `grChanged` is the one where a gain change took effect, so
        everything up to and including it predates the new gain and goes in the bin.
        """
        try:
            count = int(num_samples)
            if count <= 0:
                return
            interleaved = np.empty(count * VALUES_PER_FRAME, dtype=np.int16)
            interleaved[0::VALUES_PER_FRAME] = np.ctypeslib.as_array(
                xi, shape=(count,))
            interleaved[1::VALUES_PER_FRAME] = np.ctypeslib.as_array(
                xq, shape=(count,))
            if params and params.contents.grChanged:
                self._awaiting_gain_change = False
                self._pending = []
                self._pending_values = 0
                return
            self._pending.append(interleaved)
            self._pending_values += interleaved.size
            while self._pending_values >= self._block_values > 0:
                self._emit(self._take(self._block_values))
        except Exception:  # pragma: no cover -- the last resort in a C callback
            self._blocks_refused += 1

    def _take(self, values: int) -> np.ndarray:
        """Pull exactly `values` from the front of the pending arrays."""
        gathered = np.concatenate(self._pending)
        self._pending = [gathered[values:]] if gathered.size > values else []
        self._pending_values = gathered.size - values
        return gathered[:values]

    def _emit(self, raw: np.ndarray) -> None:
        """Offer one assembled block, or count it as refused.

        The index counts blocks the receiver produced rather than blocks that survived,
        so a gap tells a consumer that something was dropped, whether a sink had no
        room or a gain change made the samples stale.
        """
        self._produced += 1
        if self._awaiting_gain_change:
            return
        block = IqBlock(raw=raw, fmt=SDRPLAY_FORMAT, arrived_at=monotonic(),
                        index=self._produced)
        sink = self._sink or self._sync_sink
        if sink is None or not sink.offer(block):
            self._blocks_refused += 1

    def _on_event(self, event_id: int, _tuner: int, params: object,
                  _context: object) -> None:
        """Take one event from the library, on the library's own thread.

        Nothing here is allowed to raise, for the same reason `_on_stream` may not.
        Counting is all this does, and whoever reads the counter on an ordinary thread
        reports it, because logging can block on I/O.
        """
        try:
            if event_id == api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange:
                overload = params.contents.powerOverloadParams.powerOverloadChangeType
                if overload == (api.sdrplay_api_PowerOverloadCbEventIdT
                                .sdrplay_api_Overload_Detected):
                    self._overloads += 1
        except Exception:  # pragma: no cover -- the last resort in a C callback
            pass


class _SyncBlocks:
    """A `BlockSink` that holds a few blocks for `read_block` to collect.

    This exists because the SDRplay API has no synchronous read, so the only way to
    serve one is to run a stream and take from it.  The queue is short on purpose: a
    synchronous reader wants the newest samples, and a deep queue would hand it a
    backlog to work through instead.
    """

    def __init__(self, depth: int) -> None:
        self._queue: queue.Queue[IqBlock] = queue.Queue(maxsize=depth)

    def offer(self, block: IqBlock) -> bool:
        """Take a block, or return False when the reader is behind."""
        try:
            self._queue.put_nowait(block)
            return True
        except queue.Full:
            return False

    def take(self, timeout: float) -> IqBlock | None:
        """The next block, or None once nothing is arriving."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _what_an_api_failure_usually_means(code: int) -> str:
    """A next step for the failures an operator can do something about."""
    errors = api.sdrplay_api_ErrT
    if code == errors.sdrplay_api_ServiceNotResponding:
        return ('The SDRplayAPIService is not running.  Start it, or reinstall the '
                'SDRplay Hardware API.')
    if code == errors.sdrplay_api_HwVerError:
        return ('The service does not recognise this receiver, which usually means '
                'the installed API is older than the hardware.')
    if code in (errors.sdrplay_api_AlreadyInitialised,
                errors.sdrplay_api_NotInitialised):
        return 'This is a fault in this program rather than in the receiver.'
    return 'Check that no other program is using the receiver, then try again.'


def _quietly(call: object) -> None:
    """Run a cleanup call and swallow whatever it raises.

    Callers use this only on paths already unwinding from a failure, so that the
    exception an operator needs to see is the one that propagates.
    """
    try:
        call()
    except Exception:
        logger.debug('A cleanup call failed during shutdown.', exc_info=True)


def _bounded(call: object, what: str) -> bool:
    """Run a library call on a thread, and give up on it after a timeout.

    The SDRplay API talks to a background service over an interprocess channel, so a
    call can wait on a service that has stopped answering.  Waiting forever at shutdown
    would hang the monitor, where giving up leaves the receiver held until the process
    ends, which is where a blocked call left it anyway.
    """
    finished = threading.Event()

    def run() -> None:
        try:
            call()
        except Exception:
            logger.debug('%s failed.', what, exc_info=True)
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True, name='sdrplay-close').start()
    if finished.wait(timeout=_DEVICE_CLOSE_TIMEOUT_SECONDS):
        return True
    logger.warning(
        'The receiver did not finish %s within %.0f seconds and was left to the '
        'operating system.  The API waits on a background service, so waiting longer '
        'would only hang this program.', what, _DEVICE_CLOSE_TIMEOUT_SECONDS)
    return False

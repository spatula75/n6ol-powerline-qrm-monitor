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
import math
import queue
import threading
from collections.abc import Callable
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Protocol, Self

import numpy as np

from buzz import sdrplay_api as api
from buzz.config import SdrConfig
from buzz.sdr_device import VALUES_PER_FRAME, BlockSink, DeviceProfile, IqBlock, OverloadStatus, SampleFormat, SdrDevice

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

# The API delivers signed 16-bit I/Q values.  These endpoints describe that output
# format, not every stage that can overload inside the receiver.  Filtering and
# decimation can move clipped values away from an endpoint, so the hardware overload
# reports remain a separate indication.
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

# The gain an RSP1A or RSP1B has on HF before any reduction is applied, in dB.
#
# The value came from measuring, not from theory.  On an RSP1B listening at 3530 kHz,
# the gain the library reports in `gainVals.curr` was read at eleven settings
# spanning the whole ladder.  This is what each implies once the reduction is added
# back:
#
#     LNA state    0     1     2     4     5     6
#     implied   91.4  91.4  91.0  91.3  91.5  91.6
#
# Two things follow.  The vendored LNA table is right, because a wrong entry would put
# its own rows somewhere else entirely rather than within half a decibel of the others.
# And the receiver's gain is this figure minus the total reduction, which is what lets
# the ladder below be stated in real dB rather than as the negative of a reduction.
#
# This is a fallback rather than the figure normally used.  It came from one receiver
# at one frequency, and the conversion gain is not flat across HF, so a device asks its
# own hardware for the figure at open and keeps that for the session.  See
# `_learn_conversion_gain`.  This is what a receiver gets when the library declines to
# report a gain at all, which is also the only case where nobody can do better.
HF_CONVERSION_GAIN_DB = 91.4

# Fixed part of the uncalibrated API-output dBFS to receiver-input estimate.
#
# On 2026-09-18 at 3540 kHz, comparisons with a calibrated receiver put one RSP1B's
# intercept near 11 dB at gains of 13 and 19 dB.  Measurements close to the noise-floor
# knee varied more, which is why the gain probe adds the padding below.  This gives a
# new station a useful starting point; measured level calibration still replaces it.
# See docs-notebook/sdrplay-gain.md.
ESTIMATED_CALIBRATION_INTERCEPT_DB = 11.0

# How many bits of the int16 audio an RSP really resolves.
#
# See `SdrplayDevice.effective_bits` for where the figure comes from.  It is a module
# constant for the same reason FLOOR_MARGIN_DB below is: the scope's floor is derived
# from it, and a test that checks the derivation should read the figure rather than
# restate it.
EFFECTIVE_BITS = 15

# How many of its own steps this receiver's scope floor is worth.
#
# The value came from measuring rather than from theory.  `SdrplayDevice.scope_floor_steps`
# gives the two readings and the window they define, and
# docs-notebook/scope-auto-range-floor.md says how they were taken.
SCOPE_FLOOR_STEPS = 3.2

# How far above the noise-floor knee to put the gain the sweep chooses.
#
# See `SdrplayDevice.floor_margin_db` for the measurement and the reasoning.  It is a
# module constant so that a test can state the figure without an open receiver.
FLOOR_MARGIN_DB = 10.0

# Samples per block while nothing has asked for a size yet.
#
# A device initializes the library at open so that `gainVals.curr` is filled in before
# anybody asks for the gain ladder, and stops it again straight away.  Nothing is
# listening for the moment it runs, so the size only has to be a size.
_INITIAL_BLOCK_SAMPLES = 2048

# How long to let the receiver run before reading the gain it settled on.
#
# The wait is for a delivery rather than for a duration, and this only bounds it.  What
# it buys is that the library has demonstrably applied the settings, where a read taken
# the instant `sdrplay_api_Init` returns rests on the assumption that it fills
# `gainVals.curr` before returning, which nobody here has checked.
#
# Half a second is long against the milliseconds a receiver takes to start and short
# against anything an operator would notice at startup.  Reaching it means the receiver
# is not delivering at all, and the ladder then falls back to the measured constant,
# which is the same answer this program would have given before it learned to ask.
_GAIN_REPORT_WAIT_SECONDS = 0.5

# How far the hardware's own figure may sit from what the table predicts before this
# says so.  Three decibels: the spread above is 0.6 across LNA states at one frequency,
# so this allows several times that for the rest of the band before calling it wrong.
_GAIN_DISAGREEMENT_DB = 3.0

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

# How deep the backlog has to be before it is unusual rather than routine, in seconds.
#
# The figure is chosen rather than measured.  The library delivers in bursts, so the
# backlog reaches one burst period as a matter of course: measured on an RSP1B at 3530
# kHz it peaked between 71.6 and 87.6 ms every minute for ten minutes.  100 ms is
# clear of that and still catches the tens of milliseconds a drift excursion runs to.
#
# A first version of this counted every gap past 10 ms instead, which came to 953 a
# minute and described the library's delivery cadence rather than anything wrong.
_UNUSUAL_BACKLOG_SECONDS = 0.100

# How often to summarize the waits, in seconds.
#
# A minute matches _HEALTH_INTERVAL_SECONDS in buzz.sdr, so a summary sits in the log
# beside the drift figure covering the same minute.  Comparing the two is the whole
# purpose, and two different periods would make that arithmetic rather than reading.
_LATE_REPORT_INTERVAL_SECONDS = 60.0

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

# How long to keep dropping blocks while waiting for the library to mark one with
# `grChanged`.
#
# A ceiling rather than a measurement.  `grChanged` is the exact answer and this is
# what happens when it never comes: the specification says the library sets it on the
# block where a gain change took effect, and nobody here has confirmed that against
# hardware.  Without a bound, a library that never sets it drops every block from the
# first gain change onwards, and the receiver goes silent for the rest of the session.
#
# That is not hypothetical.  It is what made a gain sweep report the lowest gain on
# the ladder: every gain after the first measured nothing, so the only gain with
# readings was the one the sweep started at.
#
# Half a second for scale: an RTL-SDR's transfer pool holds about 960 ms at the
# default block and 256 kHz, so this is well inside what a comparable receiver buffers.
#
# The two ways to be wrong are not equal, which is what sets it this high.  Too short
# and a sweep measures the previous gain and picks the wrong one, silently, which is
# the fault this bound exists to prevent.  Too long and the sweep takes half a second
# more per gain step.  Nothing pays it at all while the flag arrives.
_GAIN_CHANGE_SETTLE_SECONDS = 0.5

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
    def effective_bits(cls) -> int:
        """Fifteen, which is fourteen on the converter and one recovered by decimating.

        The library delivers int16 and uses the whole of it, and the converter behind
        that resolves fourteen.  Oversampling makes up part of the difference: at the
        256 kHz this program asks for, the converter runs at 2048 kHz and decimates by
        eight, which is 9 dB and worth 1.5 bits by the ideal rule.  Taking one of them
        rather than both is the conservative reading, because the ideal rule assumes
        white quantization noise and a perfect filter.

        The recovered bit follows the rate, since the decimation does: 1024 kHz
        decimates by two and earns half a bit, where 64 kHz decimates by 32 and earns
        two and a half, which the int16 container caps at sixteen.  Fifteen is fixed
        rather than derived because it is conservative at the rate this program
        defaults to and below, and the whole span is only a bit and a half.  A station
        running well above 256 kHz is claiming a bit it does not have, and the display
        would magnify a little further than it should.

        See docs-notebook/scope-auto-range-floor.md.
        """
        return EFFECTIVE_BITS

    @classmethod
    def scope_floor_steps(cls) -> float:
        """The midpoint of a window nearly twice the RTL-SDR's, in decibels.

        The figures came from measuring on 2026-09-19 at 14 dB of gain.  With the
        antenna off this receiver asked the scope for 1.30 of its own steps, and on a
        live band it asked for 7.86.  That is 15.6 dB of window, because an RSP runs
        ten decibels above the knee where an RTL-SDR runs at it.  See
        floor_margin_db.

        The midpoint in decibels is the square root of 1.30 times 7.86, which leaves
        7.8 dB to either fault and draws a dead channel at about two fifths of the
        height.  A figure shared with the RTL-SDR would have to suit that receiver's
        narrower window and would spend most of this one, which is why the multiple
        belongs to the receiver rather than to buzz.scope.
        """
        return SCOPE_FLOOR_STEPS

    @classmethod
    def floor_margin_db(cls) -> float:
        """Ten decibels above the knee, which fourteen bits can afford.

        An RSP has about 84 dB of converter range against an RTL-SDR's 48, so the same
        32 dB arc reserve leaves about 52 dB rather than 16.  Ten of those buy a
        reported floor that reads 0.4 dB high instead of 3.0.

        Measured on an RSP1B on a rooftop antenna at 3530 kHz.  The knee sat at 1 dB of
        gain, where the reported floor read 3.8 dB high, SNR read the same amount low
        against a calibrated receiver, and the scope trace sat under the auto-range
        floor.  Ten dB of margin moves the choice to 11 dB and leaves 60 dB of arc room
        against the 32 dB reserve, so nothing is given up to buy it.

        Ten rather than some other figure because that is what this program tells an
        SDRplay operator to aim for, in schema.json and config.example.toml.  A handful
        of manual trials on the same station had arrived at about six from the other
        direction.
        """
        return FLOOR_MARGIN_DB

    @classmethod
    def estimated_calibration_offset_db(cls, gain_db: float) -> float:
        """Undo receiver gain after adding the measured API-output intercept."""
        return -gain_db + ESTIMATED_CALIBRATION_INTERCEPT_DB

    @classmethod
    def open_from(cls, settings: SdrConfig) -> Self:
        """Open the receiver these settings describe.  See `SdrDevice.open_from`.

        `api_path` is read here and nowhere else, because it is the one setting an
        RSP needs that an RTL-SDR has no use for.
        """
        return cls.open(
            settings.device_index,
            tuned_hz=settings.frequency_hz + settings.tuning_offset_hz,
            gain_db=settings.gain_db,
            iq_sample_rate=settings.iq_sample_rate,
            api_path=getattr(settings, 'api_path', None))

    @classmethod
    def supported_gains(cls, settings: SdrConfig) -> list[float]:
        """Every gain an RSP1A or RSP1B offers on HF.

        No hardware is opened, unlike an RTL-SDR's, because the reduction ladder comes
        from the gain reduction tables rather than from the unit: every RSP1A and RSP1B
        reaches the same 101 rungs below the band edge.  An RTL-SDR has to ask, since
        its steps belong to the tuner chip that happens to be fitted.

        What that costs is where the hundred decibels sit.  An open device asks the
        receiver for its conversion gain, and this cannot, so the rungs here are the
        fallback's and may be a decibel or two from the ones the device settles on.
        That is what a picker needs, because the device snaps whatever is chosen to its
        own nearest rung anyway.

        `settings` is accepted and unread, so that a caller holding a config section
        can ask either receiver the same question.
        """
        return cls._gain_ladder()

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
            installed_version = library.api_version()
            if not math.isclose(installed_version, api.API_VERSION, abs_tol=0.005):
                raise RuntimeError(
                    f'This program was built for SDRplay API {api.API_VERSION:.2f}, but '
                    f'the installed library reports {installed_version:.2f}.  Install '
                    'the matching SDRplay Hardware API before opening the receiver.')
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
        except _ReceiverStillRunning:
            raise
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
        self._overload_lock = threading.Lock()
        self._accept_overload_events = False
        self._overload_status = OverloadStatus(active=False, detections=0)
        self._overload_error: str | None = None
        self._closed = False
        self._released = False
        self._initialized = False
        self._shutdown_stuck = False
        self._sink: BlockSink | None = None
        self._sync_sink: _SyncBlocks | None = None
        self._gain_table_checked = False
        self._gain_flag_reported = False
        # Blocks to drop before anything is offered, because a gain change is in
        # flight.  The block the library marks with `grChanged` clears it, and
        # `_settle_blocks` is the ceiling that clears it when no such block arrives.
        self._awaiting_gain_change = False
        self._dropped_waiting = 0
        self._settle_blocks = 0
        # This holds the gain just written, until the library marks the block where
        # it took effect.  That block is the first moment `gainVals.curr` answers
        # about this gain rather than about the previous one, so it is where the
        # table gets checked.  This thread writes it and the library's thread clears
        # it.
        self._pending_gain_check: tuple[float, int, int] | None = None
        # Set by the stream callback on any delivery, including one nothing is
        # listening for.  What it says is that the library is running and has applied
        # what it was given, which is when `gainVals.curr` means something.
        self._delivered = threading.Event()
        # The block being filled, and how many values of it are written.  Only the
        # stream callback touches either, and the buffer is replaced rather than
        # rewritten once a consumer holds it.
        self._filling: np.ndarray = np.empty(0, dtype=np.int16)
        self._filled = 0
        # The deepest backlog in the window now open, how much audio has arrived in
        # it, how many deliveries found the backlog past the threshold, and when the
        # window started.  Only the stream callback touches any of them.
        self._worst_backlog = 0.0
        self._audio_this_window = 0.0
        self._unusual_count = 0
        self._past_threshold = False
        self._window_opened_at: float | None = None
        # Whether the window now open is the first of a stream.  See
        # _report_the_worst_backlog for why that one is reported differently.
        self._first_window = True
        self._block_values = 0

        self._check_band()
        self._params = library.device_params(int(device.dev))
        self._iq_sample_rate = self._configure(iq_sample_rate)
        # Written against the fallback, because nothing has asked the hardware yet and
        # a gain has to be set before the library will report one.
        self._conversion_gain_db = HF_CONVERSION_GAIN_DB
        self._write_gain(self.nearest_supported_gain(gain_db, self.supported_gains_db))
        # Started only to make the library answer, and stopped again before anything
        # else happens.  A library left running here spends the gap until the first
        # read filling its own buffers, and hands the backlog over in a burst as soon
        # as a consumer attaches.  That burst carries more audio than the wall clock
        # between its blocks accounts for, which `SdrSource.clock_drift_seconds`
        # reads as the receiver clock running away from the system one.
        self._begin(_INITIAL_BLOCK_SAMPLES)
        gain_report_arrived = self._delivered.wait(timeout=_GAIN_REPORT_WAIT_SECONDS)
        self._conversion_gain_db = self._learn_conversion_gain(gain_report_arrived)
        if not self._end():
            raise _ReceiverStillRunning(
                'The receiver started but did not stop while it was learning its gain.  '
                'It remains held until this process exits.  Restart the program before '
                'trying to open it again.')
        # Written again, now that the ladder means what it says.  The first write put
        # the receiver wherever the fallback pointed, which is the right rung only when
        # the fallback happened to be right for this band.
        self._gain_db = self._write_gain(self.nearest_supported_gain(
            gain_db, self.supported_gains_db))
        self._profile = DeviceProfile(
            name=f'SDRplay {"RSP1B" if device.hwVer == api.SDRPLAY_RSP1B_ID else "RSP1A"}',
            settings_section='sdrplay',
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

        Real gain rather than the negative of a reduction, so the figures mean the
        same thing here as they do on an RTL-SDR and as they do in the level offset a
        station calibrates.  The list runs from about +71 dB down to about -29 dB in
        one-dB steps.  Two knobs reach most of those more than one way, and
        `_knobs_for` picks which pair to use.

        Where the hundred decibels sit depends on this receiver at this frequency, and
        `_learn_conversion_gain` asked it rather than assuming.  So the same rung is a
        different number on another band, which is the point: the number is the gain.
        """
        return self._gain_ladder(self._conversion_gain_db)

    @property
    def blocks_refused(self) -> int:
        return self._blocks_refused

    @property
    def reported_gain_db(self) -> float:
        """What the receiver says its gain is now, straight out of the struct.

        Zero is a valid gain.  The initial delivery wait decides whether the device
        supplied this field; the value itself cannot carry that second meaning.
        """
        return float(self._params.rxChannelA.contents.tunerParams.gain.gainVals.curr)

    @property
    def overloads(self) -> int:
        """How many overload detection events the receiver has reported."""
        with self._overload_lock:
            return self._overload_status.detections

    @property
    def overload_status(self) -> OverloadStatus:
        """The last hardware state and detection count, read together.

        The callback updates both under one lock, so a reader cannot pair an old count
        with a new state.  A failed acknowledgement makes the indication unreliable.
        Report that failure here, because an exception cannot leave the C callback.
        """
        with self._overload_lock:
            status = self._overload_status
            error = self._overload_error
        if error is not None:
            raise RuntimeError(
                f'The SDRplay overload callback failed: {error.rstrip(".")}.  '
                'Hardware overload reporting is unreliable.  Close the receiver '
                'and restart the probe before using its overload readings.')
        return status

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
        self._dropped_waiting = 0
        # Whatever a synchronous reader already has in hand predates this change, so
        # it goes now.  The drop above covers blocks the library has yet to deliver
        # and cannot reach one already queued, which a reader would otherwise take as
        # a measurement of the new gain.
        if self._sync_sink is not None:
            self._sync_sink.clear()
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
        if not self._end():
            return False
        if not _bounded(lambda: self._library.release(self._device),
                        'releasing the receiver'):
            return False
        released = _bounded(self._library.close, 'closing the API session')
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

    @classmethod
    def _gain_ladder(cls,
                     conversion_gain_db: float = HF_CONVERSION_GAIN_DB) -> list[float]:
        """Every gain this receiver reaches on HF, in real dB, highest first.

        The reduction ladder subtracted from the conversion gain, and rounded to whole
        decibels, so that an operator types a round number and the round trip through
        `_knobs_for` returns the same rung.

        This has to be the scale `set_gain_db` answers on, because a gain sweep files
        each reading under whatever that returns and then looks those keys up in this
        list.  Returning the hardware's figure while this held the negative of a
        reduction put every reading under a key no lookup would ever ask for, so a
        sweep kept only its first gain and reported that as the answer.
        """
        return [float(round(conversion_gain_db - total))
                for total in cls._gain_reduction_ladder()]

    @classmethod
    def _knobs_for(cls, gain_db: float,
                   conversion_gain_db: float = HF_CONVERSION_GAIN_DB) -> tuple[int, int]:
        """Split one gain into the LNA state and the baseband reduction to write.

        Most totals are reachable more than one way, and this takes the least LNA
        reduction that leaves a baseband figure in range.  That is the quietest of the
        choices, because reduction taken at the front end costs noise figure where
        reduction taken at baseband does not.  The receiver reports an overload event
        when that choice is wrong for the signal present.
        """
        total = int(round(conversion_gain_db - gain_db))
        for lna_state, lna_reduction in enumerate(HF_LNA_GAIN_REDUCTION_DB):
            baseband = total - lna_reduction
            if MIN_GAIN_REDUCTION_DB <= baseband <= MAX_GAIN_REDUCTION_DB:
                return lna_state, baseband
        ladder = cls._gain_ladder(conversion_gain_db)
        raise ValueError(
            f'A gain of {gain_db:.0f} dB needs a total reduction of {total} dB, which '
            f'is outside what this receiver reaches.  It runs from {ladder[-1]:.0f} to '
            f'{ladder[0]:.0f} dB.  Pick a gain from supported_gains_db.')

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
        lna_state, baseband = self._knobs_for(gain_db, self._conversion_gain_db)
        gain = self._params.rxChannelA.contents.tunerParams.gain
        gain.gRdB = baseband
        gain.LNAstate = lna_state
        if self._initialized:
            # This is armed before the update rather than after it, so the library
            # cannot deliver the marked block into a check that is not yet waiting
            # for it.  See _check_reported_gain for why that block is the one that
            # can answer.
            if not self._gain_table_checked:
                self._pending_gain_check = (gain_db, lna_state, baseband)
            self._library.update(
                int(self._device.dev), int(self._device.tuner),
                api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Tuner_Gr)
        return gain_db

    def _learn_conversion_gain(self, report_arrived: bool) -> float:
        """The gain this receiver has before any reduction, asked of the receiver.

        `gainVals.curr` is what the library says the gain is now, and the reduction
        just written is known, so the sum of the two is the fixed part.  That fixed
        part is what the whole ladder is measured from, and it belongs to this unit at
        this frequency rather than to SDRplay receivers in general.

        Asked once and kept for the session.  A ladder that moved under a gain sweep
        would change what each of its readings was filed under half way through.

        A library that reports nothing leaves the fallback in place, which is the one
        case where no better answer exists.  It is logged rather than raised, because
        a figure that is a decibel or two out costs an operator a level offset they
        recalibrate anyway.
        """
        reported = self.reported_gain_db
        if not report_arrived:
            logger.info(
                'The receiver did not report its own gain within %.1f seconds, so '
                'levels use the conversion gain measured on one RSP1B at 3530 kHz.  '
                'Expect the gain figures to be a decibel or two out on other bands.',
                _GAIN_REPORT_WAIT_SECONDS)
            return HF_CONVERSION_GAIN_DB
        gain = self._params.rxChannelA.contents.tunerParams.gain
        learned = reported + HF_LNA_GAIN_REDUCTION_DB[gain.LNAstate] + gain.gRdB
        logger.debug('This receiver has %.1f dB of gain before reduction at %.4f MHz.',
                     learned, self._tuned_hz / 1e6)
        return learned

    def _check_reported_gain(self) -> None:
        """Compare the hardware's own figure against the table, and warn once.

        `gainVals.curr` is an output parameter, so this is what the receiver says its
        gain is rather than what the vendored table and the conversion gain predict.
        A disagreement means one of those is wrong for this band or this unit, and
        every level the station reports is then out by the difference.

        This only reports, and the caller returns the predicted figure regardless.  The
        two are on the same scale now, so the hardware's would be the better answer
        were it not that a gain sweep files readings under it and looks them up in
        `supported_gains_db`.  A figure half a decibel off a rung is a key that list
        does not contain, and the reading is then dropped rather than used.

        **This runs from the block the library marked, not from the gain write.**
        `sdrplay_api_Update` returns before the new gain is in force, which is the
        whole reason `_awaiting_gain_change` and `grChanged` exist, so `gainVals.curr`
        at that moment still answers about the previous gain.  Comparing the new
        prediction against the previous gain then measures the step rather than the
        table.  A sweep steps by `_COARSE_STEP_DB`, which is 10 dB against a 3 dB
        threshold, so this program would report a wrong gain table on the second gain
        of every sweep of a healthy receiver.

        So `_write_gain` records what it wrote and this runs from the marked block,
        which is the first delivery where the receiver's figure is about the same gain
        the prediction is about.  It runs on the library's own thread, so it does the
        least it can: read two fields, compare, and log at most once for the session.
        """
        pending = self._pending_gain_check
        if pending is None or self._gain_table_checked:
            return
        self._pending_gain_check = None
        self._gain_table_checked = True
        wanted_db, lna_state, baseband = pending
        gain = self._params.rxChannelA.contents.tunerParams.gain
        reported = float(gain.gainVals.curr)
        if abs(reported - wanted_db) > _GAIN_DISAGREEMENT_DB:
            logger.warning(
                'The receiver reports %.1f dB of gain at LNA state %d and %d dB of '
                'baseband reduction, where this program predicts %.1f dB.  Levels will '
                'read about %.1f dB out.  The LNA gain table in buzz.sdrplay_device is '
                'wrong for this receiver or this band.',
                reported, lna_state, baseband, wanted_db, reported - wanted_db)

    def _begin(self, block_samples: int) -> None:
        """Hand the library its callbacks and start it delivering.

        This device keeps the callback objects rather than passing them and forgetting
        them, because ctypes keeps no reference of its own.  A collected callback
        leaves the library calling into freed memory, which is a crash in C rather
        than an exception here.
        """
        self._block_values = block_samples * VALUES_PER_FRAME
        self._filling = np.empty(self._block_values, dtype=np.int16)
        self._filled = 0
        # Forget the deliveries the gain probe at open produced.  Measuring the first
        # delivery of a stream against one from seconds earlier would report the time
        # between them as a stall, on every stream this device ever starts.
        self._worst_backlog = 0.0
        self._audio_this_window = 0.0
        self._unusual_count = 0
        self._past_threshold = False
        self._window_opened_at = None
        self._first_window = True
        # How many blocks make up the settling ceiling, at this block size and rate.
        # This follows the block size rather than being fixed, so a different block
        # size does not silently change how long the device waits.
        self._settle_blocks = max(1, ceil(
            _GAIN_CHANGE_SETTLE_SECONDS * self._iq_sample_rate / block_samples))
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
        with self._overload_lock:
            self._overload_status = OverloadStatus(False, self._overload_status.detections)
            self._accept_overload_events = True
        try:
            self._library.init(int(self._device.dev), self._callbacks)
        except Exception:
            with self._overload_lock:
                self._accept_overload_events = False
            raise
        self._initialized = True

    def _end(self) -> bool:
        """Stop the library delivering, and say whether its uninit returned."""
        if not self._initialized:
            return True
        if self._shutdown_stuck:
            return False
        # An RSP1B trace with API 3.15 showed clearance during Uninit, when Update
        # rejected the acknowledgement.  See docs-notebook/sdrplay-gain.md.
        # Stop accepting events first so teardown cannot spoil the next capture.
        with self._overload_lock:
            self._accept_overload_events = False
        stopped = _bounded(lambda: self._library.uninit(int(self._device.dev)),
                           'stopping the receiver')
        if stopped:
            self._initialized = False
        else:
            self._shutdown_stuck = True
        return stopped

    def _on_stream(self, xi: object, xq: object, params: object, num_samples: int,
                   _reset: int, _context: object) -> None:
        """Take one delivery from the library, on the library's own thread.

        Nothing here is allowed to raise, because this is called from C where an
        exception has nowhere sensible to go.  The library chooses how many samples to
        deliver and it is not the block size anybody asked for, so this accumulates and
        emits whole blocks.

        A delivery marked `grChanged` is the one where a gain change took effect, so
        everything up to and including it predates the new gain and goes in the bin.

        **This allocates once per block, not once per delivery.**  The samples go
        straight into a buffer sized for one block, so a delivery costs two strided
        copies and nothing else.  This method used to build an array per delivery and
        concatenate the lot at every block boundary.

        The churn matters more on this thread than it would on another.  The callback
        runs on the library's own thread and needs the GIL, so anything else that holds
        the GIL delays the next delivery.  A long enough delay makes the library hand
        over a backlog in one burst, which `buzz.sdr` reports as the receiver clock
        running away from the system clock.  `buzz.plotter` already disables collection
        around its own work for the same reason.
        """
        try:
            self._delivered.set()
            self._note_the_backlog(int(num_samples))
            if self._sink is None and self._sync_sink is None:
                # Delivered before any consumer attached, which happens between the
                # open and the first read, because the library is initialized at open
                # so that it can report its own gain.  A refusal means a consumer had
                # no room, so this is not one.
                self._filled = 0
                return
            if params and params.contents.grChanged:
                self._awaiting_gain_change = False
                self._check_reported_gain()
                self._filled = 0
                return
            count = int(num_samples)
            if count <= 0 or self._block_values <= 0:
                return
            self._fill_from(np.ctypeslib.as_array(xi, shape=(count,)),
                            np.ctypeslib.as_array(xq, shape=(count,)), count)
        except Exception:  # pragma: no cover -- the last resort in a C callback
            self._blocks_refused += 1

    def _note_the_backlog(self, count: int) -> None:
        """Track how far behind real time the receiver has fallen in this window.

        The backlog is the time open since the window started, less the audio that has
        arrived in it.  It grows while the library holds audio and falls to nothing
        when the library hands it over, so its peak over a window is the deepest the
        library ever got behind.

        The arithmetic does not care how deliveries are sized, which an earlier version
        did and was wrong for it.  That one took the interval between two callbacks and
        subtracted the audio the delivery *before* it held.  A library that pauses and
        then hands over what it accumulated delivers the large block *after* the gap,
        so every inter-burst period read as lateness.  Measured on an RSP1B, the
        delivery before each long gap held 0.6 ms of audio and the gap read 73 ms.

        This is the direct measurement, and `SdrSource.clock_drift_seconds` is the
        indirect one.  The drift figure is the same quantity taken from the stream
        start rather than from the window start, so this one shows the burst depth and
        that one shows where the depth has got to over the whole run.
        """
        if count <= 0:
            # The library calls with an empty delivery at open and at every stream
            # start.  It carries no audio, so it moves neither term of the backlog.
            return
        now = monotonic()
        if self._window_opened_at is None:
            self._window_opened_at = now
        # Measured before this delivery is counted, because the backlog peaks in the
        # moment before audio arrives.  Counting the delivery first would let one large
        # delivery hide the gap that preceded it, which is the whole shape this exists
        # to catch.
        backlog = (now - self._window_opened_at) - self._audio_this_window
        self._audio_this_window += count / self._iq_sample_rate
        if backlog > self._worst_backlog:
            self._worst_backlog = backlog
        # Counted on the way past rather than while past.  A backlog closes only when
        # the library delivers more audio than the time it takes, so deliveries that
        # merely keep pace leave it open, and counting those would report one excursion
        # as dozens.
        was_past, self._past_threshold = (self._past_threshold,
                                          backlog >= _UNUSUAL_BACKLOG_SECONDS)
        if self._past_threshold and not was_past:
            self._unusual_count += 1
        elapsed = now - self._window_opened_at
        if elapsed >= _LATE_REPORT_INTERVAL_SECONDS:
            self._report_the_worst_backlog(now)

    def _report_the_worst_backlog(self, now: float) -> None:
        """Say how deep the backlog got in this window, and start the next one.

        Reported at DEBUG rather than as a warning, because a backlog loses no samples.
        The library holds them and hands them over.

        A window with nothing over the threshold still reports, because the absence is
        the useful reading when the drift figure for the same minute says the receiver
        fell behind.

        The first window of a stream says that it is the first, because starting a
        stream is itself a long backlog and the figure is not comparable with the ones
        after it.  Measured on an RSP1B, two runs gave 265.9 ms and 218.3 ms in that
        window, where every window after them sat between 71.6 and 87.6 ms.
        `SdrPipeline`'s drift check treats its own first interval as a baseline for
        the same reason.
        """
        period = (f'first {_LATE_REPORT_INTERVAL_SECONDS:g} seconds of this stream'
                  if self._first_window
                  else f'last {_LATE_REPORT_INTERVAL_SECONDS:g} seconds')
        self._first_window = False
        logger.debug(
            'In the %s the receiver fell at worst %.1f ms behind real time.  '
            'Crossings past %.0f ms: %d.',
            period, self._worst_backlog * 1e3, _UNUSUAL_BACKLOG_SECONDS * 1e3,
            self._unusual_count)
        self._worst_backlog = 0.0
        self._audio_this_window = 0.0
        self._unusual_count = 0
        self._window_opened_at = now

    def _fill_from(self, xi: np.ndarray, xq: np.ndarray, count: int) -> None:
        """Interleave one delivery into the block buffer, emitting each block it fills.

        A delivery is whatever size the library chose and has no relation to the block
        size anybody asked for, so one can finish a block, fill several, or not finish
        any.  The loop handles all three by taking as many frames as the buffer has
        room for and going round again.

        The offsets stay frame-aligned, because every delivery contributes whole frames
        and a block is a whole number of them.  The `[start::2]` slices therefore always
        put I on an even index and Q on the odd one after it.
        """
        taken = 0
        while taken < count:
            room = (self._block_values - self._filled) // VALUES_PER_FRAME
            frames = min(count - taken, room)
            start = self._filled
            end = start + frames * VALUES_PER_FRAME
            self._filling[start:end:VALUES_PER_FRAME] = xi[taken:taken + frames]
            self._filling[start + 1:end:VALUES_PER_FRAME] = xq[taken:taken + frames]
            self._filled = end
            taken += frames
            if self._filled >= self._block_values:
                self._finish_block()

    def _finish_block(self) -> None:
        """Hand the full buffer on, and start the next one.

        This allocates a fresh buffer only where the old one went somewhere.  A block
        dropped while a gain change is in flight reached nobody, so this fills the same
        buffer again.  That matters because a gain sweep drops blocks by the hundred.
        """
        self._filled = 0
        if self._emit(self._filling):
            self._filling = np.empty(self._block_values, dtype=np.int16)

    def _emit(self, raw: np.ndarray) -> bool:
        """Offer one assembled block, and say whether anybody else now holds it.

        The index counts blocks the receiver produced rather than blocks that survived,
        so a gap tells a consumer that something was dropped, whether a sink had no
        room or a gain change made the samples stale.

        False means the buffer was dropped before anybody saw it and may be filled
        again.  A refusal still counts as handed on, because a sink that declined a
        block does not promise that it kept no reference to the buffer.
        """
        self._produced += 1
        if self._awaiting_gain_change:
            self._dropped_waiting += 1
            if self._dropped_waiting < self._settle_blocks:
                return False
            # The ceiling, rather than the marked block, is what cleared this.  Say so
            # once: every later gain change will do the same, and a sweep moves the
            # gain hundreds of times.
            self._awaiting_gain_change = False
            # Nothing marked the block, so nothing says `gainVals.curr` has caught
            # up.  This drops the check rather than making it against a figure that
            # may still describe the previous gain, and the warning below covers the
            # receiver instead.
            self._pending_gain_check = None
            if not self._gain_flag_reported:
                self._gain_flag_reported = True
                logger.warning(
                    'The receiver delivered %d blocks after a gain change without '
                    'marking one as changed, so this discards by count instead.  '
                    'Measurements stay correct.  The first block after a change may '
                    'predate it, which matters to a gain sweep and to nothing else.',
                    self._dropped_waiting)
            return False
        block = IqBlock(raw=raw, fmt=SDRPLAY_FORMAT, arrived_at=monotonic(),
                        index=self._produced)
        sink = self._sink or self._sync_sink
        if sink is None or not sink.offer(block):
            self._blocks_refused += 1
        return True

    def _on_event(self, event_id: int, tuner: int, params: object,
                  _context: object) -> None:
        """Take one event from the library, on the library's own thread.

        Exceptions must stay inside this C callback.  Save failures for the caller
        that reads overload_status, and keep logging off the callback thread.

        The API requires an acknowledgement for both detection and clearance events.
        SDRplay's examples/sdrplay_api_example.c sends it from EventCallback with the
        tuner supplied by the event.  Release our state lock before that API call.
        """
        try:
            if event_id != api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange:
                return
            overload = params.contents.powerOverloadParams.powerOverloadChangeType
            active = overload == api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected
            with self._overload_lock:
                if not self._accept_overload_events:
                    return
                self._overload_status = OverloadStatus(
                    active=active,
                    detections=self._overload_status.detections + int(active))
            self._library.update(
                int(self._device.dev), tuner,
                api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Ctrl_OverloadMsgAck)
        except Exception as exc:
            with self._overload_lock:
                # Uninit can start after the callback releases the lock to acknowledge.
                if self._accept_overload_events:
                    self._overload_error = str(exc)


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

    def clear(self) -> None:
        """Throw away whatever is waiting, because it describes the old settings.

        Called from the thread that moves the gain rather than from the callback, so
        this races the callback's own put.  A block that arrives between the two is
        dropped by the `grChanged` wait instead, which is the check that covers
        everything the library has not delivered yet.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

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


def _quietly(call: Callable[[], None]) -> None:
    """Run a cleanup call and swallow whatever it raises.

    Callers use this only on paths already unwinding from a failure, so that the
    exception an operator needs to see is the one that propagates.
    """
    try:
        call()
    except Exception:
        logger.debug('A cleanup call failed during shutdown.', exc_info=True)


class _ReceiverStillRunning(RuntimeError):
    """An open attempt whose receiver cannot safely be released."""


def _bounded(call: Callable[[], None], what: str) -> bool:
    """Run a library call on a thread, and give up on it after a timeout.

    The SDRplay API talks to a background service over an interprocess channel, so a
    call can wait on a service that has stopped answering.  Waiting forever at shutdown
    would hang the monitor, where giving up leaves the receiver held until the process
    ends, which is where a blocked call left it anyway.
    """
    finished = threading.Event()
    succeeded = False

    def run() -> None:
        nonlocal succeeded
        try:
            call()
            succeeded = True
        except Exception as exc:
            logger.warning(
                'The receiver failed while %s: %s.  It remains held until this '
                'process exits.', what, exc)
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True, name='sdrplay-close').start()
    if finished.wait(timeout=_DEVICE_CLOSE_TIMEOUT_SECONDS):
        return succeeded
    logger.warning(
        'The receiver did not finish %s within %.0f seconds and was left to the '
        'operating system.  The API waits on a background service, so waiting longer '
        'would only hang this program.', what, _DEVICE_CLOSE_TIMEOUT_SECONDS)
    return False

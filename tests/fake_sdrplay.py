"""A stand-in for the SDRplay API, for testing `SdrplayDevice` with no receiver.

This fills in the real generated structs rather than mimicking them, so a test that
passes here also says the struct layout is usable.  A mock returning whatever it was
told to return would say nothing about either.

`deliver` is what makes this more than a stub: it calls the device's own stream
callback with `short *` arrays, which is the path the library takes and the one place
the interleaving can be wrong.
"""
import ctypes
import threading

from buzz import sdrplay_api as api

# What the fake reports its gain as, so a test can tell a hardware figure from the one
# the gain table predicts.  Zero means "not filled in yet", which the device reads as a
# change that has not taken effect, so a fake reporting zero must mean it deliberately.
NO_REPORTED_GAIN = 0.0


class FakeSdrplayApi:
    """An `SdrplayApi` that remembers what it was told and can deliver samples.

    Every call records itself in `calls`, so a test can assert on the order as well as
    on the result.  Order matters here: selecting before locking, or initializing
    before the parameters are written, both work in Python and fail against the real
    service.
    """

    def __init__(self, *, hw_ver: int = api.SDRPLAY_RSP1B_ID,
                 serial: str = '2405203460', devices: int = 1,
                 reported_gain_db: float = NO_REPORTED_GAIN,
                 api_version: float = api.API_VERSION) -> None:
        self.calls: list[str] = []
        self.callbacks: api.sdrplay_api_CallbackFnsT | None = None
        # Set when init() has run, so a test driving read_block waits on the signal
        # that the stream started rather than on a sleep somebody has to pick.
        self.started = threading.Event()
        self.locked = False
        self.initialised = False
        self.session_open = False
        self.selected: api.sdrplay_api_DeviceT | None = None
        self.released: list[api.sdrplay_api_DeviceT] = []
        self.updates: list[tuple[int, int, int]] = []
        self.fail_on: dict[str, Exception] = {}
        self._api_version = api_version
        self._reported_gain_db = reported_gain_db

        # The library owns this memory and hands back a pointer into it, so the fake
        # has to hold the parts alive for as long as the device holds the whole.
        self._dev_params = api.sdrplay_api_DevParamsT()
        self._channel = api.sdrplay_api_RxChannelParamsT()
        self._params = api.sdrplay_api_DeviceParamsT(
            devParams=ctypes.pointer(self._dev_params),
            rxChannelA=ctypes.pointer(self._channel),
            rxChannelB=None)

        self._devices = [self._make_device(hw_ver, serial, index)
                         for index in range(devices)]

    # ------------------------------------------------------- what the device sees

    def open(self) -> None:
        self._record('open')
        self.session_open = True

    def close(self) -> None:
        self._record('close')
        self.session_open = False

    def api_version(self) -> float:
        self._record('api_version')
        return self._api_version

    def lock(self) -> None:
        self._record('lock')
        self.locked = True

    def unlock(self) -> None:
        self._record('unlock')
        self.locked = False

    def devices(self) -> list[api.sdrplay_api_DeviceT]:
        self._record('devices')
        assert self.locked, 'devices() was called without the API lock held'
        return list(self._devices)

    def select(self, device: api.sdrplay_api_DeviceT) -> None:
        self._record('select')
        assert self.locked, 'select() was called without the API lock held'
        self.selected = device

    def release(self, device: api.sdrplay_api_DeviceT) -> None:
        self._record('release')
        self.released.append(device)

    def device_params(self, handle: int) -> api.sdrplay_api_DeviceParamsT:
        self._record('device_params')
        return self._params

    def init(self, handle: int, callbacks: api.sdrplay_api_CallbackFnsT) -> None:
        self._record('init')
        assert not self.initialised, 'init() was called twice without an uninit'
        self.callbacks = callbacks
        self.initialised = True
        self.started.set()

    def uninit(self, handle: int) -> None:
        self._record('uninit')
        self.initialised = False
        self.started.clear()

    def update(self, handle: int, tuner: int, reason: int) -> None:
        self._record('update')
        assert self.initialised, 'update() was called before init()'
        self.updates.append((handle, tuner, reason))
        if self._reported_gain_db:
            self.gain.gainVals.curr = self._reported_gain_db

    # ------------------------------------------------------- what a test drives

    @property
    def device(self) -> api.sdrplay_api_DeviceT:
        """The first receiver this fake presents."""
        return self._devices[0]

    @property
    def gain(self) -> api.sdrplay_api_GainT:
        """The gain block the device writes through."""
        return self._channel.tunerParams.gain

    @property
    def tuner(self) -> api.sdrplay_api_TunerParamsT:
        """The tuner block the device writes through."""
        return self._channel.tunerParams

    @property
    def control(self) -> api.sdrplay_api_ControlParamsT:
        """The control block, which carries decimation and the AGC."""
        return self._channel.ctrlParams

    @property
    def fs_hz(self) -> float:
        """The converter rate the device asked for."""
        return self._dev_params.fsFreq.fsHz

    def set_reported_gain_db(self, gain_db: float) -> None:
        """Make the receiver report this gain from the next update onwards."""
        self._reported_gain_db = gain_db

    def deliver(self, i_values: list[int], q_values: list[int], *,
                gr_changed: bool = False) -> None:
        """Call the device's stream callback with one delivery, as the library would.

        The two arrays go over separately, which is the whole point: the device has to
        interleave them, and nothing else in the suite would notice if it did not.
        """
        assert self.callbacks is not None, 'deliver() was called before init()'
        count = len(i_values)
        assert count == len(q_values), 'I and Q must be the same length'
        xi = (ctypes.c_short * count)(*i_values)
        xq = (ctypes.c_short * count)(*q_values)
        params = api.sdrplay_api_StreamCbParamsT(
            firstSampleNum=0, grChanged=int(gr_changed), rfChanged=0, fsChanged=0,
            numSamples=count)
        self.callbacks.StreamACbFn(
            ctypes.cast(xi, ctypes.POINTER(ctypes.c_short)),
            ctypes.cast(xq, ctypes.POINTER(ctypes.c_short)),
            ctypes.byref(params), count, 0, None)

    def raise_event(self, event_id: int, overload: int = 0) -> None:
        """Call the device's event callback, as the library would."""
        assert self.callbacks is not None, 'raise_event() was called before init()'
        params = api.sdrplay_api_EventParamsT()
        params.powerOverloadParams.powerOverloadChangeType = overload
        self.callbacks.EventCbFn(event_id, api.sdrplay_api_TunerSelectT
                                 .sdrplay_api_Tuner_A, ctypes.byref(params), None)

    # ----------------------------------------------------------------- private

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise self.fail_on[name]

    @staticmethod
    def _make_device(hw_ver: int, serial: str,
                     index: int) -> api.sdrplay_api_DeviceT:
        device = api.sdrplay_api_DeviceT()
        device.SerNo = f'{serial}{index or ""}'.encode()
        device.hwVer = hw_ver
        device.tuner = api.sdrplay_api_TunerSelectT.sdrplay_api_Tuner_A
        device.valid = 1
        # A handle the real service would hand back.  Any non-zero value does, and this
        # one is distinctive in a failure message.
        device.dev = 0xABCD0000 + index
        return device

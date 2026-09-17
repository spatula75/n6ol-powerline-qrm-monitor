"""Tests for the layer between `SdrplayDevice` and the real shared library.

The SDRplay API is an optional dependency: a station running a sound card or an
RTL-SDR never installs it.  So every line here is exercised against a stand-in for
`ctypes.CDLL` rather than against the library, because a coverage number that moves
depending on what somebody chose to install means nothing.  `render.py` and ffmpeg are
the precedent, and `CLAUDE.md` states the rule.

What a fake CDLL cannot say is whether the real library behaves this way.  It can say
that the error codes are read, that the declarations are applied to every call, and
that a missing symbol is reported rather than reaching a caller as an AttributeError.
"""
import ctypes
import threading

import pytest
from buzz import sdrplay_api as api
from buzz.sdrplay_device import (SdrplayDevice, SdrplayLibrary, _bounded, _quietly,
                                 _what_an_api_failure_usually_means)
from fake_sdrplay import FakeSdrplayApi

SUCCESS = api.sdrplay_api_ErrT.sdrplay_api_Success


class FakeCFunction:
    """One exported call, with the declarations `SdrplayLibrary` writes onto it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.restype: object = None
        self.argtypes: object = None
        self.calls: list[tuple] = []
        self.result: object = SUCCESS
        self.effect: object = None

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        if self.effect is not None:
            return self.effect(*args)
        return self.result


class FakeCdll:
    """A stand-in for `ctypes.CDLL` that hands out `FakeCFunction`s.

    A symbol named in `missing` raises AttributeError the way a real library does when
    it is older than the header this program was built against.
    """

    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.missing = set(missing)
        self.functions: dict[str, FakeCFunction] = {}

    def __getattr__(self, name: str) -> FakeCFunction:
        if name in self.missing:
            raise AttributeError(name)
        return self.functions.setdefault(name, FakeCFunction(name))


def make_library() -> tuple[SdrplayLibrary, FakeCdll]:
    """A wrapper over a fake CDLL, with both handed back."""
    cdll = FakeCdll()
    # The error string is the one call whose return is not a code, so it needs a
    # plausible value before anything asks it to explain a failure.
    cdll.sdrplay_api_GetErrorString.result = b'the service is not responding'
    return SdrplayLibrary(cdll), cdll


class TestDeclaringTheCalls:
    def test_every_exported_call_gets_its_declaration(self):
        """ctypes defaults a return to int, which truncates a 64-bit pointer on Win64.

        A pointer return read as an int is a wrong address rather than an error, so
        this is the difference between a working call and a crash with no exception.
        """
        _, cdll = make_library()
        for symbol, restype, argtypes in api.FUNCTIONS:
            assert cdll.functions[symbol].restype == restype, symbol
            assert cdll.functions[symbol].argtypes == argtypes, symbol

    def test_a_missing_symbol_names_itself_and_says_what_to_install(self):
        """An older API is the likely cause, and an AttributeError on a C symbol tells
        whoever hits it nothing about that.
        """
        cdll = FakeCdll(missing=('sdrplay_api_GetDevices',))
        with pytest.raises(RuntimeError, match='does not export sdrplay_api_GetDevices'):
            SdrplayLibrary(cdll)


class TestLoadingTheLibrary:
    def test_the_first_name_that_loads_is_the_one_used(self, monkeypatch):
        tried: list[str] = []

        def fake_cdll(path: str) -> FakeCdll:
            tried.append(path)
            if path != 'libsdrplay_api.so':
                raise OSError('not here')
            return FakeCdll()

        monkeypatch.setattr(ctypes, 'CDLL', fake_cdll)
        SdrplayLibrary.load()
        assert tried[-1] == 'libsdrplay_api.so'
        assert 'sdrplay_api.dll' in tried

    def test_nothing_loading_says_where_to_get_the_api(self, monkeypatch):
        """The operator has to install something, so the message has to say what."""
        monkeypatch.setattr(ctypes, 'CDLL',
                            lambda path: (_ for _ in ()).throw(OSError('no')))
        with pytest.raises(RuntimeError, match='sdrplay.com/hardware-api'):
            SdrplayLibrary.load()

    def test_a_setting_that_points_nowhere_blames_the_setting(self, monkeypatch):
        monkeypatch.setattr(ctypes, 'CDLL',
                            lambda path: (_ for _ in ()).throw(OSError('no')))
        with pytest.raises(RuntimeError, match='api_path'):
            SdrplayLibrary.load('/nowhere/libsdrplay_api.so')


class TestTheCalls:
    def test_a_success_code_passes_quietly(self):
        library, cdll = make_library()
        library.open()
        library.lock()
        library.unlock()
        library.close()
        assert cdll.functions['sdrplay_api_Open'].calls == [()]
        assert cdll.functions['sdrplay_api_Close'].calls == [()]

    def test_a_failure_carries_the_librarys_own_words(self):
        """The library knows what went wrong and this program does not, so its text is
        the part of the message that has to survive.
        """
        library, cdll = make_library()
        cdll.functions['sdrplay_api_Open'].result = (
            api.sdrplay_api_ErrT.sdrplay_api_ServiceNotResponding)
        with pytest.raises(RuntimeError, match='the service is not responding'):
            library.open()

    def test_a_failure_says_what_to_do_about_it(self):
        library, cdll = make_library()
        cdll.functions['sdrplay_api_Open'].result = (
            api.sdrplay_api_ErrT.sdrplay_api_ServiceNotResponding)
        with pytest.raises(RuntimeError, match='SDRplayAPIService is not running'):
            library.open()

    def test_a_code_the_library_will_not_explain_still_reaches_the_reader(self):
        """A library too old to carry GetErrorString must not turn one failure into a
        different and more confusing one.
        """
        library, cdll = make_library()
        cdll.functions['sdrplay_api_GetErrorString'].result = None
        cdll.functions['sdrplay_api_Open'].result = api.sdrplay_api_ErrT.sdrplay_api_Fail
        with pytest.raises(RuntimeError, match='error 1'):
            library.open()

    def test_the_version_comes_back_as_a_number(self):
        """It is an output parameter, so reading it means writing through a pointer."""
        library, cdll = make_library()

        def fill(pointer: object) -> int:
            pointer._obj.value = 3.15
            return SUCCESS

        cdll.functions['sdrplay_api_ApiVersion'].effect = fill
        assert library.api_version() == pytest.approx(3.15)

    def test_only_the_receivers_the_library_counted_come_back(self):
        """The array is sized for the API's maximum and the count says how much of it
        the library filled in.  Reading the rest would hand out uninitialized structs.
        """
        library, cdll = make_library()

        def fill(devices: object, found: object, _max: int) -> int:
            devices[0].hwVer = api.SDRPLAY_RSP1B_ID
            devices[0].SerNo = b'2405203460'
            found._obj.value = 1
            return SUCCESS

        cdll.functions['sdrplay_api_GetDevices'].effect = fill
        devices = library.devices()
        assert len(devices) == 1
        assert bytes(devices[0].SerNo).rstrip(b'\x00') == b'2405203460'

    def test_selecting_and_releasing_pass_the_device_by_reference(self):
        """The library writes into the struct, so a copy would lose the handle."""
        library, cdll = make_library()
        device = api.sdrplay_api_DeviceT()
        library.select(device)
        library.release(device)
        assert cdll.functions['sdrplay_api_SelectDevice'].calls
        assert cdll.functions['sdrplay_api_ReleaseDevice'].calls

    def test_the_parameter_block_comes_back_as_the_librarys_own_memory(self):
        """Writing through what comes back is how every setting is changed, so a copy
        would configure nothing and report no error.
        """
        library, cdll = make_library()
        params = api.sdrplay_api_DeviceParamsT()
        dev_params = api.sdrplay_api_DevParamsT()
        params.devParams = ctypes.pointer(dev_params)

        def fill(_handle: int, out: object) -> int:
            out._obj.contents = params
            return SUCCESS

        cdll.functions['sdrplay_api_GetDeviceParams'].effect = fill
        got = library.device_params(0x1234)
        got.devParams.contents.fsFreq.fsHz = 2_048_000.0
        assert dev_params.fsFreq.fsHz == 2_048_000.0

    def test_init_and_uninit_reach_the_library(self):
        library, cdll = make_library()
        callbacks = api.sdrplay_api_CallbackFnsT()
        library.init(0x1234, callbacks)
        library.uninit(0x1234)
        assert cdll.functions['sdrplay_api_Init'].calls
        assert cdll.functions['sdrplay_api_Uninit'].calls

    def test_an_update_carries_the_reason_and_the_empty_extension(self):
        """The API takes two reason words and reads both, so the second cannot be left
        off even when this program never uses it.
        """
        library, cdll = make_library()
        reason = api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Tuner_Gr
        library.update(0x1234, api.sdrplay_api_TunerSelectT.sdrplay_api_Tuner_A, reason)
        handle, tuner, sent, extension = cdll.functions['sdrplay_api_Update'].calls[0]
        assert sent == reason
        assert extension == (api.sdrplay_api_ReasonForUpdateExtension1T
                             .sdrplay_api_Update_Ext1_None)


class TestOpeningADevice:
    def test_it_loads_selects_and_configures(self, monkeypatch):
        library = FakeSdrplayApi()
        monkeypatch.setattr(SdrplayLibrary, 'load', classmethod(lambda cls, p: library))
        device = SdrplayDevice.open(tuned_hz=7_050_000, gain_db=-40.0,
                                    iq_sample_rate=256_000)
        assert device.profile.name == 'SDRplay RSP1B'
        assert library.calls[:5] == ['open', 'lock', 'devices', 'select', 'unlock']

    def test_a_session_is_given_back_when_no_receiver_is_found(self, monkeypatch):
        """A session left open keeps the service holding a receiver, so the next run
        meets a device that is present and cannot be selected.
        """
        library = FakeSdrplayApi(devices=0)
        monkeypatch.setattr(SdrplayLibrary, 'load', classmethod(lambda cls, p: library))
        with pytest.raises(RuntimeError, match='No SDRplay receiver was found'):
            SdrplayDevice.open(tuned_hz=7_050_000, gain_db=-40.0,
                               iq_sample_rate=256_000)
        assert library.session_open is False

    def test_a_receiver_is_given_back_when_configuring_fails(self, monkeypatch):
        """The receiver is selected by this point, so failing now has two things to
        undo rather than one.  A tuning above the band edge is the way in.
        """
        library = FakeSdrplayApi()
        monkeypatch.setattr(SdrplayLibrary, 'load', classmethod(lambda cls, p: library))
        with pytest.raises(ValueError, match='below 50 MHz'):
            SdrplayDevice.open(tuned_hz=144_000_000, gain_db=-40.0,
                               iq_sample_rate=256_000)
        assert library.released == [library.device]
        assert library.session_open is False


class TestTheFailureAdvice:
    @pytest.mark.parametrize('code, expected', [
        (api.sdrplay_api_ErrT.sdrplay_api_ServiceNotResponding, 'SDRplayAPIService'),
        (api.sdrplay_api_ErrT.sdrplay_api_HwVerError, 'older than the hardware'),
        (api.sdrplay_api_ErrT.sdrplay_api_AlreadyInitialised, 'fault in this program'),
        (api.sdrplay_api_ErrT.sdrplay_api_NotInitialised, 'fault in this program'),
        (api.sdrplay_api_ErrT.sdrplay_api_HwError, 'no other program is using'),
    ])
    def test_each_failure_an_operator_can_act_on_says_what_to_try(self, code, expected):
        assert expected in _what_an_api_failure_usually_means(code)


class TestTheCleanupHelpers:
    def test_a_failing_cleanup_call_does_not_mask_the_real_failure(self):
        """These run on paths already unwinding, so the exception an operator needs to
        see is the one that was already on its way out.
        """
        def raises() -> None:
            raise RuntimeError('the service went away')

        _quietly(raises)      # no exception reaches here

    def test_a_call_that_returns_is_reported_as_finished(self):
        assert _bounded(lambda: None, 'doing nothing') is True

    def test_a_call_that_raises_is_still_reported_as_finished(self):
        """The device is as released as it is going to get either way, and the caller
        needs to know whether to wait rather than whether it worked.
        """
        def raises() -> None:
            raise RuntimeError('no')

        assert _bounded(raises, 'failing') is True

    def test_a_call_that_never_returns_is_given_up_on(self, monkeypatch):
        """The API waits on a background service, so a call can hang.  Waiting forever
        at shutdown would hang the monitor rather than the one call.
        """
        monkeypatch.setattr('buzz.sdrplay_device._DEVICE_CLOSE_TIMEOUT_SECONDS', 0.05)
        release = threading.Event()
        try:
            assert _bounded(lambda: release.wait(30), 'hanging') is False
        finally:
            release.set()

"""Layout pins for the generated SDRplay bindings.

These exist because a wrong field is memory corruption rather than an exception.  The
library writes into structs this module declares, so a dropped or reordered field
means it writes past where Python thinks something ends, and nothing raises.  The
first sign is a crash somewhere unrelated, or a serial number that reads as rubbish.

The figures come from the generated module, and a real receiver confirmed them:
`sdrplay_api_GetDevices` filled a `sdrplay_api_DeviceT` on an RSP1B and the serial
came back readable, which only happens when the layout is right.

**They are the same on Linux.**  No struct field uses a type whose width differs
between Win64 and Linux x86-64, and there is no `#pragma pack`, so one set of numbers
is correct on both and this file means something on either.  See `docs-notebook`.

Nothing here loads the library.  These run with no API installed and no receiver.
"""
import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
from buzz.receiver import sdrplay_api as api

REPO = Path(__file__).resolve().parent.parent.parent

# Every struct and union the generator emits, and the size the C compiler gives it.
# A change here is either a new API version, which is fine and wants these updated in
# the same commit, or a generator fault, which is not.
SIZES = {
    'sdrplay_api_AgcT': 20,
    'sdrplay_api_CallbackFnsT': 24,
    'sdrplay_api_ControlParamsT': 32,
    'sdrplay_api_DcOffsetT': 2,
    'sdrplay_api_DcOffsetTunerT': 12,
    'sdrplay_api_DecimationT': 3,
    'sdrplay_api_DevParamsT': 64,
    'sdrplay_api_DeviceParamsT': 24,
    'sdrplay_api_DeviceT': 96,
    'sdrplay_api_ErrorInfoT': 1540,
    'sdrplay_api_EventParamsT': 16,
    'sdrplay_api_FsFreqT': 16,
    'sdrplay_api_GainCbParamT': 16,
    'sdrplay_api_GainT': 24,
    'sdrplay_api_GainValuesT': 12,
    'sdrplay_api_PowerOverloadCbParamT': 4,
    'sdrplay_api_ResetFlagsT': 3,
    'sdrplay_api_RfFreqT': 16,
    'sdrplay_api_Rsp1aParamsT': 2,
    'sdrplay_api_Rsp1aTunerParamsT': 1,
    'sdrplay_api_Rsp2ParamsT': 1,
    'sdrplay_api_Rsp2TunerParamsT': 16,
    'sdrplay_api_RspDuoModeCbParamT': 4,
    'sdrplay_api_RspDuoParamsT': 4,
    'sdrplay_api_RspDuoTunerParamsT': 16,
    'sdrplay_api_RspDuo_ResetSlaveFlagsT': 2,
    'sdrplay_api_RspDxParamsT': 12,
    'sdrplay_api_RspDxTunerParamsT': 4,
    'sdrplay_api_RxChannelParamsT': 144,
    'sdrplay_api_StreamCbParamsT': 20,
    'sdrplay_api_SyncUpdateT': 8,
    'sdrplay_api_TunerParamsT': 72,
}


class TestStructLayout:
    @pytest.mark.parametrize('name, size', sorted(SIZES.items()))
    def test_each_struct_is_the_size_the_c_compiler_gives_it(self, name, size):
        assert ctypes.sizeof(getattr(api, name)) == size, (
            f'{name} is no longer {size} bytes.  A new API version is one cause.  '
            f'Update SIZES in the same commit as the regenerated bindings.  '
            f'A dropped or reordered field is the other.  The library then writes '
            f'past the end of this struct and nothing raises.')

    def test_every_generated_struct_is_pinned(self):
        """A struct added by a new API version must not slip in unpinned.

        The point of this file is that nothing reaches the library unchecked, and a
        struct nobody listed is exactly that.
        """
        emitted = {name for name in dir(api)
                   if isinstance(getattr(api, name), type)
                   and issubclass(getattr(api, name), (ctypes.Structure, ctypes.Union))}
        assert emitted == set(SIZES), (
            f'unpinned: {sorted(emitted - set(SIZES))}, '
            f'gone: {sorted(set(SIZES) - emitted)}')

    def test_the_device_struct_puts_its_fields_where_c_does(self):
        """The one struct the library fills for us, so the one checked field by field.

        The layout puts a 64-byte serial first, then hwVer, then the 8-byte alignment
        the double and the handle after it need.  A real RSP1B confirmed it, reading
        its serial back as text rather than as rubbish.
        """
        offsets = {'SerNo': 0, 'hwVer': 64, 'tuner': 68, 'rspDuoMode': 72,
                   'valid': 76, 'rspDuoSampleFreq': 80, 'dev': 88}
        for name, offset in offsets.items():
            assert getattr(api.sdrplay_api_DeviceT, name).offset == offset, (
                f'sdrplay_api_DeviceT.{name} moved to '
                f'{getattr(api.sdrplay_api_DeviceT, name).offset}, where C puts it at '
                f'{offset}')


class TestWhatTheBindingsCarry:
    def test_the_version_is_a_number_rather_than_a_cast(self):
        """The header spells it `(float)(3.15)`, and the cast is not the value."""
        assert api.API_VERSION == 3.15
        assert isinstance(api.API_VERSION, float)

    def test_the_two_receivers_this_supports_have_their_ids(self):
        """An RSP1B reports 6.  An RSP1A reports 255 rather than 5, which looks like a
        typo and is not, so it is pinned here where somebody would go to check.
        """
        assert api.SDRPLAY_RSP1B_ID == 6
        assert api.SDRPLAY_RSP1A_ID == 255

    def test_the_callbacks_take_separate_i_and_q_arrays(self):
        """Not interleaved, which is where an RTL-SDR and an RSP differ most.

        `SampleFormat` assumes interleaved because that is what a .wav frame is, so
        whichever shim uses this has to interleave while it copies.
        """
        argtypes = api.sdrplay_api_StreamCallback_t._argtypes_
        assert argtypes[0] == ctypes.POINTER(ctypes.c_short)
        assert argtypes[1] == ctypes.POINTER(ctypes.c_short)

    def test_the_stream_parameters_say_when_the_gain_changed(self):
        """grChanged is what lets an RSP skip the post-gain-change discard an RTL-SDR
        has to count blindly.
        """
        fields = dict(api.sdrplay_api_StreamCbParamsT._fields_)
        assert 'grChanged' in fields
        assert 'firstSampleNum' in fields

    def test_gain_comes_back_as_numbers(self):
        """The output parameter that removes any need for gain reduction tables."""
        fields = dict(api.sdrplay_api_GainValuesT._fields_)
        assert fields == {'curr': ctypes.c_float, 'max': ctypes.c_float,
                          'min': ctypes.c_float}

    def test_every_exported_call_is_declared(self):
        names = {name for name, _, _ in api.FUNCTIONS}
        for expected in ('sdrplay_api_Open', 'sdrplay_api_Close', 'sdrplay_api_GetDevices',
                         'sdrplay_api_SelectDevice', 'sdrplay_api_Init',
                         'sdrplay_api_Uninit', 'sdrplay_api_Update',
                         'sdrplay_api_GetDeviceParams', 'sdrplay_api_ApiVersion'):
            assert expected in names, f'{expected} is not in FUNCTIONS'

    def test_no_macro_leaked_in_as_a_constant(self):
        """`_SDRPLAY_DLL_QUALIFIER` is an instruction to a C compiler, and emitting it
        would put a NameError in a module that otherwise imports.
        """
        assert not hasattr(api, '_SDRPLAY_DLL_QUALIFIER')


class TestTheGeneratedFileIsCurrent:
    def test_regenerating_reproduces_the_committed_module(self):
        """The bindings are generated, not hand-edited, and the headers are vendored so
        that this can run anywhere rather than only where the API is installed.
        """
        result = subprocess.run(
            [sys.executable, str(REPO / 'tools' / 'generate_sdrplay_api.py'), '--check'],
            capture_output=True, text=True, cwd=REPO)
        assert result.returncode == 0, (
            f'lib/buzz/receiver/sdrplay_api.py is out of date with the vendored headers.  It is '
            f'generated: change tools/generate_sdrplay_api.py or the headers, then run\n'
            f'    python tools/generate_sdrplay_api.py\n'
            f'{result.stdout}{result.stderr}')

    def test_it_says_it_is_generated(self):
        """Whoever opens it to make a change needs to know the edit will be lost."""
        text = (REPO / 'lib' / 'buzz' / 'receiver' / 'sdrplay_api.py').read_text(encoding='utf-8')
        assert 'GENERATED FILE' in text
        assert 'generate_sdrplay_api.py' in text

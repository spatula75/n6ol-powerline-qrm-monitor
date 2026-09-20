"""Tests for tools/generate_sdrplay_api.py.

These run against small headers written here rather than against the vendored ones,
because each states one rule and says which rule broke when it fails.
`tests/receiver/test_sdrplay_api.py` covers the real headers end to end, by pinning what comes
out of them.

Most of these exist because the generator got that case wrong first.  Every one of the
six faults found while writing it is here, since each would have been a hand
transcription error too, and these catch a generator that grows one back.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import generate_sdrplay_api as gen  # noqa: E402


def _api(tmp_path, **files):
    """Write `files` into a directory and parse them in HEADER_ORDER."""
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding='utf-8')
    for name in gen.HEADER_ORDER:
        if name not in files:
            (tmp_path / name).write_text('', encoding='utf-8')
    return gen.parse(tmp_path)


class TestDefines:
    def test_a_numeric_define_becomes_a_constant(self, tmp_path):
        api = _api(tmp_path, **{'sdrplay_api.h': '#define SDRPLAY_RSP1B_ID (6)\n'})
        assert ('SDRPLAY_RSP1B_ID', '6') in api.defines

    def test_a_cast_is_not_the_value(self, tmp_path):
        """The header spells the version `(float)(3.15)`, and a naive read of the first
        parenthesised group takes the cast and emits `API_VERSION = float`.
        """
        api = _api(tmp_path, **{'sdrplay_api.h': '#define VER (float)(3.15)\n'})
        assert ('VER', '3.15') in api.defines

    def test_a_macro_is_not_a_constant(self, tmp_path):
        """`_SDRPLAY_DLL_QUALIFIER __declspec(dllimport)` is an instruction to a C
        compiler.  Emitting it puts a NameError in a module that otherwise imports.
        """
        header = '#define _SDRPLAY_DLL_QUALIFIER __declspec(dllimport)\n'
        assert _api(tmp_path, **{'sdrplay_api.h': header}).defines == []

    def test_one_name_defined_two_ways_is_refused(self, tmp_path):
        """The generator has no preprocessor, so it reads both branches of a `#if`.

        Two numeric defines of one name would both be emitted, and Python would keep
        whichever came last.  On these headers that would hand a Linux build the
        Windows value with nothing to notice.  The headers do this today only for
        `_SDRPLAY_DLL_QUALIFIER`, which is not a number and is dropped before here.
        """
        header = """#ifdef _WIN32
#define SLOTS 4
#else
#define SLOTS 8
#endif
"""
        with pytest.raises(ValueError, match='SLOTS is 4 .* and 8 '):
            _api(tmp_path, **{'sdrplay_api.h': header})

    def test_one_name_defined_the_same_way_twice_is_allowed(self, tmp_path):
        """The check is on the value rather than on the name, because a repeat that
        agrees with itself costs a reader nothing and cannot pick a wrong branch.
        """
        header = """#define SLOTS 4
#define SLOTS 4
"""
        assert _api(tmp_path, **{'sdrplay_api.h': header}).defines == [
            ('SLOTS', '4'), ('SLOTS', '4')]

    def test_a_header_guard_does_not_swallow_the_line_after_it(self, tmp_path):
        """The fault that cost an entire enum.

        A guard has a name and no value, and matching the separator with `\\s+` lets it
        cross the newline and take the next line as its value, consuming whatever was
        declared there.
        """
        header = (
            '#ifndef SDRPLAY_API_H\n'
            '#define SDRPLAY_API_H\n'
            'typedef enum\n{\n    a = 1,\n} thingT;\n'
        )
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.enum_names == {'thingT'}, (
            'the guard ate the enum that followed it')
        assert api.defines == []


class TestEnums:
    def test_members_and_values_survive(self, tmp_path):
        header = 'typedef enum\n{\n    one = 1,\n    big = 255,\n} idT;\n'
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.enums == [('idT', [('one', 1), ('big', 255)])]

    def test_a_directive_in_the_body_is_refused(self, tmp_path):
        """A member inside `#if 0` reaches Python looking exactly like a live one.

        This is the quiet half of the same fault as the duplicate define above.  The
        member name comes out mangled and the renderer tidies it back to a bare
        identifier, so the generated module imports and the dead value is simply
        there.  Nothing downstream can tell it apart from a real one.
        """
        header = """typedef enum
{
    a = 0,
#if 0
    b = 1,
#endif
    c = 2,
} idT;
"""
        with pytest.raises(ValueError, match='preprocessor directive in its body'):
            _api(tmp_path, **{'sdrplay_api.h': header})

    def test_an_implicit_value_is_refused_rather_than_guessed(self, tmp_path):
        """Every enum in these headers states every value, so a counter would be
        machinery nothing needs and a silent source of wrong numbers if it drifted.
        """
        header = 'typedef enum\n{\n    a = 0,\n    b,\n} idT;\n'
        with pytest.raises(ValueError, match='no explicit value'):
            _api(tmp_path, **{'sdrplay_api.h': header})


class TestStructs:
    def test_scalars_map_to_ctypes(self, tmp_path):
        header = ('typedef struct\n{\n    unsigned char a;\n    double b;\n'
                  '    float c;\n} thingT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs == [('thingT', 'ctypes.Structure', [
            ('a', 'ctypes.c_ubyte'), ('b', 'ctypes.c_double'), ('c', 'ctypes.c_float')])]

    def test_an_enum_field_is_an_int(self, tmp_path):
        """A C enum is an int.  The IntEnum generated beside it is for reading values in
        Python and cannot appear in _fields_, which needs a real ctypes type.
        """
        header = ('typedef enum\n{\n    a = 1,\n} modeT;\n'
                  'typedef struct\n{\n    modeT mode;\n} thingT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs[0][2] == [('mode', 'ctypes.c_int')]

    def test_a_nested_struct_is_held_by_value(self, tmp_path):
        """sdrplay_api_DevParamsT embeds the RSPduo and RSPdx structs this way, which is
        why every struct has to be right even to support one receiver.
        """
        header = ('typedef struct\n{\n    int x;\n} innerT;\n'
                  'typedef struct\n{\n    innerT inner;\n} outerT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs[1][2] == [('inner', 'innerT')]

    def test_a_pointer_to_a_struct_keeps_the_type(self, tmp_path):
        header = ('typedef struct\n{\n    int x;\n} innerT;\n'
                  'typedef struct\n{\n    innerT *inner;\n} outerT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs[1][2] == [('inner', 'ctypes.POINTER(innerT)')]

    def test_an_array_keeps_the_name_of_its_extent(self, tmp_path):
        """`char SerNo[SDRPLAY_MAX_SER_NO_LEN]` stays readable rather than becoming 64,
        because the generator emits that constant too.
        """
        header = ('#define LEN (64)\n'
                  'typedef struct\n{\n    char SerNo[LEN];\n} thingT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs[0][2] == [('SerNo', 'ctypes.c_char * LEN')]

    def test_a_union_is_a_union(self, tmp_path):
        """sdrplay_api_EventParamsT is one, and ctypes lays every member at offset 0."""
        header = ('typedef struct\n{\n    int x;\n} aT;\n'
                  'typedef union\n{\n    aT a;\n} eitherT;\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.structs[1][:2] == ('eitherT', 'ctypes.Union')

    def test_an_unknown_type_is_refused(self, tmp_path):
        header = 'typedef struct\n{\n    mystery m;\n} thingT;\n'
        with pytest.raises(ValueError, match='No ctypes equivalent'):
            _api(tmp_path, **{'sdrplay_api.h': header})

    def test_a_field_it_cannot_read_names_itself(self, tmp_path):
        header = 'typedef struct\n{\n    int (*weird[3])(void);\n} thingT;\n'
        with pytest.raises(ValueError, match='Could not read the field'):
            _api(tmp_path, **{'sdrplay_api.h': header})


class TestSourceOrder:
    def test_a_struct_a_callback_and_a_struct_keep_their_order(self, tmp_path):
        """The fault that took two goes to fix, once in parsing and once in rendering.

        A header states a struct, then a callback typedef pointing at it, then the
        struct holding that callback.  The dependency runs both ways inside one file,
        so processing by category cannot be repaired by reordering the categories.
        """
        header = (
            'typedef struct\n{\n    int n;\n} paramsT;\n'
            'typedef void (*cbT)(paramsT *p, void *ctx);\n'
            'typedef struct\n{\n    cbT fn;\n} fnsT;\n'
        )
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.order == [('struct', 'paramsT'), ('callback', 'cbT'),
                             ('struct', 'fnsT')]
        assert api.structs[1][2] == [('fn', 'cbT')]

    def test_the_rendered_module_follows_the_same_order(self, tmp_path):
        header = (
            'typedef struct\n{\n    int n;\n} paramsT;\n'
            'typedef void (*cbT)(paramsT *p);\n'
            'typedef struct\n{\n    cbT fn;\n} fnsT;\n'
        )
        text = gen.render(_api(tmp_path, **{'sdrplay_api.h': header}), tmp_path)
        assert text.index('class paramsT') < text.index('cbT = ctypes.CFUNCTYPE')
        assert text.index('cbT = ctypes.CFUNCTYPE') < text.index('class fnsT')


class TestExportedFunctions:
    def test_they_come_from_the_headers_own_export_block(self, tmp_path):
        """Read from there rather than from the function-pointer typedefs beside them,
        because that block is the header's statement of what the library exports.  A
        typedef with no matching export would be looked up and fail at load.
        """
        header = (
            'typedef enum\n{\n    ok = 0,\n} errT;\n'
            'typedef struct\n{\n    int n;\n} devT;\n'
            '_SDRPLAY_DLL_QUALIFIER errT sdrplay_api_Open(void);\n'
            '_SDRPLAY_DLL_QUALIFIER errT sdrplay_api_GetDevices(devT *d, unsigned int *n);\n'
        )
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.functions == [
            ('sdrplay_api_Open', 'ctypes.c_int', []),
            ('sdrplay_api_GetDevices', 'ctypes.c_int',
             ['ctypes.POINTER(devT)', 'ctypes.POINTER(ctypes.c_uint)']),
        ]

    def test_a_const_char_return_is_a_string(self, tmp_path):
        """sdrplay_api_GetErrorString returns one, and c_char_p hands back bytes where
        c_void_p would hand back an address.
        """
        header = '_SDRPLAY_DLL_QUALIFIER const char* sdrplay_api_GetErrorString(int e);\n'
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.functions[0][1] == 'ctypes.c_char_p'

    def test_a_pointer_to_an_enum_is_a_pointer_to_an_int(self, tmp_path):
        header = ('typedef enum\n{\n    a = 1,\n} tunerT;\n'
                  '_SDRPLAY_DLL_QUALIFIER int sdrplay_api_Swap(tunerT *t);\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.functions[0][2] == ['ctypes.POINTER(ctypes.c_int)']

    def test_a_pointer_to_a_pointer_survives(self, tmp_path):
        """sdrplay_api_GetDeviceParams takes one, to hand back a struct it owns."""
        header = ('typedef struct\n{\n    int n;\n} paramsT;\n'
                  '_SDRPLAY_DLL_QUALIFIER int sdrplay_api_Get(paramsT **p);\n')
        api = _api(tmp_path, **{'sdrplay_api.h': header})
        assert api.functions[0][2] == ['ctypes.POINTER(ctypes.POINTER(paramsT))']


class TestRunningIt:
    def test_a_missing_header_says_which_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='sdrplay_api_tuner.h'):
            gen.parse(tmp_path)

    def test_check_reports_a_stale_output(self, tmp_path, capsys):
        headers = tmp_path / 'inc'
        headers.mkdir()
        for name in gen.HEADER_ORDER:
            (headers / name).write_text('#define A (1)\n', encoding='utf-8')
        output = tmp_path / 'out.py'
        output.write_text('stale\n', encoding='utf-8')
        assert gen.main(['--headers', str(headers), '--output', str(output),
                         '--check']) == 1
        assert 'out of date' in capsys.readouterr().out

    def test_check_passes_once_it_has_been_written(self, tmp_path, capsys):
        headers = tmp_path / 'inc'
        headers.mkdir()
        for name in gen.HEADER_ORDER:
            (headers / name).write_text('#define A (1)\n', encoding='utf-8')
        output = tmp_path / 'out.py'
        assert gen.main(['--headers', str(headers), '--output', str(output)]) == 0
        assert gen.main(['--headers', str(headers), '--output', str(output),
                         '--check']) == 0
        assert 'is current' in capsys.readouterr().out

    def test_the_written_module_imports(self, tmp_path):
        """The end of the job: what comes out has to be Python that runs."""
        headers = tmp_path / 'inc'
        headers.mkdir()
        for name in gen.HEADER_ORDER:
            (headers / name).write_text('', encoding='utf-8')
        (headers / 'sdrplay_api.h').write_text(
            '#define SDRPLAY_API_VERSION (float)(3.15)\n'
            'typedef enum\n{\n    ok = 0,\n} errT;\n'
            'typedef struct\n{\n    int n;\n} devT;\n', encoding='utf-8')
        output = tmp_path / 'out.py'
        gen.main(['--headers', str(headers), '--output', str(output)])
        namespace: dict = {}
        exec(compile(output.read_text(encoding='utf-8'), str(output), 'exec'), namespace)
        assert namespace['API_VERSION'] == 3.15
        assert namespace['errT'].ok == 0

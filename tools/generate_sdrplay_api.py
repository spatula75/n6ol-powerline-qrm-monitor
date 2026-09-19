"""Generate `lib/buzz/sdrplay_api.py` from SDRplay's own C headers.

The SDRplay API is a C library, and no published Python binding suits this project.
What is published is either not on PyPI at all, or is a SWIG build needing a compiler,
so this project talks to `sdrplay_api.dll` with `ctypes` instead.  That needs the structs
declared in Python, laid out exactly as the C compiler laid them out, and one wrong
field is memory corruption rather than an exception.

So they are generated rather than transcribed.  `sdrplay_api_DevParamsT` embeds the
RSPduo and RSPdx parameter structs by value, which means every struct has to be right
even to support only an RSP1B, and that is more hand-copying than anybody should do
accurately.

The headers live in `vendor/`, copied unmodified from an API install and redistributed
under their own license.  See the NOTICE beside them.  Keeping them here is what lets
this run on a machine with no receiver and no API installed, so the generated file can
be checked in CI rather than only where the hardware happens to be.

Run it after installing a new API version:

    python tools/generate_sdrplay_api.py

It writes the module and prints what changed.  `--check` exits non-zero instead of
writing, which is what the staleness test uses.
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_HEADERS = REPO / 'vendor' / 'sdrplay-api-3.15' / 'inc'
DEFAULT_OUTPUT = REPO / 'lib' / 'buzz' / 'sdrplay_api.py'

# Dependency order, not alphabetical.  ctypes needs a nested struct defined before the
# struct that embeds it, and this is the order the headers include each other in.
HEADER_ORDER = [
    'sdrplay_api_tuner.h',
    'sdrplay_api_control.h',
    'sdrplay_api_rsp1a.h',
    'sdrplay_api_rsp2.h',
    'sdrplay_api_rspDuo.h',
    'sdrplay_api_rspDx.h',
    'sdrplay_api_rx_channel.h',
    'sdrplay_api_dev.h',
    'sdrplay_api_callback.h',
    'sdrplay_api.h',
]

# C scalar to ctypes.  A C enum is an int, so enum-typed fields resolve here too, by
# way of `_enums` below.  HANDLE is a pointer on both platforms: a real handle on
# Windows, and the header itself typedefs it to `void *` under GCC.
SCALARS = {
    'char': 'ctypes.c_char',
    'signed char': 'ctypes.c_byte',
    'unsigned char': 'ctypes.c_ubyte',
    'short': 'ctypes.c_short',
    'unsigned short': 'ctypes.c_ushort',
    'int': 'ctypes.c_int',
    'unsigned int': 'ctypes.c_uint',
    'long': 'ctypes.c_long',
    'unsigned long': 'ctypes.c_ulong',
    'long long': 'ctypes.c_longlong',
    'unsigned long long': 'ctypes.c_ulonglong',
    'float': 'ctypes.c_float',
    'double': 'ctypes.c_double',
    'void': 'None',
    'HANDLE': 'ctypes.c_void_p',
}


@dataclass
class Api:
    """Everything the generator emits, in the order the headers declared it."""

    defines: list[tuple[str, str]] = field(default_factory=list)
    enums: list[tuple[str, list[tuple[str, int]]]] = field(default_factory=list)
    callbacks: list[tuple[str, str, list[str]]] = field(default_factory=list)
    structs: list[tuple[str, str, list[tuple[str, str]]]] = field(default_factory=list)
    functions: list[tuple[str, str, list[str]]] = field(default_factory=list)
    # (kind, name) for every declaration, in the order the headers state them.  The
    # lists above are for lookup; this is what rendering walks, because a callback
    # typedef can sit between two structs and has to be emitted between them.
    order: list[tuple[str, str]] = field(default_factory=list)

    @property
    def enum_names(self) -> set[str]:
        return {name for name, _ in self.enums}

    @property
    def struct_names(self) -> set[str]:
        return {name for name, _, _ in self.structs}

    @property
    def callback_names(self) -> set[str]:
        return {name for name, _, _ in self.callbacks}


def _is_number(value: str) -> bool:
    """Whether a `#define` value is a literal this module can carry into Python."""
    try:
        int(value, 0)
        return True
    except ValueError:
        pass
    try:
        float(value)
        return True
    except ValueError:
        return False


def strip_comments(text: str) -> str:
    """Remove C comments, which otherwise sit in the middle of field declarations."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)



def take_enum(name: str, body: str, api: Api) -> None:
    """Record one `typedef enum { NAME = n, ... } name_t;`.

    Values are explicit in every enum here, so nothing has to track an implicit
    counter.  An enum whose members are not all explicit would raise rather than
    guess.

    A `#if` inside the body raises too.  The generator has no preprocessor and reads
    every branch, so a member that a compiler skips still reaches Python.  Nothing
    downstream can tell that member from a live one.
    """
    if '#' in body:
        raise ValueError(
            f'{name} has a preprocessor directive in its body.  This generator reads '
            'every branch of a #if, so a member that a C compiler skips would still '
            'reach Python as a live value.  Teach take_enum which branch to take, or '
            'preprocess the headers before running this.')
    members = []
    for entry in body.split(','):
        entry = entry.strip()
        if not entry:
            continue
        member, sep, value = entry.partition('=')
        if not sep:
            raise ValueError(
                f'{name}.{member.strip()} has no explicit value.  This generator '
                'does not track implicit enum counters, because every enum in the '
                'SDRplay headers states every value.  Add counter handling if that '
                'has changed.')
        members.append((member.strip(), int(value.strip(), 0)))
    api.enums.append((name, members))
    api.order.append(('enum', name))


def ctype_for(c_type: str, pointer: bool, api: Api) -> str:
    """The ctypes spelling of one field's type."""
    c_type = c_type.removeprefix('const ').strip()
    if pointer:
        if c_type in api.struct_names:
            return f'ctypes.POINTER({c_type})'
        if c_type in api.enum_names:
            # A pointer to a C enum is a pointer to an int, the same way a C enum is an
            # int.  sdrplay_api_SwapRspDuoActiveTuner takes one as an out parameter.
            return 'ctypes.POINTER(ctypes.c_int)'
        if c_type == 'char':
            # A C string rather than a buffer.  sdrplay_api_GetErrorString returns one,
            # and c_char_p hands back bytes where c_void_p would hand back an address.
            return 'ctypes.c_char_p'
        if c_type == 'void':
            return 'ctypes.c_void_p'
        if c_type not in SCALARS:
            raise ValueError(
                f'No ctypes equivalent for a pointer to {c_type!r}.  Add it to SCALARS, '
                'or check whether the type is declared after its first use, which the '
                'order in HEADER_ORDER exists to prevent.')
        return f'ctypes.POINTER({SCALARS[c_type]})'
    if c_type in api.enum_names:
        # A C enum is an int.  The IntEnum generated for it is for reading values in
        # Python, and cannot appear in _fields_, which needs a real ctypes type.
        return 'ctypes.c_int'
    if c_type in api.struct_names:
        return c_type
    if c_type in api.callback_names:
        # A function-pointer typedef, which reaches here when a struct holds one.
        # sdrplay_api_CallbackFnsT is the case: the API takes the callbacks as a
        # struct of three function pointers.
        return c_type
    if c_type in SCALARS:
        return SCALARS[c_type]
    raise ValueError(
        f'No ctypes equivalent for C type {c_type!r}.  Add it to SCALARS.  '
        'Or check whether a struct is declared after its first use.  HEADER_ORDER '
        'exists to prevent that.')


def take_struct(name: str, body: str, kind: str, api: Api) -> None:
    """Record one `typedef struct { ... } name_t;`, or the one union shaped the same.

    `sdrplay_api_EventParamsT` is a union of the three event parameter structs, which
    ctypes spells `Union` and lays out with every member at offset zero.  Nothing else
    about reading it differs, so it shares this.

    Arrays keep their size expression verbatim, because it is a `#define` this module
    also emits, so `char SerNo[SDRPLAY_MAX_SER_NO_LEN]` stays readable rather than
    becoming a bare 64.
    """
    fields = []
    for line in body.split(';'):
        line = ' '.join(line.split())
        if not line:
            continue
        match = re.match(r'^([\w ]+?)\s*(\*?)\s*(\w+)\s*(?:\[\s*(\w+)\s*\])?$', line)
        if not match:
            raise ValueError(
                f'Could not read the field {line!r} of {name}.  The generator '
                'handles scalars, enums, nested structs, one level of pointer, and '
                'one-dimensional arrays.  Anything else needs adding here.')
        c_type, star, member, extent = match.groups()
        spelling = ctype_for(c_type.strip(), bool(star), api)
        if extent:
            spelling = f'{spelling} * {extent}'
        fields.append((member, spelling))
    base = 'ctypes.Union' if kind == 'union' else 'ctypes.Structure'
    api.structs.append((name, base, fields))
    api.order.append(('struct', name))


def signature(arg_text: str, ret_type: str, ret_star: str, api: Api,
               where: str) -> tuple[str, list[str]]:
    """Turn one C parameter list and return type into ctypes spellings."""
    args = []
    for arg in arg_text.split(','):
        arg = ' '.join(arg.split())
        if not arg or arg == 'void':
            continue
        match = re.match(r'^(?:const\s+)?([\w ]+?)\s*(\*{0,2})\s*(\w+)?$', arg)
        if not match:
            raise ValueError(f'Could not read the argument {arg!r} of {where}.')
        c_type, stars, _ = match.groups()
        c_type = c_type.strip()
        if len(stars) == 2:
            args.append(f'ctypes.POINTER({ctype_for(c_type, True, api)})')
        else:
            args.append(ctype_for(c_type, bool(stars), api))
    return ctype_for(ret_type.strip(), bool(ret_star), api), args




DECLARATION = re.compile(
    r'(?P<define>^[ \t]*\#define[ \t]+(?P<define_name>\w+)[ \t]+(?P<define_value>[^\n]+))'
    r'|(?P<enum>typedef\s+enum\s*\{(?P<enum_body>.*?)\}\s*(?P<enum_name>\w+)\s*;)'
    r'|(?P<fnptr>typedef\s+(?P<fn_ret>[\w ]+?)\s*(?P<fn_star>\*?)\s*\(\s*\*\s*'
    r'(?P<fn_name>\w+)\s*\)\s*\((?P<fn_args>.*?)\)\s*;)'
    r'|(?P<struct>typedef\s+(?P<struct_kind>struct|union)\s*\{(?P<struct_body>.*?)\}'
    r'\s*(?P<struct_name>\w+)\s*;)'
    r'|(?P<export>_SDRPLAY_DLL_QUALIFIER\s+(?P<ex_ret>[\w ]+?)\s*(?P<ex_star>\*?)\s*'
    r'(?P<ex_name>sdrplay_api_\w+)\s*\((?P<ex_args>.*?)\)\s*;)',
    re.S | re.M)


def parse(headers: Path) -> Api:
    """Read every header in dependency order and collect what it declares."""
    api = Api()
    # This maps each name to its (value, header).  The headers already define
    # _SDRPLAY_DLL_QUALIFIER twice, in exclusive branches, and `_is_number` drops both
    # because neither is a number.  A numeric define written that way would be emitted
    # twice instead.  Python would keep whichever branch came last, so the check below
    # refuses the pair rather than picking one.
    defined: dict[str, tuple[str, str]] = {}
    for filename in HEADER_ORDER:
        path = headers / filename
        if not path.exists():
            raise FileNotFoundError(
                f'{path} is missing.  The vendored headers are what this generates '
                f'from, and {filename} is one of them.  See vendor/*/NOTICE.md.')
        text = strip_comments(path.read_text(encoding='utf-8', errors='replace'))
        for match in DECLARATION.finditer(text):
            if match.group('define'):
                value = match.group('define_value').strip()
                # A define may carry a C cast, as SDRPLAY_API_VERSION does with
                # `(float)(3.15)`.  Drop the cast and the parentheses around the value,
                # so the constant reaches Python as a number rather than a builtin.
                value = re.sub(r'^\((?:float|double|int|unsigned int)\)\s*', '', value)
                value = value.strip()
                if value.startswith('(') and value.endswith(')'):
                    value = value[1:-1].strip()
                # Only numbers.  The headers also define macros such as
                # _SDRPLAY_DLL_QUALIFIER, which are instructions to a C compiler and
                # mean nothing here, and emitting one would put a NameError in the
                # generated module.
                if _is_number(value):
                    name = match.group('define_name')
                    if name in defined and defined[name][0] != value:
                        previous, where = defined[name]
                        raise ValueError(
                            f'{name} is {previous} in {where} and {value} in '
                            f'{filename}.  This generator reads every branch of a #if, '
                            f'so it cannot tell which value a C compiler would take.  '
                            f'Pick one in the generator, or preprocess the headers '
                            f'before running this.')
                    defined[name] = (value, filename)
                    api.defines.append((name, value))
            elif match.group('enum'):
                take_enum(match.group('enum_name'), match.group('enum_body'), api)
            elif match.group('fnptr'):
                restype, args = signature(
                    match.group('fn_args'), match.group('fn_ret'),
                    match.group('fn_star'), api, match.group('fn_name'))
                api.callbacks.append((match.group('fn_name'), restype, args))
                api.order.append(('callback', match.group('fn_name')))
            elif match.group('struct'):
                take_struct(match.group('struct_name'), match.group('struct_body'),
                            match.group('struct_kind'), api)
            elif match.group('export'):
                restype, args = signature(
                    match.group('ex_args'), match.group('ex_ret'),
                    match.group('ex_star'), api, match.group('ex_name'))
                api.functions.append((match.group('ex_name'), restype, args))
    return api


def _call(prefix: str, parts: list[str], suffix: str, indent: str = '    ') -> list[str]:
    """One call per line where it fits, and one argument per line where it does not.

    An emitted line has to pass the same lint as a written one, and some of these run
    past 200 characters on one line.  sdrplay_api_SwapRspDuoMode takes eight arguments.
    """
    single = f'{prefix}{", ".join(parts)}{suffix}'
    if len(single) <= 96:
        return [single]
    return [prefix.rstrip()] + [f'{indent}{part},' for part in parts] + [f'{indent[:-4]}{suffix.lstrip()}']


def render(api: Api, source: Path) -> str:
    """Build the module text."""
    version = next((v for n, v in api.defines if n == 'SDRPLAY_API_VERSION'), 'unknown')
    out = [
        '"""ctypes declarations for the SDRplay API.  GENERATED FILE - do not edit.',
        '',
        f'tools/generate_sdrplay_api.py wrote this from the SDRplay API version {version}',
        f'headers in {source.parent.name}.  Edit the generator or install a newer API and',
        'regenerate, because the next run discards whatever anybody changes here.',
        '',
        'Nothing in here talks to a device.  This module states the shapes the SDRplay',
        'library expects, so that a device shim can call that library.  One wrong field is',
        'memory corruption rather than an exception, so these declarations are generated',
        'rather than transcribed by hand.',
        '"""',
        'import ctypes',
        'from enum import IntEnum',
        '',
        f"API_VERSION = {version}",
        '',
        '# The headers define these constants.',
    ]
    for name, value in api.defines:
        if name == 'SDRPLAY_API_VERSION':
            continue
        out.append(f'{name} = {value}')

    enums = {name: members for name, members in api.enums}
    callbacks = {name: (r, a) for name, r, a in api.callbacks}
    structs = {name: (base, fields) for name, base, fields in api.structs}

    out += ['', '']
    for kind, name in api.order:
        if kind == 'enum':
            out.append(f'class {name}(IntEnum):')
            for member, value in enums[name]:
                out.append(f'    {member} = {value}')
            out += ['', '']
        elif kind == 'callback':
            restype, args = callbacks[name]
            out += _call(f'{name} = ctypes.CFUNCTYPE(', [restype] + args, ')')
            out += ['', '']
        else:
            base, fields = structs[name]
            out.append(f'class {name}({base}):')
            out.append('    _fields_ = [')
            for member, spelling in fields:
                out.append(f"        ('{member}', {spelling}),")
            out += ['    ]', '', '']

    out += [
        '# Each entry is (symbol, restype, argtypes).  The header declares every call as',
        '# a function-pointer typedef whose name is the symbol plus "_t", so one typedef',
        '# gives both the signature and what to look up in the library.',
        'FUNCTIONS = [',
    ]
    for name, restype, args in api.functions:
        joined = ', '.join(args)
        entry = f"    ('{name}', {restype}, [{joined}]),"
        if len(entry) <= 96:
            out.append(entry)
            continue
        out.append(f"    ('{name}', {restype}, [")
        out += [f'        {arg},' for arg in args]
        out.append('    ]),')
    out.append(']')
    out.append('')
    return '\n'.join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--headers', type=Path, default=DEFAULT_HEADERS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true',
                        help='if the output is stale, exit 1 and write nothing')
    args = parser.parse_args(argv)

    text = render(parse(args.headers), args.headers)
    if args.check:
        current = args.output.read_text(encoding='utf-8') if args.output.exists() else ''
        if current == text:
            print(f'{args.output} is current.')
            return 0
        print(f'{args.output} is out of date with {args.headers}.  Regenerate with\n'
              f'    python tools/generate_sdrplay_api.py')
        return 1
    args.output.write_text(text, encoding='utf-8')
    print(f'Wrote {args.output} ({len(text.splitlines())} lines).')
    return 0


if __name__ == '__main__':
    sys.exit(main())

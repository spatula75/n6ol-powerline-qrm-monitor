"""
Generate `config.example.toml` from the schema.

The sample config is documentation in its own right.  It used to be a third place that
described every setting, after the dataclass comments in `buzz.config` and the setup
program's own field labels.  Three copies of one piece of prose give it three chances to
disagree, and a shared constant in this project has already drifted that way.  So the
schema renders the sample now, and a test fails when the committed file stops matching
what the schema produces.

Some settings go in commented out, because a static file cannot state their real
default.  A home-directory path is one case.  A setting whose default is "work it out"
is the other, such as the recording directory.  `x-example` then supplies something to
edit, which serves a reader better than a blank.
"""

from pathlib import Path
from typing import Any

from buzz.setup.schema import (
    ConfigValues,
    field_names,
    field_schema,
    load_schema,
    section_names,
)

HEADER = """\
# N6OL Powerline QRM Monitor - example configuration
#
# Copy this file to ~/.buzz/config.toml and edit to suit your setup, or let the
# setup program write it for you:
#
#     setup.bat                (Windows)
#     ./setup.sh               (Linux / macOS / BSD)
#
# Every option the monitor understands appears here, with the value shown being the
# default. Options that are commented out have no useful default: either the monitor
# works it out, the feature is off, or the setup program fills them in.
#
# GENERATED FILE - edit lib/buzz/setup/schema.json and regenerate, do not edit here.\
"""

# Wrap comment prose at a width that stays readable beside the values it describes.
_COMMENT_WIDTH = 76


def _wrap(text: str, prefix: str = '# ') -> list[str]:
    """`text` as wrapped comment lines. No word breaks across a line."""
    words = text.split()
    lines: list[str] = []
    current = prefix
    for word in words:
        candidate = f'{current}{word} ' if current != prefix else f'{prefix}{word} '
        if len(candidate.rstrip()) > _COMMENT_WIDTH and current != prefix:
            lines.append(current.rstrip())
            current = f'{prefix}{word} '
        else:
            current = candidate
    if current.strip() != prefix.strip():
        lines.append(current.rstrip())
    return lines


def _format_value(value: Any) -> str:
    """One TOML scalar, with backslashes doubled.  TOML reads one as an escape.

    Nothing calls this with None.  TOML cannot spell "unset", so _field_lines() writes
    a setting that has no value as a comment before it reaches here.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return repr(value)


def _field_lines(spec: dict[str, Any], name: str, value: Any) -> list[str]:
    """One setting: the description, the choices, the notes, then the assignment."""
    lines = _wrap(spec['description'])

    # An enum's options are the most useful thing the sample file can tell a reader, so
    # they go directly under the description instead of into the notes.
    titles = spec.get('x-enum-titles')
    if titles:
        lines.append('#')
        for choice in spec['enum']:
            lines.extend(_wrap(titles[str(choice)], prefix='#   '))

    for note in spec.get('x-notes', []):
        lines.append('#')
        lines.extend(_wrap(note))

    # A setting goes in commented out for one of three reasons.  Its real default comes
    # from the runtime, so a literal would be wrong on every other machine.  It has no
    # value at all.  Or it has one that nobody should copy, which x-sample-commented
    # marks.  The last covers the audio device, whose default names one sound card.  As
    # an active setting, that would put somebody else's hardware in every new config.
    reason = spec.get('x-default-from-runtime') or spec.get('x-sample-commented')
    if reason or value is None:
        lines.append('#')
        lines.extend(_wrap(reason, prefix='# Default: ') if reason
                     else _wrap('Unset by default.'))
        example = spec.get('x-example', value if reason else None)
        lines.append(f'# {name} = {_format_value(example)}' if example is not None
                     else f'# {name} =')
    else:
        if 'x-example' in spec:
            lines.extend(_wrap(f'Example: {spec["x-example"]}'))
        lines.append(f'{name} = {_format_value(value)}')
    return lines


def render(values: ConfigValues | None = None) -> str:
    """The whole of `config.example.toml` as text.

    `values` overrides what each setting gets.  Leave it out and the schema's own
    defaults apply, which is what the committed sample file holds.
    """
    schema = load_schema()
    if values is None:
        from buzz.setup.schema import defaults
        values = defaults(schema)

    lines = [HEADER]
    for section in section_names(schema):
        section_spec = schema['properties'][section]
        lines.append('')
        lines.append('')
        lines.extend(_wrap(section_spec['description'], prefix='# '))
        lines.append(f'[{section}]')
        for index, field in enumerate(field_names(schema, section)):
            if index:
                lines.append('')
            lines.extend(_field_lines(field_schema(schema, section, field),
                                      field, values[section][field]))
    return '\n'.join(lines) + '\n'


def write(path: Path) -> None:
    """Regenerate the sample config at `path`.

    Written with explicit newline='\\n' so a checkout on Windows produces the same
    bytes as one on Linux.  Without it, the staleness test fails on whichever platform
    did not generate the committed copy.
    """
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(render())


if __name__ == '__main__':  # pragma: no cover
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _TARGET = _REPO_ROOT / 'config.example.toml'
    write(_TARGET)
    print(f'Wrote {_TARGET}')

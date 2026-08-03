"""
Loading, validating, and pre-filling from `schema.json`.

Nothing here draws anything.  The wizard's screens, the example-config generator, and
the tests all need the same three operations, and keeping them free of any terminal
means the arithmetic and the merge rules are testable without one.

The schema carries three custom keywords.  JSON Schema ignores unknown keywords, so
the document stays a valid schema while saying things a validator has no opinion on:

  * `x-visible-when` - which field this one depends on, for the wizard to gray it out.
    Visibility and validity are different questions and are kept apart deliberately:
    `if`/`then` states what must be true of a *saved* config, which is what
    `validate()` enforces, while this states what is worth *showing* while editing.
    Deriving one from the other would mean reverse-engineering `if`/`then` blocks to
    guess at intent.
  * `x-notes` - paragraphs too long for a form field, used by `example_toml`.  The
    wizard shows `description`, which is deliberately kept to a line or two, because
    a form field has no room for four paragraphs.
  * `x-default-from-runtime` - this field's default depends on the machine (the home
    directory), so it cannot be written into a static document.  `defaults()` fills
    those from `BuzzConfig` instead, which is where they are already computed.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema

from buzz.config import BuzzConfig

SCHEMA_PATH = Path(__file__).with_name('schema.json')

# One section's worth of settings, as they appear in TOML and in the wizard.
SectionValues = dict[str, Any]
ConfigValues = dict[str, SectionValues]


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    """Read the schema document."""
    with open(path, 'rb') as handle:
        return json.load(handle)


def section_names(schema: dict[str, Any]) -> list[str]:
    """The sections, in the order the document lists them.

    Order matters: it is the order the wizard walks and the order
    `config.example.toml` is written in, so both follow the schema rather than each
    choosing for themselves.
    """
    return list(schema['properties'])


def field_names(schema: dict[str, Any], section: str) -> list[str]:
    """The fields of one section, in document order."""
    return list(schema['properties'][section]['properties'])


def field_schema(schema: dict[str, Any], section: str, field: str) -> dict[str, Any]:
    """One field's own sub-schema."""
    return schema['properties'][section]['properties'][field]


def defaults(schema: dict[str, Any], config: BuzzConfig | None = None) -> ConfigValues:
    """Every setting at its default, as a section -> field -> value mapping.

    `x-default-from-runtime` fields take theirs from `config` (a fresh `BuzzConfig`
    unless one is supplied), because a default derived from the home directory cannot
    be written into a static document and duplicating the derivation here would be a
    second place for it to drift.
    """
    config = config or BuzzConfig()
    values: ConfigValues = {}
    for section in section_names(schema):
        section_values: SectionValues = {}
        for field in field_names(schema, section):
            spec = field_schema(schema, section, field)
            if 'default' in spec:
                section_values[field] = spec['default']
            else:
                section_values[field] = getattr(getattr(config, section), field)
        values[section] = section_values
    return values


def from_config(schema: dict[str, Any], config: BuzzConfig) -> ConfigValues:
    """The wizard's starting values: every setting as `config` currently has it.

    Read straight off the dataclasses rather than re-parsing the TOML, so a config
    file missing a key gets the same default the running program would use for it.
    """
    return {section: {field: getattr(getattr(config, section), field)
                      for field in field_names(schema, section)}
            for section in section_names(schema)}


def validate(schema: dict[str, Any], values: ConfigValues) -> list[str]:
    """Every way `values` fails the schema, as messages naming the setting.

    A list rather than an exception because a form wants to mark up all its bad
    fields at once, not stop at the first.  An empty list means the config is good.
    """
    validator = jsonschema.Draft202012Validator(schema)
    problems = []
    for error in sorted(validator.iter_errors(values), key=lambda e: list(e.absolute_path)):
        where = '.'.join(str(part) for part in error.absolute_path)
        problems.append(f'{where}: {error.message}' if where else error.message)
    return problems


def is_visible(schema: dict[str, Any], section: str, field: str,
               section_values: SectionValues) -> bool:
    """Whether `field` is worth showing, given what else in its section is set.

    A field with no `x-visible-when` is always shown.  One that has it is shown only
    when the field it names holds the stated value - so the whole of [server] stays
    out of the way until uploads are switched on.
    """
    condition = field_schema(schema, section, field).get('x-visible-when')
    if condition is None:
        return True
    return section_values.get(condition['field']) == condition['equals']

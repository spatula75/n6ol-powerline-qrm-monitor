"""
Load `schema.json`, validate a config against it, and fill a form from one.

Nothing here draws.  The wizard's screens, the example-config generator, and the tests
all need these same operations.  Keeping them clear of any terminal makes the merge
rules testable without one.

The schema carries six custom keywords.  JSON Schema ignores a keyword it does not
know, so the document stays valid while it says things a validator has no opinion on.

  * `x-visible-when` names the field this one depends on, so the wizard can gray it
    out.  Visibility and validity are separate questions, and this project keeps them
    apart on purpose.  `if`/`then` says what a *saved* config must satisfy, and
    `validate()` enforces that.  This says what is worth *showing* during an edit.  To
    derive one from the other, a reader would have to reverse-engineer `if`/`then`
    blocks and guess at the intent behind them.
  * `x-notes` holds paragraphs too long for a form field.  Only `example_toml` renders
    them.  The wizard shows `description`, which stays a line or two, because a form
    field has no room for four paragraphs.
  * `x-default-from-runtime` marks a field whose default depends on the machine, such
    as the home directory.  No static document can hold it.  `defaults()` reads those
    from `BuzzConfig`, which already computes them.
  * `x-sample-commented` marks a field that has a real default which nobody should
    copy.  The audio device is the example: its default names one sound card.
  * `x-example` supplies a value to edit for a field the sample config comments out.
  * `x-enum-titles` gives each `enum` choice a label for the wizard and the sample
    config to show.
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

    The order matters.  The wizard walks the sections in it, and `config.example.toml`
    is written in it.  Both follow the schema instead of each choosing for itself.
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

    An `x-default-from-runtime` field takes its default from `config`, which is a fresh
    `BuzzConfig` unless the caller supplies one.  A default derived from the home
    directory cannot go into a static document.  Repeating the derivation here would
    give it a second place to drift.
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
    """The wizard's starting values: every setting as `config` now holds it.

    This reads the dataclasses instead of parsing the TOML again.  A config file that
    omits a key therefore gets the same default the running program uses.
    """
    return {section: {field: getattr(getattr(config, section), field)
                      for field in field_names(schema, section)}
            for section in section_names(schema)}


def validate(schema: dict[str, Any], values: ConfigValues) -> list[str]:
    """Every way `values` fails the schema, as messages that name the setting.

    This returns a list instead of raising, because a form marks up all its bad fields
    at once and does not stop at the first.  An empty list means the config is good.
    """
    validator = jsonschema.Draft202012Validator(schema)
    problems = []
    for error in sorted(validator.iter_errors(values), key=lambda e: list(e.absolute_path)):
        where = '.'.join(str(part) for part in error.absolute_path)
        problems.append(f'{where}: {error.message}' if where else error.message)
    return problems


def is_visible(schema: dict[str, Any], section: str, field: str,
               section_values: SectionValues) -> bool:
    """Whether to show `field`, given what else in its section is set.

    A field with no `x-visible-when` always shows.  A field that has one shows only
    when the field it names holds the stated value.  The whole of [server] therefore
    stays out of the way until you switch uploads on.
    """
    condition = field_schema(schema, section, field).get('x-visible-when')
    if condition is None:
        return True
    return section_values.get(condition['field']) == condition['equals']

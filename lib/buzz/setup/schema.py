"""
Load `schema.json`, validate a config against it, and fill a form from one.

Nothing here draws.  The setup program's screens, the example-config generator, and
the tests all need these same operations.  Keeping them clear of any terminal makes
the merge rules testable without one.

The schema carries nine custom keywords.  JSON Schema ignores a keyword it does not
know, so the document stays valid while it says things a validator has no opinion on.

  * `x-visible-when` names the field this one depends on, so the setup program can
    leave it off a section's menu entirely rather than show it disabled.  Visibility
    and validity are separate questions, and this project keeps them apart on
    purpose.  `if`/`then` says what a *saved* config must satisfy, and `validate()`
    enforces that.  This says what is worth *showing* during an edit.  To derive one
    from the other, a reader would have to reverse-engineer `if`/`then` blocks and
    guess at the intent behind them.

    Gate a field on a master switch only, meaning one whose off position puts the
    other settings out of reach for the whole run.  `server.enabled` qualifies:
    `main.py` builds no `Publisher` when it is off.  `recording.enabled` does not,
    and gated the whole recording section by mistake.  It only seeds the recorder's
    opening state.  The R key and `--enable-recording` both arm a run that started
    disarmed, and every other recording setting governs that run.
  * `x-file-only` marks a setting that the monitor reads and `config.example.toml`
    documents, but that nobody should meet in a menu, such as the decimation.
    `menu_field_names()` leaves it out.
  * `x-drop-when-hidden` marks a field that is left out of the written config file
    while its `x-visible-when` condition is unsatisfied, rather than merely left off
    the menu.  Only `station.audio_rf_conversion_db` carries it, because another
    setting holds the same quantity for a receiver and exactly one of the two is ever
    in use.  Every other hidden field is still written, since an operator who turns
    uploads off for a week expects the host and the key path to be there when they
    turn them back on.  See `file_field_names()`.
  * `x-notes` holds paragraphs too long for a form field.  Only `example_toml` renders
    them.  The setup program shows `description`, which stays a line or two, because a
    form field has no room for four paragraphs.
  * `x-default-from-runtime` marks a field whose default depends on the machine, such
    as the home directory.  No static document can hold it.  `defaults()` reads those
    from `BuzzConfig`, which already computes them.
  * `x-sample-commented` marks a field that has a real default which nobody should
    copy.  The audio device is the example: its default names one sound card.
  * `x-example` supplies a value to edit for a field the sample config comments out.
  * `x-enum-titles` gives each `enum` choice a label for the setup program and the
    sample config to show.
  * `x-widget` names a dialog other than the type-driven default (a text box, a
    switch, or an enum's radio list) for `field_dialogs.open_field_dialog()` to open
    instead.  `audio.input_device_name` uses `device-picker`,
    `station.audio_rf_conversion_db` uses `calibration`, and `station.timezone` uses
    `timezone-picker` - see `screens/device_picker.py`, `screens/calibration.py`, and
    `screens/timezone_picker.py`.  All three dialogs still return the field's new
    value on confirm and `CANCELLED` on cancel, the same contract every other field
    dialog honors, so section_menu.py never has to know which one it opened.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema

from buzz.config import BuzzConfig

SCHEMA_PATH = Path(__file__).with_name('schema.json')

# One section's worth of settings, as they appear in TOML and in the setup program.
SectionValues = dict[str, Any]
ConfigValues = dict[str, SectionValues]


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    """Read the schema document."""
    with open(path, 'rb') as handle:
        return json.load(handle)


def section_names(schema: dict[str, Any]) -> list[str]:
    """The sections, in the order the document lists them.

    The order matters.  The setup program walks the sections in it, and
    `config.example.toml` is written in it.  Both follow the schema instead of each
    choosing for itself.
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
    """The setup program's starting values: every setting as `config` now holds it.

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
               values: ConfigValues) -> bool:
    """Whether to show `field`, given what is set anywhere.

    A field with no `x-visible-when` always shows.  A field that has one shows only
    when the field it names holds the stated value.  The whole of [server] therefore
    stays out of the way until you switch uploads on.

    The condition may name a `section`, and defaults to the field's own.  Most gates
    are local, and one is not: station.audio_rf_conversion_db describes a sound card
    and is overwritten at startup when the source is a receiver, so it has to read
    audio.source to know whether it applies at all.  This is the same shape
    section_is_visible already uses, rather than a second spelling of one idea.

    `equals` may be a list, which means any one of those values.  recording.record_iq
    is the case: IQ comes from a receiver and there is more than one kind, so the gate
    has to name each rather than being written again per receiver.
    """
    condition = field_schema(schema, section, field).get('x-visible-when')
    if condition is None:
        return True
    where = values.get(condition.get('section', section), {})
    return _matches(where.get(condition['field']), condition['equals'])


def section_is_visible(schema: dict[str, Any], section: str,
                       values: ConfigValues) -> bool:
    """Whether to show `section` at all, given what is set elsewhere.

    A section with no `x-visible-when` always shows.  Unlike a field gate, this one
    names the section it reads as well as the field, because the setting that decides
    whether a whole section applies is rarely inside that section.  `[rtlsdr]` stays
    hidden until the audio source is set to `rtlsdr`, and the source cannot live in
    `[rtlsdr]`, since it is what chooses between the receiver and the sound card.
    """
    condition = schema['properties'][section].get('x-visible-when')
    if condition is None:
        return True
    where = values.get(condition['section'], {})
    return _matches(where.get(condition['field']), condition['equals'])


def _matches(actual: Any, expected: Any) -> bool:
    """Whether a setting satisfies what an `x-visible-when` condition asks for.

    A list means any one of its values, so a gate that applies to several receivers
    names them rather than being repeated per receiver.
    """
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def file_field_names(schema: dict[str, Any], section: str,
                     values: ConfigValues) -> list[str]:
    """The fields of `section` that belong in the written config file, in order.

    Every field except one marked `x-drop-when-hidden` whose `x-visible-when`
    condition is unsatisfied.  A hidden field is normally still written, because
    hidden means inapplicable to the choices made so far rather than unwanted: an
    operator who switches uploads off for a week expects the host, the username and
    the key path to still be there afterwards, and the backup is the only other copy.

    The marked field is the exception because a second setting holds the same
    quantity.  A receiver's file that carried a [station] audio_rf_conversion_db
    would show a live-looking figure the monitor ignores, in a section that applies
    to every station, and somebody would edit it and watch nothing happen.

    This is the file's counterpart to menu_field_names, and the two differ on
    purpose.  Being off a menu says a setting does not apply now.  Being out of the
    file says it never applied.
    """
    return [field for field in field_names(schema, section)
            if is_visible(schema, section, field, values)
            or not field_schema(schema, section, field).get('x-drop-when-hidden')]


def menu_field_names(schema: dict[str, Any], section: str,
                     values: ConfigValues) -> list[str]:
    """The fields of `section` the setup program offers, in order.

    Two things take a field out of the menu.  `x-file-only` marks a setting that is
    real and documented in `config.example.toml` but that nobody should meet in a
    menu, such as the decimation.  `x-visible-when` hides one that does not apply to
    the choices already made, such as the sound card device when the source is a
    receiver.

    Both screens go through here rather than filtering for themselves, so a field
    cannot be offered on one and withheld on the other.
    """
    return [field for field in field_names(schema, section)
            if not field_schema(schema, section, field).get('x-file-only')
            and is_visible(schema, section, field, values)]

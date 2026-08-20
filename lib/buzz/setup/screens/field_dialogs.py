"""Modal dialogs for editing one field, chosen by its schema type.

Every dialog dismisses with the field's new value on confirm, or with `CANCELLED`
on Cancel or Escape - a sentinel rather than `None`, because `None` is itself a
legal value for a nullable field (an unset weather coordinate, for instance) and
the caller has to tell "the user cleared this" apart from "the user backed out".

open_field_dialog() is the one entry point section_menu.py calls.  It picks the
right dialog from the field's `x-widget`, if it has one, or its JSON Schema `type`
otherwise, so a caller never has to know the mapping itself.
"""

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, OptionList, RadioButton, RadioSet, Static
from textual.widgets.option_list import Option

from buzz.setup.screens.base import CANCELLED, ScopeModalScreen
from buzz.setup.screens.calibration import OffsetCalibrationDialog
from buzz.setup.screens.device_picker import DevicePickerDialog
from buzz.setup.screens.timezone_picker import TimezonePickerDialog


def _types(spec: dict[str, Any]) -> list[str]:
    """A field's JSON Schema `type` as a list, whether it was written as one or several."""
    declared = spec['type']
    return declared if isinstance(declared, list) else [declared]


def _kind(spec: dict[str, Any]) -> str:
    """Which dialog a field needs: 'boolean', 'calibration', 'device-picker', 'enum', 'number',
    'text', or 'timezone-picker'.

    An explicit `x-widget` always wins over the type-driven default - see
    schema.py's module docstring for the fields that set one.
    """
    widget = spec.get('x-widget')
    if widget is not None:
        return widget
    if 'enum' in spec:
        return 'enum'
    types = _types(spec)
    if 'boolean' in types:
        return 'boolean'
    if 'integer' in types or 'number' in types:
        return 'number'
    return 'text'


def _parse_number(spec: dict[str, Any], raw: str) -> tuple[bool, Any]:
    """Parse `raw` against a number/integer field's type.

    Returns (ok, value).  A blank entry on a nullable field parses to (True, None).
    An unparsable entry returns (False, the offending text) so the dialog can show it.
    """
    types = _types(spec)
    if raw.strip() == '' and 'null' in types:
        return True, None
    try:
        return True, int(raw) if 'integer' in types else float(raw)
    except ValueError:
        return False, raw


class TextFieldDialog(ScopeModalScreen[Any]):
    """Edit a string or number field as free text."""

    DEFAULT_CSS = """
    TextFieldDialog {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #dialog Horizontal {
        height: auto;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #description {
        text-align: center;
        padding-bottom: 1;
    }
    #error {
        color: $error;
    }
    """
    # Input already owns left/right for its text cursor, so that pair only takes
    # effect once a Button has focus - see ConfirmDialog's identical note.
    #
    # Escape is a binding rather than a `key_escape` method on purpose.  Textual's
    # `dispatch_key()` (which calls `key_<name>` methods) never calls
    # `event.stop()` or `event.prevent_default()` on the event it dispatches, so a
    # `key_escape` method that dismisses this screen does not stop the same key
    # press from also being resolved through the BINDINGS chain - and by the time
    # that resolution runs, this screen has already closed, so it resolves against
    # whichever screen is now on top instead.  That let one Escape press close both
    # this dialog and the section screen underneath it in a single keystroke.
    # BINDINGS-based handling does not have this problem: matching a binding here
    # is what stops the chain, the same mechanism that already made left/right
    # behave correctly above.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, spec: dict[str, Any], current: Any) -> None:
        super().__init__()
        self._spec = spec
        self._current = current

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static(self._spec['description'], id='description'),
            Input(value='' if self._current is None else str(self._current), id='value'),
            Static('', id='error'),
            Horizontal(Button('OK', id='ok', variant='primary'), Button('Cancel', id='cancel')),
            id='dialog',
        )

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'ok':
            self._confirm()
        else:
            self.dismiss(CANCELLED)

    def on_input_submitted(self, event) -> None:
        self._confirm()

    def _confirm(self) -> None:
        raw = self.query_one('#value', Input).value
        if _kind(self._spec) == 'number':
            ok, value = _parse_number(self._spec, raw)
            if not ok:
                self.query_one('#error', Static).update(f'"{value}" is not a number.')
                return
            self.dismiss(value)
        else:
            types = _types(self._spec)
            self.dismiss(None if raw == '' and 'null' in types else raw)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)


# The two choices of a boolean dialog, On first so that the pressed row index maps
# onto the value without a lookup: row _ON_INDEX is True.
#
# "On" and "Off" rather than "Enabled"/"Disabled" or "Yes"/"No", because
# section_menu.display_value() already renders a boolean row as On or Off.  The menu
# and the dialog have to call the same two states by the same two names.
_ON_INDEX = 0
_BOOLEAN_LABELS = ('On', 'Off')


class BooleanFieldDialog(ScopeModalScreen[Any]):
    """Edit a boolean field as a pair of labeled radio buttons.

    This was a bare `Switch`, an unlabeled square sliding between two ends, with the
    words "on" and "off" appearing nowhere in the dialog.

    A radio pair names both choices and marks the current one.  The mark and the
    cursor stay separate: an arrow key moves `_selected`, and Space or Enter presses
    a button.  Moving between the choices therefore never obscures what is set.
    """

    DEFAULT_CSS = """
    BooleanFieldDialog {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #dialog Horizontal {
        height: auto;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #description {
        text-align: center;
        padding-bottom: 1;
    }
    #value {
        width: 100%;
        margin-bottom: 1;
    }
    """
    # See TextFieldDialog's identical notes on Escape.  Left and right are not bound
    # here as they are there: RadioSet binds `up,left` and `down,right` itself, and
    # those move between the two choices, which is what an arrow key should do while
    # the choices have focus.  Tab still reaches OK and Cancel, and once a Button
    # holds focus its own bindings take over.
    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, spec: dict[str, Any], current: Any) -> None:
        super().__init__()
        self._spec = spec
        self._current = current

    def compose(self):
        current = bool(self._current)
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static(self._spec['description'], id='description'),
            RadioSet(
                RadioButton(_BOOLEAN_LABELS[0], value=current),
                RadioButton(_BOOLEAN_LABELS[1], value=not current),
                id='value',
            ),
            Horizontal(Button('OK', id='ok', variant='primary'), Button('Cancel', id='cancel')),
            id='dialog',
        )

    def on_mount(self) -> None:
        """Focus the choices, with the cursor on the value already set.

        `RadioSet._on_mount` calls `action_next_button()`, parking its cursor on row
        0 whatever is pressed.  On an Off field the cursor would therefore appear on
        "On" while the mark sits on "Off", which is the ambiguity this dialog
        replaced.  `_selected` is private with no setter, so the assignment is
        guarded against an upstream rename.  test_setup_app.py pins the placement.
        """
        choices = self.query_one('#value', RadioSet)
        if hasattr(choices, '_selected'):
            choices._selected = _ON_INDEX if bool(self._current) else _ON_INDEX + 1
        choices.focus()

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'ok':
            self.dismiss(self.query_one('#value', RadioSet).pressed_index == _ON_INDEX)
        else:
            self.dismiss(CANCELLED)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)


class EnumFieldDialog(ScopeModalScreen[Any]):
    """Choose one of a field's `enum` values from a labeled list.

    Selecting a row confirms it immediately - there is no separate OK button, the
    same way a radio list works in raspi-config.  `x-enum-titles` supplies the
    label.  A choice without one falls back to its raw value.
    """

    DEFAULT_CSS = """
    EnumFieldDialog {
        align: center middle;
    }
    #dialog {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #description {
        text-align: center;
        padding-bottom: 1;
    }
    """
    # See TextFieldDialog's note: Escape is a binding rather than a `key_escape`
    # method so that dismissing this screen actually stops the key press there.
    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, spec: dict[str, Any], current: Any) -> None:
        super().__init__()
        self._spec = spec
        self._current = current

    def compose(self):
        titles = self._spec.get('x-enum-titles', {})
        options = [Option(titles.get(str(choice), str(choice)), id=str(choice))
                  for choice in self._spec['enum']]
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static(self._spec['description'], id='description'),
            OptionList(*options, id='value', classes='scope-options'),
            id='dialog',
        )

    def on_mount(self) -> None:
        option_list = self.query_one('#value', OptionList)
        # An OptionList starts with nothing highlighted at all, not row 0, so without
        # this Enter would silently do nothing until the user pressed an arrow key
        # first.  Highlight the current value if it is one of the choices, else the
        # first choice, rather than leave the list looking unselected either way.
        option_list.highlighted = (self._spec['enum'].index(self._current)
                                   if self._current in self._spec['enum'] else 0)
        option_list.focus()

    def on_option_list_option_selected(self, event) -> None:
        # event.option.id is always a str (Option's id parameter), but an enum's own
        # values need not be - pulse_rate's are 120/100 as integers, not "120"/"100" -
        # so this maps the selected row back to the original enum value by identity
        # of its string form, rather than returning the row's id as the new value.
        for choice in self._spec['enum']:
            if str(choice) == event.option.id:
                self.dismiss(choice)
                return

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)


async def open_field_dialog(screen, spec: dict[str, Any], current: Any) -> Any:
    """Open the right dialog for `spec` and return the new value, or CANCELLED."""
    kind = _kind(spec)
    if kind == 'boolean':
        return await screen.app.push_screen_wait(BooleanFieldDialog(spec, current))
    if kind == 'enum':
        return await screen.app.push_screen_wait(EnumFieldDialog(spec, current))
    if kind == 'device-picker':
        # Only ever set on audio.input_device_name, which is why this reaches into
        # the audio section's own in-progress sample rate directly rather than
        # threading a parameter through every other dialog that would ignore it.
        sample_rate = screen.app.values['audio']['sample_rate']
        return await screen.app.push_screen_wait(DevicePickerDialog(spec, current, sample_rate))
    if kind == 'calibration':
        # Only ever set on station.audio_rf_conversion_db, which needs to know the
        # in-progress audio section to know which device to open - same reasoning
        # as device-picker above.
        return await screen.app.push_screen_wait(
            OffsetCalibrationDialog(spec, current, screen.app.values['audio']))
    if kind == 'timezone-picker':
        return await screen.app.push_screen_wait(TimezonePickerDialog(spec, current))
    return await screen.app.push_screen_wait(TextFieldDialog(spec, current))

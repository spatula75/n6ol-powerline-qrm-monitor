"""Modal dialogs for editing one field, chosen by its schema type.

Every dialog dismisses with the field's new value on confirm, or with `CANCELLED`
on Cancel or Escape - a sentinel rather than `None`, because `None` is itself a
legal value for a nullable field (an unset weather coordinate, for instance) and
the caller has to tell "the user cleared this" apart from "the user backed out".

open_field_dialog() is the one entry point section_menu.py calls.  It picks the
right dialog from the field's JSON Schema `type`, so a caller never has to know
the mapping itself.
"""

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, OptionList, Static, Switch
from textual.widgets.option_list import Option

from buzz.setup.screens.base import ScopeModalScreen

CANCELLED = object()


def _types(spec: dict[str, Any]) -> list[str]:
    """A field's JSON Schema `type` as a list, whether it was written as one or several."""
    declared = spec['type']
    return declared if isinstance(declared, list) else [declared]


def _kind(spec: dict[str, Any]) -> str:
    """Which dialog a field needs: 'boolean', 'enum', 'number', or 'text'."""
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
    # Input already owns left/right for its text cursor, so this only takes effect
    # once a Button has focus - see ConfirmDialog's identical note.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
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

    def key_escape(self) -> None:
        self.dismiss(CANCELLED)


class BooleanFieldDialog(ScopeModalScreen[Any]):
    """Edit a boolean field with a switch."""

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
    """
    # See ConfirmDialog's identical note: Tab already cycles this row, arrows do not.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
    ]

    def __init__(self, spec: dict[str, Any], current: Any) -> None:
        super().__init__()
        self._spec = spec
        self._current = current

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static(self._spec['description'], id='description'),
            Switch(value=bool(self._current), id='value'),
            Horizontal(Button('OK', id='ok', variant='primary'), Button('Cancel', id='cancel')),
            id='dialog',
        )

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'ok':
            self.dismiss(self.query_one('#value', Switch).value)
        else:
            self.dismiss(CANCELLED)

    def key_escape(self) -> None:
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
            OptionList(*options, id='value'),
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

    def key_escape(self) -> None:
        self.dismiss(CANCELLED)


async def open_field_dialog(screen, spec: dict[str, Any], current: Any) -> Any:
    """Open the right dialog for `spec` and return the new value, or CANCELLED."""
    kind = _kind(spec)
    if kind == 'boolean':
        return await screen.app.push_screen_wait(BooleanFieldDialog(spec, current))
    if kind == 'enum':
        return await screen.app.push_screen_wait(EnumFieldDialog(spec, current))
    return await screen.app.push_screen_wait(TextFieldDialog(spec, current))

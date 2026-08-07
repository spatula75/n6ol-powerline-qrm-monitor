"""The submenu for one config section: a row per visible field, each opening an edit dialog."""

from typing import Any

from textual import work
from textual.containers import Vertical
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from buzz.setup.schema import field_names, field_schema, is_visible
from buzz.setup.screens.base import CANCELLED, ScopeScreen, scope_header
from buzz.setup.screens.calibration import CalibrationMeterDialog
from buzz.setup.screens.field_dialogs import open_field_dialog

# Not a field - audio_rf_conversion_db lives in the station section, and it stays
# there (see schema.py's docstring on x-widget).  This is a shortcut to a read-only
# meter, shown only on the audio section, matching the current in-progress device
# and sample rate rather than whatever is already saved.  Same sentinel-id shape as
# main_menu.py's _FINISH_ID, for the same reason: a row on_option_list_option_selected
# needs to recognize as not a real field before it looks one up in the schema.
_CALIBRATE_ID = '__calibrate__'


def display_value(spec: dict[str, Any], value: Any) -> str:
    """How one field's current value reads in its menu row.

    Short by design: this is a row label, not the dialog.  An enum shows its raw
    value here rather than its full `x-enum-titles` label, which can run to a
    sentence and would crowd out every other row's value.
    """
    if value is None:
        return '(unset)'
    if 'enum' not in spec and spec['type'] == 'boolean':
        return 'On' if value else 'Off'
    return str(value)


class SectionMenuScreen(ScopeScreen[None]):
    """One section's fields as a menu.  Entering it marks the section visited."""

    DEFAULT_CSS = """
    SectionMenuScreen {
        align: center middle;
    }
    #body {
        width: 80%;
        max-width: 100;
        height: auto;
    }
    #title {
        padding: 1 2 0 2;
        text-align: center;
        text-style: bold;
    }
    #intro {
        padding: 0 2 1 2;
        text-align: center;
    }
    """
    BINDINGS = [('escape', 'back', 'Back to main menu')]

    def __init__(self, section: str) -> None:
        super().__init__()
        self.section = section

    def compose(self):
        section_spec = self.app.schema['properties'][self.section]
        yield scope_header()
        yield Vertical(
            Static(section_spec['title'], id='title'),
            Static(section_spec['description'], id='intro'),
            OptionList(id='fields', classes='scope-options'),
            id='body',
        )
        yield Footer()

    def on_mount(self) -> None:
        self.app.visited.add(self.section)
        self._refresh_options()

    def _refresh_options(self) -> None:
        schema = self.app.schema
        section_values = self.app.values[self.section]
        options: list[Option | None] = []
        for field in field_names(schema, self.section):
            if not is_visible(schema, self.section, field, section_values):
                continue
            spec = field_schema(schema, self.section, field)
            label = f"{spec['title']}: {display_value(spec, section_values[field])}"
            options.append(Option(label, id=field))
        if self.section == 'audio':
            options.append(None)  # a separator, per OptionList.add_option's own convention
            options.append(Option('Calibration meter...', id=_CALIBRATE_ID))
        option_list = self.query_one('#fields', OptionList)
        option_list.clear_options()
        option_list.add_options(options)
        # See main_menu.py's identical note: clear_options() always drops the
        # highlight, so without this Enter would silently do nothing until the user
        # pressed an arrow key first.  Row 0 is always a field, never the separator.
        if options:
            option_list.highlighted = 0

    @work
    async def on_option_list_option_selected(self, event) -> None:
        # See main_menu.py's identical @work note: open_field_dialog and
        # push_screen_wait both await, which Textual only allows inside a worker.
        field = event.option.id
        if field == _CALIBRATE_ID:
            await self.app.push_screen_wait(
                CalibrationMeterDialog(self.app.values['audio'],
                                       self.app.values['station']['audio_rf_conversion_db']))
            return
        schema = self.app.schema
        spec = field_schema(schema, self.section, field)
        current = self.app.values[self.section][field]
        new_value = await open_field_dialog(self, spec, current)
        if new_value is not CANCELLED:
            self.app.values[self.section][field] = new_value
            self._refresh_options()

    def action_back(self) -> None:
        self.dismiss()

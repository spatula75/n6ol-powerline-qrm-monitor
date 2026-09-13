"""The submenu for one config section: a row per visible field, each opening an edit dialog."""

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

from textual import work
from textual.containers import Vertical
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from buzz.config import RtlSdrConfig
from buzz.setup.schema import ConfigValues, SectionValues, field_schema, menu_field_names
from buzz.setup.screens.base import CANCELLED, ScopeScreen, scope_header
from buzz.setup.screens.calibration import CalibrationMeterDialog, level_offset_for
from buzz.setup.screens.field_dialogs import open_field_dialog
from buzz.setup.screens.gain_calibration import GainCalibrationDialog

# Not a field - audio_rf_conversion_db lives in the station section, and it stays
# there (see schema.py's docstring on x-widget).  This is a shortcut to a read-only
# meter, shown only on the audio section, matching the current in-progress device
# and sample rate rather than whatever is already saved.  Same sentinel-id shape as
# main_menu.py's _FINISH_ID, for the same reason: a row on_option_list_option_selected
# needs to recognize as not a real field before it looks one up in the schema.
_CALIBRATE_ID = '__calibrate__'

# The receiver section lists its steps in order, and this is the second: pick a
# frequency, measure a gain, then read back the gain it chose and the offset derived
# from it.  It sits between two fields rather than below everything because it is a
# step in that procedure, which is what _ActionRow.after exists for.
_SWEEP_ID = '__sweep__'


class _ActionRow(NamedTuple):
    """A menu row that runs something instead of editing a value.

    `after` names the field the row follows, or is None to put it at the end.  A
    position is needed because an action can be a step in a procedure rather than an
    afterthought: calibrating the receiver's gain belongs between choosing a frequency
    and reading back the gain it chose, not below everything.

    `shown_for` decides whether the row appears at all, given every section's current
    values.  A predicate rather than the schema's `x-visible-when` shape, because what
    gates a row is Python in the same way its handler is, and one row is hidden by a
    condition that shape cannot state: not equal to a value.
    """

    id: str
    label: str
    after: str | None
    shown_for: Callable[[ConfigValues], bool] | None = None


def _has_no_front_panel(values: ConfigValues) -> bool:
    """Whether the read-only meter would be of any use to this station.

    It exists for a radio whose own front panel carries RF and AF gain, so that an
    operator can turn those while watching a reading.  A receiver has no such knobs:
    its gain is a config field and its level offset is the only thing to move, which
    is what the offset dialog is for.  Offering a meter that adjusts nothing, under a
    hint telling somebody to adjust two controls they do not have, is worse than
    offering nothing.
    """
    return values.get('audio', {}).get('source') != 'rtlsdr'


# Keyed by section.  Kept here rather than in the schema because the handler for each
# row is Python, and a schema entry naming a dialog it cannot open would be a second
# place to keep in step with this file.
_ACTIONS: dict[str, tuple[_ActionRow, ...]] = {
    'audio': (_ActionRow(_CALIBRATE_ID, 'Calibration meter...', None,
                         shown_for=_has_no_front_panel),),
    'rtlsdr': (_ActionRow(_SWEEP_ID, 'Auto-calibrate gain...', 'frequency_khz'),),
}


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


def row_value(section: str, field: str, spec: dict[str, Any],
              section_values: SectionValues) -> str:
    """The same thing, except that an unset field which something derives shows what
    the monitor will actually use.

    `(unset)` is honest and unhelpful for the receiver's level calibration, because
    the monitor does not run without an offset.  It estimates one from the tuner gain,
    and an operator deciding whether to go and calibrate wants to see the figure they
    would be accepting.  The marker says where it came from, so a borrowed number and
    a measured one never look alike.

    The estimate is read from `RtlSdrConfig.level_offset_db` rather than worked out
    here.  Writing `-gain_db` a second time would be a second place to keep in step
    with the first, and nothing would notice them drifting apart.
    """
    if (section == 'rtlsdr' and field == 'calibrated_offset_db'
            and section_values.get(field) is None):
        known = {f.name for f in dataclasses.fields(RtlSdrConfig)}
        settings = RtlSdrConfig(**{k: v for k, v in section_values.items() if k in known})
        return f'{settings.level_offset_db:g} (estimated)'
    return display_value(spec, section_values[field])


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
        actions = tuple(action for action in _ACTIONS.get(self.section, ())
                        if action.shown_for is None or action.shown_for(self.app.values))
        for field in menu_field_names(schema, self.section, self.app.values):
            spec = field_schema(schema, self.section, field)
            label = f"{spec['title']}: {row_value(self.section, field, spec, section_values)}"
            options.append(Option(label, id=field))
            for action in actions:
                if action.after == field:
                    options.append(Option(action.label, id=action.id))
        for action in actions:
            if action.after is None:
                # A separator earns its place only at the end, where it marks the
                # break between the settings and what can be done with them.  One in
                # the middle of a procedure would cut the procedure in half.
                options.append(None)  # a separator, per OptionList.add_option's own convention
                options.append(Option(action.label, id=action.id))
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
        if field == _SWEEP_ID:
            await self._calibrate_gain()
            return
        if field == _CALIBRATE_ID:
            await self.app.push_screen_wait(
                CalibrationMeterDialog(
                    self.app.values['audio'],
                    level_offset_for(self.app.values['audio'],
                                     self.app.values['station'],
                                     self.app.values.get('rtlsdr')),
                    self.app.values.get('rtlsdr')))
            return
        schema = self.app.schema
        spec = field_schema(schema, self.section, field)
        current = self.app.values[self.section][field]
        new_value = await open_field_dialog(self, spec, current)
        if new_value is not CANCELLED:
            self.app.values[self.section][field] = new_value
            if self.section == 'rtlsdr' and field == 'gain_db':
                self._carry_the_calibration_to(current, new_value)
            self._refresh_options()

    def _carry_the_calibration_to(self, old_gain: float | None,
                                  new_gain: float) -> None:
        """Move the level offset with the tuner gain, so the pair stays consistent.

        The offset is what converts audio level to dBm, and the tuner gain is most of
        that conversion.  Change the gain and leave the offset, and every level the
        station reports is wrong by the difference, which is exactly what
        calibrated_at_gain_db exists to notice at startup.  Fixing it here means the
        operator never has to see that warning.

        An uncalibrated station needs nothing done: calibrated_offset_db is unset,
        and RtlSdrConfig.level_offset_db already derives the estimate from whatever
        gain_db currently says, so the menu row re-renders against the new one.

        A calibrated station keeps its measurement.  The offset is the negative of the
        gain plus a residual for the rest of the chain, and only the gain term moved,
        so shifting by the difference carries the residual across.  That is an
        approximation rather than a fresh measurement, because the nominal step labels
        carry their own error, but it is right to within that where doing nothing is
        wrong by the whole change.  See docs-notebook/sdr-gain-calibration.md for why
        the true gain per step could not be measured.
        """
        values = self.app.values['rtlsdr']
        if values.get('calibrated_offset_db') is None or old_gain is None:
            return
        values['calibrated_offset_db'] = round(
            values['calibrated_offset_db'] - (new_gain - old_gain), 2)
        # The calibration now describes the new gain, so the startup check compares
        # against that.  Leaving the old figure would warn about a difference this
        # has already corrected.
        values['calibrated_at_gain_db'] = new_gain

    async def _calibrate_gain(self) -> None:
        """Run the sweep and take its answer as the gain, and the offset with it.

        The offset starts at the negative of the gain because that is the whole of
        what is known: the true gain per step cannot be measured without a signal
        strong enough to reference, and on a narrowband antenna there may be none.
        See docs-notebook/sdr-gain-calibration.md.  An operator with a second receiver
        tunes it afterwards from the offset's own dialog.
        """
        chosen = await self.app.push_screen_wait(
            GainCalibrationDialog(self.app.values['rtlsdr']))
        if chosen is CANCELLED or chosen is None:
            return
        self.app.values['rtlsdr']['gain_db'] = chosen
        self.app.values['rtlsdr']['calibrated_offset_db'] = -chosen
        self.app.values['rtlsdr']['calibrated_at_gain_db'] = chosen
        self._refresh_options()

    def action_back(self) -> None:
        self.dismiss()

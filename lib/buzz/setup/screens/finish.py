"""The closing screen: show what changed, back out if something looks wrong, or save.

With nothing to save it offers to leave the program instead, because Back would be
the only way out of a screen somebody opened in order to finish.

Saving always backs up an existing config first.  If the backup cannot be written,
the config is not touched - complaining about a failed backup and then overwriting
the file anyway would destroy the one copy a failed backup was supposed to protect.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Static

from buzz.setup.schema import ConfigValues, field_names, is_visible, section_names
from buzz.setup.screens.base import ScopeScreen, scope_header

_BACKUP_TIMESTAMP = '%Y%m%d-%H%M%S'


def changed_fields(schema: dict[str, Any], original: ConfigValues,
                   current: ConfigValues) -> list[tuple[str, str, Any, Any]]:
    """Every (section, field, old, new) where `current` differs from `original`.

    In schema order, not dict order, so the summary reads the same way the setup
    program's own menus do.  The result does not depend on which fields happen to be
    visible right now, because it reports what will actually be written rather than
    what the last-opened submenu showed.
    """
    changes = []
    for section in section_names(schema):
        for field in field_names(schema, section):
            old = original[section][field]
            new = current[section][field]
            if old != new:
                changes.append((section, field, old, new))
    return changes


def backup_path(config_path: Path, now: datetime | None = None) -> Path:
    """Where the pre-save copy of `config_path` goes: alongside it, timestamped."""
    now = now or datetime.now()
    return config_path.with_name(f'config-{now:{_BACKUP_TIMESTAMP}}.toml.bak')


def toml_ready(values: ConfigValues,
               schema: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """`values` ready to write: no unset fields, and nothing that does not apply.

    TOML cannot spell "unset", so a None is dropped rather than written.

    A setting hidden by `x-visible-when` is dropped too, when a schema is given.  Two
    sections carry a level offset and exactly one is ever used, so writing both
    left a receiver's config file holding a live-looking [station] figure that the
    monitor ignores.  A value nobody can act on is worse than a missing one,
    because it invites somebody to edit it and watch nothing happen.

    The schema is optional so that a caller with only values still gets the None
    filtering, which is the older of the two jobs.
    """
    if schema is None:
        return {section: {k: v for k, v in fields.items() if v is not None}
                for section, fields in values.items()}
    return {section: {name: field_values[name]
                      for name in field_names(schema, section)
                      if name in field_values
                      and field_values[name] is not None
                      and is_visible(schema, section, name, values)}
            for section, field_values in values.items()
            if section in section_names(schema)}


class FinishScreen(ScopeScreen[None]):
    """Show the pending changes, then back out or save.  With none, back out or exit."""

    DEFAULT_CSS = """
    FinishScreen {
        align: center middle;
    }
    #body {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 100%;
    }
    #intro {
        padding: 1 2;
        text-align: center;
    }
    #changes {
        padding: 0 2;
        height: auto;
        max-height: 15;
    }
    #error {
        color: $error;
        padding: 0 2;
    }
    #actions {
        height: auto;
        padding: 1 2;
    }
    """
    # See ConfirmDialog's identical note: a Horizontal's children already take
    # Tab/Shift+Tab, but not the left/right arrows a button row naturally invites.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
        ('escape', 'back', 'Back to main menu'),
    ]

    def compose(self):
        self._changes = changed_fields(self.app.schema, self.app.original_values, self.app.values)
        yield scope_header()
        if self._changes:
            yield Vertical(
                Static('The following will be saved:', id='intro'),
                VerticalScroll(*(Static(self._change_line(c)) for c in self._changes), id='changes'),
                Static('', id='error'),
                Horizontal(Button('Save', id='save', variant='primary'), Button('Back', id='back'),
                          id='actions'),
                id='body',
            )
        else:
            # Exit as well as Back, because Back alone is a dead end: somebody who
            # reached this screen to leave the program is told there is nothing to
            # save and then sent to the menu they came from.  Exit is safe here in a
            # way it is not in the branch above, since there is nothing to discard.
            yield Vertical(
                Static(self._nothing_to_save(), id='intro'),
                Horizontal(Button('Exit', id='exit', variant='primary'),
                           Button('Back', id='back'),
                           id='actions'),
                id='body',
            )
        yield Footer()

    def _nothing_to_save(self) -> str:
        """Why there is nothing to write, and what that leaves behind.

        The two cases differ in what the monitor will read afterwards, so they say so.
        An operator who ran setup on a machine with no config at all should not have
        to guess whether one now exists.
        """
        if self.app.had_existing_config:
            return f'No changes to save.  {self.app.config_path} is unchanged.'
        return ('No changes to save, so no config file was written.  The monitor uses '
                f'its built-in defaults until {self.app.config_path} exists.')

    def on_mount(self) -> None:
        # Neither button focuses itself, and nothing else on this screen is
        # focusable - without this, arrow keys and Enter do nothing until Tab is
        # pressed first, and no row shows which one Enter would confirm.  Back is
        # the safe default where there is something to save, the same reasoning as
        # ConfirmDialog focusing Cancel: Enter should not save by accident.  With
        # nothing to save there is nothing to do by accident, so Exit takes the
        # focus and Enter finishes the job the operator came here for.
        self.query_one('#back' if self._changes else '#exit', Button).focus()

    def _change_line(self, change: tuple[str, str, Any, Any]) -> str:
        section, field, old, new = change
        title = self.app.schema['properties'][section]['properties'][field]['title']
        return f'{section}.{field} ({title}): {old!r} -> {new!r}'

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'back':
            self.dismiss()
        elif event.button.id == 'exit':
            # No confirmation, unlike the main menu's Escape.  That one asks because
            # it cannot tell whether somebody meant to leave; this button says Exit
            # and there is nothing unsaved for a misfire to cost.
            self.app.exit(message=self._nothing_to_save())
        elif event.button.id == 'save':
            self._save()

    def action_back(self) -> None:
        self.dismiss()

    def _save(self) -> None:
        config_path: Path = self.app.config_path
        if config_path.exists():
            destination = backup_path(config_path)
            try:
                shutil.copy2(config_path, destination)
            except OSError as exc:
                self.query_one('#error', Static).update(
                    f'Could not back up {config_path} to {destination}: {exc}.  '
                    'The config was not changed.  Free up space or fix permissions, '
                    'then try Save again.')
                return

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'wb') as handle:
            tomli_w.dump(toml_ready(self.app.values, self.app.schema), handle)
        self.app.exit(message=f'Config saved to {config_path}')

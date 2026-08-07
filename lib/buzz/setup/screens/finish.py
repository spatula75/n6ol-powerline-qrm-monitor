"""The closing screen: show what changed, back out if something looks wrong, or save.

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

from buzz.setup.schema import ConfigValues, field_names, section_names
from buzz.setup.screens.base import ScopeScreen, scope_header

_BACKUP_TIMESTAMP = '%Y%m%d-%H%M%S'


def changed_fields(schema: dict[str, Any], original: ConfigValues,
                   current: ConfigValues) -> list[tuple[str, str, Any, Any]]:
    """Every (section, field, old, new) where `current` differs from `original`.

    In schema order, not dict order, so the summary reads the same way the setup
    program's own menus do.  Unaffected by which fields happen to be visible right
    now: this reports what will actually be written, not what the last-opened
    submenu showed.
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


def toml_ready(values: ConfigValues) -> dict[str, dict[str, Any]]:
    """`values` with every unset (None) field dropped.  TOML cannot spell "unset"."""
    return {section: {k: v for k, v in fields.items() if v is not None}
            for section, fields in values.items()}


class FinishScreen(ScopeScreen[None]):
    """Show the pending changes, then back out or save."""

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
            yield Vertical(
                Static('No changes to save.', id='intro'),
                Horizontal(Button('Back', id='back'), id='actions'),
                id='body',
            )
        yield Footer()

    def on_mount(self) -> None:
        # Neither button focuses itself, and nothing else on this screen is
        # focusable - without this, arrow keys and Enter do nothing until Tab is
        # pressed first, and no row shows which one Enter would confirm.  Back is
        # the safe default, the same reasoning as ConfirmDialog focusing Cancel:
        # Enter should not save by accident.
        self.query_one('#back', Button).focus()

    def _change_line(self, change: tuple[str, str, Any, Any]) -> str:
        section, field, old, new = change
        title = self.app.schema['properties'][section]['properties'][field]['title']
        return f'{section}.{field} ({title}): {old!r} -> {new!r}'

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'back':
            self.dismiss()
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
            tomli_w.dump(toml_ready(self.app.values), handle)
        self.app.exit(message=f'Config saved to {config_path}')

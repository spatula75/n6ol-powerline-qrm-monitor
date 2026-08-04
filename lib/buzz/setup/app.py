"""The setup program's Textual application.

Owns the state every screen shares: the schema, the config values being edited,
a copy of what they were when the program opened (so `finish.py` can show a diff),
and which sections have been visited this run (so `main_menu.py` can mark them).
No screen reaches into another screen for this - they all read and write it here,
on `self.app`, the same "one shared place, not scattered" reasoning `buzz.config`
follows for the dataclasses this mirrors.

Usage:
    PYTHONPATH=lib python -m buzz.setup
"""

from pathlib import Path

from textual.app import App

from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.setup.schema import ConfigValues, defaults, from_config, load_schema
from buzz.setup.screens.main_menu import MainMenuScreen
from buzz.setup.theme import SCOPE_THEME


def _copy_values(values: ConfigValues) -> ConfigValues:
    """A copy of `values` independent enough for before/after comparison.

    One dict per section is enough: every leaf is a JSON Schema primitive
    (str, int, float, bool, or None), so nothing nested needs a deep copy.
    """
    return {section: dict(fields) for section, fields in values.items()}


class SetupApp(App[None]):
    """The terminal setup program for ~/.buzz/config.toml."""

    TITLE = 'N6OL Powerline QRM Monitor - Setup'
    # The command palette (Ctrl+P) has nothing to offer here - a handful of
    # screens with no commands registered - and its Footer hint only crowds out
    # the bindings that do something.  This turns it off entirely, rather than
    # merely hiding the hint.
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        super().__init__()
        self.register_theme(SCOPE_THEME)
        self.theme = SCOPE_THEME.name
        self.config_path = config_path
        self.schema = load_schema()
        self.had_existing_config = config_path.exists()
        config = BuzzConfig.from_toml(config_path) if self.had_existing_config else BuzzConfig()
        self.values: ConfigValues = (from_config(self.schema, config) if self.had_existing_config
                                     else defaults(self.schema, config))
        self.original_values: ConfigValues = _copy_values(self.values)
        self.visited: set[str] = set()

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())

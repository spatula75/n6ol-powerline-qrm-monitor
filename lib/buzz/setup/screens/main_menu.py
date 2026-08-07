"""The setup program's main menu: one row per config section, plus Finish."""

from textual import work
from textual.containers import Vertical
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from buzz.setup.schema import section_names
from buzz.setup.screens.base import ScopeScreen, scope_header
from buzz.setup.screens.confirm import ConfirmDialog
from buzz.setup.screens.finish import FinishScreen, changed_fields
from buzz.setup.screens.section_menu import SectionMenuScreen

_FINISH_ID = '__finish__'

_NEW_CONFIG_HELP = (
    'Because you are creating a new configuration, be sure to step through each '
    "section, making all your selections before choosing 'Finish'."
)
_EXISTING_CONFIG_HELP = (
    'Pick a section to review or change a setting, or choose Finish to save.'
)

# The brackets are always present.  Only the character between them changes.  There
# are two reasons for this: first, Rich treats square brackets as markup - '[x]
# Station' renders as plain 'Station' with the '[x]' silently eaten, because Rich
# parses it as a (meaningless, so discarded) markup tag, where '[ ]' happens to
# survive because a bare space is not valid tag syntax.  Toggling only the inside
# avoids relying on that accident.  Second, a mark that changes width when a section
# is visited shifts every title after it out of alignment - '*' is exactly as wide
# as the space it replaces, so the column of titles never moves.
_VISITED_MARK = '*'
_UNVISITED_MARK = ' '


class MainMenuScreen(ScopeScreen[None]):
    """The screen the setup program opens on.  Sections show a checkmark once visited this run."""

    DEFAULT_CSS = """
    MainMenuScreen {
        align: center middle;
    }
    #body {
        width: 80%;
        max-width: 100;
        height: auto;
    }
    #intro {
        padding: 1 2;
        text-align: center;
    }
    """
    # 'q'/'Q' alongside Escape: the conventional "quit" key in terminal programs,
    # and OptionList does not bind either letter for anything of its own, so there
    # is nothing for this to shadow regardless of which row has focus.
    BINDINGS = [
        ('escape', 'try_exit', 'Exit'),
        ('q', 'try_exit', 'Exit'),
        ('Q', 'try_exit', 'Exit'),
    ]

    def compose(self):
        yield scope_header()
        yield Vertical(
            Static(self._intro_text(), id='intro'),
            OptionList(id='sections', classes='scope-options'),
            id='body',
        )
        yield Footer()

    def _intro_text(self) -> str:
        specific = _NEW_CONFIG_HELP if not self.app.had_existing_config else _EXISTING_CONFIG_HELP
        return (f'Use the arrow keys and Enter to choose.  Escape or Q exits, asking '
                f'first to confirm.\n{specific}')

    def on_mount(self) -> None:
        self._refresh_options()

    def _refresh_options(self) -> None:
        schema = self.app.schema
        options: list[Option | None] = []
        for section in section_names(schema):
            mark = _VISITED_MARK if section in self.app.visited else _UNVISITED_MARK
            title = schema['properties'][section]['title']
            options.append(Option(f'[{mark}] {title}', id=section))
        options.append(None)  # a separator, per OptionList.add_option's own convention
        options.append(Option('Finish', id=_FINISH_ID))
        option_list = self.query_one('#sections', OptionList)
        option_list.clear_options()
        option_list.add_options(options)
        # clear_options() always drops the highlight, and OptionList otherwise starts
        # with nothing highlighted at all - so without this, the first screen a user
        # sees offers no visible cue that Enter does anything until they press an
        # arrow key first.  Row 0 is always a section, never the separator.
        option_list.highlighted = 0

    @work
    async def on_option_list_option_selected(self, event) -> None:
        # push_screen_wait suspends this handler until the pushed screen dismisses,
        # which Textual only allows inside a worker - see the @work decorator above.
        if event.option.id == _FINISH_ID:
            await self.app.push_screen_wait(FinishScreen())
        else:
            await self.app.push_screen_wait(SectionMenuScreen(event.option.id))
        self._refresh_options()

    @work
    async def action_try_exit(self) -> None:
        """Escape on the main menu: always confirm before exiting.

        There is nowhere else Escape could sensibly back out to - this is the top of
        the screen stack - so it means "leave the program" instead, the same way Back
        does on every other screen.  Asking every time, not only when something is
        unsaved, means Escape can never exit on its own.  Escape on the confirmation
        itself cancels, the same as everywhere else it appears in this program.
        """
        changes = changed_fields(self.app.schema, self.app.original_values, self.app.values)
        if changes:
            question = f'You have {len(changes)} unsaved change(s).  Discard them and exit?'
            confirm_label, cancel_label = 'Discard and exit', 'Keep editing'
        else:
            question = 'Exit the setup program?'
            confirm_label, cancel_label = 'Exit', 'Cancel'
        discard = await self.app.push_screen_wait(
            ConfirmDialog(question, confirm_label=confirm_label, cancel_label=cancel_label))
        if discard:
            self.app.exit()

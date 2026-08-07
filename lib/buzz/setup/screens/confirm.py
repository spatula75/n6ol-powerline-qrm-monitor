"""A yes/no confirmation dialog, reused wherever the setup program needs one."""

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from buzz.setup.screens.base import ScopeModalScreen


class ConfirmDialog(ScopeModalScreen[bool]):
    """Ask a yes/no question.  Dismisses with True (confirmed) or False (cancelled/Escape)."""

    DEFAULT_CSS = """
    ConfirmDialog {
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
    #question {
        text-align: center;
        padding-bottom: 1;
    }
    """
    # A Horizontal's children take Tab/Shift+Tab already (Screen.BINDINGS binds
    # those to app.focus_next/previous), but not the left/right arrows a button
    # row like this one naturally invites.  Binding them to the same actions
    # covers both without touching how Tab already works.
    #
    # Escape is a binding rather than a `key_escape` method for the same reason
    # as every dialog in field_dialogs.py: `dispatch_key()` never stops the key
    # press it dispatches, so a `key_escape` method that dismisses this screen
    # does not stop the same press from also resolving through the BINDINGS
    # chain, by which point this screen has closed and MainMenuScreen - which
    # binds Escape to open this very dialog - is what the press resolves against
    # instead.  On this dialog specifically that means a second ConfirmDialog
    # opening on top of the first from one Escape press, not just a skipped
    # screen.  A matched binding stops the chain where a bare method does not.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, question: str, confirm_label: str, cancel_label: str) -> None:
        super().__init__()
        self._question = question
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self):
        yield Vertical(
            Static(self._question, id='question'),
            Horizontal(
                Button(self._confirm_label, id='confirm', variant='error'),
                Button(self._cancel_label, id='cancel', variant='primary'),
            ),
            id='dialog',
        )

    def on_mount(self) -> None:
        self.query_one('#cancel', Button).focus()

    def on_button_pressed(self, event) -> None:
        self.dismiss(event.button.id == 'confirm')

    def action_cancel(self) -> None:
        self.dismiss(False)

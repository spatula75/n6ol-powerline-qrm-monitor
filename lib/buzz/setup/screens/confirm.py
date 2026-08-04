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
    # row like this one naturally invites. Binding them to the same actions
    # covers both without touching how Tab already works.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
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

    def key_escape(self) -> None:
        self.dismiss(False)

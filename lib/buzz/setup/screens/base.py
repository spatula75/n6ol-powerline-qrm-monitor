"""Shared base screens that fix the color Textual's ansi mode otherwise resets.

`Screen.DEFAULT_CSS` carries its own `&:ansi { color: ansi_default; background:
ansi_default; }` rule, which fires whenever the active theme sets `ansi=True` - this
is deliberate on Textual's part, and is exactly what its own built-in `ansi-dark` and
`ansi-light` themes rely on: general body text defers to whatever the terminal's own
ambient colors already are, rather than to a color the theme names.

This project's theme wants a specific pair instead (bright cyan on black), not the
terminal's ambient default. Overriding that takes a selector at equal specificity -
a bare type selector such as `MainMenuScreen { color: ...; }` cannot beat
`Screen:ansi`, because CSS resolves specificity before source order, and a lone type
selector is less specific than a type-plus-pseudo-class one. Only another
`:ansi`-qualified selector can win, confirmed against a live app instance: `Screen`'s
own `:ansi` rule left a child Static rendering white-on-default until the screen
class itself carried a matching `:ansi` override, at which point the widget's actual
`rich_style` (not its own unset `styles.color`, which never reflects inheritance)
showed the intended color pair.

ScopeScreen and ScopeModalScreen do this once, so no individual screen repeats it.
Every screen and dialog in this package inherits from one of the two instead of from
`Screen`/`ModalScreen` directly.
"""

from typing import TypeVar

from textual.screen import ModalScreen, Screen

ScreenResultType = TypeVar('ScreenResultType')


class ScopeScreen(Screen[ScreenResultType]):
    DEFAULT_CSS = """
    ScopeScreen:ansi {
        color: ansi_bright_cyan;
        background: ansi_black;
    }
    """


class ScopeModalScreen(ModalScreen[ScreenResultType]):
    DEFAULT_CSS = """
    ScopeModalScreen:ansi {
        color: ansi_bright_cyan;
        background: ansi_black;
    }
    """

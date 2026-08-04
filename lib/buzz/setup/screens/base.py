"""Shared base screens that fix colors Textual's own defaults otherwise get wrong.

`Screen.DEFAULT_CSS` carries its own `&:ansi { color: ansi_default; background:
ansi_default; }` rule, which fires whenever the active theme sets `ansi=True` - this
is deliberate on Textual's part, and is exactly what its own built-in `ansi-dark` and
`ansi-light` themes rely on: general body text defers to whatever the terminal's own
ambient colors already are, rather than to a color the theme names.

This project's theme wants a specific pair instead (bright cyan on black), not the
terminal's ambient default.  Overriding that takes a selector at equal specificity -
a bare type selector such as `MainMenuScreen { color: ...; }` cannot beat
`Screen:ansi`, because CSS resolves specificity before source order, and a lone type
selector is less specific than a type-plus-pseudo-class one.  Only another
`:ansi`-qualified selector can win, confirmed against a live app instance: `Screen`'s
own `:ansi` rule left a child Static rendering white-on-default until the screen
class itself carried a matching `:ansi` override, at which point the widget's actual
`rich_style` (not its own unset `styles.color`, which never reflects inheritance)
showed the intended color pair.

`OptionList.DEFAULT_CSS` has a second, unrelated problem: it defines the same
selector, `OptionList > .option-list--option-highlighted`, twice at equal
specificity, once early (using the `block-cursor-blurred-*` variables) and once
later (using `color: $foreground` and `background: $block-cursor-blurred-
background`).  The later one always wins when the widget is not focused, which
makes `block-cursor-blurred-foreground` unreachable dead configuration - setting
it has no effect no matter what it is set to.  Confirmed by simulating an
`AppBlur` event against a live app: the highlighted row's resolved style came
back as `color(14) on color(14)`, this theme's bright cyan on itself, invisible,
which is exactly the vanishing selection this fix exists for.

Repeating the identical bare-type selector once more, later still, was tried
first and did not work: the live style after a simulated blur came back as
`color(14) on default`, matching neither the built-in rule nor the intended
override, which means source order alone was not deciding the winner here the
way it did for `Screen:ansi` above.  A selector using a CSS class instead of the
built-in rule's plain type selector was tried next - every `OptionList` this
package creates carries the `scope-options` class (see main_menu.py,
section_menu.py, and field_dialogs.py's EnumFieldDialog) - and by hand-counted
specificity should have settled it outright, but it did not move the result
either, for a mundane reason rather than a deeper one: the CSS text itself still
read the old bare-type selector, an editing mistake caught only by rereading the
file rather than by the specificity theory being wrong.  `!important` on both
declarations below is what actually closed the question, correctly this time -
reconfirmed post-blur as `color(0) on color(14)`, the intended pair, matching
the focused state exactly rather than only nominally trying to.

ScopeScreen and ScopeModalScreen do both of these once, so no individual screen
repeats them.  Every screen and dialog in this package inherits from one of the two
instead of from `Screen`/`ModalScreen` directly.
"""

from typing import TypeVar

from textual.screen import ModalScreen, Screen
from textual.widgets import Header

ScreenResultType = TypeVar('ScreenResultType')

# Shared between ScopeScreen and ScopeModalScreen: EnumFieldDialog is a
# ScopeModalScreen and has its own OptionList, so it needs the same fix.  Targets
# the `scope-options` class every OptionList in this package carries (not the
# bare `OptionList` type Textual's own rule uses), and `!important`, because
# even the class selector alone was not reliably winning - see the docstring.
#
# The separator rule rides along here for the same reason, not a new one: it is
# the divider main_menu.py draws between the sections and Finish, and reads
# `$border` rather than repeating `ansi_magenta` a second place, so the border
# and the separator can never drift into two different accent colors.
_OPTION_LIST_HIGHLIGHT_FIX = """
    .scope-options > .option-list--option-highlighted {
        color: $block-cursor-foreground !important;
        background: $block-cursor-background !important;
    }
    .scope-options > .option-list--separator {
        color: $border !important;
    }
    """


class ScopeScreen(Screen[ScreenResultType]):
    DEFAULT_CSS = f"""
    ScopeScreen:ansi {{
        color: ansi_bright_cyan;
        background: ansi_black;
    }}
    {_OPTION_LIST_HIGHLIGHT_FIX}
    HeaderIcon:hover {{
        background: $panel !important;
    }}
    """


class ScopeModalScreen(ModalScreen[ScreenResultType]):
    DEFAULT_CSS = f"""
    ScopeModalScreen:ansi {{
        color: ansi_bright_cyan;
        background: ansi_black;
    }}
    {_OPTION_LIST_HIGHLIGHT_FIX}
    """


def scope_header() -> Header:
    """A `Header` with its command-palette icon blanked out.

    The icon (the circle docked at the top left) exists only to open the command
    palette on click, which `SetupApp.ENABLE_COMMAND_PALETTE = False` already
    disables - but a circle that still lights up on mouse hover and does nothing
    on click reads as broken, not as inert.

    Two separate fixes were needed, not one.  `icon=''` blanks the glyph itself.
    Hiding the whole widget with CSS (`display: none`) was tried first and
    rejected: it also dropped the icon's docked width from the header's layout,
    which shifted the centered title over to fill the gap.  `Header.__init__`
    data-binds `icon` down to the child `HeaderIcon` widget directly (see
    Textual's own `_header.py`), so `icon=''` renders nothing while still
    reserving the width the title-centering math depends on.

    That alone still left the icon highlighting on mouse hover, because
    `HeaderIcon`'s own `:hover` rule is not conditioned on the widget's
    `disabled` state at all - being disabled only stops the click action, not
    the CSS hover match.  `ScopeScreen`'s `HeaderIcon:hover` override, matching
    the header's own ambient `$panel` background, makes hovering visually
    inert too, without touching layout a second time - though `!important` is
    needed to make it actually take effect.  Textual's own `HeaderIcon:hover`
    rule and this override share the identical selector, so without it the
    live style still changed on hover (confirmed via `rich_style` on a live
    widget) regardless of which stylesheet was merged in later, the same
    surprise the OptionList highlight fix above ran into.
    """
    return Header(icon='')

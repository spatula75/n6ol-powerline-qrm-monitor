"""The setup program's color theme: black screen, bright cyan text, matching the scope.

Built from the ANSI palette (`ansi=True` below), not RGB hex.  An arbitrary hex like
the scope's own phosphor color has no fixed rendering outside a truecolor terminal -
an SSH session, a serial console, or any terminal running in 16- or 256-color mode
approximates it however that terminal's color-matching happens to work, and the
result can drift far from the intended cyan.  `ansi_bright_cyan` asks for a palette
slot instead, which every ANSI terminal already defines and renders consistently.

The plain (non-bright) `ansi_cyan` was tried first.  It reads correctly against a
component that references it by name - `#dialog { border: round $primary; }`, for
instance - but general screen text stayed the terminal's own default color rather
than cyan at all, because `Screen.DEFAULT_CSS` carries its own `&:ansi { color:
ansi_default; }` rule that Textual applies whenever a theme sets `ansi=True`,
independently of what that theme's own `foreground` names.  See `screens/base.py`
for the fix.  Once that was in place and the color was confirmed actually changing,
bright cyan read clearly on a real terminal, so this uses it throughout rather than
mixing bright and non-bright cyans in the same palette.

The selected row in a menu had a second, separate problem: asking for literal
`ansi_black` text read as dark grey rather than black on a real terminal, even
though the resolved style was confirmed (via `rich_style` on a live widget) to be
exactly `color(0)`, ANSI palette index 0.  The likely cause: `block-cursor-text-
style` was never set here, so it fell back to Textual's own ansi-mode default of
`bold`, and a great many terminals implement bold text by substituting the
*bright* variant of whatever color is set (a historical SGR-1 convention) rather
than a heavier font weight.  Bold black over such a terminal renders as ANSI
color 8, bright black, which is exactly the mid-grey reported.  Set to `none`
below so the color painted is the one actually named, not its bold-shifted
neighbor.

`Button`'s own focused state sidesteps the whole question with `text-style:
reverse` (real terminal reverse video) instead of a named foreground, and its
`rich_style` confirmed the effect.  The identical technique was tried for the
highlighted row below first and made things worse, not better - the highlight
stopped rendering at all on the same real terminal that showed the grey-black
correctly - which is why this uses the `bold` diagnosis instead: it explains the
specific grey observed, where `reverse` was a different mechanism entirely and
its failure explains nothing about why black read as grey in the first place.

One more variable needs a note, for the opposite of the usual reason: `border`
was never set at all until this was noticed, so every OptionList's outer box was
rendering with Textual's own ansi-mode default of `ansi_magenta`, confirmed via
`option_list.styles.border` on a live widget returning `Color(128, 0, 128,
ansi=5)` - literal magenta.  That was headed for a fix until it turned out to be
liked as it stood, breaking up what would otherwise be cyan on black throughout,
so `border` is set here explicitly to the same magenta Textual's default already
supplied, which pins the color as a deliberate choice rather than leaving it as
an unstyled leftover that could change out from under this theme if that default
ever did.  `border-blurred` was a real bug, not a color choice: it was black,
which is invisible against the black background the moment the terminal loses
OS focus, reading as the whole selection having vanished.  Set to the same
magenta as the focused border, for the same reason the block-cursor pair no
longer has a separate blurred version in `screens/base.py` - the box should stay
exactly as visible as it was, not go dark the instant focus moves elsewhere.
"""

from textual.theme import Theme

SCOPE_THEME = Theme(
    name='n6ol-scope',
    primary='ansi_bright_cyan',
    foreground='ansi_bright_cyan',
    background='ansi_black',
    surface='ansi_black',
    panel='ansi_black',
    accent='ansi_bright_cyan',
    error='ansi_red',
    dark=True,
    ansi=True,
    # `ansi=True` themes bypass Textual's usual RGB-derived variables, so these
    # have to be supplied explicitly - the built-in widgets' own CSS references
    # them by name (a box's border, Input's cursor and selection, OptionList's
    # highlighted row).  The block-cursor pair is black-on-bright-cyan, and
    # text-style is pinned to `none` rather than left at Textual's ansi-mode
    # default of `bold` - see the module docstring for why `bold` was very likely
    # the actual cause of black reading as grey.  There is no separate blurred
    # block-cursor pair: `screens/base.py` makes the highlighted row use this
    # same pair regardless of focus, so setting one here would do nothing - see
    # that file for why the built-in blurred variables cannot be reached at all.
    variables={
        'ansi-background': 'ansi_black',
        'ansi-foreground': 'ansi_white',
        'border': 'ansi_magenta',
        'border-blurred': 'ansi_magenta',
        'block-cursor-foreground': 'ansi_black',
        'block-cursor-background': 'ansi_bright_cyan',
        'block-cursor-text-style': 'none',
        'input-cursor-background': 'ansi_black',
        'input-cursor-foreground': 'ansi_white',
        'input-cursor-text-style': 'none',
        'input-selection-background': 'ansi_bright_cyan',
        'input-selection-foreground': 'ansi_black',
        'screen-selection-background': 'ansi_bright_cyan',
        'screen-selection-foreground': 'ansi_black',
    },
)

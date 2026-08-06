"""Modal timezone picker for station.timezone: filters the IANA database as you type.

Nobody outside this codebase can recite "America/Chicago" from a blank prompt, so this
opens a search box instead of the plain text box `station.timezone` would otherwise get.
`zoneinfo.available_timezones()` reads the same tzdata `zoneinfo.ZoneInfo` resolves at
runtime, so a name this dialog offers can never be one the monitor itself later rejects.
"""

import asyncio
import importlib.resources
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

from textual import work
from textual.containers import Vertical
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from buzz.setup.screens.base import CANCELLED, ScopeModalScreen

# "align: center middle" splits whatever space is left after #dialog's own height by
# floor division, not evenly - 1 row of slack becomes 0 rows above #dialog and 1 below,
# not half a row each.  A margin of 1 therefore left no room at all above the dialog,
# and the row that cost - part of #description, wrapped onto a second line - was the
# first one clipped on a terminal with even slightly less headroom than the one this
# was checked against.  3 is enough to floor to at least 1 on both edges.
_VERTICAL_SLACK = 3


def _utc_offset_label(zone: str) -> str:
    """A zone's current offset from UTC, as `+HH:MM` or `-HH:MM`.

    A bare zone name does not say whether it currently sits in daylight or
    standard time, and the two are six characters apart in the alphabet at
    best - `America/Denver` and `America/Phoenix` differ by a whole hour right
    now and give no hint of that in the name alone.  This reads the offset off
    the system clock rather than off a fixed table, so it already accounts for
    whichever of the two the zone is in today.
    """
    offset = datetime.now(ZoneInfo(zone)).utcoffset()
    total_minutes = int(offset.total_seconds()) // 60
    sign = '+' if total_minutes >= 0 else '-'
    hours, minutes = divmod(abs(total_minutes), 60)
    return f'{sign}{hours:02d}:{minutes:02d}'


def _canonical_zone_names() -> frozenset[str]:
    """Every zone name tzdata itself treats as a real zone, not a deprecated alias.

    `zoneinfo.available_timezones()` does not make this distinction - `UTC` and
    `Zulu` both come back alongside `Etc/UTC` even though all three share the
    identical rules, because the tz database keeps `UTC` and `Zulu` only for
    backward compatibility with software that expects the old name.  There is no
    `zoneinfo` API that says which is which, but the `tzdata` package - already a
    pinned dependency, both for `zoneinfo` itself on platforms with no system tz
    database and for this lookup - ships its own compiled source, `tzdata.zi`, in
    the exact format the real tz database is built from: a `Z name ...` line
    defines a zone, an `L target alias` line defines one of the names this filters
    out.  Reading it once here does not depend on which of the two `zoneinfo`
    itself ends up resolving names against at runtime - the zone-vs-alias
    structure this reads has held for decades of tzdata releases and is not
    something a single release would change.
    """
    zi_text = importlib.resources.files('tzdata.zoneinfo').joinpath('tzdata.zi').read_text(encoding='utf-8')
    return frozenset(line.split()[1] for line in zi_text.splitlines() if line.startswith('Z '))


def _offset_index(zones: list[str]) -> dict[str, str]:
    """Every zone's `_utc_offset_label()`, computed once and reused.

    `ZoneInfo(name)` reads and parses a tzdata file the first time each name is
    seen, and the roughly 400 names together cost close to a second of real
    file I/O - fine for a one-time cost paid off the UI thread (see
    `_load_offsets()`), not fine repeated on every keystroke `_show_matches()`
    makes while narrowing the list.
    """
    return {zone: _utc_offset_label(zone) for zone in zones}


class TimezonePickerDialog(ScopeModalScreen[Any]):
    """Filter the IANA timezone names by substring and choose one from the results.

    Selecting a row confirms it immediately, the same radio-list behavior
    EnumFieldDialog uses - there is no separate OK button.  Typing narrows the list
    by region or city, matched anywhere in the name, so "chicago" and "america/ch"
    both find America/Chicago.  Each row shows its current UTC offset alongside the
    name, since the name alone does not say whether a zone is in daylight time today.
    A blank search shows every real zone the database has, not a truncated sample,
    so scrolling through it by hand rather than typing a name works too - "real"
    meaning `_canonical_zone_names()` has already dropped backward-compatibility
    aliases such as `UTC` and `Zulu`, both of which are the same zone as `Etc/UTC`
    under a different name.
    """

    DEFAULT_CSS = """
    TimezonePickerDialog {
        align: center middle;
    }
    #dialog {
        width: 70;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #description {
        text-align: center;
        padding-bottom: 1;
    }
    #value {
        height: auto;
    }
    #status {
        padding-top: 1;
    }
    """
    # Input does not bind Down itself, so the key press bubbles up to here rather
    # than moving the text cursor - this is what lets Down step from the search box
    # into the result list in one press.  See TextFieldDialog's identical note on
    # why Escape is a binding rather than a `key_escape` method: matching a binding
    # is what stops the chain, so this dialog's Escape does not also resolve
    # against the section screen underneath it.
    BINDINGS = [
        ('down', 'focus_list', 'Results'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, spec: dict[str, Any], current: Any) -> None:
        super().__init__()
        self._spec = spec
        self._current = current
        self._zones = sorted(available_timezones() & _canonical_zone_names())
        self._offsets: dict[str, str] = {}

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static(self._spec['description'], id='description'),
            Input(placeholder='Type a region or city, such as Chicago', id='filter'),
            OptionList(id='value', classes='scope-options'),
            Static('', id='status'),
            id='dialog',
        )

    def on_mount(self) -> None:
        self.query_one('#filter', Input).focus()
        self._load_offsets()

    @work(exclusive=True)
    async def _load_offsets(self) -> None:
        status = self.query_one('#status', Static)
        status.update('Reading timezone offsets...')
        # Off the UI thread for the same reason DevicePickerDialog's probe is - see
        # its module docstring and "push, don't poll" in CLAUDE.md.  Nothing in this
        # dialog is usable yet without the offsets, so there is no partial state to
        # show meanwhile beyond the status line above.
        self._offsets = await asyncio.to_thread(_offset_index, self._zones)
        # #value is still empty at this point - _show_matches() has not run yet - so
        # sizing it now cannot cause the flash _fit_results_to_the_terminal()'s own
        # docstring describes: there is nothing tall to shrink down from.
        self._fit_results_to_the_terminal()
        self._show_matches(self.query_one('#filter', Input).value)

    def _fit_results_to_the_terminal(self) -> None:
        """Bound #value to whatever room is actually left below the chrome around
        it, measured for real rather than guessed.

        The chrome widgets it measures - #title, #description, #filter, #status -
        are deliberately left at `height: auto` in DEFAULT_CSS rather than each
        given an explicit row count.  An earlier version did the opposite,
        reasoning that a fixed height would make this method's total predictable.
        It instead silently broke the two wrapping ones: `height: 2` on
        #description covers the whole padding box, and `padding-bottom: 1` already
        claims one of those two rows, so only one was ever left for content - the
        wrapped second line ("as America/Los_Angeles.") had nowhere to render and
        simply never appeared, in any terminal, at any size, which is why
        widening or heightening the window never helped.  #status's `height: 1`
        plus its own `padding-top: 1` was worse - zero rows of content, so no
        status message ever rendered at all.  `outer_size` still reported the
        declared 2 and 1 correctly in both cases; only the actual rendered lines,
        not the widget's own reported box size, would have shown either bug -
        see `render_line()` in a debugging session, not `outer_size` alone, next
        time a widget's content looks short of what its box implies.  Auto-sizing
        avoids the whole class of bug: whatever height a widget's content and
        padding actually need is the height it reports, with nothing to fall out
        of sync.

        Two still-earlier versions guessed at #value's own share instead of
        measuring it at all: a `max-height: 60vh` drifted from the real
        remaining space as the terminal got shorter, because the chrome above it
        is a fixed row count, not a percentage of the screen; a hand counted
        `_CHROME_ROWS` constant that replaced it matched the chrome's *intended*
        size but not what it actually rendered at, for the padding reason above.
        Reading `outer_size` after the fact removes the guessing entirely -
        whatever the chrome really needed is what gets subtracted, on this
        platform, in this terminal, this run.

        This does not flash the way an earlier `call_after_refresh()` version
        did, which measured only after #value had already been populated with
        every match and painted once at its full, unclamped height.  Here,
        `_load_offsets()` calls this method before `_show_matches()` ever adds a
        row - #value is still empty when its height gets bounded, so there is
        nothing tall to shrink down from.  The `await asyncio.to_thread(...)`
        two lines above this call already gives Textual the time it needs to lay
        out that still-empty chrome for real, the same wait this method's
        measurement depends on - not a second, separate deferral back to the
        framework.
        """
        chrome = sum(self.query_one(selector).outer_size.height
                    for selector in ('#title', '#description', '#filter', '#status'))
        border_and_padding = 4  # #dialog's round border (2 rows) plus `padding: 1 2` (2 rows)
        available = self.screen.size.height - chrome - border_and_padding - _VERTICAL_SLACK
        # A floor below 1 would give the list no visible rows at all, but flooring any
        # higher than that can itself push the dialog past the screen on a truly
        # tiny terminal - the fixed chrome already has nowhere left to shrink into.
        self.query_one('#value', OptionList).styles.max_height = max(available, 1)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == 'filter':
            self._show_matches(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        option_list = self.query_one('#value', OptionList)
        if option_list.option_count > 0:
            highlighted = option_list.highlighted or 0
            self._confirm(option_list.get_option_at_index(highlighted).id)

    def on_option_list_option_selected(self, event) -> None:
        self._confirm(event.option.id)

    def _show_matches(self, text: str) -> None:
        # Nothing to show yet while _load_offsets() is still reading tzdata - the
        # status line already says so, and on_input_changed can fire before that
        # worker finishes if someone starts typing right away.
        if not self._offsets:
            return
        needle = text.strip().lower()
        matches = [zone for zone in self._zones if needle in zone.lower()]
        option_list = self.query_one('#value', OptionList)
        option_list.clear_options()
        status = self.query_one('#status', Static)
        if not matches:
            status.update('No timezone matches that text.')
            return
        status.update('1 match.' if len(matches) == 1 else f'{len(matches)} matches.')
        option_list.add_options([Option(f'{zone} ({self._offsets[zone]})', id=zone)
                                 for zone in matches])
        # Highlight the configured zone if the current search still shows it, so
        # opening the dialog on an already-set timezone starts with it selected -
        # the same reasoning as EnumFieldDialog's on_mount.
        option_list.highlighted = matches.index(self._current) if self._current in matches else 0
        # Setting `.highlighted` above already asks OptionList to scroll the row
        # into view, but on this method's very first call - right after
        # `_load_offsets()` populates a previously empty list - that scroll runs
        # before OptionList has been through a layout pass with any rows in it,
        # so it silently computes against a stale, empty viewport and leaves the
        # highlighted row off-screen with nothing scrolled to reveal it.  A
        # timezone set to something well past the top of the alphabet (station's
        # own default, "America/Los_Angeles") only became visible again once the
        # first arrow key press forced a second, now-correct scroll - which read
        # as the row "jumping" there rather than opening already selected.
        # Repeating the same scroll after the next real layout removes the wait.
        self.call_after_refresh(option_list.scroll_to_highlight, top=True)

    def action_focus_list(self) -> None:
        option_list = self.query_one('#value', OptionList)
        if option_list.option_count > 0:
            option_list.focus()

    def _confirm(self, zone: str) -> None:
        self.dismiss(zone)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)

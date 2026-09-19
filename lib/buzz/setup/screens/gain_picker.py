"""Modal picker for [rtlsdr] gain_db: offers the steps the tuner actually has.

A tuner accepts a fixed set of gains and snaps anything else to the nearest, so a
typed 41.0 becomes 40.2 and nothing tells the operator.  A field that silently changes
what somebody entered should not be a text box, which is the rule in CLAUDE.md.

The list lives on the hardware rather than in the schema, so this has to open the
receiver.  That is the one thing separating it from EnumFieldDialog, and it is why
the work happens in a worker with the list filling afterwards, the same shape
DevicePickerDialog uses for the sound card.

A receiver that will not open is not a dead end.  The dialog says why and hands back
UNAVAILABLE, and open_field_dialog then falls through to the plain number box, because
somebody has to be able to set a gain before the device is working.
"""

import asyncio
from typing import Any

from textual import work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from buzz.config import receiver_settings_from
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen

# Dismissal value meaning "the receiver could not be reached", as distinct from the
# operator backing out.  The caller reopens the field as a text box for this and does
# nothing at all for CANCELLED, so the two cannot share a sentinel.
UNAVAILABLE = object()


def supported_gains(source: str, values: SectionValues) -> list[float]:
    """The gains this receiver offers, leaving it as it was found.

    The import sits inside the function for the reason buzz.main.open_live_source
    gives: a station should never load a driver for hardware it does not own.

    This asks for the list rather than opening a configured device, because the two
    are different requests.  Configuring writes a sample rate, a tuning, an AGC
    setting and a gain, and it logs that the operator's gain was snapped to a step
    while the operator is part way through choosing that gain.  The steps a receiver
    offers do not depend on any of it.

    An RTL-SDR has to be opened to answer, and an RSP does not: its ladder comes from
    the published gain tables rather than from the unit.  Whichever it is, the device
    is released rather than held, because holding it would stop the monitor and the
    sweep from opening it.  Whatever this raises carries wording the device wrote for
    whoever is standing at the radio.
    """
    from buzz.sdr_device import supported_gains as gains_for

    settings = receiver_settings_from(source, values)
    if settings is None:
        return []
    return sorted(gains_for(source, settings))


class GainPickerDialog(ScopeModalScreen[Any]):
    """Choose a tuner gain from the steps the receiver reports.

    Selecting a row confirms it immediately, the same radio-list behavior
    EnumFieldDialog and DevicePickerDialog both use.  There is no OK button.
    """

    DEFAULT_CSS = """
    GainPickerDialog {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        max-height: 90%;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #status {
        text-align: center;
        padding: 0 0 1 0;
    }
    """
    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, spec: dict[str, Any], current: Any, source: str = '',
                 receiver_values: SectionValues | None = None) -> None:
        super().__init__()
        self._spec = spec
        self._current = current
        self._source = source
        self._receiver_values = receiver_values or {}
        self._gains: list[float] = []
        # Set when the receiver could not be reached, which changes what Escape means:
        # backing out of a list is a cancel, and leaving a dialog that has no list is
        # a request for the text box instead.
        self._unreachable = False

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static('Reading the gains this receiver offers...', id='status'),
            OptionList(id='value', classes='scope-options'),
            id='dialog',
        )

    def on_mount(self) -> None:
        self._load_gains()

    @work
    async def _load_gains(self) -> None:
        try:
            self._gains = await asyncio.to_thread(
                supported_gains, self._source, self._receiver_values)
        except Exception as exc:
            self._give_up(f'Could not read the gains: {exc}')
            return
        if not self._gains:
            self._give_up('The receiver reported no gain settings.')
            return
        self._show_gains()

    def _show_gains(self) -> None:
        # Escape can dismiss this while the read above is still running.  Textual
        # cancels the worker at its next await rather than mid-statement, so a read
        # that finishes in that instant still resumes and arrives after the widgets
        # have gone.  Same guard, and same reason, as DevicePickerDialog._show_devices.
        try:
            status = self.query_one('#status', Static)
            option_list = self.query_one('#value', OptionList)
        except NoMatches:
            return
        status.update(f'{len(self._gains)} steps.  Enter chooses one.')
        option_list.clear_options()
        option_list.add_options(
            [Option(self._label(gain), id=str(index))
             for index, gain in enumerate(self._gains)])
        option_list.highlighted = self._nearest_index()
        option_list.focus()

    def _label(self, gain: float) -> str:
        """One row.  The current value is marked, since the list is long enough that
        finding where you already are otherwise means reading all 29."""
        marker = ' (current)' if gain == self._nearest_gain() else ''
        return f'{gain:.1f} dB{marker}'

    def _nearest_gain(self) -> float:
        """The offered step the stored value would snap to.

        Marked rather than the stored value itself, because the stored value may be
        one the tuner does not have, and the row shown as current has to be the one
        the hardware would actually use.
        """
        return self._gains[self._nearest_index()]

    def _nearest_index(self) -> int:
        if self._current is None:
            return 0
        return min(range(len(self._gains)),
                   key=lambda i: abs(self._gains[i] - float(self._current)))

    def _give_up(self, message: str) -> None:
        """Report why there is no list, and wait rather than vanishing.

        Dismissing here would be tidier and would throw the message away: the dialog
        would close in the same instant it explained itself, and the operator would
        see a text box appear for no stated reason.  RtlSdrDevice words these for
        whoever is standing at the radio, naming the driver to install or what else
        holds the receiver, so it has to stay on screen long enough to read.
        """
        self._unreachable = True
        try:
            self.query_one('#status', Static).update(
                f'{message}\n\nPress Escape to type a gain instead.')
        except NoMatches:
            pass

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(self._gains[int(event.option.id)])

    def action_cancel(self) -> None:
        self.dismiss(UNAVAILABLE if self._unreachable else CANCELLED)

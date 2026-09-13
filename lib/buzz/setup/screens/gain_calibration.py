"""Choose the receiver's tuner gain by measuring the band.

A modal that runs buzz.gain_sweep against the receiver described by the in-progress
[rtlsdr] section, shows where it has got to, and offers the answer.  The measurement
and the arithmetic live in gain_sweep; this file is the screen around them.

The dialog warns rather than refuses when the sweep cannot reach an answer.  A quiet
antenna has no gain that satisfies both bounds, so refusing to close would trap
exactly the operator who most needs to set a gain by hand.  What it can always do is
say which of the two bounds failed and what to do about it.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from textual import work
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from buzz.config import RtlSdrConfig
from buzz.gain_sweep import GainSweep, SweepResult
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen

if TYPE_CHECKING:
    from buzz.sdr import RtlSdrSource

# IQ samples per callback during a sweep.
#
# The pipeline uses 16384 because it has a deadline to beat; a sweep has none, and the
# transfer pool it must drain after every gain change is buf_num blocks deep whatever
# their size.  At 2048 the pool is 120 ms rather than 960, which turns 30 seconds of
# discarding across 29 gain steps into under 4.  See docs-notebook/rtl-sdr-hardware.md.
_SWEEP_BLOCK_SAMPLES = 2048


def open_sweep(rtlsdr_values: SectionValues) -> tuple['RtlSdrSource', GainSweep]:
    """Open the receiver and build a sweep over it.

    The imports sit inside the function for the reason buzz.main.open_live_source
    gives: a station using a sound card should never load pyrtlsdr, which resolves a
    symbol as it imports and so fails at import rather than at first call.

    Whatever this raises carries a message written for whoever is standing at the
    radio, because open_device rewords libusb's own wording.  The dialog shows it
    rather than letting a traceback through.
    """
    from buzz.sdr import RtlSdrSource, open_device

    settings = RtlSdrConfig(**rtlsdr_values)
    source = RtlSdrSource(
        open_device(settings.device_index),
        frequency_hz=settings.frequency_hz, gain_db=settings.gain_db,
        iq_sample_rate=settings.iq_sample_rate,
        tuning_offset_hz=settings.tuning_offset_hz,
        block_samples=_SWEEP_BLOCK_SAMPLES)
    source.start()
    return source, GainSweep(source, settings.arc_headroom_db)


class GainCalibrationDialog(ScopeModalScreen[Any]):
    """Runs the sweep and offers what it found.

    Dismisses with the chosen gain, or with CANCELLED if the operator backs out or
    the sweep reached no answer.
    """

    DEFAULT_CSS = """
    GainCalibrationDialog {
        align: center middle;
    }
    #dialog {
        width: auto;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }
    #title {
        text-style: bold;
        padding-bottom: 1;
    }
    #status {
        width: 64;
        padding-bottom: 1;
    }
    #outcome {
        width: 64;
        padding-bottom: 1;
    }
    #buttons {
        width: auto;
        height: auto;
    }
    #buttons Button {
        margin-right: 2;
    }
    .hidden {
        display: none;
    }
    """
    # Left and Right move between the buttons, the same way ConfirmDialog does it and
    # for the same reason: a Horizontal's children take Tab already, because
    # Screen.BINDINGS binds it to app.focus_next, but not the arrows a row of buttons
    # invites somebody to reach for.  Without these, the only ways out of a finished
    # sweep were the mouse and Escape.
    BINDINGS = [
        ('left', 'app.focus_previous', 'Previous'),
        ('right', 'app.focus_next', 'Next'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, rtlsdr_values: SectionValues) -> None:
        super().__init__()
        self._rtlsdr_values = rtlsdr_values
        self._sweep: GainSweep | None = None
        self._result: SweepResult | None = None
        self._offers_gain = False

    def compose(self):
        yield Vertical(
            Static('Calibrate receiver gain', id='title'),
            Static('Measuring the band at every gain the tuner offers, five times '
                   'over.  This takes a little over a minute.  Leave the antenna '
                   'connected and the receiver tuned where it will run.', id='status'),
            Static('', id='outcome'),
            Horizontal(
                # Hidden until there is a gain to accept.  Offering it during the
                # sweep would invite somebody to take an answer that does not exist
                # yet.
                Button('Use this gain', id='accept', classes='hidden',
                       variant='primary'),
                Button('Cancel', id='cancel'),
                id='buttons',
            ),
            id='dialog',
        )

    def on_mount(self) -> None:
        self._run_sweep()

    @work
    async def _run_sweep(self) -> None:
        try:
            source, sweep = await asyncio.to_thread(open_sweep, self._rtlsdr_values)
        except Exception as exc:
            self._set('#outcome', f'Could not open the receiver: {exc}')
            self._finish()
            return
        self._sweep = sweep
        try:
            result = await asyncio.to_thread(sweep.run, self._on_progress)
        except Exception as exc:
            self._set('#outcome', f'The sweep failed: {exc}')
            self._finish()
            return
        finally:
            # Off the event loop and shielded from cancellation, for the reason
            # calibration.close_without_blocking_the_ui gives: closing a receiver
            # joins two threads with five second timeouts, and doing that on the
            # event loop freezes the whole program on the way out of the dialog.
            await asyncio.shield(asyncio.to_thread(source.close))
        self._result = result
        self._show_result(result)

    def _on_progress(self, step: int, total: int, gain_db: float) -> None:
        """Called from the sweep's own thread, so the update is posted rather than made.

        Textual is not thread-safe, and this runs on the worker asyncio.to_thread put
        the sweep on.  call_from_thread is the supported way back onto the event loop.
        """
        self.app.call_from_thread(
            self._set, '#status',
            f'Step {step + 1} of {total}: measuring {gain_db:.1f} dB...')

    def _show_result(self, result: SweepResult) -> None:
        if result.chosen_db is None:
            self._set('#status', 'The sweep finished without an answer.')
            self._set('#outcome', result.reason)
            self._finish()
            return
        self._set('#status', f'Measured gain: {result.chosen_db:.1f} dB')
        self._set('#outcome',
                  f'{result.reason}  The antenna supplies '
                  f'{result.antenna_share * 100:.0f}% of the noise floor at this '
                  f'gain, so the reported floor reads about '
                  f'{result.floor_error_db:.1f} dB high.')
        self._finish(accept=True)

    def _finish(self, accept: bool = False) -> None:
        """Offer whatever the operator can do now.

        A measured gain gets two buttons rather than one.  Taking the figure and
        declining it are both reasonable: somebody may have run the sweep to see what
        it says, or may disagree with it, and a dialog whose only exit applies the
        change makes refusing it feel like an error.  Escape has always cancelled, but
        an operator should not have to know that.

        A sweep that reached no answer has nothing to accept, so its one button says
        Close.  The labels change and the ids do not, because Textual refuses to
        change a widget's id once set and raises inside the worker when asked, which
        takes the whole app down rather than the dialog.
        """
        self._offers_gain = accept
        try:
            accept_button = self.query_one('#accept', Button)
            cancel_button = self.query_one('#cancel', Button)
        except NoMatches:
            return
        accept_button.set_class(not accept, 'hidden')
        cancel_button.label = 'Cancel' if accept else 'Close'
        # Focus whichever button is the useful one, so the keyboard works without a
        # Tab press first.  Accepting a measured gain is what an operator came for;
        # a sweep with no answer has only the one way out.
        (accept_button if accept else cancel_button).focus()

    def _set(self, selector: str, text: str) -> None:
        # Same reasoning as CalibrationMeterDialog._show: a worker is cancelled at its
        # next await rather than mid-statement, so an update can arrive after the
        # widget has left the DOM while the screen still reads as mounted.
        try:
            self.query_one(selector, Static).update(text)
        except NoMatches:
            pass

    def on_button_pressed(self, event) -> None:
        if event.button.id == 'accept' and self._result is not None:
            self.dismiss(self._result.chosen_db)
            return
        self.action_cancel()

    def action_cancel(self) -> None:
        if self._sweep is not None:
            self._sweep.cancel()
        self.dismiss(CANCELLED)

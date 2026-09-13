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
import logging
from typing import TYPE_CHECKING, Any

from textual import work
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from buzz.config import RtlSdrConfig
from buzz.gain_sweep import GainSweep, ProgressCallback, SweepResult
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen

if TYPE_CHECKING:
    from buzz.sdr import SweepReader

logger = logging.getLogger(__name__)

def open_sweep(rtlsdr_values: SectionValues) -> tuple['SweepReader', GainSweep]:
    """Open the receiver and build a sweep over it.

    A SweepReader rather than the RtlSdrSource the monitor uses.  It reads
    synchronously on one thread, so changing gain cannot race a capture thread that
    is driving libusb's event loop, which is what left a receiver wedged and the
    program hung.  See its own docstring.

    The imports sit inside the function for the reason buzz.main.open_live_source
    gives: a station using a sound card should never load pyrtlsdr, which resolves a
    symbol as it imports and so fails at import rather than at first call.

    Whatever this raises carries a message written for whoever is standing at the
    radio, because open_device rewords libusb's own wording.  The dialog shows it
    rather than letting a traceback through.
    """
    from buzz.sdr import SweepReader, open_device

    settings = RtlSdrConfig(**rtlsdr_values)
    reader = SweepReader(
        open_device(settings.device_index),
        frequency_hz=settings.frequency_hz, gain_db=settings.gain_db,
        iq_sample_rate=settings.iq_sample_rate,
        tuning_offset_hz=settings.tuning_offset_hz)
    return reader, GainSweep(reader, settings.arc_headroom_db)


def _sweep_then_release(source: 'SweepReader', sweep: GainSweep,
                        on_progress: ProgressCallback) -> tuple[SweepResult, bool]:
    """Run the sweep and release the receiver, both on the calling thread.

    The release used to be an awaited call in the worker's `finally`, which meant the
    event loop decided whether it happened.  It did not always happen.  A sweep that
    reached its last step and then failed anywhere afterwards left the device held,
    and the next attempt to open it came back as LIBUSB_ERROR_ACCESS: a permissions
    error that is nothing of the sort.

    Here the close is in the same thread as the work, after a `finally` that no task
    cancellation can skip, because a thread started by asyncio.to_thread runs to
    completion whatever happens to the task awaiting it.

    Returns the result and whether the device was actually released.  RtlSdrSource
    leaves it open on purpose when its capture thread will not stop, since freeing a
    handle that thread is still reading through is a crash in C rather than an
    exception, so this can be False after an otherwise perfect sweep.
    """
    try:
        result = sweep.run(on_progress)
    finally:
        # Assigned here and returned below rather than built into the return above,
        # because a return expression is evaluated before the finally runs, so the
        # tuple would have carried the value released had before the close.
        # RtlSdrSource.close bounds itself, including the rtlsdr_close that can
        # block inside libusb and never return.  See its own docstring; the bound
        # lives there because every other path that closes a receiver needs it too,
        # the atexit hook among them.
        released = source.close()
        if not released:
            logger.warning(
                'The receiver was still held after the sweep, so the next attempt to '
                'open it fails until this program exits.')
    return result, released


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
    #instructions {
        width: 64;
        padding-bottom: 1;
    }
    #status {
        width: 64;
        text-style: bold;
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
            # Standing advice, in its own widget.  It shared one with the progress
            # line until somebody pointed out that it vanished before it could be
            # read: the first step overwrote it about four hundred milliseconds in.
            # Worded so that it still reads correctly after the sweep has finished.
            Static('This measures the band at every gain the tuner offers, five '
                   'times over, and takes a little over a minute.  Leave the antenna '
                   'connected and the receiver tuned where it will run.  Calibrate '
                   'when the band is quiet if you can: a running arc raises the '
                   'noise floor, and the gain then comes out low for the hours '
                   'either side.', id='instructions'),
            Static('Starting...', id='status'),
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
            result, released = await asyncio.to_thread(
                _sweep_then_release, source, sweep,
                self._progress_reporter(asyncio.get_running_loop()))
        except Exception as exc:
            self._set('#outcome', f'The sweep failed: {exc}')
            self._finish()
            return
        try:
            self._result = result
            self._show_result(result, released)
        except Exception as exc:
            # Anything raised past this point used to die inside the worker, which
            # Textual reports nowhere the operator can see: the dialog simply sat
            # there having measured everything and said nothing.
            logger.exception('Showing the sweep result failed.')
            self._set('#outcome', f'The sweep finished and the result would not '
                                  f'display: {exc}')
            self._finish()

    def _progress_reporter(self, loop: asyncio.AbstractEventLoop) -> ProgressCallback:
        """A progress callback the sweep's own thread can use without waiting.

        Textual is not thread-safe and this runs on the worker asyncio.to_thread put
        the sweep on, so the update has to cross back to the event loop.  It crosses
        with call_soon_threadsafe rather than App.call_from_thread, which blocks the
        caller until the loop has run the callback.

        Blocking there hung the program.  A loop that is shutting down never runs the
        callback, so the sweep thread waited on it for ever, and CPython's own
        ThreadPoolExecutor joins every worker it ever made during interpreter exit.
        One stuck thread therefore hangs the whole process rather than the dialog that
        orphaned it, which is the trap sampler.LevelStream.close already documents.

        A failure to post is swallowed for the same reason.  The loop being gone is
        not a reason to stop sweeping, and it is certainly not a reason to raise
        inside a thread nobody is watching.
        """
        def report(step: int, total: int, gain_db: float) -> None:
            try:
                loop.call_soon_threadsafe(
                    self._set, '#status',
                    f'Step {step + 1} of {total}: measuring {gain_db:.1f} dB...')
            except RuntimeError:
                pass

        return report

    def _show_result(self, result: SweepResult, released: bool = True) -> None:
        """Report the outcome, and say if the receiver is still held.

        A held receiver matters more than it sounds.  The next attempt to open one
        fails with LIBUSB_ERROR_ACCESS, which reads as a permissions problem and sends
        people to Zadig for something Zadig cannot fix.  Saying it here, where it
        happened, costs one line and saves that hunt.
        """
        if result.chosen_db is None:
            self._set('#status', 'The sweep finished without an answer.')
            self._set('#outcome', result.reason + self._held_note(released))
            self._finish()
            return
        self._set('#status', f'Measured gain: {result.chosen_db:.1f} dB')
        self._set('#outcome',
                  f'{result.reason}  The antenna supplies '
                  f'{result.antenna_share * 100:.0f}% of the noise floor at this '
                  f'gain, so the reported floor reads about '
                  f'{result.floor_error_db:.1f} dB high.{self._held_note(released)}')
        self._finish(accept=True)

    @staticmethod
    def _held_note(released: bool) -> str:
        """What to add when the receiver could not be released."""
        if released:
            return ''
        return ('\n\nThe receiver did not stop cleanly and is still held, so running '
                'this again will fail until the setup program is restarted.')

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

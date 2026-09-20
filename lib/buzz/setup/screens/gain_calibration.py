"""Choose the receiver's tuner gain by measuring the band.

A modal that runs buzz.receiver.gain_sweep against the receiver described by the in-progress
[rtlsdr] section, shows where it has got to, and offers the answer.  The measurement
and the arithmetic live in gain_sweep; this file is the screen around them.

The dialog warns rather than refuses when the sweep cannot reach an answer.  A quiet
antenna has no gain that satisfies both bounds, so refusing to close would trap
exactly the operator who most needs to set a gain by hand.  What it can always do is
say which of the two bounds failed and what to do about it.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from textual import work
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from buzz.config import receiver_settings_from
from buzz.receiver.gain_sweep import GainSweep, ProgressCallback, SweepResult
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen

if TYPE_CHECKING:
    from buzz.receiver.source import SweepReader

logger = logging.getLogger(__name__)

# Told the dialog that the receiver answered: the sweep now built over it, and the
# gains the receiver reported.  Both belong to the device, so nothing can state them
# before it is open.  The ladder rather than a count of it, because how long a sweep
# takes depends on how far apart the rungs are as well as how many there are.
OpenedCallback = Callable[[GainSweep, list[float]], None]


def open_sweep(source: str,
               values: SectionValues) -> tuple['SweepReader', GainSweep]:
    """Open whichever receiver `source` names and build a sweep over it.

    A SweepReader rather than the SdrSource the monitor uses.  It reads
    synchronously on one thread, so changing gain cannot race a capture thread that
    is driving libusb's event loop, which is what left a receiver wedged and the
    program hung.  See its own docstring.  An SDRplay has no synchronous read at all,
    so its own SweepReader runs a stream and takes one block from it.

    The imports sit inside the function for the reason buzz.main.open_live_source
    gives: a station using a sound card should never load a driver for hardware it does
    not own.

    Whatever this raises carries a message written for whoever is standing at the
    radio, because each device rewords its own driver's wording.  The dialog shows it
    rather than letting a traceback through.

    The device is closed again if anything after the open raises, because nothing else
    would.  Each `open` covers its own handle, so what is left for this guard is the
    read size SweepReader refuses and whatever building the sweep does.  Either one
    leaves a receiver no object owns and no atexit hook covers.
    """
    from buzz.receiver.device import open_receiver
    from buzz.receiver.source import SweepReader

    settings = receiver_settings_from(source, values)
    if settings is None:
        raise RuntimeError(
            f'[audio] source is {source!r}, and a gain sweep needs a receiver.  Set it '
            f'to a receiver first, then calibrate its gain.')
    device = open_receiver(source, settings)
    try:
        reader = SweepReader(device)
        return reader, GainSweep(reader, settings.arc_headroom_db)
    except BaseException:
        device.close()
        raise


def _open_sweep_then_release(source: str, values: SectionValues,
                             on_open: OpenedCallback,
                             on_progress: ProgressCallback) -> tuple[SweepResult, bool]:
    """Open the receiver, sweep it, and release it, all on the calling thread.

    The release used to be an awaited call in the worker's `finally`, which meant the
    event loop decided whether it happened.  It did not always happen.  A sweep that
    reached its last step and then failed anywhere afterwards left the device held,
    and the next attempt to open it came back as LIBUSB_ERROR_ACCESS: a permissions
    error that is nothing of the sort.

    Here the close is in the same thread as the work, after a `finally` that no task
    cancellation can skip, because a thread started by asyncio.to_thread runs to
    completion whatever happens to the task awaiting it.

    The open moved in here for the same reason, rather than staying in an
    asyncio.to_thread call of its own.  Textual cancels a screen's workers when it
    unmounts, and the CancelledError that raises is a BaseException, so an `except
    Exception` around the await never saw it.  The thread went on to build the reader
    and hand it back to a task that had gone, which held the receiver for the rest of
    the session.  Escape pressed during the opening second was enough, and it
    surfaced as the same LIBUSB_ERROR_ACCESS above.  Ownership of the device now
    never leaves this function.

    `on_open` is called with the sweep and the tuner gain count as soon as the
    receiver answers, so that the dialog can say how long the sweep will take.  It
    runs on this thread and must not block, which is the contract `on_progress` has
    as well.

    Returns the result and whether the device was actually released.  SweepReader
    leaves it open on purpose when the driver will not give it back, because waiting
    longer inside libusb only hangs the program, so this can be False after an
    otherwise perfect sweep.
    """
    reader, sweep = open_sweep(source, values)
    try:
        on_open(sweep, reader.supported_gains_db)
        result = sweep.run(on_progress)
    finally:
        # Assigned here and returned below rather than built into the return above,
        # because a return expression is evaluated before the finally runs, so the
        # tuple would have carried the value released had before the close.
        # SweepReader.close bounds itself, including the rtlsdr_close that can
        # block inside libusb and never return.  See its own docstring; the bound
        # lives there because every other path that closes a receiver needs it too,
        # the atexit hook among them.
        released = reader.close()
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

    def __init__(self, source: str, values: SectionValues) -> None:
        super().__init__()
        self._source = source
        self._values = values
        self._sweep: GainSweep | None = None
        self._result: SweepResult | None = None
        self._offers_gain = False
        # Set once something final is on screen, so that a progress update still in
        # flight cannot overwrite it.  See _show_progress.
        self._settled = False
        self._cancel_requested = False

    def compose(self):
        yield Vertical(
            Static('Calibrate receiver gain', id='title'),
            Static(self._instructions('every gain the tuner offers, several times '
                                      'over'), id='instructions'),
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
        """Drive the whole measurement from one worker, over one thread.

        One thread call rather than an open and then a sweep, because the receiver is
        released in that thread's `finally` and anything opened outside it is owned by
        a task that cancellation can take away.  See `_open_sweep_then_release`.

        Which of the two messages a failure gets is decided by whether the sweep ever
        reached this screen, because `_opening_reporter` is what puts it there and
        only a receiver that opened calls that.
        """
        loop = asyncio.get_running_loop()
        try:
            result, released = await asyncio.to_thread(
                _open_sweep_then_release, self._source, self._values,
                self._opening_reporter(loop), self._progress_reporter(loop))
        except Exception as exc:
            opened = self._sweep is not None
            self._set('#outcome', f'The sweep failed: {exc}' if opened
                      else f'Could not open the receiver: {exc}')
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

    @staticmethod
    def _instructions(what_it_measures: str) -> str:
        """The standing advice, with what the sweep is about to do written into it.

        It sits in its own widget rather than in the progress line, because the two
        shared one until somebody pointed out that the first step overwrote the advice
        about four hundred milliseconds in.  Worded so that it still reads correctly
        after the sweep has finished.
        """
        return (f'This measures the band at {what_it_measures}.  Leave the antenna '
                'connected and the receiver tuned where it will run.  Calibrate when '
                'the band is quiet if you can.  A running arc raises the noise floor, '
                'so the sweep sees a louder band than usual and picks a gain that is '
                'then too low once the arc stops.')

    @staticmethod
    def _duration_phrase(seconds: float) -> str:
        """Round a sweep estimate to a figure somebody can plan around.

        The estimate is a per-step figure times a step count, so it is not accurate to
        the second and saying 75 of them would claim it was.  Anything under three
        quarters of a minute is rounded to a quarter minute, and anything above it to
        a half minute.
        """
        if seconds < 45.0:
            return f'about {max(15, round(seconds / 15.0) * 15)} seconds'
        minutes = max(2, round(seconds / 30.0)) / 2.0
        return f'about {minutes:g} minute' + ('' if minutes == 1.0 else 's')

    def _opening_reporter(self, loop: asyncio.AbstractEventLoop) -> OpenedCallback:
        """A callback for the sweep thread to report that the receiver is open.

        The sweep is stored from that thread rather than posted to the loop with the
        screen update, so that Escape pressed a moment later finds something to
        cancel.  Posting it would leave a window in which the sweep is running and
        the dialog believes it has not started.  An attribute assignment is atomic and
        Textual's thread rules are about widgets, which this does not touch.

        The flag covers the other order.  Escape can arrive before the receiver
        answers at all, and the sweep to cancel does not exist yet, so the thread
        checks whether one was asked for as soon as it has something to ask.
        """
        def opened(sweep: GainSweep, gains: list[float]) -> None:
            self._sweep = sweep
            if self._cancel_requested:
                sweep.cancel()
            try:
                loop.call_soon_threadsafe(self._say_what_the_sweep_will_do, sweep,
                                          gains)
            except RuntimeError:
                pass

        return opened

    def _say_what_the_sweep_will_do(self, sweep: GainSweep, gains: list[float]) -> None:
        """Replace the opening advice with the figures the device has now supplied.

        These belong to the device: a V4 has 29 gains and an RSP has 101, so the
        opening text cannot state either before the receiver is open.

        It says the sweep narrows rather than quoting one gain count, because the two
        phases visit different numbers of gains and a single figure would be wrong for
        both.  What an operator is deciding here is whether to wait, so the duration is
        the part that has to be right.
        """
        self._set('#instructions', self._instructions(
            f'the {len(gains)} gains the receiver offers, coarsely at first and then '
            f'closely around the answer, which takes '
            f'{self._duration_phrase(sweep.estimated_seconds(gains))}'))

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
                loop.call_soon_threadsafe(self._show_progress, step, total, gain_db)
            except RuntimeError:
                pass

        return report

    def _show_progress(self, step: int, total: int, gain_db: float) -> None:
        """Say where the sweep has got to, unless it has already finished.

        The guard is the whole reason this is a method rather than the _set call it
        used to be.  A progress update crosses from the sweep's thread by
        call_soon_threadsafe, which queues it rather than running it, so one posted
        just before the sweep returned can still be waiting when the result reaches
        the screen.  Running it then replaces the answer with a progress line for a
        sweep that has already finished, and nothing puts the answer back.

        Caught by a test that read 'Step 1 of 2: measuring 0.0 dB...' where the
        measured gain should have been.
        """
        if self._settled:
            return
        self._set('#status', f'Step {step + 1} of {total}: measuring {gain_db:.1f} dB...')

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
        # Just the reason.  This used to append the antenna's share of the floor and
        # the decibels the floor reads high, which are one figure said twice, and the
        # reason already carries the decibels.
        self._set('#status', f'Measured gain: {result.chosen_db:.1f} dB')
        self._set('#outcome', result.reason + self._held_note(released))
        self._finish(accept=True)

    @staticmethod
    def _held_note(released: bool) -> str:
        """What to add when the receiver could not be released."""
        if released:
            return ''
        return ('\n\nThe receiver did not stop cleanly and is still held, so running '
                'this again will fail until the setup program is restarted.')

    def _finish(self, accept: bool = False) -> None:
        """Offer whatever the operator can do now, and stop progress overwriting it.

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
        # Every path that reaches an outcome comes through here, which makes it the
        # one place that can say the screen is now showing something final.
        self._settled = True
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
        # The flag first, so that a sweep still being built on the other thread sees
        # it.  See _opening_reporter for the two orders this has to survive.
        self._cancel_requested = True
        if self._sweep is not None:
            self._sweep.cancel()
        self.dismiss(CANCELLED)

"""Tests for the gain calibration dialog and the menu row that opens it.

The sweep's own arithmetic is covered in test_gain_sweep.py.  What is checked here is
the screen around it: that the row sits where the procedure needs it, that an answer
reaches the config, and that a sweep with no answer says so rather than writing a
number picked from a rule that failed.
"""
import threading
import time
from asyncio import run
from pathlib import Path
from unittest.mock import patch

import pytest

from textual.widgets import Button, OptionList, Static

from buzz.gain_sweep import GainMeasurement, SweepResult
from buzz.setup.app import SetupApp
from buzz.setup.screens.base import CANCELLED
from buzz.setup.screens.gain_calibration import GainCalibrationDialog
from buzz.setup.screens.section_menu import _ACTIONS, _SWEEP_ID, SectionMenuScreen

RTLSDR_VALUES = {'frequency_khz': 3588.0, 'gain_db': 40.2, 'device_index': 0,
                 'iq_sample_rate': 256_000, 'decimation': 16, 'bandwidth_khz': 4.0,
                 'tuning_offset_khz': 50.0, 'sideband': 'upper',
                 'arc_headroom_db': 32.0, 'calibrated_offset_db': -40.2,
                 'calibrated_at_gain_db': None}


def _result(chosen=36.4, reason='Measured.', share=0.81):
    return SweepResult(chosen_db=chosen, reason=reason, antenna_share=share,
                       floor_bound_db=chosen, headroom_bound_db=44.5,
                       measurements=(GainMeasurement(
                           gain_db=36.4, quiet_dbfs=-41.0, peak_dbfs=-12.0,
                           clipped=0, raw_values=640_000, passes=5),))


async def _wait_until(pilot, condition, description, timeout=5.0):
    """Pump the app until the condition holds.  See test_setup_app's own copy for why
    a single pause would be racing the worker that fills these dialogs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if condition():
            return
    raise AssertionError(f'timed out waiting for {description}')


class _FakeSweep:
    """Stands in for GainSweep, so the dialog is tested without the arithmetic."""

    passes = 5

    def __init__(self, result):
        self._result = result
        self.cancelled = False
        self.progress_reported = []

    def estimated_seconds(self, gain_count):
        return gain_count * 2.6

    def run(self, on_progress=None):
        if on_progress is not None:
            on_progress(0, 2, 0.0)
            self.progress_reported.append((0, 2, 0.0))
        return self._result

    def cancel(self):
        self.cancelled = True


class _FakeSource:
    # Fewer steps than a V4, so a test can tell a derived count from a hard-coded one.
    supported_gains_db = [0.0, 14.4, 25.4, 32.8, 40.2, 49.6]

    def __init__(self, released=True):
        self.closed = False
        self._released = released

    def close(self):
        self.closed = True
        return self._released


class TestTheMenuRowSitsInTheProcedure:
    """The receiver section lists its steps in order.  A row that runs a step of that
    procedure would read as an afterthought at the bottom of the list.
    """

    def test_the_receiver_section_has_an_auto_calibrate_row(self):
        assert 'rtlsdr' in _ACTIONS
        assert any(action.id == _SWEEP_ID for action in _ACTIONS['rtlsdr'])

    def test_it_follows_the_frequency_rather_than_sitting_at_the_end(self):
        """Frequency first, because the sweep measures whatever the receiver is tuned
        to and a gain measured on the wrong band is wrong.
        """
        row = next(action for action in _ACTIONS['rtlsdr'] if action.id == _SWEEP_ID)
        assert row.after == 'frequency_khz'

    def test_it_renders_between_the_frequency_and_the_gain(self, tmp_path):
        """The rendered order rather than the declaration, since positioning a row is
        the whole job of _ActionRow.after and nothing else checks it comes out right.
        """
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                app.values['audio']['source'] = 'rtlsdr'
                await app.push_screen(SectionMenuScreen('rtlsdr'))
                await pilot.pause()
                options = app.screen.query_one('#fields', OptionList)
                ids = [options.get_option_at_index(i).id
                       for i in range(options.option_count)]
                assert ids[:3] == ['frequency_khz', _SWEEP_ID, 'gain_db'], ids

        run(scenario())


class TestTheDialogReportsWhatTheSweepFound:
    def _open(self, tmp_path, sweep):
        """Drive the dialog to the point where it has an outcome, and read the screen."""
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), sweep)):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: app.screen.query_one('#outcome', Static).content != '',
                        'the dialog to report an outcome')
                    accept = app.screen.query_one('#accept', Button)
                    return (app.screen.query_one('#status', Static).content,
                            app.screen.query_one('#outcome', Static).content,
                            str(app.screen.query_one('#cancel', Button).label),
                            not accept.has_class('hidden'))

        return run(scenario())

    def test_a_measured_gain_is_shown_with_what_it_cost(self, tmp_path):
        """The cost comes from the sweep's own reason rather than being appended here.
        The dialog used to add the antenna's share of the floor and the decibels it
        reads high, which are one figure said twice.
        """
        status, outcome, label, offers = self._open(tmp_path, _FakeSweep(_result()))
        assert '36.4' in status
        assert outcome.startswith('Measured.'), outcome
        assert '%' not in outcome, 'the share is the decibels said again'
        assert offers, 'the measured gain was not offered'
        assert label == 'Cancel', (
            'a measured gain has to be refusable without pressing Escape')

    def test_a_late_progress_update_cannot_replace_the_result(self, tmp_path):
        """Progress crosses from the sweep's thread by call_soon_threadsafe, which
        queues rather than runs, so one posted just before the sweep returned can still
        be waiting when the result reaches the screen.  Running it then puts a progress
        line back over the answer, and nothing puts the answer back.

        Seen in CI, where this class read 'Step 1 of 2: measuring 0.0 dB...' where the
        measured gain belonged.  Driven directly here rather than hoping for the race,
        since it only showed up under a fully loaded machine.
        """
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), _FakeSweep(_result()))):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: app.screen.query_one('#outcome', Static).content != '',
                        'the dialog to report an outcome')
                    # The straggler, arriving after the answer is already on screen.
                    app.screen._show_progress(0, 145, 0.0)
                    await pilot.pause()
                    return app.screen.query_one('#status', Static).content

        status = run(scenario())
        assert '36.4' in status, (
            f'a late progress update replaced the measured gain: {status!r}')

    def test_no_answer_offers_no_gain_to_accept(self, tmp_path):
        """The quiet-antenna case.  The dialog closes rather than refusing, because a
        station whose sweep cannot succeed is exactly the one that has to set a gain by
        hand.  It must not hand back a number anyway.
        """
        nothing = SweepResult(None, 'The antenna is too quiet.', 0.1, None, 49.6, ())
        status, outcome, label, offers = self._open(tmp_path, _FakeSweep(nothing))
        assert 'without an answer' in status
        assert 'too quiet' in outcome
        assert not offers, 'there is no gain to accept'
        assert label == 'Close'

    def test_progress_reaches_the_screen_while_the_sweep_runs(self, tmp_path):
        """The sweep calls back from its own thread, so this also covers the hop back
        onto the event loop that Textual requires of anything touching a widget.
        """
        sweep = _FakeSweep(_result())
        self._open(tmp_path, sweep)
        assert sweep.progress_reported, 'the dialog never passed a progress callback'

    def test_the_receiver_is_closed_even_though_the_dialog_stays_open(self, tmp_path):
        """Nothing else will close it: the dialog holds no handle after the sweep, and
        a receiver left streaming is one the monitor cannot open afterwards.
        """
        source = _FakeSource()

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(source, _FakeSweep(_result()))):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(pilot, lambda: source.closed,
                                      'the receiver to be closed')

        run(scenario())

    def test_a_receiver_that_will_not_open_is_reported_rather_than_raised(self, tmp_path):
        """open_device rewords libusb's own message for whoever is at the radio, so
        the wording has to reach the screen rather than a traceback.
        """
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           side_effect=RuntimeError('no driver is bound to it')):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: 'driver' in app.screen.query_one('#outcome', Static).content,
                        'the open failure to reach the screen')

        run(scenario())


class TestAcceptingTheAnswerWritesTheConfig:
    def _run_with(self, tmp_path, answer):
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                app.values['audio']['source'] = 'rtlsdr'
                screen = SectionMenuScreen('rtlsdr')
                await app.push_screen(screen)
                await pilot.pause()
                before = dict(app.values['rtlsdr'])
                with patch.object(app, 'push_screen_wait', return_value=answer):
                    await screen._calibrate_gain()
                return before, dict(app.values['rtlsdr'])

        return run(scenario())

    def test_it_sets_the_gain_the_offset_and_the_calibration_mark(self, tmp_path):
        """All three move together.  The offset starts at the negative of the gain
        because the true gain per step cannot be measured without a reference signal,
        and calibrated_at_gain_db is what lets startup notice the gain moved later.
        """
        _, after = self._run_with(tmp_path, 36.4)
        assert after['gain_db'] == 36.4
        assert after['calibrated_offset_db'] == -36.4
        assert after['calibrated_at_gain_db'] == 36.4

    def test_cancelling_changes_nothing(self, tmp_path):
        before, after = self._run_with(tmp_path, CANCELLED)
        assert after == before

    def test_a_dialog_that_reached_no_answer_changes_nothing(self, tmp_path):
        """dismiss(None) is what a closed-without-an-answer dialog sends back, and it
        must not be written as a gain of None.
        """
        before, after = self._run_with(tmp_path, None)
        assert after == before


class TestTheDialogSurvivesTheWaysItCanGoWrong:
    """Every one of these used to be a traceback in a worker, which Textual turns into
    the whole app coming down rather than one dialog reporting a problem.
    """

    def test_a_sweep_that_raises_is_reported_and_the_receiver_still_closed(self, tmp_path):
        """The receiver is released in a finally, so a failed sweep does not leave a
        device the monitor cannot open afterwards.
        """
        source = _FakeSource()

        class _ExplodingSweep(_FakeSweep):
            def run(self, on_progress=None):
                raise RuntimeError('the tuner stopped answering')

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(source, _ExplodingSweep(_result()))):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: 'stopped answering' in app.screen.query_one(
                            '#outcome', Static).content,
                        'the sweep failure to reach the screen')
                    assert str(app.screen.query_one('#cancel', Button).label) == 'Close'
                    # The close runs in the sweep's own thread, so it has already
                    # happened by the time the failure reaches the screen.
                    assert source.closed

        run(scenario())

    def test_pressing_the_button_after_an_answer_returns_the_gain(self, tmp_path):
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), _FakeSweep(_result()))):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog)
                    await _wait_until(
                        pilot,
                        lambda: not app.screen.query_one(
                            '#accept', Button).has_class('hidden'),
                        'the dialog to offer a gain')
                    await pilot.click('#accept')
                    await pilot.pause()
                    assert dialog._result.chosen_db == 36.4

        run(scenario())

    def test_cancelling_mid_sweep_asks_the_sweep_to_stop(self, tmp_path):
        """Escape has to reach the sweep, or the receiver keeps stepping through gains
        for another minute behind a dialog that is already gone.
        """
        sweep = _FakeSweep(_result())

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), sweep)):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog)
                    await _wait_until(pilot, lambda: dialog._sweep is not None,
                                      'the sweep to start')
                    dialog.action_cancel()
                    await pilot.pause()
        run(scenario())
        assert sweep.cancelled

    def test_an_update_arriving_after_the_widget_is_gone_is_dropped(self, tmp_path):
        """A worker is cancelled at its next await rather than mid-statement, so an
        update can reach a widget that has already left the DOM.  Swallowing NoMatches
        is what keeps that from taking the app down.
        """
        dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
        dialog._set('#status', 'no screen is mounted, so this must not raise')
        dialog._finish(accept=True)
        assert dialog._offers_gain is True

    def test_the_row_opens_the_dialog(self, tmp_path):
        """The wiring from the menu row through to the screen, which nothing else
        covers: the two halves can each be right while the row opens nothing.
        """
        opened = []

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                app.values['audio']['source'] = 'rtlsdr'
                screen = SectionMenuScreen('rtlsdr')
                await app.push_screen(screen)
                await pilot.pause()
                with patch.object(app, 'push_screen_wait',
                                  side_effect=lambda s: opened.append(s) or CANCELLED):
                    rows = app.screen.query_one('#fields', OptionList)
                    rows.highlighted = 1
                    await pilot.press('enter')
                    await _wait_until(pilot, lambda: bool(opened),
                                      'the sweep dialog to be opened')

        run(scenario())
        assert isinstance(opened[0], GainCalibrationDialog)


class TestOpeningTheRealReceiver:
    """open_sweep is the boundary against pyrtlsdr, so it is tested with the library
    mocked.  Its absence must not move the coverage number, the same rule ffmpeg
    follows in render.py.
    """

    def test_it_opens_the_configured_device(self, tmp_path):
        from buzz.setup.screens.gain_calibration import open_sweep

        with patch('buzz.sdr.open_device') as open_device, \
             patch('buzz.sdr.SweepReader') as reader_class:
            values = dict(RTLSDR_VALUES, device_index=2)
            source, sweep = open_sweep(values)

        open_device.assert_called_once_with(2)
        # The receiver is set in Hz; only the config key moved to kHz.
        assert reader_class.call_args.kwargs['frequency_hz'] == 3_588_000
        assert source is reader_class.return_value

    def test_it_reads_synchronously_rather_than_streaming(self):
        """The whole reason the sweep has a reader of its own.  RtlSdrSource streams
        on a capture thread, and changing gain against that is two threads on one
        device, which wedged the receiver and hung the program.
        """
        from buzz.setup.screens.gain_calibration import open_sweep

        with patch('buzz.sdr.open_device'), \
             patch('buzz.sdr.SweepReader'), \
             patch('buzz.sdr.RtlSdrSource') as streaming:
            open_sweep(dict(RTLSDR_VALUES))
        streaming.assert_not_called()

    def test_the_headroom_comes_from_the_config_rather_than_a_literal(self):
        """arc_headroom_db is the reserve the whole choice turns on, so a sweep built
        with the default instead of the operator's value would quietly ignore it.
        """
        from buzz.setup.screens.gain_calibration import open_sweep

        with patch('buzz.sdr.open_device'), patch('buzz.sdr.SweepReader'):
            _, sweep = open_sweep(dict(RTLSDR_VALUES, arc_headroom_db=26.0))
        assert sweep._headroom_db == 26.0


class TestAMeasuredGainCanBeRefused:
    """Running the sweep to see what it says is a reasonable thing to do, and so is
    disagreeing with the answer.  A dialog whose only exit applies the change makes
    refusing it feel like an error.  Escape always worked; an operator should not have
    to know that.
    """

    def _dialog_after_sweep(self, tmp_path, sweep, then):
        result = {}

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), sweep)):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog, lambda value: result.update(value=value))
                    await _wait_until(
                        pilot,
                        lambda: not app.screen.query_one(
                            '#accept', Button).has_class('hidden'),
                        'the dialog to offer a gain')
                    await then(pilot)
                    await _wait_until(pilot, lambda: 'value' in result,
                                      'the dialog to dismiss')

        run(scenario())
        return result['value']

    def test_cancel_returns_no_gain(self, tmp_path):
        async def press_cancel(pilot):
            await pilot.click('#cancel')

        assert self._dialog_after_sweep(
            tmp_path, _FakeSweep(_result()), press_cancel) is CANCELLED

    def test_accept_returns_the_measured_gain(self, tmp_path):
        async def press_accept(pilot):
            await pilot.click('#accept')

        assert self._dialog_after_sweep(
            tmp_path, _FakeSweep(_result()), press_accept) == 36.4

    def test_escape_still_cancels(self, tmp_path):
        """It was the only way out before, so it keeps working for anybody used to it."""
        async def press_escape(pilot):
            await pilot.press('escape')

        assert self._dialog_after_sweep(
            tmp_path, _FakeSweep(_result()), press_escape) is CANCELLED

    def test_cancelling_leaves_the_stored_gain_alone(self, tmp_path):
        """The property behind the button, checked where the value actually lives."""
        from buzz.setup.screens.section_menu import SectionMenuScreen

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                app.values['audio']['source'] = 'rtlsdr'
                screen = SectionMenuScreen('rtlsdr')
                await app.push_screen(screen)
                await pilot.pause()
                before = dict(app.values['rtlsdr'])
                with patch.object(app, 'push_screen_wait', return_value=CANCELLED):
                    await screen._calibrate_gain()
                return before, dict(app.values['rtlsdr'])

        before, after = run(scenario())
        assert after == before


class TestTheKeyboardReachesBothButtons:
    """The mouse and Escape worked; the arrow keys did not.  A Horizontal's children
    take Tab already, because Screen.BINDINGS binds it to app.focus_next, but not the
    arrows a row of buttons invites somebody to reach for.
    """

    def _after_sweep(self, tmp_path, keys):
        result = {}

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), _FakeSweep(_result()))):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog, lambda value: result.update(value=value))
                    await _wait_until(
                        pilot,
                        lambda: not app.screen.query_one(
                            '#accept', Button).has_class('hidden'),
                        'the dialog to offer a gain')
                    result['focused_first'] = app.screen.focused.id
                    for key in keys:
                        await pilot.press(key)
                    await _wait_until(pilot, lambda: 'value' in result,
                                      'the dialog to dismiss')

        run(scenario())
        return result

    def test_the_accept_button_starts_focused(self, tmp_path):
        """So Enter works without a Tab press first.  Taking the measured gain is what
        an operator opened the sweep for.
        """
        assert self._after_sweep(tmp_path, ['enter'])['focused_first'] == 'accept'

    def test_enter_on_the_focused_button_accepts(self, tmp_path):
        assert self._after_sweep(tmp_path, ['enter'])['value'] == 36.4

    def test_right_then_enter_cancels(self, tmp_path):
        """The bug: this sequence did nothing, and the only ways out were the mouse
        and Escape.
        """
        assert self._after_sweep(tmp_path, ['right', 'enter'])['value'] is CANCELLED

    def test_left_comes_back_to_accept(self, tmp_path):
        assert self._after_sweep(tmp_path, ['right', 'left', 'enter'])['value'] == 36.4

    def test_tab_still_works(self, tmp_path):
        """Screen.BINDINGS already bound it, and the arrows were added beside it
        rather than in place of it.
        """
        assert self._after_sweep(tmp_path, ['tab', 'enter'])['value'] is CANCELLED


class TestTheReceiverIsAlwaysReleased:
    """A sweep that reached its last step and then failed anywhere afterwards left the
    device held, and the next attempt to open one came back as LIBUSB_ERROR_ACCESS: a
    permissions error that is nothing of the sort.

    The release is in the same thread as the work now, after a finally no task
    cancellation can skip, because a thread asyncio.to_thread started runs to
    completion whatever happens to the task awaiting it.

    The open is in that thread too, for the same reason read the other way round: a
    receiver opened by a thread nobody is awaiting belongs to nothing.  Between them
    these cover the three ways the device was still reachable without an owner, which
    are a cancelled open, a reader that would not configure, and a callback that
    raised between the two.
    """

    def _run(self, source, sweep, on_open=lambda *a: None):
        """Drive the thread function with the receiver already stood in for."""
        from buzz.setup.screens.gain_calibration import _open_sweep_then_release

        with patch('buzz.setup.screens.gain_calibration.open_sweep',
                   return_value=(source, sweep)):
            return _open_sweep_then_release(dict(RTLSDR_VALUES), on_open,
                                            lambda *a: None)

    def test_a_sweep_that_raises_still_releases_it(self):
        source = _FakeSource()

        class _Exploding(_FakeSweep):
            def run(self, on_progress=None):
                raise RuntimeError('the tuner stopped')

        with pytest.raises(RuntimeError):
            self._run(source, _Exploding(_result()))
        assert source.closed

    def test_a_callback_that_raises_still_releases_it(self):
        """on_open runs on this thread between the open and the sweep, which is the one
        stretch where the device is held and no finally had covered it.
        """
        source = _FakeSource()

        def explode(sweep, gain_count):
            raise ValueError('the screen went away mid-update')

        with pytest.raises(ValueError):
            self._run(source, _FakeSweep(_result()), explode)
        assert source.closed

    def test_it_reports_whether_the_device_came_back(self):
        for released in (True, False):
            _, reported = self._run(_FakeSource(released), _FakeSweep(_result()))
            assert reported is released

    def test_the_release_value_is_read_after_the_close_not_before(self):
        """A return expression is evaluated before the finally runs, so building the
        tuple inside the try would have carried whatever the flag held beforehand.
        """
        _, reported = self._run(_FakeSource(released=True), _FakeSweep(_result()))
        assert reported is True

    def test_it_says_how_many_gains_the_tuner_offers_once_it_answers(self):
        """The count reaches the screen through the callback now, because the open
        happens on the sweep's thread rather than in an await of its own.
        """
        opened = []
        sweep = _FakeSweep(_result())
        self._run(_FakeSource(), sweep, lambda s, count: opened.append((s, count)))
        assert opened == [(sweep, len(_FakeSource.supported_gains_db))]

    def test_a_receiver_that_will_not_configure_is_not_left_open(self):
        """The other half of the leak: open_device succeeds and SweepReader raises
        somewhere inside configure_device, which leaves a handle no object owns.  The
        atexit hook is registered on the constructor's last line, so it covers nothing
        here.
        """
        from buzz.setup.screens.gain_calibration import open_sweep

        with patch('buzz.sdr.open_device') as open_device, \
             patch('buzz.sdr.close_device') as close_device, \
             patch('buzz.sdr.SweepReader',
                   side_effect=RuntimeError('the tuner would not take a rate')):
            with pytest.raises(RuntimeError):
                open_sweep(dict(RTLSDR_VALUES))

        close_device.assert_called_once_with(open_device.return_value)

    def test_cancelling_while_the_receiver_opens_still_releases_it(self, tmp_path):
        """The defect this class exists for, in the one place it was still possible.

        Textual cancels a screen's workers when it unmounts, and CancelledError is a
        BaseException, so the `except Exception` around the await never saw it.  The
        open ran in a thread of its own, finished after the task had gone, and handed
        back a receiver nobody closed.  Opening one takes about 0.72 s on this
        hardware, so Escape during the opening line was enough to do it.

        The open is held here until after the cancel rather than timed, so the test
        pins the ordering rather than racing it.
        """
        source = _FakeSource()
        opening = threading.Event()
        cancelled = threading.Event()

        def blocking_open(values):
            opening.set()
            cancelled.wait(5.0)
            return source, _FakeSweep(_result())

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           side_effect=blocking_open):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog)
                    await _wait_until(pilot, opening.is_set,
                                      'the receiver to start opening')
                    dialog.action_cancel()
                    await pilot.pause()
                    cancelled.set()
                    await _wait_until(pilot, lambda: source.closed,
                                      'the receiver to be released')

        run(scenario())

    def test_cancelling_before_the_receiver_answers_still_stops_the_sweep(self, tmp_path):
        """Escape can arrive before there is a sweep to cancel, so the request is
        remembered and applied by the thread as soon as one exists.  Without it the
        receiver goes on stepping through gains behind a dialog that has gone.
        """
        sweep = _FakeSweep(_result())
        opening = threading.Event()
        cancelled = threading.Event()

        def blocking_open(values):
            opening.set()
            cancelled.wait(5.0)
            return _FakeSource(), sweep

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           side_effect=blocking_open):
                    dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                    app.push_screen(dialog)
                    await _wait_until(pilot, opening.is_set,
                                      'the receiver to start opening')
                    assert dialog._sweep is None, 'the fixture must cancel first'
                    dialog.action_cancel()
                    await pilot.pause()
                    cancelled.set()
                    await _wait_until(pilot, lambda: sweep.cancelled,
                                      'the sweep to be cancelled')

        run(scenario())

    def test_a_held_receiver_is_said_on_screen(self):
        """The consequence falls on the next run as a libusb error that sends people
        to Zadig for something Zadig cannot fix.  Saying it where it happened costs a
        line.
        """
        note = GainCalibrationDialog._held_note(released=False)
        assert 'still held' in note
        assert GainCalibrationDialog._held_note(released=True) == ''

    def test_a_failure_showing_the_result_reaches_the_screen(self, tmp_path):
        """It used to die inside the worker, which Textual reports nowhere an operator
        can see: the dialog sat there having measured everything and said nothing.
        """
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), _FakeSweep(_result()))), \
                     patch.object(GainCalibrationDialog, '_show_result',
                                  side_effect=ValueError('cannot format that')):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: 'would not display' in app.screen.query_one(
                            '#outcome', Static).content,
                        'the display failure to be reported')

        run(scenario())


class TestProgressDoesNotBlockTheSweep:
    """App.call_from_thread waits until the loop has run the callback.  A loop that is
    shutting down never runs it, so the sweep thread waited for ever, and CPython's
    ThreadPoolExecutor joins every worker it made during interpreter exit.  One stuck
    thread hangs the whole process, which is what exiting the program did.
    """

    def test_it_posts_rather_than_waiting(self, tmp_path):
        posted = []

        class _Loop:
            def call_soon_threadsafe(self, fn, *args):
                posted.append(args)

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test():
                dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                report = dialog._progress_reporter(_Loop())
                report(0, 145, 32.8)

        run(scenario())
        assert posted, 'nothing was posted to the loop'
        # The step, not a formatted line: the formatting moved behind _show_progress so
        # that a late update can be dropped rather than overwriting the result.  What
        # reaches the screen is checked by
        # test_progress_reaches_the_screen_while_the_sweep_runs.
        assert posted[0] == (0, 145, 32.8)

    def test_a_dead_loop_does_not_stop_the_sweep(self, tmp_path):
        """The loop being gone is not a reason to stop sweeping, and certainly not a
        reason to raise inside a thread nobody is watching.
        """
        class _DeadLoop:
            def call_soon_threadsafe(self, fn, *args):
                raise RuntimeError('Event loop is closed')

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test():
                dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                dialog._progress_reporter(_DeadLoop())(0, 145, 32.8)

        run(scenario())

    def test_a_dead_loop_does_not_lose_the_sweep_that_just_opened(self, tmp_path):
        """The opening report crosses back the same way and swallows the same failure.
        The sweep is still stored, because the dialog needs it to cancel whatever the
        loop is doing, and the receiver is open by this point either way.
        """
        class _DeadLoop:
            def call_soon_threadsafe(self, fn, *args):
                raise RuntimeError('Event loop is closed')

        sweep = _FakeSweep(_result())

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test():
                dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                dialog._opening_reporter(_DeadLoop())(sweep, 6)
                return dialog._sweep

        assert run(scenario()) is sweep

    def test_the_blocking_bridge_is_not_called(self):
        """The blocking call is the defect, so its absence is what is worth pinning.
        The name still appears in a docstring saying why it is not used, so this looks
        for the call rather than the mention.
        """
        source = Path(
            'lib/buzz/setup/screens/gain_calibration.py').read_text(encoding='utf-8')
        assert '.call_from_thread(' not in source
        assert '.call_soon_threadsafe(' in source


class TestTheInstructionsStayOnScreen:
    """They shared a widget with the progress line, so the first step overwrote them
    about four hundred milliseconds in and nobody could read them.
    """

    def _shown(self, tmp_path, sweep):
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           return_value=(_FakeSource(), sweep)):
                    app.push_screen(GainCalibrationDialog(dict(RTLSDR_VALUES)))
                    await _wait_until(
                        pilot,
                        lambda: app.screen.query_one('#outcome', Static).content != '',
                        'the dialog to report an outcome')
                    return app.screen.query_one('#instructions', Static).content

        return run(scenario())

    def test_they_survive_the_whole_sweep(self, tmp_path):
        instructions = self._shown(tmp_path, _FakeSweep(_result()))
        assert 'Leave the antenna connected' in instructions

    def test_they_survive_a_sweep_that_found_nothing(self, tmp_path):
        nothing = SweepResult(None, 'The antenna is too quiet.', 0.1, None, 49.6, ())
        assert 'antenna' in self._shown(tmp_path, _FakeSweep(nothing))

    def test_the_gain_count_and_the_duration_come_from_the_device(self, tmp_path):
        """A V4 offers 29 steps and another tuner offers its own number, so a fixed
        "five times over, a little over a minute" is right for one receiver and wrong
        for the rest.  The stand-in offers six gains at 2.6 seconds each.
        """
        instructions = self._shown(tmp_path, _FakeSweep(_result()))
        assert 'each of the 6 gains' in instructions
        assert '5 times over' in instructions
        assert 'about 15 seconds' in instructions, instructions

    def test_the_opening_text_claims_no_duration_before_the_device_answers(self):
        """Nothing knows how many gains there are until the receiver is open, and a
        figure stated before that would be a guess.
        """
        opening = GainCalibrationDialog._instructions(
            'every gain the tuner offers, several times over')
        assert 'minute' not in opening and 'second' not in opening
        assert 'Leave the antenna connected' in opening

    @pytest.mark.parametrize('seconds, expected', [
        (3.0, 'about 15 seconds'),      # never rounds away to nothing
        (26.0, 'about 30 seconds'),
        (44.9, 'about 45 seconds'),
        (60.0, 'about 1 minute'),       # singular
        (75.4, 'about 1.5 minutes'),    # the V4, measured at about 75 seconds
        (130.0, 'about 2 minutes'),
    ])
    def test_the_duration_is_rounded_to_something_worth_reading(self, seconds, expected):
        """The estimate is a per-step figure times a step count, so saying 75 seconds
        would claim an accuracy it does not have.
        """
        assert GainCalibrationDialog._duration_phrase(seconds) == expected

    def test_progress_and_instructions_are_different_widgets(self, tmp_path):
        """The defect stated directly: one widget cannot hold both, because the
        progress line is rewritten 145 times.
        """
        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                dialog = GainCalibrationDialog(dict(RTLSDR_VALUES))
                with patch('buzz.setup.screens.gain_calibration.open_sweep',
                           side_effect=RuntimeError('no receiver')):
                    app.push_screen(dialog)
                    await pilot.pause()
                    instructions = app.screen.query_one('#instructions', Static)
                    status = app.screen.query_one('#status', Static)
                    assert instructions is not status

        run(scenario())

    def test_they_say_to_calibrate_on_a_quiet_band(self, tmp_path):
        """The one piece of advice that is not obvious from the screen: a running arc
        raises the floor, so the sweep picks a gain for a louder band than the station
        usually has, and that gain is too low once the arc stops.
        """
        assert 'quiet' in self._shown(tmp_path, _FakeSweep(_result()))

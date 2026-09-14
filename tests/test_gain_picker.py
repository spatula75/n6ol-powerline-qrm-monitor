"""Tests for the tuner gain picker.

A tuner accepts a fixed set of gains and snaps anything else to the nearest, so a
typed 41.0 became 40.2 and nothing told the operator.  The picker exists so that the
number shown is the number in use.

The list comes from hardware, so every test here supplies it through a patched
open_device.  The absence of a receiver must not move the coverage number, the same
rule ffmpeg follows in render.py.
"""
import asyncio
import time
from asyncio import run
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import OptionList, Static

from buzz.setup.app import SetupApp
from buzz.setup.screens.base import CANCELLED
from buzz.setup.screens.gain_picker import (
    UNAVAILABLE,
    GainPickerDialog,
    supported_gains,
)

V4_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
            22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
            43.9, 44.5, 48.0, 49.6]

SPEC = {'type': 'number', 'title': 'Tuner gain (dB)', 'default': 40.2,
        'x-widget': 'gain-picker'}
RTLSDR_VALUES = {'device_index': 0, 'frequency_khz': 3588.0, 'gain_db': 40.2}


def _fake_device(gains=None):
    device = MagicMock()
    device.valid_gains_db = list(V4_GAINS if gains is None else gains)
    return device


async def _wait_until(pilot, condition, description, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if condition():
            return
    raise AssertionError(f'timed out waiting for {description}')


class TestReadingTheGainsOffTheReceiver:
    def test_it_returns_the_receivers_own_list_sorted(self):
        with patch('buzz.sdr.open_device', return_value=_fake_device()) as open_device:
            gains = supported_gains(RTLSDR_VALUES)
        open_device.assert_called_once_with(0)
        assert gains == sorted(V4_GAINS)

    def test_it_opens_the_configured_receiver(self):
        with patch('buzz.sdr.open_device', return_value=_fake_device()) as open_device:
            supported_gains(dict(RTLSDR_VALUES, device_index=3))
        open_device.assert_called_once_with(3)

    def test_the_receiver_is_released_again(self):
        """Held open, it would stop the monitor and the sweep from opening it.  The
        dialog needs one answer, not a stream.
        """
        device = _fake_device()
        with patch('buzz.sdr.open_device', return_value=device):
            supported_gains(RTLSDR_VALUES)
        device.close.assert_called_once()

    def test_it_is_released_even_when_reading_the_list_fails(self):
        device = _fake_device()
        type(device).valid_gains_db = property(
            lambda self: (_ for _ in ()).throw(RuntimeError('the tuner stopped')))
        with patch('buzz.sdr.open_device', return_value=device):
            with pytest.raises(RuntimeError):
                supported_gains(RTLSDR_VALUES)
        device.close.assert_called_once()

    def test_missing_receiver_settings_fall_back_to_the_defaults(self):
        with patch('buzz.sdr.open_device', return_value=_fake_device()) as open_device:
            supported_gains({})
        open_device.assert_called_once_with(0)


class TestTheDialogOffersWhatTheTunerHas:
    def _open(self, tmp_path, current=40.2, device=None, error=None):
        """Drive the dialog until it has either a list or a message."""
        state = {}

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                patcher = (patch('buzz.sdr.open_device', side_effect=error) if error
                           else patch('buzz.sdr.open_device',
                                      return_value=device or _fake_device()))
                with patcher:
                    dialog = GainPickerDialog(SPEC, current, RTLSDR_VALUES)
                    app.push_screen(dialog)
                    await _wait_until(
                        pilot,
                        lambda: 'Reading' not in app.screen.query_one(
                            '#status', Static).content,
                        'the dialog to finish reading the receiver')
                    state['status'] = app.screen.query_one('#status', Static).content
                    options = app.screen.query_one('#value', OptionList)
                    state['labels'] = [str(options.get_option_at_index(i).prompt)
                                       for i in range(options.option_count)]
                    state['highlighted'] = options.highlighted
                    state['gains'] = list(dialog._gains)

        run(scenario())
        return state

    def test_every_step_the_tuner_reports_becomes_a_row(self, tmp_path):
        state = self._open(tmp_path)
        assert len(state['labels']) == len(V4_GAINS)
        assert state['gains'] == sorted(V4_GAINS)

    def test_nothing_outside_the_receivers_list_is_offered(self, tmp_path):
        """The whole point.  A row the hardware does not have would be the text box's
        problem all over again.
        """
        state = self._open(tmp_path)
        for label, gain in zip(state['labels'], sorted(V4_GAINS)):
            assert label.startswith(f'{gain:.1f} dB')

    def test_a_different_receiver_gives_a_different_list(self, tmp_path):
        """The list is read rather than assumed, so a tuner with three steps gets
        three rows and no invented ones.
        """
        state = self._open(tmp_path, device=_fake_device([0.0, 20.0, 40.0]))
        assert state['gains'] == [0.0, 20.0, 40.0]

    def test_the_current_value_starts_highlighted(self, tmp_path):
        """29 rows is long enough that finding where you already are otherwise means
        reading all of them.
        """
        state = self._open(tmp_path, current=32.8)
        assert state['labels'][state['highlighted']].startswith('32.8 dB')
        assert '(current)' in state['labels'][state['highlighted']]

    def test_a_stored_value_the_tuner_lacks_marks_what_it_would_snap_to(self, tmp_path):
        """41.0 is not a step.  The row marked current has to be the one the hardware
        would actually use, or the dialog repeats the lie it exists to stop.
        """
        state = self._open(tmp_path, current=41.0)
        assert state['labels'][state['highlighted']].startswith('40.2 dB')
        assert sum('(current)' in label for label in state['labels']) == 1

    def test_an_unset_value_highlights_the_first_row(self, tmp_path):
        state = self._open(tmp_path, current=None)
        assert state['highlighted'] == 0

    def test_the_status_says_how_many_steps_there_are(self, tmp_path):
        assert '29 steps' in self._open(tmp_path)['status']


class TestWhenTheReceiverCannotBeReached:
    """Not a dead end.  Somebody has to be able to set a gain before the device is
    working, so the caller offers a text box instead.
    """

    def test_it_reports_why_rather_than_raising(self, tmp_path):
        state = TestTheDialogOffersWhatTheTunerHas()._open(
            tmp_path, error=RuntimeError('no driver is bound to it'))
        assert 'no driver is bound to it' in state['status']

    def test_escape_hands_back_unavailable_rather_than_cancelled(self, tmp_path):
        """The two have to differ, because the caller does nothing for a cancel and
        opens the number box for an unreachable receiver.
        """
        result = {}

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.sdr.open_device',
                           side_effect=RuntimeError('nothing there')):
                    dialog = GainPickerDialog(SPEC, 40.2, RTLSDR_VALUES)
                    app.push_screen(dialog, lambda value: result.update(value=value))
                    await _wait_until(pilot, lambda: dialog._unreachable,
                                      'the dialog to report the receiver missing')
                    await pilot.press('escape')
                    await _wait_until(pilot, lambda: 'value' in result,
                                      'the dialog to dismiss')

        run(scenario())
        assert result['value'] is UNAVAILABLE
        assert result['value'] is not CANCELLED

    def test_escape_after_a_successful_read_is_an_ordinary_cancel(self, tmp_path):
        """Which must leave the stored gain alone rather than reopening a text box."""
        result = {}

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                with patch('buzz.sdr.open_device', return_value=_fake_device()):
                    dialog = GainPickerDialog(SPEC, 40.2, RTLSDR_VALUES)
                    app.push_screen(dialog, lambda value: result.update(value=value))
                    await _wait_until(pilot, lambda: bool(dialog._gains),
                                      'the list to fill')
                    await pilot.press('escape')
                    await _wait_until(pilot, lambda: 'value' in result,
                                      'the dialog to dismiss')

        run(scenario())
        assert result['value'] is CANCELLED

    def test_the_message_stays_on_screen_rather_than_vanishing(self, tmp_path):
        """open_device words these for whoever is standing at the radio, naming the
        driver to install.  A dialog that closed as it explained itself would throw
        that away and show a text box for no stated reason.
        """
        state = TestTheDialogOffersWhatTheTunerHas()._open(
            tmp_path, error=RuntimeError('no driver is bound to it'))
        assert 'no driver is bound to it' in state['status']
        assert 'Escape' in state['status']

    def test_a_receiver_reporting_no_gains_is_the_same_case(self, tmp_path):
        state = TestTheDialogOffersWhatTheTunerHas()._open(
            tmp_path, device=_fake_device([]))
        assert 'no gain settings' in state['status']

    def test_showing_the_list_after_dismissal_does_not_raise(self, tmp_path):
        """Escape cancels the worker at its next await, not mid-statement, so a read
        that finishes in that instant still resumes and reaches a screen whose widgets
        have gone.  Called directly, the way the device picker's own test does.
        """
        dialog = GainPickerDialog(SPEC, 40.2, RTLSDR_VALUES)
        dialog._gains = list(V4_GAINS)
        dialog._show_gains()
        dialog._give_up('no receiver')
        assert dialog._unreachable is True


class TestChangingTheGainCarriesTheLevelCalibration:
    """The offset converts audio level to dBm and the tuner gain is most of that
    conversion, so moving one without the other makes every reported level wrong by
    the difference.  That is the very thing calibrated_at_gain_db warns about at
    startup, and it should not need to.
    """

    def _change_gain_to(self, tmp_path, new_gain, **rtlsdr):
        from buzz.setup.screens.section_menu import SectionMenuScreen

        async def scenario():
            app = SetupApp(config_path=tmp_path / 'config.toml')
            async with app.run_test() as pilot:
                app.values['audio']['source'] = 'rtlsdr'
                app.values['rtlsdr'].update(rtlsdr)
                screen = SectionMenuScreen('rtlsdr')
                await app.push_screen(screen)
                await pilot.pause()
                screen.app.values['rtlsdr']['gain_db'] = new_gain
                screen._carry_the_calibration_to(rtlsdr['gain_db'], new_gain)
                return dict(app.values['rtlsdr'])

        return run(scenario())

    def test_a_calibrated_offset_moves_with_the_gain(self, tmp_path):
        """Dropping the gain by 7.4 dB means the same signal makes 7.4 dB less audio,
        so the offset has to rise by 7.4 to report the same dBm.
        """
        after = self._change_gain_to(tmp_path, 32.8, gain_db=40.2,
                                     calibrated_offset_db=-38.5,
                                     calibrated_at_gain_db=40.2)
        assert after['calibrated_offset_db'] == pytest.approx(-31.1)
        assert after['gain_db'] == 32.8

    def test_the_residual_from_the_calibration_is_kept(self, tmp_path):
        """The offset is the negative of the gain plus a residual for the rest of the
        chain.  Only the gain term moved, so the residual has to survive.
        """
        residual = -38.5 - (-40.2)
        after = self._change_gain_to(tmp_path, 25.4, gain_db=40.2,
                                     calibrated_offset_db=-38.5,
                                     calibrated_at_gain_db=40.2)
        assert after['calibrated_offset_db'] - (-25.4) == pytest.approx(residual)

    def test_the_calibration_mark_follows_so_startup_stays_quiet(self, tmp_path):
        """Leaving the old figure would warn about a difference this just corrected."""
        after = self._change_gain_to(tmp_path, 32.8, gain_db=40.2,
                                     calibrated_offset_db=-38.5,
                                     calibrated_at_gain_db=40.2)
        assert after['calibrated_at_gain_db'] == 32.8

    def test_an_uncalibrated_station_is_left_alone(self, tmp_path):
        """Nothing to carry: level_offset_db already derives the estimate from
        whatever gain_db says, so writing a figure here would turn an estimate into
        something that looks measured.
        """
        after = self._change_gain_to(tmp_path, 32.8, gain_db=40.2,
                                     calibrated_offset_db=None,
                                     calibrated_at_gain_db=None)
        assert after['calibrated_offset_db'] is None
        assert after['calibrated_at_gain_db'] is None

    def test_the_estimate_still_tracks_the_new_gain(self, tmp_path):
        """Which is why doing nothing above is right rather than merely harmless."""
        from buzz.config import RtlSdrConfig
        after = self._change_gain_to(tmp_path, 32.8, gain_db=40.2,
                                     calibrated_offset_db=None,
                                     calibrated_at_gain_db=None)
        assert RtlSdrConfig(**after).level_offset_db == -32.8

    def test_the_reported_level_is_unchanged_by_the_pair_moving(self, tmp_path):
        """The point of all of it, stated as the property.  A signal producing a given
        audio level at one gain produces less at a lower one, and the offset makes up
        the difference, so the dBm the operator reads stays put.
        """
        before_gain, after_gain = 40.2, 32.8
        after = self._change_gain_to(tmp_path, after_gain, gain_db=before_gain,
                                     calibrated_offset_db=-38.5,
                                     calibrated_at_gain_db=before_gain)
        audio_dbfs_before = -45.0
        audio_dbfs_after = audio_dbfs_before - (before_gain - after_gain)
        assert (audio_dbfs_after + after['calibrated_offset_db']
                == pytest.approx(audio_dbfs_before + -38.5))


class TestTheOffsetNudgesInTenths:
    """Half a dB could not reach the figure matching a gain of 40.2, which is exactly
    the case an operator calibrating a receiver is in: the arithmetic they correct is
    given in tenths and the control was not.
    """

    def _dialog(self, current=-32.0):
        from buzz.setup.screens.calibration import OffsetCalibrationDialog
        spec = {'type': 'number', 'title': 'Audio-to-RF offset (dB)', 'default': -32.0}
        dialog = OffsetCalibrationDialog.__new__(OffsetCalibrationDialog)
        dialog._spec = spec
        dialog._offset = current
        dialog._default_db = -32.0
        dialog._stream = None
        dialog.query_one = lambda *args, **kwargs: MagicMock()
        return dialog

    def test_up_and_down_move_a_tenth(self):
        dialog = self._dialog()
        dialog.action_increase()
        assert dialog._offset == pytest.approx(-31.9)
        dialog.action_decrease()
        dialog.action_decrease()
        assert dialog._offset == pytest.approx(-32.1)

    def test_page_keys_move_a_whole_decibel(self):
        """A tenth is slow across a wide correction, so the coarse step has its own
        keys rather than the fine one being coarsened to suit both.
        """
        dialog = self._dialog()
        dialog.action_increase_coarse()
        assert dialog._offset == pytest.approx(-31.0)
        dialog.action_decrease_coarse()
        dialog.action_decrease_coarse()
        assert dialog._offset == pytest.approx(-33.0)

    def test_a_tenth_reaches_the_figure_that_matches_a_tuner_step(self):
        """The reason for the change.  From -40.0, a gain of 40.2 needs -40.2, and
        half-decibel steps step straight over it.
        """
        dialog = self._dialog(current=-40.0)
        dialog.action_decrease()
        dialog.action_decrease()
        assert dialog._offset == pytest.approx(-40.2)

    def test_repeated_nudges_leave_no_floating_point_dirt(self):
        """A tenth is not exact in binary, and the result is written to a config file.
        Thirty of them reached -31.000000000000004 before the rounding.
        """
        dialog = self._dialog()
        for _ in range(30):
            dialog.action_increase()
        assert dialog._offset == -29.0
        assert repr(dialog._offset) == '-29.0'

    def test_reset_still_returns_to_the_default(self):
        dialog = self._dialog()
        for _ in range(7):
            dialog.action_increase()
        dialog.action_reset()
        assert dialog._offset == -32.0

    def test_a_live_stream_sees_every_nudge(self):
        """The offset is applied fresh on each block, so the meter moves as the
        operator holds the key down rather than after a restart.
        """
        dialog = self._dialog()
        dialog._stream = MagicMock()
        dialog.action_increase()
        assert dialog._stream.offset_db == pytest.approx(-31.9)


class TestLeavingTheMeterDoesNotFreezeTheUi:
    """Closing a receiver joins two threads with five second timeouts.  Run from the
    worker directly, that happened on the Textual event loop, so leaving the level
    meter stopped the whole program for as long as the close took.  A receiver whose
    cancel does not take effect holds it for the full ten seconds, which is
    indistinguishable from a hang.
    """

    class _SlowStream:
        """A stream whose close blocks, standing in for a receiver being released."""

        def __init__(self, seconds=0.6):
            self.seconds = seconds
            self.closed = False
            self.offset_db = -32.0

        def read(self, timeout=None):
            time.sleep(0.01)
            return -50.0

        def close(self):
            time.sleep(self.seconds)
            self.closed = True

    def test_the_loop_keeps_running_while_the_receiver_closes(self, tmp_path):
        """The property, measured rather than asserted: the event loop has to get a
        turn while the close is in flight.  Before the fix it got none at all.
        """
        from buzz.setup.screens.calibration import close_without_blocking_the_ui

        stream = self._SlowStream(seconds=0.6)
        ticks = []

        async def scenario():
            async def tick():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.02)

            ticker = asyncio.create_task(tick())
            await close_without_blocking_the_ui(stream)
            ticker.cancel()

        run(scenario())
        assert stream.closed
        assert len(ticks) > 5, (
            f'the event loop got {len(ticks)} turns during a 0.6 s close, so it was '
            'blocked rather than yielding')

    def test_a_cancelled_worker_still_closes_the_receiver(self, tmp_path):
        """The reason for the shield.  Escape cancels the worker, and a plain await in
        the finally would be cancelled with it, leaving the device open and unusable
        until the process ends.
        """
        from buzz.setup.screens.calibration import close_without_blocking_the_ui

        stream = self._SlowStream(seconds=0.3)

        async def scenario():
            task = asyncio.ensure_future(close_without_blocking_the_ui(stream))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # The shielded close is still running; give it room to finish.
            await asyncio.sleep(0.5)

        run(scenario())
        assert stream.closed, 'cancelling the worker left the receiver open'

    def test_the_meter_dialog_uses_it(self, tmp_path):
        """Wiring, since the helper being right helps nobody if the dialog still calls
        close() straight from the worker.
        """
        source = Path('lib/buzz/setup/screens/calibration.py').read_text(encoding='utf-8')
        assert 'stream.close()' not in source
        assert source.count('close_without_blocking_the_ui') == 3

    def test_the_sweep_dialog_closes_in_the_sweeps_own_thread(self, tmp_path):
        """Stronger than closing off the loop, and it replaced that.  A shielded close
        still left the event loop deciding whether it ran, and it did not always run:
        a sweep that failed after its last step left the receiver held, and the next
        attempt to open one came back as LIBUSB_ERROR_ACCESS.

        In the sweep's own thread the close cannot be skipped, because a thread
        asyncio.to_thread started runs to completion whatever happens to the task.

        The open belongs to that same thread, which is what the single to_thread call
        pins.  A second one meant a receiver opened by a thread the screen no longer
        awaited, and cancelling the dialog during the opening second leaked it.
        """
        source = Path(
            'lib/buzz/setup/screens/gain_calibration.py').read_text(encoding='utf-8')
        assert 'def _open_sweep_then_release(' in source
        assert 'source.close()' in source
        assert source.count('asyncio.to_thread(') == 1, (
            'the device has to be opened and released by one thread, so that nothing '
            'the event loop does can strand it')
        assert 'asyncio.shield' not in source, (
            'the shielded close left the loop deciding whether the device came back')

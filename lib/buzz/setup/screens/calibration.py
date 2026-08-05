"""Live S-meter dialogs for calibrating audio_rf_conversion_db.

Two dialogs, not one, because they answer two different questions with two
different controls.  CalibrationMeterDialog (opened from an action row on the
Audio section) is read-only: for a receiver whose own front panel has separate RF
and AF gain, that is what should move, not the stored offset, so this dialog has
nothing to adjust - only a live reading to watch while turning those two knobs.
OffsetCalibrationDialog (opened when audio_rf_conversion_db itself is chosen, from
the Station section) is for the opposite case: a receiver with no separate AF gain
to reach - an internal sound device, for instance - where the offset is the only
knob left.  Up and Down nudge it, Space resets it to the schema default, and Enter
confirms, all against the same live reading the other dialog only shows.

Both open LevelStream directly rather than going through AudioSampler, which also
opens a second input stream on the same device for the continuous-analysis
pipeline - a pipeline neither of these dialogs has any use for.
"""

import asyncio
from typing import Any

import sounddevice as sd
from textual import work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from buzz.config import AudioConfig, BuzzConfig, StationConfig
from buzz.sampler import LevelStream
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen
from buzz.smeter import SCALE_ROW, TENS_ROW, dbm_to_s_string, s_meter_bar

# 20 ms at 16 kHz - matches LevelStream's own default in AudioSampler.level_stream().
_METER_BLOCKSIZE = 320
_NUDGE_STEP_DB = 0.5


def _open_level_stream(audio_values: SectionValues, offset_db: float) -> LevelStream:
    """Open a LevelStream on `audio_values`, starting at `offset_db`.

    Raises whatever sd.query_devices() or sd.InputStream() raise - a device that no
    longer exists, or will not open - which both callers turn into an on-screen
    message rather than letting it crash the dialog.
    """
    config = BuzzConfig(audio=AudioConfig(**audio_values),
                        station=StationConfig(audio_rf_conversion_db=offset_db))
    device = sd.query_devices(config.audio.input_device_name, 'input')
    return LevelStream(config, device['index'], _METER_BLOCKSIZE)


def _meter_block(reading_line: str) -> str:
    """The scale rows plus one reading line, as a single block of text.

    All three lines have to share a left edge for the bar and the scale ticks
    above it to line up in the same columns - `text-align: center` on separate
    Static widgets centers each one independently around its own (different)
    width instead, which is what visibly misaligned them.  One Static, sized to
    its own content (`width: auto` in DEFAULT_CSS) and left-aligned within
    that, keeps all three lines pinned to column 0 - the widget itself is what
    gets centered in the dialog, not the text inside it.
    """
    return f'{TENS_ROW}\n{SCALE_ROW}\n{reading_line}'


def _format_reading(dbm: float) -> str:
    """A live meter line: the same bracketed bar, dBm figure, and S-string
    level_meter.py's console meter prints, so the two tools read alike.

    Fixed-width fields (`+7.1f`, a 6-wide S-string) matter here beyond matching
    level_meter.py's own formatting: _meter_block's parent Static is sized to
    its widest line, so a reading whose width changed from one update to the
    next - "-5.0" against "-15.0", say - would resize that widget and visibly
    shift the whole block sideways on every tick.
    """
    return f'[{s_meter_bar(dbm)}]  {dbm:+7.1f} dBm  {dbm_to_s_string(dbm):<6}'


class CalibrationMeterDialog(ScopeModalScreen[None]):
    """A live, read-only dBm/S-meter for matching a receiver's own S-meter.

    Adjust the RF and AF gain on the receiver, not this offset - see the module
    docstring.  Always dismisses with None: there is nothing here for
    section_menu.py to write back.
    """

    DEFAULT_CSS = """
    CalibrationMeterDialog {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
        align-horizontal: center;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #offset {
        text-align: center;
        padding-top: 1;
    }
    #hint {
        text-align: center;
        padding-bottom: 1;
    }
    #meter {
        width: auto;
        text-align: left;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    BINDINGS = [('escape', 'close', 'Close')]

    def __init__(self, audio_values: SectionValues, offset_db: float) -> None:
        super().__init__()
        self._audio_values = audio_values
        self._offset_db = offset_db

    def compose(self):
        yield Vertical(
            Static('Calibration meter', id='title'),
            Static(f'Offset: {self._offset_db:+.1f} dB (audio_rf_conversion_db, unchanged here)',
                  id='offset'),
            Static("Adjust the RF and AF gain on your receiver until this reading "
                  "matches your own S-meter.", id='hint'),
            Static(_meter_block('Starting...'), id='meter'),
            Button('Close', id='close'),
            id='dialog',
        )

    def on_mount(self) -> None:
        self._run_meter()

    @work
    async def _run_meter(self) -> None:
        try:
            stream = _open_level_stream(self._audio_values, self._offset_db)
        except Exception as exc:
            self._show(f'Could not open the input device: {exc}')
            return
        try:
            while True:
                dbm = await asyncio.to_thread(stream.read)
                self._show(_meter_block(_format_reading(dbm)))
        finally:
            stream.close()

    def _show(self, text: str) -> None:
        # Textual cancels this worker on unmount (Widget._on_unmount), but
        # cancellation arrives at the next await - not mid-statement - so a read
        # that completes in the same instant the dialog is dismissed can still
        # resume and reach here after '#meter' has already been torn down, even
        # while self.is_mounted still reads True: child widgets are removed from
        # the DOM before the screen's own mounted flag flips.  Confirmed by
        # dismissing mid-read in a test - catching NoMatches is what actually
        # stopped it from taking the whole app down, where checking is_mounted
        # first did not.
        try:
            self.query_one('#meter', Static).update(text)
        except NoMatches:
            pass

    def on_button_pressed(self, event) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class OffsetCalibrationDialog(ScopeModalScreen[Any]):
    """Live-adjust audio_rf_conversion_db while watching the reading it produces.

    For a receiver with no separate AF gain to reach, the offset itself is the
    only thing left to calibrate against.  Up/Down nudge it by _NUDGE_STEP_DB,
    Space resets it to the schema default, Enter confirms, and Escape cancels and
    discards every nudge made here - see CalibrationMeterDialog above for the
    opposite case, where the offset stays fixed and the receiver's own gain
    controls are what move.
    """

    DEFAULT_CSS = """
    OffsetCalibrationDialog {
        align: center middle;
    }
    #dialog {
        width: 60;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface;
        align-horizontal: center;
    }
    #title {
        text-align: center;
        text-style: bold;
    }
    #hint {
        text-align: center;
        padding: 1 0;
    }
    #offset {
        text-align: center;
        text-style: bold;
    }
    #meter {
        width: auto;
        text-align: left;
        padding-top: 1;
        padding-bottom: 1;
    }
    """
    BINDINGS = [
        ('up', 'increase', 'Increase'),
        ('down', 'decrease', 'Decrease'),
        ('space', 'reset', 'Reset to default'),
        ('enter', 'confirm', 'Confirm'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, spec: dict[str, Any], current: float, audio_values: SectionValues) -> None:
        super().__init__()
        self._spec = spec
        self._audio_values = audio_values
        self._offset = float(current)
        self._stream: LevelStream | None = None

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static('Up and Down nudge the offset, Space resets it to the default, '
                  'Enter confirms, Escape cancels.', id='hint'),
            Static(self._offset_text(), id='offset'),
            Static(_meter_block('Starting...'), id='meter'),
            id='dialog',
        )

    def _offset_text(self) -> str:
        return f'Offset: {self._offset:+.1f} dB'

    def on_mount(self) -> None:
        self._run_meter()

    @work
    async def _run_meter(self) -> None:
        try:
            self._stream = _open_level_stream(self._audio_values, self._offset)
        except Exception as exc:
            self._show(f'Could not open the input device: {exc}')
            return
        try:
            while True:
                dbm = await asyncio.to_thread(self._stream.read)
                self._show(_meter_block(_format_reading(dbm)))
        finally:
            self._stream.close()

    def _show(self, text: str) -> None:
        # See CalibrationMeterDialog's identical guard and comment above: a read
        # that completes the instant this dialog is dismissed can still resume and
        # touch a widget that is already gone.
        try:
            self.query_one('#meter', Static).update(text)
        except NoMatches:
            pass

    def action_increase(self) -> None:
        self._nudge(_NUDGE_STEP_DB)

    def action_decrease(self) -> None:
        self._nudge(-_NUDGE_STEP_DB)

    def action_reset(self) -> None:
        self._nudge(self._spec['default'] - self._offset)

    def _nudge(self, delta: float) -> None:
        self._offset += delta
        self.query_one('#offset', Static).update(self._offset_text())
        # See LevelStream's own docstring: a live write here is exactly what it is
        # for, and the running stream picks it up on its very next callback.
        if self._stream is not None:
            self._stream.offset_db = self._offset

    def action_confirm(self) -> None:
        self.dismiss(self._offset)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)

"""Live S-meter dialogs for calibrating the level offset.

Two dialogs, not one, because they answer two different questions with two
different controls.  CalibrationMeterDialog (opened from an action row on the
Audio section) is read-only: for a receiver whose own front panel has separate RF
and AF gain, that is what should move, not the stored offset, so this dialog has
nothing to adjust - only a live reading to watch while turning those two knobs.
OffsetCalibrationDialog (opened by choosing the offset itself, [station]
audio_rf_conversion_db for a sound card or [rtlsdr] calibrated_offset_db for a
receiver) is for the opposite case: a receiver with no separate AF gain
to reach - an internal sound device, for instance - where the offset is the only
knob left.  Up and Down nudge it, Space resets it to the schema default, and Enter
confirms, all against the same live reading the other dialog only shows.

Both open the stream directly rather than going through AudioSampler, which also
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

from buzz.config import AudioConfig, BuzzConfig, RtlSdrConfig, StationConfig
from buzz.dsp import SILENCE_DBFS
from buzz.sampler import LevelStream, SoundCardLevelStream
from buzz.setup.schema import SectionValues
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen
from buzz.setup.smeter import SCALE_ROW, TENS_ROW, dbm_to_s_string, s_meter_bar

# 20 ms at 16 kHz - matches the default in AudioSampler.level_stream().
_METER_BLOCKSIZE = 320

# IQ samples per receiver callback while metering.  8 ms at 256 kHz, against the
# pipeline's 16384, because the pool that has to drain before a change shows up is
# buf_num blocks deep whatever their size.  See docs-notebook/rtl-sdr-hardware.md.
_SDR_METER_BLOCK_SAMPLES = 2048
# How far Up and Down move the offset, and how far PageUp and PageDown move it.
#
# A tenth, because the tuner's gain steps are given to a tenth and the offset starts
# at the negative of one.  At half a dB the offset can never reach the figure that
# matches a gain of 40.2, which is the exact case an operator calibrating a receiver
# is in: the arithmetic they are correcting is in tenths and the control was not.
#
# A tenth is slow across a wide correction, so a whole dB has its own pair of keys
# rather than the fine step being coarsened to suit both.
_NUDGE_STEP_DB = 0.1
_COARSE_STEP_DB = 1.0


def level_offset_for(audio_values: SectionValues, station_values: SectionValues,
                     rtlsdr_values: SectionValues | None) -> float:
    """The offset the meter should apply, from whichever section owns it.

    A receiver keeps its own, because [station] audio_rf_conversion_db describes a
    radio feeding a sound card and means nothing to an SDR: the tuner gain is the
    conversion, and the two numbers are unrelated.  A receiver that has never been
    calibrated has no stored offset at all, and RtlSdrConfig.level_offset_db falls
    back to the negative of the tuner gain, which is the estimate the menu shows.

    Reading the station's figure for both is how the meter came to show -32.0 dB, the
    sound-card default, against a receiver whose own setting said -40.2.  The reading
    was right and the label on it was somebody else's.
    """
    if audio_values.get('source') != 'rtlsdr':
        return station_values['audio_rf_conversion_db']
    return RtlSdrConfig(**(rtlsdr_values or {})).level_offset_db


def _open_level_stream(audio_values: SectionValues, offset_db: float,
                       rtlsdr_values: SectionValues | None = None) -> LevelStream:
    """Open a level stream on whichever source the config selects.

    Both kinds produce the same reading through the same arithmetic, which is the
    point of LevelStream being split the way it is: an operator calibrating against
    this meter has to be calibrating against the figure the monitor itself reports.

    Raises whatever the underlying device raises - one that no longer exists, or will
    not open, or has no driver bound to it - which every caller turns into an
    on-screen message rather than letting it crash the dialog.
    """
    if audio_values.get('source') == 'rtlsdr':
        return _open_sdr_level_stream(rtlsdr_values or {}, offset_db)
    config = BuzzConfig(audio=AudioConfig(**audio_values),
                        station=StationConfig(audio_rf_conversion_db=offset_db))
    device = sd.query_devices(config.audio.input_device_name, 'input')
    return SoundCardLevelStream(config, device['index'], _METER_BLOCKSIZE)


def _open_sdr_level_stream(rtlsdr_values: SectionValues, offset_db: float) -> LevelStream:
    """Open the receiver and a converter for it, and meter what comes out.

    The imports sit inside the function for the reason open_live_source gives: a
    station using a sound card should never load pyrtlsdr, which resolves a symbol as
    it imports and so fails at import rather than at first call.

    A small block is used rather than the pipeline's, because a meter has no deadline
    to beat and a smaller block makes the transfer pool shallow, so the reading starts
    moving promptly instead of after most of a second.
    """
    from buzz.iq import IqToAudio
    from buzz.sdr import RtlSdrSource, SdrLevelStream
    from buzz.sdr_device import RtlSdrDevice

    settings = RtlSdrConfig(**rtlsdr_values)
    source = RtlSdrSource(
        RtlSdrDevice.open(
            settings.device_index,
            tuned_hz=settings.frequency_hz + settings.tuning_offset_hz,
            gain_db=settings.gain_db,
            iq_sample_rate=settings.iq_sample_rate),
        block_samples=_SDR_METER_BLOCK_SAMPLES)
    converter = IqToAudio(source.iq_sample_rate, settings.decimation,
                          settings.bandwidth_hz, settings.tuning_offset_hz,
                          settings.sideband)
    return SdrLevelStream(source, converter, offset_db)


async def close_without_blocking_the_ui(stream: LevelStream) -> None:
    """Close a level stream without stopping the event loop while it happens.

    Closing a receiver is slow and can be very slow.  RtlSdrSource.close cancels the
    async read and then joins the capture thread with a five second timeout, and
    SdrLevelStream adds a second join of its own for the thread that drains it.  Run
    straight from a worker, all of that happens on the Textual event loop, so leaving
    the meter freezes the entire program for as long as it takes.  A receiver whose
    cancel does not take effect holds it for the full ten seconds, which is
    indistinguishable from a hang and was reported as one.

    The shield is what makes it safe in a `finally`.  Escape cancels the worker, and
    a plain await here would be cancelled with it, leaving the receiver open and the
    device unusable until the process ends.  Shielded, the close finishes on its own
    thread whatever happens to the task that started it.
    """
    await asyncio.shield(asyncio.to_thread(stream.close))


def _stalled_reading() -> str:
    """The reading line when no audio has arrived for a second.

    This is built to the same width as _format_reading rather than written out,
    because _meter_block's parent Static is sized to its widest line: a shorter line
    would shrink the widget and shift the whole block sideways the moment the source
    stopped, which is the one instant the operator is trying to read it.
    """
    bar = ' ' * len(s_meter_bar(SILENCE_DBFS))
    tail = len(_format_reading(SILENCE_DBFS)) - len(bar) - 2   # minus the brackets
    return f'[{bar}]{"no audio":^{tail}}'


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
    """A live meter line: a bracketed bar, a dBm figure, and an S-string.

    Fixed-width fields (`+7.1f`, a 6-wide S-string) matter here: _meter_block's
    parent Static is sized to its widest line, so a reading whose width changed
    from one update to the next - "-5.0" against "-15.0", say - would resize
    that widget and visibly shift the whole block sideways on every tick.
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

    def __init__(self, audio_values: SectionValues, offset_db: float,
                 rtlsdr_values: SectionValues | None = None) -> None:
        super().__init__()
        self._audio_values = audio_values
        self._offset_db = offset_db
        self._rtlsdr_values = rtlsdr_values

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
            stream = _open_level_stream(self._audio_values, self._offset_db,
                                        self._rtlsdr_values)
        except Exception as exc:
            self._show(f'Could not open the input device: {exc}')
            return
        try:
            while True:
                dbm = await asyncio.to_thread(stream.read)
                reading = _stalled_reading() if dbm is None else _format_reading(dbm)
                self._show(_meter_block(reading))
        finally:
            await close_without_blocking_the_ui(stream)

    def _show(self, text: str) -> None:
        # Textual cancels this worker on unmount (Widget._on_unmount), but
        # cancellation arrives at the next await - not mid-statement - so a read
        # that completes in the same instant the dialog is dismissed can still
        # resume and reach here after '#meter' has already been torn down, even
        # while self.is_mounted still reads True: child widgets are removed from
        # the DOM before the screen's own mounted flag flips.  A test confirmed it
        # by dismissing mid-read: catching NoMatches is what actually stopped it
        # from taking the whole app down, where checking is_mounted first did not.
        try:
            self.query_one('#meter', Static).update(text)
        except NoMatches:
            pass

    def on_button_pressed(self, event) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class OffsetCalibrationDialog(ScopeModalScreen[Any]):
    """Live-adjust the level offset while watching the reading it produces.

    For a receiver with no separate AF gain to reach, the offset itself is the
    only thing left to calibrate against.  Up/Down nudge it by _NUDGE_STEP_DB and
    PageUp/PageDown by _COARSE_STEP_DB, Space resets it to the schema default, Enter
    confirms, and Escape cancels and
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
        ('up', 'increase', '+0.1 dB'),
        ('down', 'decrease', '-0.1 dB'),
        ('pageup', 'increase_coarse', '+1 dB'),
        ('pagedown', 'decrease_coarse', '-1 dB'),
        ('space', 'reset', 'Reset to default'),
        ('enter', 'confirm', 'Confirm'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, spec: dict[str, Any], current: float, audio_values: SectionValues,
                 rtlsdr_values: SectionValues | None = None,
                 default_db: float | None = None) -> None:
        super().__init__()
        self._spec = spec
        self._audio_values = audio_values
        self._rtlsdr_values = rtlsdr_values
        self._offset = float(current)
        # What Space resets to.  Passed in rather than read from the schema, because
        # the receiver's own field defaults to null: unset there means "estimate it
        # from the tuner gain", and resetting to nothing would be resetting to a
        # TypeError.  The caller knows which section this is and what its estimate is.
        self._default_db = float(spec['default'] if default_db is None else default_db)
        self._stream: LevelStream | None = None

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static('Up and Down move the offset by 0.1 dB, PageUp and PageDown by '
                  '1 dB, Space resets it to the default, Enter confirms, Escape '
                  'cancels.', id='hint'),
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
            self._stream = _open_level_stream(self._audio_values, self._offset,
                                              self._rtlsdr_values)
        except Exception as exc:
            self._show(f'Could not open the input device: {exc}')
            return
        try:
            while True:
                dbm = await asyncio.to_thread(self._stream.read)
                reading = _stalled_reading() if dbm is None else _format_reading(dbm)
                self._show(_meter_block(reading))
        finally:
            await close_without_blocking_the_ui(self._stream)

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

    def action_increase_coarse(self) -> None:
        self._nudge(_COARSE_STEP_DB)

    def action_decrease_coarse(self) -> None:
        self._nudge(-_COARSE_STEP_DB)

    def action_reset(self) -> None:
        self._nudge(self._default_db - self._offset)

    def _nudge(self, delta: float) -> None:
        # Rounded every time rather than at the end, because a tenth is not exact in
        # binary and twenty of them accumulate visible dirt: -32.0 nudged up ten times
        # reaches -31.000000000000004, which is what would be written to the file.
        self._offset = round(self._offset + delta, 2)
        self.query_one('#offset', Static).update(self._offset_text())
        # See LevelStream's own docstring: a live write here is exactly what it is
        # for, and the running stream picks it up on its very next callback.
        if self._stream is not None:
            self._stream.offset_db = self._offset

    def action_confirm(self) -> None:
        self.dismiss(self._offset)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)

"""Modal device picker for audio.input_device_name: probes every input device once.

Reuses device_setup.py's own probing.  The probe is a one-shot pass, not a
continuously refreshing meter: it already samples 100 ms of real audio per device to
measure its level, so a device already reads true the moment it appears, and R reruns
the whole pass for anyone who plugs something in after the dialog opens.
"""

import asyncio
from typing import Any

from textual import work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from buzz.setup.device_setup import DeviceInfo, current_device, enumerate_input_devices
from buzz.setup.screens.base import CANCELLED, ScopeModalScreen


class DevicePickerDialog(ScopeModalScreen[Any]):
    """Probe every input device once and let the user choose one by its level.

    A row that is not selectable - the wrong native rate, or the device would not
    open - still shows, with the reason baked into its bar by device_setup.py's own
    _reason_bar(), but is disabled so arrow-key navigation and Enter both skip it.
    Selecting a selectable row confirms it immediately, the same radio-list
    behavior EnumFieldDialog uses - there is no separate OK button.
    """

    DEFAULT_CSS = """
    DevicePickerDialog {
        align: center middle;
    }
    #dialog {
        width: 90%;
        max-width: 100;
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
    BINDINGS = [
        ('r', 'rescan', 'Rescan'),
        ('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, spec: dict[str, Any], current: Any, sample_rate: int) -> None:
        super().__init__()
        self._spec = spec
        self._current_name = current
        self._sample_rate = sample_rate
        self._devices: list[DeviceInfo] = []

    def compose(self):
        yield Vertical(
            Static(self._spec['title'], id='title'),
            Static('Scanning audio input devices...', id='status'),
            OptionList(id='value', classes='scope-options'),
            id='dialog',
        )

    def on_mount(self) -> None:
        self.action_rescan()

    @work(exclusive=True)
    async def action_rescan(self) -> None:
        status = self.query_one('#status', Static)
        status.update('Scanning audio input devices...')
        self.query_one('#value', OptionList).clear_options()
        # enumerate_input_devices() blocks on real device I/O for up to 100 ms per
        # device, so this runs off the event loop rather than freezing the dialog
        # while it scans - see "push, don't poll" in CLAUDE.md, the same reasoning
        # applied here to a UI thread instead of a publisher's.
        self._devices = await asyncio.to_thread(enumerate_input_devices, self._sample_rate)
        self._show_devices()

    def _show_devices(self) -> None:
        # Escape can dismiss this dialog while the probe above is still running -
        # Textual then cancels this worker, but only at the next await, so a probe
        # that finishes in that same instant still resumes and reaches here after
        # the screen's own widgets are already gone.  See
        # test_device_picker_show_devices_survives_being_called_after_dismissal in
        # test_setup_app.py, which calls this method directly after dismissal
        # rather than depending on the real race actually occurring in a timed
        # test - it is far narrower here, with one probe instead of
        # CalibrationMeterDialog's continuous loop.
        try:
            status = self.query_one('#status', Static)
            option_list = self.query_one('#value', OptionList)
        except NoMatches:
            return
        if not self._devices:
            status.update('No input devices found.  Press R to rescan.')
            return
        current = current_device(self._devices, self._current_name)
        selectable = [d for d in self._devices if d.selectable]
        status.update('Choose a device.  Press R to rescan.  Compatible devices show the audio amplitude found during '
                      'the last scan.  Non-compatible devices show the reason for being non-compatible.' if selectable
                      else 'No compatible input device found.  Press R to rescan.')
        options = [Option(f'{d.bar}  {d.display_name}',
                          id=str(d.real_index), disabled=not d.selectable)
                  for d in self._devices]
        option_list.add_options(options)
        # Only ever pre-highlight a selectable row - current_device() matches by
        # name alone, so the configured device can still come back disabled if the
        # sample rate changed since it was chosen.
        if current is not None and current.selectable:
            option_list.highlighted = self._devices.index(current)
        elif selectable:
            option_list.highlighted = self._devices.index(selectable[0])

    def on_option_list_option_selected(self, event) -> None:
        real_index = int(event.option.id)
        device = next(d for d in self._devices if d.real_index == real_index)
        self.dismiss(device.name)

    def action_cancel(self) -> None:
        self.dismiss(CANCELLED)

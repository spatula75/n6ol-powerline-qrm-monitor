"""
Device enumeration for the setup program's device-picker dialog.

Probes all PortAudio input devices in parallel: checks sample-rate/format
compatibility, then samples 100 ms of audio to measure ambient amplitude.
Results are displayed as a logarithmic ASCII level bar so the user can
visually identify which device is carrying the RF signal.

On Windows the same physical input is often listed three times, once each
under MME, DirectSound, and WASAPI. The deduplication logic in _best_api_devices()
collapses these to a single entry, preferring WASAPI, which routes through the
Windows audio engine and respects system input level controls. On Linux, macOS,
and BSD each device typically appears once, so the deduplication is a no-op.

Every DeviceInfo carries two names, for two different jobs.  name is "device, host
API" - PortAudio's own device name and host API name, joined the same way
sounddevice's own device lookup expects, since that string is what gets saved to
input_device_name and matched against it at every later startup.  The host API is
part of what makes it distinct: the same device can appear once per API, so the
name alone would not always tell two entries apart.  display_name is just the
device name, with no host API at all, because the setup program's device picker is
showing one row per already-deduplicated physical device, so the
host API adds nothing there but width - "Line In (Realtek(R) Audio), Windows
DirectSound" wraps a device picker sized for an 80-column terminal; "Line In
(Realtek(R) Audio)" does not.
"""

import concurrent.futures
from dataclasses import dataclass
from math import log10
from typing import Any

import numpy as np
import sounddevice as sd

from buzz.constants import DB_PER_S_UNIT, FULL_SCALE_COUNTS

# Segments span 1 LSB to full scale, each one DB_PER_S_UNIT wide - matching what the
# waterfall's S-meters call an S-unit, so the two displays read the same way. This is
# derived rather than a literal count, so a change to either shared constant moves it
# with this one instead of silently drifting, which is exactly how this bar ended up
# at 4.75 dB/segment while the S-meters stayed at 6.
_BAR_PAD = 2
_BAR_WIDTH = round(20 * log10(FULL_SCALE_COUNTS) / DB_PER_S_UNIT)
_FILL = '▒'
_EMPTY = ' '
_LEFT = '▕'
_RIGHT = '▏'

# Keys are the 'name' field from sd.query_hostapis()[n], which comes from PortAudio.
# If sounddevice ever changes how it reports API names these will silently stop matching.
_API_PRIORITY = {
    'Windows WASAPI': 2,
    'Windows DirectSound': 1,
    'MME': 0,
}


@dataclass
class DeviceInfo:
    real_index: int       # PortAudio device index, as used by sounddevice
    name: str             # "device, host API", saved to input_device_name and matched against it
    display_name: str     # device name alone, for the picker - see module docstring
    selectable: bool      # False if the device doesn't support the configured sample rate
    amplitude: float      # mean absolute amplitude from the 100 ms probe recording
    bar: str              # pre-rendered level or reason bar, always _BAR_WIDTH + _BAR_PAD chars wide
    display_index: int = 0  # 1-based index, assigned in enumeration order


def _amplitude_bar(amplitude: float) -> str:
    """Render amplitude as a logarithmic bar: empty at 1 LSB, full at full scale,
    each filled char spanning DB_PER_S_UNIT dB - see _BAR_WIDTH."""
    n = min(_BAR_WIDTH, int(_BAR_WIDTH * log10(max(1.0, amplitude)) / log10(FULL_SCALE_COUNTS)))
    return _LEFT + (_FILL * n) + (_EMPTY * (_BAR_WIDTH - n)) + _RIGHT


def _reason_bar(text: str) -> str:
    return f'({text[:_BAR_WIDTH]})'.center(_BAR_WIDTH + _BAR_PAD)


def enumerate_input_devices(sample_rate: int) -> list[DeviceInfo]:
    """Probe deduplicated input devices in parallel, one entry per physical device.

    Public: the setup program's device-picker dialog uses this directly.
    """
    candidates = _best_api_devices(sample_rate)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_probe, idx, dev, sample_rate)
                   for idx, dev in candidates]
        probed = [f.result() for f in futures]

    probed.sort(key=lambda d: (d.selectable, d.amplitude), reverse=True)

    for display_idx, entry in enumerate(probed, start=1):
        entry.display_index = display_idx
    return probed


def current_device(devices: list[DeviceInfo], current_name: str | None) -> DeviceInfo | None:
    """The entry matching the configured device name, if it is still present.

    Matched by name rather than by a stored index, because that is how the running
    program resolves the device too: an index is only true until Windows next
    reassigns audio hardware, so one written into the config last month may now point
    at something else entirely.  A device that has been unplugged simply does not
    match, and the picker then offers no current selection, which is honest.

    Public for the same reason as enumerate_input_devices() above: the setup
    program's device-picker dialog uses this to mark the current device.
    """
    if not current_name:
        return None
    return next((d for d in devices if d.name == current_name), None)


def _best_api_devices(sample_rate: int) -> list[tuple[int, dict[str, Any]]]:
    """Return one (real_index, device_dict) per physical device.

    When the same device appears under multiple host APIs, this prefers the
    highest-priority API that is actually compatible with sample_rate. It only
    falls back to an incompatible variant if no compatible one exists, so the
    device still appears in the list with a reason rather than silently
    disappearing. Ordering follows the first appearance of each device name in
    PortAudio's enumeration.
    """
    hostapis = sd.query_hostapis()
    all_input = [(i, d) for i, d in enumerate(sd.query_devices())
                 if d['max_input_channels'] > 0]

    # Group all variants by the first 31 chars of the device name.
    # Windows WAVE API (MME) caps names at 31 characters (MAXPNAMELEN=32 including
    # null terminator), so an MME entry and its DirectSound/WASAPI counterpart can
    # have different dict keys even though they refer to the same physical device.
    # Truncating to 31 chars normalizes both sides of that pair.
    groups: dict[str, list[tuple[int, dict, int, int]]] = {}
    for order, (real_idx, dev) in enumerate(all_input):
        key = dev['name'][:31]
        host = hostapis[dev['hostapi']]['name']
        priority = _API_PRIORITY.get(host, 0)
        groups.setdefault(key, []).append((real_idx, dev, priority, order))

    chosen = []
    for variants in groups.values():
        compatible = [v for v in variants if _supports_rate(v[0], sample_rate)]
        pool = compatible if compatible else variants
        best = max(pool, key=lambda v: v[2])  # highest priority in pool
        chosen.append(best)

    chosen.sort(key=lambda v: v[3])  # restore first-appearance order
    return [(real_idx, dev) for real_idx, dev, _, _ in chosen]


def _supports_rate(device_index: int, sample_rate: int) -> bool:
    try:
        sd.check_input_settings(device=device_index, channels=1,
                                dtype='int16', samplerate=sample_rate)
        return True
    except sd.PortAudioError:
        return False


def _probe(real_index: int, device: dict[str, Any], sample_rate: int) -> DeviceInfo:
    host_name = sd.query_hostapis(device['hostapi'])['name']
    name = f"{device['name']}, {host_name}"
    display_name = device['name']

    try:
        sd.check_input_settings(device=real_index, channels=1,
                                dtype='int16', samplerate=sample_rate)
    except sd.PortAudioError:
        native_hz = int(device['default_samplerate'])
        return DeviceInfo(real_index=real_index, name=name, display_name=display_name,
                          selectable=False, amplitude=0.0,
                          bar=_reason_bar(f'needs {native_hz} Hz'))

    try:
        rec = sd.rec(int(sample_rate * 0.1), samplerate=sample_rate,
                     channels=1, blocking=True, dtype='int16', device=real_index)
        amplitude = float(np.mean(np.abs(rec.astype(np.int32))))
        return DeviceInfo(real_index=real_index, name=name, display_name=display_name,
                          selectable=True, amplitude=amplitude, bar=_amplitude_bar(amplitude))
    except Exception:
        return DeviceInfo(real_index=real_index, name=name, display_name=display_name,
                          selectable=False, amplitude=0.0, bar=_reason_bar('could not open'))

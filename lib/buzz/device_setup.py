"""
Device enumeration and interactive selection for the audio input configurator.

Probes all PortAudio input devices in parallel: checks sample-rate/format
compatibility, then samples 100 ms of audio to measure ambient amplitude.
Results are displayed as a logarithmic ASCII level bar so the user can
visually identify which device is carrying the RF signal.

On Windows the same physical input is often listed three times — once each
under MME, DirectSound, and WASAPI.  The deduplication logic in _best_api_devices()
collapses these to a single entry, preferring WASAPI (which routes through the
Windows audio engine and respects system input level controls).  On Linux, macOS,
and BSD each device typically appears once, so the deduplication is a no-op.
"""

import concurrent.futures
from dataclasses import dataclass
from math import log10
from typing import Any

import numpy as np
import sounddevice as sd

_BAR_WIDTH = 19
_FILL = '█'
_EMPTY = '░'

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
    name: str             # display name: "device, host API" with "Windows " stripped
    selectable: bool      # False if the device doesn't support the configured sample rate
    amplitude: float      # mean absolute amplitude from the 100 ms probe recording
    bar: str              # pre-rendered level or reason bar, always _BAR_WIDTH chars wide
    display_index: int = 0  # 1-based index shown to the user in the selection table


def _amplitude_bar(amplitude: float) -> str:
    """Render amplitude as a logarithmic bar: empty at 1 LSB, full at 16-bit
    full scale (32768), so each filled char spans an equal number of dB."""
    n = min(_BAR_WIDTH, int(_BAR_WIDTH * log10(max(1.0, amplitude)) / log10(32768.0)))
    return _FILL * n + _EMPTY * (_BAR_WIDTH - n)


def _reason_bar(text: str) -> str:
    return text[:_BAR_WIDTH].center(_BAR_WIDTH)


def _best_api_devices(sample_rate: int) -> list[tuple[int, dict[str, Any]]]:
    """Return one (real_index, device_dict) per physical device.

    When the same device appears under multiple host APIs, prefer the
    highest-priority API that is actually compatible with sample_rate.
    Only fall back to an incompatible variant if no compatible one exists
    (so the device still appears in the list with a reason rather than
    silently disappearing).  Ordering follows the first appearance of each
    device name in PortAudio's enumeration.
    """
    hostapis = sd.query_hostapis()
    all_input = [(i, d) for i, d in enumerate(sd.query_devices())
                 if d['max_input_channels'] > 0]

    # Group all variants by the first 31 chars of the device name.
    # Windows WAVE API (MME) caps names at 31 characters (MAXPNAMELEN=32 including
    # null terminator), so an MME entry and its DirectSound/WASAPI counterpart can
    # have different dict keys even though they refer to the same physical device.
    # Truncating to 31 chars normalises both sides of that pair.
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
    host_name = sd.query_hostapis(device['hostapi'])['name'].replace('Windows ', '')
    name = f"{device['name']}, {host_name}"

    try:
        sd.check_input_settings(device=real_index, channels=1,
                                dtype='int16', samplerate=sample_rate)
    except sd.PortAudioError:
        native_hz = int(device['default_samplerate'])
        return DeviceInfo(real_index=real_index, name=name, selectable=False,
                          amplitude=0.0, bar=_reason_bar(f'needs {native_hz} Hz'))

    try:
        rec = sd.rec(int(sample_rate * 0.1), samplerate=sample_rate,
                     channels=1, blocking=True, dtype='int16', device=real_index)
        amplitude = float(np.mean(np.abs(rec.astype(np.int32))))
        return DeviceInfo(real_index=real_index, name=name, selectable=True,
                          amplitude=amplitude, bar=_amplitude_bar(amplitude))
    except Exception:
        return DeviceInfo(real_index=real_index, name=name, selectable=False,
                          amplitude=0.0, bar=_reason_bar('could not open'))


def _enumerate_input_devices(sample_rate: int) -> list[DeviceInfo]:
    """Probe deduplicated input devices in parallel, one entry per physical device."""
    candidates = _best_api_devices(sample_rate)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_probe, idx, dev, sample_rate)
                   for idx, dev in candidates]
        probed = [f.result() for f in futures]

    probed.sort(key=lambda d: (d.selectable, d.amplitude), reverse=True)

    for display_idx, entry in enumerate(probed, start=1):
        entry.display_index = display_idx
    return probed


def _print_device_table(devices: list[DeviceInfo], current_real_index: int | None = None) -> None:
    print()
    print('  Audio level is logarithmic — each █ ≈ 6 dB above silence (16-bit)')
    print()
    idx_w = len(str(len(devices)))
    print(f"  {'#':>{idx_w}}  [{'LEVEL'.center(_BAR_WIDTH)}]  DEVICE")
    print(f"  {'-' * idx_w}  {'-' * (_BAR_WIDTH + 2)}  {'-' * 50}")
    for dev in devices:
        marker = '  ← current' if dev.real_index == current_real_index else ''
        print(f"  {dev.display_index:>{idx_w}}  [{dev.bar}]  {dev.name}{marker}")
    print()


def _build_selection_prompt(devices: list[DeviceInfo], selectable: list[DeviceInfo],
                            current_real_index: int | None) -> tuple[str, set[int]]:
    """Build the "Select device (...)" prompt text and the set of valid numbers."""
    valid_nums = sorted(d.display_index for d in selectable)
    current_display = next(
        (d.display_index for d in devices if d.real_index == current_real_index),
        None,
    )
    default_str = f', or Enter to keep current [{current_display}]' if current_display else ''
    nums_str = ', '.join(str(n) for n in valid_nums)
    prompt = f'Select device ({nums_str}{default_str}): '
    return prompt, set(valid_nums)


def _prompt_for_device_number(prompt: str, valid_set: set[int], selectable: list[DeviceInfo],
                              current_real_index: int | None) -> int:
    """Read device numbers from stdin until the user picks a valid one.

    Enter alone keeps current_real_index if one was given; anything else that
    isn't a valid number re-prompts rather than failing.
    """
    nums_str = ', '.join(str(n) for n in sorted(valid_set))
    while True:
        raw = input(prompt).strip()
        if not raw and current_real_index is not None:
            return current_real_index
        try:
            n = int(raw)
            if n in valid_set:
                return next(d.real_index for d in selectable if d.display_index == n)
        except ValueError:
            pass
        print(f'  Please enter one of: {nums_str}')


def select_device(sample_rate: int, current_real_index: int | None = None) -> int | None:
    """Display the device table and prompt for selection.

    Returns the real PortAudio index of the chosen device, or None if no
    compatible devices are found.  Returns current_real_index unchanged if
    the user presses Enter without a selection.
    """
    print('Scanning audio input devices...')
    devices = _enumerate_input_devices(sample_rate)
    _print_device_table(devices, current_real_index)

    selectable = [d for d in devices if d.selectable]
    if not selectable:
        print('No compatible input devices found.')
        return None

    prompt, valid_set = _build_selection_prompt(devices, selectable, current_real_index)
    return _prompt_for_device_number(prompt, valid_set, selectable, current_real_index)

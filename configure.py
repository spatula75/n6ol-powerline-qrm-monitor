"""
Audio device configuration tool.

Lists all available input devices with a live amplitude level bar so you can
identify which device is connected to your radio.  Saves your selection —
along with the full configuration — to ~/.buzz/config.toml.

Usage:
    python configure.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))

import sounddevice as sd
import tomli_w

from buzz.config import BuzzConfig, CONFIG_PATH
from buzz.device_setup import select_device


def _section_dict(obj) -> dict:
    return {k: v for k, v in asdict(obj).items() if v is not None}


def main():
    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()

    new_index = select_device(config.audio.sample_rate, current_real_index=config.audio.device_index)
    if new_index is None:
        print('No device selected.')
        return

    device = sd.query_devices(new_index)
    hostapis = sd.query_hostapis()
    full_name = f"{device['name']}, {hostapis[device['hostapi']]['name']}"

    config.audio.device_index = new_index
    config.audio.input_device_name = full_name

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'audio':   _section_dict(config.audio),
        'station': _section_dict(config.station),
        'weather': _section_dict(config.weather),
        'server':  _section_dict(config.server),
    }
    with open(CONFIG_PATH, 'wb') as f:
        tomli_w.dump(data, f)

    print(f'\nDevice set to: {full_name} (index {new_index})')
    print(f'Config saved to: {CONFIG_PATH}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled.')

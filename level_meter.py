"""
Real-time RF level meter for calibrating receiver gain settings.

Continuously reads from the configured audio input via a persistent stream and
displays a live text S-meter. Use this to match your radio's S-meter reading
against what the monitor will report before starting a monitoring run.

Usage:
    python level_meter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.constants import DB_PER_S_UNIT, S9_DBM
from buzz.sampler import AudioSampler
from buzz.smeter import S1_DBM, SCALE_ROW, TENS_ROW, dbm_to_s_string, s_meter_bar


def main() -> None:
    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    sampler = AudioSampler(config)

    print(f'Device : {config.audio.input_device_name}')
    print(f'Offset : {config.station.audio_rf_conversion_db:+.1f} dB  '
          f'(audio_rf_conversion_db from config)')
    print(f'Scale  : S1 = {S1_DBM:g} dBm  -  S9 = {S9_DBM:g} dBm  '
          f'(each S-unit = {DB_PER_S_UNIT:g} dB)')
    print()
    print('Adjust RF and AF gain until the reading here matches your S-meter.')
    print('Press Ctrl+C to exit.')
    print()
    print(TENS_ROW)
    print(SCALE_ROW)

    first = True
    try:
        with sampler.level_stream() as stream:
            while True:
                dbm = stream.read()
                s_str = dbm_to_s_string(dbm)
                line = f'[{s_meter_bar(dbm)}]  {dbm:+7.1f} dBm  {s_str:<6}'

                if not first:
                    sys.stdout.write('\033[F')  # up to start of previous line
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                first = False

    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()

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

_S1_DBM = S9_DBM - 8 * DB_PER_S_UNIT      # -121 dBm (S1)
_S9P60_DBM = S9_DBM + 60                  # -13 dBm  (S9+60)

# Bar layout mirrors a standard analog S-meter:
#   S1-S9  = 9 chars (1 char per S-unit, each unit = 6 dB)
#   S9-+60 = 12 chars (3 ticks between each labeled mark, each mark = 20 dB)
# Total bar width = 21 chars.
_S1_TO_S9_CHARS = 9
_S9_TO_S9P60_CHARS = 12
_BAR_WIDTH = _S1_TO_S9_CHARS + _S9_TO_S9P60_CHARS  # = 21

# Scale decoration rows printed once above the live bar.
# Tick positions for S2/S4/S6/S8 between labeled odd S-units, and three ticks
# between each +20/+40/+60 section.  Double-digit labels have their tens digit
# on the top row and units digit on the scale row.
#
#  pos: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# mark: 1  |  3  |  5  |  7  |  9  |  |  |  0  |  |  |  0  |  |  |  0
# tens:                                           2           4           6
_SCALE_MARKS = {
    0: '1',  1: '.',  2: '3',  3: '.',  4: '5',  5: '.',  6: '7',  7: '.',  8: '9',
    9: '.',  10: '.', 11: '.', 12: '0',
    13: '.', 14: '.', 15: '.', 16: '0',
    17: '.', 18: '.', 19: '.', 20: '0',
}
_TENS_MARKS = {12: '2', 16: '4', 20: '6'}

_SCALE_ROW = '[' + ''.join(_SCALE_MARKS.get(i, ' ') for i in range(_BAR_WIDTH)) + ']'
_TENS_ROW  = ' ' + ''.join(_TENS_MARKS.get(i, ' ') for i in range(_BAR_WIDTH)) + ' '


def _dbm_to_s_string(dbm: float) -> str:
    if dbm >= S9_DBM:
        over = dbm - S9_DBM
        if over < 10:
            return 'S9'
        if over < 20:
            return 'S9+10'
        if over < 30:
            return 'S9+20'
        if over < 40:
            return 'S9+30'
        if over < 50:
            return 'S9+40'
        return 'S9+60'
    return f'S{max(1, round(9 + (dbm - S9_DBM) / DB_PER_S_UNIT))}'


def _bar(dbm: float) -> str:
    # Piecewise fill matching the two-section scale layout:
    #   S1-S9  : 9 positions, one per S-unit (DB_PER_S_UNIT dB).  S_n lights exactly
    #            n positions.
    #   S9-+60 : 12 positions, one per 5 dB.  Each labeled mark (+20/+40/+60) lights
    #            exactly 4 more positions than the previous labeled mark.
    if dbm <= S9_DBM:
        filled = min(_S1_TO_S9_CHARS, max(0, int((dbm - _S1_DBM) // DB_PER_S_UNIT) + 1))
    else:
        fraction = min(1.0, (dbm - S9_DBM) / (_S9P60_DBM - S9_DBM))
        filled = _S1_TO_S9_CHARS + int(fraction * _S9_TO_S9P60_CHARS)
    filled = max(0, min(filled, _BAR_WIDTH))
    return '█' * filled + '░' * (_BAR_WIDTH - filled)


def main() -> None:
    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    sampler = AudioSampler(config)

    print(f'Device : {config.audio.input_device_name}')
    print(f'Offset : {config.station.audio_rf_conversion_db:+.1f} dB  '
          f'(audio_rf_conversion_db from config)')
    print(f'Scale  : S1 = {_S1_DBM:g} dBm  -  S9 = {S9_DBM:g} dBm  '
          f'(each S-unit = {DB_PER_S_UNIT:g} dB)')
    print()
    print('Adjust RF and AF gain until the reading here matches your S-meter.')
    print('Press Ctrl+C to exit.')
    print()
    print(_TENS_ROW)
    print(_SCALE_ROW)

    first = True
    try:
        with sampler.level_stream() as stream:
            while True:
                dbm = stream.read()
                s_str = _dbm_to_s_string(dbm)
                line = f'[{_bar(dbm)}]  {dbm:+7.1f} dBm  {s_str:<6}'

                if not first:
                    sys.stdout.write('\033[F')  # up to start of previous line
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                first = False

    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()

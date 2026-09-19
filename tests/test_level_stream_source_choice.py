"""Tests that the setup program's meters open the source the config actually names.

The dialogs used to open a sound card unconditionally.  An SDR station reaching the
level meter would have opened whatever sound card was named in [audio], metered it,
and let the operator calibrate an offset against a device the monitor was never going
to use.  Nothing about the reading would have looked wrong.
"""
from unittest.mock import MagicMock, patch

import pytest

from buzz.setup.screens.calibration import _open_level_stream
from buzz.sdr_device import RtlSdrDevice

AUDIO_SOUNDCARD = {'source': 'soundcard', 'sample_rate': 16000, 'pulse_rate': 120,
                   'input_device_name': 'Test'}
AUDIO_RTLSDR = dict(AUDIO_SOUNDCARD, source='rtlsdr')
RTLSDR_VALUES = {'frequency_khz': 3588.0, 'gain_db': 40.2, 'device_index': 0,
                 'iq_sample_rate': 256_000, 'decimation': 16, 'bandwidth_khz': 4.0,
                 'tuning_offset_khz': 50.0, 'sideband': 'upper'}


class TestTheMeterOpensTheConfiguredSource:
    def test_a_sound_card_station_gets_a_sound_card_stream(self):
        with patch('buzz.setup.screens.calibration.sd.query_devices',
                   return_value={'index': 3}) as query, \
             patch('buzz.setup.screens.calibration.SoundCardLevelStream') as stream:
            result = _open_level_stream(AUDIO_SOUNDCARD, -32.0)
        assert result is stream.return_value
        query.assert_called_once()
        assert stream.call_args[0][1] == 3, 'the device index query_devices returned'

    def test_a_receiver_station_gets_a_receiver_stream(self):
        """The bug this file exists for: no sound card is opened at all."""
        with patch.object(RtlSdrDevice, 'open') as open_device, \
             patch('buzz.sdr.SdrSource') as source, \
             patch('buzz.sdr.SdrLevelStream') as stream, \
             patch('buzz.iq.IqToAudio') as converter, \
             patch('buzz.setup.screens.calibration.sd.query_devices') as query:
            result = _open_level_stream(AUDIO_RTLSDR, -40.2, RTLSDR_VALUES)
        assert result is stream.return_value
        query.assert_not_called()
        assert open_device.call_args.args[0] == 0
        assert source.called and converter.called

    def test_the_receiver_is_opened_with_a_small_block(self):
        """A meter has no deadline, and a smaller block makes the transfer pool
        shallow, so the reading starts moving promptly instead of after most of a
        second.
        """
        with patch.object(RtlSdrDevice, 'open'), \
             patch('buzz.sdr.SdrSource') as source, \
             patch('buzz.sdr.SdrLevelStream'), \
             patch('buzz.iq.IqToAudio'):
            _open_level_stream(AUDIO_RTLSDR, -40.2, RTLSDR_VALUES)
        assert source.call_args.kwargs['block_samples'] == 2048

    def test_the_offset_reaches_the_stream_whichever_source_it_is(self):
        """The offset is what the operator is calibrating, so a source that dropped it
        would meter correctly and calibrate nothing.
        """
        with patch('buzz.setup.screens.calibration.sd.query_devices',
                   return_value={'index': 0}), \
             patch('buzz.setup.screens.calibration.SoundCardLevelStream') as card:
            _open_level_stream(AUDIO_SOUNDCARD, -12.5)
        assert card.call_args[0][0].station.audio_rf_conversion_db == -12.5

        with patch.object(RtlSdrDevice, 'open'), patch('buzz.sdr.SdrSource'), \
             patch('buzz.sdr.SdrLevelStream') as sdr, patch('buzz.iq.IqToAudio'):
            _open_level_stream(AUDIO_RTLSDR, -12.5, RTLSDR_VALUES)
        assert sdr.call_args[0][2] == -12.5

    def test_a_receiver_station_with_no_receiver_settings_still_opens_defaults(self):
        """values.get('rtlsdr') is None for a config written before the section
        existed.  Falling back to the dataclass defaults beats raising a KeyError at
        an operator who only wanted to look at a meter.
        """
        with patch.object(RtlSdrDevice, 'open') as open_device, \
             patch('buzz.sdr.SdrSource'), patch('buzz.sdr.SdrLevelStream'), \
             patch('buzz.iq.IqToAudio'):
            _open_level_stream(AUDIO_RTLSDR, -40.2, None)
        assert open_device.call_args.args[0] == 0

    def test_an_unknown_source_is_treated_as_a_sound_card(self):
        """Matching what the rest of the setup program does with a value the schema
        does not allow.  main.py is the layer that refuses one outright, because it is
        the layer that would otherwise log a day of the wrong input.
        """
        with patch('buzz.setup.screens.calibration.sd.query_devices',
                   return_value={'index': 0}), \
             patch('buzz.setup.screens.calibration.SoundCardLevelStream') as card:
            _open_level_stream(dict(AUDIO_SOUNDCARD, source='nonsense'), -32.0)
        assert card.called


class TestBothStreamsShareTheirArithmetic:
    """The drift pin for the split.  An operator calibrates against whichever meter
    their station gives them, and the monitor then reports levels through the same
    code, so the two must not be able to disagree.
    """

    def test_neither_subclass_overrides_how_a_block_becomes_a_reading(self):
        from buzz.sampler import LevelStream, SoundCardLevelStream
        from buzz.sdr import SdrLevelStream
        shared = ('_on_block', 'read', 'dc_ema_alpha')
        for subclass in (SoundCardLevelStream, SdrLevelStream):
            for name in shared:
                assert getattr(subclass, name) is getattr(LevelStream, name), (
                    f'{subclass.__name__} overrides {name}, so a station using it '
                    'would calibrate against a number the monitor never reports')

    def test_the_same_samples_give_the_same_reading_through_either(self):
        import numpy as np
        from buzz.sampler import LevelStream, SoundCardLevelStream
        from buzz.sdr import SdrLevelStream

        block = np.full(320, 1000.0, dtype=np.float32)
        readings = []
        for subclass in (SoundCardLevelStream, SdrLevelStream):
            stream = subclass.__new__(subclass)
            LevelStream.__init__(stream, -32.0, 16000, 320)
            stream._on_block(block)
            readings.append(stream.read(timeout=0.1))
        assert readings[0] == readings[1]
        assert readings[0] is not None


class TestTheMeterLabelsTheReadingWithTheRightOffset:
    """[station] audio_rf_conversion_db describes a radio feeding a sound card.  For a
    receiver the tuner gain is the conversion, and the two numbers are unrelated.

    Reading the station's figure for both is how the meter came to show -32.0 dB, the
    sound-card default, against a receiver whose own setting said -40.2.  The reading
    was right and the label on it belonged to another station.
    """

    STATION = {'audio_rf_conversion_db': -32.0}

    def test_a_sound_card_station_uses_the_station_offset(self):
        from buzz.setup.screens.calibration import level_offset_for
        assert level_offset_for(AUDIO_SOUNDCARD, self.STATION, None) == -32.0

    def test_a_receiver_uses_its_own_calibrated_offset(self):
        from buzz.setup.screens.calibration import level_offset_for
        values = dict(RTLSDR_VALUES, calibrated_offset_db=-38.5)
        assert level_offset_for(AUDIO_RTLSDR, self.STATION, values) == -38.5

    def test_an_uncalibrated_receiver_falls_back_to_the_negative_of_its_gain(self):
        """Which is the figure the menu row already shows as an estimate, so the meter
        and the menu agree instead of describing different stations.
        """
        from buzz.setup.screens.calibration import level_offset_for
        values = dict(RTLSDR_VALUES, calibrated_offset_db=None, gain_db=40.2)
        assert level_offset_for(AUDIO_RTLSDR, self.STATION, values) == -40.2

    def test_the_station_default_never_reaches_a_receiver(self):
        """The bug itself, stated as the thing that must not happen again."""
        from buzz.setup.screens.calibration import level_offset_for
        for gain in (20.7, 32.8, 40.2, 49.6):
            values = dict(RTLSDR_VALUES, calibrated_offset_db=None, gain_db=gain)
            assert level_offset_for(AUDIO_RTLSDR, self.STATION, values) == -gain

    def test_the_offset_the_meter_shows_is_the_one_it_applies(self):
        """The dialog prints the offset and hands the same number to the stream, so a
        mismatch would put a correct reading under a wrong label.
        """
        from buzz.setup.screens.calibration import level_offset_for
        values = dict(RTLSDR_VALUES, calibrated_offset_db=None, gain_db=40.2)
        offset = level_offset_for(AUDIO_RTLSDR, self.STATION, values)
        with patch.object(RtlSdrDevice, 'open'), patch('buzz.sdr.SdrSource'), \
             patch('buzz.sdr.SdrLevelStream') as stream, patch('buzz.iq.IqToAudio'):
            _open_level_stream(AUDIO_RTLSDR, offset, values)
        assert stream.call_args[0][2] == -40.2

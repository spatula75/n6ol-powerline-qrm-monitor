"""Tests that exactly one level offset applies, and that nothing can disagree about it.

Two sections carry an `audio_rf_conversion_db`, because the two mean different things:
a sound card's is a property of the wiring and the radio, and a receiver's depends on
the tuner gain beside it.  Only one can ever be in use.

Before `BuzzConfig.level_offset_db`, startup resolved that by *copying* the receiver's
figure over the sound card's.  Two consequences followed, and both were reported: a
config file held the same setting name twice with nothing saying which was live, and
the running config no longer matched the file it came from.
"""
import pytest

from buzz.config import RTLSDR, SOUNDCARD, BuzzConfig
from buzz.setup.schema import defaults, load_schema
from buzz.setup.screens.finish import toml_ready


def _config(source, *, station=-32.0, gain=32.8, receiver=None):
    config = BuzzConfig()
    config.audio.source = source
    config.station.audio_rf_conversion_db = station
    config.rtlsdr.gain_db = gain
    config.rtlsdr.calibrated_offset_db = receiver
    return config


class TestOneQuestionHasOneAnswer:
    def test_a_sound_card_station_uses_the_station_figure(self):
        assert _config(SOUNDCARD, station=-28.0).level_offset_db == -28.0

    def test_a_receiver_uses_its_own_calibrated_figure(self):
        assert _config(RTLSDR, receiver=-38.5).level_offset_db == -38.5

    def test_an_uncalibrated_receiver_is_estimated_from_its_gain(self):
        assert _config(RTLSDR, gain=32.8, receiver=None).level_offset_db == -32.8

    def test_the_station_figure_never_reaches_a_receiver(self):
        """However it is set.  This is the defect stated directly: the two must not be
        able to act independently, and a receiver must not pick up a sound card's.
        """
        for station in (-32.0, 0.0, -99.0):
            config = _config(RTLSDR, station=station, gain=32.8, receiver=None)
            assert config.level_offset_db == -32.8

    def test_changing_the_source_changes_the_answer_with_nothing_else_moving(self):
        """The resolution is computed rather than stored, so the answer follows the
        source immediately instead of depending on whether startup has run.
        """
        config = _config(SOUNDCARD, station=-28.0, gain=32.8, receiver=-38.5)
        assert config.level_offset_db == -28.0
        config.audio.source = RTLSDR
        assert config.level_offset_db == -38.5

    def test_nothing_is_written_when_the_answer_is_read(self):
        """A resolver that assigned as a side effect would put the two back into the
        state this exists to prevent.
        """
        config = _config(RTLSDR, station=-32.0, receiver=-38.5)
        for _ in range(3):
            config.level_offset_db
        assert config.station.audio_rf_conversion_db == -32.0
        assert config.rtlsdr.calibrated_offset_db == -38.5


class TestThePlaybackOverrideBeatsBoth:
    """A recording carries the figure it was made with, and --audio-rf-conversion-db
    carries one somebody typed.  Neither describes this station, so neither belongs in
    a section that does.
    """

    def test_it_wins_over_a_sound_card(self):
        config = _config(SOUNDCARD, station=-28.0)
        config.level_offset_override_db = -18.5
        assert config.level_offset_db == -18.5

    def test_it_wins_over_a_receiver(self):
        config = _config(RTLSDR, receiver=-38.5)
        config.level_offset_override_db = -18.5
        assert config.level_offset_db == -18.5

    def test_clearing_it_restores_the_stations_own_answer(self):
        config = _config(RTLSDR, receiver=-38.5)
        config.level_offset_override_db = -18.5
        config.level_offset_override_db = None
        assert config.level_offset_db == -38.5

    def test_it_is_not_a_setting_and_is_absent_from_the_schema(self):
        """It holds runtime state, so writing it to a file would invite somebody to
        set it.  config.RUNTIME is what keeps the drift pins from reporting that
        absence as drift.
        """
        schema = load_schema()
        assert 'level_offset_override_db' not in schema['properties']
        assert BuzzConfig().level_offset_override_db is None


class TestTheConfigFileCarriesOnlyTheOneThatApplies:
    """The operator-facing half.  The setup program used to write every value it held,
    so a receiver's file gained a live-looking [station] audio_rf_conversion_db that
    the monitor ignores, next to an absent [rtlsdr] one.
    """

    def _written(self, source):
        schema = load_schema()
        values = defaults(schema)
        values['audio']['source'] = source
        return toml_ready(values, schema)

    def test_a_receiver_does_not_write_the_sound_cards_figure(self):
        assert 'audio_rf_conversion_db' not in self._written('rtlsdr')['station']

    def test_a_sound_card_still_writes_its_own(self):
        assert 'audio_rf_conversion_db' in self._written('soundcard')['station']

    @pytest.mark.parametrize('source', ['soundcard', 'rtlsdr'])
    def test_at_most_one_offset_is_ever_written(self, source):
        """The property the whole change exists for, stated so it cannot regress."""
        written = self._written(source)
        present = [section for section in ('station', 'rtlsdr')
                   if 'audio_rf_conversion_db' in written.get(section, {})]
        assert len(present) <= 1, f'{source} wrote the offset in {present}'

    def test_an_unset_value_is_still_dropped(self):
        """The older of the two jobs, which must survive the new one."""
        written = self._written('rtlsdr')
        assert 'audio_rf_conversion_db' not in written['rtlsdr'], (
            'an uncalibrated receiver has no figure, so nothing should be written'
        )

    def test_a_caller_with_no_schema_still_gets_the_none_filtering(self):
        values = {'weather': {'latitude': None, 'longitude': 47.6}}
        assert toml_ready(values) == {'weather': {'longitude': 47.6}}

    def test_what_is_written_reloads_to_the_same_answer(self, tmp_path):
        """The round trip, which is what any of this is for.  A file that drops the
        inapplicable setting still has to produce the same offset when read back.
        """
        import tomli_w

        schema = load_schema()
        values = defaults(schema)
        values['audio']['source'] = 'rtlsdr'
        values['rtlsdr']['gain_db'] = 32.8
        path = tmp_path / 'config.toml'
        with open(path, 'wb') as handle:
            tomli_w.dump(toml_ready(values, schema), handle)

        reloaded = BuzzConfig.from_toml(path)
        assert reloaded.level_offset_db == -32.8


class TestAStaleCalibrationIsReported:
    """calibrated_at_gain_db records the gain the offset was measured at.  Until this
    check existed, nothing read it at all, while config.py and schema.json both told
    the operator that "startup compares the two and says so".

    It matters because the failure is silent.  The offset is mostly the negative of
    the gain, so moving one without the other makes every dBm reading wrong, and lock,
    SNR, phase and burst shape all survive it untouched.  The monitor runs perfectly
    and reports the wrong levels.
    """

    def _check(self, caplog, **settings):
        import logging

        from buzz.config import RtlSdrConfig
        from buzz.main import _warn_if_the_calibration_predates_the_gain

        with caplog.at_level(logging.WARNING, logger='buzz.main'):
            _warn_if_the_calibration_predates_the_gain(RtlSdrConfig(**settings))
        return caplog.messages

    def test_a_gain_that_moved_since_the_calibration_warns(self, caplog):
        messages = self._check(caplog, gain_db=32.8, calibrated_offset_db=-38.5,
                               calibrated_at_gain_db=40.2)
        assert any('40.2' in m and '32.8' in m for m in messages), messages

    def test_it_says_how_far_out_the_levels_will_read(self, caplog):
        messages = self._check(caplog, gain_db=32.8, calibrated_offset_db=-38.5,
                               calibrated_at_gain_db=40.2)
        assert any('7.4 dB out' in m for m in messages), messages

    def test_it_offers_the_figure_that_carries_the_measurement_across(self, caplog):
        """The same arithmetic the setup program applies when the gain changes there,
        so an operator fixing it by hand lands on the value the tool would have set.
        """
        messages = self._check(caplog, gain_db=32.8, calibrated_offset_db=-38.5,
                               calibrated_at_gain_db=40.2)
        assert any('-31.1' in m for m in messages), messages

    def test_a_matching_gain_says_nothing(self, caplog):
        assert self._check(caplog, gain_db=32.8, calibrated_offset_db=-31.1,
                           calibrated_at_gain_db=32.8) == []

    def test_a_calibration_with_no_recorded_gain_says_nothing(self, caplog):
        """It came from a file written before the figure was stored, or from somebody
        setting the offset directly.  Neither is evidence of drift.
        """
        assert self._check(caplog, gain_db=32.8, calibrated_offset_db=-31.1,
                           calibrated_at_gain_db=None) == []

    def test_an_uncalibrated_receiver_is_never_checked(self, caplog):
        """Its offset is derived from the gain every time, so the two cannot drift.
        Warning would be telling somebody to fix something that is already right.
        """
        from buzz.config import RTLSDR

        config = BuzzConfig()
        config.audio.source = RTLSDR
        config.rtlsdr.gain_db = 32.8
        config.rtlsdr.calibrated_offset_db = None
        config.rtlsdr.calibrated_at_gain_db = 40.2
        assert config.level_offset_db == -32.8

    def test_the_setup_programs_own_change_never_leaves_it_stale(self, caplog):
        """The carry in section_menu sets both together, so the only way to reach the
        warning is by editing the file, which is what it is for.
        """
        offset, old_gain, new_gain = -38.5, 40.2, 32.8
        carried = round(offset - (new_gain - old_gain), 2)
        assert self._check(caplog, gain_db=new_gain,
                           calibrated_offset_db=carried,
                           calibrated_at_gain_db=new_gain) == []




class TestADroppedKeyIsReported:
    """An unknown key is still ignored rather than fatal, so a file written by another
    build still starts.  What it no longer does is happen in silence.

    Silence cost more than it saved.  A key renamed during development left a
    receiver's calibration behind without a word: the figure reverted to the default,
    the menu went on describing an estimate as an estimate, and nothing said a line had
    been dropped.  A typo does exactly the same thing, which is what this keeps
    catching now that the renaming has stopped.
    """

    def _load(self, tmp_path, caplog, lines):
        import logging

        path = tmp_path / 'config.toml'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        with caplog.at_level(logging.WARNING, logger='buzz.config'):
            config = BuzzConfig.from_toml(path)
        return config, caplog.messages

    def test_a_misspelled_setting_is_named(self, tmp_path, caplog):
        _, messages = self._load(tmp_path, caplog, ['[rtlsdr]', 'gian_db = 40.2'])
        assert any('gian_db' in m for m in messages), messages

    def test_it_says_the_default_was_used(self, tmp_path, caplog):
        """The consequence, not just the fact.  A dropped key means a setting silently
        reverting, which is the part that costs somebody a day.
        """
        config, messages = self._load(tmp_path, caplog,
                                      ['[rtlsdr]', 'gian_db = 44.5'])
        assert config.rtlsdr.gain_db == BuzzConfig().rtlsdr.gain_db
        assert any('default used instead' in m for m in messages), messages

    def test_a_key_renamed_during_development_is_caught_the_same_way(self, tmp_path,
                                                                     caplog):
        """The case that found this: [rtlsdr] audio_rf_conversion_db became
        calibrated_offset_db, and a file still using the old name lost its calibration
        with nothing said.
        """
        config, messages = self._load(
            tmp_path, caplog, ['[rtlsdr]', 'audio_rf_conversion_db = -38.5'])
        assert config.rtlsdr.calibrated_offset_db is None
        assert any('audio_rf_conversion_db' in m for m in messages), messages

    def test_a_correct_file_says_nothing(self, tmp_path, caplog):
        config, messages = self._load(
            tmp_path, caplog,
            ['[audio]', 'source = "rtlsdr"', '', '[rtlsdr]',
             'calibrated_offset_db = -38.5'])
        assert config.rtlsdr.calibrated_offset_db == -38.5
        assert messages == []

    def test_an_unknown_key_does_not_stop_the_program(self, tmp_path, caplog):
        """Ignoring it is still the behavior.  A file from a newer build, or one
        carrying a setting since removed, has to start.
        """
        config, _ = self._load(
            tmp_path, caplog,
            ['[station]', 'callsign = "N0ONE"', 'something_new = 7'])
        assert config.station.callsign == 'N0ONE'

    def test_each_unknown_key_is_named_once(self, tmp_path, caplog):
        _, messages = self._load(
            tmp_path, caplog,
            ['[rtlsdr]', 'gian_db = 40.2', 'frequency_hz = 7074000'])
        assert len(messages) == 2, messages

    def test_the_station_offset_is_not_swept_up_by_the_receiver_rename(self, tmp_path,
                                                                       caplog):
        """[station] audio_rf_conversion_db kept its name and is a different setting.
        Only the receiver's moved.
        """
        config, messages = self._load(
            tmp_path, caplog,
            ['[audio]', 'source = "soundcard"', '', '[station]',
             'audio_rf_conversion_db = -28.0'])
        assert config.station.audio_rf_conversion_db == -28.0
        assert config.level_offset_db == -28.0
        assert messages == []

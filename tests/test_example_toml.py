"""Tests for lib/buzz/setup/example_toml.py and the file it generates.

The point of generating `config.example.toml` is that the sample config and the
setup program cannot describe a setting differently.  That only holds while the committed
file matches what the schema currently produces, which is what the staleness test
here is for - nothing else in the suite would notice a schema edit that was never
regenerated, and the sample config would quietly start lying.
"""
import tomllib
from pathlib import Path

import pytest
from buzz.config import BuzzConfig
from buzz.setup import example_toml
from buzz.setup.schema import defaults, field_names, load_schema, section_names

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / 'config.example.toml'


@pytest.fixture(scope='module')
def rendered() -> str:
    return example_toml.render()


class TestTheCommittedFileIsCurrent:
    def test_regenerating_reproduces_the_committed_file(self, rendered):
        assert EXAMPLE_PATH.read_text(encoding='utf-8') == rendered, (
            'config.example.toml is out of date with lib/buzz/setup/schema.json. It is '
            'generated, not hand-edited: change the schema, then regenerate with\n'
            '    PYTHONPATH=lib python -m buzz.setup.example_toml > config.example.toml')

    def test_it_says_it_is_generated(self, rendered):
        """Whoever opens it to make a change needs to know the edit will be lost."""
        assert 'GENERATED FILE' in rendered
        assert 'schema.json' in rendered


class TestItIsValidToml:
    def test_the_generated_file_parses(self, rendered):
        tomllib.loads(rendered)

    def test_every_section_appears_as_a_table(self, rendered):
        parsed = tomllib.loads(rendered)
        assert list(parsed) == section_names(load_schema())

    def test_the_values_it_writes_are_the_schema_defaults(self, rendered):
        """A sample file that claims a different default from the one the program uses
        is worse than no sample file."""
        schema = load_schema()
        parsed = tomllib.loads(rendered)
        expected = defaults(schema)
        for section in section_names(schema):
            for field, value in parsed[section].items():
                assert value == expected[section][field], f'{section}.{field}'

    def test_it_loads_as_a_config_the_program_accepts(self, tmp_path, rendered):
        """The strongest check available: BuzzConfig.from_toml() reading the sample the
        way it would read a real one, so a mis-typed value is caught here rather than
        by whoever copied the file."""
        path = tmp_path / 'config.toml'
        path.write_text(rendered, encoding='utf-8')
        config = BuzzConfig.from_toml(path)
        assert config.audio.sample_rate == 16000
        assert config.station.callsign == 'N0CALL'
        assert config.server.enabled is False


class TestWhatGetsCommentedOut:
    def test_settings_with_no_static_default_are_commented(self, rendered):
        """A home-directory path written as a literal would be wrong on every machine
        but the one that generated it."""
        assert '# path =' in rendered or '# path = ' in rendered
        assert '# key_path' in rendered

    def test_a_commented_setting_does_not_reach_the_parsed_config(self, rendered):
        parsed = tomllib.loads(rendered)
        assert 'path' not in parsed['station']
        assert 'key_path' not in parsed['server']

    def test_the_sample_does_not_ship_one_particular_sound_card(self, rendered):
        """input_device_name has a real default, but it names the hardware of whoever
        wrote it. Active in the sample, every new operator's config would start by
        claiming a Realtek line-in they may not have."""
        assert 'input_device_name' not in tomllib.loads(rendered)['audio']
        assert '# input_device_name =' in rendered

    def test_the_recording_directory_is_left_for_the_program_to_derive(self, rendered):
        """Empty means 'a recordings folder under the station path'. Writing the empty
        string works, but says nothing; the commented form explains itself."""
        assert 'directory' not in tomllib.loads(rendered)['recording']

    def test_the_program_still_has_a_value_for_them(self, tmp_path, rendered):
        """Commented out in the sample, but never missing at runtime: the dataclass
        default fills in, which is the whole reason it is safe to omit them."""
        path = tmp_path / 'config.toml'
        path.write_text(rendered, encoding='utf-8')
        config = BuzzConfig.from_toml(path)
        assert config.station.path
        assert config.server.key_path


class TestWrite:
    def test_it_writes_what_render_produced(self, tmp_path, rendered):
        target = tmp_path / 'config.example.toml'
        example_toml.write(target)
        assert target.read_text(encoding='utf-8') == rendered

    def test_it_writes_unix_newlines_on_every_platform(self, tmp_path):
        """Without newline='\\n' Windows writes CRLF, and the committed file would then
        differ by line ending from one generated on Linux - failing the staleness test
        on whichever platform did not produce the copy in the repo, for no real reason."""
        target = tmp_path / 'config.example.toml'
        example_toml.write(target)
        assert b'\r\n' not in target.read_bytes()


class TestProse:
    def test_every_field_is_described(self, rendered):
        """The sample config is documentation; a bare assignment with no comment above
        it is the one thing it must never contain."""
        schema = load_schema()
        for section in section_names(schema):
            for field in field_names(schema, section):
                assert f'{field} =' in rendered or f'# {field} =' in rendered

    def test_enum_choices_are_spelled_out(self, rendered):
        assert '60 Hz grid' in rendered and '50 Hz grid' in rendered
        assert 'Open-Meteo' in rendered

    def test_notes_reach_the_file(self, rendered):
        """x-notes carry the long-form reasoning that will not fit in a form field.
        If they were dropped the sample would lose most of what makes it useful."""
        assert 'above Nyquist' in rendered
        assert 'does not roll over into a second file' in rendered

    def test_no_comment_line_runs_absurdly_long(self, rendered):
        for line in rendered.splitlines():
            assert len(line) <= 100, f'unwrapped line: {line[:60]}...'

    def test_it_points_at_the_setup_program_rather_than_hand_editing(self, rendered):
        assert 'setup.sh' in rendered or 'setup.bat' in rendered

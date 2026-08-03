"""Tests for lib/buzz/setup/schema.json and the loader over it.

The important ones here are the drift pins.  The schema restates every field of
BuzzConfig - its type, its default, what it means - and two descriptions of the same
thing are two chances to disagree.  Nothing else in the suite would notice: a field
added to the dataclass and forgotten in the schema simply would not appear in the
wizard, silently, and a schema default that drifted from the dataclass default would
make the wizard offer a value the program does not actually use.
"""
import dataclasses
import json

import pytest
from buzz.config import BuzzConfig
from buzz.setup.schema import (
    SCHEMA_PATH,
    defaults,
    field_names,
    field_schema,
    from_config,
    is_visible,
    load_schema,
    section_names,
    validate,
)

# The two settings whose real default depends on the machine - the home directory -
# and so cannot be written into a static document.  Named here rather than derived,
# so that a third one appearing has to be a deliberate decision rather than a quiet
# widening of the exemption.
RUNTIME_DEFAULT_FIELDS = {('station', 'path'), ('server', 'key_path')}


@pytest.fixture(scope='module')
def schema():
    return load_schema()


def _dataclass_sections() -> dict[str, type]:
    return {f.name: f.type for f in dataclasses.fields(BuzzConfig)}


class TestSchemaMatchesTheDataclasses:
    """The drift pins. Each states an identity that has to keep holding."""

    def test_every_config_section_is_in_the_schema(self, schema):
        assert section_names(schema) == list(_dataclass_sections()), (
            'schema sections must match BuzzConfig field-for-field and in the same '
            'order - the order is what the wizard walks and what config.example.toml '
            'is written in')

    def test_every_field_of_every_section_is_in_the_schema(self, schema):
        config = BuzzConfig()
        for section in section_names(schema):
            in_dataclass = [f.name for f in dataclasses.fields(getattr(config, section))]
            assert field_names(schema, section) == in_dataclass, (
                f'[{section}] differs between schema.json and buzz.config')

    def test_every_schema_default_equals_the_dataclass_default(self, schema):
        config = BuzzConfig()
        for section in section_names(schema):
            for field in field_names(schema, section):
                spec = field_schema(schema, section, field)
                if 'default' not in spec:
                    continue
                assert spec['default'] == getattr(getattr(config, section), field), (
                    f'{section}.{field}: schema default disagrees with BuzzConfig, so '
                    'the wizard would offer a value the program does not use')

    def test_only_the_known_machine_dependent_fields_lack_a_default(self, schema):
        missing = {(section, field)
                   for section in section_names(schema)
                   for field in field_names(schema, section)
                   if 'default' not in field_schema(schema, section, field)}
        assert missing == RUNTIME_DEFAULT_FIELDS

    def test_the_machine_dependent_fields_say_why_they_have_no_default(self, schema):
        """Without the marker, defaults() cannot tell 'derived at runtime' from
        'somebody forgot', and would hand the wizard a missing value either way."""
        for section, field in RUNTIME_DEFAULT_FIELDS:
            assert 'x-default-from-runtime' in field_schema(schema, section, field)


class TestSchemaIsWellFormed:
    def test_it_is_a_valid_schema_document(self, schema):
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_every_field_has_a_title_and_a_description(self, schema):
        """Both are user-facing: the title labels the form field, the description sits
        under it and is the first line of the sample config."""
        for section in section_names(schema):
            for field in field_names(schema, section):
                spec = field_schema(schema, section, field)
                assert spec.get('title'), f'{section}.{field} has no title'
                assert spec.get('description'), f'{section}.{field} has no description'

    def test_descriptions_stay_short_enough_for_a_form_field(self, schema):
        """Long prose belongs in x-notes, which only the sample config renders. A
        description that runs to a paragraph has nowhere to go on screen."""
        for section in section_names(schema):
            for field in field_names(schema, section):
                spec = field_schema(schema, section, field)
                assert len(spec['description']) <= 200, (
                    f'{section}.{field}: move the detail into x-notes')

    def test_every_visibility_gate_names_a_field_in_its_own_section(self, schema):
        """A gate pointing at a field that does not exist would silently hide the
        setting for ever, since the lookup would just never match."""
        for section in section_names(schema):
            siblings = field_names(schema, section)
            for field in siblings:
                gate = field_schema(schema, section, field).get('x-visible-when')
                if gate is not None:
                    assert gate['field'] in siblings, (
                        f'{section}.{field} is gated on {gate["field"]}, which is not '
                        f'a field of [{section}]')

    def test_enum_titles_cover_every_choice(self, schema):
        for section in section_names(schema):
            for field in field_names(schema, section):
                spec = field_schema(schema, section, field)
                titles = spec.get('x-enum-titles')
                if titles is None:
                    continue
                assert {str(c) for c in spec['enum']} == set(titles), (
                    f'{section}.{field}: every enum choice needs a title, and no title '
                    'may name a choice that is not offered')

    def test_the_file_on_disk_is_the_one_that_loads(self):
        assert json.loads(SCHEMA_PATH.read_text(encoding='utf-8')) == load_schema()


class TestDefaults:
    def test_it_covers_every_field(self, schema):
        """Derived from the dataclasses rather than a written-down count, so adding a
        setting does not mean editing a number here as well - the field-by-field pin
        above is what actually catches a field going missing."""
        expected = sum(len(dataclasses.fields(getattr(BuzzConfig(), section)))
                       for section in section_names(schema))
        values = defaults(schema)
        assert sum(len(v) for v in values.values()) == expected
        assert set(values) == set(section_names(schema))

    def test_the_defaults_validate_against_the_schema(self, schema):
        """A shipped default that its own schema rejects would fail the first time
        anybody opened the wizard without a config file."""
        assert validate(schema, defaults(schema)) == []

    def test_machine_dependent_defaults_come_from_the_running_config(self, schema):
        config = BuzzConfig()
        values = defaults(schema, config)
        assert values['station']['path'] == config.station.path
        assert values['server']['key_path'] == config.server.key_path

    def test_a_supplied_config_supplies_those_defaults(self, schema):
        config = BuzzConfig()
        config.station.path = '/somewhere/else'
        assert defaults(schema, config)['station']['path'] == '/somewhere/else'


class TestFromConfig:
    def test_it_reads_current_values_not_defaults(self, schema):
        """This is what makes re-entering setup show what you already have rather
        than starting over from the shipped defaults."""
        config = BuzzConfig()
        config.station.callsign = 'N6OL'
        config.audio.sample_rate = 48000
        values = from_config(schema, config)
        assert values['station']['callsign'] == 'N6OL'
        assert values['audio']['sample_rate'] == 48000

    def test_it_covers_every_field_the_schema_knows(self, schema):
        values = from_config(schema, BuzzConfig())
        for section in section_names(schema):
            assert set(values[section]) == set(field_names(schema, section))


class TestValidate:
    def test_a_good_config_has_no_problems(self, schema):
        assert validate(schema, defaults(schema)) == []

    def test_a_sample_rate_below_the_band_is_refused(self, schema):
        values = defaults(schema)
        values['audio']['sample_rate'] = 4000
        problems = validate(schema, values)
        assert any('audio.sample_rate' in p for p in problems)

    def test_a_sample_rate_above_the_band_is_refused(self, schema):
        values = defaults(schema)
        values['audio']['sample_rate'] = 96000
        assert any('audio.sample_rate' in p for p in problems_of(schema, values))

    def test_an_unknown_weather_source_is_refused(self, schema):
        values = defaults(schema)
        values['weather']['source'] = 'guesswork'
        assert any('weather.source' in p for p in problems_of(schema, values))

    def test_a_pulse_rate_that_is_not_a_real_grid_is_refused(self, schema):
        values = defaults(schema)
        values['audio']['pulse_rate'] = 60
        assert any('audio.pulse_rate' in p for p in problems_of(schema, values))

    def test_uploads_switched_on_with_no_host_are_refused(self, schema):
        """The if/then case: these fields are optional until enabled is true, at which
        point an empty one is a configuration that cannot possibly work."""
        values = defaults(schema)
        values['server']['enabled'] = True
        assert validate(schema, values) != []

    def test_uploads_switched_on_and_filled_in_are_accepted(self, schema):
        values = defaults(schema)
        values['server'].update(enabled=True, host='h.example.com', username='u',
                                remote_path='/var/www/noise/', key_path='/home/u/k.pem')
        assert validate(schema, values) == []

    def test_uploads_left_off_do_not_require_anything(self, schema):
        """Everything under [server] stays empty by default, and that has to remain a
        valid config or a fresh install would fail validation out of the box."""
        assert validate(schema, defaults(schema)) == []

    def test_openmeteo_without_coordinates_is_refused(self, schema):
        values = defaults(schema)
        values['weather']['source'] = 'openmeteo'
        assert validate(schema, values) != []

    def test_openmeteo_with_coordinates_is_accepted(self, schema):
        values = defaults(schema)
        values['weather'].update(source='openmeteo', latitude=34.05, longitude=-118.25)
        assert validate(schema, values) == []

    def test_a_latitude_off_the_planet_is_refused(self, schema):
        values = defaults(schema)
        values['weather'].update(source='openmeteo', latitude=91.0, longitude=0.0)
        assert any('weather.latitude' in p for p in problems_of(schema, values))

    def test_problems_name_the_setting_they_are_about(self, schema):
        """The wizard marks up the offending field, and whoever hand-edits the TOML
        needs to know which line to look at."""
        values = defaults(schema)
        values['audio']['sample_rate'] = 1
        assert validate(schema, values)[0].startswith('audio.sample_rate:')

    def test_every_problem_is_reported_not_just_the_first(self, schema):
        values = defaults(schema)
        values['audio']['sample_rate'] = 1
        values['weather']['source'] = 'guesswork'
        assert len(validate(schema, values)) >= 2


def problems_of(schema, values):
    return validate(schema, values)


class TestIsVisible:
    def test_an_ungated_field_is_always_shown(self, schema):
        assert is_visible(schema, 'station', 'callsign', {}) is True

    def test_a_gated_field_is_hidden_until_its_gate_opens(self, schema):
        assert is_visible(schema, 'server', 'host', {'enabled': False}) is False
        assert is_visible(schema, 'server', 'host', {'enabled': True}) is True

    def test_the_cumulusmx_url_is_shown_only_for_cumulusmx(self, schema):
        assert is_visible(schema, 'weather', 'url', {'source': 'cumulusmx'}) is True
        assert is_visible(schema, 'weather', 'url', {'source': 'openmeteo'}) is False
        assert is_visible(schema, 'weather', 'url', {'source': 'none'}) is False

    def test_the_coordinates_are_shown_only_for_openmeteo(self, schema):
        for field in ('latitude', 'longitude'):
            assert is_visible(schema, 'weather', field, {'source': 'openmeteo'}) is True
            assert is_visible(schema, 'weather', field, {'source': 'cumulusmx'}) is False

    def test_a_missing_gate_field_hides_rather_than_crashes(self, schema):
        """A half-built form has not filled in the gate yet; hiding is the safe answer."""
        assert is_visible(schema, 'server', 'host', {}) is False

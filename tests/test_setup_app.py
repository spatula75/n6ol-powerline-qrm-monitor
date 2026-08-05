"""Tests for the setup program: pure helpers directly, screen behavior via Textual's Pilot."""

import asyncio
from datetime import datetime
from pathlib import Path

from textual import events
from textual.widgets import Button, OptionList
from buzz.device_setup import DeviceInfo
from buzz.setup.screens.calibration import _format_reading, _meter_block
from buzz.setup.screens.field_dialogs import _kind, _parse_number
from buzz.setup.screens.finish import backup_path, changed_fields, toml_ready
from buzz.setup.screens.main_menu import MainMenuScreen
from buzz.setup.screens.section_menu import display_value
from buzz.setup.app import SetupApp


def run(coro):
    """Run an async test scenario without pulling in a pytest-asyncio dependency."""
    return asyncio.run(coro)


class _FakeLevelStream:
    """Stands in for buzz.sampler.LevelStream in the calibration dialog tests.

    Matches the constructor signature _open_level_stream() calls with (config,
    device_index, blocksize) and read()/close()/offset_db, the only parts of the
    real class either dialog touches.  read() always reports -50.0 dBm regardless
    of offset_db - screens/calibration.py never adds it a second time, so a
    constant reading is what a correct dialog should show no matter how many times
    the offset gets nudged.  instances records every one created, so a test can
    reach in and confirm a nudge really reached the stream object, not only the
    on-screen text - see LevelStream's own offset_db docstring for why that must
    work without reopening anything.
    """

    instances: list['_FakeLevelStream'] = []

    def __init__(self, config, device_index, blocksize) -> None:
        self.offset_db = config.station.audio_rf_conversion_db
        self.closed = False
        _FakeLevelStream.instances.append(self)

    def read(self) -> float:
        return -50.0

    def close(self) -> None:
        self.closed = True


class TestKind:
    def test_boolean(self):
        assert _kind({'type': 'boolean'}) == 'boolean'

    def test_enum_wins_over_type(self):
        assert _kind({'type': 'integer', 'enum': [120, 100]}) == 'enum'

    def test_integer(self):
        assert _kind({'type': 'integer'}) == 'number'

    def test_nullable_number(self):
        assert _kind({'type': ['number', 'null']}) == 'number'

    def test_string(self):
        assert _kind({'type': 'string'}) == 'text'

    def test_x_widget_wins_over_type(self):
        assert _kind({'type': 'string', 'x-widget': 'device-picker'}) == 'device-picker'

    def test_x_widget_wins_over_enum(self):
        """audio_rf_conversion_db has no enum, but the precedence must hold even if
        a future field somehow carried both."""
        assert _kind({'type': 'number', 'enum': [1, 2], 'x-widget': 'calibration'}) == 'calibration'


class TestParseNumber:
    def test_integer(self):
        assert _parse_number({'type': 'integer'}, '10') == (True, 10)

    def test_float(self):
        assert _parse_number({'type': 'number'}, '-98.5') == (True, -98.5)

    def test_blank_nullable_is_none(self):
        assert _parse_number({'type': ['number', 'null']}, '  ') == (True, None)

    def test_blank_non_nullable_fails(self):
        ok, _ = _parse_number({'type': 'number'}, '')
        assert ok is False

    def test_unparsable_reports_the_offending_text(self):
        assert _parse_number({'type': 'integer'}, 'abc') == (False, 'abc')


class TestDisplayValue:
    def test_unset(self):
        assert display_value({'type': 'string'}, None) == '(unset)'

    def test_boolean_on(self):
        assert display_value({'type': 'boolean'}, True) == 'On'

    def test_boolean_off(self):
        assert display_value({'type': 'boolean'}, False) == 'Off'

    def test_plain_value(self):
        assert display_value({'type': 'string'}, 'N0CALL') == 'N0CALL'

    def test_enum_shows_the_raw_value_not_the_title(self):
        spec = {'type': 'integer', 'enum': [120, 100], 'x-enum-titles': {'120': 'x', '100': 'y'}}
        assert display_value(spec, 120) == '120'


class TestChangedFields:
    def test_no_changes(self):
        values = {'station': {'callsign': 'N0CALL'}}
        assert changed_fields({'properties': {'station': {'properties': {'callsign': {}}}}},
                              values, {'station': {'callsign': 'N0CALL'}}) == []

    def test_one_change_reported_in_schema_order(self):
        schema = {'properties': {'station': {'properties': {'callsign': {}, 'path': {}}}}}
        original = {'station': {'callsign': 'N0CALL', 'path': '/a'}}
        current = {'station': {'callsign': 'N6OL', 'path': '/a'}}
        assert changed_fields(schema, original, current) == [('station', 'callsign', 'N0CALL', 'N6OL')]


class TestBackupPath:
    def test_names_a_sibling_bak_file(self):
        path = backup_path(Path('/home/x/.buzz/config.toml'), datetime(2026, 8, 3, 14, 5, 9))
        assert path == Path('/home/x/.buzz/config-20260803-140509.toml.bak')


class TestTomlReady:
    def test_drops_none_values(self):
        values = {'weather': {'latitude': None, 'longitude': 47.6}}
        assert toml_ready(values) == {'weather': {'longitude': 47.6}}


class TestSetupAppWalkthrough:
    """End-to-end screen behavior, driven headless through Textual's Pilot."""

    def test_main_menu_has_something_highlighted_without_pressing_an_arrow_key(self, tmp_path):
        """Enter must do something the instant the screen appears, not only after Up/Down."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                assert app.screen.query_one('#sections').highlighted == 0
                await pilot.press('enter')
                await pilot.pause()
                # Row 0 is a section (never the Finish row or the separator), so
                # Enter with nothing touched must have opened a section screen.
                assert app.visited != set()

        run(scenario())

    def test_highlighted_row_stays_visible_when_the_terminal_loses_focus(self, tmp_path):
        """Regression test for a real bug: OptionList.DEFAULT_CSS defines the
        highlighted-row style twice at equal specificity, and the copy that wins
        once the widget is not focused reads `color: $foreground` rather than the
        theme's own block-cursor variables - so losing OS focus (Alt+Tab, clicking
        another window) made the selected row render as this theme's bright cyan
        on itself, invisible.  Confirmed against a live app by posting an AppBlur
        event.  Fixed in screens/base.py."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await pilot.pause()
                option_list = app.screen.query_one('#sections', OptionList)
                focused_style = option_list.get_component_rich_style('option-list--option-highlighted')

                app.post_message(events.AppBlur())
                await pilot.pause()
                blurred_style = option_list.get_component_rich_style('option-list--option-highlighted')

                assert blurred_style.color == focused_style.color
                assert blurred_style.bgcolor == focused_style.bgcolor
                assert blurred_style.color != blurred_style.bgcolor

        run(scenario())

    def test_header_icon_is_blank_but_still_reserves_its_width(self, tmp_path):
        """Regression test: the command-palette icon is disabled (ENABLE_COMMAND_
        PALETTE is off), so it should show nothing and do nothing, but hiding it
        outright with `display: none` was tried first and rejected - it dropped
        the icon's reserved width from the header's layout, which visibly shifted
        the centered title over. `icon=''` (see screens/base.py's scope_header())
        blanks the glyph while keeping the width, so the title's region should
        start exactly where the icon's reserved width ends, not at 0."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                icon = app.screen.query_one('HeaderIcon')
                title = app.screen.query_one('HeaderTitle')
                assert icon.icon == ''
                assert icon.region.width > 0
                assert title.region.x == icon.region.width

        run(scenario())

    def test_header_icon_does_not_highlight_on_hover(self, tmp_path):
        """Regression test: HeaderIcon's own `:hover` rule is not conditioned on
        its `disabled` state, so a disabled icon still visibly highlighted on
        mouse hover despite doing nothing on click - reading as broken rather
        than inert.  See screens/base.py's `HeaderIcon:hover` override."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                icon = app.screen.query_one('HeaderIcon')
                normal_style = icon.rich_style
                await pilot.hover(icon)
                await pilot.pause()
                assert icon.rich_style == normal_style

        run(scenario())

    def test_section_menu_has_something_highlighted_without_pressing_an_arrow_key(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await pilot.press('enter')  # into whichever section row 0 is
                await pilot.pause()
                assert app.screen.query_one('#fields').highlighted == 0

        run(scenario())

    def test_fresh_config_walkthrough_edits_and_saves(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                assert app.had_existing_config is False
                assert app.visited == set()

                main_menu = app.screen
                option_list = main_menu.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                section_screen = app.screen
                assert app.visited == {'station'}
                fields = section_screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                dialog_input = app.screen.query_one('#value')
                dialog_input.value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()

                assert app.values['station']['callsign'] == 'N6OL'

                await pilot.press('escape')
                await pilot.pause()

                option_list = app.screen.query_one('#sections')
                station_row = option_list.get_option_at_index(option_list.get_option_index('station'))
                assert str(station_row.prompt).startswith('[*]')

                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                await pilot.click('#save')
                await pilot.pause()

            return app

        app = run(scenario())
        assert config_path.exists()
        assert 'N6OL' in config_path.read_text(encoding='utf-8')
        assert list(config_path.parent.glob('*.toml.bak')) == []

    def test_reconfigure_seeds_from_existing_file_and_backs_it_up(self, tmp_path):
        config_path = tmp_path / 'config.toml'
        config_path.write_text('[station]\ncallsign = "N0CALL"\n', encoding='utf-8')

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                assert app.had_existing_config is True
                assert app.values['station']['callsign'] == 'N0CALL'

                main_menu = app.screen
                option_list = main_menu.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                app.screen.query_one('#value').value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                await pilot.click('#save')
                await pilot.pause()

        run(scenario())
        assert 'N6OL' in config_path.read_text(encoding='utf-8')
        backups = list(config_path.parent.glob('config-*.toml.bak'))
        assert len(backups) == 1
        assert 'N0CALL' in backups[0].read_text(encoding='utf-8')

    def test_backup_failure_leaves_the_config_untouched(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'
        original_text = '[station]\ncallsign = "N0CALL"\n'
        config_path.write_text(original_text, encoding='utf-8')

        def _boom(*_args, **_kwargs):
            raise OSError('disk full')

        monkeypatch.setattr('shutil.copy2', _boom)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                app.screen.query_one('#value').value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                await pilot.click('#save')
                await pilot.pause()

                assert 'disk full' in app.screen.query_one('#error').content
                # The app is still running: a failed backup must not have exited it
                # the way a successful save does.
                assert app.is_running

        run(scenario())
        assert config_path.read_text(encoding='utf-8') == original_text
        assert list(config_path.parent.glob('*.toml.bak')) == []

    def test_no_changes_skips_backup_and_write(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                assert 'No changes' in app.screen.query_one('#intro').content
                # Nothing to save means no Save button - only a way back.
                assert len(app.screen.query('#save')) == 0

        run(scenario())
        assert not config_path.exists()

    def test_enum_field_round_trips_a_non_string_value(self, tmp_path):
        """pulse_rate's enum values are integers.  The dialog must not turn 120 into "120"."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('pulse_rate')
                await pilot.press('enter')
                await pilot.pause()

                enum_list = app.screen.query_one('#value')
                enum_list.highlighted = enum_list.get_option_index('100')
                await pilot.press('enter')
                await pilot.pause()

                assert app.values['audio']['pulse_rate'] == 100
                assert isinstance(app.values['audio']['pulse_rate'], int)

        run(scenario())

    def test_enum_dialog_highlights_the_current_value_without_an_arrow_key(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('pulse_rate')
                await pilot.press('enter')
                await pilot.pause()

                # pulse_rate defaults to 120, which is the first enum choice - so this
                # would pass even with no highlight-on-mount logic at all.  What it
                # actually guards is that *something* is highlighted the instant the
                # dialog opens, whatever the value.  Enter here must confirm 120, not
                # do nothing.
                enum_list = app.screen.query_one('#value')
                assert enum_list.highlighted is not None
                await pilot.press('enter')
                await pilot.pause()
                assert app.values['audio']['pulse_rate'] == 120

        run(scenario())

    def test_cancel_leaves_the_value_unchanged(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()
                section_screen = type(app.screen)

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                before = app.values['station']['callsign']
                await pilot.press('escape')
                await pilot.pause()

                assert app.values['station']['callsign'] == before
                # Regression coverage: this used to close all the way to MainMenuScreen instead,
                # because dismissing the dialog on Escape did not stop that same
                # key press from also resolving against the section screen's own
                # Escape binding once it became the top screen again.  See
                # test_escape_cancel_lands_on_the_section_menu_not_the_main_menu
                # for the direct reproduction of that bug.
                assert type(app.screen) is section_screen

        run(scenario())

    def test_escape_cancel_lands_on_the_section_menu_not_the_main_menu(self, tmp_path):
        """Regression test for a real bug: one Escape press in a field dialog used
        to close both the dialog and the section menu underneath it, ending back
        up on the main menu instead.  The cause was `key_escape()` dismissing the
        dialog without stopping the key event, so the same press was then also
        resolved against the section screen's own Escape binding, which by then
        was the new top of the stack.  Fixed by moving Escape handling in every
        dialog onto the BINDINGS chain, which does stop the event where a bare
        `key_escape` method does not - see field_dialogs.py and confirm.py."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()
                section_screen = type(app.screen)

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('sample_rate')
                await pilot.press('enter')
                await pilot.pause()

                await pilot.press('escape')
                await pilot.pause()
                assert type(app.screen) is section_screen

        run(scenario())

    def test_text_field_dialog_ok_and_cancel_buttons(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                # Cancel via the button (not Escape) must leave the value untouched.
                before = app.values['station']['callsign']
                await pilot.click('#cancel')
                await pilot.pause()
                assert app.values['station']['callsign'] == before

                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                app.screen.query_one('#value').value = 'N6OL'
                await pilot.click('#ok')
                await pilot.pause()
                assert app.values['station']['callsign'] == 'N6OL'

        run(scenario())

    def test_text_field_dialog_box_does_not_fill_the_screen(self, tmp_path):
        """A Vertical/Horizontal defaults to height: 1fr, not auto - regression
        coverage for a dialog box that once expanded to the full screen height."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test(size=(100, 30)) as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()
                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                # Before the fix this was 26-30, essentially the whole 30-row screen.
                # This dialog's real content (title, description, input, error line,
                # button row, padding, border) comes to well under 20.
                dialog_box = app.screen.query_one('#dialog')
                assert dialog_box.region.height < 20

        run(scenario())

    def test_left_right_arrows_move_focus_between_dialog_buttons(self, tmp_path):
        """Tab already cycles a dialog's buttons.  The arrow keys did not until this
        was bound explicitly - regression coverage for that gap."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()
                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()

                app.screen.query_one('#ok', Button).focus()
                await pilot.pause()
                assert app.focused.id == 'ok'
                await pilot.press('right')
                await pilot.pause()
                assert app.focused.id == 'cancel'
                await pilot.press('left')
                await pilot.pause()
                assert app.focused.id == 'ok'

        run(scenario())

    def test_number_field_parse_failure_shows_error_and_stays_open(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('noise_floor')
                await pilot.press('enter')
                await pilot.pause()

                before = app.values['station']['noise_floor']
                app.screen.query_one('#value').value = 'not a number'
                await pilot.press('enter')
                await pilot.pause()

                # The dialog is still open, the value unchanged, and it says why.
                assert app.values['station']['noise_floor'] == before
                assert 'not a number' in app.screen.query_one('#error').content

                app.screen.query_one('#value').value = '-95'
                await pilot.press('enter')
                await pilot.pause()
                assert app.values['station']['noise_floor'] == -95.0

        run(scenario())

    def test_boolean_field_dialog_ok_and_cancel(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('server')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('enabled')
                await pilot.press('enter')
                await pilot.pause()

                # Cancel leaves it off.
                await pilot.click('#cancel')
                await pilot.pause()
                assert app.values['server']['enabled'] is False

                fields.highlighted = fields.get_option_index('enabled')
                await pilot.press('enter')
                await pilot.pause()
                app.screen.query_one('#value').toggle()
                await pilot.click('#ok')
                await pilot.pause()
                assert app.values['server']['enabled'] is True

                # host, username, etc. only appear now that publishing is on.
                fields = app.screen.query_one('#fields')
                assert fields.get_option_index('host') is not None

        run(scenario())

    def test_enum_field_escape_cancels(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('weather')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('source')
                await pilot.press('enter')
                await pilot.pause()

                before = app.values['weather']['source']
                await pilot.press('escape')
                await pilot.pause()
                assert app.values['weather']['source'] == before

        run(scenario())

    def test_boolean_field_escape_cancels(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('server')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('enabled')
                await pilot.press('enter')
                await pilot.pause()

                await pilot.press('escape')
                await pilot.pause()
                assert app.values['server']['enabled'] is False

        run(scenario())

    def test_finish_screen_back_button_and_escape_save_nothing(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()
                app.screen.query_one('#value').value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                # Back button: no write, edit is still staged.
                await pilot.click('#back')
                await pilot.pause()
                assert not config_path.exists()
                assert app.values['station']['callsign'] == 'N6OL'

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('__finish__')
                await pilot.press('enter')
                await pilot.pause()

                # Escape does the same as Back.
                await pilot.press('escape')
                await pilot.pause()
                assert not config_path.exists()

        run(scenario())

    def test_escape_on_main_menu_always_confirms_even_with_nothing_changed(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await pilot.press('escape')
                await pilot.pause()

                # Still running: Escape must never exit without asking, even when
                # there is nothing to lose.
                assert app.is_running
                assert 'Exit the setup program?' in app.screen.query_one('#question').content

                # Escape on the confirmation itself means "do not exit".
                await pilot.press('escape')
                await pilot.pause()
                assert app.is_running
                # Regression coverage: MainMenuScreen also binds Escape (to reopen
                # this very dialog), so the same bug that skipped a section screen
                # could instead stack a second ConfirmDialog on top of the first
                # from one Escape press.  Confirm there is exactly one screen back.
                assert type(app.screen) is MainMenuScreen

        run(scenario())

    def test_confirming_exit_with_nothing_changed_exits(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await pilot.press('escape')
                await pilot.pause()
                await pilot.click('#confirm')
                await pilot.pause()
                assert not app.is_running

        run(scenario())

    def test_escape_on_main_menu_asks_first_when_something_changed(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()
                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()
                app.screen.query_one('#value').value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()
                # Back to the main menu - one Escape backs out of the section screen,
                # it does not yet ask about exiting.
                await pilot.press('escape')
                await pilot.pause()

                await pilot.press('escape')
                await pilot.pause()

                # Still running: pressing Escape must not have exited by itself.
                assert app.is_running
                assert 'unsaved change' in app.screen.query_one('#question').content

                # "Keep editing" (Escape, the safe default) leaves the app running
                # and the edit intact.
                await pilot.press('escape')
                await pilot.pause()
                assert app.is_running
                assert app.values['station']['callsign'] == 'N6OL'

        run(scenario())

    def test_discarding_from_the_confirm_dialog_exits_without_saving(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()
                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('callsign')
                await pilot.press('enter')
                await pilot.pause()
                app.screen.query_one('#value').value = 'N6OL'
                await pilot.press('enter')
                await pilot.pause()
                await pilot.press('escape')  # section screen -> main menu
                await pilot.pause()
                await pilot.press('escape')  # main menu -> confirm dialog
                await pilot.pause()

                await pilot.click('#confirm')
                await pilot.pause()
                assert not app.is_running

        run(scenario())
        assert not config_path.exists()

    def test_device_picker_lists_probed_devices_and_confirming_writes_the_name(
            self, tmp_path, monkeypatch):
        """Regression coverage for screens/device_picker.py: the row shows the bare
        display_name, not the full name with its host API, but a disabled row must
        stay disabled and unreachable, and confirming the selectable one round-trips
        its exact (full) name into audio.input_device_name."""
        config_path = tmp_path / 'config.toml'
        devices = [
            DeviceInfo(real_index=3, name='USB Mic, WASAPI', display_name='USB Mic',
                      selectable=True, amplitude=500.0, bar='####', display_index=1),
            DeviceInfo(real_index=7, name='Line In, WASAPI', display_name='Line In',
                      selectable=False, amplitude=0.0, bar='needs 44100 Hz', display_index=2),
        ]
        monkeypatch.setattr('buzz.setup.screens.device_picker.enumerate_input_devices',
                            lambda sample_rate: devices)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('input_device_name')
                await pilot.press('enter')
                await pilot.pause()
                # The probe runs off the event loop (see device_picker.py's
                # action_rescan), so pause() alone is not enough to see it finish -
                # workers.wait_for_complete() cannot help either, since the section
                # screen's own on_option_list_option_selected worker is still
                # suspended awaiting this very dialog and would never resolve.
                await asyncio.sleep(0.05)
                await pilot.pause()

                value_list = app.screen.query_one('#value', OptionList)
                assert value_list.option_count == 2
                first_row = str(value_list.get_option_at_index(0).prompt)
                assert 'USB Mic' in first_row
                assert 'WASAPI' not in first_row
                assert value_list.get_option_at_index(0).disabled is False
                assert value_list.get_option_at_index(1).disabled is True
                assert value_list.highlighted == 0  # the only selectable row

                await pilot.press('enter')
                await pilot.pause()

                assert app.values['audio']['input_device_name'] == 'USB Mic, WASAPI'

        run(scenario())

    def test_device_picker_pre_highlights_the_configured_device(self, tmp_path, monkeypatch):
        """The configured device, when it is still present and still selectable,
        must be the row already highlighted - not merely the first selectable one,
        which test_device_picker_lists_probed_devices_and_confirming_writes_the_name
        above cannot tell apart from this."""
        config_path = tmp_path / 'config.toml'
        configured_name = 'Line In (Realtek(R) Audio), Windows DirectSound'
        devices = [
            DeviceInfo(real_index=3, name='USB Mic, WASAPI', display_name='USB Mic',
                      selectable=True, amplitude=500.0, bar='####', display_index=1),
            DeviceInfo(real_index=9, name=configured_name, display_name='Line In (Realtek(R) Audio)',
                      selectable=True, amplitude=50.0, bar='##', display_index=2),
        ]
        monkeypatch.setattr('buzz.setup.screens.device_picker.enumerate_input_devices',
                            lambda sample_rate: devices)

        async def scenario():
            app = SetupApp(config_path=config_path)
            assert app.values['audio']['input_device_name'] == configured_name
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('input_device_name')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                value_list = app.screen.query_one('#value', OptionList)
                assert value_list.highlighted == 1

        run(scenario())

    def test_device_picker_rescan_reruns_the_probe(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'
        calls = []

        def _probe(sample_rate):
            calls.append(sample_rate)
            return []

        monkeypatch.setattr('buzz.setup.screens.device_picker.enumerate_input_devices', _probe)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('input_device_name')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()
                assert len(calls) == 1
                assert 'No input devices found' in app.screen.query_one('#status').content

                await pilot.press('r')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()
                assert len(calls) == 2

        run(scenario())

    def test_device_picker_cancel_leaves_the_device_unchanged(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'
        monkeypatch.setattr('buzz.setup.screens.device_picker.enumerate_input_devices',
                            lambda sample_rate: [])

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                before = app.values['audio']['input_device_name']

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('input_device_name')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                await pilot.press('escape')
                await pilot.pause()

                assert app.values['audio']['input_device_name'] == before

        run(scenario())

    def test_device_picker_show_devices_survives_being_called_after_dismissal(
            self, tmp_path, monkeypatch):
        """Direct regression test for _show_devices()'s NoMatches guard: Escape can
        dismiss this dialog while its background probe is still in flight, and
        Textual only cancels that worker at its next await, so a probe finishing in
        that same instant can still resume and call this method after the screen's
        own widgets are gone.  The real race is too narrow to hit reliably in a
        timed test - there is only one probe, not a running loop, unlike
        CalibrationMeterDialog's equivalent test above - so this calls the method
        directly, post-dismissal, which reproduces the same NoMatches without
        depending on real scheduling."""
        config_path = tmp_path / 'config.toml'
        monkeypatch.setattr('buzz.setup.screens.device_picker.enumerate_input_devices',
                            lambda sample_rate: [])

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('input_device_name')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                dialog = app.screen
                await pilot.press('escape')
                await pilot.pause()

                # The widgets this reaches for are gone now - it must not raise.
                dialog._show_devices()

        run(scenario())

    def test_calibration_meter_shows_a_live_reading_and_does_not_touch_the_offset(
            self, tmp_path, monkeypatch):
        """Regression coverage for screens/calibration.py's CalibrationMeterDialog:
        it shows what the running stream reports, and closing it via its Close
        button must not have written anything back - there is nothing here for it
        to write."""
        config_path = tmp_path / 'config.toml'
        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices',
                            lambda name, kind: {'index': 0})
        monkeypatch.setattr('buzz.setup.screens.calibration.LevelStream', _FakeLevelStream)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                before = app.values['station']['audio_rf_conversion_db']

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('__calibrate__')
                await pilot.press('enter')
                await pilot.pause()
                # The read loop runs forever until the dialog is dismissed (Textual
                # cancels the worker on unmount - see Widget._on_unmount), so there
                # is no worker completion to await here, only real elapsed time for
                # at least one asyncio.to_thread round trip to finish.
                await asyncio.sleep(0.05)
                await pilot.pause()

                assert app.screen.query_one('#meter').content == _meter_block(_format_reading(-50.0))

                await pilot.click('#close')
                await pilot.pause()

                assert app.values['station']['audio_rf_conversion_db'] == before

        run(scenario())

    def test_calibration_meter_reports_a_device_that_will_not_open(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'

        def _boom(name, kind):
            raise ValueError('device not found')

        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices', _boom)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('audio')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('__calibrate__')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                meter = app.screen.query_one('#meter').content
                assert 'Could not open the input device' in meter
                assert 'device not found' in meter

        run(scenario())

    def test_offset_calibration_nudges_and_confirms_the_new_value(self, tmp_path, monkeypatch):
        """Regression coverage for screens/calibration.py's OffsetCalibrationDialog:
        Up/Down must change both the on-screen offset and the value Enter confirms,
        and the running stream must pick up each nudge live - see LevelStream's own
        offset_db docstring."""
        config_path = tmp_path / 'config.toml'
        _FakeLevelStream.instances.clear()
        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices',
                            lambda name, kind: {'index': 0})
        monkeypatch.setattr('buzz.setup.screens.calibration.LevelStream', _FakeLevelStream)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                before = app.values['station']['audio_rf_conversion_db']

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('audio_rf_conversion_db')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                assert f'{before:+.1f} dB' in app.screen.query_one('#offset').content
                assert app.screen.query_one('#meter').content == _meter_block(_format_reading(-50.0))

                await pilot.press('up')
                await pilot.press('up')
                await pilot.pause()

                assert f'{before + 1.0:+.1f} dB' in app.screen.query_one('#offset').content
                assert _FakeLevelStream.instances[-1].offset_db == before + 1.0

                await pilot.press('enter')
                await pilot.pause()

                assert app.values['station']['audio_rf_conversion_db'] == before + 1.0

        run(scenario())

    def test_offset_calibration_space_resets_to_the_schema_default(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'
        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices',
                            lambda name, kind: {'index': 0})
        monkeypatch.setattr('buzz.setup.screens.calibration.LevelStream', _FakeLevelStream)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                default = app.schema['properties']['station']['properties'][
                    'audio_rf_conversion_db']['default']

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('audio_rf_conversion_db')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                await pilot.press('up')
                await pilot.press('up')
                await pilot.press('space')
                await pilot.pause()

                assert f'{default:+.1f} dB' in app.screen.query_one('#offset').content

                await pilot.press('enter')
                await pilot.pause()

                assert app.values['station']['audio_rf_conversion_db'] == default

        run(scenario())

    def test_offset_calibration_escape_discards_every_nudge(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'
        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices',
                            lambda name, kind: {'index': 0})
        monkeypatch.setattr('buzz.setup.screens.calibration.LevelStream', _FakeLevelStream)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                before = app.values['station']['audio_rf_conversion_db']

                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('audio_rf_conversion_db')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                await pilot.press('up')
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()

                assert app.values['station']['audio_rf_conversion_db'] == before

        run(scenario())

    def test_offset_calibration_reports_a_device_that_will_not_open(self, tmp_path, monkeypatch):
        config_path = tmp_path / 'config.toml'

        def _boom(name, kind):
            raise ValueError('device not found')

        monkeypatch.setattr('buzz.setup.screens.calibration.sd.query_devices', _boom)

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                option_list = app.screen.query_one('#sections')
                option_list.highlighted = option_list.get_option_index('station')
                await pilot.press('enter')
                await pilot.pause()

                fields = app.screen.query_one('#fields')
                fields.highlighted = fields.get_option_index('audio_rf_conversion_db')
                await pilot.press('enter')
                await pilot.pause()
                await asyncio.sleep(0.05)
                await pilot.pause()

                meter = app.screen.query_one('#meter').content
                assert 'Could not open the input device' in meter
                assert 'device not found' in meter

        run(scenario())


class TestMainEntryPoint:
    """`python -m buzz.setup` resolves to this module.  Importing it must not run the app."""

    def test_importing_does_not_launch_the_app(self):
        import buzz.setup.__main__ as entry_point
        assert entry_point.SetupApp is SetupApp

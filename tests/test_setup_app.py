"""Tests for the setup program: pure helpers directly, screen behavior via Textual's Pilot."""

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import available_timezones

from textual import events
from textual.widgets import Button, OptionList, RadioButton, RadioSet
from buzz.setup.device_setup import DeviceInfo
from buzz.setup.screens.calibration import _format_reading, _meter_block
from buzz.setup.screens.field_dialogs import _kind, _parse_number
from buzz.setup.screens.finish import backup_path, changed_fields, toml_ready
from buzz.setup.screens.main_menu import MainMenuScreen
from buzz.setup.screens.section_menu import display_value
from buzz.setup.screens.timezone_picker import _canonical_zone_names, _utc_offset_label
from buzz.setup.app import SetupApp


def run(coro):
    """Run an async test scenario without pulling in a pytest-asyncio dependency."""
    return asyncio.run(coro)


async def _wait_until(pilot, condition, description: str, timeout: float = 5.0) -> None:
    """Pump the app until `condition()` holds, and fail saying what never happened.

    Some dialogs fill themselves from a worker that leaves the event loop: the
    timezone picker reads tzdata through asyncio.to_thread, the device picker probes
    the sound card.  A single pilot.pause() only guarantees that the messages queued
    so far were handled, not that such a worker finished, so a test that pauses once
    and asserts is racing the worker.  It wins on a developer's machine and loses on a
    loaded CI runner.

    A fixed sleep is the obvious alternative, and this replaces one.  It has to be
    long enough for the slowest machine that will ever run it, so every run pays that
    cost, and it still fails on a machine slower than whoever picked the number.  This
    returns as soon as the condition holds and spends the timeout only when something
    is really wrong.

    workers.wait_for_complete() cannot be used here.  The section screen's own
    on_option_list_option_selected worker sits suspended awaiting the very dialog
    under test, so waiting for every worker would never return.
    """
    deadline = time.monotonic() + timeout
    while True:
        await pilot.pause()
        if condition():
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f'Waited {timeout:.0f} s for {description} and it never happened.  '
                'The dialog fills itself from a background worker, so either that '
                'worker failed or it now reports through a different route.  Check '
                'the worker this dialog starts in on_mount.')
        await asyncio.sleep(0.01)


async def _open_field(pilot, app, section: str, field: str) -> None:
    """Walk from the main menu into one field's dialog, the way a keyboard would.

    Both menus start with nothing highlighted until something sets `highlighted`,
    which is why each step assigns it before pressing Enter.
    """
    sections = app.screen.query_one('#sections', OptionList)
    sections.highlighted = sections.get_option_index(section)
    await pilot.press('enter')
    await pilot.pause()
    fields = app.screen.query_one('#fields', OptionList)
    fields.highlighted = fields.get_option_index(field)
    await pilot.press('enter')
    await pilot.pause()


async def _open_the_timezone_picker(pilot, app) -> None:
    """Open the timezone dialog, and wait for it to fill itself from tzdata.

    Nine tests open this dialog, and every one of them has to wait before touching the
    filter box or the list, because the zones arrive from a worker that leaves the
    event loop.  Doing that by hand in each was how one of them came to pass on every
    developer's machine and fail on CI.
    """
    await _open_field(pilot, app, 'station', 'timezone')
    await _wait_until(pilot, lambda: app.screen.query_one('#value', OptionList).option_count > 0,
                      'the timezone list to fill from tzdata')


def _rendered_text(widget) -> str:
    """Every character a widget actually paints, read straight off `render_line()`.

    `widget.outer_size` and a Static's own `.content` string both describe what
    the widget was *asked* to show, not what made it onto the screen - a fixed
    CSS height that leaves too little content space silently drops whatever
    does not fit, and neither of those two would notice.  This is the check
    that would have caught it: see the timezone picker's own tests for the bug
    it was written for.
    """
    return ''.join(''.join(segment.text for segment in widget.render_line(y))
                   for y in range(widget.size.height))


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

    def test_x_widget_timezone_picker(self):
        assert _kind({'type': 'string', 'x-widget': 'timezone-picker'}) == 'timezone-picker'

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


class TestCanonicalZoneNames:
    def test_drops_utc_and_zulu_but_keeps_their_target(self):
        """UTC and Zulu are backward-compatibility aliases for the same zone as
        Etc/UTC - see _canonical_zone_names()'s docstring for how tzdata itself
        says so.  The zone they point at must stay in the list; only the
        deprecated alternate names for it should go."""
        canonical = _canonical_zone_names()
        assert 'UTC' not in canonical
        assert 'Zulu' not in canonical
        assert 'Etc/UTC' in canonical

    def test_keeps_ordinary_geographic_zones(self):
        assert 'America/Chicago' in _canonical_zone_names()

    def test_is_a_strict_subset_of_every_available_timezone(self):
        canonical = _canonical_zone_names()
        assert canonical <= available_timezones()
        assert canonical < available_timezones()  # some alias must have been dropped


class TestUtcOffsetLabel:
    def test_utc_has_no_offset(self):
        """UTC never observes daylight time, so this is the one zone whose offset
        is a known constant regardless of when the test runs."""
        assert _utc_offset_label('UTC') == '+00:00'

    def test_format_is_signed_hh_mm(self):
        assert re.fullmatch(r'[+-]\d{2}:\d{2}', _utc_offset_label('America/Chicago'))

    def test_west_of_utc_is_negative(self):
        # America/Chicago is UTC-5 or UTC-6 depending on the time of year - never
        # positive, never zero.
        assert _utc_offset_label('America/Chicago').startswith('-')


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
                # Arrow up to "On" and press it, the way an operator would.  The dialog
                # opens with its cursor on "Off", which is the value currently set.
                await pilot.press('up')
                await pilot.press('space')
                await pilot.click('#ok')
                await pilot.pause()
                assert app.values['server']['enabled'] is True

                # host, username, etc. only appear now that publishing is on.
                fields = app.screen.query_one('#fields')
                assert fields.get_option_index('host') is not None

        run(scenario())

    def test_boolean_dialog_names_both_choices(self, tmp_path):
        """The complaint that replaced the Switch: it was an unlabeled square, and
        the words "on" and "off" appeared nowhere in the dialog."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_field(pilot, app, 'recording', 'enabled')
                labels = [str(button.label)
                          for button in app.screen.query_one('#value', RadioSet).query(RadioButton)]
                assert labels == ['On', 'Off'], (
                    f'The boolean dialog offers {labels} rather than On and Off.  The '
                    'section menu row renders a boolean as On/Off (see '
                    'section_menu.display_value), so the two have to agree.')

        run(scenario())

    def test_boolean_dialog_marks_the_current_value_away_from_the_cursor(self, tmp_path):
        """A switch could not show this, which is why the choices are radio buttons.

        The mark says what is set and the cursor says where the keyboard is.  Textual
        parks its cursor on row 0 whatever is pressed, so BooleanFieldDialog.on_mount
        moves it onto the pressed row; without that an Off field opens with the cursor
        drawn on "On", which is the ambiguity this replaced.  An arrow key then moves
        the cursor and leaves the mark where it was.
        """
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                # recording.enabled defaults to False, so Off is row 1 - a row the
                # cursor would not be on by Textual's own default.
                await _open_field(pilot, app, 'recording', 'enabled')
                choices = app.screen.query_one('#value', RadioSet)
                assert choices.pressed_index == 1, 'Off should be the pressed row'
                assert choices._selected == 1, (
                    f'The cursor opened on row {choices._selected} while row '
                    f'{choices.pressed_index} is the one set.  Either on_mount stopped '
                    'placing the cursor, or Textual renamed RadioSet._selected and the '
                    'guarded assignment in on_mount silently did nothing.')

                # Moving the cursor must not move the mark.
                await pilot.press('up')
                assert choices._selected == 0
                assert choices.pressed_index == 1, (
                    'An arrow key changed the value.  The mark must stay put until Space or '
                    'Enter presses a button.  Without that the dialog is no better '
                    'than the switch it replaced.')

        run(scenario())

    def test_boolean_dialog_arrows_move_the_choices_then_the_buttons(self, tmp_path):
        """Left and right have to do something wherever focus is.

        The screen binds them to focus_previous/focus_next, and RadioSet binds
        up/left and down/right itself.  The widget's binding resolves first, so the
        arrows move between On and Off while the choices have focus and between OK
        and Cancel once a Button does.  Without the screen binding the arrows are
        inert on the button row, because Button.BINDINGS carries only `enter` - and
        an operator who reached OK with Tab then has no way back but Tab again.
        """
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_field(pilot, app, 'recording', 'enabled')
                choices = app.screen.query_one('#value', RadioSet)
                assert choices._selected == 1, 'the dialog opens on the pressed row, Off'

                # The RadioSet has focus, so left is its own binding, not the screen's.
                await pilot.press('left')
                assert choices._selected == 0, (
                    'Left did not move the cursor between the choices.  The screen '
                    'binding for left is resolving ahead of the RadioSet, which means '
                    'an arrow key no longer picks a value.')

                # Tab to OK, where the screen's binding is the only thing bound.
                await pilot.press('tab')
                assert isinstance(app.screen.focused, Button)
                await pilot.press('right')
                assert isinstance(app.screen.focused, Button), (
                    'Focus left the button row entirely.')
                assert app.screen.focused.id == 'cancel', (
                    'Right did nothing on the button row.  Button.BINDINGS is only '
                    '`enter`, so the screen has to bind left and right or the arrows '
                    'are inert once OK or Cancel has focus.')

        run(scenario())

    def test_recording_settings_show_even_when_it_starts_disarmed(self, tmp_path):
        """recording.enabled only seeds the recorder's opening state.

        The Record button, the R key and `--enable-recording` all arm a run that
        started disarmed, and the monitor honors every other recording setting when
        they do.  Hiding those settings left an operator unable to choose a directory
        or an event budget that was going to be used anyway.
        """
        config_path = tmp_path / 'config.toml'
        gated = ('directory', 'max_events', 'rearm_reset_minutes', 'max_seconds',
                 'stop_after_seconds', 'min_lock_seconds', 'min_lock_snr')

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                sections = app.screen.query_one('#sections', OptionList)
                sections.highlighted = sections.get_option_index('recording')
                await pilot.press('enter')
                await pilot.pause()

                assert app.values['recording']['enabled'] is False, (
                    'This test needs recording to start disarmed, which is the '
                    'default it is written against.')
                fields = app.screen.query_one('#fields', OptionList)
                shown = {option.id for option in fields.options}
                missing = [f for f in gated if f not in shown]
                assert not missing, (
                    f'{len(missing)} recording settings are hidden while recording '
                    f'starts disarmed: {", ".join(missing)}.  They are not gated on a '
                    'master switch.  See schema.py on when x-visible-when applies.')

        run(scenario())

    def test_publishing_settings_stay_hidden_until_it_is_switched_on(self, tmp_path):
        """The counterpart, and the reason the keyword still exists.

        server.enabled is a master switch: main.py builds no Publisher at all while
        it is off, so the host and the key path really are out of reach.  Removing
        the recording gates must not remove this one.
        """
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                sections = app.screen.query_one('#sections', OptionList)
                sections.highlighted = sections.get_option_index('server')
                await pilot.press('enter')
                await pilot.pause()

                assert app.values['server']['enabled'] is False
                fields = app.screen.query_one('#fields', OptionList)
                shown = {option.id for option in fields.options}
                offered = [f for f in ('host', 'username', 'remote_path', 'key_path')
                           if f in shown]
                assert not offered, (
                    f'{", ".join(offered)} offered while publishing is off.  The '
                    'monitor builds no Publisher in that state, so it reads none of '
                    'them.')

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

    def test_finish_screen_buttons_sit_side_by_side_and_take_arrow_keys(self, tmp_path):
        """Regression test for a real bug: Save and Back sat in a Vertical, so they
        stacked one above the other instead of side by side like every other
        button row in the program (ConfirmDialog's Exit/Cancel, for instance).
        Worse, nothing focused either button on mount, so no row showed which one
        Enter would confirm, and the arrow keys - bound everywhere else a button
        row appears - had nothing to move between.  See FinishScreen.on_mount()
        and the Horizontal in its compose()."""
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

                save = app.screen.query_one('#save', Button)
                back = app.screen.query_one('#back', Button)
                assert save.region.y == back.region.y, 'Save and Back should sit on the same row'

                assert back.has_focus, 'Back should be the default focus, so Enter cannot save by accident'
                # Save is the first child of #actions and Back the second, so
                # moving to the previous widget from Back reaches Save.
                await pilot.press('left')
                await pilot.pause()
                assert save.has_focus

                await pilot.press('enter')
                await pilot.pause()
                assert config_path.exists()

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
                # action_rescan), so pause() alone cannot see it finish.  _wait_until
                # says why workers.wait_for_complete() is no use here either.
                await _wait_until(pilot, lambda: app.screen.query_one('#value', OptionList).option_count > 0,
                                  'the device probe to fill the list')

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
                await _wait_until(pilot, lambda: app.screen.query_one('#value', OptionList).option_count > 0,
                                  'the device probe to fill the list')

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
                await _wait_until(pilot, lambda: 'Scanning' not in app.screen.query_one('#status').content,
                                  'the device probe to report its result')
                assert len(calls) == 1
                assert 'No input devices found' in app.screen.query_one('#status').content

                await pilot.press('r')
                await pilot.pause()
                await _wait_until(pilot, lambda: len(calls) == 2,
                                  'the rescan to probe the devices a second time')

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
                # This station has no devices, so the list stays empty and the status
                # line is the only thing that says the probe finished.
                await _wait_until(pilot, lambda: 'Scanning' not in app.screen.query_one('#status').content,
                                  'the device probe to report that it found nothing')

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
                # No devices here either, so wait on the status rather than the list.
                await _wait_until(pilot, lambda: 'Scanning' not in app.screen.query_one('#status').content,
                                  'the device probe to report that it found nothing')

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
                # cancels the worker on unmount - see Widget._on_unmount), so there is
                # no worker completion to await.  The meter replacing its 'Starting...'
                # placeholder is the signal that a first reading arrived.
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

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
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

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
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

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
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

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
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

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
                await _wait_until(pilot, lambda: 'Starting' not in app.screen.query_one('#meter').content,
                                  'the calibration meter to take its first reading')

                meter = app.screen.query_one('#meter').content
                assert 'Could not open the input device' in meter
                assert 'device not found' in meter

        run(scenario())

    def test_timezone_picker_highlights_the_current_value_without_typing(self, tmp_path):
        """The same regression shape as test_enum_dialog_highlights_the_current_value_
        without_an_arrow_key: opening the dialog on an already-set timezone must
        highlight it immediately, so Enter with no typing confirms the value
        already in the config rather than doing nothing."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                zone_list = app.screen.query_one('#value')
                assert zone_list.highlighted is not None
                assert zone_list.get_option_at_index(zone_list.highlighted).id == 'America/Los_Angeles'
                # Regression coverage: setting `.highlighted` on a still-empty
                # OptionList scrolls against a stale viewport and leaves the row
                # off-screen - see _show_matches()'s note on why it calls
                # scroll_to_highlight() a second time after the next refresh.
                # A scroll offset of 0 here would mean the list opened still
                # showing the alphabet's start (Africa/...) with the configured
                # zone highlighted but invisible below the fold, only revealed
                # once something else forced a second scroll - which read as the
                # row "jumping" into view on the first arrow key instead of the
                # dialog opening on it already.  The scroll is scheduled with
                # call_after_refresh, so this waits for it for the same reason.
                await _wait_until(pilot, lambda: zone_list.scroll_offset.y > 0,
                                  'the configured zone to be scrolled into view')
                await pilot.press('enter')
                await pilot.pause()
                assert app.values['station']['timezone'] == 'America/Los_Angeles'

        run(scenario())

    def test_timezone_picker_description_does_not_lose_its_wrapped_second_line(self, tmp_path):
        """Regression test for a real bug: #description's own CSS once forced
        `height: 2` while also declaring `padding-bottom: 1` - a fixed height
        covers the whole padding box, so only one row was ever left for content,
        and the wrapped second line ("as America/Los_Angeles.") had nowhere to
        render.  It simply never appeared, at any terminal size, because the
        bug was in the widget's own box, not in how much room the screen gave
        it.  #description is auto-height now, so whatever the wrapped text
        needs is what it gets."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                description = app.screen.query_one('#description')
                assert 'America/Los_Angeles' in _rendered_text(description)

        run(scenario())

    def test_timezone_picker_status_line_is_actually_visible(self, tmp_path):
        """Regression test for the same bug in #status: `height: 1` plus its own
        `padding-top: 1` left zero rows of content space, so no status message -
        not even the match count - ever rendered at all."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                status = app.screen.query_one('#status')
                assert status.size.height > 0
                assert 'match' in _rendered_text(status)

        run(scenario())

    def test_timezone_picker_filters_by_typed_text_and_confirms(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                app.screen.query_one('#filter').value = 'Chicago'
                await pilot.pause()

                zone_list = app.screen.query_one('#value')
                shown = [zone_list.get_option_at_index(i).id for i in range(zone_list.option_count)]
                assert shown == ['America/Chicago']
                assert zone_list.highlighted == 0

                await pilot.press('enter')
                await pilot.pause()
                assert app.values['station']['timezone'] == 'America/Chicago'

        run(scenario())

    def test_timezone_picker_rows_show_the_utc_offset(self, tmp_path):
        """UTC is the one zone whose offset never changes with the season, which is
        what makes its row's exact text predictable enough to assert on."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                app.screen.query_one('#filter').value = 'UTC'
                await pilot.pause()

                # Filtering out the backward-compatibility aliases (see
                # _canonical_zone_names()) leaves exactly one "UTC" match: the
                # zone itself, Etc/UTC.  Plain "UTC" is one of the aliases this
                # dialog no longer offers.
                zone_list = app.screen.query_one('#value')
                row = zone_list.get_option('Etc/UTC')
                assert str(row.prompt) == 'Etc/UTC (+00:00)'

        run(scenario())

    def test_timezone_picker_down_arrow_moves_focus_into_the_result_list(self, tmp_path):
        """Down does not move the text cursor in an Input, so it bubbles up to this
        dialog's own binding instead - covering the path Enter-in-the-filter-box
        never takes: selecting a row directly in the OptionList."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                app.screen.query_one('#filter').value = 'Chicago'
                await pilot.pause()
                await pilot.press('down')
                await pilot.pause()

                zone_list = app.screen.query_one('#value')
                assert zone_list.has_focus
                await pilot.press('enter')
                await pilot.pause()
                assert app.values['station']['timezone'] == 'America/Chicago'

        run(scenario())

    def test_timezone_picker_fits_a_short_terminal_and_still_scrolls_to_the_highlight(
            self, tmp_path):
        """Regression test for a real bug reported against a 25-line terminal: the
        dialog's own border ran past the bottom of the screen, and the option list
        scrolled a highlighted row into an area already clipped away by #dialog's
        own edge, with no scrollbar and no way back to it.  A first fix that gave
        #value its own percentage-of-screen max-height only shrank the bug -
        that guess and #dialog's separate percentage guess still did not agree
        with each other on a short terminal.  The real fix measures the actual
        chrome around #value after it lays out and sizes #value to what is
        actually left, in `_fit_results_to_the_terminal()`, so nothing here is
        two independent guesses anymore."""
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test(size=(80, 25)) as pilot:
                await _open_the_timezone_picker(pilot, app)

                dialog = app.screen.query_one('#dialog')
                assert dialog.region.y + dialog.region.height <= 25

                zone_list = app.screen.query_one('#value')
                zone_list.highlighted = zone_list.option_count - 1
                await pilot.pause()
                # A scroll offset of 0 would mean the list never moved to reveal
                # the highlighted row - exactly the symptom reported: the row
                # highlighted last, past the fold, with nothing to show for it.
                assert zone_list.scroll_offset.y > 0

        run(scenario())

    def test_timezone_picker_reports_when_nothing_matches(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                app.screen.query_one('#filter').value = 'not a real place'
                await pilot.pause()

                zone_list = app.screen.query_one('#value')
                assert zone_list.option_count == 0
                assert 'No timezone matches' in app.screen.query_one('#status').content

        run(scenario())

    def test_timezone_picker_cancel_leaves_the_value_unchanged(self, tmp_path):
        config_path = tmp_path / 'config.toml'

        async def scenario():
            app = SetupApp(config_path=config_path)
            async with app.run_test() as pilot:
                await _open_the_timezone_picker(pilot, app)

                app.screen.query_one('#filter').value = 'Chicago'
                await pilot.pause()
                await pilot.press('escape')
                await pilot.pause()

                assert app.values['station']['timezone'] == 'America/Los_Angeles'

        run(scenario())


class TestMainEntryPoint:
    """`python -m buzz.setup` resolves to this module.  Importing it must not run the app."""

    def test_importing_does_not_launch_the_app(self):
        import buzz.setup.__main__ as entry_point
        assert entry_point.SetupApp is SetupApp

"""Font loading, without needing a display.

The part that can break silently is whether the .ttf files are where we believe they
are: matplotlib is a third-party package that could reorganise its data directory, and
the failure mode downstream is a display full of tofu boxes rather than an exception.
That is worth a test that runs everywhere, not only where there is a Qt platform.
"""

from pathlib import Path
from unittest.mock import patch

from buzz import fonts
from buzz.fonts import FAMILY, display_family, font_files


def clear_cache():
    display_family.cache_clear()


def _font_dir() -> str:
    """Where the fonts are expected to live, for failure messages that can be acted on
    without first going and reading lib/buzz/fonts.py."""
    try:
        import matplotlib
        return str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf')
    except Exception as exc:                                  # noqa: BLE001
        return f'(matplotlib font data could not be located: {exc})'


class TestFontFiles:

    def setup_method(self):
        clear_cache()

    def test_the_font_files_are_where_we_think_they_are(self):
        """Guards against matplotlib moving its data directory: the display would
        still run, drawing every label as a box, and nothing would raise."""
        files = font_files()
        assert files, (
            f'No DejaVu Sans Mono found under {_font_dir()}.\n'
            '\n'
            'buzz.fonts loads the display typeface out of matplotlib\'s bundled font '
            'data rather than from the system, so that a headless render (whose Qt '
            'platform reports no fonts at all) and Windows (where "Monospace" '
            'resolves to a proportional face) both get a real monospace font.\n'
            '\n'
            'The likely cause is that matplotlib reorganised mpl-data, or that a '
            'slimmed-down install dropped its fonts.\n'
            '\n'
            'To fix: find the new location of DejaVuSansMono.ttf in the matplotlib '
            'package and update the directory or _FILES in lib/buzz/fonts.py. Until '
            'then the display falls back to "Monospace" with a logged warning, which '
            'means proportional labels on Windows and empty boxes in a headless '
            'render -- ugly and wrong respectively, but not a crash.')
        missing = [path for path in files if not path.exists()]
        assert not missing, (
            f'font_files() returned paths that do not exist: {missing}.\n'
            'It filters on existence, so this means the files were removed between '
            'that check and this one -- most likely the environment is being rebuilt '
            'underneath the test run. Re-run the suite before investigating further.')

    def test_both_the_regular_and_the_bold_are_loaded(self):
        """Without the bold file Qt synthesises bold by smearing the regular
        sideways, which on a monospace face at 7 points looks like a rendering bug."""
        names = {path.name for path in font_files()}
        assert {'DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf'} <= names, (
            f'Expected both the regular and bold DejaVu Sans Mono in {_font_dir()}, '
            f'found {sorted(names) or "nothing"}.\n'
            '\n'
            'Both are loaded so that bold labels -- the scope header and the meter '
            'legends -- use the face\'s own bold. With only the regular present Qt '
            'fakes it by smearing glyphs sideways, which at 7 points on a monospace '
            'face reads as a rendering fault rather than as bold text.\n'
            '\n'
            'To fix: check what matplotlib now ships under fonts/ttf and update '
            '_FILES in lib/buzz/fonts.py. If the bold file is genuinely gone, drop it '
            'from _FILES deliberately and accept synthesised bold, rather than '
            'leaving this failing.')

    def test_a_missing_matplotlib_is_survivable(self):
        """A display with proportional labels is ugly; one that refuses to start is
        broken. This is the ugly branch, deliberately."""
        with patch.dict('sys.modules', {'matplotlib': None}):
            assert font_files() == []


class TestDisplayFamily:
    """What the paint handlers actually ask Qt for."""

    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    def test_it_reports_the_family_qt_loaded(self):
        with patch.object(fonts.QFontDatabase, 'addApplicationFont', return_value=0), \
             patch.object(fonts.QFontDatabase, 'applicationFontFamilies',
                          return_value=[FAMILY]):
            assert display_family() == FAMILY

    def test_it_loads_the_files_only_once(self):
        """Called from paint handlers running ten times a second on two widgets;
        reloading would add an entry to Qt's font database on every frame."""
        with patch.object(fonts.QFontDatabase, 'addApplicationFont',
                          return_value=0) as add, \
             patch.object(fonts.QFontDatabase, 'applicationFontFamilies',
                          return_value=[FAMILY]):
            display_family()
            display_family()
            display_family()
            assert add.call_count == len(font_files())

    def test_no_files_falls_back_rather_than_failing(self):
        with patch.object(fonts, 'font_files', return_value=[]):
            assert display_family() == 'Monospace'

    def test_qt_refusing_the_file_falls_back(self):
        """addApplicationFont returns -1 rather than raising, so an unchecked return
        would leave the family name of a font that was never loaded."""
        with patch.object(fonts, 'font_files', return_value=[Path('nope.ttf')]), \
             patch.object(fonts.QFontDatabase, 'addApplicationFont', return_value=-1):
            assert display_family() == 'Monospace'

    def test_an_unexpected_family_name_is_used_anyway(self):
        """If DejaVu is ever renamed, using what Qt reports beats asking for a name
        that is no longer there and silently getting the system default."""
        with patch.object(fonts.QFontDatabase, 'addApplicationFont', return_value=0), \
             patch.object(fonts.QFontDatabase, 'applicationFontFamilies',
                          return_value=['Some Other Mono']):
            assert display_family() == 'Some Other Mono'

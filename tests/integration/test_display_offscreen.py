"""The display, rendered to Qt's offscreen platform and inspected pixel by pixel.

    pytest -m integration --no-cov

Tier 2 - headless, but a real Qt paint pass.  These are the only tests in the
project that assert on what the operator actually sees, and they exist because two
bugs got all the way to a running program without a single test noticing: the
toolbar drew in the desktop's default gray instead of the dark theme, and the Record
button stayed lit after it had been pressed, so the one control whose whole job is
to show a state showed the wrong one.  Neither is visible anywhere but the pixels.

What they cannot do is see that something looks wrong in a way nobody predicted --
the elapsed timer that started at nine needed a person.  These pin down the two
specific things known to have broken, and no more than that.
"""

import pytest
from harness import LOUD_PULSES, Monitor

pytest.importorskip('PySide6', reason='the display needs Qt')

from PySide6.QtCore import QPoint                                       # noqa: E402
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter        # noqa: E402
from PySide6.QtWidgets import QPushButton                               # noqa: E402

from buzz.config import BuzzConfig                                      # noqa: E402
from buzz.fonts import FAMILY, display_family, display_font             # noqa: E402
from buzz.waterfall import _BAR_BG, _BAR_H, MainWindow, RecordingBarWidget  # noqa: E402

_BAR_WIDTH = 600            # wide enough that the stretch leaves bare background


def mean_lightness(image: QImage) -> float:
    """Average lightness over the image, sampled on a grid.

    Every pixel would be exact and pointlessly slow; every fourth one is plenty for
    telling a lit control from a dimmed one, which is a difference of tens of levels
    spread over the whole face of a button.
    """
    values = [image.pixelColor(x, y).lightness()
              for y in range(0, image.height(), 2)
              for x in range(0, image.width(), 2)]
    return sum(values) / len(values)


@pytest.fixture(scope='module')
def monitor(tmp_path_factory):
    """A live monitor with a signal in its buffer, so the panels have data to draw.

    Disarmed at the start: the Record button's appearance when armed is the thing
    under test, and a button that begins armed cannot show the change.
    """
    monitor = Monitor(tmp_path_factory.mktemp('display'), enabled=False)
    try:
        monitor.play(1.5, LOUD_PULSES)
        yield monitor
    finally:
        monitor.stop()


@pytest.mark.integration
class TestToolbarIsPainted:
    """The strip draws its own background, not the desktop's.

    A plain QWidget paints its palette background and ignores the stylesheet's unless
    WA_StyledBackground is set, which left the bar in the system gray with only the
    button and the label dark on top of it - an unmistakable pale band across the top
    of an otherwise black window, and nothing in the unit suite could see it.
    """

    def test_the_bare_strip_is_the_theme_background(self, qt_app, monitor):
        bar = RecordingBarWidget(monitor.recorder, None, monitor.analyzer)
        bar.resize(_BAR_WIDTH, _BAR_H)
        image = bar.grab().toImage()
        # Far right, past the button and the label, where the layout's stretch leaves
        # nothing but background.
        painted = image.pixelColor(_BAR_WIDTH - 10, _BAR_H // 2)
        assert painted == QColor(_BAR_BG)

    def test_the_strip_is_painted_inside_the_assembled_window(self, qt_app, monitor):
        """In situ, rather than alone: the bar is the one panel that spans the whole
        width, and getting it into the window is where the layout could still lose it.
        """
        window = MainWindow(monitor.pipeline, monitor.analyzer, BuzzConfig(),
                            recorder=monitor.recorder)
        try:
            window.show()
            qt_app.processEvents()
            image = window.grab().toImage()
            bar = window._bar
            corner = bar.mapTo(window, QPoint(bar.width() - 10, bar.height() // 2))
            assert image.pixelColor(corner) == QColor(_BAR_BG)
        finally:
            window.close()


@pytest.mark.integration
class TestRecordButtonShowsItsState:
    """Armed reads as spent, not as inviting.

    The button offers an action that has already been taken once it is armed, and a
    lit button says the opposite - it looks like the control you still need to press.
    It dims rather than graying out because it is also the only way to switch
    recording off with the mouse, so it has to stay clickable.
    """

    @pytest.fixture(scope='class')
    @staticmethod
    def faces(qt_app, monitor):
        """The button's rendering before and after arming, and what it thinks it is."""
        bar = RecordingBarWidget(monitor.recorder, None, monitor.analyzer)
        bar.resize(_BAR_WIDTH, _BAR_H)
        button = next(child for child in bar.findChildren(QPushButton)
                      if child.text() == 'Record')
        bar.ensurePolished()
        idle = mean_lightness(button.grab().toImage())

        # Through the toolbar's own handler, which is what the button click and the R
        # key both reach -- not by setting the checked property directly, which would
        # prove only that a stylesheet works.
        assert bar.toggle(), 'the toolbar had no recorder to arm'
        qt_app.processEvents()
        armed = mean_lightness(button.grab().toImage())
        return {'idle': idle, 'armed': armed, 'checked': button.isChecked(),
                'enabled': button.isEnabled(),
                'recorder_armed': monitor.recorder.status().armed}

    def test_pressing_it_arms_the_recorder(self, faces):
        assert faces['recorder_armed']

    def test_the_button_follows_the_recorder(self, faces):
        assert faces['checked']

    def test_armed_is_dimmer_than_idle(self, faces):
        """The pixels, not the property: `checked` was already true for the whole life
        of the bug -- what was missing was the stylesheet rule that acts on it."""
        assert faces['armed'] < faces['idle']

    def test_it_stays_clickable_while_armed(self, faces):
        """Dimmed is not disabled.  Disabling it would leave the mouse no way to stop
        a recording that is running."""
        assert faces['enabled']


@pytest.mark.integration
class TestTheDisplayFontSurvivesAHeadlessPlatform:
    """Text has to draw where the platform provides no fonts at all.

    Qt's offscreen platform on Windows reports zero font families, so every label drew
    as an empty box -- and that is precisely the platform a headless render uses. The
    display font is therefore loaded from a file into the application's own database,
    where the platform's cannot matter.

    These run on Linux CI too, where offscreen *does* see fontconfig. That case is
    worth covering as well: asking for the bundled family by name has to win over
    whatever the system would otherwise have supplied, or a render made on one machine
    would not look like a render made on another.
    """

    def test_the_bundled_family_is_what_gets_used(self, qt_app):
        assert display_family() == FAMILY, (
            f'The display resolved to {display_family()!r} rather than {FAMILY!r}. '
            'Falling back means buzz.fonts could not load the .ttf from matplotlib -- '
            'see the warning it logs, and tests/test_fonts.py for where it looks. On '
            'a platform with no font database this leaves every label an empty box.')

    def test_it_is_genuinely_monospace(self, qt_app):
        """The bug underneath the tofu: 'Monospace' is a fontconfig generic that
        resolves to Tahoma on Windows and to Alef when a bare font directory is
        scanned -- both proportional, on a display whose labels are all tabular."""
        metrics = QFontMetrics(display_font(10))
        widths = {character: metrics.horizontalAdvance(character)
                  for character in 'iWm1.'}
        assert len(set(widths.values())) == 1, (
            f'Advance widths differ across characters: {widths}. The display font is '
            'proportional, so columns of frequencies and S-units will not line up. '
            'Check that display_family() found DejaVu Sans Mono rather than falling '
            'back to the "Monospace" generic.')

    def test_it_actually_puts_ink_on_the_canvas(self, qt_app):
        """The end of the chain, and the only part that would have caught the original
        bug: a font can resolve, report metrics, and still draw nothing but boxes."""
        image = QImage(240, 40, QImage.Format.Format_RGBA8888)
        image.fill(0xFF000000)
        painter = QPainter(image)
        painter.setPen(0xFFFFFFFF)
        painter.setFont(display_font(10))
        painter.drawText(6, 26, 'S9 +40 3500 Hz')
        painter.end()
        lit = sum(1 for y in range(image.height()) for x in range(image.width())
                  if image.pixelColor(x, y).lightness() > 100)
        assert lit > 50, (
            f'Only {lit} lit pixels from a 14-character string, which means the text '
            'did not render. Empty tofu boxes would still light pixels, so near-zero '
            'here means no glyphs at all were available.')

    def test_the_toolbar_uses_the_same_face_as_the_panels(self, qt_app, monitor):
        """The strip sets a font size in its stylesheet; without a family beside it Qt
        supplies its default UI font, leaving proportional text above two panels of
        monospace -- and a transport time index whose digits shift as it counts."""
        bar = RecordingBarWidget(monitor.recorder, None, monitor.analyzer)
        assert f'font-family: "{FAMILY}"' in bar.styleSheet()

"""The display, rendered to Qt's offscreen platform and inspected pixel by pixel.

    pytest -m integration --no-cov

Tier 2 — headless, but a real Qt paint pass.  These are the only tests in the
project that assert on what the operator actually sees, and they exist because two
bugs got all the way to a running program without a single test noticing: the
toolbar drew in the desktop's default grey instead of the dark theme, and the Record
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
from PySide6.QtGui import QColor, QImage                                # noqa: E402
from PySide6.QtWidgets import QPushButton                               # noqa: E402

from buzz.config import BuzzConfig                                      # noqa: E402
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
    WA_StyledBackground is set, which left the bar in the system grey with only the
    button and the label dark on top of it — an unmistakable pale band across the top
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
    lit button says the opposite — it looks like the control you still need to press.
    It dims rather than greying out because it is also the only way to switch
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

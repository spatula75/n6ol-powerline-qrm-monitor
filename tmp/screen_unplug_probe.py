"""Throwaway probe: what happens to a repainting Qt window when its monitor is unplugged.

Run it from a console, so the log lines are visible:

    venv\\Scripts\\python tmp\\screen_unplug_probe.py

The window shows a counter that a timer advances ten times a second.  Once a second the
console prints how many timer ticks and how many paint calls happened in that second,
so a frozen window can be told apart three ways:

    ticks 10, paints 10, picture frozen   -> painting works and never reaches the screen
    ticks 10, paints  0                   -> the timer runs and Qt stopped painting
    ticks  0                              -> the timer stopped

Every screen added or removed, every change of the window's screen, and every window
state change is logged with a timestamp.

After reproducing the freeze, press these keys with the window focused, one at a time,
and watch whether the counter moves again:

    0  repaint() now, which forces a paint without anything else
    1  hide() and show() the window
    2  move the window onto the primary screen explicitly
    3  destroy the native window and create it again
"""

import sys
import time

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import QApplication, QWidget


def log(message: str) -> None:
    print(f'{time.strftime("%H:%M:%S")}  {message}', flush=True)


def describe(screen) -> str:
    if screen is None:
        return 'None'
    geometry = screen.geometry()
    return (f'{screen.name()!r} {geometry.width()}x{geometry.height()} at '
            f'({geometry.x()},{geometry.y()}) dpr={screen.devicePixelRatio():g}')


class Probe(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Screen unplug probe')
        self.setFixedSize(480, 200)
        self._counter = 0
        self._ticks = 0
        self._paints = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self._report = QTimer(self)
        self._report.timeout.connect(self._report_second)
        self._report.start(1000)

    def _tick(self) -> None:
        self._counter += 1
        self._ticks += 1
        self.update()

    def _report_second(self) -> None:
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else None
        log(f'ticks {self._ticks:2d}, paints {self._paints:2d}, counter {self._counter}, '
            f'visible={self.isVisible()} minimized={self.isMinimized()} '
            f'screen={screen.name() if screen else None!r}')
        self._ticks = self._paints = 0

    def paintEvent(self, event) -> None:  # noqa: N802
        self._paints += 1
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.green)
        painter.setFont(QFont('Consolas', 48))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(self._counter))
        painter.end()

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.Type.WindowStateChange:
            log(f'window state changed: minimized={self.isMinimized()} '
                f'state={self.windowState()}')
        super().changeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        log('show event')
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        log('hide event')
        super().hideEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_0:
            log('recovery 0: repaint()')
            self.repaint()
        elif key == Qt.Key.Key_1:
            log('recovery 1: hide() and show()')
            self.hide()
            self.show()
        elif key == Qt.Key.Key_2:
            primary = QGuiApplication.primaryScreen()
            log(f'recovery 2: move onto the primary screen, {describe(primary)}')
            self.windowHandle().setScreen(primary)
            self.move(primary.availableGeometry().center() - self.rect().center())
        elif key == Qt.Key.Key_3:
            log('recovery 3: destroy and recreate the native window')
            self.hide()
            self.windowHandle().destroy()
            self.show()
        else:
            super().keyPressEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.screenAdded.connect(lambda screen: log(f'screen added: {describe(screen)}'))
    app.screenRemoved.connect(lambda screen: log(f'screen removed: {describe(screen)}'))
    app.primaryScreenChanged.connect(lambda screen: log(f'primary screen is now {describe(screen)}'))
    for screen in app.screens():
        log(f'screen at start: {describe(screen)}')

    probe = Probe()
    probe.show()
    probe.windowHandle().screenChanged.connect(
        lambda screen: log(f'window moved to screen {describe(screen)}'))
    log(f'window starts on {describe(probe.windowHandle().screen())}')
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

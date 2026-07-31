"""Fixtures shared across the integration tiers.

Everything expensive here is session-scoped on purpose.  These tests cost real
seconds because real seconds are what they are testing, so anything that can be made
once and read many times should be.
"""

import os

import pytest

from harness import LOUD_PULSES, Monitor


@pytest.fixture(scope='session')
def recorded_event(tmp_path_factory):
    """A .wav of a loud event, written by the real recorder.

    Shared by every test that needs something to replay.  Made once because making it
    costs fifteen seconds of wall clock and each consumer wants the same thing from
    it: a file with an unmistakable pulse train in the middle and quiet either side.

    Read-only by contract.  The replay tests open it repeatedly and must not depend
    on the order they happen to run in.
    """
    monitor = Monitor(tmp_path_factory.mktemp('recorded'),
                      max_seconds=6.0, stop_after_seconds=2.0)
    try:
        monitor.play(2, None)
        monitor.play(10, LOUD_PULSES)
        monitor.play(3, None)
    finally:
        monitor.stop()
    recordings = monitor.recordings()
    assert recordings, 'the recorder wrote nothing to replay'
    return recordings[0]


@pytest.fixture(scope='session')
def qt_app():
    """A QApplication rendering to Qt's offscreen platform plugin.

    Offscreen is a real raster paint device with no window server behind it: widgets
    lay out and paint exactly as they would on a desktop, and grab() returns the
    pixels they painted.  That is what makes the display testable on a headless
    runner at all — and both display bugs this suite is answering for (a toolbar that
    drew in the desktop's grey, a Record button that stayed lit once armed) were
    visible in nothing but those pixels.

    One per session: Qt permits a single QApplication per process and does not fully
    release one, so building a second is a crash rather than a test failure.
    """
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # Not app.quit() or deleteLater(): nothing here runs an event loop, so there is
    # no exec() to return from and no deferred-delete pass to run.  Widgets are
    # closed by the tests that made them; the application object outlives them and
    # dies with the process.
    app.processEvents()

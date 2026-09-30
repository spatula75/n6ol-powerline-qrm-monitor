"""The application icon's files, and the Windows call that lets the taskbar show it.

The ctypes names patched here exist only on Windows, so each patch passes
`create=True`, for the reasons test_windows_qos.py gives in full.
"""

import ctypes
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtGui import QImage

from buzz.display import app_icon


class TestTheIconFiles:
    def test_the_full_size_icon_is_there(self):
        """The window falls back to the platform's icon without it, and nothing else
        would notice the file had moved."""
        names = [path.name for path in app_icon.icon_files()]
        assert 'app-icon_256x256.png' in names, (
            f'app-icon_256x256.png was not found in {app_icon.RESOURCES}.  It holds {names}.  '
            'The window shows the default icon without it.  If resources/ moved, '
            'RESOURCES in buzz.display.app_icon has to follow it.')

    def test_each_file_is_the_size_its_name_says(self):
        """A drift pin between each file name and its pixels, because Qt picks a size by
        what the image holds and a person picks a file by its name."""
        for path in app_icon.icon_files():
            width, height = map(int, app_icon.ICON_NAME.fullmatch(path.name).groups())
            image = QImage(str(path))
            assert (image.width(), image.height()) == (width, height), (
                f'{path.name} holds a {image.width()}x{image.height()} image.  '
                'Rename the file or redraw it at the size in its name.')

    def test_each_file_has_a_transparent_background(self):
        for path in app_icon.icon_files():
            image = QImage(str(path))
            assert image.hasAlphaChannel(), (
                f'{path.name} has no alpha channel, so it draws as a square on the '
                'taskbar and the title bar.  Save it as a PNG with transparency.')

    def test_the_sizes_come_smallest_first(self, tmp_path):
        for name in ('app-icon_256x256.png', 'app-icon_16x16.png', 'app-icon_32x32.png',
                     'app-icon_draft.png', 'notes.png'):
            (tmp_path / name).write_bytes(b'')
        with patch.object(app_icon, 'RESOURCES', tmp_path):
            names = [path.name for path in app_icon.icon_files()]
        assert names == ['app-icon_16x16.png', 'app-icon_32x32.png', 'app-icon_256x256.png']


def _shell32(result: int = 0) -> SimpleNamespace:
    return SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=MagicMock(return_value=result))


class TestTheTaskbarIdentity:
    def test_other_platforms_do_not_load_a_windows_library(self):
        with patch.object(app_icon.sys, 'platform', 'linux'), \
                patch.object(app_icon.ctypes, 'WinDLL', create=True) as load:
            assert app_icon.use_own_taskbar_identity() is True
        load.assert_not_called()

    def test_windows_sets_the_monitors_own_identity(self):
        shell32 = _shell32()
        with patch.object(app_icon.sys, 'platform', 'win32'), \
                patch.object(app_icon.ctypes, 'WinDLL', create=True, return_value=shell32):
            assert app_icon.use_own_taskbar_identity() is True
        shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with(
            'N6OL.PowerlineQrmMonitor')

    def test_the_identity_is_one_windows_accepts(self):
        """Windows documents at most 128 characters and no spaces."""
        identity = app_icon._APP_USER_MODEL_ID
        assert len(identity) <= 128 and ' ' not in identity

    def test_a_failure_code_warns_and_carries_on(self, caplog):
        # E_INVALIDARG, as the signed value a c_long restype gives back.
        shell32 = _shell32(result=-2147024809)
        with patch.object(app_icon.sys, 'platform', 'win32'), \
                patch.object(app_icon.ctypes, 'WinDLL', create=True, return_value=shell32), \
                patch.object(app_icon.ctypes, 'FormatError', create=True,
                             return_value='The parameter is incorrect.') as format_error, \
                caplog.at_level(logging.WARNING, logger='buzz.display.app_icon'):
            assert app_icon.use_own_taskbar_identity() is False
        format_error.assert_called_once_with(0x80070057)
        assert 'The parameter is incorrect.' in caplog.text
        assert 'shows the Python icon instead' in caplog.text

    def test_a_windows_without_the_call_warns_rather_than_crashing(self, caplog):
        class WithoutTheCall:
            def __getattr__(self, name):
                raise AttributeError(f'function {name!r} not found')

        with patch.object(app_icon.sys, 'platform', 'win32'), \
                patch.object(app_icon.ctypes, 'WinDLL', create=True, return_value=WithoutTheCall()), \
                caplog.at_level(logging.WARNING, logger='buzz.display.app_icon'):
            assert app_icon.use_own_taskbar_identity() is False
        assert 'SetCurrentProcessExplicitAppUserModelID' in caplog.text

    def test_a_missing_library_warns_rather_than_crashing(self, caplog):
        with patch.object(app_icon.sys, 'platform', 'win32'), \
                patch.object(app_icon.ctypes, 'WinDLL', create=True, side_effect=OSError('no shell32')), \
                caplog.at_level(logging.WARNING, logger='buzz.display.app_icon'):
            assert app_icon.use_own_taskbar_identity() is False
        assert 'no shell32' in caplog.text

    def test_the_names_this_patches_are_real_on_windows(self):
        if sys.platform != 'win32':
            pytest.skip('ctypes only defines these on Windows, so only Windows can check '
                        'the spelling.  The tests above run everywhere.')
        for name in ('WinDLL', 'FormatError'):
            assert hasattr(ctypes, name), f'ctypes on this Windows build has no {name}.'
        assert hasattr(ctypes.WinDLL('shell32'), 'SetCurrentProcessExplicitAppUserModelID')

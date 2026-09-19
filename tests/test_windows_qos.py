"""Tests for the Windows execution-speed policy used by the monitor.

Every ctypes name patched here exists only on Windows: CPython defines `WinDLL`,
`get_last_error` and `FormatError` inside `if os.name == "nt"`.  So each patch passes
`create=True`, without which `patch.object` has nothing to replace and raises on any
other platform.  CI runs on Linux and these four tests failed there while passing on
the machine they were written on.

Skipping them off Windows was the other option and is worse.  `windows_qos` would then
be uncovered on Linux and covered on Windows, so the coverage gate would mean two
different things on the two platforms, and CI would quietly stop exercising the path
that matters most: the one that runs on an operator's machine.

What `create=True` costs is that a misspelled attribute would be invented rather than
refused.  `test_the_names_this_patches_are_real_on_windows` covers that, by checking
the same names against ctypes itself wherever ctypes has them.
"""

import ctypes
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from buzz import windows_qos


def _kernel32(changed: bool = True):
    return SimpleNamespace(
        GetCurrentProcess=MagicMock(return_value=1234),
        SetProcessInformation=MagicMock(return_value=changed),
    )


def test_other_platforms_do_not_load_a_windows_library():
    with patch.object(windows_qos.sys, 'platform', 'linux'), \
            patch.object(windows_qos.ctypes, 'WinDLL', create=True) as load:
        assert windows_qos.keep_execution_speed_while_hidden() is True
    load.assert_not_called()


def test_windows_clears_only_execution_speed_throttling():
    kernel32 = _kernel32()
    with patch.object(windows_qos.sys, 'platform', 'win32'), \
            patch.object(windows_qos.ctypes, 'WinDLL', create=True,
                         return_value=kernel32):
        assert windows_qos.keep_execution_speed_while_hidden() is True

    process, information_class, pointer, size = kernel32.SetProcessInformation.call_args.args
    state = pointer._obj
    assert process == 1234
    assert information_class == 4
    assert size == ctypes.sizeof(state)
    assert state.version == 1
    assert state.control_mask == 1
    assert state.state_mask == 0


def test_a_windows_failure_warns_and_leaves_the_monitor_usable(caplog):
    kernel32 = _kernel32(changed=False)
    with patch.object(windows_qos.sys, 'platform', 'win32'), \
            patch.object(windows_qos.ctypes, 'WinDLL', create=True,
                         return_value=kernel32), \
            patch.object(windows_qos.ctypes, 'get_last_error', create=True,
                         return_value=5), \
            patch.object(windows_qos.ctypes, 'FormatError', create=True,
                         return_value='Access denied'), \
            caplog.at_level(logging.WARNING, logger='buzz.windows_qos'):
        assert windows_qos.keep_execution_speed_while_hidden() is False

    assert 'Access denied' in caplog.text
    assert 'Leave the window visible or run headlessly' in caplog.text


def test_a_windows_library_failure_uses_the_same_survivable_path(caplog):
    with patch.object(windows_qos.sys, 'platform', 'win32'), \
            patch.object(windows_qos.ctypes, 'WinDLL', create=True,
                         side_effect=OSError('no kernel32')), \
            caplog.at_level(logging.WARNING, logger='buzz.windows_qos'):
        assert windows_qos.keep_execution_speed_while_hidden() is False

    assert 'no kernel32' in caplog.text


def test_a_windows_without_the_call_warns_rather_than_crashing(caplog):
    """SetProcessInformation arrived in Windows 8, and a Wine build may not carry it.

    ctypes raises AttributeError from the argtypes line rather than from the call, so
    the OSError this once caught never fired.  main() calls this before it reads the
    config, so the monitor died with a traceback on the oldest machines it claims to
    run on.  A SimpleNamespace always has the attribute, which is why nothing here
    saw it.
    """

    class WithoutTheCall:
        GetCurrentProcess = MagicMock(return_value=1234)

        def __getattr__(self, name):
            raise AttributeError(f'function {name!r} not found')

    with patch.object(windows_qos.sys, 'platform', 'win32'),             patch.object(windows_qos.ctypes, 'WinDLL', create=True,
                         return_value=WithoutTheCall()),             caplog.at_level(logging.WARNING, logger='buzz.windows_qos'):
        assert windows_qos.keep_execution_speed_while_hidden() is False

    assert 'SetProcessInformation' in caplog.text
    assert 'Leave the window visible or run headlessly' in caplog.text


def test_the_names_this_patches_are_real_on_windows():
    """What `create=True` gives up, bought back where it can be.

    A patch that creates the attribute it wanted also creates a misspelled one, so
    these tests would keep passing against a name ctypes does not have.  On Windows
    ctypes does have them, so there the spelling can be checked for real.  Elsewhere
    there is nothing to check against and this says so rather than passing quietly.
    """
    if sys.platform != 'win32':
        pytest.skip('ctypes only defines these on Windows, so only Windows can check '
                    'the spelling.  The four tests above run everywhere.')
    for name in ('WinDLL', 'get_last_error', 'FormatError'):
        assert hasattr(ctypes, name), (
            f'The tests above patch ctypes.{name} with create=True, and ctypes on this '
            f'Windows build has no such attribute.  Either the name is misspelled or '
            f'the one the module calls has moved.')

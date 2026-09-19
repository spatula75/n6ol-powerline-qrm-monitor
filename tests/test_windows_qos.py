"""Tests for the Windows execution-speed policy used by the monitor."""

import ctypes
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from buzz import windows_qos


def _kernel32(changed: bool = True):
    return SimpleNamespace(
        GetCurrentProcess=MagicMock(return_value=1234),
        SetProcessInformation=MagicMock(return_value=changed),
    )


def test_other_platforms_do_not_load_a_windows_library():
    with patch.object(windows_qos.sys, 'platform', 'linux'), \
            patch.object(windows_qos.ctypes, 'WinDLL') as load:
        assert windows_qos.keep_execution_speed_while_hidden() is True
    load.assert_not_called()


def test_windows_clears_only_execution_speed_throttling():
    kernel32 = _kernel32()
    with patch.object(windows_qos.sys, 'platform', 'win32'), \
            patch.object(windows_qos.ctypes, 'WinDLL', return_value=kernel32):
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
            patch.object(windows_qos.ctypes, 'WinDLL', return_value=kernel32), \
            patch.object(windows_qos.ctypes, 'get_last_error', return_value=5), \
            patch.object(windows_qos.ctypes, 'FormatError', return_value='Access denied'), \
            caplog.at_level(logging.WARNING, logger='buzz.windows_qos'):
        assert windows_qos.keep_execution_speed_while_hidden() is False

    assert 'Access denied' in caplog.text
    assert 'Leave the window visible or run headlessly' in caplog.text


def test_a_windows_library_failure_uses_the_same_survivable_path(caplog):
    with patch.object(windows_qos.sys, 'platform', 'win32'), \
            patch.object(windows_qos.ctypes, 'WinDLL', side_effect=OSError('no kernel32')), \
            caplog.at_level(logging.WARNING, logger='buzz.windows_qos'):
        assert windows_qos.keep_execution_speed_while_hidden() is False

    assert 'no kernel32' in caplog.text

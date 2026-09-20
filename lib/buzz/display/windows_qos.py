"""Windows scheduling policy for a monitor with continuous audio deadlines."""

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger(__name__)

_PROCESS_POWER_THROTTLING = 4
_POWER_THROTTLING_CURRENT_VERSION = 1
_POWER_THROTTLING_EXECUTION_SPEED = 0x1


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ('version', wintypes.DWORD),
        ('control_mask', wintypes.DWORD),
        ('state_mask', wintypes.DWORD),
    ]


def keep_execution_speed_while_hidden() -> bool:
    """Keep Windows from slowing the process when its window is minimized.

    Windows can assign a lower execution-speed QoS to a process whose window is not
    visible.  That policy affects CPU frequency and core selection independently of
    process priority.  The monitor keeps audio deadlines while minimized, so it clears
    only that policy bit and leaves priority and timer resolution under system control.

    Other platforms need no action and report success.  A Windows failure is also
    survivable: the monitor can still run, and the warning gives the operator a way to
    avoid the measured slowdown.
    """
    if sys.platform != 'win32':
        return True

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetProcessInformation.restype = wintypes.BOOL

        requested = _PowerThrottlingState(
            _POWER_THROTTLING_CURRENT_VERSION,
            _POWER_THROTTLING_EXECUTION_SPEED,
            0,
        )
        changed = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            _PROCESS_POWER_THROTTLING,
            ctypes.byref(requested),
            ctypes.sizeof(requested),
        )
        if changed:
            return True
        code = ctypes.get_last_error()
        reason = ctypes.FormatError(code).strip()
    except (OSError, AttributeError) as error:
        # This catches AttributeError because SetProcessInformation arrived in
        # Windows 8, and ctypes raises that from the argtypes line rather than from
        # the call when the symbol is absent.  A build of Wine without the symbol
        # answers the same way.  main() calls this before it reads the config, so
        # letting either exception out would end the monitor with a traceback where
        # the warning below is the whole remedy.
        reason = str(error)

    logger.warning(
        'Windows could not keep full execution speed while the monitor is minimized: '
        '%s.  Audio handling can slow down in that state.  Leave the window visible '
        'or run headlessly.', reason)
    return False

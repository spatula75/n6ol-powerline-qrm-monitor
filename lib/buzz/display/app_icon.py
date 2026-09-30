"""The monitor's own icon, for its window and its taskbar entry.

The artwork lives in `resources/` at the top of the tree, one PNG per size, each named
`app-icon_<width>x<height>.png`.  Qt picks the size nearest to what each place asks
for, so a hand-drawn small size is used where one exists, and the largest image is
scaled down everywhere else.

Windows needs one more step.  Its taskbar groups a window under the program that owns
the process, which is `python.exe` here, and shows that program's icon whatever the
window's icon is.  An explicit AppUserModelID gives this process an identity of its
own, and the taskbar then shows the window's icon.
"""

import ctypes
import logging
import re
import sys
from ctypes import wintypes
from pathlib import Path

from PySide6.QtGui import QIcon

logger = logging.getLogger(__name__)

# lib/buzz/display/app_icon.py, so the top of the tree is three directories up.
RESOURCES = Path(__file__).resolve().parents[3] / 'resources'
ICON_NAME = re.compile(r'app-icon_(\d+)x(\d+)\.png')

# Windows' documented form is Company.Product, with no spaces and at most 128
# characters.  The value only has to differ from every other program's.
_APP_USER_MODEL_ID = 'N6OL.PowerlineQrmMonitor'


def icon_files() -> list[Path]:
    """Every size of the icon in `resources/`, smallest first.

    This is separate from building the QIcon so it can be tested without a
    QApplication, because Qt ends the process if a QIcon is made before one exists.
    """
    found = [path for path in RESOURCES.glob('app-icon_*.png') if ICON_NAME.fullmatch(path.name)]
    return sorted(found, key=lambda path: int(ICON_NAME.fullmatch(path.name).group(1)))


def application_icon() -> QIcon:  # pragma: no cover -- a QIcon needs a QApplication
    """The icon at every size `resources/` holds.  Call only after QApplication exists.

    With no files the icon is empty, and the window keeps the platform's default.
    That is a cosmetic loss rather than a fault, so it warns and carries on.
    """
    icon = QIcon()
    for path in icon_files():
        icon.addFile(str(path))
    if icon.isNull():
        logger.warning('The application icon was not found in %s, so the window shows '
                       'the default icon.  Check that the app-icon PNG files are still '
                       'in that directory.', RESOURCES)
    return icon


def use_own_taskbar_identity() -> bool:
    """Give this process a Windows taskbar identity of its own, so it shows this icon.

    Call this before the first window opens, because Windows reads the identity when
    it creates the taskbar entry.  Other platforms need no action and report success.
    A failure costs only the icon, so it warns and the monitor carries on.
    """
    if sys.platform != 'win32':
        return True
    try:
        shell32 = ctypes.WinDLL('shell32')
        set_identity = shell32.SetCurrentProcessExplicitAppUserModelID
        set_identity.argtypes = [wintypes.LPCWSTR]
        set_identity.restype = ctypes.c_long
        result = set_identity(_APP_USER_MODEL_ID)
        if result >= 0:
            return True
        # A negative HRESULT is a failure code, and FormatError reads it as unsigned.
        reason = ctypes.FormatError(result & 0xFFFFFFFF).strip()
    except (OSError, AttributeError) as error:
        # AttributeError for the same reason as in windows_qos: ctypes raises it when
        # the symbol is absent, as it can be under Wine.
        reason = str(error)
    logger.warning('Windows would not give the monitor its own taskbar entry: %s.  The '
                   'taskbar shows the Python icon instead.  Nothing else is affected.',
                   reason)
    return False

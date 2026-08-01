"""The display's typeface, carried with the program rather than borrowed from the OS.

Every label on the display is tabular - frequencies up a frequency axis, S-units up a
meter, a time base across a scope - so they want a monospace face, and they want the
same one everywhere.  Asking the system for one delivers neither.

`QFont('Monospace')` is a fontconfig generic.  It resolves on Linux and means nothing
on Windows, where Qt substitutes its default UI face: this display asked for Monospace
for years and drew Tahoma, which is proportional, so columns of numbers never lined up.

Worse for rendering, a platform with no font database at all resolves it to nothing and
draws every character as a tofu box.  Qt's offscreen platform on Windows is exactly
that -- it reports zero font families -- which is the platform a headless render would
use.  A video of empty rectangles is not a video anybody wants, and nothing about the
failure announces itself until you look at the output.

So the font is loaded from a file, into the application's own database, where neither
the platform nor the machine's installed fonts can affect it.  DejaVu Sans Mono comes
with matplotlib, which is already a hard dependency, so this adds no asset to ship and
no licence to think about.  Measured both ways, the same string draws the identical
number of lit pixels under the native platform and under offscreen.
"""

import logging
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

logger = logging.getLogger(__name__)

FAMILY = 'DejaVu Sans Mono'
# Fontconfig's generic, which is right on Linux and the best guess left if the file
# below cannot be found.  Reaching this means the labels may be proportional, which is
# ugly rather than broken, so it warns and carries on.
_FALLBACK_FAMILY = 'Monospace'
_FILES = ('DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf')


def font_files() -> list[Path]:
    """The .ttf files to load, from matplotlib's bundled copy of DejaVu.

    Separated from the loading so it can be tested without a QApplication: whether the
    files are where we think they are is the part that breaks when a dependency
    reorganises itself, and it is worth knowing without a display.

    Bold is loaded alongside the regular so that the bold labels are the face's own
    bold rather than one Qt synthesises by smearing the regular sideways.
    """
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('Could not locate matplotlib font data (%s); '
                       'display labels will use whatever the system offers.', exc)
        return []
    return [path for name in _FILES if (path := ttf / name).exists()]


@lru_cache(maxsize=1)
def display_family() -> str:
    """Load the display font once and return the family name to ask for.

    Cached because it is called from paint handlers, which run ten times a second on
    two widgets, and because loading the same file repeatedly would grow Qt's font
    database an entry at a time.

    Must be called with a QApplication already built -- Qt has nowhere to put an
    application font before that -- which is why it happens on first paint rather than
    at import.
    """
    files = font_files()
    if not files:
        return _FALLBACK_FAMILY
    families: list[str] = []
    for path in files:
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id == -1:
            logger.warning('Qt refused to load the display font %s', path.name)
            continue
        families.extend(str(f) for f in QFontDatabase.applicationFontFamilies(font_id))
    if not families:
        logger.warning('No display font could be loaded; falling back to %s.',
                       _FALLBACK_FAMILY)
        return _FALLBACK_FAMILY
    return FAMILY if FAMILY in families else families[0]


def display_font(size: int, bold: bool = False) -> QFont:  # pragma: no cover
    """A label font at `size` points, in the display's own face.

    Excluded from coverage for the same reason the widget classes are: constructing a
    QFont needs a live Qt application, which the unit suite has no platform for.  The
    offscreen integration tier exercises it, and does so against the case that matters
    -- a platform reporting no fonts at all.
    """
    font = QFont(display_family(), size)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    return font

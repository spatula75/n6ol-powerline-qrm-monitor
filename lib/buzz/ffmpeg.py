"""Finding ffmpeg and running it.

Everything in the monitor that shells out to ffmpeg comes through here, so that the
question "where is ffmpeg and did it work" is answered once. What each caller asks it
to *do* stays with the caller: buzz.render owns the encode, buzz.loudness owns the
measurement, and neither knows how the other invokes it.

ffmpeg is optional. It is needed for rendering a replay to video and for the loudness
probe that feeds it, and for nothing else, so nothing here is imported and no binary is
looked for unless one of those is asked for.
"""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    """ffmpeg could not be found, could not be run, or failed while running."""


def find_ffmpeg(configured: str | None = None) -> str:
    """Locate the ffmpeg binary, or say clearly why the caller cannot proceed.

    This checks PATH first, then `render.ffmpeg_path` from the config. That order
    means a normal install needs no configuration at all, and the setting exists for
    the installs that do not appear on PATH: a Windows build unzipped into a folder,
    or winget's shim directory in a terminal that has not been restarted since.

    The consequence worth knowing: a copy on PATH wins over a configured one, so the
    setting cannot be used to override a working ffmpeg with a different build. That
    has not been needed. If it ever is, the order is one line.
    """
    on_path = shutil.which('ffmpeg')
    if on_path:
        return on_path
    if configured:
        # Accept a directory as well as the binary itself. Pointing at the folder is
        # the more natural reading of "where ffmpeg is", and getting it wrong would
        # otherwise fail with a message about a file that is plainly right there.
        candidate = Path(configured)
        if candidate.is_dir():
            candidate = candidate / 'ffmpeg.exe' if (candidate / 'ffmpeg.exe').exists() \
                else candidate / 'ffmpeg'
        if candidate.exists():
            return str(candidate)
        raise FfmpegError(
            f'ffmpeg is not on PATH and render.ffmpeg_path points at {configured}, '
            'where there is no ffmpeg. Correct the setting to the folder holding '
            'ffmpeg, or install it so it appears on PATH.')
    raise FfmpegError(
        'ffmpeg was not found on PATH, and this needs it. Install it (on Windows, '
        '"winget install Gyan.FFmpeg") or set render.ffmpeg_path in the config to '
        'where it lives. ffmpeg is required only for --render and the loudness probe '
        'that feeds it; nothing else in the monitor uses it.')


def run(command: list[str], *, timeout: float = 300.0) -> str:
    """Run ffmpeg to completion and return everything it said.

    ffmpeg writes its filter summaries and its errors to stderr, so both streams are
    captured and returned together: a caller parsing a measurement and a caller
    reporting a failure want the same text.

    For invocations that produce a file, use subprocess directly. This waits for the
    process and buffers its output, which is wrong for anything being fed on a pipe.
    """
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, check=False)
    except OSError as exc:
        raise FfmpegError(
            f'Could not run ffmpeg at {command[0]}: {exc}. The file was found but the '
            'operating system refused to execute it, which usually means it is not '
            'executable, is a broken symlink, or was built for a different '
            'architecture. Running it directly from a terminal normally gives a more '
            'specific error than this one.') from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(
            f'ffmpeg did not finish within {timeout:g} s and was stopped. It was '
            f'asked to process {command[-1] if command else "nothing"}; an unusually '
            'long recording or a stalled read from a network drive would explain it. '
            'Retry, or raise the timeout at the call site.') from exc
    output = (finished.stdout or '') + (finished.stderr or '')
    if finished.returncode != 0:
        raise FfmpegError(
            f'ffmpeg exited with status {finished.returncode}. Its own explanation '
            f'follows, and is the thing to read:\n\n{output[-2000:]}')
    return output

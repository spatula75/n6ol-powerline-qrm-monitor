"""Finding and running ffmpeg, on a machine that need not have it.

Everything here mocks the boundary, so the whole file passes with no ffmpeg installed
— which is the point: it is needed by --render and the loudness probe alone, and a
contributor who uses neither should still get a green suite.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from buzz.ffmpeg import FfmpegError, find_ffmpeg, run


class TestFindFfmpeg:
    """PATH first, then the configured location.

    That order keeps a normal install configuration-free; the setting is for installs
    that do not land on PATH, like a Windows build unzipped into a folder.
    """

    def test_it_finds_ffmpeg_on_the_path(self):
        with patch('buzz.ffmpeg.shutil.which', return_value='/usr/bin/ffmpeg'):
            assert find_ffmpeg() == '/usr/bin/ffmpeg'

    def test_the_path_wins_over_the_configured_location(self, tmp_path):
        """Documented consequence of searching PATH first: the setting cannot override
        a working ffmpeg with a different build."""
        configured = tmp_path / 'ffmpeg.exe'
        configured.write_bytes(b'')
        with patch('buzz.ffmpeg.shutil.which', return_value='/usr/bin/ffmpeg'):
            assert find_ffmpeg(str(configured)) == '/usr/bin/ffmpeg'

    def test_the_configured_location_is_used_when_the_path_has_none(self, tmp_path):
        configured = tmp_path / 'ffmpeg.exe'
        configured.write_bytes(b'')
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            assert find_ffmpeg(str(configured)) == str(configured)

    def test_a_configured_directory_is_accepted_as_well_as_a_binary(self, tmp_path):
        """"Where ffmpeg is" reads more naturally as the folder, and a config that
        failed on the folder would report no ffmpeg at a path where it plainly is."""
        (tmp_path / 'ffmpeg.exe').write_bytes(b'')
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            assert find_ffmpeg(str(tmp_path)) == str(tmp_path / 'ffmpeg.exe')

    def test_a_missing_binary_says_what_to_do_about_it(self):
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            with pytest.raises(FfmpegError, match='not found on PATH'):
                find_ffmpeg()

    def test_the_message_says_what_actually_needs_it(self):
        """A user who never renders should not be left thinking the monitor is broken
        or that they have acquired a dependency."""
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            with pytest.raises(FfmpegError, match='only for --render'):
                find_ffmpeg()

    def test_a_configured_location_with_nothing_in_it_names_the_setting(self, tmp_path):
        """The operator set something and it was wrong; the message has to say which
        setting, or they are left guessing which of the two lookups failed."""
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            with pytest.raises(FfmpegError, match='render.ffmpeg_path'):
                find_ffmpeg(str(tmp_path / 'nowhere'))


class TestRun:
    """Running ffmpeg to completion and handing back what it said."""

    def test_it_returns_both_streams_together(self):
        """ffmpeg writes filter summaries and errors to stderr and little to stdout, so
        a caller parsing a measurement and a caller reporting a failure want the same
        text rather than having to guess which stream it landed on."""
        finished = subprocess.CompletedProcess([], 0, stdout='out', stderr='err')
        with patch('buzz.ffmpeg.subprocess.run', return_value=finished):
            assert run(['ffmpeg']) == 'outerr'

    def test_a_nonzero_exit_carries_ffmpegs_own_explanation(self):
        """The operator needs what ffmpeg said, not a restatement of the exit code."""
        finished = subprocess.CompletedProcess([], 1, stdout='', stderr='No such filter')
        with patch('buzz.ffmpeg.subprocess.run', return_value=finished):
            with pytest.raises(FfmpegError, match='No such filter'):
                run(['ffmpeg'])

    def test_an_unrunnable_binary_names_the_path_that_failed(self, tmp_path):
        """Found but not runnable is a different failure from not found, and wants a
        different remedy."""
        with patch('buzz.ffmpeg.subprocess.run', side_effect=OSError('Exec format error')):
            with pytest.raises(FfmpegError, match='Could not run ffmpeg at /bad/ffmpeg'):
                run(['/bad/ffmpeg'])

    def test_a_timeout_says_what_it_was_working_on(self, tmp_path):
        """A stalled read from a network drive looks identical to a hang otherwise."""
        with patch('buzz.ffmpeg.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('ffmpeg', 300)):
            with pytest.raises(FfmpegError, match='did not finish within'):
                run(['ffmpeg', str(Path('recording.wav'))])

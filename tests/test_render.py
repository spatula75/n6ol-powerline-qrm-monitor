"""The render core, without ffmpeg and without Qt.

Everything here runs on a machine that has never had ffmpeg installed, which is the
point: the binary is required only by `--render`, so the suite must not need it.  The
process is mocked; what is under test is the arithmetic that decides where a frame
lands, the argument list handed to ffmpeg, and the refusals that happen before it is
ever started.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buzz import render
from buzz.render import (FRAME_RATE, RenderError, RenderSession, _FrameGrid,
                         _nearest_slot, ffmpeg_command, find_ffmpeg)

WIDTH, HEIGHT = 734, 248
FRAME_BYTES = WIDTH * HEIGHT * 4


def frame(fill: int = 0) -> bytes:
    return bytes([fill]) * FRAME_BYTES


class TestNearestSlot:
    """Which output slot a moment belongs to."""

    def test_a_whole_second_is_a_whole_number_of_slots(self):
        assert _nearest_slot(1.0, 30) == 30

    def test_a_tenth_of_a_second_survives_binary_representation(self):
        """0.3 * 30 is 8.999999999999998 in binary floating point.  Truncating drops a
        frame from an ordinary tenth of a second; this is why it rounds."""
        assert _nearest_slot(0.3, 30) == 9

    def test_every_tenth_of_a_second_lands_where_it_should(self):
        for tenth in range(1, 60):
            assert _nearest_slot(tenth / 10, 30) == tenth * 3

    def test_it_rounds_to_nearest_rather_than_truncating(self):
        assert _nearest_slot(0.049, 30) == 1        # 1.47 slots
        assert _nearest_slot(0.0166, 30) == 0       # 0.498 slots

    def test_it_does_not_round_half_to_even(self):
        """round() would give 0 and 2 here; the project rounds half away from zero
        wherever staying in phase matters."""
        assert _nearest_slot(0.5 / 30, 30) == 1
        assert _nearest_slot(1.5 / 30, 30) == 2


class TestFrameGrid:
    """Placing 10 fps captures onto the 30 fps grid."""

    def test_the_first_frame_occupies_the_first_slot(self):
        assert _FrameGrid().slots_before(0.0) == 0

    def test_a_tenth_of_a_second_of_display_is_three_slots(self):
        grid = _FrameGrid()
        grid.slots_before(0.0)
        assert grid.slots_before(0.1) == 3

    def test_ten_fps_capture_fills_the_grid_evenly(self):
        """Three output frames per capture, with no slot skipped and none doubled —
        which is what keeps a 30 fps file honest about 10 fps content."""
        grid = _FrameGrid()
        counts = [grid.slots_before(tenth / 10) for tenth in range(30)]
        assert counts[0] == 0
        assert set(counts[1:]) == {3}
        assert grid.filled == 29 * 3

    def test_a_capture_within_the_same_slot_writes_nothing(self):
        """Captures are not aligned to the grid and can arrive twice in one slot.  The
        answer is zero, not a duplicated frame."""
        grid = _FrameGrid()
        grid.slots_before(0.1)
        assert grid.slots_before(0.11) == 0

    def test_position_going_backwards_does_not_rewind_the_file(self):
        """Nothing should restart playback mid-render, but a negative count would
        corrupt a file already partly written, so it cannot be left to chance."""
        grid = _FrameGrid()
        grid.slots_before(2.0)
        assert grid.slots_before(0.5) == 0
        assert grid.filled == 60

    def test_the_last_frame_is_held_to_the_end_of_the_audio(self):
        """Otherwise the video stops at the final capture and ends short of its own
        sound, which reads as the picture freezing early."""
        grid = _FrameGrid()
        grid.slots_before(1.0)
        assert grid.slots_to_fill(2.0) == 30

    def test_nothing_is_owed_once_the_grid_is_full(self):
        grid = _FrameGrid()
        grid.slots_before(2.0)
        assert grid.slots_to_fill(2.0) == 0
        assert grid.slots_to_fill(1.0) == 0

    def test_placement_error_stays_within_half_a_slot(self):
        """The property the whole design rests on: a frame captured at t is shown
        within half a grid slot of t, so the conform contributes less error than the
        32 ms quantum the playback position is measured in."""
        grid = _FrameGrid()
        for capture in [0.0, 0.093, 0.211, 0.298, 0.417, 0.505, 0.62, 0.71]:
            before = grid.filled
            grid.slots_before(capture)
            shown_at = max(before, grid.filled) / FRAME_RATE
            assert abs(shown_at - capture) <= 0.5 / FRAME_RATE + 1e-9


class TestFindFfmpeg:
    """PATH first, then the configured location.

    That order keeps a normal install configuration-free; the setting is for installs
    that do not land on PATH, like a Windows build unzipped into a folder.
    """

    def test_it_finds_ffmpeg_on_the_path(self):
        with patch('buzz.render.shutil.which', return_value='/usr/bin/ffmpeg'):
            assert find_ffmpeg() == '/usr/bin/ffmpeg'

    def test_the_path_wins_over_the_configured_location(self, tmp_path):
        """Documented consequence of searching PATH first: the setting cannot override
        a working ffmpeg with a different build."""
        configured = tmp_path / 'ffmpeg.exe'
        configured.write_bytes(b'')
        with patch('buzz.render.shutil.which', return_value='/usr/bin/ffmpeg'):
            assert find_ffmpeg(str(configured)) == '/usr/bin/ffmpeg'

    def test_the_configured_location_is_used_when_the_path_has_none(self, tmp_path):
        configured = tmp_path / 'ffmpeg.exe'
        configured.write_bytes(b'')
        with patch('buzz.render.shutil.which', return_value=None):
            assert find_ffmpeg(str(configured)) == str(configured)

    def test_a_configured_directory_is_accepted_as_well_as_a_binary(self, tmp_path):
        """"Where ffmpeg is" reads more naturally as the folder, and a config that
        failed on the folder would report no ffmpeg at a path where it plainly is."""
        (tmp_path / 'ffmpeg.exe').write_bytes(b'')
        with patch('buzz.render.shutil.which', return_value=None):
            assert find_ffmpeg(str(tmp_path)) == str(tmp_path / 'ffmpeg.exe')

    def test_a_missing_binary_says_what_to_do_about_it(self):
        with patch('buzz.render.shutil.which', return_value=None):
            with pytest.raises(RenderError, match='not found on PATH'):
                find_ffmpeg()

    def test_the_message_says_it_is_needed_only_for_rendering(self):
        """A user who does not render should not be left thinking the monitor is
        broken or that they now have a dependency to satisfy."""
        with patch('buzz.render.shutil.which', return_value=None):
            with pytest.raises(RenderError, match='only for --render'):
                find_ffmpeg()

    def test_a_configured_location_with_nothing_in_it_names_the_setting(self, tmp_path):
        """The operator set something and it was wrong; the message has to say which
        setting, or they are left guessing which of the two lookups failed."""
        with patch('buzz.render.shutil.which', return_value=None):
            with pytest.raises(RenderError, match='render.ffmpeg_path'):
                find_ffmpeg(str(tmp_path / 'nowhere'))


class TestFfmpegCommand:
    """The argument list, asserted as a value rather than by running anything."""

    @pytest.fixture
    def command(self, tmp_path):
        return ffmpeg_command('ffmpeg', tmp_path / 'out.mp4', tmp_path / 'in.wav',
                              WIDTH, HEIGHT)

    def test_the_frames_come_in_on_the_pipe(self, command):
        assert '-i' in command and 'pipe:0' in command

    def test_the_audio_comes_from_the_recording_itself(self, command, tmp_path):
        """Not piped and not re-encoded from the ring buffer: the file on disk is
        already the audio, and handing it over whole removes it from the sync
        problem entirely."""
        assert str(tmp_path / 'in.wav') in command

    def test_frames_are_declared_at_the_output_rate(self, command):
        """They arrive already conformed, so the declared rate is the real one — the
        pipe carries no timestamps that could say otherwise."""
        assert command[command.index('-r') + 1] == str(FRAME_RATE)

    def test_the_frame_size_is_declared(self, command):
        assert f'{WIDTH}x{HEIGHT}' in command

    def test_it_encodes_for_firefox(self, command):
        """yuv420p and Main profile: Firefox will not decode High 4:4:4 Predictive,
        which is what rules out the 4:4:4 this content would otherwise like."""
        assert command[command.index('-pix_fmt', command.index('-c:v')) + 1] == 'yuv420p'
        assert command[command.index('-profile:v') + 1] == 'main'

    def test_the_moov_atom_goes_at_the_front(self, command):
        """A Firefox target means the file may be served over HTTP, where a trailing
        moov atom forces the whole download before the first frame appears."""
        assert command[command.index('-movflags') + 1] == '+faststart'

    def test_there_is_a_keyframe_every_second(self, command):
        """Chosen for scrubbing rather than for size: at this frame size an I-frame is
        small enough that the interval is not a size decision at all."""
        assert command[command.index('-g') + 1] == str(FRAME_RATE)

    def test_the_audio_is_resampled_for_players(self, command):
        assert command[command.index('-ar') + 1] == '48000'
        assert command[command.index('-b:a') + 1] == '192k'

    def test_the_recordings_tags_are_carried_into_the_container(self, command):
        """Input 1 is the .wav.  Its LIST/INFO tags say what the event was, which
        station heard it, when, and against what calibration -- without them the file
        reaches whoever is shown it carrying nothing but pixels."""
        assert command[command.index('-map_metadata') + 1] == '1'

    def test_the_chapters_are_not(self, command):
        """wavmeta marks the lock with a `cue` chunk; ffmpeg reads it as a chapter and
        the mp4 muxer writes chapters as a track, so one marker became a third stream
        of bin_data. The offset survives in the comment tag as lead_in_seconds."""
        assert command[command.index('-map_chapters') + 1] == '-1'

    def test_it_refuses_to_overwrite_without_prompting(self, command):
        """-n rather than -y: a prompt on a pipe would hang rather than ask."""
        assert '-n' in command and '-y' not in command

    def test_no_gain_filter_when_there_is_no_gain(self, command):
        assert '-af' not in command

    def test_playback_gain_is_applied_to_the_rendered_audio(self, tmp_path):
        """Deliberately unlike live playback, where gain reaches the speakers only.
        Nothing measured is involved in a rendered file, and a demo nobody can hear is
        useless."""
        command = ffmpeg_command('ffmpeg', tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                 WIDTH, HEIGHT, gain_db=6.0)
        assert command[command.index('-af') + 1] == 'volume=6.0dB'


class TestRenderSession:
    """Feeding the process, with the process mocked out."""

    @pytest.fixture
    def started(self, tmp_path):
        """A session with ffmpeg found and Popen replaced, already started."""
        process = MagicMock()
        process.stdin = MagicMock()
        process.wait.return_value = 0
        with patch('buzz.render.shutil.which', return_value='ffmpeg'), \
             patch('buzz.render.subprocess.Popen', return_value=process) as popen:
            session = RenderSession(tmp_path / 'out.mp4', tmp_path / 'in.wav',
                                    WIDTH, HEIGHT)
            session.start()
            yield session, process, popen

    def test_it_starts_ffmpeg_with_a_pipe_for_frames(self, started):
        _, _, popen = started
        assert popen.call_args.kwargs['stdin'] is subprocess.PIPE

    def test_it_will_not_overwrite_an_existing_file(self, tmp_path):
        """Checked here rather than left to ffmpeg so the operator gets a message
        naming the file, before a process is started."""
        (tmp_path / 'out.mp4').write_bytes(b'')
        with patch('buzz.render.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'out.mp4', tmp_path / 'in.wav',
                                    WIDTH, HEIGHT)
            with pytest.raises(RenderError, match='already exists'):
                session.start()

    def test_a_missing_ffmpeg_fails_before_anything_is_built(self, tmp_path):
        with patch('buzz.render.shutil.which', return_value=None):
            with pytest.raises(RenderError):
                RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav', WIDTH, HEIGHT)

    def test_a_wrongly_sized_frame_is_refused(self, started):
        """Silently writing a short frame desynchronises the whole rest of the stream,
        since rawvideo has no framing of its own to resynchronise against."""
        session, _, _ = started
        with pytest.raises(RenderError, match='expected'):
            session.submit(b'too short', 0.0)

    def test_the_first_frame_writes_nothing_yet(self, started):
        """It has not been on screen for any time yet; what it is owed is decided by
        where the *next* capture lands."""
        session, process, _ = started
        session.submit(frame(1), 0.0)
        process.stdin.write.assert_not_called()

    def test_a_late_first_frame_backfills_rather_than_starting_late(self, started):
        """The bug this exists for lost the opening 1.5 s of a render and desynched
        everything after it.  A first capture that arrives after playback has already
        moved fills the slots it missed with itself, so the video still starts at
        zero and still lines up with its audio."""
        session, process, _ = started
        session.submit(frame(3), 1.0)
        written = b''.join(call.args[0] for call in process.stdin.write.call_args_list)
        assert len(written) == FRAME_BYTES * FRAME_RATE
        assert written[:1] == bytes([3])
        assert session.frames_written == FRAME_RATE

    def test_the_counter_never_claims_frames_that_were_not_written(self, started):
        """The same bug also lied about it: the grid advanced over slots that _write
        skipped for want of anything to write, so the log reported 89 frames into a
        file holding 46."""
        session, process, _ = started
        session.submit(frame(1), 0.7)
        session.submit(frame(2), 1.3)
        session.finish(2.0)
        written = sum(len(c.args[0]) for c in process.stdin.write.call_args_list)
        assert written == session.frames_written * FRAME_BYTES

    def test_a_frame_is_written_for_the_time_it_was_on_screen(self, started):
        session, process, _ = started
        session.submit(frame(1), 0.0)
        session.submit(frame(2), 0.1)
        written = b''.join(call.args[0] for call in process.stdin.write.call_args_list)
        assert written == frame(1) * 3

    def test_the_earlier_frame_is_the_one_written(self, started):
        """The frame captured at t shows what was on screen up to t, so the slots
        before t belong to the frame before it.  Getting this backwards is exactly the
        bug that shows up as video running ahead of sound."""
        session, process, _ = started
        session.submit(frame(7), 0.0)
        session.submit(frame(9), 0.1)
        first = process.stdin.write.call_args_list[0].args[0]
        assert first[:1] == bytes([7])

    def test_finishing_holds_the_last_frame_to_the_end(self, started):
        session, process, _ = started
        session.submit(frame(1), 0.0)
        session.submit(frame(2), 1.0)
        session.finish(2.0)
        assert session.frames_written == 60

    def test_finishing_closes_the_pipe_and_waits(self, started):
        """Waiting matters: +faststart rewrites the file at the end to move the moov
        atom, and a process killed before that leaves an .mp4 no player will open."""
        session, process, _ = started
        session.submit(frame(1), 0.0)
        session.finish(0.1)
        process.stdin.close.assert_called_once()
        process.wait.assert_called_once()

    def test_a_failing_ffmpeg_is_reported(self, started):
        session, process, _ = started
        process.wait.return_value = 1
        session.submit(frame(1), 0.0)
        with pytest.raises(RenderError, match='exited with status 1'):
            session.finish(0.1)

    def test_the_failure_says_where_to_look(self, started):
        """Context, cause and remedy: which file, that ffmpeg explains itself on
        stderr, and that the command is in the log to re-run by hand."""
        session, process, _ = started
        process.wait.return_value = 1
        session.submit(frame(1), 0.0)
        with pytest.raises(RenderError) as raised:
            session.finish(0.1)
        message = str(raised.value)
        assert 'out.mp4' in message
        assert 'stderr' in message
        assert 'log' in message

    def test_a_broken_pipe_is_reported_as_a_render_error(self, started):
        """ffmpeg dying mid-render surfaces here as a write failure; the operator needs
        to be pointed at its output rather than at a bare BrokenPipeError."""
        session, process, _ = started
        process.stdin.write.side_effect = BrokenPipeError('gone')
        session.submit(frame(1), 0.0)
        with pytest.raises(RenderError, match='stopped accepting frames'):
            session.submit(frame(2), 0.1)

    def test_an_unrunnable_ffmpeg_is_reported_with_its_path(self, tmp_path):
        """Found but not runnable is a different failure from not found, and needs a
        different remedy — so it gets its own message naming the path that failed."""
        with patch('buzz.render.shutil.which', return_value='/bad/ffmpeg'), \
             patch('buzz.render.subprocess.Popen',
                   side_effect=OSError('Exec format error')):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
            with pytest.raises(RenderError, match='Could not start ffmpeg'):
                session.start()

    def test_a_pipe_already_closed_does_not_mask_the_real_failure(self, started):
        """If ffmpeg has already exited, closing its stdin raises — and that error is
        a symptom. The exit status underneath is the useful thing, so the close is
        allowed to fail and the status is what gets reported."""
        session, process, _ = started
        process.stdin.close.side_effect = OSError('pipe is gone')
        process.wait.return_value = 3
        session.submit(frame(1), 0.0)
        with pytest.raises(RenderError, match='exited with status 3'):
            session.finish(0.1)

    def test_finishing_without_starting_is_harmless(self, tmp_path):
        with patch('buzz.render.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
        session.finish(1.0)

    def test_the_command_is_logged_so_it_can_be_run_by_hand(self, tmp_path, caplog):
        """The single most useful thing in the log when a render comes out wrong."""
        process = MagicMock()
        process.stdin = MagicMock()
        with patch('buzz.render.shutil.which', return_value='ffmpeg'), \
             patch('buzz.render.subprocess.Popen', return_value=process):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
            with caplog.at_level('INFO', logger=render.__name__):
                session.start()
        assert any('ffmpeg command:' in r.message for r in caplog.records)

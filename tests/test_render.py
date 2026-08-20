"""The render core, without ffmpeg and without Qt.

Everything here runs on a machine that has never had ffmpeg installed, which is the
point: the binary is required only by `--render`, so the suite must not need it.  The
process is mocked; what is under test is the arithmetic that decides where a frame
falls, the argument list handed to ffmpeg, and the refusals that happen before it is
ever started.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buzz import render
from buzz.ffmpeg import FfmpegError
from buzz.render import (FRAME_RATE, RenderError, RenderSession, _FrameGrid,
                         _nearest_slot, ffmpeg_command)

WIDTH, HEIGHT = 742, 248
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
        """Three output frames per capture, with no slot skipped and none doubled -
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
        """They arrive already conformed, so the declared rate is the real one - the
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
        assert command[command.index('-af') + 1] == 'volume=6.00dB'


class TestRenderSession:
    """Feeding the process, with the process mocked out."""

    @pytest.fixture
    def started(self, tmp_path):
        """A session with ffmpeg found and Popen replaced, already started."""
        process = MagicMock()
        process.stdin = MagicMock()
        process.wait.return_value = 0
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'), \
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
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'out.mp4', tmp_path / 'in.wav',
                                    WIDTH, HEIGHT)
            with pytest.raises(RenderError, match='already exists'):
                session.start()

    def test_a_missing_ffmpeg_fails_before_anything_is_built(self, tmp_path):
        """Raised by buzz.ffmpeg as the parent FfmpegError, not RenderError: the
        toolchain being absent is not a fact about rendering."""
        with patch('buzz.ffmpeg.shutil.which', return_value=None):
            with pytest.raises(FfmpegError):
                RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav', WIDTH, HEIGHT)

    def test_a_wrongly_sized_frame_is_refused(self, started):
        """Silently writing a short frame desynchronises the whole rest of the stream,
        since rawvideo has no framing of its own to resynchronise against."""
        session, _, _ = started
        with pytest.raises(RenderError, match='expected'):
            session.submit(b'too short', 0.0)

    def test_the_first_frame_writes_nothing_yet(self, started):
        """It has not been on screen for any time yet; what it is owed is decided by
        where the *next* capture falls."""
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
        different remedy - so it gets its own message naming the path that failed."""
        with patch('buzz.ffmpeg.shutil.which', return_value='/bad/ffmpeg'), \
             patch('buzz.render.subprocess.Popen',
                   side_effect=OSError('Exec format error')):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
            with pytest.raises(RenderError, match='Could not start ffmpeg'):
                session.start()

    def test_a_pipe_already_closed_does_not_mask_the_real_failure(self, started):
        """If ffmpeg has already exited, closing its stdin raises - and that error is
        a symptom. The exit status underneath is the useful thing, so the close is
        allowed to fail and the status is what gets reported."""
        session, process, _ = started
        process.stdin.close.side_effect = OSError('pipe is gone')
        process.wait.return_value = 3
        session.submit(frame(1), 0.0)
        with pytest.raises(RenderError, match='exited with status 3'):
            session.finish(0.1)

    def test_finishing_without_starting_is_harmless(self, tmp_path):
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
        session.finish(1.0)

    def test_the_command_is_logged_so_it_can_be_run_by_hand(self, tmp_path, caplog):
        """The single most useful thing in the log when a render comes out wrong."""
        process = MagicMock()
        process.stdin = MagicMock()
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'), \
             patch('buzz.render.subprocess.Popen', return_value=process):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
            with caplog.at_level('INFO', logger=render.__name__):
                session.start()
        assert any('ffmpeg command:' in r.message for r in caplog.records)


class TestRenderedComment:
    """The recording's settings line, with the gain this render applied added to it.

    A demo raised by 20-odd dB should say so: someone shown the video has no other way
    to know the audio is not at the level it was recorded at.
    """

    def test_the_gain_is_added_to_the_recordings_settings(self, tmp_path):
        source = tmp_path / 'event.wav'
        with patch('buzz.render.wavmeta.read_settings',
                   return_value={'sample_rate': '16000', 'ended': 'capped'}):
            comment = render.rendered_comment(source, 19.0)
        assert 'render_gain_db=+19.0' in comment
        assert 'sample_rate=16000' in comment, 'the original settings must survive'

    def test_a_negative_gain_keeps_its_sign(self, tmp_path):
        with patch('buzz.render.wavmeta.read_settings', return_value={'ended': 'capped'}):
            assert 'render_gain_db=-2.2' in render.rendered_comment(tmp_path / 'e.wav', -2.2)

    def test_a_recording_with_no_settings_is_left_alone(self, tmp_path):
        """One made by other software, or by a version predating the settings line.
        Returning None leaves the copied tags untouched rather than inventing a
        comment that claims to describe a recording it does not."""
        with patch('buzz.render.wavmeta.read_settings', return_value={}):
            assert render.rendered_comment(tmp_path / 'e.wav', 19.0) is None

    def test_an_unreadable_recording_warns_and_carries_on(self, tmp_path, caplog):
        """Failing to annotate the metadata is not a reason to abandon a render that
        would otherwise be fine."""
        with patch('buzz.render.wavmeta.read_settings', side_effect=OSError('gone')):
            with caplog.at_level('WARNING', logger='buzz.render'):
                assert render.rendered_comment(tmp_path / 'e.wav', 19.0) is None
        assert 'tags unchanged' in caplog.text

    def test_the_comment_reaches_the_command_after_map_metadata(self, tmp_path):
        """Order is the whole mechanism: -map_metadata copies the recording's comment,
        and this has to override it rather than be overridden."""
        command = ffmpeg_command('ffmpeg', tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                 WIDTH, HEIGHT, comment='sample_rate=16000 render_gain_db=+19.0')
        assert command.index('-metadata') > command.index('-map_metadata')
        assert command[command.index('-metadata') + 1].startswith('comment=')

    def test_no_metadata_override_when_there_is_nothing_to_say(self, tmp_path):
        command = ffmpeg_command('ffmpeg', tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                 WIDTH, HEIGHT)
        assert '-metadata' not in command

    def test_the_gain_filter_is_not_a_float_repr(self, tmp_path):
        """A measured gain is a float like 18.990000000000002, which would otherwise
        appear both in the filter and in the command the operator is invited to paste
        into a terminal."""
        command = ffmpeg_command('ffmpeg', tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                 WIDTH, HEIGHT, gain_db=18.990000000000002)
        assert command[command.index('-af') + 1] == 'volume=18.99dB'


class TestAbort:
    """Giving up on a render without leaving ffmpeg behind.

    The path that reaches this is a frame capture that raised, which has already lost
    the render.  Nothing here is about salvaging the file -- it cannot be salvaged,
    having no moov atom -- and everything is about the subprocess: Python does not reap
    its children on the way out, so an ffmpeg still holding an open pipe outlives the
    monitor and goes on writing.
    """

    @pytest.fixture
    def started(self, tmp_path):
        process = MagicMock()
        process.stdin = MagicMock()
        process.wait.return_value = 0
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'), \
             patch('buzz.render.subprocess.Popen', return_value=process):
            session = RenderSession(tmp_path / 'out.mp4', tmp_path / 'in.wav',
                                    WIDTH, HEIGHT)
            session.start()
            yield session, process

    def test_it_closes_the_pipe(self, started):
        """ffmpeg reads until end of input; without this it waits forever for frames
        that are never coming."""
        session, process = started
        session.abort()
        process.stdin.close.assert_called_once()

    def test_it_waits_for_the_process(self, started):
        session, process = started
        session.abort()
        process.wait.assert_called_once()

    def test_it_writes_no_further_frames(self, started):
        """finish() pads out to the recording's length; abort() must not, or a render
        that failed at two seconds still costs the whole encode."""
        session, process = started
        session.submit(frame(1), 0.0)
        process.stdin.write.reset_mock()
        session.abort()
        process.stdin.write.assert_not_called()

    def test_it_is_idempotent(self, started):
        """Wired to a Qt failure path and to aboutToQuit, so it can be reached twice."""
        session, process = started
        session.abort()
        session.abort()
        process.wait.assert_called_once()

    def test_a_session_never_started_is_a_no_op(self, tmp_path):
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'o.mp4', tmp_path / 'i.wav',
                                    WIDTH, HEIGHT)
        session.abort()          # no process to abandon; must not raise

    def test_it_does_not_raise_when_the_pipe_is_already_gone(self, started):
        """The usual reason for aborting is that ffmpeg died, which is exactly when
        closing its stdin fails. Raising here would replace the real error."""
        session, process = started
        process.stdin.close.side_effect = OSError('already closed')
        session.abort()

    def test_an_ffmpeg_that_will_not_exit_is_killed(self, started):
        """Nothing it is still doing is of any use, and the monitor is on its way out."""
        session, process = started
        process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 10.0), 0]
        session.abort()
        process.kill.assert_called_once()

    def test_finish_after_an_abort_does_nothing(self, started):
        """aboutToQuit calls stop(), which calls finish(); the abort has already
        disowned the process, and waiting on it again would hang."""
        session, process = started
        session.abort()
        process.wait.reset_mock()
        session.finish(10.0)
        process.wait.assert_not_called()


class TestSourceChannels:
    """How many channels the recording has, read from the header before encoding."""

    def _wav(self, path: Path, channels: int) -> Path:
        import wave
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b'\x00\x00' * 64 * channels)
        return path

    def test_a_recording_this_program_made_is_mono(self, tmp_path):
        assert render._source_channels(self._wav(tmp_path / 'mono.wav', 1)) == 1

    def test_a_file_from_elsewhere_may_be_stereo(self, tmp_path):
        """Which is what puts pan=mono|c0=c0 in the filter chain: the display was drawn
        from channel 0, so the video's audio has to be channel 0 too."""
        assert render._source_channels(self._wav(tmp_path / 'stereo.wav', 2)) == 2

    def test_the_channel_count_reaches_the_command(self, tmp_path):
        source = self._wav(tmp_path / 'stereo.wav', 2)
        with patch('buzz.ffmpeg.shutil.which', return_value='ffmpeg'):
            session = RenderSession(tmp_path / 'out.mp4', source, WIDTH, HEIGHT)
        assert 'pan=mono|c0=c0' in ' '.join(session._command)

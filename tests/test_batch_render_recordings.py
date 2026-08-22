"""Tests for scripts/batch_render_recordings.py: planning, running, and reporting.

No test here renders anything.  A render plays a recording through in real time, so
the real thing belongs in nobody's unit suite; subprocess.run is the boundary and it
is mocked.  What is tested is everything around it: which recordings become jobs, what
command each one produces, what happens to a failure, and that --jobs actually runs
more than one at a time.
"""

import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import batch_render_recordings as batch  # noqa: E402


def _write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> Path:
    """A .wav of a given length, in the 16-bit mono the recorder writes."""
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(int(seconds * sample_rate), dtype='<i2').tobytes())
    return path


def _job(tmp_path: Path, name: str = 'event.wav', seconds: float = 5.0) -> batch.Job:
    return batch.Job(_write_wav(tmp_path / name, seconds),
                     tmp_path / 'renders' / f'{Path(name).stem}.mp4', seconds)


def _completed(returncode: int = 0, stderr: str = '') -> MagicMock:
    finished = MagicMock()
    finished.returncode = returncode
    finished.stdout = ''
    finished.stderr = stderr
    return finished


class TestRecordingSeconds:
    def test_reads_the_length_from_the_header(self, tmp_path):
        assert batch.recording_seconds(_write_wav(tmp_path / 'a.wav', 2.5)) == 2.5

    def test_an_unreadable_file_measures_zero(self, tmp_path):
        """Not an error here, deliberately.

        The renderer reports a corrupt or missing file far better than this can, naming
        what is actually wrong with it.  Zero sends the file through on the shortest
        timeout, which suits something about to fail immediately anyway.
        """
        broken = tmp_path / 'broken.wav'
        broken.write_bytes(b'this is not a wav')
        assert batch.recording_seconds(broken) == 0.0

    def test_a_missing_file_measures_zero(self, tmp_path):
        assert batch.recording_seconds(tmp_path / 'absent.wav') == 0.0


class TestPlan:
    def test_every_recording_becomes_a_job_by_default(self, tmp_path):
        """No --max-length means render everything, however long it runs."""
        sources = [_write_wav(tmp_path / 'a.wav', 5), _write_wav(tmp_path / 'b.wav', 600)]
        jobs, skipped = batch.plan(sources, tmp_path / 'renders', max_length_s=None)
        assert [j.source.name for j in jobs] == ['a.wav', 'b.wav']
        assert skipped == []

    def test_max_length_skips_the_long_ones(self, tmp_path):
        sources = [_write_wav(tmp_path / 'short.wav', 5), _write_wav(tmp_path / 'long.wav', 30)]
        jobs, skipped = batch.plan(sources, tmp_path / 'renders', max_length_s=20)
        assert [j.source.name for j in jobs] == ['short.wav']
        assert skipped[0][0].name == 'long.wav' and '20 s limit' in skipped[0][1]

    def test_a_recording_exactly_at_the_limit_is_rendered(self, tmp_path):
        """The limit is a maximum, not a threshold to be under."""
        jobs, skipped = batch.plan([_write_wav(tmp_path / 'a.wav', 20)],
                                   tmp_path / 'renders', max_length_s=20)
        assert len(jobs) == 1 and skipped == []

    def test_an_existing_video_is_skipped(self, tmp_path):
        """What makes an interrupted batch resumable rather than a restart."""
        renders = tmp_path / 'renders'
        renders.mkdir()
        (renders / 'a.mp4').write_bytes(b'video')
        jobs, skipped = batch.plan([_write_wav(tmp_path / 'a.wav', 5)], renders, None)
        assert jobs == [] and skipped[0][1] == 'already rendered'

    def test_the_video_is_named_after_the_recording(self, tmp_path):
        jobs, _ = batch.plan([_write_wav(tmp_path / 'event-20260821-151720-0700.wav', 5)],
                             tmp_path / 'renders', None)
        assert jobs[0].output == tmp_path / 'renders' / 'event-20260821-151720-0700.mp4'


class TestTheTimeout:
    def test_it_scales_with_the_recording(self, tmp_path):
        """A render cannot beat real time, so the length of the file is its floor."""
        long_job = batch.Job(tmp_path / 'a.wav', tmp_path / 'a.mp4', 600)
        assert long_job.timeout_s > 600, (
            'A timeout below the length of the recording would kill every long render '
            'part-way through, since a render plays the recording at real speed.'
        )

    def test_a_zero_length_recording_still_gets_a_usable_timeout(self, tmp_path):
        """Startup costs the same whatever the file holds, so the floor is not zero."""
        assert batch.Job(tmp_path / 'a.wav', tmp_path / 'a.mp4', 0).timeout_s >= 60


class TestRenderCommand:
    def test_it_runs_the_monitor_on_this_interpreter(self, tmp_path):
        command = batch.render_command(_job(tmp_path))
        assert command[:3] == [sys.executable, '-m', 'buzz.main'], (
            'The child has to be this same interpreter.  Anything else runs the batch '
            'against whatever python happens to be first on PATH, with its own '
            'packages and possibly no PySide6 at all.'
        )

    def test_it_renders_headless_and_measures_the_gain(self, tmp_path):
        command = batch.render_command(_job(tmp_path))
        assert '--headless' in command, (
            'Without --headless a batch throws one window in front of the operator per '
            'recording.  With it, --render paints the same frames offscreen.'
        )
        assert command[command.index('--playback-gain') + 1] == 'auto'
        assert command[command.index('--render') + 1].endswith('.mp4')

    def test_the_child_can_import_buzz(self, tmp_path):
        """lib/ is not an installed package, so the child needs it on PYTHONPATH."""
        environment = batch.child_environment()
        assert environment['PYTHONPATH'].endswith('lib')
        assert environment['QT_QPA_PLATFORM'] == 'offscreen'


class TestRender:
    def test_a_written_file_and_a_zero_exit_is_a_success(self, tmp_path):
        job = _job(tmp_path)
        job.output.parent.mkdir()
        job.output.write_bytes(b'video')
        with patch('batch_render_recordings.subprocess.run', return_value=_completed()):
            outcome = batch.render(job)
        assert outcome.ok is True and outcome.message == ''

    def test_a_nonzero_exit_is_a_failure_quoting_the_child(self, tmp_path):
        job = _job(tmp_path)
        with patch('batch_render_recordings.subprocess.run',
                   return_value=_completed(2, 'ffmpeg was not found on PATH')):
            outcome = batch.render(job)
        assert outcome.ok is False
        assert 'ffmpeg was not found' in outcome.message, (
            'The child says why it failed.  Swallowing that leaves the operator with a '
            'count of failures and no way to tell what went wrong.'
        )

    def test_a_zero_exit_that_wrote_nothing_is_a_failure(self, tmp_path):
        """Exit code alone is not proof: the file is what was asked for."""
        with patch('batch_render_recordings.subprocess.run', return_value=_completed(0)):
            outcome = batch.render(_job(tmp_path))
        assert outcome.ok is False and 'no file was written' in outcome.message

    def test_a_hung_render_is_a_failure_rather_than_a_wait(self, tmp_path):
        with patch('batch_render_recordings.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('cmd', 60)):
            outcome = batch.render(_job(tmp_path))
        assert outcome.ok is False and 'no exit within' in outcome.message

    def test_a_failure_leaves_no_video_behind(self, tmp_path):
        """Otherwise the next run skips a recording that never rendered.

        A render that dies part-way can leave a partial .mp4, and plan() treats any
        existing output as done.  The recording would then be silently missing from
        every later batch, which is worse than the original failure because nothing
        reports it a second time.
        """
        job = _job(tmp_path)
        job.output.parent.mkdir()
        job.output.write_bytes(b'partial')
        with patch('batch_render_recordings.subprocess.run', return_value=_completed(1)):
            batch.render(job)
        assert not job.output.exists(), (
            'A partial video left behind makes the recording look rendered, so the '
            'retry skips it.'
        )

    def test_a_timed_out_render_leaves_no_video_behind(self, tmp_path):
        job = _job(tmp_path)
        job.output.parent.mkdir()
        job.output.write_bytes(b'partial')
        with patch('batch_render_recordings.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('cmd', 60)):
            batch.render(job)
        assert not job.output.exists()


class TestRunJobs:
    def test_every_job_is_rendered_and_reported(self, tmp_path):
        jobs = [_job(tmp_path, f'{n}.wav') for n in 'abc']
        lines = []
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)):
            outcomes = batch.run_jobs(jobs, workers=1, announce=lines.append)
        assert len(outcomes) == 3
        assert [line.startswith(f'[{n}/3]') for n, line in enumerate(lines, start=1)] == [True] * 3

    def test_jobs_run_concurrently_when_asked(self, tmp_path):
        """--jobs has to actually overlap the work, not just accept the number.

        The renders are separate processes, so this thread pool exists purely to have
        more than one of them outstanding at a time.  A pool that ran them one after
        another would look identical from the outside: same outcomes, same order, only
        slower.  So this counts how many were in flight at once.
        """
        import threading
        in_flight, peak, lock = 0, 0, threading.Lock()
        started = threading.Barrier(3, timeout=5)

        def slow_render(job):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            started.wait()          # only passes if three are running together
            with lock:
                in_flight -= 1
            return batch.Outcome(job, True, 1.0)

        jobs = [_job(tmp_path, f'{n}.wav') for n in 'abc']
        with patch('batch_render_recordings.render', side_effect=slow_render):
            batch.run_jobs(jobs, workers=3, announce=lambda _: None)
        assert peak == 3, f'Only {peak} render(s) ran at once with --jobs 3.'

    def test_one_worker_means_one_at_a_time(self, tmp_path):
        import threading
        in_flight, peak, lock = 0, 0, threading.Lock()

        def counted_render(job):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time_to_overlap = 0.02
            import time as _time
            _time.sleep(time_to_overlap)
            with lock:
                in_flight -= 1
            return batch.Outcome(job, True, 1.0)

        jobs = [_job(tmp_path, f'{n}.wav') for n in 'abc']
        with patch('batch_render_recordings.render', side_effect=counted_render):
            batch.run_jobs(jobs, workers=1, announce=lambda _: None)
        assert peak == 1, (
            f'{peak} renders overlapped at --jobs 1.  The default has to stay '
            'sequential, since each render is a whole monitor process.'
        )


class TestSummary:
    def test_it_counts_each_kind(self, tmp_path):
        jobs = [_job(tmp_path, f'{n}.wav') for n in 'abc']
        outcomes = [batch.Outcome(jobs[0], True, 1.0),
                    batch.Outcome(jobs[1], False, 1.0, 'ffmpeg exploded')]
        text = batch.summary(outcomes, [(jobs[2].source, 'already rendered')],
                             tmp_path / 'renders', elapsed_s=120)
        assert '1 rendered, 1 skipped, 1 failed' in text
        assert '2.0 min' in text

    def test_each_failure_is_named(self, tmp_path):
        """A count of failures with no names is not actionable."""
        job = _job(tmp_path)
        text = batch.summary([batch.Outcome(job, False, 1.0, 'ffmpeg exploded\nline two')],
                             [], tmp_path, elapsed_s=1)
        assert 'event.wav: ffmpeg exploded' in text

    def test_a_failure_with_nothing_to_say_is_still_named(self, tmp_path):
        text = batch.summary([batch.Outcome(_job(tmp_path), False, 1.0, '')],
                             [], tmp_path, elapsed_s=1)
        assert 'event.wav: unknown' in text


class TestMain:
    def _recordings(self, tmp_path, count=2, seconds=5.0):
        for n in range(count):
            _write_wav(tmp_path / f'event{n}.wav', seconds)
        return tmp_path

    def test_it_renders_what_it_finds(self, tmp_path, capsys):
        source_dir = self._recordings(tmp_path)
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)) as rendered:
            assert batch.main(['--recordings', str(source_dir)]) == 0
        assert rendered.call_count == 2
        assert '2 rendered' in capsys.readouterr().out

    def test_a_failed_render_makes_the_whole_run_fail(self, tmp_path, capsys):
        """The exit code is what an operator running this from a shell script reads."""
        source_dir = self._recordings(tmp_path, count=1)
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, False, 1.0, 'no ffmpeg')):
            assert batch.main(['--recordings', str(source_dir)]) == 1

    def test_an_empty_directory_says_so_rather_than_succeeding_silently(self, tmp_path, capsys):
        """A batch that rendered nothing because it looked in the wrong place must not
        report success.  The operator would come back to an empty directory and no
        indication of why."""
        assert batch.main(['--recordings', str(tmp_path)]) == 1
        assert 'No .wav recordings' in capsys.readouterr().out

    def test_nothing_left_to_do_is_a_success(self, tmp_path, capsys):
        """Every recording already rendered is the resumed-batch case, not a failure."""
        source_dir = self._recordings(tmp_path, count=1)
        renders = source_dir / batch.DEFAULT_OUTPUT_NAME
        renders.mkdir()
        (renders / 'event0.mp4').write_bytes(b'video')
        assert batch.main(['--recordings', str(source_dir)]) == 0
        assert 'Nothing to render' in capsys.readouterr().out

    def test_limit_stops_after_the_given_count(self, tmp_path):
        source_dir = self._recordings(tmp_path, count=5)
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)) as rendered:
            batch.main(['--recordings', str(source_dir), '--limit', '2'])
        assert rendered.call_count == 2

    def test_max_length_reaches_the_plan(self, tmp_path):
        _write_wav(tmp_path / 'short.wav', 5)
        _write_wav(tmp_path / 'long.wav', 40)
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)) as rendered:
            batch.main(['--recordings', str(tmp_path), '--max-length', '20'])
        assert [c.args[0].source.name for c in rendered.call_args_list] == ['short.wav']

    def test_the_output_directory_is_created(self, tmp_path):
        source_dir = self._recordings(tmp_path, count=1)
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)):
            batch.main(['--recordings', str(source_dir)])
        assert (source_dir / batch.DEFAULT_OUTPUT_NAME).is_dir()

    def test_the_output_directory_can_be_chosen(self, tmp_path):
        source_dir = self._recordings(tmp_path, count=1)
        elsewhere = tmp_path / 'videos'
        with patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)) as rendered:
            batch.main(['--recordings', str(source_dir), '--output-dir', str(elsewhere)])
        assert rendered.call_args.args[0].output.parent == elsewhere

    def test_it_falls_back_to_the_configured_recording_directory(self, tmp_path):
        """Nobody should have to name the directory the monitor already writes to."""
        self._recordings(tmp_path, count=1)
        with patch('batch_render_recordings.default_recordings_directory',
                   return_value=tmp_path), \
             patch('batch_render_recordings.render',
                   side_effect=lambda j: batch.Outcome(j, True, 1.0)) as rendered:
            batch.main([])
        assert rendered.call_count == 1


class TestArgumentChecking:
    @pytest.mark.parametrize('flag', ['--jobs', '--limit'])
    def test_zero_and_negative_counts_are_refused(self, flag):
        """--jobs 0 would otherwise reach ThreadPoolExecutor and raise there instead,
        with a message about max_workers rather than about the flag that was typed."""
        with pytest.raises(SystemExit):
            batch.main([flag, '0'])
        with pytest.raises(SystemExit):
            batch.main([flag, '-3'])


class TestTheConfiguredDirectory:
    def test_it_comes_from_the_station_config(self, tmp_path):
        config = MagicMock()
        config.recording.directory_path.return_value = tmp_path / 'recordings'
        with patch('batch_render_recordings.CONFIG_PATH') as config_path, \
             patch('batch_render_recordings.BuzzConfig') as buzz_config:
            config_path.exists.return_value = True
            buzz_config.from_toml.return_value = config
            assert batch.default_recordings_directory() == tmp_path / 'recordings'
        config.recording.directory_path.assert_called_once_with(config.station)

    def test_defaults_are_used_when_there_is_no_config_file(self, tmp_path):
        with patch('batch_render_recordings.CONFIG_PATH') as config_path:
            config_path.exists.return_value = False
            assert isinstance(batch.default_recordings_directory(), Path)

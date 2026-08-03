"""Tests for the pure logic in tools/release_render_check.py.

Everything that shells out to ffmpeg, ffprobe, or a second Python process is mocked
at that boundary, the way tests/test_render.py and tests/test_ffmpeg.py mock theirs -
this file is exercised with no ffmpeg installed and proves the plumbing, not that
ffmpeg's filters do what their documentation says.
"""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from buzz.config import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE
from buzz.ffmpeg import FfmpegError

from tools.release_render_check import (
    RenderAttempt,
    RenderCheck,
    age_days,
    audio_levels,
    check_render,
    count_black_segments,
    decoded_frame_count,
    describe_staleness,
    find_ffprobe,
    flags_for,
    format_report,
    logged_frame_count,
    luma_series,
    main,
    make_rate_variants,
    native_sample_rate,
    newest_recording,
    render_variant,
    resolve_source,
)


def _write_wav(path: Path, rate: int = 16000) -> Path:
    import wave
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b'\x00\x00' * rate)   # one second of silence
    return path


class TestNewestRecording:
    def test_empty_directory_has_no_newest(self, tmp_path):
        assert newest_recording(tmp_path) is None

    def test_picks_the_most_recently_modified_file(self, tmp_path):
        older = _write_wav(tmp_path / 'older.wav')
        newer = _write_wav(tmp_path / 'newer.wav')
        # mtimes from wave.open() alone can land in the same tick on a fast disk;
        # force the ordering the test is actually about.
        import os
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        assert newest_recording(tmp_path) == newer

    def test_ignores_non_wav_files(self, tmp_path):
        (tmp_path / 'notes.txt').write_text('not a recording')
        assert newest_recording(tmp_path) is None


class TestAgeDays:
    def test_a_file_from_exactly_one_day_ago(self, tmp_path):
        import os
        f = _write_wav(tmp_path / 'event.wav')
        now = datetime(2026, 8, 1, 12, 0, 0)
        yesterday = now - timedelta(days=1)
        os.utime(f, (yesterday.timestamp(), yesterday.timestamp()))
        assert age_days(f, now) == pytest.approx(1.0, abs=0.01)

    def test_a_brand_new_file_is_zero_days_old(self, tmp_path):
        f = _write_wav(tmp_path / 'event.wav')
        now = datetime.fromtimestamp(f.stat().st_mtime)
        assert age_days(f, now) == pytest.approx(0.0, abs=0.01)


class TestDescribeStaleness:
    def test_no_recordings_at_all(self, tmp_path):
        message = describe_staleness(tmp_path, None, 7.0)
        assert str(tmp_path) in message
        assert 'nothing here to validate against' in message

    def test_names_the_file_the_age_and_the_threshold(self, tmp_path):
        f = _write_wav(tmp_path / 'event-old.wav')
        now = datetime.fromtimestamp(f.stat().st_mtime) + timedelta(days=10)
        message = describe_staleness(tmp_path, f, 7.0, now)
        assert 'event-old.wav' in message
        assert '10.0 days old' in message
        assert '7-day' in message


class TestResolveSource:
    def test_an_explicit_source_always_wins(self, tmp_path):
        """Matches --audio-rf-conversion-db and the rest of this program's rule that
        something named on the command line beats anything guessed from disk -
        the freshness check is not even consulted."""
        explicit = tmp_path / 'anything.wav'
        assert resolve_source(tmp_path, 7.0, explicit) == explicit

    def test_a_fresh_recording_is_used_without_asking(self, tmp_path):
        fresh = _write_wav(tmp_path / 'fresh.wav')
        with patch('builtins.input', side_effect=AssertionError('should not be asked')):
            assert resolve_source(tmp_path, 7.0, None) == fresh

    def test_operator_can_abort_a_stale_recording(self, tmp_path):
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        with patch('builtins.input', return_value='a'):
            assert resolve_source(tmp_path, 7.0, None) is None

    def test_operator_can_proceed_with_a_stale_recording_anyway(self, tmp_path):
        """A quiet band can mean there is genuinely nothing newer, and that is an
        acceptable reason to proceed - the check asks, it does not refuse."""
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        with patch('builtins.input', return_value='p'):
            assert resolve_source(tmp_path, 7.0, None) == stale

    def test_proceed_is_not_offered_when_there_is_nothing_to_proceed_with(self, tmp_path):
        """With no recording at all, 'p' has nothing behind it and must not be
        accepted as though it named the (nonexistent) newest file."""
        with patch('builtins.input', side_effect=['p', 'a']) as mocked:
            assert resolve_source(tmp_path, 7.0, None) is None
        assert mocked.call_count == 2, 'an invalid choice must re-prompt, not proceed'

    def test_operator_can_specify_a_different_file(self, tmp_path):
        """The replacement has to live outside the directory being scanned.  Written
        into it, a freshly created .wav is simply the newest recording, so
        resolve_source returns it without ever prompting and this test passes while
        exercising nothing - which is what it did before."""
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        elsewhere = tmp_path.parent / 'elsewhere'
        elsewhere.mkdir(exist_ok=True)
        replacement = _write_wav(elsewhere / 'from-elsewhere.wav')
        with patch('builtins.input', side_effect=['s', str(replacement)]) as mocked:
            assert resolve_source(tmp_path, 7.0, None) == replacement
        assert mocked.call_count == 2, 'the operator must actually have been prompted'

    def test_a_nonexistent_specified_path_re_prompts_rather_than_crashing(self, tmp_path):
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        with patch('builtins.input', side_effect=['s', str(tmp_path / 'ghost.wav'), 'a']):
            assert resolve_source(tmp_path, 7.0, None) is None

    def test_an_unrecognised_choice_re_prompts(self, tmp_path):
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        with patch('builtins.input', side_effect=['what?', 'a']):
            assert resolve_source(tmp_path, 7.0, None) is None


class TestNativeSampleRate:
    def test_reads_the_rate_from_the_header(self, tmp_path):
        f = _write_wav(tmp_path / 'event.wav', rate=44100)
        assert native_sample_rate(f) == 44100


class TestLoggedFrameCount:
    def test_parses_the_count_render_py_itself_logged(self, tmp_path):
        log = tmp_path / 'clip-16000hz.log'
        log.write_text(
            '2026-08-01 15:42:55  INFO  buzz.render: '
            'Rendered 497 frames to tmp\\clip-16000hz.mp4\n')
        assert logged_frame_count(log) == 497

    def test_a_missing_log_is_not_an_error(self, tmp_path):
        assert logged_frame_count(tmp_path / 'never-written.log') is None

    def test_a_log_with_no_matching_line_returns_none(self, tmp_path):
        log = tmp_path / 'clip.log'
        log.write_text('nothing relevant happened\n')
        assert logged_frame_count(log) is None


class TestFindFfprobe:
    def test_prefers_the_binary_beside_ffmpeg(self, tmp_path):
        ffmpeg = tmp_path / 'ffmpeg.exe'
        ffmpeg.touch()
        (tmp_path / 'ffprobe.exe').touch()
        assert find_ffprobe(str(ffmpeg)) == str(tmp_path / 'ffprobe.exe')

    def test_falls_back_to_path_when_no_sibling_exists(self, tmp_path):
        ffmpeg = tmp_path / 'ffmpeg.exe'
        ffmpeg.touch()
        with patch('shutil.which', return_value='/usr/bin/ffprobe'):
            assert find_ffprobe(str(ffmpeg)) == '/usr/bin/ffprobe'

    def test_raises_a_message_naming_what_was_searched(self, tmp_path):
        ffmpeg = tmp_path / 'ffmpeg.exe'
        ffmpeg.touch()
        with patch('shutil.which', return_value=None):
            with pytest.raises(Exception, match='ffprobe'):
                find_ffprobe(str(ffmpeg))


class TestFlagsFor:
    def test_a_clean_render_has_no_flags(self):
        assert flags_for(0, -29.0, 480, 480, 28.0, 105.0) == []

    def test_black_segments_are_flagged(self):
        flags = flags_for(2, -29.0, 480, 480, 28.0, 105.0)
        assert any('black segment' in f for f in flags)

    def test_near_silent_audio_is_flagged(self):
        flags = flags_for(0, -75.0, 480, 480, 28.0, 105.0)
        assert any('silent' in f for f in flags)

    def test_ordinary_quiet_audio_is_not_flagged(self):
        """-29 dB is what a real recording measures; the floor is for a dead track,
        not a quiet one."""
        assert flags_for(0, -45.0, 480, 480, 28.0, 105.0) == []

    def test_frame_count_mismatch_is_flagged(self):
        flags = flags_for(0, -29.0, 480, 475, 28.0, 105.0)
        assert any('frame count mismatch' in f for f in flags)

    def test_an_absent_logged_count_is_flagged_rather_than_skipped(self):
        """flags_for only runs for a render that finished, and buzz.render logs its
        frame count on every render that finishes.  So no count in the log means the
        plumbing broke - the log went astray, or the wording moved and the regex now
        matches nothing - and the check has stopped checking.  Reporting that as a
        pass would leave the tool green while one of its four probes was dead."""
        flags = flags_for(0, -29.0, None, 480, 28.0, 105.0)
        assert any('no frame count' in f for f in flags), (
            'A render log with no "Rendered N frames" line must be flagged. Left '
            'unflagged, a change to that log line silently disables the frame-count '
            'check and every release afterwards passes it without testing anything.')

    def test_the_absent_count_flag_names_what_was_decoded(self):
        """The reader still needs the one number that was measurable, or they cannot
        tell a plumbing fault from a render that produced no frames at all."""
        assert 'decoded 480' in flags_for(0, -29.0, None, 480, 28.0, 105.0)[0]

    def test_a_frozen_picture_is_flagged_as_low_motion(self):
        flags = flags_for(0, -29.0, 480, 480, 50.0, 50.1)
        assert any('low motion' in f for f in flags)

    def test_real_motion_is_not_flagged(self):
        assert flags_for(0, -29.0, 480, 480, 28.0, 105.0) == []


class TestRenderCheckOk:
    def test_ok_when_rendered_and_unflagged(self):
        assert RenderCheck(rate=16000, render_ok=True, flags=[]).ok is True

    def test_not_ok_when_render_failed(self):
        assert RenderCheck(rate=16000, render_ok=False, flags=['render failed: x']).ok is False

    def test_not_ok_when_rendered_but_flagged(self):
        assert RenderCheck(rate=16000, render_ok=True, flags=['low motion']).ok is False


def _clean_check(rate: int) -> RenderCheck:
    """A RenderCheck shaped the way check_render() actually populates one on success:
    every measured field filled in together, never left at the dataclass defaults."""
    return RenderCheck(rate=rate, render_ok=True, frames_logged=480, frames_decoded=480,
                       black_segments=0, mean_db=-29.0, peak_db=-11.0,
                       luma_lo=28.0, luma_hi=105.0, luma_stdev=20.0, flags=[])


class TestFormatReport:
    def test_names_the_source_recording(self, tmp_path):
        source = tmp_path / 'event.wav'
        report = format_report(source, [_clean_check(16000)])
        assert str(source) in report

    def test_a_failed_render_shows_its_error_not_a_blank_row(self):
        checks = [RenderCheck(rate=8000, render_ok=False, render_error='exit code 1',
                              flags=['render failed: exit code 1'])]
        report = format_report(Path('event.wav'), checks)
        assert 'FAILED' in report
        assert 'exit code 1' in report

    def test_summarises_how_many_were_flagged(self):
        flagged = _clean_check(16000)
        flagged.flags.append('low motion (luma range 0.1)')
        checks = [_clean_check(8000), flagged]
        report = format_report(Path('event.wav'), checks)
        assert '2 rate(s) checked, 1 flagged.' in report

    def test_every_column_starts_where_its_heading_does(self):
        """The table is read by eye down a column, so a heading that has drifted off
        its own numbers costs the reader the thing the table was for.  The widths are
        measured from the cells, which is what lets a wide value push its heading
        along instead of shoving every later column out of register: 384000 Hz here
        is nine characters against 'rate''s four.
        """
        checks = [_clean_check(8000), _clean_check(384000)]
        lines = format_report(Path('event.wav'), checks).splitlines()
        heading, rows = lines[2], lines[4:6]
        starts = [heading.index(column) for column in ('frames', 'audio', 'luma', 'flags')]
        for row in rows:
            for column, start in zip(('frames', 'audio', 'luma', 'flags'), starts):
                assert row[start] != ' ', (
                    f'The {column!r} column starts at position {start} in the heading, '
                    f'and this row has whitespace there:\n{heading}\n{row}\n'
                    'Header and cells are padded to different widths.')

    def test_an_unmeasurable_audio_track_prints_rather_than_raising(self):
        """audio_levels() returns None for a track it could not measure, and
        flags_for() deliberately does not call that silence.  So a render can reach
        the report with no dB figures, and formatting None as a float raises - which
        would lose the whole report, including the rates that were fine."""
        unmeasured = RenderCheck(rate=16000, render_ok=True, frames_logged=480,
                                 frames_decoded=480, mean_db=None, peak_db=None,
                                 luma_lo=28.0, luma_hi=105.0, luma_stdev=20.0, flags=[])
        assert '?' in format_report(Path('event.wav'), [unmeasured])


# --------------------------------------------------------------------------- clip prep

class TestMakeRateVariants:
    """The ffmpeg calls are mocked; what is under test is which rates get asked for
    and how each output is named, not that ffmpeg resamples correctly."""

    def test_renders_the_native_rate_alongside_the_foreign_ones(self, tmp_path):
        source = _write_wav(tmp_path / 'event.wav', rate=16000)
        with patch('tools.release_render_check.run') as mocked:
            variants = make_rate_variants(source, 16.0, tmp_path, 'ffmpeg')
        assert sorted(variants) == [8000, 11025, 16000, 22050, 44100, 48000]
        assert mocked.call_count == 6

    def test_both_ends_of_the_admitted_band_are_rendered(self, tmp_path):
        """The reason this tool renders at more than one rate at all.

        MIN_SAMPLE_RATE and MAX_SAMPLE_RATE are the boundaries buzz.config's
        validate_sample_rate admits, and a rate-dependent bug shows up at a boundary
        long before it shows up in the middle - MAX_SAMPLE_RATE especially, where the
        fixed-size ring buffer holds the least history.  Both numbers come from
        buzz.config rather than from this file, so widening the admitted band and
        leaving the check rendering at the old edges fails here instead of shipping.
        """
        source = _write_wav(tmp_path / 'event.wav', rate=16000)
        with patch('tools.release_render_check.run'):
            variants = make_rate_variants(source, 16.0, tmp_path, 'ffmpeg')
        for boundary in (MIN_SAMPLE_RATE, MAX_SAMPLE_RATE):
            assert boundary in variants, (
                f'{boundary} Hz is an end of the band validate_sample_rate admits, and '
                f'the release check renders at {sorted(variants)}. Add it to '
                '_RATES_UNDER_TEST in tools/release_render_check.py, or the release '
                'goes out with that boundary never rendered.')

    def test_a_native_rate_already_in_the_set_is_not_duplicated(self, tmp_path):
        """44100 is both this recording's own rate and one of the foreign rates; it
        must be rendered once, not twice."""
        source = _write_wav(tmp_path / 'event.wav', rate=44100)
        with patch('tools.release_render_check.run') as mocked:
            variants = make_rate_variants(source, 16.0, tmp_path, 'ffmpeg')
        assert sorted(variants) == [8000, 11025, 22050, 44100, 48000]
        assert mocked.call_count == 5

    def test_each_variant_asks_ffmpeg_for_its_own_rate_and_pcm_wav(self, tmp_path):
        """--playback only reads .wav, so every variant has to come back as PCM WAVE
        rather than whatever ffmpeg would infer from the extension."""
        source = _write_wav(tmp_path / 'event.wav', rate=16000)
        with patch('tools.release_render_check.run') as mocked:
            variants = make_rate_variants(source, 12.5, tmp_path, 'ffmpeg')
        for rate, path in variants.items():
            command = next(c.args[0] for c in mocked.call_args_list if str(path) in c.args[0])
            assert command[command.index('-ar') + 1] == str(rate)
            assert command[command.index('-t') + 1] == '12.5'
            assert command[command.index('-c:a') + 1] == 'pcm_s16le'
            assert path.name == 'clip-{}hz.wav'.format(rate)


# --------------------------------------------------------------------------- rendering

def _completed(returncode: int = 0, stdout: str = '', stderr: str = ''):
    from subprocess import CompletedProcess
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestRenderVariant:
    def test_a_successful_render_reports_the_mp4_it_made(self, tmp_path):
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        (tmp_path / 'clip-16000hz.mp4').touch()     # stands in for what the render wrote
        with patch('tools.release_render_check.subprocess.run', return_value=_completed()):
            attempt = render_variant(wav, tmp_path)
        assert attempt.ok is True and attempt.error == ''
        assert attempt.mp4 == tmp_path / 'clip-16000hz.mp4'

    def test_the_attempt_carries_the_log_it_wrote(self, tmp_path):
        """check_render reads the frame count back out of this file.  Returning the
        path rather than letting the caller rebuild it from the same naming rule is
        what stops the two drifting apart - and a frame-count check pointed at a file
        that is not there reports nothing wrong, so the drift would be silent."""
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        (tmp_path / 'clip-16000hz.mp4').touch()
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed(stderr='Rendered 497 frames\n')):
            attempt = render_variant(wav, tmp_path)
        assert attempt.log.exists(), 'the returned log path must be the file just written'
        assert 'Rendered 497 frames' in attempt.log.read_text()

    def test_it_renders_headless_and_offscreen(self, tmp_path):
        """A release check that throws a window in front of whoever is running it
        would not get run.  Same rule as the integration tier."""
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        (tmp_path / 'clip-16000hz.mp4').touch()
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed()) as mocked:
            render_variant(wav, tmp_path)
        command = mocked.call_args.args[0]
        assert '--headless' in command
        assert mocked.call_args.kwargs['env']['QT_QPA_PLATFORM'] == 'offscreen'

    def test_it_exercises_the_numba_path_production_actually_runs(self, tmp_path):
        """NUMBA_DISABLE_JIT=0: the interpreted and JIT'd paths are not automatically
        equivalent, so checking the interpreted one before a release proves little."""
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        (tmp_path / 'clip-16000hz.mp4').touch()
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed()) as mocked:
            render_variant(wav, tmp_path)
        assert mocked.call_args.kwargs['env']['NUMBA_DISABLE_JIT'] == '0'

    def test_the_render_log_is_kept_for_the_frame_count_check(self, tmp_path):
        """logged_frame_count() reads this file back, so both streams have to reach it."""
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        (tmp_path / 'clip-16000hz.mp4').touch()
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed(stdout='on stdout\n',
                                           stderr='Rendered 497 frames\n')):
            render_variant(wav, tmp_path)
        written = (tmp_path / 'clip-16000hz.log').read_text()
        assert 'on stdout' in written and 'Rendered 497 frames' in written

    def test_a_timeout_is_reported_rather_than_raised(self, tmp_path):
        import subprocess as sp
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        with patch('tools.release_render_check.subprocess.run',
                   side_effect=sp.TimeoutExpired(cmd='buzz.main', timeout=120.0)):
            attempt = render_variant(wav, tmp_path, timeout=120.0)
        assert attempt.ok is False
        assert 'did not finish within 120s' in attempt.error

    def test_a_nonzero_exit_reports_the_code_and_the_tail_of_the_output(self, tmp_path):
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed(returncode=2, stderr='ffmpeg exploded')):
            attempt = render_variant(wav, tmp_path)
        assert attempt.ok is False
        assert 'exit code 2' in attempt.error and 'ffmpeg exploded' in attempt.error

    def test_a_clean_exit_that_produced_no_file_is_still_a_failure(self, tmp_path):
        """buzz.main can exit 0 having written nothing; the file is the real evidence."""
        wav = tmp_path / 'clip-16000hz.wav'
        wav.touch()
        with patch('tools.release_render_check.subprocess.run', return_value=_completed()):
            attempt = render_variant(wav, tmp_path)
        assert attempt.ok is False


# --------------------------------------------------------------------------- probes

class TestContentProbes:
    """Each probe parses one ffmpeg filter's output.  The text fed in is the shape
    ffmpeg really emits; what is under test is the parsing, not the filter."""

    def test_black_segments_are_counted(self, tmp_path):
        output = ('[blackdetect @ 0x1] black_start:0 black_end:0.4 black_duration:0.4\n'
                  '[blackdetect @ 0x1] black_start:9.2 black_end:9.7 black_duration:0.5\n')
        with patch('tools.release_render_check.run', return_value=output):
            assert count_black_segments(tmp_path / 'x.mp4', 'ffmpeg') == 2

    def test_a_clean_render_reports_no_black_segments(self, tmp_path):
        with patch('tools.release_render_check.run', return_value='frame= 480 fps=30'):
            assert count_black_segments(tmp_path / 'x.mp4', 'ffmpeg') == 0

    def test_audio_levels_are_parsed(self, tmp_path):
        output = ('[Parsed_volumedetect_0 @ 0x1] mean_volume: -29.4 dB\n'
                  '[Parsed_volumedetect_0 @ 0x1] max_volume: -11.2 dB\n')
        with patch('tools.release_render_check.run', return_value=output):
            assert audio_levels(tmp_path / 'x.mp4', 'ffmpeg') == (-29.4, -11.2)

    def test_missing_audio_levels_read_as_unknown_not_as_silence(self, tmp_path):
        """None and 'very quiet' are different claims: flags_for() must not treat a
        track it could not measure as a near-silent one."""
        with patch('tools.release_render_check.run', return_value='no audio stream'):
            assert audio_levels(tmp_path / 'x.mp4', 'ffmpeg') == (None, None)

    def test_decoded_frames_are_read_from_ffprobe(self, tmp_path):
        with patch('tools.release_render_check.subprocess.run',
                   return_value=_completed(stdout='480\n')):
            assert decoded_frame_count(tmp_path / 'x.mp4', 'ffprobe') == 480

    def test_luma_series_collects_every_frames_yavg(self, tmp_path):
        output = ('lavfi.signalstats.YAVG=28.500\n'
                  'lavfi.signalstats.YAVG=71.250\n'
                  'lavfi.signalstats.YAVG=105.000\n')
        with patch('tools.release_render_check.run', return_value=output):
            assert luma_series(tmp_path / 'x.mp4', 'ffmpeg') == [28.5, 71.25, 105.0]

    def test_no_yavg_lines_give_an_empty_series(self, tmp_path):
        with patch('tools.release_render_check.run', return_value='nothing here'):
            assert luma_series(tmp_path / 'x.mp4', 'ffmpeg') == []


# --------------------------------------------------------------------------- check_render

class TestCheckRender:
    """The probes are mocked individually here: what is under test is how their
    answers are assembled into one RenderCheck, not what any of them measures."""

    def _probes(self, black=0, levels=(-29.0, -11.0), decoded=480, luma=None):
        luma = [28.0, 60.0, 105.0] if luma is None else luma
        return {
            'tools.release_render_check.count_black_segments': black,
            'tools.release_render_check.audio_levels': levels,
            'tools.release_render_check.decoded_frame_count': decoded,
            'tools.release_render_check.luma_series': luma,
        }

    def test_a_failed_render_short_circuits_the_content_checks(self, tmp_path):
        """Nothing was produced, so probing it would only add noise; the flag says
        what went wrong instead."""
        with patch('tools.release_render_check.render_variant',
                   return_value=RenderAttempt(tmp_path / 'x.mp4', tmp_path / 'x.log',
                                              False, 'exit code 1')):
            with patch('tools.release_render_check.count_black_segments',
                       side_effect=AssertionError('must not probe a file that was never made')):
                check = check_render(16000, tmp_path / 'clip.wav', tmp_path, 'ffmpeg', 'ffprobe')
        assert check.render_ok is False
        assert check.ok is False
        assert check.flags == ['render failed: exit code 1']

    def test_a_clean_render_fills_in_every_measured_field(self, tmp_path):
        (tmp_path / 'clip.log').write_text('Rendered 480 frames to clip.mp4\n')
        with patch('tools.release_render_check.render_variant',
                   return_value=RenderAttempt(tmp_path / 'clip.mp4', tmp_path / 'clip.log',
                                              True)):
            with patch.multiple('tools.release_render_check',
                                count_black_segments=lambda *a: 0,
                                audio_levels=lambda *a: (-29.0, -11.0),
                                decoded_frame_count=lambda *a: 480,
                                luma_series=lambda *a: [28.0, 60.0, 105.0]):
                check = check_render(16000, tmp_path / 'clip.wav', tmp_path, 'ffmpeg', 'ffprobe')
        assert check.ok is True
        assert (check.frames_decoded, check.frames_logged) == (480, 480)
        assert (check.luma_lo, check.luma_hi) == (28.0, 105.0)
        assert check.mean_db == -29.0
        assert check.luma_stdev > 0

    def test_a_frozen_render_is_flagged_from_its_luma_range(self, tmp_path):
        """The frozen-opening-frame bug class: a well-formed file whose picture
        never changes."""
        (tmp_path / 'clip.log').write_text('Rendered 480 frames\n')
        with patch('tools.release_render_check.render_variant',
                   return_value=RenderAttempt(tmp_path / 'clip.mp4', tmp_path / 'clip.log',
                                              True)):
            with patch.multiple('tools.release_render_check',
                                count_black_segments=lambda *a: 0,
                                audio_levels=lambda *a: (-29.0, -11.0),
                                decoded_frame_count=lambda *a: 480,
                                luma_series=lambda *a: [50.0, 50.0, 50.05]):
                check = check_render(16000, tmp_path / 'clip.wav', tmp_path, 'ffmpeg', 'ffprobe')
        assert check.ok is False
        assert any('low motion' in f for f in check.flags)

    def test_an_unmeasurable_luma_series_does_not_crash_the_statistics(self, tmp_path):
        """min()/max()/pstdev() of an empty list all raise; the zeros are what let
        the low-motion flag speak instead."""
        (tmp_path / 'clip.log').write_text('Rendered 480 frames\n')
        with patch('tools.release_render_check.render_variant',
                   return_value=RenderAttempt(tmp_path / 'clip.mp4', tmp_path / 'clip.log',
                                              True)):
            with patch.multiple('tools.release_render_check',
                                count_black_segments=lambda *a: 0,
                                audio_levels=lambda *a: (-29.0, -11.0),
                                decoded_frame_count=lambda *a: 480,
                                luma_series=lambda *a: []):
                check = check_render(16000, tmp_path / 'clip.wav', tmp_path, 'ffmpeg', 'ffprobe')
        assert (check.luma_lo, check.luma_hi, check.luma_stdev) == (0.0, 0.0, 0.0)
        assert any('low motion' in f for f in check.flags)


# --------------------------------------------------------------------------- entry point

class _MainHarness:
    """Everything main() reaches outside itself, stubbed in one place.

    main() is orchestration: find the tools, choose a source, build the variants,
    check each one, report.  These tests are about that sequence and the exit code,
    which is what the release procedure actually keys on.
    """

    def __init__(self, tmp_path, checks=None, source=None, variants=None):
        self.tmp_path = tmp_path
        self.checks = checks if checks is not None else [_clean_check(16000)]
        self.source = source if source is not None else tmp_path / 'event.wav'
        self.variants = variants if variants is not None else {16000: tmp_path / 'clip.wav'}

    def run(self, argv_extra=(), **overrides):
        from unittest.mock import MagicMock
        config_path = MagicMock()
        config_path.exists.return_value = False
        patches = {
            'CONFIG_PATH': config_path,
            'find_ffmpeg': lambda *a: 'ffmpeg',
            'find_ffprobe': lambda *a: 'ffprobe',
            'resolve_source': lambda *a: self.source,
            'make_rate_variants': lambda *a: self.variants,
            'check_render': lambda rate, *a: next(
                (c for c in self.checks if c.rate == rate), self.checks[0]),
        }
        patches.update(overrides)
        argv = ['release_render_check.py', '--output-dir', str(self.tmp_path / 'out')]
        argv.extend(argv_extra)
        with patch('sys.argv', argv):
            with patch.multiple('tools.release_render_check', **patches):
                return main()


class TestMain:
    def test_a_clean_run_exits_zero(self, tmp_path):
        assert _MainHarness(tmp_path).run() == 0

    def test_a_flagged_render_exits_nonzero(self, tmp_path):
        """The exit code is what makes this usable as a gate rather than a report
        somebody has to read carefully."""
        flagged = RenderCheck(rate=16000, render_ok=True, frames_logged=480,
                              frames_decoded=480, mean_db=-29.0, peak_db=-11.0,
                              luma_lo=50.0, luma_hi=50.1, luma_stdev=0.01,
                              flags=['low motion (luma range 0.100)'])
        assert _MainHarness(tmp_path, checks=[flagged]).run() == 1

    def test_the_report_is_written_beside_the_renders(self, tmp_path):
        _MainHarness(tmp_path).run()
        report = tmp_path / 'out' / 'report.txt'
        assert report.exists()
        assert 'rate(s) checked' in report.read_text()

    def test_the_report_is_also_printed(self, tmp_path, capsys):
        _MainHarness(tmp_path).run()
        assert 'rate(s) checked' in capsys.readouterr().out

    def test_a_missing_ffmpeg_explains_itself_and_stops(self, tmp_path, capsys):
        def explode(*args):
            raise FfmpegError('ffmpeg was not found on PATH, and --render needs it.')
        exit_code = _MainHarness(tmp_path).run(find_ffmpeg=explode)
        assert exit_code == 1
        assert 'not found on PATH' in capsys.readouterr().out

    def test_a_missing_ffprobe_also_stops_before_rendering_anything(self, tmp_path):
        def explode(*args):
            raise FfmpegError('ffprobe was not found beside ffmpeg or on PATH.')
        assert _MainHarness(tmp_path).run(find_ffprobe=explode) == 1

    def test_aborting_at_the_freshness_prompt_renders_nothing(self, tmp_path, capsys):
        exit_code = _MainHarness(tmp_path).run(resolve_source=lambda *a: None)
        assert exit_code == 1
        assert 'Aborted' in capsys.readouterr().out
        assert not (tmp_path / 'out').exists(), 'nothing should be created after an abort'

    def test_a_failure_building_the_clips_is_reported_not_raised(self, tmp_path, capsys):
        def explode(*args):
            raise FfmpegError('ffmpeg could not resample the recording.')
        exit_code = _MainHarness(tmp_path).run(make_rate_variants=explode)
        assert exit_code == 1
        assert 'could not resample' in capsys.readouterr().out

    def test_every_variant_gets_checked(self, tmp_path):
        seen = []

        def record(rate, *args):
            seen.append(rate)
            return _clean_check(rate)

        harness = _MainHarness(tmp_path, variants={8000: tmp_path / 'a.wav',
                                                   16000: tmp_path / 'b.wav',
                                                   44100: tmp_path / 'c.wav'})
        assert harness.run(check_render=record) == 0
        assert seen == [8000, 16000, 44100], 'checked in rate order, one per variant'

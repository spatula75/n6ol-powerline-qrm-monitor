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

from tools.release_render_check import (
    RenderCheck,
    age_days,
    describe_staleness,
    find_ffprobe,
    flags_for,
    format_report,
    logged_frame_count,
    native_sample_rate,
    newest_recording,
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

    def test_no_prompt_mode_treats_staleness_as_abort(self, tmp_path):
        """prompt=False is what a non-interactive caller (this test file, or a future
        scripted use) gets: no input() call, and staleness reads the same as the
        operator choosing to abort."""
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        assert resolve_source(tmp_path, 7.0, None, prompt=False) is None

    def test_no_recordings_and_no_prompt_is_also_abort(self, tmp_path):
        assert resolve_source(tmp_path, 7.0, None, prompt=False) is None

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
        import os
        stale = _write_wav(tmp_path / 'stale.wav')
        os.utime(stale, (1_000_000, 1_000_000))
        replacement = _write_wav(tmp_path / 'from-elsewhere.wav')
        with patch('builtins.input', side_effect=['s', str(replacement)]):
            assert resolve_source(tmp_path, 7.0, None) == replacement

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

    def test_no_logged_count_is_not_a_mismatch(self):
        """A missing log (render_variant timed out, say) has nothing to compare
        against, which is not the same claim as the counts disagreeing."""
        assert flags_for(0, -29.0, None, 480, 28.0, 105.0) == []

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

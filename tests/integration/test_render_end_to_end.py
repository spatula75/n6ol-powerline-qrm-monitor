"""Rendering a replay to .mp4, driven the way an operator drives it.

    pytest -m integration --no-cov

Tier 2 — a real ffmpeg, a real Qt paint pass, and a real replay at real speed.  The
monitor is launched as a subprocess rather than assembled in-process, so what is under
test includes the parts nothing else covers: argument parsing, the lazy import that
keeps ffmpeg optional, the window built without its control strip, and the ordering
that decides whether the opening of the file exists at all.

Skipped, not failed, without ffmpeg on PATH.  It is needed for --render and nothing
else, and a contributor who never renders should still get a green suite.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from harness import RATE

FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')

# Skipping is right on a contributor's machine and wrong in CI.  A skip is invisible
# in a green run, so an environment that was meant to have ffmpeg and lost it would
# quietly stop testing the render path and nobody would find out.  CI sets this, and
# the missing toolchain becomes a failure with an explanation instead.
_REQUIRED = os.environ.get('BUZZ_REQUIRE_FFMPEG') == '1'
_MISSING = not FFMPEG or not FFPROBE

if _MISSING and _REQUIRED:  # pragma: no cover -- only reached in a broken CI image
    raise RuntimeError(
        'BUZZ_REQUIRE_FFMPEG=1 is set, so the render integration tests must run, but '
        f'ffmpeg={FFMPEG!r} and ffprobe={FFPROBE!r} on PATH.\n\n'
        'This environment was configured expecting both — see the ffmpeg step in '
        '.github/actions/setup. The likely cause is a runner image that no longer '
        'ships them, or an install step that succeeded without putting them on PATH '
        'for this step.\n\n'
        'To fix: install ffmpeg (it provides ffprobe too) before running the suite, '
        'or unset BUZZ_REQUIRE_FFMPEG to let these tests skip as they do locally.')

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _MISSING,
        reason='needs ffmpeg and ffprobe on PATH; they are used by --render and by '
               'nothing else, so the rest of the suite runs without them'),
]

# The window with its control strip removed: 640 waterfall + 8 gap + 86 meters wide,
# 120 scope + 8 gap + 120 waterfall tall.  Both even, which yuv420p requires.
EXPECTED_SIZE = (734, 248)
FRAME_RATE = 30
ROOT = Path(__file__).resolve().parent.parent.parent


def probe(path: Path) -> dict[str, dict]:
    """Stream properties from ffprobe, keyed by codec_type."""
    result = subprocess.run(
        [FFPROBE, '-v', 'error', '-show_streams', '-show_format',
         '-of', 'json', str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    streams = {stream['codec_type']: stream for stream in data['streams']}
    streams['format'] = data['format']
    return streams


@pytest.fixture(scope='module')
def rendered(recorded_event, tmp_path_factory):
    """One render of a real recording, produced by running the program.

    Module-scoped because it costs the length of the recording in wall clock -- the
    render is real-time by construction -- and every assertion below reads the same
    file.
    """
    output = tmp_path_factory.mktemp('render') / 'event.mp4'
    environment = {
        **os.environ,
        # No window server: this is the platform a headless render would use, and the
        # one that reports no fonts at all on Windows.  Running here rather than on
        # the desktop keeps the suite from throwing a window at whoever runs it, and
        # tests the case that actually broke.
        'QT_QPA_PLATFORM': 'offscreen',
        'PYTHONPATH': str(ROOT / 'lib'),
        # The compiled path, as production runs it -- and not the interpreted one
        # conftest.py selects for coverage, which this subprocess would otherwise
        # inherit.  It is faithfulness that matters here rather than speed: compiling
        # the FFT kernels costs the first seconds of the process, the event loop is
        # blocked while it happens, and playback is already running on its own thread.
        # That is exactly the window in which the opening of a render was once lost,
        # so a test that skips the compile cannot see the bug it exists to catch.
        'NUMBA_DISABLE_JIT': '0',
    }
    result = subprocess.run(
        [sys.executable, '-m', 'buzz.main',
         '--playback', str(recorded_event), '--mute', '--render', str(output)],
        capture_output=True, text=True, env=environment, cwd=str(ROOT), timeout=300)
    assert result.returncode == 0, (
        f'Rendering exited {result.returncode} instead of 0, so no usable .mp4 was '
        f'produced.\n\nstderr:\n{result.stderr[-3000:]}\n\n'
        'The monitor logs the full ffmpeg command at INFO before it starts; that line '
        'can be pasted into a terminal to reproduce the failure by hand.')
    assert output.exists(), (
        f'Rendering reported success but {output} does not exist. Either the output '
        'path was rewritten somewhere between the flag and the encoder, or ffmpeg '
        'exited zero without writing -- check the logged command.')
    return output


class TestTheFileIsPlayable:
    """What VLC and Firefox need, asserted on the artefact rather than the arguments.

    The unit tests already check the command that gets built.  These check what came
    out of it, which is a different question: an argument list can be perfectly correct
    and still produce a file no browser will open.
    """

    def test_it_is_h264_in_the_profile_firefox_decodes(self, rendered):
        video = probe(rendered)['video']
        assert video['codec_name'] == 'h264'
        assert video['profile'] == 'Main'

    def test_the_pixel_format_is_the_widely_decodable_one(self, rendered):
        """4:4:4 would suit thin scope traces and axis text better, and Firefox will
        not decode it."""
        assert probe(rendered)['video']['pix_fmt'] == 'yuv420p'

    def test_the_audio_is_aac_at_a_rate_players_like(self, rendered):
        audio = probe(rendered)['audio']
        assert audio['codec_name'] == 'aac'
        assert int(audio['sample_rate']) == 48000

    def test_the_audio_was_resampled_up_from_the_recording(self, rendered):
        """The recording is 16 kHz, which some browsers handle poorly."""
        assert int(probe(rendered)['audio']['sample_rate']) != RATE

    def test_the_frame_rate_is_exactly_thirty(self, rendered):
        """Not 29.97.  A 1001/1000 timebase would put every frame time off a round
        number against an audio timeline measured in exact 48000ths, for the sake of
        a 1953 broadcast compromise this file has no part in."""
        assert probe(rendered)['video']['r_frame_rate'] == f'{FRAME_RATE}/1'

    def test_the_moov_atom_is_at_the_front(self, rendered):
        """+faststart.  Without it a browser must fetch the whole file before showing
        a frame, which matters as soon as one of these is served over HTTP."""
        head = rendered.read_bytes()
        moov, mdat = head.find(b'moov'), head.find(b'mdat')
        assert 0 <= moov < mdat, (
            f'moov is at {moov} and mdat at {mdat}, so the index is behind the media '
            'and progressive playback will stall until the whole file arrives. '
            'Check that -movflags +faststart survived in ffmpeg_command().')


class TestTheRecordingsIdentityTravelsWithIt:
    """A rendered event should still say what it is.

    The .wav carries LIST/INFO tags naming the event, the station, the moment, and the
    calibration behind the numbers. They are the difference between a video somebody
    can act on and one that is just a picture of some noise.
    """

    def test_the_tags_are_in_the_container(self, rendered, recorded_event):
        tags = probe(rendered)['format'].get('tags', {})
        source_tags = probe(recorded_event)['format'].get('tags', {})
        for key in ('title', 'artist', 'comment'):
            assert tags.get(key) == source_tags.get(key), (
                f'The {key!r} tag did not survive into the .mp4: the recording has '
                f'{source_tags.get(key)!r} and the render has {tags.get(key)!r}. '
                'Check that -map_metadata names the .wav\'s input index; input 0 is '
                'the frame pipe, which has no metadata of its own to copy.')

    def test_there_is_no_third_stream(self, rendered):
        """The lock cue used to arrive as a chapter, and mp4 chapters are a *track* --
        so a single marker became a bin_data stream sitting beside the video and
        audio. Dropping the chapters removes the track; lead_in_seconds in the comment
        tag still says where the lock was."""
        kinds = sorted(stream['codec_type'] for stream in probe(rendered).values()
                       if isinstance(stream, dict) and 'codec_type' in stream)
        assert kinds == ['audio', 'video'], (
            f'Expected only a video and an audio stream, found {kinds}. A "data" '
            'stream means chapters came through from the recording again — see '
            '-map_chapters in ffmpeg_command().')


class TestTheControlsAreGone:
    """A render is of the display, not of the program's chrome."""

    def test_the_frame_is_the_window_without_its_toolbar(self, rendered):
        video = probe(rendered)['video']
        assert (video['width'], video['height']) == EXPECTED_SIZE, (
            f"Rendered {video['width']}x{video['height']}, expected {EXPECTED_SIZE}. "
            'The taller size (734x284) means the control strip was drawn into the '
            'video; a different width means the panel geometry moved. MainWindow '
            'takes show_controls=False for renders and drops the bar and its layout '
            'spacing together.')

    def test_both_dimensions_stay_even(self, rendered):
        """yuv420p subsamples chroma by two in each direction, so an odd dimension is
        not encodable — a constraint that only bites when the layout changes."""
        video = probe(rendered)['video']
        assert video['width'] % 2 == 0 and video['height'] % 2 == 0


class TestAudioAndVideoLineUp:
    """The sync property the whole design exists to protect."""

    def test_the_streams_are_the_same_length(self, rendered):
        streams = probe(rendered)
        video = float(streams['video']['duration'])
        audio = float(streams['audio']['duration'])
        assert abs(video - audio) < 0.05, (
            f'Video runs {video:.3f}s and audio {audio:.3f}s, a gap of '
            f'{abs(video - audio):.3f}s. They are laid on one timeline from the same '
            'source, so a gap means frames were lost or duplicated rather than that '
            'the clocks drifted.')

    def test_the_video_lasts_as_long_as_the_replay_did(self, rendered):
        """The bug this catches shipped a 1.5 s video from a 3 s recording while
        reporting the right number of frames: the first capture arrived after startup
        had already burned a second, and the frames before it were counted but never
        written."""
        streams = probe(rendered)
        source = float(streams['format']['duration'])
        video = float(streams['video']['duration'])
        # Within a couple of frames, not within ten percent.  Whatever is lost at the
        # start is lost silently and the file still looks plausible, so the tolerance
        # has to be tight enough to notice a fraction of a second.  The legitimate
        # shortfall is one dropped partial chunk, 32 ms, plus grid rounding.
        assert source - video < 0.2, (
            f'Video is {video:.3f}s against a {source:.3f}s recording, {source - video:.3f}s '
            'short. Time missing from the front means the first frame was captured '
            'after playback had already moved: DisplayRecorder.start() captures one '
            'frame before starting the transport, and RenderSession.submit() backfills '
            'if that first frame still arrives late. Check both are intact.')

    def test_the_frame_count_matches_the_duration(self, rendered):
        """Every slot on the grid holds a frame: no gaps, no padding at the end."""
        video = probe(rendered)['video']
        expected = round(float(video['duration']) * FRAME_RATE)
        assert abs(int(video['nb_frames']) - expected) <= 1

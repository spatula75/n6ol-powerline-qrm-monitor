"""A .wav from somebody else, played straight into the analyzer.

    pytest -m integration --no-cov

The case this is written for: an operator is sent a capture by another ham and wants to
know whether it locks at 120 pps.  It will not be 16 kHz mono with this program's
metadata — it will be whatever their sound card produced, most likely 44.1 kHz stereo
with no tags at all.

Tier 1 apart from the render, which needs ffmpeg and skips without it.  The monitor is
launched as a subprocess so the warnings under test are the ones an operator actually
sees, in the order they see them.
"""

import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = shutil.which('ffmpeg')

# What a sound card in a shack somewhere is likely to hand over.
FOREIGN_RATE = 44100
FOREIGN_CHANNELS = 2


def write_foreign_wav(path: Path, seconds: float = 3.0) -> Path:
    """A stereo 44.1 kHz recording of a 120 pps arc, carrying no metadata at all.

    The interference sits on channel 0 and the other channel holds unrelated noise, so
    a downmix would measurably dilute it -- which is why the program takes one channel
    rather than averaging, and why it says so.
    """
    count = int(seconds * FOREIGN_RATE)
    rng = np.random.default_rng(5)
    left = rng.integers(-300, 301, size=count).astype(np.int16)
    right = rng.integers(-3000, 3001, size=count).astype(np.int16)
    spacing = FOREIGN_RATE / 120
    width = int(0.003 * FOREIGN_RATE)          # a 3 ms burst, as the real thing is
    for index in range(int(count / spacing)):
        at = round(index * spacing)
        if at + width < count:
            left[at:at + width] = 12000
    interleaved = np.empty(count * 2, dtype=np.int16)
    interleaved[0::2], interleaved[1::2] = left, right
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(FOREIGN_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(FOREIGN_RATE)
        handle.writeframes(interleaved.tobytes())
    return path


def run_monitor(source: Path, *extra: str, seconds: float = 8) -> tuple[str, int]:
    """Play `source` and return (everything it said, exit status).

    Always `--mute` and always `--headless`: a test suite must not take over the
    speakers or throw a window at whoever is running it, and neither adds anything to
    what is under test here -- the audio path to the sound card is not involved in any
    of it.

    Headless playback does not end by itself; it waits for an interrupt, exactly as a
    monitor left running does.  So it is stopped after `seconds` and whatever it said
    by then is the result.  Everything asserted below is logged in the first moments,
    while the file is being opened and its settings worked out.  A run that exits on
    its own -- a refusal, or a render -- reports its real status instead.
    """
    environment = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
                   'PYTHONPATH': str(ROOT / 'lib'), 'NUMBA_DISABLE_JIT': '0'}
    process = subprocess.Popen(
        [sys.executable, '-m', 'buzz.main', '--playback', str(source),
         '--headless', '--mute', *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=environment, cwd=str(ROOT))
    try:
        output = process.communicate(timeout=seconds)[0]
    except subprocess.TimeoutExpired:
        process.terminate()
        output = process.communicate(timeout=30)[0]
    return output, process.returncode


@pytest.fixture(scope='module')
def foreign_wav(tmp_path_factory):
    return write_foreign_wav(tmp_path_factory.mktemp('foreign') / 'from-a-ham.wav')


@pytest.fixture(scope='module')
def played(foreign_wav):
    """One short run, read by everything below.  Stopped rather than waited out: a
    headless playback runs until interrupted, and every line asserted here is logged
    while the file is being opened."""
    return run_monitor(foreign_wav)[0]


class TestItPlaysAtAll:
    """The point of the exercise: an unfamiliar file should just work."""

    def test_it_gets_as_far_as_playing(self, played):
        """Not "exits cleanly": a headless playback runs until interrupted, exactly as
        a monitor left running does, so reaching playback at all is the thing to check
        here. The render test below covers a run that ends by itself."""
        assert 'Playing back' in played, (
            f'A {FOREIGN_RATE} Hz stereo file never reached playback.\n\n{played[-2000:]}')

    def test_it_adopts_the_files_sample_rate(self, played):
        """44.1 kHz is not this program's rate, and the file's own is authoritative --
        analysing it as 16 kHz would put the pulse grid at the wrong spacing."""
        assert 'sample rate 44100' in played


class TestItSaysWhatItIsAssuming:
    """Everything a foreign file cannot tell us has to be stated, or a plausible number
    gets read off the display as a real one."""

    def test_it_says_only_one_channel_is_analysed(self, played):
        assert 'analysing channel 0 only' in played, (
            'A stereo file was analysed without saying half of it was ignored.')

    def test_it_says_which_pulse_rate_it_assumed(self, played):
        assert 'does not record its pulse rate' in played

    def test_it_says_which_calibration_it_assumed(self, played):
        assert 'does not record its level calibration' in played

    def test_it_says_which_readings_are_untrustworthy(self, played):
        """The distinction that makes the file useful anyway: levels are suspect,
        lock and burst shape are not."""
        assert 'may be wrong by any amount' in played

    def test_it_names_the_remedy(self, played):
        assert '--audio-rf-conversion-db' in played


class TestASuppliedCalibration:

    @pytest.fixture(scope='class')
    @staticmethod
    def supplied(foreign_wav):
        return run_monitor(foreign_wav, '--audio-rf-conversion-db', '-28.5')[0]

    def test_it_is_reported_as_used(self, supplied):
        assert '-28.5 dB from --audio-rf-conversion-db' in supplied

    def test_it_stops_the_advice_being_repeated(self, supplied):
        """Telling somebody to pass a flag they have just passed is noise."""
        assert 'does not record its level calibration' not in supplied


class TestRejections:
    """Refused rather than analysed into plausible nonsense."""

    def test_a_rate_below_the_floor_is_refused(self, tmp_path):
        source = tmp_path / 'slow.wav'
        with wave.open(str(source), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(4000)
            handle.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
        output, status = run_monitor(source)
        assert status != 0
        assert '4000 Hz' in output

    def test_a_wrong_sample_width_is_refused(self, tmp_path):
        """int16 end to end; converting silently would put this replay at odds with
        what the same audio measured live."""
        source = tmp_path / 'wide.wav'
        with wave.open(str(source), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(4)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(16000 * 4, dtype=np.int8).tobytes())
        output, status = run_monitor(source)
        assert status != 0
        assert '16-bit' in output


@pytest.mark.skipif(not FFMPEG, reason='rendering needs ffmpeg, which is optional')
class TestItRendersToo:
    """The reply: a video that can be sent back to whoever supplied the recording."""

    def test_a_foreign_file_renders(self, foreign_wav, tmp_path):
        video = tmp_path / 'reply.mp4'
        # A render ends by itself when the file does, so this one is waited out.
        output, status = run_monitor(foreign_wav, '--render', str(video), seconds=180)
        assert status == 0, output[-2000:]
        assert video.exists() and video.stat().st_size > 0

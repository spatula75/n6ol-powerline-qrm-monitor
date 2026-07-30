"""Tests for .wav loading, path resolution, and the file-backed playback pipeline."""

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from buzz.playback import FilePlaybackPipeline, load_wav, resolve_playback_path
from buzz.sampler import RingBufferPipeline

CHUNK = RingBufferPipeline.CHUNK_SIZE

# Playback is paced from the file's own sample rate, so an absurdly high rate makes
# the feeder thread run through a test file in milliseconds while still exercising
# the real deadline schedule.  One chunk is 1 ms of "audio" at this rate.
FAST_RATE = CHUNK * 1000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = FAST_RATE,
               channels: int = 1, sampwidth: int = 2) -> Path:
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype('<i2').tobytes())
    return path


def _ramp(n_chunks: int) -> np.ndarray:
    """n_chunks worth of samples whose values are their own index."""
    return np.arange(n_chunks * CHUNK, dtype=np.int16)


def _wait_for_finish(pipeline: FilePlaybackPipeline, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pipeline.finished:
            return True
        time.sleep(0.005)
    return False


class TestLoadWav:
    def test_returns_the_samples(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.array([1, -2, 3], dtype=np.int16))
        assert list(load_wav(path)[0]) == [1, -2, 3]

    def test_returns_the_sample_rate(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(4, dtype=np.int16), sample_rate=16000)
        assert load_wav(path)[1] == 16000

    def test_dtype_is_int16(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(4, dtype=np.int16))
        assert load_wav(path)[0].dtype == np.int16

    def test_samples_are_writable(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(4, dtype=np.int16))
        samples = load_wav(path)[0]
        samples[0] = 5                      # a read-only view would raise here
        assert samples[0] == 5

    def test_stereo_keeps_the_first_channel(self, tmp_path):
        interleaved = np.array([10, 99, 20, 99, 30, 99], dtype=np.int16)
        path = _write_wav(tmp_path / 'a.wav', interleaved, channels=2)
        assert list(load_wav(path)[0]) == [10, 20, 30]

    def test_rejects_non_16_bit_audio(self, tmp_path):
        path = tmp_path / 'a.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(8000)
            wav.writeframes(b'\x00\x01\x02')
        with pytest.raises(ValueError, match='16-bit'):
            load_wav(path)


class TestResolvePlaybackPath:
    def test_bare_filename_resolves_against_the_recording_directory(self):
        assert resolve_playback_path('event.wav', Path('/rec')) == Path('/rec/event.wav')

    def test_path_with_a_directory_is_used_as_given(self):
        assert resolve_playback_path(Path('/other/event.wav'), Path('/rec')) \
            == Path('/other/event.wav')

    def test_explicitly_relative_path_is_used_as_given(self):
        assert resolve_playback_path('./event.wav', Path('/rec')) == Path('./event.wav')


class TestFilePlaybackPipeline:
    def test_exposes_the_files_sample_rate(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(2), sample_rate=16000)
        with FilePlaybackPipeline(path) as pipeline:
            assert pipeline.sample_rate == 16000

    def test_exposes_the_files_duration(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(8000, dtype=np.int16),
                          sample_rate=16000)
        with FilePlaybackPipeline(path) as pipeline:
            assert pipeline.duration == pytest.approx(0.5)

    def test_feeds_the_whole_file(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(4))
        with FilePlaybackPipeline(path) as pipeline:
            assert _wait_for_finish(pipeline)
            assert pipeline.total_samples == 4 * CHUNK

    def test_audio_arrives_in_order(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with FilePlaybackPipeline(path) as pipeline:
            _wait_for_finish(pipeline)
            snapshot = pipeline.get_snapshot(3 * CHUNK)
        assert np.array_equal(snapshot, _ramp(3))

    def test_partial_trailing_chunk_is_dropped(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(2 * CHUNK + 17, dtype=np.int16))
        with FilePlaybackPipeline(path) as pipeline:
            assert _wait_for_finish(pipeline)
            assert pipeline.total_samples == 2 * CHUNK

    def test_not_finished_before_the_file_runs_out(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(4), sample_rate=CHUNK * 4)
        with FilePlaybackPipeline(path) as pipeline:   # 4 s of audio: still playing
            assert pipeline.finished is False

    def test_close_stops_the_feeder(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(40), sample_rate=CHUNK * 4)
        pipeline = FilePlaybackPipeline(path)
        pipeline.close()
        assert pipeline._thread.is_alive() is False

    def test_position_starts_at_the_beginning(self, tmp_path):
        """Not exactly zero: the first chunk is due immediately, and the position is
        quantised to whole chunks, so it has already moved by the time anyone looks."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(20), sample_rate=CHUNK * 4)
        with FilePlaybackPipeline(path) as pipeline:
            assert pipeline.position <= pipeline.duration / 10

    def test_position_advances_with_playback(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(8))
        with FilePlaybackPipeline(path) as pipeline:
            _wait_for_finish(pipeline)
            assert pipeline.position == pytest.approx(pipeline.duration)


class TestTransport:
    """Pause, resume and restart, driven the way the toolbar drives them."""

    def _slow(self, tmp_path, chunks=40):
        """A file long enough to still be playing while a test pokes at it."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(chunks), sample_rate=CHUNK * 10)
        return FilePlaybackPipeline(path)

    def test_starts_playing(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            assert pipeline.paused is False

    def test_pause_stops_the_audio(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.15)
            pipeline.pause()
            time.sleep(0.05)                 # let any in-flight chunk land
            settled = pipeline.total_samples
            time.sleep(0.3)
            assert pipeline.total_samples == settled

    def test_pause_holds_the_position(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.15)
            pipeline.pause()
            time.sleep(0.05)
            settled = pipeline.position
            time.sleep(0.3)
            assert pipeline.position == settled

    def test_resume_continues_from_where_it_stopped(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.15)
            pipeline.pause()
            time.sleep(0.05)
            settled = pipeline.position
            pipeline.resume()
            time.sleep(0.2)
            assert pipeline.position > settled

    def test_resume_does_not_replay_what_was_already_fed(self, tmp_path):
        """Audio is a stream to everything downstream, so a resume that re-fed the
        chunk it paused on would show up as the same instant happening twice."""
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.15)
            pipeline.pause()
            time.sleep(0.05)
            before = pipeline.total_samples
            pipeline.resume()
            time.sleep(0.2)
            after = pipeline.total_samples
        assert (after - before) % CHUNK == 0 and after > before

    def test_toggle_pauses_then_resumes(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            pipeline.toggle_pause()
            paused = pipeline.paused
            pipeline.toggle_pause()
            assert (paused, pipeline.paused) == (True, False)

    def test_restart_returns_to_the_beginning(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.25)
            pipeline.restart()
            assert pipeline.position < 0.1

    def test_restart_resumes_a_paused_file(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            pipeline.pause()
            pipeline.restart()
            assert pipeline.paused is False

    def test_restart_plays_the_file_again(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.2)
            before = pipeline.total_samples
            pipeline.restart()
            time.sleep(0.2)
            assert pipeline.total_samples > before

    def test_restart_empties_the_ring_buffer(self, tmp_path):
        """Otherwise an analyzer reset alongside the restart is handed the tail of
        the pass just abandoned, and locks onto it before the new pass arrives."""
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.25)
            pipeline.pause()
            time.sleep(0.05)
            pipeline.restart()
            assert pipeline.read_from(0).samples.size == 0

    def test_restart_keeps_the_sample_counter_monotonic(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.25)
            pipeline.pause()
            time.sleep(0.05)
            before = pipeline.total_samples
            pipeline.restart()
            assert pipeline.total_samples >= before

    def test_restart_clears_the_finished_flag(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with FilePlaybackPipeline(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.restart()
            assert pipeline.finished is False

    def test_resume_at_the_end_does_nothing(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with FilePlaybackPipeline(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.resume()
            assert pipeline.finished is True

    def test_pausing_does_not_spin(self, tmp_path):
        """The feeder blocks on its condition rather than polling.  Measured as CPU
        time, because a spin loop looks identical to a blocked thread from outside
        until you ask how much of a core it is eating."""
        with self._slow(tmp_path) as pipeline:
            pipeline.pause()
            time.sleep(0.05)
            start = time.process_time()
            time.sleep(0.5)
            assert time.process_time() - start < 0.05

    def test_sitting_at_the_end_does_not_spin(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with FilePlaybackPipeline(path) as pipeline:
            _wait_for_finish(pipeline)
            start = time.process_time()
            time.sleep(0.5)
            assert time.process_time() - start < 0.05

    def test_close_wakes_a_paused_feeder(self, tmp_path):
        pipeline = self._slow(tmp_path)
        pipeline.pause()
        pipeline.close()
        assert pipeline._thread.is_alive() is False

    def test_playback_runs_at_the_files_rate(self, tmp_path):
        # 20 chunks at 20 chunks/second should take about a second, not zero: the
        # point of playback is that a screen recording of it runs at real speed.
        path = _write_wav(tmp_path / 'a.wav', _ramp(20), sample_rate=CHUNK * 20)
        start = time.monotonic()
        with FilePlaybackPipeline(path) as pipeline:
            assert _wait_for_finish(pipeline, timeout=5.0)
        assert time.monotonic() - start == pytest.approx(1.0, abs=0.3)

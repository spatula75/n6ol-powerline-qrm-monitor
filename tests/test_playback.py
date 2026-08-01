"""Tests for .wav loading, path resolution, and the file-backed playback pipeline."""

import threading
import time
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from buzz.playback import (
    FilePlaybackPipeline, apply_gain, load_wav, resolve_playback_path,
)
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


def _playing(path, **kwargs) -> FilePlaybackPipeline:
    """A pipeline that has been told to play, which is the only way one is used.

    Opening a file no longer starts it: the display has to exist first.  Tests want
    the started state almost everywhere, so they say so once, here.
    """
    pipeline = FilePlaybackPipeline(path, **kwargs)
    pipeline.start()
    return pipeline


def _paced(stream, period: float = 0.01) -> None:
    """Make a mocked OutputStream block in write() the way a real one does.

    write() on a real device returns when the card has room, and that is what paces
    the feeder whenever audio is being heard - the deadline schedule stands down and
    the sound card becomes the clock.  A bare MagicMock returns instantly, so an
    unmuted replay reaches the end of the file in microseconds and any test meaning
    to ask what happens *during* playback is quietly asking about afterwards instead.
    """
    stream.return_value.write.side_effect = lambda _: time.sleep(period)


def _eventually(predicate, timeout: float = 2.0) -> bool:
    """Wait for something the feeder thread is expected to get round to."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _wait_for_finish(pipeline: FilePlaybackPipeline, timeout: float = 5.0) -> bool:
    return _eventually(lambda: pipeline.finished, timeout)


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
        with _playing(path) as pipeline:
            assert pipeline.sample_rate == 16000

    def test_exposes_the_files_duration(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(16 * CHUNK, dtype=np.int16),
                          sample_rate=16000)
        with _playing(path) as pipeline:
            assert pipeline.duration == pytest.approx(16 * CHUNK / 16000)

    def test_duration_counts_only_what_will_be_played(self, tmp_path):
        """The trailing partial chunk is never fed, so counting it would leave the
        transport's time index unable to reach its own total."""
        path = _write_wav(tmp_path / 'a.wav', np.zeros(2 * CHUNK + 17, dtype=np.int16),
                          sample_rate=16000)
        with _playing(path) as pipeline:
            assert pipeline.duration == pytest.approx(2 * CHUNK / 16000)

    def test_feeds_the_whole_file(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(4))
        with _playing(path) as pipeline:
            assert _wait_for_finish(pipeline)
            assert pipeline.total_samples == 4 * CHUNK

    def test_audio_arrives_in_order(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            snapshot = pipeline.get_snapshot(3 * CHUNK)
        assert np.array_equal(snapshot, _ramp(3))

    def test_partial_trailing_chunk_is_dropped(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', np.zeros(2 * CHUNK + 17, dtype=np.int16))
        with _playing(path) as pipeline:
            assert _wait_for_finish(pipeline)
            assert pipeline.total_samples == 2 * CHUNK

    def test_not_finished_before_the_file_runs_out(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(4), sample_rate=CHUNK * 4)
        with _playing(path) as pipeline:   # 4 s of audio: still playing
            assert pipeline.finished is False

    def test_close_stops_the_feeder(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(40), sample_rate=CHUNK * 4)
        pipeline = _playing(path)
        pipeline.close()
        assert pipeline._thread.is_alive() is False

    def test_position_starts_at_the_beginning(self, tmp_path):
        """Not exactly zero: the first chunk is due immediately, and the position is
        quantised to whole chunks, so it has already moved by the time anyone looks."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(20), sample_rate=CHUNK * 4)
        with _playing(path) as pipeline:
            assert pipeline.position <= pipeline.duration / 10

    def test_position_advances_with_playback(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(8))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            assert pipeline.position == pytest.approx(pipeline.duration)


class TestTransport:
    """Pause, resume and restart, driven the way the toolbar drives them."""

    def _slow(self, tmp_path, chunks=40):
        """A file long enough to still be playing while a test pokes at it."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(chunks), sample_rate=CHUNK * 10)
        return _playing(path)

    def test_starts_playing(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            assert pipeline.paused is False

    def test_pause_stops_the_audio(self, tmp_path):
        with self._slow(tmp_path) as pipeline:
            time.sleep(0.15)
            pipeline.pause()
            time.sleep(0.05)                 # let any in-flight chunk arrive
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
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.restart()
            assert pipeline.finished is False

    def test_resume_at_the_end_does_nothing(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.resume()
            assert pipeline.finished is True

    def test_toggle_at_the_end_does_not_pause(self, tmp_path):
        """The space bar reaches this even though the Play button is greyed out at the
        end of the file.  Pausing there would leave the transport paused at a position
        resume() rightly refuses to continue from, and disagreeing with the button."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.toggle_pause()
            assert pipeline.paused is False

    def test_restart_after_a_toggle_at_the_end_plays_again(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            pipeline.toggle_pause()
            pipeline.restart()
            assert (pipeline.finished, pipeline.paused) == (False, False)

    def test_pausing_does_not_spin(self, tmp_path):
        """The feeder blocks on its condition rather than polling.  Measured as CPU
        time, because a spin loop looks identical to a blocked thread from outside
        until you ask how much of a core it is eating.

        process_time() counts every thread in the process, so this reads as near zero
        only because nothing else here is working while it sleeps.  Running the suite
        in parallel - pytest-xdist and friends - would make it measure the machine
        rather than the feeder.
        """
        with self._slow(tmp_path) as pipeline:
            pipeline.pause()
            time.sleep(0.05)
            start = time.process_time()
            time.sleep(0.5)
            assert time.process_time() - start < 0.05

    def test_sitting_at_the_end_does_not_spin(self, tmp_path):
        path = _write_wav(tmp_path / 'a.wav', _ramp(3))
        with _playing(path) as pipeline:
            _wait_for_finish(pipeline)
            start = time.process_time()
            time.sleep(0.5)
            assert time.process_time() - start < 0.05

    def test_close_wakes_a_paused_feeder(self, tmp_path):
        pipeline = self._slow(tmp_path)
        pipeline.pause()
        pipeline.close()
        assert pipeline._thread.is_alive() is False

    def test_no_output_device_is_opened_by_default(self, tmp_path):
        """Muted is the default so that building a pipeline - in a test, on a
        headless box - never reaches for a sound card nobody asked for."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._slow(tmp_path):
                time.sleep(0.15)
        stream.assert_not_called()


class TestAudioOutput:
    """Muting is the absence of an output stream, not a flag that zeroes samples.

    That keeps muted playback on exactly the code path that existed before playback
    could be heard: paced by a monotonic deadline, touching no audio device at all.
    """

    # A mute is picked up once per chunk, so these run at 100 chunks/second: fast
    # enough that a tenth of a second covers several passes of the feeder, long
    # enough overall that the file does not run out mid-test.
    def _pipeline(self, tmp_path, muted=True, chunks=200):
        path = _write_wav(tmp_path / 'a.wav', _ramp(chunks), sample_rate=CHUNK * 100)
        return _playing(path, muted=muted)

    def test_muted_by_default(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path) as pipeline:
                assert pipeline.muted is True

    def test_unmuting_opens_a_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.15)
        stream.assert_called_once()

    def test_the_stream_matches_the_file(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.15)
        assert stream.call_args.kwargs['samplerate'] == CHUNK * 100
        assert stream.call_args.kwargs['dtype'] == 'int16'

    def test_audio_is_written_to_the_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.15)
        assert stream.return_value.write.called

    def test_muting_again_closes_the_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.set_muted(True)
                time.sleep(0.1)
        stream.return_value.close.assert_called()

    def test_muting_aborts_rather_than_draining(self, tmp_path):
        """stop() plays out what is queued; a mute button with a fifth of a second
        of lag in it is a broken one."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.set_muted(True)
                time.sleep(0.1)
        stream.return_value.abort.assert_called()

    def test_pausing_releases_the_stream(self, tmp_path):
        """Nothing is written to the card while the feeder is parked, and a stream
        left open with nothing arriving underruns for as long as the pause lasts."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.pause()
                time.sleep(0.1)
                assert pipeline._output is None

    def test_pausing_aborts_rather_than_draining(self, tmp_path):
        """Pause is a "stop now" for the same reason mute is."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                stream.return_value.abort.reset_mock()
                pipeline.pause()
                time.sleep(0.1)
        stream.return_value.abort.assert_called()

    def test_resuming_reopens_the_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.pause()
                time.sleep(0.1)
                pipeline.resume()
                time.sleep(0.1)
                assert pipeline._output is not None

    def test_the_end_of_the_file_releases_the_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path, muted=False, chunks=5) as pipeline:
                assert _wait_for_finish(pipeline)
                assert _eventually(lambda: pipeline._output is None)

    def test_the_end_of_the_file_drains_rather_than_aborting(self, tmp_path):
        """The queue there is the last of the recording, not something anyone asked
        to silence; cutting it off would end the replay on the step the recorder's
        fade-out exists to prevent."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path, muted=False, chunks=5) as pipeline:
                assert _wait_for_finish(pipeline)
                assert _eventually(lambda: pipeline._output is None)
        stream.return_value.stop.assert_called()
        stream.return_value.abort.assert_not_called()

    def test_restarting_from_the_end_opens_a_stream_again(self, tmp_path):
        """The device is released at the end of the file, so Restart has to reach for
        it again rather than assuming it still holds one."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path, muted=False, chunks=5) as pipeline:
                assert _wait_for_finish(pipeline)
                assert _eventually(lambda: pipeline._output is None)
                opened = stream.call_count
                pipeline.restart()
                assert _eventually(lambda: stream.call_count == opened + 1)

    def test_muting_does_not_lose_our_place(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.pause()
                time.sleep(0.05)
                before = pipeline.position
                pipeline.set_muted(True)
                time.sleep(0.1)
                assert pipeline.position == before

    def test_switching_clocks_keeps_the_audio_contiguous(self, tmp_path):
        """The point of the exercise: muting must not drop a chunk or repeat one, or
        the analyzer loses phase over a discontinuity nobody asked for."""
        path = _write_wav(tmp_path / 'a.wav', _ramp(30), sample_rate=CHUNK * 200)
        with patch('buzz.playback.sd.OutputStream'):
            with _playing(path, muted=True) as pipeline:
                time.sleep(0.05)
                pipeline.set_muted(False)
                time.sleep(0.05)
                pipeline.set_muted(True)
                _wait_for_finish(pipeline)
                span = pipeline.read_from(0)
        assert np.array_equal(span.samples, _ramp(30)[span.start:span.end])

    def test_muting_rebases_the_schedule(self, tmp_path):
        """Without this the origin is left where it was before the stream opened,
        and the loop races through the rest of the file catching up to it."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                switched = time.monotonic()
                pipeline.set_muted(True)
                time.sleep(0.1)
                assert pipeline._origin >= switched

    def test_a_device_that_will_not_open_leaves_playback_silent(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream', side_effect=OSError('no device')):
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.15)
                assert pipeline._output is None

    def test_a_device_that_will_not_open_does_not_stop_playback(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream', side_effect=OSError('no device')):
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.15)
                assert pipeline.position > 0

    def test_a_failed_device_is_not_retried_every_chunk(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream', side_effect=OSError('no')) as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.2)
        assert stream.call_count == 1

    def test_asking_again_retries_the_device(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream', side_effect=OSError('no')) as stream:
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                pipeline.set_muted(True)
                pipeline.set_muted(False)
                time.sleep(0.1)
        assert stream.call_count == 2

    def test_availability_follows_the_device_check(self, tmp_path):
        with patch('buzz.playback.sd.check_output_settings', side_effect=OSError('nope')):
            with self._pipeline(tmp_path) as pipeline:
                assert pipeline.output_available is False

    def test_toggle_flips_the_mute_state(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path) as pipeline:
                pipeline.toggle_mute()
                assert pipeline.muted is False

    def test_restart_discards_queued_audio(self, tmp_path):
        """Otherwise the abandoned pass plays out after the click, and every chunk
        after it trails the display by an output buffer."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                stream.return_value.abort.reset_mock()
                pipeline.restart()
                time.sleep(0.1)
        stream.return_value.abort.assert_called()

    def test_restart_keeps_playing_the_audio(self, tmp_path):
        """Discarded, not switched off: the stream is started again, not closed."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            _paced(stream)
            with self._pipeline(tmp_path) as pipeline:
                pipeline.set_muted(False)
                time.sleep(0.1)
                opened = stream.call_count
                pipeline.restart()
                time.sleep(0.1)
                assert pipeline._output is not None
        assert stream.call_count == opened      # the same device, not a new one

    def test_restart_while_muted_touches_no_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path) as pipeline:
                time.sleep(0.05)
                pipeline.restart()
                time.sleep(0.05)
        stream.assert_not_called()

    def test_a_stuck_feeder_keeps_its_stream(self, tmp_path):
        """close() joins with a timeout, so a feeder blocked in write() on a device
        that stopped draining outlives it.  Closing the stream from under that write
        is undefined behaviour; leaking it on a misbehaving device is the lesser
        evil, and the process is on its way out regardless."""
        with patch('buzz.playback.sd.OutputStream') as stream:
            stuck = threading.Event()
            stream.return_value.write.side_effect = lambda _: stuck.wait()
            pipeline = self._pipeline(tmp_path, muted=False)
            time.sleep(0.15)
            pipeline.close()
            still_running = pipeline._thread.is_alive()
            stuck.set()                     # let the feeder go before the test ends
        assert still_running is True
        stream.return_value.close.assert_not_called()

    def test_close_releases_the_stream(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            pipeline = self._pipeline(tmp_path)
            pipeline.set_muted(False)
            time.sleep(0.1)
            pipeline.close()
        stream.return_value.close.assert_called()


class TestDeferredStart:
    """Nothing plays until the caller says so, so a replay cannot start before the
    window that is supposed to be showing it."""

    def _pipeline(self, tmp_path, **kwargs):
        path = _write_wav(tmp_path / 'a.wav', _ramp(20), sample_rate=CHUNK * 100)
        return FilePlaybackPipeline(path, **kwargs)

    def test_nothing_is_fed_before_start(self, tmp_path):
        with self._pipeline(tmp_path) as pipeline:
            time.sleep(0.1)
            assert pipeline.total_samples == 0

    def test_the_position_stays_at_the_beginning(self, tmp_path):
        with self._pipeline(tmp_path) as pipeline:
            time.sleep(0.1)
            assert pipeline.position == 0.0

    def test_no_audio_device_is_opened_before_start(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path, muted=False):
                time.sleep(0.1)
        stream.assert_not_called()

    def test_start_begins_playback(self, tmp_path):
        with self._pipeline(tmp_path) as pipeline:
            pipeline.start()
            time.sleep(0.1)
            assert pipeline.total_samples > 0

    def test_start_is_idempotent(self, tmp_path):
        with self._pipeline(tmp_path) as pipeline:
            pipeline.start()
            pipeline.start()            # a second call must not raise
            time.sleep(0.05)
            assert pipeline.total_samples > 0

    def test_closing_without_starting_is_safe(self, tmp_path):
        pipeline = self._pipeline(tmp_path)
        pipeline.close()
        assert pipeline.total_samples == 0


class TestApplyGain:
    def test_unity_returns_the_audio_untouched(self):
        samples = np.array([1, -2, 3], dtype=np.int16)
        assert apply_gain(samples, 1.0) is samples

    def test_six_db_roughly_doubles_amplitude(self):
        samples = np.array([1000, -1000], dtype=np.int16)
        assert list(apply_gain(samples, 10 ** (6 / 20))) == [1995, -1995]

    def test_negative_gain_attenuates(self):
        samples = np.array([1000], dtype=np.int16)
        assert apply_gain(samples, 10 ** (-6 / 20))[0] == 501

    def test_output_is_still_int16(self):
        samples = np.array([1000], dtype=np.int16)
        assert apply_gain(samples, 2.0).dtype == np.int16

    def test_too_much_gain_clips_rather_than_wrapping(self):
        """The reason this goes through float32: in int16 the multiply would wrap,
        and a passage a few dB too loud would come back as noise, not as loud."""
        samples = np.array([20000, -20000], dtype=np.int16)
        assert list(apply_gain(samples, 10 ** (10 / 20))) == [32767, -32768]

    def test_clipping_holds_at_the_rail(self):
        samples = np.array([30000], dtype=np.int16)
        assert apply_gain(samples, 100.0)[0] == 32767


class TestPlaybackGain:
    def _pipeline(self, tmp_path, gain_db, chunks=200):
        path = _write_wav(tmp_path / 'a.wav', _ramp(chunks), sample_rate=CHUNK * 100)
        return _playing(path, muted=False, gain_db=gain_db)

    def test_gain_reaches_the_sound_card(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path, gain_db=6.0):
                time.sleep(0.1)
        written = stream.return_value.write.call_args_list[0].args[0]
        assert np.array_equal(written, apply_gain(_ramp(1)[:CHUNK], 10 ** (6 / 20)))

    def test_gain_does_not_reach_the_analyzer(self, tmp_path):
        """The whole point: the monitor must report the level that was recorded, not
        the level someone chose in order to hear it on laptop speakers."""
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path, gain_db=20.0) as pipeline:
                time.sleep(0.1)
                span = pipeline.read_from(0)
        assert np.array_equal(span.samples, _ramp(200)[span.start:span.end])

    def test_no_gain_writes_the_file_untouched(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream') as stream:
            with self._pipeline(tmp_path, gain_db=0.0):
                time.sleep(0.1)
        written = stream.return_value.write.call_args_list[0].args[0]
        assert np.array_equal(written, _ramp(1)[:CHUNK])

    def test_gain_is_reported_in_db(self, tmp_path):
        with patch('buzz.playback.sd.OutputStream'):
            with self._pipeline(tmp_path, gain_db=10.0) as pipeline:
                assert pipeline.gain_db == 10.0


class TestPacing:
    def test_playback_runs_at_the_files_rate(self, tmp_path):
        # 20 chunks at 20 chunks/second should take about a second, not zero: the
        # point of playback is that a screen recording of it runs at real speed.
        path = _write_wav(tmp_path / 'a.wav', _ramp(20), sample_rate=CHUNK * 20)
        start = time.monotonic()
        with _playing(path) as pipeline:
            assert _wait_for_finish(pipeline, timeout=5.0)
        assert time.monotonic() - start == pytest.approx(1.0, abs=0.3)

"""Tests for AudioPipeline: ring buffer, callback, snapshot, and wait_for_data."""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline

SAMPLE_RATE = 16000
CHUNK = AudioPipeline.CHUNK_SIZE


def _make_config() -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.device_index = 0
    cfg.audio.input_device_name = 'Test, DirectSound'
    cfg.audio.sample_rate = SAMPLE_RATE
    return cfg


def _make_pipeline(config=None):
    """Return (pipeline, mock_sd_stream, callback) with sd.InputStream mocked."""
    if config is None:
        config = _make_config()
    with patch('buzz.sampler.sd.InputStream') as mock_cls:
        mock_sd = MagicMock()
        mock_cls.return_value = mock_sd
        pipeline = AudioPipeline(config, 0)
        callback = mock_cls.call_args.kwargs['callback']
    return pipeline, mock_sd, callback


def _fire(callback, amplitude: int = 5000, n: int = CHUNK) -> None:
    """Simulate one PortAudio callback delivering n samples at the given amplitude."""
    data = np.full((n, 1), amplitude, dtype=np.int16)
    callback(data, n, None, None)


class TestAudioPipelineInit:
    def test_stream_started_on_init(self):
        _, mock_sd, _ = _make_pipeline()
        mock_sd.start.assert_called_once()

    def test_buffer_empty_on_init(self):
        pipeline, _, _ = _make_pipeline()
        assert len(pipeline._buffer) == 0

    def test_chunk_size_is_power_of_two(self):
        assert CHUNK > 0 and (CHUNK & (CHUNK - 1)) == 0

    def test_blocksize_passed_to_input_stream(self):
        config = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            AudioPipeline(config, 0)
        assert mock_cls.call_args.kwargs['blocksize'] == CHUNK

    def test_device_index_passed_to_input_stream(self):
        config = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            AudioPipeline(config, 7)
        assert mock_cls.call_args.kwargs['device'] == 7


class TestAudioPipelineCallback:
    def test_callback_appends_chunk_to_buffer(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback)
        assert len(pipeline._buffer) == 1

    def test_callback_stores_correct_amplitude(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=1234)
        assert pipeline._buffer[0][0] == 1234

    def test_callback_copies_data(self):
        pipeline, _, callback = _make_pipeline()
        data = np.full((CHUNK, 1), 999, dtype=np.int16)
        callback(data, CHUNK, None, None)
        data[0, 0] = 0  # mutate original
        assert pipeline._buffer[0][0] == 999  # buffer unaffected

    def test_callback_notifies_condition(self):
        pipeline, _, callback = _make_pipeline()
        notified = []

        def _waiter():
            with pipeline._condition:
                pipeline._condition.wait(timeout=1.0)
                notified.append(True)

        t = threading.Thread(target=_waiter)
        t.start()
        _fire(callback)
        t.join(timeout=2.0)
        assert notified

    def test_buffer_drops_oldest_when_full(self):
        pipeline, _, callback = _make_pipeline()
        # fill the buffer to capacity
        for i in range(pipeline._buffer.maxlen + 5):
            _fire(callback, amplitude=i)
        assert len(pipeline._buffer) == pipeline._buffer.maxlen
        # oldest chunks (low amplitude) have been dropped; newest are at the end
        assert pipeline._buffer[-1][0] == pipeline._buffer.maxlen + 4


class TestAudioPipelineGetSnapshot:
    def test_returns_zeros_when_buffer_empty(self):
        pipeline, _, _ = _make_pipeline()
        result = pipeline.get_snapshot(CHUNK)
        assert result.shape == (CHUNK,)
        assert np.all(result == 0)

    def test_returns_correct_length(self):
        pipeline, _, callback = _make_pipeline()
        for _ in range(10):
            _fire(callback)
        result = pipeline.get_snapshot(CHUNK * 3)
        assert len(result) == CHUNK * 3

    def test_returns_most_recent_data(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=100)
        _fire(callback, amplitude=200)
        _fire(callback, amplitude=300)
        result = pipeline.get_snapshot(CHUNK)
        assert np.all(result == 300)

    def test_assembles_multiple_chunks(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=111)
        _fire(callback, amplitude=222)
        result = pipeline.get_snapshot(CHUNK * 2)
        assert result[0] == 111
        assert result[CHUNK] == 222

    def test_truncates_to_requested_length(self):
        pipeline, _, callback = _make_pipeline()
        for _ in range(5):
            _fire(callback)
        result = pipeline.get_snapshot(100)
        assert len(result) == 100

    def test_offset_shifts_window_back(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=111)
        _fire(callback, amplitude=222)
        # with offset=CHUNK, skip the last chunk and read the one before it
        result = pipeline.get_snapshot(CHUNK, offset=CHUNK)
        assert np.all(result == 111)

    def test_offset_zero_is_most_recent(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=111)
        _fire(callback, amplitude=222)
        result = pipeline.get_snapshot(CHUNK, offset=0)
        assert np.all(result == 222)

    def test_non_overlapping_windows_are_distinct(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback, amplitude=100)
        _fire(callback, amplitude=200)
        _fire(callback, amplitude=300)
        w0 = pipeline.get_snapshot(CHUNK, offset=2 * CHUNK)
        w1 = pipeline.get_snapshot(CHUNK, offset=CHUNK)
        w2 = pipeline.get_snapshot(CHUNK, offset=0)
        assert np.all(w0 == 100)
        assert np.all(w1 == 200)
        assert np.all(w2 == 300)


class TestAudioPipelineWaitForData:
    def test_returns_true_immediately_when_data_available(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback)
        assert pipeline.wait_for_data(CHUNK, timeout=0.1) is True

    def test_returns_false_on_timeout_when_no_data(self):
        pipeline, _, _ = _make_pipeline()
        assert pipeline.wait_for_data(CHUNK, timeout=0.05) is False

    def test_blocks_until_callback_fires(self):
        pipeline, _, callback = _make_pipeline()
        results = []

        def _fire_after_start():
            _fire(callback)

        t = threading.Thread(target=_fire_after_start)
        t.start()
        results.append(pipeline.wait_for_data(CHUNK, timeout=2.0))
        t.join()
        assert results[0] is True

    def test_requires_enough_chunks_for_requested_samples(self):
        pipeline, _, callback = _make_pipeline()
        _fire(callback)  # one chunk = CHUNK samples
        # requesting more than one chunk should still wait
        assert pipeline.wait_for_data(CHUNK + 1, timeout=0.05) is False
        _fire(callback)  # now two chunks cover it
        assert pipeline.wait_for_data(CHUNK + 1, timeout=0.05) is True


class TestAudioPipelineClose:
    def test_close_stops_stream(self):
        config = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            pipeline = AudioPipeline(config, 0)
        pipeline.close()
        mock_sd.stop.assert_called_once()

    def test_close_closes_stream(self):
        config = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            pipeline = AudioPipeline(config, 0)
        pipeline.close()
        mock_sd.close.assert_called_once()


class TestAudioPipelineContextManager:
    def test_enter_returns_self(self):
        pipeline, _, _ = _make_pipeline()
        assert pipeline.__enter__() is pipeline

    def test_exit_closes_stream(self):
        config = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            with AudioPipeline(config, 0):
                pass
        mock_sd.stop.assert_called_once()
        mock_sd.close.assert_called_once()

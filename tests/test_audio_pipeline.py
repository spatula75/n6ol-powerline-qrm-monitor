"""Tests for AudioPipeline: ring buffer, callback, snapshot, and wait_for_data."""
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.config import BuzzConfig
from buzz.sampler import AudioPipeline, RingBufferPipeline, buffer_chunks

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


def _fire_indexed(callback, chunk_index: int) -> None:
    """Deliver a chunk whose sample values are their global stream indices."""
    start = chunk_index * CHUNK
    data = (np.arange(start, start + CHUNK) % 32768).astype(np.int16).reshape(-1, 1)
    callback(data, CHUNK, None, None)


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

class TestAudioPipelineGetSnapshotAligned:
    """Phase-aligned snapshots: the window must end at a multiple of `align`
    samples in the global stream, so every window shares a phase origin no
    matter where the chunk-quantised tail happens to be."""

    ALIGN = 400   # 3 pulse periods at 16 kHz / 120 pps

    def _pipeline_with_chunks(self, n_chunks: int):
        pipeline, _, callback = _make_pipeline()
        for i in range(n_chunks):
            _fire_indexed(callback, i)
        return pipeline, callback

    def test_default_window_ends_at_tail(self):
        pipeline, _ = self._pipeline_with_chunks(33)
        result = pipeline.get_snapshot(1600)
        assert result[-1] == 33 * CHUNK - 1

    def test_aligned_window_ends_at_align_multiple(self):
        # tail = 33 × 512 = 16896; greatest multiple of 400 below it is 16800
        pipeline, _ = self._pipeline_with_chunks(33)
        result = pipeline.get_snapshot(1600, align=self.ALIGN)
        assert len(result) == 1600
        assert result[-1] == 16800 - 1
        assert result[0] == 16800 - 1600

    def test_aligned_windows_share_phase_origin_as_tail_advances(self):
        pipeline, callback = self._pipeline_with_chunks(33)
        for i in range(33, 40):
            _fire_indexed(callback, i)
            window = pipeline.get_snapshot(800, align=self.ALIGN)
            assert (int(window[-1]) + 1) % self.ALIGN == 0

    def test_unaligned_windows_move_with_tail(self):
        """Documents the problem align solves: default windows end mid-period."""
        pipeline, _ = self._pipeline_with_chunks(33)
        assert (int(pipeline.get_snapshot(800)[-1]) + 1) % self.ALIGN != 0

    def test_empty_buffer_returns_zeros_with_align(self):
        pipeline, _, _ = _make_pipeline()
        result = pipeline.get_snapshot(CHUNK, align=self.ALIGN)
        assert np.all(result == 0)


class TestAudioPipelineTotalSamples:
    def test_zero_before_any_callback(self):
        pipeline, _, _ = _make_pipeline()
        assert pipeline.total_samples == 0

    def test_counts_every_captured_sample(self):
        pipeline, _, callback = _make_pipeline()
        for _ in range(3):
            _fire(callback)
        assert pipeline.total_samples == 3 * CHUNK

    def test_keeps_growing_after_buffer_discards_old_chunks(self):
        pipeline, _, callback = _make_pipeline()
        n = pipeline._buffer.maxlen + 10
        for _ in range(n):
            _fire(callback)
        assert pipeline.total_samples == n * CHUNK


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


class TestReadFrom:
    """Sequential reads: every sample exactly once, in order, gaps visible."""

    def _pipeline(self, n_chunks: int):
        pipeline = RingBufferPipeline()
        for i in range(n_chunks):
            pipeline._append(np.full(CHUNK, i + 1, dtype=np.int16))
        return pipeline

    def test_empty_buffer_returns_nothing(self):
        span = RingBufferPipeline().read_from(0)
        assert span.samples.size == 0

    def test_reads_everything_buffered_from_zero(self):
        span = self._pipeline(3).read_from(0)
        assert len(span.samples) == 3 * CHUNK

    def test_span_is_tagged_with_absolute_positions(self):
        span = self._pipeline(3).read_from(0)
        assert (span.start, span.end) == (0, 3 * CHUNK)

    def test_reads_only_what_is_new(self):
        pipeline = self._pipeline(2)
        first = pipeline.read_from(0)
        pipeline._append(np.full(CHUNK, 99, dtype=np.int16))
        second = pipeline.read_from(first.end)
        assert np.all(second.samples == 99)

    def test_consecutive_reads_do_not_overlap(self):
        pipeline = self._pipeline(2)
        first = pipeline.read_from(0)
        pipeline._append(np.zeros(CHUNK, dtype=np.int16))
        assert pipeline.read_from(first.end).start == first.end

    def test_nothing_new_returns_an_empty_span(self):
        pipeline = self._pipeline(2)
        end = pipeline.read_from(0).end
        assert pipeline.read_from(end).samples.size == 0

    def test_nothing_new_still_reports_the_current_position(self):
        pipeline = self._pipeline(2)
        end = pipeline.read_from(0).end
        assert pipeline.read_from(end).start == end

    def test_a_reader_that_falls_behind_gets_what_survived(self):
        pipeline = self._pipeline(400)          # past the ring buffer's capacity
        span = pipeline.read_from(0)
        assert span.start > 0

    def test_a_reader_that_falls_behind_still_reads_to_the_tail(self):
        pipeline = self._pipeline(400)
        span = pipeline.read_from(0)
        assert span.end == 400 * CHUNK

    def test_dropped_audio_is_the_difference_between_request_and_start(self):
        pipeline = self._pipeline(400)
        span = pipeline.read_from(0)
        assert len(span.samples) == span.end - span.start

    def test_a_read_starting_mid_chunk_begins_at_that_sample(self):
        """Only whole chunks are joined, so a position part-way into one has to be
        trimmed off the front of the join rather than assumed to fall on a seam."""
        pipeline = RingBufferPipeline()
        pipeline._append(np.arange(CHUNK, dtype=np.int16))
        span = pipeline.read_from(10)
        assert (span.samples[0], span.start) == (10, 10)

    def test_a_read_starting_mid_chunk_has_the_right_length(self):
        pipeline = RingBufferPipeline()
        pipeline._append(np.arange(CHUNK, dtype=np.int16))
        assert len(pipeline.read_from(10).samples) == CHUNK - 10

    def _uneven(self):
        """Nothing promises every chunk is CHUNK_SIZE - only the live capture callback
        is fixed-size - so a span is assembled by length, never by chunk count."""
        pipeline = RingBufferPipeline()
        for value, length in ((1, 100), (2, 7), (3, 300)):
            pipeline._append(np.full(length, value, dtype=np.int16))
        return pipeline

    def test_uneven_chunks_read_back_in_order(self):
        span = self._uneven().read_from(0)
        assert list(span.samples) == [1] * 100 + [2] * 7 + [3] * 300

    def test_uneven_chunks_trim_from_the_right_place(self):
        # Positions 0-99 hold 1, 100-106 hold 2, 107 on hold 3; so a read from 105
        # starts two samples into the middle chunk.
        span = self._uneven().read_from(105)
        assert list(span.samples) == [2] * 2 + [3] * 300

    def test_a_small_read_does_not_join_the_whole_buffer(self, monkeypatch):
        """A sequential reader asks for the fraction of a second that arrived since it
        last called, five times a second for as long as a recording lasts.  Joining the
        whole ~10 second buffer to keep the newest 512 samples of it is several hundred
        kilobytes copied to throw away, so the span takes only the chunks it touches."""
        # A full buffer at the default rate, which is what the capacity is sized for.
        pipeline = self._pipeline(buffer_chunks(16000, RingBufferPipeline.CHUNK_SIZE))
        end = pipeline.read_from(0).end
        pipeline._append(np.zeros(CHUNK, dtype=np.int16))

        joined, real = [], np.concatenate

        def spy(arrays, *args, **kwargs):
            joined.append(sum(len(a) for a in arrays))
            return real(arrays, *args, **kwargs)

        monkeypatch.setattr(np, 'concatenate', spy)
        pipeline.read_from(end)
        assert joined == [CHUNK]


class TestClear:
    def _pipeline(self, n_chunks: int):
        pipeline = RingBufferPipeline()
        for _ in range(n_chunks):
            pipeline._append(np.full(CHUNK, 1000, dtype=np.int16))
        return pipeline

    def test_discards_buffered_audio(self):
        pipeline = self._pipeline(3)
        pipeline.clear()
        assert pipeline.read_from(0).samples.size == 0

    def test_sample_counter_keeps_going(self):
        """It is the analyzer's audio clock and every phase's origin; winding it back
        would read as time running backwards rather than as a fresh start."""
        pipeline = self._pipeline(3)
        pipeline.clear()
        assert pipeline.total_samples == 3 * CHUNK

    def test_new_audio_is_buffered_again(self):
        pipeline = self._pipeline(3)
        pipeline.clear()
        pipeline._append(np.full(CHUNK, 7, dtype=np.int16))
        assert np.all(pipeline.get_snapshot(CHUNK) == 7)

    def test_snapshot_of_an_emptied_buffer_is_silence(self):
        pipeline = self._pipeline(3)
        pipeline.clear()
        assert np.all(pipeline.get_snapshot(CHUNK) == 0)


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


class TestBufferSizedInSeconds:
    """The buffer holds a duration, not a sample count.

    It doubles as the analyzer's history and as the lead-in an event recording opens
    with, so a fixed sample count made both shrink with the sample rate: 9.6 s at
    16 kHz but only 3.5 s at 44.1 kHz, in proportion to a setting nobody would connect
    to either of them.
    """

    RATES = (8000, 11025, 16000, 22050, 44100, 48000)

    @pytest.mark.parametrize('rate', RATES)
    def test_it_holds_the_same_duration_at_every_rate(self, rate):
        held = RingBufferPipeline(rate).capacity_samples / rate
        assert held == pytest.approx(9.6, abs=0.02), (
            f'At {rate} Hz the buffer holds {held:.2f} s rather than 9.6. The capacity '
            'is meant to be a duration converted to chunks, not a fixed chunk count.')

    def test_the_default_rate_is_unchanged(self):
        """9.6 s at 16 kHz is exactly the 300 chunks of 512 this used to be, so nothing
        moves at the rate the program records at."""
        assert RingBufferPipeline().capacity_samples == 300 * RingBufferPipeline.CHUNK_SIZE

    @pytest.mark.parametrize('rate', RATES)
    def test_it_is_never_shorter_than_it_promises(self, rate):
        """Rounded up: a buffer a chunk short of its stated duration would trim the
        oldest fraction of a second off every recording's lead-in."""
        assert RingBufferPipeline(rate).capacity_samples >= 9.6 * rate

    def test_the_cost_stays_reasonable_at_the_top_of_the_range(self):
        """Sizing by duration is only affordable because the audio is mono int16;
        this is the number that makes it a non-decision."""
        kilobytes = RingBufferPipeline(48000).capacity_samples * 2 / 1024
        assert kilobytes < 1024

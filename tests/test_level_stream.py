"""Tests for LevelStream and AudioSampler.level_stream()."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from buzz.config import BuzzConfig
from buzz.sampler import AudioSampler, LevelStream

SAMPLE_RATE = 16000
PULSE_RATE = 120


def _make_config(offset_db: float = 0.0) -> BuzzConfig:
    cfg = BuzzConfig()
    cfg.audio.device_index = 0
    cfg.audio.input_device_name = 'Test, DirectSound'
    cfg.audio.sample_rate = SAMPLE_RATE
    cfg.audio.pulse_rate = PULSE_RATE
    cfg.station.audio_rf_conversion_db = offset_db
    return cfg


def _make_sampler() -> AudioSampler:
    cfg = _make_config()
    device = {'index': 0, 'name': 'Test', 'hostapi': 0}
    with patch('buzz.sampler.sd.query_devices', return_value=device), \
         patch('buzz.sampler.sd.InputStream', return_value=MagicMock()):
        return AudioSampler(cfg)


def _make_level_stream(cfg=None, blocksize=320):
    """Return (stream, mock_sd_stream_instance, callback) with sd.InputStream mocked."""
    if cfg is None:
        cfg = _make_config()
    with patch('buzz.sampler.sd.InputStream') as mock_cls:
        mock_sd = MagicMock()
        mock_cls.return_value = mock_sd
        stream = LevelStream(cfg, 0, blocksize)
        callback = mock_cls.call_args.kwargs['callback']
    return stream, mock_sd, callback


def _audio(amplitude: int, n: int = 320) -> np.ndarray:
    """One block of zero-mean audio whose mean-absolute level is `amplitude`.

    Alternating +/-amplitude rather than a constant: LSB receiver audio is bipolar,
    and LevelStream removes DC before rectifying, so a constant block is exactly the
    signal the DC correction is designed to null to zero.  A square wave carries the
    intended level through unchanged.
    """
    block = np.full((n, 1), amplitude, dtype=np.int16)
    block[1::2] = -amplitude
    return block


class TestLevelStreamInit:
    def test_stream_started_on_init(self):
        _, mock_sd, _ = _make_level_stream()
        mock_sd.start.assert_called_once()

    def test_initial_dbm_is_sentinel(self):
        stream, _, _ = _make_level_stream()
        assert stream._latest_dbm == -128.0

    def test_offset_stored_from_config(self):
        cfg = _make_config(offset_db=12.5)
        stream, _, _ = _make_level_stream(cfg=cfg)
        assert stream._offset_db == pytest.approx(12.5)

    def test_blocksize_passed_to_input_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            LevelStream(cfg, 0, 160)
        assert mock_cls.call_args.kwargs['blocksize'] == 160


class TestLevelStreamCallback:
    def test_nonzero_audio_sets_dbm_above_sentinel(self):
        stream, _, callback = _make_level_stream()
        callback(_audio(1000), 320, None, None)
        assert stream._latest_dbm > -128.0

    def test_zero_audio_sets_sentinel(self):
        stream, _, callback = _make_level_stream()
        callback(_audio(0), 320, None, None)
        assert stream._latest_dbm == -128.0

    def test_callback_sets_event(self):
        stream, _, callback = _make_level_stream()
        assert not stream._event.is_set()
        callback(_audio(1000), 320, None, None)
        assert stream._event.is_set()

    def test_offset_shifts_dbm(self):
        stream_base, _, cb_base = _make_level_stream(cfg=_make_config(offset_db=0.0))
        stream_offset, _, cb_offset = _make_level_stream(cfg=_make_config(offset_db=10.0))
        audio = _audio(5000)
        cb_base(audio, 320, None, None)
        cb_offset(audio, 320, None, None)
        assert stream_offset._latest_dbm == pytest.approx(stream_base._latest_dbm + 10.0)

    def test_higher_amplitude_gives_higher_dbm(self):
        stream_lo, _, cb_lo = _make_level_stream()
        stream_hi, _, cb_hi = _make_level_stream()
        cb_lo(_audio(500), 320, None, None)
        cb_hi(_audio(5000), 320, None, None)
        assert stream_hi._latest_dbm > stream_lo._latest_dbm


class TestLevelStreamRead:
    def test_read_returns_latest_dbm(self):
        stream, _, callback = _make_level_stream()
        callback(_audio(5000), 320, None, None)
        assert stream.read() == pytest.approx(stream._latest_dbm)

    def test_read_clears_event(self):
        stream, _, callback = _make_level_stream()
        callback(_audio(1000), 320, None, None)
        stream.read()
        assert not stream._event.is_set()

    def test_read_blocks_until_callback_fires(self):
        import threading
        stream, _, callback = _make_level_stream()
        results = []

        def _fire():
            callback(_audio(2000), 320, None, None)

        t = threading.Thread(target=_fire)
        t.start()
        results.append(stream.read())
        t.join()
        assert results[0] > -128.0


class TestLevelStreamClose:
    def test_close_stops_stream(self):
        _, mock_sd, _ = _make_level_stream()
        # need to reconstruct to call close on a live stream object
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd2 = MagicMock()
            mock_cls.return_value = mock_sd2
            stream = LevelStream(cfg, 0, 320)
        stream.close()
        mock_sd2.stop.assert_called_once()

    def test_close_closes_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            stream = LevelStream(cfg, 0, 320)
        stream.close()
        mock_sd.close.assert_called_once()


class TestLevelStreamContextManager:
    def test_enter_returns_self(self):
        stream, _, _ = _make_level_stream()
        assert stream.__enter__() is stream

    def test_exit_closes_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            stream = LevelStream(cfg, 0, 320)
        stream.__exit__(None, None, None)
        mock_sd.stop.assert_called_once()
        mock_sd.close.assert_called_once()

    def test_with_statement_closes_on_normal_exit(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            with LevelStream(cfg, 0, 320):
                pass
        mock_sd.stop.assert_called_once()
        mock_sd.close.assert_called_once()


class TestAudioSamplerLevelStream:
    def test_returns_level_stream_instance(self):
        sampler = _make_sampler()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            ls = sampler.level_stream()
        assert isinstance(ls, LevelStream)

    def test_default_blocksize_is_320(self):
        sampler = _make_sampler()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            sampler.level_stream()
        assert mock_cls.call_args.kwargs['blocksize'] == 320

    def test_custom_blocksize_passed_through(self):
        sampler = _make_sampler()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            sampler.level_stream(blocksize=160)
        assert mock_cls.call_args.kwargs['blocksize'] == 160

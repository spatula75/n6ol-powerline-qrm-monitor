"""Tests for LevelStream and AudioSampler.level_stream()."""
import threading
import time

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from buzz.config import BuzzConfig
from buzz.sampler import (
    AudioSampler,
    LevelStream,
    SoundCardLevelStream,
)

SAMPLE_RATE = 16000
PULSE_RATE = 120


def _make_config(offset_db: float = 0.0) -> BuzzConfig:
    cfg = BuzzConfig()
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
        stream = SoundCardLevelStream(cfg, 0, blocksize)
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
        assert stream.offset_db == pytest.approx(12.5)

    def test_blocksize_passed_to_input_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            SoundCardLevelStream(cfg, 0, 160)
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

    def test_offset_can_be_changed_live_without_reopening_the_stream(self):
        """The setup program's calibration dialog nudges offset_db while a single
        LevelStream keeps running - see LevelStream's own docstring.  A later
        callback must pick up the new value, not the one the stream opened with."""
        stream, _, callback = _make_level_stream(cfg=_make_config(offset_db=0.0))
        audio = _audio(5000)
        callback(audio, 320, None, None)
        before = stream._latest_dbm

        stream.offset_db = 20.0
        callback(audio, 320, None, None)

        assert stream._latest_dbm == pytest.approx(before + 20.0)


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
            stream = SoundCardLevelStream(cfg, 0, 320)
        stream.close()
        mock_sd2.stop.assert_called_once()

    def test_close_closes_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            stream = SoundCardLevelStream(cfg, 0, 320)
        stream.close()
        mock_sd.close.assert_called_once()

    def test_close_unblocks_a_thread_waiting_in_read(self):
        """Regression test: the setup program's calibration dialogs call read()
        via asyncio.to_thread(), and cancelling that Task on dismiss does not stop
        the thread pool worker actually running it - only closing the stream while
        it is parked in Event.wait() does, since no further callback is ever
        coming to set that event once the stream is stopped.  See close()'s own
        comment: an unblocked thread here is what stands between a dismissed
        dialog and a setup program that hangs on exit rather than closing, since
        CPython joins every ThreadPoolExecutor worker before the process can
        exit.  This test hung instead of failing when the fix was reverted -
        exactly the hang this exists to prevent - which is why it bounds the
        wait with a join timeout rather than calling stream.read() directly."""
        import threading
        import time
        stream, _, _ = _make_level_stream()
        result = []

        def _block_in_read():
            result.append(stream.read())

        # daemon=True: if the fix regresses, this thread must not be able to hang
        # the whole test run's own process exit - only this test should fail.
        t = threading.Thread(target=_block_in_read, daemon=True)
        t.start()
        time.sleep(0.05)  # give the thread time to actually reach Event.wait()
        stream.close()
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert result == [stream._latest_dbm]


class TestLevelStreamContextManager:
    def test_enter_returns_self(self):
        stream, _, _ = _make_level_stream()
        assert stream.__enter__() is stream

    def test_exit_closes_stream(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            stream = SoundCardLevelStream(cfg, 0, 320)
        stream.__exit__(None, None, None)
        mock_sd.stop.assert_called_once()
        mock_sd.close.assert_called_once()

    def test_with_statement_closes_on_normal_exit(self):
        cfg = _make_config()
        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_sd = MagicMock()
            mock_cls.return_value = mock_sd
            with SoundCardLevelStream(cfg, 0, 320):
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


class TestTheEmaWeightIsDerivedNotGuessed:
    """`DC_EMA_ALPHA = 0.002` was correct for one block size and one sample rate, and
    silently meant something else for any other.  It is computed from both now.
    """

    def test_it_reproduces_the_literal_it_replaced(self):
        """The old value, at the settings its comment described: 320 samples at
        16 kHz, a 50 Hz block rate, ten seconds.  Equal rather than close, so this
        refactor provably changed nothing for a sound card.
        """
        assert LevelStream.dc_ema_alpha(16_000, 320, 10.0) == 0.002

    def test_a_longer_block_needs_a_heavier_weight(self):
        """Fewer blocks per second means each has to carry more, or the time
        constant stretches.
        """
        assert (LevelStream.dc_ema_alpha(16_000, 640, 10.0)
                == 2 * LevelStream.dc_ema_alpha(16_000, 320, 10.0))

    def test_a_faster_rate_needs_a_lighter_one(self):
        assert LevelStream.dc_ema_alpha(48_000, 320, 10.0) == pytest.approx(
            LevelStream.dc_ema_alpha(16_000, 320, 10.0) / 3)

    def test_the_time_constant_is_what_it_claims(self):
        """One time constant of blocks should leave about 1/e of a step remaining."""
        rate, block, seconds = 16_000, 320, 10.0
        alpha = LevelStream.dc_ema_alpha(rate, block, seconds)
        blocks = round(seconds * rate / block)
        remaining = (1 - alpha) ** blocks
        assert remaining == pytest.approx(1 / np.e, rel=0.01)


class TestBothSourcesAgree:
    """The drift pin the split exists for.

    An operator calibrates audio_rf_conversion_db against whichever meter their
    station uses.  If the two disagreed about DC, about what "level" means, or about
    the conversion to dBm, they would calibrate against a figure the monitor never
    reports and bake the difference into every level that station ever logs.  Nothing
    else in the suite compares them.
    """

    @staticmethod
    def _readings(block, offset_db=0.0, rate=16_000, blocksize=320):
        """The same samples through each subclass, by way of the shared base."""
        from buzz.sampler import LevelStream
        out = []
        for _ in range(2):
            stream = LevelStream.__new__(LevelStream)
            LevelStream.__init__(stream, offset_db, rate, blocksize)
            stream._on_block(block)
            out.append(stream.read())
        return out

    def test_neither_subclass_overrides_the_arithmetic(self):
        """Stated as a test rather than a comment, because a comment would go quietly
        out of date the first time somebody added a method.
        """
        from buzz.sdr import SdrLevelStream
        shared = {'_on_block', 'read', 'close', '__enter__', '__exit__'}
        for cls in (SoundCardLevelStream, SdrLevelStream):
            assert not shared & set(vars(cls)), (
                f'{cls.__name__} overrides {shared & set(vars(cls))}, which decides '
                'the number an operator calibrates against')

    def test_the_same_samples_give_the_same_dbm(self):
        """Driven through each subclass's own entry point: a PortAudio callback for
        one, a converted IQ block for the other.
        """
        from unittest.mock import MagicMock, patch

        from buzz.sdr import SdrLevelStream

        samples = np.array([1000, -1000] * 160, dtype=np.int16)

        with patch('buzz.sampler.sd.InputStream') as mock_cls:
            mock_cls.return_value = MagicMock()
            card = SoundCardLevelStream(_make_config(offset_db=0.0), 0, 320)
        card._callback(samples.reshape(-1, 1), 320, None, None)
        from_card = card.read()

        source, converter = MagicMock(), MagicMock()
        source.block_samples, source.iq_sample_rate = 5_120, 256_000
        converter.audio_sample_rate = 16_000
        converter.convert.return_value = samples
        sdr = SdrLevelStream.__new__(SdrLevelStream)
        LevelStream.__init__(sdr, 0.0, 16_000, 320)
        sdr._converter = converter
        sdr._consume(MagicMock())
        from_sdr = sdr.read()

        assert from_card == from_sdr, (
            f'the sound card reads {from_card} dBm where the receiver reads '
            f'{from_sdr} dBm for identical samples')

    def test_an_empty_converted_block_poisons_nothing(self):
        """The filter returns nothing until it has enough samples, which is always
        true of the first call.

        Checking `_latest_dbm` alone would prove nothing, which is how the first
        version of this test passed against the unguarded code: np.median of an
        empty array is NaN rather than an error, and amplitude_to_dbm reads NaN as
        not-greater-than-zero and hands back the silence sentinel.  The damage is
        upstream of that.  `_dc` becomes NaN, the EMA feeds itself, and every later
        reading is NaN for the life of the stream.
        """
        from unittest.mock import MagicMock

        from buzz.sdr import SdrLevelStream

        converter = MagicMock()
        converter.convert.return_value = np.array([], dtype=np.int16)
        sdr = SdrLevelStream.__new__(SdrLevelStream)
        LevelStream.__init__(sdr, 0.0, 16_000, 320)
        sdr._converter = converter

        sdr._consume(MagicMock())
        assert sdr._dc is None, 'an empty block seeded the DC estimate'
        assert not sdr._event.is_set(), 'an empty block woke a reader with nothing'

        # And the stream still works afterwards, which NaN would have prevented.
        converter.convert.return_value = np.array([1000, -1000] * 160, dtype=np.int16)
        sdr._consume(MagicMock())
        reading = sdr.read(timeout=0.05)
        assert reading is not None and reading > -128.0, (
            f'the reading after an empty block came back {reading}')


class TestReadGivesUpRatherThanHanging:
    """read() used to wait with no timeout, so a source that stopped delivering
    parked the caller on an event nothing would ever set.  close() was the only
    escape, which covers shutdown but not a receiver unplugged mid-calibration.
    """

    def test_nothing_arriving_reports_a_stall(self):
        stream, _, _ = _make_level_stream()
        assert stream.read(timeout=0.05) is None

    def test_a_block_that_does_arrive_still_reads(self):
        stream, _, callback = _make_level_stream()
        callback(_audio(1000), 320, None, None)
        assert stream.read(timeout=0.05) is not None

    def test_it_honors_the_timeout_it_was_given(self):
        """Run on a thread so a read that ignores its timeout fails here rather than
        hanging the suite.  A test that can only be caught by a CI timeout is a bad
        way to learn this broke.
        """
        stream, _, _ = _make_level_stream()
        done = threading.Event()
        threading.Thread(target=lambda: (stream.read(timeout=0.05), done.set()),
                         daemon=True).start()
        assert done.wait(timeout=5.0), (
            'read() ignored its timeout and was still blocked after 5 s')

    def test_a_stall_does_not_clear_the_last_reading(self):
        """The caller decides what to show.  Throwing the value away here would
        take the choice away from it.
        """
        stream, _, callback = _make_level_stream()
        callback(_audio(1000), 320, None, None)
        last = stream.read(timeout=0.05)
        assert stream.read(timeout=0.05) is None
        assert stream._latest_dbm == pytest.approx(last)

    def test_close_still_frees_a_blocked_reader(self):
        """The existing guarantee, unchanged: without it a stuck worker thread hangs
        the whole process at exit, because ThreadPoolExecutor joins every worker it
        ever made.
        """
        stream, _, _ = _make_level_stream()
        freed = []
        reader = threading.Thread(
            target=lambda: freed.append(stream.read(timeout=30.0)), daemon=True)
        reader.start()
        time.sleep(0.05)
        stream.close()
        reader.join(timeout=2.0)
        assert not reader.is_alive(), 'close() left the reader blocked'
        assert freed == [-128.0]

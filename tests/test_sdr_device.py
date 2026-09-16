"""Tests for lib/buzz/sdr_device.py, the hardware contract and its RTL-SDR shim.

Everything here runs against a fake handle, so the whole module is exercised with no
receiver attached.  That is the point of RtlSdrHandle existing at all.

The tests worth reading twice are TestTheSampleFormat, which pins the conversion to
the arithmetic pyrtlsdr has always used, and TestGainWhileStreaming, which turns a
hazard that used to live only in a docstring into something that fails loudly.
"""
import threading
import time

import numpy as np
import pytest
from buzz.sdr_device import (
    RTL_SDR_FORMAT, DeviceProfile, IqBlock, RtlSdrDevice, SampleFormat,
)

WHY = RtlSdrDevice._why_the_receiver_would_not_open
SETTINGS = dict(tuned_hz=3_638_000, gain_db=22.9, iq_sample_rate=256_000)

# A 14-bit converter's format, which is what an SDRplay would present.  Declared in the
# tests rather than in the module because no device here produces it yet, and its job
# is to prove the descriptor is not shaped around one receiver.
INT16_FORMAT = SampleFormat(
    dtype=np.dtype(np.int16), bytes_per_frame=4, half_span=32768.0,
    zero_offset=0j, rail_low=-32768, rail_high=32767,
)


class FakeHandle:
    """Stands in for pyrtlsdr's RtlSdr.  Records what was set, and streams on demand."""

    def __init__(self, gains=None, rate_returns=None, read_raises=False):
        self.sample_rate = 0.0
        self.center_freq = 0
        self.gain = 0.0
        self.valid_gains_db = gains if gains is not None else [0.0, 8.7, 22.9, 40.2]
        self.agc_mode = None
        self.closed = False
        self.cancelled = False
        self.close_blocks = False
        self._rate_returns = rate_returns
        self._read_raises = read_raises
        self._cancel = threading.Event()
        self.async_bytes = None

    def __setattr__(self, name, value):
        if name == 'sample_rate' and getattr(self, '_rate_returns', None) is not None:
            value = self._rate_returns
        object.__setattr__(self, name, value)

    def set_agc_mode(self, enabled):
        self.agc_mode = enabled
        return 0

    def read_bytes(self, num_bytes):
        if self._read_raises:
            raise OSError('libusb says no')
        return np.full(num_bytes, 200, np.uint8)

    def read_bytes_async(self, callback, num_bytes):
        self.async_bytes = num_bytes
        while not self._cancel.is_set():
            callback(np.full(num_bytes, 7, np.uint8))
            time.sleep(0.005)

    def cancel_read_async(self):
        self.cancelled = True
        self._cancel.set()

    def close(self):
        while self.close_blocks:
            time.sleep(0.01)
        self.closed = True


class CountingSink:
    """Takes `accepts` blocks and then has no room, which is what a full queue does."""

    def __init__(self, accepts=1_000_000):
        self.blocks = []
        self._accepts = accepts

    def offer(self, block):
        if len(self.blocks) >= self._accepts:
            return False
        self.blocks.append(block)
        return True


def _device(handle=None, **kwargs):
    settings = dict(tuned_hz=3_638_000, gain_db=22.9, iq_sample_rate=256_000)
    settings.update(kwargs)
    return RtlSdrDevice(handle or FakeHandle(), **settings)


class TestTheSampleFormat:
    """The conversion has to keep producing the samples it always produced."""

    def test_it_reproduces_pyrtlsdrs_own_arithmetic_exactly(self):
        """Bit for bit, not merely close.

        Golden files pin DSP behavior downstream, and the tidier (raw - midpoint) /
        half_span form differs from this one by a unit in the last place.  The
        refactor that introduced SampleFormat is meant to change nothing at all.
        """
        raw = np.concatenate([np.arange(256, dtype=np.uint8),
                              np.random.default_rng(0).integers(0, 256, 4096,
                                                                dtype=np.uint8)])
        pyrtlsdr_does = raw.astype(np.float64).view(np.complex128) / 127.5 - (1 + 1j)
        ours = IqBlock(raw, RTL_SDR_FORMAT, 0.0, 1).as_complex()
        assert np.array_equal(ours, pyrtlsdr_does), (
            'the IQ conversion no longer matches the expression pyrtlsdr uses, so '
            'every golden file downstream now describes a path nobody runs')

    def test_the_rails_reach_exactly_minus_one_and_plus_one(self):
        rails = IqBlock(np.array([0, 0, 255, 255], np.uint8), RTL_SDR_FORMAT,
                        0.0, 1).as_complex()
        assert rails[0] == -1 - 1j
        assert rails[1] == 1 + 1j

    def test_a_sixteen_bit_device_reaches_the_same_rails(self):
        """The descriptor is not shaped around one receiver.

        A 14-bit converter delivers 16-bit signed samples, and the same two fields
        have to carry it without any branch on device type.
        """
        edges = np.array([-32768, -32768, 32767, 32767], np.int16)
        rails = IqBlock(edges, INT16_FORMAT, 0.0, 1).as_complex()
        assert rails[0] == -1 - 1j
        assert rails[1] == pytest.approx(1 + 1j, abs=1e-4)

    def test_samples_counts_frames_rather_than_bytes(self):
        """The units trap that already bit the IQ ring buffer once.

        For an 8-bit device the element count and the byte count are equal, so
        dividing by bytes_per_frame is right by coincidence.  A 16-bit device spends
        four bytes per sample and two values, and counting bytes halves every block.
        """
        six_values = np.zeros(6, np.int16)
        assert IqBlock(six_values, INT16_FORMAT, 0.0, 1).samples == 3
        assert IqBlock(np.zeros(6, np.uint8), RTL_SDR_FORMAT, 0.0, 1).samples == 3

    def test_clipping_counts_each_value_at_a_rail(self):
        raw = np.array([0, 128, 255, 255], np.uint8)
        assert IqBlock(raw, RTL_SDR_FORMAT, 0.0, 1).clipped_samples == 3

    def test_clipping_uses_the_formats_own_rails(self):
        raw = np.array([-32768, 0, 32767, 100], np.int16)
        assert IqBlock(raw, INT16_FORMAT, 0.0, 1).clipped_samples == 2


class TestConfiguring:
    def test_it_sets_rate_tuning_and_gain(self):
        handle = FakeHandle()
        device = _device(handle)
        assert handle.center_freq == 3_638_000
        assert device.iq_sample_rate == 256_000
        assert device.tuned_hz == 3_638_000
        assert handle.gain == 22.9

    def test_it_turns_the_digital_agc_off(self):
        """An AGC riding the impulses would compress what this program measures."""
        handle = FakeHandle()
        _device(handle)
        assert handle.agc_mode is False

    def test_it_snaps_the_gain_to_a_step_the_tuner_offers(self):
        device = _device(gain_db=23.0)
        assert device.gain_db == 22.9

    def test_a_rate_that_comes_back_fractional_is_not_worth_a_warning(self, caplog):
        """Measured on this hardware, 250000 comes back as 250000.000414.

        That rounds to what was asked for, so nothing downstream is misled and the
        operator has nothing to act on.
        """
        handle = FakeHandle(rate_returns=250_000.000414)
        with caplog.at_level('WARNING'):
            device = _device(handle, iq_sample_rate=250_000)
        assert device.iq_sample_rate == 250_000
        assert caplog.text == ''

    def test_a_rate_the_hardware_could_not_reach_says_so_in_ppm(self, caplog):
        """The device derives the rate from a 28.8 MHz divider and misses some."""
        handle = FakeHandle(rate_returns=251_234.5)
        with caplog.at_level('WARNING'):
            device = _device(handle, iq_sample_rate=250_000)
        assert device.iq_sample_rate == 251_234, (
            'downstream has to be told the rate in use, not the rate requested')
        assert 'ppm' in caplog.text

    def test_the_profile_says_what_the_hardware_is(self):
        profile = _device().profile
        assert isinstance(profile, DeviceProfile)
        assert profile.sample_format is RTL_SDR_FORMAT
        assert profile.blocks_to_discard_streaming == 16
        assert profile.blocks_to_discard_reading == 2, (
            'a synchronous read has no transfer pool, so it discards far fewer')
        assert profile.gain_changes_while_streaming is False


class TestGain:
    def test_setting_returns_the_step_actually_used(self):
        device = _device()
        assert device.set_gain_db(41.0) == 40.2
        assert device.gain_db == 40.2

    def test_supported_gains_come_from_the_device(self):
        device = _device(FakeHandle(gains=[1.0, 2.0]))
        assert device.supported_gains_db == [1.0, 2.0]

    def test_nearest_picks_the_closest_step(self):
        assert RtlSdrDevice.nearest_supported_gain(20.0, [0.0, 19.0, 25.0]) == 19.0


class TestGainWhileStreaming:
    """A hazard that used to live only in a docstring.

    Changing an RTL-SDR's gain during an async read is two threads on one device, and
    it left a receiver that never answered again, twice in a few dozen sweeps.  The
    profile says the device refuses it and this makes the refusal real, so the next
    person to wire a gain control to a live source gets an error rather than a wedged
    receiver.  See docs-notebook/rtl-sdr-hardware.md.
    """

    def test_it_refuses_while_a_stream_runs(self):
        device = _device()
        device.start_stream(CountingSink(), 512)
        try:
            with pytest.raises(RuntimeError, match='cannot move while it is streaming'):
                device.set_gain_db(8.7)
        finally:
            device.close()

    def test_it_allows_a_change_once_the_stream_has_stopped(self):
        device = _device()
        device.start_stream(CountingSink(), 512)
        device.stop_stream()
        assert device.set_gain_db(8.7) == 8.7
        device.close()


class TestStreaming:
    def test_blocks_reach_the_sink_the_caller_supplied(self):
        device, sink = _device(), CountingSink()
        device.start_stream(sink, 512)
        deadline = time.monotonic() + 2.0
        while not sink.blocks and time.monotonic() < deadline:
            time.sleep(0.01)
        device.close()
        assert sink.blocks, 'nothing reached the sink within two seconds'
        assert sink.blocks[0].samples == 512

    def test_it_asks_the_driver_for_bytes_rather_than_samples(self):
        handle = FakeHandle()
        device = _device(handle)
        device.start_stream(CountingSink(), 512)
        deadline = time.monotonic() + 2.0
        while handle.async_bytes is None and time.monotonic() < deadline:
            time.sleep(0.01)
        device.close()
        assert handle.async_bytes == 1024, (
            'the read size is in bytes, and 512 samples of 8-bit IQ is 1024 of them'
        )

    def test_a_sink_with_no_room_is_counted_rather_than_raised_at(self):
        """A full queue must not raise inside a callback invoked from C."""
        device, sink = _device(), CountingSink(accepts=2)
        device.start_stream(sink, 512)
        deadline = time.monotonic() + 2.0
        while device.blocks_refused < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        device.close()
        assert len(sink.blocks) == 2
        assert device.blocks_refused >= 1

    def test_a_block_offered_before_any_sink_exists_is_counted(self):
        """_on_block can run once more after stop_stream clears the sink."""
        device = _device()
        device._on_block(np.zeros(4, np.uint8))
        assert device.blocks_refused == 1

    def test_is_streaming_reports_the_thread(self):
        device = _device()
        assert device.is_streaming is False
        device.start_stream(CountingSink(), 512)
        assert device.is_streaming is True
        device.close()
        assert device.is_streaming is False


class TestSynchronousReads:
    def test_it_returns_a_block_of_the_size_asked_for(self):
        assert _device().read_block(256).samples == 256

    def test_it_refuses_a_size_the_driver_cannot_serve(self):
        """rtlsdr_read_sync wants whole 512-byte USB packets and fails quietly."""
        with pytest.raises(ValueError, match='whole 512-byte USB packets'):
            _device().read_block(100)

    def test_zero_is_refused_too(self):
        with pytest.raises(ValueError):
            _device().read_block(0)

    def test_a_read_failure_ends_the_session_rather_than_retrying(self, caplog):
        """pyrtlsdr closes the device itself on any read error."""
        device = _device(FakeHandle(read_raises=True))
        with caplog.at_level('WARNING'):
            assert device.read_block(256) is None
        assert device.read_block(256) is None
        assert caplog.text.count('cannot continue') == 1, (
            'a failed read reported itself more than once')

    def test_blocks_are_numbered_in_the_order_the_device_made_them(self):
        device = _device()
        assert [device.read_block(256).index for _ in range(3)] == [1, 2, 3]


class TestClosing:
    def test_it_releases_the_device(self):
        handle = FakeHandle()
        device = _device(handle)
        assert device.close() is True
        assert handle.closed is True

    def test_it_is_safe_to_call_twice(self):
        device = _device()
        assert device.close() is True
        assert device.close() is True

    def test_it_cancels_a_running_stream_first(self):
        handle = FakeHandle()
        device = _device(handle)
        device.start_stream(CountingSink(), 512)
        assert device.close() is True
        assert handle.cancelled is True

    def test_a_driver_that_never_returns_does_not_hang_the_program(self, caplog):
        """rtlsdr_close blocks inside libusb when transfers were never cancelled.

        A daemon thread costs nothing at exit, and the device stays held until the
        process ends, which is what happened anyway.
        """
        handle = FakeHandle()
        handle.close_blocks = True
        device = _device(handle)
        import buzz.sdr_device as module
        original = module._DEVICE_CLOSE_TIMEOUT_SECONDS
        module._DEVICE_CLOSE_TIMEOUT_SECONDS = 0.05
        try:
            with caplog.at_level('WARNING'):
                assert device.close() is False
        finally:
            module._DEVICE_CLOSE_TIMEOUT_SECONDS = original
            handle.close_blocks = False
        assert 'left to the operating system' in caplog.text

    def test_a_close_failure_is_swallowed_rather_than_raised(self):
        handle = FakeHandle()
        handle.close = lambda: (_ for _ in ()).throw(OSError('gone'))
        assert _device(handle).close() is True

    def test_a_capture_thread_that_will_not_stop_leaves_the_device_open(self, caplog):
        """Closing a handle a thread is still reading through is a crash in C."""
        handle = FakeHandle()
        device = _device(handle)
        device.start_stream(CountingSink(), 512)
        handle.cancel_read_async = lambda: None      # the read never ends
        import buzz.sdr_device as module
        original = module._THREAD_JOIN_TIMEOUT_SECONDS
        module._THREAD_JOIN_TIMEOUT_SECONDS = 0.05
        try:
            with caplog.at_level('WARNING'):
                assert device.close() is False
        finally:
            module._THREAD_JOIN_TIMEOUT_SECONDS = original
            handle._cancel.set()
        assert handle.closed is False, (
            'the device was closed while a thread was still reading through it')
        assert 'did not stop within' in caplog.text

    def test_cancelling_is_allowed_to_fail(self):
        handle = FakeHandle()
        device = _device(handle)
        device.start_stream(CountingSink(), 512)
        real_cancel = handle.cancel_read_async

        def angry():
            real_cancel()
            raise OSError('cancel failed')

        handle.cancel_read_async = angry
        assert device.close() is True


class TestOpening:
    """RtlSdrDevice.open is the contract, and it never hands back a raw handle."""

    def _with_rtlsdr_module(self, monkeypatch, factory):
        import sys
        import types
        module = types.ModuleType('rtlsdr')
        module.RtlSdr = factory
        monkeypatch.setitem(sys.modules, 'rtlsdr', module)

    def test_it_returns_a_configured_device_rather_than_a_handle(self, monkeypatch):
        handle = FakeHandle()
        self._with_rtlsdr_module(monkeypatch, lambda index: handle)
        device = RtlSdrDevice.open(2, **SETTINGS)
        assert isinstance(device, RtlSdrDevice), (
            'open handed back something other than a device, so the driver object '
            'escapes into code that has no business holding it')
        assert device.gain_db == 22.9
        assert handle.agc_mode is False, 'open returned a device it had not configured'

    def test_a_subclass_gets_its_own_type_back(self, monkeypatch):
        """open is an alternative constructor, so cls rather than a hardcoded name.

        A variant of this receiver that overrode the diagnostics or the profile would
        otherwise be handed a plain RtlSdrDevice by its own open.
        """
        class Variant(RtlSdrDevice):
            pass

        self._with_rtlsdr_module(monkeypatch, lambda index: FakeHandle())
        assert type(Variant.open(0, **SETTINGS)) is Variant

    def test_it_opens_the_index_it_was_given(self, monkeypatch):
        seen = []
        self._with_rtlsdr_module(
            monkeypatch, lambda index: seen.append(index) or FakeHandle())
        RtlSdrDevice.open(3, **SETTINGS)
        assert seen == [3]

    def test_a_missing_library_names_what_to_install(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'rtlsdr':
                raise ImportError('no librtlsdr')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)
        with pytest.raises(RuntimeError, match='pyrtlsdr'):
            RtlSdrDevice.open(0, **SETTINGS)

    def test_a_library_that_refuses_to_open_is_reworded(self, monkeypatch):
        def refuse(index):
            raise OSError('busy')

        self._with_rtlsdr_module(monkeypatch, refuse)
        with pytest.raises(RuntimeError, match='Receiver 2'):
            RtlSdrDevice.open(2, **SETTINGS)

    def test_an_unbound_driver_is_told_apart_from_a_busy_device(self):
        """libusb's own wording sends people the wrong way."""
        not_found = OSError('Entity not found')
        not_found.errno = -5
        assert 'Zadig' in WHY(0, not_found)
        assert 'Zadig' not in WHY(0, OSError('busy'))

    def test_a_busy_device_names_both_likely_causes(self):
        message = WHY(3, OSError('busy'))
        assert 'Receiver 3' in message
        assert 'using it' in message and 'plugged in' in message

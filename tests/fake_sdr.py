"""A stand-in receiver, for every test above `buzz.sdr_device`.

There is one fake here rather than one in each test file.  Before the device shim
existed there were two `FakeDevice` classes mirroring pyrtlsdr's own members, so
anything testing the layer above had to know how a driver behaves.  It only has to
know `SdrDevice` now.

Tests drive delivery themselves with `deliver`, rather than waiting for a thread.  A
real device calls its sink from a thread the driver owns, and a test that waits on one
is a test that can hang, so this offers blocks on the calling thread and returns what
the sink said.

`tests/test_sdr_device.py` does not use this.  It tests the shim itself, so it fakes
pyrtlsdr's handle instead, one layer further down.
"""
from time import monotonic

import numpy as np
from buzz.sdr_device import (
    RTL_SDR_FORMAT, VALUES_PER_FRAME, BlockSink, DeviceProfile, IqBlock, SdrDevice,
)

# The 29 steps an RTL-SDR Blog V4 reports, in the order it reports them.  Real values,
# because the snapping tests are about what this hardware actually offers.
V4_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
            22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
            43.9, 44.5, 48.0, 49.6]


class FakeSdrDevice(SdrDevice):
    """An SdrDevice that answers from memory and records what it was told."""

    @classmethod
    def open_from(cls, settings):
        """Build one from a config section, so this satisfies the contract.

        A test builds this directly with the answers it wants, because the point of a
        fake is to skip opening anything.  This exists so that the abstract method is
        implemented and so that a caller holding a config section can still use it.
        """
        return cls(iq_sample_rate=settings.iq_sample_rate,
                   tuned_hz=settings.frequency_hz + settings.tuning_offset_hz,
                   gain_db=settings.gain_db)

    @classmethod
    def supported_gains(cls, settings):
        """The V4's steps, which is what the rest of this fake reports by default."""
        return list(V4_GAINS)

    def __init__(self, *, gains=None, iq_sample_rate=256_000, tuned_hz=3_638_000,
                 gain_db=40.2, fmt=RTL_SDR_FORMAT, blocks_to_discard_streaming=16,
                 blocks_to_discard_reading=2, gain_changes_while_streaming=False,
                 read_returns_none=False):
        self._gains = list(V4_GAINS if gains is None else gains)
        self._iq_sample_rate = iq_sample_rate
        self._tuned_hz = tuned_hz
        self._gain_db = self._nearest(gain_db)
        self._profile = DeviceProfile(
            name='fake',
            sample_format=fmt,
            blocks_to_discard_streaming=blocks_to_discard_streaming,
            blocks_to_discard_reading=blocks_to_discard_reading,
            gain_changes_while_streaming=gain_changes_while_streaming,
        )
        self._read_returns_none = read_returns_none

        self.sink: BlockSink | None = None
        self.gains_written: list[float] = []
        self.started_with: int | None = None
        self.closed = False
        self.stopped = False
        self._refused = 0
        self._produced = 0

    # ------------------------------------------------------------- SdrDevice

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    @property
    def iq_sample_rate(self) -> int:
        return self._iq_sample_rate

    @property
    def tuned_hz(self) -> int:
        return self._tuned_hz

    @property
    def gain_db(self) -> float:
        return self._gain_db

    @property
    def supported_gains_db(self) -> list[float]:
        return list(self._gains)

    @property
    def blocks_refused(self) -> int:
        return self._refused

    @property
    def is_streaming(self) -> bool:
        return self.sink is not None and not self.stopped

    def set_gain_db(self, gain_db: float) -> float:
        if self.is_streaming and not self._profile.gain_changes_while_streaming:
            raise RuntimeError('gain cannot move while it is streaming')
        self._gain_db = self._nearest(gain_db)
        self.gains_written.append(self._gain_db)
        return self._gain_db

    def start_stream(self, sink: BlockSink, block_samples: int) -> None:
        self.sink = sink
        self.started_with = block_samples
        self.stopped = False

    def stop_stream(self) -> bool:
        self.stopped = True
        return True

    def read_block(self, block_samples: int) -> IqBlock | None:
        if self._read_returns_none:
            return None
        return self.block(block_samples)

    def close(self) -> bool:
        self.closed = True
        self.stopped = True
        return True

    # ------------------------------------------------------------- test hooks

    def block(self, samples=64, value=None, arrived_at=None) -> IqBlock:
        """Build one block in this device's own format, without delivering it."""
        fmt = self._profile.sample_format
        if value is None:
            value = 0 if fmt.dtype.kind == 'i' else int(fmt.half_span)
        raw = np.full(samples * VALUES_PER_FRAME, value, dtype=fmt.dtype)
        self._produced += 1
        return IqBlock(raw=raw, fmt=fmt, index=self._produced,
                       arrived_at=monotonic() if arrived_at is None else arrived_at)

    def deliver(self, block=None, **kwargs) -> bool:
        """Offer one block to the sink, the way a driver's callback would.

        Returns what the sink said, and counts a refusal the way a real device does,
        so a test can fill a queue and then check the count.
        """
        if self.sink is None:
            raise AssertionError(
                'deliver() was called before start_stream().  The test is wrong.')
        accepted = self.sink.offer(self.block(**kwargs) if block is None else block)
        if not accepted:
            self._refused += 1
        return accepted

    def _nearest(self, gain_db: float) -> float:
        return min(self._gains, key=lambda candidate: abs(candidate - gain_db))

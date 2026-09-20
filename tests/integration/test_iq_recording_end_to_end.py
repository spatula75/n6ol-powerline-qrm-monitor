"""IQ event recording end to end, from a receiver's own sample format to the .wav.

This tier is deselected by default because it costs real seconds.  Run it with:

    pytest -m integration --no-cov

What this adds over the unit tests is the whole chain at once, over the threads it
really runs on: a device delivering blocks, `SdrSource` queueing them, `SdrPipeline`
converting on its feeder thread while `IqRingBuffer` keeps the raw samples, a real
`ContinuousAnalyzer` locking onto the recovered audio, and a real `RecordingTrigger`
polling on a thread of its own until it writes two files.

**Every one of those carries the width of a sample, and none of them owns it.**
The device states it in its `DeviceProfile`, the ring buffer has to size itself from
that, and the recorder reads the buffer to size the `.wav` header.  Each of the three
is correct on its own reading, and the file is wrong if any pair disagrees.  That is
the failure this tier is for: the buffer took unsigned bytes whatever the receiver
was, so every SDRplay capture held the low byte of a 16-bit sample under a header
saying it was a whole one, and no unit test of any single component saw it.

The device is the only stand-in, which is the hardware boundary and the one thing a
machine with no receiver cannot supply.  Its sample format is the production constant
rather than a value invented here, so a change to either `RTL_SDR_FORMAT` or
`SDRPLAY_FORMAT` has to come past these tests.
"""

import time
import wave
from pathlib import Path

import numpy as np
import pytest
from tests.receiver.fake_sdr import FakeSdrDevice
from harness import StateLog, config_for

from buzz.analyzer import AnalyzerState, ContinuousAnalyzer
from buzz.receiver.iq import IqToAudio
from buzz.recorder import build_recording
from buzz.receiver.source import SdrPipeline, SdrSource
from buzz.receiver.device import RTL_SDR_FORMAT, IqBlock, SampleFormat
from buzz.receiver.sdrplay import SDRPLAY_FORMAT

IQ_RATE = 256_000
DECIMATION = 16
BANDWIDTH_HZ = 4_000
TUNING_OFFSET_HZ = 50_000
AUDIO_RATE = IQ_RATE // DECIMATION
PULSE_RATE = 120

# 64 ms of IQ, which is what the monitor asks a receiver for at this rate.
BLOCK_SAMPLES = 16_384

# A gap fires for as long as the line voltage is over its breakdown threshold, so the
# pulse is a burst of milliseconds rather than an impulse.  2.5 ms is the short end of
# the range this program is written for, and the shape is the symmetric football that
# such a discharge produces.  An impulse would be the wrong signal to test with, for a
# concrete reason: three samples at the audio rate is broadband, the 4 kHz filter in
# the IQ chain throws nearly all of it away, and the recovered audio is then too faint
# to lock onto.
BURST_SECONDS = 0.0025

# Burst envelope and background, as fractions of the converter's full scale.  Both sit
# well inside the rails of either format, so neither receiver clips and the two are
# measured on the same signal.  The ratio is about 43 dB, which is an ordinary arc
# rather than a marginal one, because this tier is about the plumbing and not the DSP.
BURST_AMPLITUDE = 0.15
BACKGROUND_AMPLITUDE = 0.001

# How long to keep feeding before giving up on a lock.  The analyzer wants most of the
# ring buffer before it will search, so this is several times what a lock costs in
# practice and still bounds a broken run.
LOCK_TIMEOUT_SECONDS = 25.0

# How much more to feed once the recording has started, so the file holds an arc
# rather than the lead-in alone.
RECORD_SECONDS = 2.0

# Every receiver this program supports, with the sample width its recordings must
# have.  The byte counts are stated here rather than read from the format, because a
# test taking both sides from one constant would pass whatever that constant said.
RECEIVERS = [
    pytest.param(('rtlsdr', RTL_SDR_FORMAT, 1), id='8-bit-RTL-SDR'),
    pytest.param(('sdrplay', SDRPLAY_FORMAT, 2), id='16-bit-SDRplay'),
]


def quantize(values: np.ndarray, fmt: SampleFormat) -> np.ndarray:
    """One axis of the wave, as the converter would deliver it.

    This inverts `IqBlock.as_complex`, so an unsigned format comes back centered on
    its own midpoint and a signed one on zero.
    """
    counts = (values + fmt.zero_offset.real) * fmt.half_span
    return np.clip(np.round(counts), fmt.rail_low, fmt.rail_high)


def iq_block(index: int, fmt: SampleFormat,
             rng: np.random.Generator) -> IqBlock:
    """One block of raw IQ, as a receiver hearing a 120 pps arc would deliver it.

    This builds the envelope at the IQ rate and puts it on a carrier at
    `TUNING_OFFSET_HZ`, because that is where the monitor tunes and where `IqToAudio`
    mixes back down from.  A signal built at DC instead would sit on the tuner artifact the offset
    exists to avoid, and the one-sided filter would take half of it.

    `index` places the block in the stream, so the pulse grid and the carrier both run
    continuously across block boundaries.  Restarting either per block would put a
    step in at the block rate, and a step at a steady rate is exactly what this
    program detects and believes.
    """
    start = index * BLOCK_SAMPLES
    envelope = rng.normal(0.0, BACKGROUND_AMPLITUDE, BLOCK_SAMPLES)
    burst = int(BURST_SECONDS * IQ_RATE)
    shape = np.sin(np.linspace(0, np.pi, burst))
    spacing = IQ_RATE / PULSE_RATE
    for pulse in range(int(start / spacing), int((start + BLOCK_SAMPLES) / spacing) + 1):
        at = round(pulse * spacing) - start
        if 0 <= at and at + burst <= BLOCK_SAMPLES:
            envelope[at:at + burst] += (BURST_AMPLITUDE * shape
                                        * rng.normal(0.0, 1.0, burst))
    phase = 2j * np.pi * TUNING_OFFSET_HZ * (start + np.arange(BLOCK_SAMPLES)) / IQ_RATE
    carried = envelope * np.exp(phase)
    raw = np.empty(BLOCK_SAMPLES * 2, dtype=fmt.dtype)
    raw[0::2] = quantize(carried.real, fmt)
    raw[1::2] = quantize(carried.imag, fmt)
    return IqBlock(raw=raw, fmt=fmt, arrived_at=time.monotonic(), index=index)


def excursion(samples: np.ndarray, fmt: SampleFormat) -> float:
    """How far the samples swing from the format's own zero, in counts."""
    centered = samples.astype(np.float64) - fmt.zero_offset.real * fmt.half_span
    return float(np.abs(centered).max())


class Receiver:
    """A real capture chain over a stand-in device, recording into `directory`.

    Everything but the device is the production object on its production thread.  The
    test hands blocks to the device at the speed a receiver would, because the trigger
    and the analyzer both work in wall-clock seconds.
    """

    def __init__(self, directory: Path, source_name: str,
                 fmt: SampleFormat) -> None:
        self.directory = directory
        self.delivered: list[np.ndarray] = []
        self.device = FakeSdrDevice(fmt=fmt, iq_sample_rate=IQ_RATE)
        self.source = SdrSource(self.device, block_samples=BLOCK_SAMPLES)
        self.pipeline = SdrPipeline(
            self.source,
            IqToAudio(IQ_RATE, DECIMATION, BANDWIDTH_HZ, TUNING_OFFSET_HZ),
            keep_iq=True)
        config = config_for(directory, record_iq=True, max_seconds=30.0)
        # An IQ recording reads the section of the receiver in use, and refuses
        # outright for a sound card.  See IqEventRecorder.
        config.audio.source = source_name
        config.audio.sample_rate = AUDIO_RATE
        getattr(config, source_name).iq_sample_rate = IQ_RATE
        self.analyzer = ContinuousAnalyzer(self.pipeline, config)
        self.log = StateLog()
        self.analyzer.add_state_listener(self.log)
        self.recorder = build_recording(self.pipeline, self.analyzer, config)
        self._fmt = fmt
        self._index = 0
        self._rng = np.random.default_rng(7)

    def __enter__(self) -> 'Receiver':
        self.pipeline.start()
        self.analyzer.start()
        self.recorder.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.recorder.stop()
        self.analyzer.stop()
        self.pipeline.close()

    def feed(self, seconds: float) -> None:
        """Deliver blocks at the rate the hardware would, for `seconds` of signal."""
        block_seconds = BLOCK_SAMPLES / IQ_RATE
        for _ in range(int(seconds / block_seconds)):
            block = iq_block(self._index, self._fmt, self._rng)
            self.delivered.append(block.raw)
            self.device.deliver(block)
            self._index += 1
            time.sleep(block_seconds)

    def feed_until_locked(self) -> bool:
        """Feed until the analyzer publishes a lock, and say whether it did."""
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            self.feed(0.5)
            if AnalyzerState.LOCKED in self.log.states:
                return True
        return False

    def iq_file(self) -> Path:
        return [f for f in sorted(self.directory.glob('*.wav'))
                if f.name.endswith('-iq.wav')][0]

    def written(self) -> np.ndarray:
        """Every sample in the IQ recording, read at the width the file declares.

        The width comes from the header rather than from the format the device sent,
        because a reader with their own tools has nothing else to go on.  Reading at
        the sent width instead would paper over the whole failure: a truncated file
        read back as 16-bit pairs adjacent bytes into plausible-looking samples.
        """
        with wave.open(str(self.iq_file()), 'rb') as wav:
            dtype = np.uint8 if wav.getsampwidth() == 1 else np.int16
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype=dtype)


@pytest.mark.integration
class TestTheRecordingKeepsTheReceiversOwnSampleWidth:
    """An IQ capture is the one output of this program that nothing downstream can
    repair, because it exists to be handed to somebody with their own tools.

    A width the header gets wrong is not a degraded recording.  It is a file that
    opens cleanly, plays as noise, and says nothing about having lost half its bits.
    """

    @pytest.fixture(scope='class', params=RECEIVERS)
    @staticmethod
    def captured(request, tmp_path_factory):
        source_name, fmt, width_bytes = request.param
        directory = tmp_path_factory.mktemp(f'iq-{source_name}')
        with Receiver(directory, source_name, fmt) as receiver:
            assert receiver.feed_until_locked(), (
                f'the analyzer never locked onto the {source_name} signal within '
                f'{LOCK_TIMEOUT_SECONDS:.0f} seconds, so no recording ever started '
                f'and nothing below is about sample widths.  States seen: '
                f'{[s.name for s in receiver.log.states]}')
            receiver.feed(RECORD_SECONDS)
        return receiver, fmt, width_bytes

    def test_the_event_wrote_both_files(self, captured):
        """The audio and the IQ come from one lock and one budget, so a tier finding
        only one file would be measuring a half-built event in everything below.
        """
        receiver = captured[0]
        names = sorted(f.name for f in receiver.directory.glob('*.wav'))
        assert len(names) == 2, names

    def test_the_sample_width_is_the_receivers_own(self, captured):
        """The width travels device to DeviceProfile to IqRingBuffer to .wav header,
        and each stage reads it from the one before.  This is the assertion an 8-bit
        buffer failed on a 16-bit receiver.
        """
        receiver, _, width_bytes = captured
        with wave.open(str(receiver.iq_file()), 'rb') as wav:
            assert wav.getsampwidth() == width_bytes

    def test_it_is_stereo_at_the_receivers_own_rate(self, captured):
        """I on the left and Q on the right, at the IQ rate rather than the audio one.
        A file tagged with the audio rate would describe a span of time 16 times too
        long, and every frequency read off it would be wrong by that factor.
        """
        receiver = captured[0]
        with wave.open(str(receiver.iq_file()), 'rb') as wav:
            assert wav.getnchannels() == 2
            assert wav.getframerate() == IQ_RATE

    def test_the_samples_keep_the_excursion_the_receiver_sent(self, captured):
        """A right header over wrong numbers is still a wrong file, so the width is
        not the whole assertion.

        The arc repeats 120 times a second through a file of several seconds, so the
        loudest sample in the file is the loudest the receiver produced, to well
        inside this tolerance.  Truncation cannot pass it: dropping the high byte of a
        16-bit sample moves the excursion by orders of magnitude.
        """
        receiver, fmt, _ = captured
        written = receiver.written()
        assert written.size, 'the recording is empty'
        sent = np.concatenate(receiver.delivered)
        assert excursion(written, fmt) == pytest.approx(excursion(sent, fmt), rel=0.2)

    def test_a_sixteen_bit_capture_holds_what_no_byte_could(self, captured):
        """The corruption itself, named, and asserted on the samples rather than on
        the header the test above covers.

        Every value a byte can take falls in 0 through 255, so a capture holding
        anything outside that range cannot have been through the 8-bit buffer,
        whatever its header says.  A real one leaves the range constantly, because a
        signed converter centers on zero and half of every arc is negative.
        """
        receiver, _, width_bytes = captured
        if width_bytes == 1:
            pytest.skip('an 8-bit receiver has nothing to lose to a byte')
        written = receiver.written()
        assert written.min() < 0 or written.max() > 255

"""Raw IQ from a receiver, turned into the audio and the measurements above it.

The hardware itself is `buzz.sdr_device`, which owns every operation performed against
a device.  This module is what sits between that and the rest of the program:
`RtlSdrSource` queues the blocks a streaming device delivers, `SweepReader` reads one
at a time for a gain sweep, `RtlSdrPipeline` converts and fills the shared ring buffer,
`IqRingBuffer` keeps the raw bytes when an IQ recording wants a lead-in, and
`SdrLevelStream` answers the setup program's meters.

Nothing here imports a driver library.  That is the point of the split: the arithmetic
and the buffering are exercised exhaustively with no receiver attached, and a second
kind of receiver arrives as another `SdrDevice` without touching any of this.

Why the draining thread is the one with a deadline
--------------------------------------------------
A device copies each block on the driver's own thread and does nothing else there, for
the reasons `buzz.sdr_device` gives.  The work falls to whichever thread drains it,
which is `RtlSdrPipeline`'s feeder, and that thread has to average less than a block's
own duration.  That is 64 ms at the default settings against roughly 2 ms of work.

Run longer than it and librtlsdr's pool of USB transfers drains.  The pool is what
meets the receiver's real deadline: its FIFO holds 1880 bytes, which is 940 complex
samples, so at 256 kHz it overflows 3.67 ms after collection stops.  That is far
shorter than a Windows scheduler quantum, and nothing anywhere reports the loss,
because it happens in hardware upstream of every piece of software.  A sound card can
report an overflow because the driver owns the buffer that overflowed.  Here nobody
owns it.  Measured on this hardware the pool holds about 960 ms at our block size, so
the 3.67 ms deadline is met by the USB stack rather than by Python.

The pool absorbs bursts rather than sustained slowness.  Measurement showed that a
callback stalled 60 ms against a 64 ms block stayed clean, while one stalled 200 ms
lost 67% of the stream and kept losing it.  So the feeder loop does the least it can,
which is to convert, count and append.

What cannot be counted, and what can
------------------------------------
A loss inside the receiver cannot be counted at all, which is why
`RtlSdrSource.clock_drift_seconds` exists: elapsed time minus the audio that arrived
for it is the only evidence available.  Read it as a symptom rather than a
measurement, since the receiver's crystal and the system clock separate slowly even
when nothing is wrong.

Blocks this program refused because its own queue was full are a different thing and
stay apart on purpose.  The device counts those, because a refusal happens on the
driver's thread where logging can raise and can block on I/O, and `RtlSdrSource`
reports them from the thread that fell behind.
"""

import logging
import queue
import threading
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np

from buzz.sampler import LevelStream, RingBufferPipeline
from buzz.sdr_device import VALUES_PER_FRAME, IqBlock, OverloadStatus, SdrDevice

if TYPE_CHECKING:
    from buzz.iq import IqToAudio

logger = logging.getLogger(__name__)

# Samples per callback, which is the deadline the draining thread has to beat.
#
# 16384 samples is 64 ms at 256 kHz, against roughly 2 ms of conversion work, so the
# margin is about thirty to one.  Smaller blocks tighten the deadline and buy
# nothing.  Larger ones delay the first samples and coarsen the drop check, which
# compares arrival times against the sample count.
DEFAULT_BLOCK_SAMPLES = 16_384

# How many blocks may wait for the draining thread before the oldest is refused.
#
# The queue is deliberately small, because a deep one would hide a slow consumer for
# a while and then fail anyway, where a shallow one reports the problem at once.
#
# It also keeps the two kinds of loss apart.  A refusal here is our own thread
# running late, which is countable and fixable.  A loss inside the receiver is
# neither.
DEFAULT_BUFFER_BLOCKS = 8

# How far the discard count must move before the warning is repeated, in blocks.  The
# first is always reported and then every hundred after it, so a sustained problem
# stays visible without costing a log line per block.
#
# A distance rather than a multiple, because this counter is not read at every value.
# The device increments it on its own thread and a consumer that has fallen behind
# refuses several blocks between two reads, so the exact multiple is usually stepped
# over: refusals arriving three at a time run 3, 6, 9 and up to 99, 102, and a test
# for `% 100 == 0` then never fires again after the first report.
_DISCARD_LOG_EVERY = 100

# How long to wait for a thread to finish during shutdown, in seconds.
#
# The two threads stop by different means.  The feeder notices its stop flag when its
# next read times out, which is within 0.5 s.  The capture thread returns from
# read_bytes_async once cancel_read_async has taken effect.  Five seconds is several
# times either, so a thread still running afterwards is stuck rather than slow, and
# RtlSdrSource.close treats it that way.
_THREAD_JOIN_TIMEOUT_SECONDS = 5.0

# How long a draining thread waits for a block before it rechecks its stop flag.
# It sets how quickly close() returns on a receiver that has gone quiet, so it is
# chosen short against the join above rather than against the block rate.
_FEED_READ_TIMEOUT_SECONDS = 0.5

# The share of raw values that has to clip before it means anything.
#
# A fraction rather than a count, so it means the same thing at any sample rate and
# over any interval.  That is what lets one figure serve two readers: the monitor
# measures it over a minute of streaming, and GainMeasurement over a second or so at
# one gain step.  4 parts per million is 123 values a minute at 256 kHz, which is the
# figure the one station running this settled on as the boundary between noise and
# news: it saw over seven hundred a minute when its gain really was a step too high,
# and single digits once it was not.
#
# What that costs if it is wrong is small and known.  123 clipped values is at most
# 6% of a single 4 ms burst, so the worst that slips through unreported is a fraction
# of a decibel on one event.  What the old behavior of treating any clipping at all as
# evidence cost was larger: it advised lowering the gain a step for 0.46 parts per
# million, and a step below the knee is one to three decibels on every noise floor
# from then on.
CLIPPING_WORTH_NOTICING = 4e-6

# How often the pipeline looks at its own health counters, in seconds.
#
# A minute matches the collector's own cadence, so a warning reaches the log beside
# the CSV row it spoiled.  Looking more often would find nothing sooner, because the
# counters only move when a block arrives.
_HEALTH_INTERVAL_SECONDS = 60.0

# How far the receiver clock may run from the system clock, in parts per million,
# before the difference means lost samples rather than two crystals disagreeing.
#
# Only a positive movement is checked against it, because only a positive movement can
# be a loss: the interval then holds more time than audio.  A negative one is a buffer
# in the receiver library emptying, which loses nothing.  See _warn_about_drift.
#
# The figure is chosen rather than measured.  RTL-SDR crystals are specified in the
# tens of parts per million, so 500 leaves room for a poor one and still catches a
# loss, which runs to thousands.  See RtlSdrSource.clock_drift_seconds.
#
# Raising this figure is almost never the answer to a wide spread, and twice on one
# evening it looked like it was.  An RSP1B ran to 12.2 ms of standard deviation, which
# turned out to be a garbage collection the plotter forced twice a minute, and it
# settled to 4.4 ms once that was narrowed.  What remained was a buffer in the receiver
# library filling and emptying, which crosses this limit about every eight minutes and
# is not a fault at all.  See _warn_about_drift and
# docs-notebook/receiver-clock-drift.md.
_DRIFT_PPM_LIMIT = 500

# How far the receiver clock may stand from its baseline before that is a fault rather
# than a buffer cycling, in seconds.
#
# The figure is chosen, from a measurement.  On an RSP1B at 3530 kHz the library's
# buffer filled for eight or nine minutes to between +48 and +64 ms and then emptied in
# one interval, three times across two runs, always returning to within 20 ms of zero.
# 300 ms is about five times the largest excursion seen, so a cycle cannot reach it
# while a rate error or a steady loss will.  See
# docs-notebook/receiver-clock-drift.md.
_CUMULATIVE_DRIFT_LIMIT_SECONDS = 0.300


def _what_a_loss_means() -> str:
    """What one interval short of audio is, and what to do about it.

    Split out so the wording can be read and tested without a receiver.
    """
    return ('Less audio arrived than that interval holds, so samples were lost.  '
            'Levels and grid frequency from this period are suspect.  Check what else '
            'on this machine is taking the CPU.')


def _what_a_sustained_drift_means(total: float) -> str:
    """What a clock that has walked away from where it started is, by sign.

    The sign says which fault it is.  A positive total means audio keeps going missing,
    and a negative one means the receiver produces more audio than the configured rate
    accounts for.
    """
    if total > 0:
        return ('Less audio has arrived than the run accounts for, so samples are '
                'going missing steadily rather than once.  Check what else on this '
                'machine is taking the CPU.')
    return ('More audio has arrived than the run accounts for, so the receiver runs '
            'faster than the rate it was configured at.  Check that rate against what '
            'the receiver reports.')


class RtlSdrSource:
    """Pulls raw IQ off a receiver and queues it for somebody else to convert.

    Call start(), then read() until it returns None, then close().  The caller injects
    the device rather than this opening one, so a test can pass a stand-in.  See
    `RtlSdrDevice.open` for the real one.

    This owns the queue and hands it to the device as a sink.  The depth belongs here
    rather than in the device, because it is set by how much history the ring buffer
    needs, which is a fact about the monitor rather than about hardware.  The device
    fills it from the driver's own thread and counts what it could not place.
    """

    class _Sink:
        """The queue, in the shape a device fills.

        `offer` never raises, because a device calls it from a thread the driver owns,
        where an exception has nowhere to go.  A full queue is a refusal rather than a
        failure, and the device counts refusals for this class to report.
        """

        def __init__(self, blocks: queue.Queue) -> None:
            self._blocks = blocks

        def offer(self, block: IqBlock) -> bool:
            try:
                self._blocks.put_nowait(block)
            except queue.Full:
                return False
            return True

    def __init__(self, device: SdrDevice, *,
                 block_samples: int = DEFAULT_BLOCK_SAMPLES,
                 buffer_blocks: int = DEFAULT_BUFFER_BLOCKS) -> None:
        self._device = device
        self._block_samples = block_samples
        self._blocks: queue.Queue[IqBlock] = queue.Queue(maxsize=buffer_blocks)
        self._sink = self._Sink(self._blocks)

        self._reported_discards = 0
        self._samples_delivered = 0
        self._first_arrival: float | None = None
        self._last_arrival: float | None = None
        self._closed = False

    # ------------------------------------------------------------------ public

    @property
    def block_samples(self) -> int:
        """Complex samples per callback, which is how long one block lasts.

        Public because the block duration is the deadline every consumer of this class
        works against, and because it sets the time constant of anything smoothing
        across blocks.
        """
        return self._block_samples

    @property
    def iq_sample_rate(self) -> int:
        """The rate the device is actually running at, rounded to whole samples."""
        return self._device.iq_sample_rate

    @property
    def supported_gains_db(self) -> list[float]:
        """Every gain this device offers, in the order it reports them."""
        return self._device.supported_gains_db

    @property
    def gain_db(self) -> float:
        """The gain in use."""
        return self._device.gain_db

    @property
    def tuned_hz(self) -> int:
        """Where the device is tuned, which is not the frequency of interest.

        A receiver puts a strong false signal at exactly its tuning frequency, from the
        tuner leaking into its own mixer.  `buzz.iq` tunes to one side and mixes back,
        so that false signal falls outside the measured band.
        """
        return self._device.tuned_hz

    @property
    def blocks_discarded(self) -> int:
        """Blocks refused because the draining thread had not kept up.

        This counts something different from anything the receiver lost, and the two
        stay apart on purpose.  A count here means our own consumer is too slow, which
        is a bug with a fix.  A loss inside the receiver cannot be counted at all.
        """
        return self._device.blocks_refused

    @property
    def clock_drift_seconds(self) -> float:
        """Elapsed time minus the audio the device delivered for it.

        The only available evidence that a driver dropped samples, since nothing
        reports that.  A drop on this side of the callback is counted instead, by
        _emit and _report_any_discards.

        The two signs are not symmetric, because audio cannot be created.  So a
        negative figure can only be audio that already existed arriving late, and is
        never a loss.  A positive one is ambiguous: audio is either missing or being
        held, and one reading cannot say which.

        | Reading                             | What it is                    |
        |-------------------------------------|-------------------------------|
        | Negative                            | A buffer draining.            |
        | Positive, small, and recovering     | A buffer filling.             |
        | Positive, and the total keeps going | A loss nothing else reports.  |

        Only the third is a fault, which is why _warn_about_drift watches the total
        since its baseline rather than one interval.  Measured on an RSP1B, the
        receiver library fills a buffer for eight or nine minutes at about 130 ppm and
        then empties it in one interval, so the first two rows both happen every eight
        minutes on a receiver with nothing wrong with it.  See
        docs-notebook/receiver-clock-drift.md.

        Read it as a symptom rather than a measurement.  The receiver's crystal and the
        system clock differ by some parts per million that nobody here has measured, so
        the two separate slowly even when nothing is wrong.
        """
        if self._first_arrival is None or self._last_arrival is None:
            return 0.0
        elapsed = self._last_arrival - self._first_arrival
        return elapsed - self._samples_delivered / self.iq_sample_rate

    def start(self) -> None:
        """Begin capture.  The device runs it on a thread of its own."""
        self._device.start_stream(self._sink, self._block_samples)

    def read(self, timeout: float = 1.0) -> IqBlock | None:
        """Take the next block, or None if none arrived within `timeout`.

        None means the device has gone quiet rather than that capture has finished.  A
        caller looping on this should check whether it is still meant to be running
        rather than treat None as the end.
        """
        try:
            block = self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None
        self._account_for(block)
        self._report_any_discards()
        return block

    def close(self) -> bool:
        """Stop capture and release the device.

        This is safe to call twice, and it will be called twice.  Shutdown calls it
        explicitly, and the device's own atexit hook fires afterwards regardless.

        Returns whether the device was actually released.  False means it is still
        held, and the consequence falls on whatever opens a receiver next: libusb
        refuses with LIBUSB_ERROR_ACCESS, which reads as a permissions problem and is
        not one.
        """
        if self._closed:
            return False
        self._closed = True
        return self._device.close()

    # ----------------------------------------------------------------- private

    def _account_for(self, block: IqBlock) -> None:
        """Track arrivals so clock_drift_seconds has something to compare.

        The first block establishes the origin and contributes no samples.  Its audio
        was collected before that instant, so counting it against an interval starting
        there would show a healthy stream permanently one block in deficit.
        """
        if self._first_arrival is None:
            self._first_arrival = block.arrived_at
        else:
            self._samples_delivered += block.samples
        self._last_arrival = block.arrived_at

    def _report_any_discards(self) -> None:
        """Say that blocks were refused, without saying it on every block.

        Reported here rather than by the device, because the device counts refusals on
        the driver's own callback thread, where logging can raise and can block on I/O.
        This runs on the consumer's thread, which is also the thread that fell behind.

        Rate-limited because it will not happen once.  A line per block would flood the
        log while stealing time from a thread that is already behind.
        """
        discarded = self._device.blocks_refused
        if not discarded:
            return
        first = self._reported_discards == 0
        if first or discarded - self._reported_discards >= _DISCARD_LOG_EVERY:
            self._reported_discards = discarded
            logger.warning(
                'Discarded %d block(s) of receiver samples because the conversion '
                'thread fell behind.  The audio now has gaps in it, so levels and '
                'grid frequency from this period are both suspect.  Something else on '
                'this machine is probably taking the CPU.', discarded)

# Samples per synchronous read during a gain sweep.
#
# 2048 samples is 4096 bytes, eight USB packets, and 8 ms at 256 kHz.  Small enough
# that the discard after a gain change costs little, large enough that the per-read
# overhead is not what the sweep spends its time on.  The device refuses a size it
# cannot serve exactly, so this only has to be a sensible default.
DEFAULT_SWEEP_BLOCK_SAMPLES = 2048


class SweepReader:
    """Reads IQ one block at a time, on the calling thread, for a gain sweep.

    The monitor cannot miss a sample, so it streams and the device hands blocks to a
    sink.  A gain sweep is the opposite: it throws away most of what it reads, measures
    a statistical property of noise, and has no deadline at all.  So it reads
    synchronously on one thread, and the difference is not an optimization but the
    removal of a defect.

    Changing gain during a stream is two threads touching one device, and it left a
    receiver that never answered again, twice in a few dozen sweeps.  Here there is no
    second thread, so there is nothing to race.  The device refuses the unsafe order
    outright, and `docs-notebook/rtl-sdr-hardware.md` records what else was tried.

    What it gives up is continuity, since samples between one read and the next are
    simply missed.  That costs a gain sweep nothing and would ruin the monitor.
    """

    def __init__(self, device: SdrDevice, *,
                 block_samples: int = DEFAULT_SWEEP_BLOCK_SAMPLES) -> None:
        # Asked here as well as at each read, so that a size the device cannot serve is
        # refused while the caller is still building the reader.  If read_block were
        # the only check, this would construct, the dialog would report how long the
        # sweep will take, and the first measurement would raise out of GainSweep.run
        # into a Textual worker.
        device.validate_sync_block(block_samples)
        self._device = device
        self._block_samples = block_samples
        self._closed = False

    @property
    def supported_gains_db(self) -> list[float]:
        """Every gain this device offers, in the order it reports them."""
        return self._device.supported_gains_db

    @property
    def iq_sample_rate(self) -> int:
        """The rate the device settled on, rounded to whole samples."""
        return self._device.iq_sample_rate

    @property
    def gain_db(self) -> float:
        """The gain last written, which is the only figure a V4 will ever admit to."""
        return self._device.gain_db

    @property
    def overload_status(self) -> OverloadStatus | None:
        """Hardware overload reports, or None when the receiver provides none."""
        return self._device.overload_status

    @property
    def reported_gain_db(self) -> float | None:
        """What the receiver says its gain is, or None where it says nothing.

        Distinct from `gain_db`, which falls back to the figure that was written when
        the hardware has none to offer.  Nothing in the sweep reads this: it is here so
        that `tools/sdr_gain_probe` can show the fallback and the hardware side by side,
        which is how a gain that never reaches the receiver is told from one that does.
        """
        return getattr(self._device, 'reported_gain_db', None)

    @property
    def floor_margin_db(self) -> float:
        """How far above the knee this receiver puts the floor bound, in dB."""
        return self._device.floor_margin_db()

    @property
    def blocks_to_discard_after_gain_change(self) -> int:
        """Blocks to read and throw away after moving the gain.

        This takes the reading figure rather than the streaming one, because a
        synchronous read has no transfer pool behind it.  See DeviceProfile.
        """
        return self._device.profile.blocks_to_discard_reading

    def set_gain(self, gain_db: float) -> float:
        """Move the gain, and return the value actually set.

        Nothing else is touching the device, so this is an ordinary call.  That is the
        whole point of reading synchronously.
        """
        return self._device.set_gain_db(gain_db)

    def drain(self) -> int:
        """Nothing is buffered here, so there is nothing to throw away.

        The streaming source has a queue between the device and its reader, and stale
        entries in it have to go before a measurement.  A synchronous read has no queue
        at all, which is one of the things this design removes rather than manages.
        """
        return 0

    def read(self, timeout: float = 1.0) -> IqBlock | None:
        """One block, or None when the device has stopped answering.

        `timeout` is accepted and ignored, because a synchronous read has no timeout to
        give it.  The signature matches the streaming source so that a gain sweep does
        not have to know which one it is holding.
        """
        return self._device.read_block(self._block_samples)

    def close(self) -> bool:
        """Release the device, bounded the way every other close here is."""
        if self._closed:
            return False
        self._closed = True
        return self._device.close()


class IqRingBuffer(RingBufferPipeline):
    """The last several seconds of raw IQ, kept so an IQ recording has a lead-in.

    The audio ring buffer gives an event recording its run-up for free, because the
    audio is already sitting there when the lock happens.  Raw IQ has no such buffer:
    RtlSdrPipeline converts each block and keeps only the audio, and the bytes go out
    of scope immediately after.  This holds them for the same duration instead.

    It stores the bytes the device delivered rather than the complex samples they
    convert to.  That is eight times smaller, and it is also exactly what a recording
    writes, since the format on disk is the device's own: unsigned bytes, I then Q.
    Converting to complex and back would cost the work twice and gain nothing.

    This appends one whole device block at a time rather than in CHUNK_SIZE pieces.
    That slicing exists so get_snapshot returns a full window to the analyzer, and
    nothing reads this by chunk count - a recording reads it sequentially with
    read_from.

    The pipeline builds this only when [recording] record_iq is on, because it is not
    small: 4.7 MB at the default 256 kHz, and 44 MB at the 2.4 MHz the hardware will
    accept.
    """

    def __init__(self, iq_sample_rate: int, block_samples: int) -> None:
        super().__init__(sample_rate=iq_sample_rate, chunk_size=block_samples,
                         dtype=np.uint8)

    def add(self, block: IqBlock) -> None:
        """Keep one block's raw bytes, shaped one complex sample per row.

        Public where every other pipeline here fills itself from inside a subclass,
        because this one is filled by RtlSdrPipeline, which is a buffer in its own
        right for the audio.  The push crosses an object boundary, so it gets a name.

        The reshape is what keeps the buffer's arithmetic honest.  `raw` is interleaved
        bytes, so its length counts two per complex sample, while the capacity this
        buffer was sized to counts one - appending it flat would leave total_samples
        and capacity_samples in different units, and every duration derived from them
        wrong by a factor of two.  A row per complex sample also happens to be the
        frame layout a stereo recording writes, I then Q.
        """
        self._append(block.raw.reshape(-1, 2))


class RtlSdrPipeline(RingBufferPipeline):
    """Feeds the shared ring buffer from a receiver, converting on the way.

    The last of four pieces.  SdrDevice holds the hardware, RtlSdrSource holds the
    queue between the driver's thread and this one, IqToAudio holds the arithmetic,
    and this owns the thread that carries blocks from the queue to the buffer.
    Everything downstream reads this exactly as it reads the sound card, and cannot
    tell which it has.

    This thread is the one with a deadline, because it must average less than a
    block's own duration.  That is 64 ms at the default settings against roughly 2 ms
    of work.  Run longer than it and the pool inside librtlsdr drains, and samples are
    lost where nothing can count them.  So the loop does the least it can, which is to
    convert, count and append.  See the module docstring.
    """

    def __init__(self, source: RtlSdrSource, converter: 'IqToAudio', *,
                 clock: Callable[[], float] = monotonic, keep_iq: bool = False) -> None:
        super().__init__(converter.audio_sample_rate)
        self._source = source
        self._converter = converter
        # Off unless an IQ recording is going to want it.  See IqRingBuffer for what
        # it costs, which is why nothing pays for it by default.
        self._iq_buffer = (IqRingBuffer(source.iq_sample_rate, source.block_samples)
                           if keep_iq else None)
        self._clipped = 0
        self._leftover = np.empty(0, dtype=np.int16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, daemon=True, name='sdr-feeder')
        # The clock is injected so a test can drive a minute of monitoring in no time
        # at all.  Waiting for the real one would put _HEALTH_INTERVAL_SECONDS into
        # the suite for every case.
        self._clock = clock
        self._health_checked_at = clock()
        self._clipped_reported = 0
        self._saturated_reported = 0
        # None until the first report, which takes the baseline rather than assuming
        # the stream started at zero drift.  See _warn_about_drift.
        self._drift_reported: float | None = None
        # Where the clock stood when the baseline was taken, and whether it has been
        # reported as having walked away from it.  The second stops one fault being
        # repeated every minute for as long as it lasts.
        self._drift_baseline = 0.0
        self._drift_walked_away = False

    @property
    def clipped_samples(self) -> int:
        """Raw values that reached the converter's rail since capture began.

        A count above zero means the receiver gain is too high for what the antenna is
        hearing, and that loud events are being measured smaller than they are.
        """
        return self._clipped

    @property
    def iq_buffer(self) -> IqRingBuffer | None:
        """The raw IQ history, or None when nothing asked for one to be kept."""
        return self._iq_buffer

    @property
    def source(self) -> RtlSdrSource:
        """The capture this is draining, for anything that wants its counters."""
        return self._source

    def start(self) -> None:
        """Begin capture and begin draining it."""
        self._source.start()
        self._thread.start()

    def close(self) -> None:
        """Stop capture, let the feeder finish, and release the device.

        The source is closed first so no further blocks arrive.  The feeder then sees
        its read time out, notices the stop, and ends.
        """
        self._stop.set()
        self._source.close()
        if self._thread.is_alive():
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    def _feed(self) -> None:  # pragma: no cover -- thread body; _consume is tested directly
        """Drain the source until told to stop."""
        while not self._stop.is_set():
            try:
                block = self._source.read(timeout=_FEED_READ_TIMEOUT_SECONDS)
                if block is not None:
                    self._consume(block)
            except Exception:
                logger.exception(
                    'Converting a block of receiver samples failed.  That block is '
                    'lost and capture continues, so the audio has a gap in it.')

    def _consume(self, block: IqBlock) -> None:
        """Convert one block and hand the audio to the ring buffer.

        Counting the clipping here rather than in the callback is deliberate.  The raw
        bytes are sitting in the block either way, and the callback has a deadline
        this does not.
        """
        self._clipped += block.clipped_samples
        # Kept before the conversion, so that a conversion that raises still leaves the
        # raw bytes behind.  They are what a recording writes, and they are the one
        # thing this block carries that nothing else can reconstruct afterward.
        if self._iq_buffer is not None:
            self._iq_buffer.add(block)
        self._append_in_chunks(self._converter.convert(block.as_complex()))
        self._report_health()

    def _report_health(self) -> None:
        """Say once a minute whether the receiver is still delivering honest samples.

        Each failure this reports is silent in the data it spoils.  A receiver whose
        gain is too high clips every loud arc, measures it smaller than it is, and
        writes that figure to the CSV with nothing to mark it.  Lost samples do the
        same to the grid frequency.  Both read as a quiet band, which is the answer
        the operator is hoping for, so neither prompts anybody to look.

        A log line is the smallest thing that makes the counters visible.  The
        counters stay public so that a CSV column or a light on the display can read
        them later instead.

        The check runs from _consume rather than from the feeder loop, so a test can
        drive it directly.  A stream that stops entirely therefore stops reporting,
        which is correct.  _run already says the receiver went quiet, and a second
        voice for one failure would only add noise.
        """
        now = self._clock()
        elapsed = now - self._health_checked_at
        if elapsed < _HEALTH_INTERVAL_SECONDS:
            return
        self._health_checked_at = now
        self._warn_about_clipping(elapsed)
        self._warn_about_drift(elapsed)

    def _warn_about_clipping(self, elapsed: float) -> None:
        """Report clipping at a rate that could move a measurement, and not below it.

        The two counts are kept apart because they have separate causes.  A raw value
        at the converter's rail means the antenna is louder than the tuner gain
        allows.  A clipped output sample can happen without that, because the filter
        can leave a peak slightly above where its input sat.

        Every clipped sample used to be reported, with advice to lower the gain a
        step.  The docstring said a threshold would need a figure nobody had measured,
        which was true when it was written.  It is not now, and the figures say the
        advice was costing more than it saved.

        A station running at its calibrated gain saw a handful of clipped values a
        minute, from the first burst of an intermittent arc, which is far louder than
        the train that follows.  Fourteen raw values in sixty seconds at 256 kHz is
        0.46 parts per million, and moves an averaged burst amplitude by eight
        millionths of a decibel.  Acting on it costs a whole gain step, and a step
        below the knee costs one to three decibels on every noise floor the station
        reports from then on.  That is a bad trade in every direction.

        Hundreds a minute is different and worth acting on, which is what the
        threshold separates.
        """
        clipped = max(0, self._clipped - self._clipped_reported)
        saturated = max(0, self._converter.saturated_samples - self._saturated_reported)
        self._clipped_reported = self._clipped
        self._saturated_reported = self._converter.saturated_samples
        raw_values = elapsed * self._source.iq_sample_rate * VALUES_PER_FRAME
        if clipped < raw_values * CLIPPING_WORTH_NOTICING and not saturated:
            return
        logger.warning(
            'The receiver clipped %d raw value(s) in the last %.0f seconds, and the '
            'conversion clipped %d output sample(s).  Loud events are measured smaller '
            'than they are.  Run the gain calibration again on a quiet band, which '
            'measures what [rtlsdr] gain_db should be rather than guessing a step.',
            clipped, elapsed, saturated)

    def _warn_about_drift(self, elapsed: float) -> None:
        """Report a receiver clock that has run away from the system clock.

        This one needs a limit where the counters above do not, because the figure is
        never exactly zero.  Two crystals always disagree by some parts per million,
        so "any movement" would report every minute of a healthy run.  What is
        measured here is the change since the last report rather than the total, so a
        steady offset settles instead of accumulating into a warning.

        The first interval sets the baseline instead of being measured against zero,
        because a receiver's own startup falls entirely inside it.

        The two directions are different faults and the message says which.  Less
        audio than the interval means samples went missing, and a machine with nothing
        left to give is the usual cause.  More audio than the interval cannot be a
        loss, because the blocks arrived.  Only the first case spoils a measurement,
        so only the first case says so.

        This reports every interval at DEBUG and not only the ones over the limit,
        because one interval says almost nothing.  What a run of them shows is a
        cumulative figure that climbs for seven or eight minutes and then discharges in
        one interval, which is what warns.

        Measured on an RSP1B at 3530 kHz on 2026-09-18, over three cycles in two runs:
        the cumulative climbed to +48.1, +63.5 and +48.7 ms, and discharged -52.3, -56.4
        and -59.4 ms.  Every one returned to within 20 ms of zero, so nothing was lost
        or gained.  The third was predicted before it happened, which is the reason to
        believe the first two.

        `SdrplayDevice._note_the_backlog` measures the receiver side directly, and
        it rules out a stall.  Over the nine paired minutes the worst wait between
        deliveries held between 71.6 and 87.6 ms while this figure swung from -56.4 to
        +22.3, and the minute that warned was 74.4 ms, which is the middle of that
        band.  The two are uncorrelated at r = -0.416.

        So a negative figure here is a buffer in the library emptying rather than a
        fault, and the per-interval movement measures buffer depth as well as audio
        going missing.  The cumulative figure separates them, because a buffer cycle
        returns to zero and a rate error does not.  See
        docs-notebook/receiver-clock-drift.md, which holds both tables and says what
        would be needed to warn on the cumulative instead.
        """
        drift = self._source.clock_drift_seconds
        if self._drift_reported is None:
            # The first interval is the baseline rather than a measurement.  A receiver
            # fills its pipeline as it starts and delivers that first stretch faster
            # than real time, so the drift accumulated by the end of the first interval
            # describes the startup and not the run.  Measured on an SDRplay RSP1B, it
            # came to 37 ms, which is twenty times what a crystal explains and entirely
            # gone by the next interval.
            #
            # What this gives up is a loss during the first interval, which goes
            # unreported.  A rate that is wrong still shows up, because it keeps
            # moving and this only absorbs what had already happened.
            self._drift_reported = drift
            self._drift_baseline = drift
            logger.debug('Receiver clock baseline is %+.0f ms after the first %.0f '
                         'seconds.', drift * 1e3, elapsed)
            return
        moved = drift - self._drift_reported
        self._drift_reported = drift
        total = drift - self._drift_baseline
        logger.debug('Receiver clock moved %+.1f ms over the last %.0f seconds, and '
                     'stands %+.1f ms from its baseline.', moved * 1e3, elapsed,
                     total * 1e3)
        lost = moved > elapsed * _DRIFT_PPM_LIMIT / 1e6
        if lost:
            logger.warning(
                'The receiver and system clocks moved %+.0f ms apart over the last '
                '%.0f seconds, which is more than a crystal explains.  %s',
                moved * 1e3, elapsed, _what_a_loss_means())
        self._warn_if_the_clock_has_walked_away(total, already_warned=lost)

    def _warn_if_the_clock_has_walked_away(self, total: float,
                                           already_warned: bool) -> None:
        """Report a clock that has left its baseline and stayed away.

        This is the check the per-interval one cannot do.  A buffer that fills and
        empties moves a single interval by tens of milliseconds and comes back, where a
        rate error or a steady loss keeps going.  Only a total since the baseline tells
        those apart, and a leak too slow to cross the per-interval limit reaches this
        one eventually.

        Said once per excursion rather than once a minute for as long as it lasts.  A
        fault that persists is still the same fault, and repeating it every minute
        teaches an operator to filter the log.

        `already_warned` says the interval check has spoken about this same minute.  A
        loss large enough to trip both is one event, and describing it twice buries the
        part the operator has to act on.  The flag still moves, so the next excursion
        after a recovery is reported.
        """
        outside = abs(total) > _CUMULATIVE_DRIFT_LIMIT_SECONDS
        if outside and not self._drift_walked_away and not already_warned:
            logger.warning(
                'The receiver clock stands %+.0f ms from where it started, which is '
                'more than a buffer cycle explains.  %s',
                total * 1e3, _what_a_sustained_drift_means(total))
        self._drift_walked_away = outside

    def _append_in_chunks(self, audio: np.ndarray) -> None:
        """Hand the audio over in pieces of exactly CHUNK_SIZE, holding any remainder.

        get_snapshot works out how many chunks to read by dividing the sample count
        by CHUNK_SIZE, so a shorter chunk makes it return
        less audio than it was asked for, with no error anywhere.  Measured on the real buffer,
        appending 256-sample chunks to one asked for 4000 samples returns 2064.  The
        analyzer would then be handed a window shorter than the one it sized its
        arithmetic for.

        A converted block does not divide evenly into chunks, and the first one is
        short while the filter fills, so what does not make a whole chunk waits here
        for the next block.
        """
        pending = np.concatenate([self._leftover, audio])
        whole = len(pending) // self.CHUNK_SIZE * self.CHUNK_SIZE
        for start in range(0, whole, self.CHUNK_SIZE):
            self._append(pending[start:start + self.CHUNK_SIZE])
        self._leftover = pending[whole:]


class SdrLevelStream(LevelStream):
    """A live level in dBm, fed by a receiver, for the setup program's meter.

    This owns a thread rather than reaching through RtlSdrPipeline, because a meter
    wants the newest reading rather than a history.  The ring buffer would only add
    its own latency to a number somebody is watching while they turn a knob.

    Everything that turns a block into a reading stays in LevelStream, and none of it
    is overridden here.  That is the point of the split: an operator calibrating
    against this meter has to be calibrating against the figure the monitor itself
    would report.
    """

    def __init__(self, source: RtlSdrSource, converter: 'IqToAudio',
                 offset_db: float) -> None:
        # One IQ block converts to one audio block, so they last the same time and
        # either one gives the smoothing its time constant.
        block_seconds = source.block_samples / source.iq_sample_rate
        super().__init__(offset_db, converter.audio_sample_rate,
                         round(block_seconds * converter.audio_sample_rate))
        self._source = source
        self._converter = converter
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name='sdr-level')
        self._source.start()
        self._thread.start()

    def _drain(self) -> None:  # pragma: no cover -- thread body; _consume is tested directly
        """Read from the receiver until told to stop."""
        while not self._stopping.is_set():
            try:
                block = self._source.read(timeout=_FEED_READ_TIMEOUT_SECONDS)
                if block is not None:
                    self._consume(block)
            except Exception:
                logger.exception(
                    'Converting a block for the level meter failed.  The reading is '
                    'now stale, and the meter keeps running.')

    def _consume(self, block: IqBlock) -> None:
        """Convert one block and fold it into the reading.

        An empty result is normal rather than an error: the filter needs samples it
        has not been given yet, which is always true of the first call.  Passing an
        empty block on would take the median of nothing and poison the reading with
        a NaN that never clears.
        """
        audio = self._converter.convert(block.as_complex())
        if len(audio):
            self._on_block(audio)

    def _stop(self) -> None:
        self._stopping.set()
        self._source.close()
        if self._thread.is_alive():
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

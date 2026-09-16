"""Raw IQ capture from an RTL-SDR receiver, and nothing else.

This module holds the hardware.  It opens the device, configures it, and hands out
blocks of raw bytes exactly as they arrived.  It has no opinion about what happens
next.  The conversion to audio lives in buzz.iq, and the ring buffer everything else
reads lives in buzz.sampler.  Keeping the three apart is what lets the conversion be
tested exhaustively with no receiver attached.  It also leaves somewhere for IQ
recording to tap in later, without disturbing either neighbor.

Why the callback does almost nothing
------------------------------------
The receiver's own FIFO holds 1880 bytes, which is 940 complex samples, so at 256 kHz
it overflows 3.67 ms after collection stops.  That is far shorter than a Windows
scheduler quantum, and nothing anywhere reports the loss, because it happens in
hardware upstream of every piece of software.  A sound card can report an overflow
because the driver owns the buffer that overflowed.  Here nobody owns it.

What saves the arrangement is librtlsdr's pool of USB transfers, which the host
controller fills by DMA without our thread being scheduled at all.  Measured on this
hardware, that pool holds about 960 ms at our block size, so the 3.67 ms deadline is
met by the USB stack rather than by Python.

The deadline that does fall to us is softer and different.  Each callback carries
`block_samples` worth of audio, so the callback must average less than that duration
or the pool drains and never recovers.  Measurement showed that a callback stalled
60 ms against a 64 ms block stayed clean, while one stalled 200 ms lost 67% of the
stream and kept losing it.  The pool absorbs bursts, not sustained slowness.

So the callback copies its block, timestamps it, and returns.  Every other piece of
work, including converting to complex and counting clipped samples, belongs to
whatever thread drains this class.

Why close() is also registered with atexit
------------------------------------------
A receiver that is never closed keeps running.  It goes on streaming with nothing
collecting the samples, which wastes power and warms the tuner for no purpose.  See
https://github.com/librtlsdr/librtlsdr/issues/116

Be careful what that does and does not cost, because the obvious guess is wrong.
Measured on this hardware, across a process that exits without closing, the next
process opened the device in 0.72 s on its first attempt.  Closing first made no
difference, at 0.75 s and also on the first attempt.  The operating system reclaims the USB handle
when a process ends, so a skipped close does not strand the device for anybody else.

The hook is therefore ordinary resource hygiene rather than a fix for a measured
failure.  It costs nothing, the explicit call is the one that normally runs, and the
hook covers paths that skip it such as an unhandled exception on another thread.
close() is idempotent because during an orderly shutdown both will fire.

Nothing saves a device from a hard kill, and nothing can.
"""

import logging
import queue
import threading
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np

from buzz.sampler import LevelStream, RingBufferPipeline
from buzz.sdr_device import VALUES_PER_FRAME, IqBlock, SdrDevice

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

# How often to repeat the discard warning, counted in discarded blocks.  The first is
# always reported and then every hundredth, so a sustained problem stays visible
# without costing a log line per block.
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
# The figure is chosen rather than measured.  RTL-SDR crystals are specified in the
# tens of parts per million, so 500 leaves room for a poor one and still catches a
# loss, which runs to thousands.  See RtlSdrSource.clock_drift_seconds.
_DRIFT_PPM_LIMIT = 500


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
    def blocks_to_discard_after_gain_change(self) -> int:
        """Blocks that may predate a gain change, and so have to be thrown away.

        The streaming figure, because this drains a transfer pool.  A synchronous read
        has none and takes the smaller one.  See DeviceProfile.
        """
        return self._device.profile.blocks_to_discard_streaming

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

        The only available evidence that samples went missing, since nothing reports a
        drop.  A positive figure means less audio arrived than the wall clock says it
        should have.

        Read it as a symptom rather than a measurement.  The receiver's crystal and the
        system clock differ by some parts per million that nobody here has measured, so
        the two separate slowly even when nothing is wrong.  Milliseconds over a few
        seconds mean lost samples.  Microseconds mean clocks.
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

    def drain(self) -> int:
        """Throw away every block already queued, and say how many that was.

        There are two buffers between the tuner and a caller, and counting only one of
        them is not enough.  blocks_to_discard_after_gain_change covers the driver's
        transfer pool, which is the buffering nobody here can see.  This queue is the
        other one, and anything sitting in it when the gain changes was captured before
        the change.

        Without this, the counted discard spends itself on stale queue entries first
        and lets that many true post-change callbacks through in their place, so up to
        buffer_blocks blocks of the previous gain reach the measurement.  It is worst
        at the first step of a gain sweep, where the queue has been filling since
        start() with nobody reading.
        """
        dropped = 0
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return dropped
            dropped += 1

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
        if discarded == self._reported_discards:
            return
        first = self._reported_discards == 0
        self._reported_discards = discarded
        if first or discarded % _DISCARD_LOG_EVERY == 0:
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
    def blocks_to_discard_after_gain_change(self) -> int:
        """Blocks to read and throw away after moving the gain.

        The reading figure rather than the streaming one, because a synchronous read
        has no transfer pool behind it.  See DeviceProfile.
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

    This is the third of the three pieces.  RtlSdrSource holds the hardware,
    IqToAudio holds the arithmetic, and this owns the thread that carries blocks from
    one to the other.  Everything downstream reads this exactly as it reads the sound card, and
    cannot tell which it has.

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
        self._drift_reported = 0.0

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
        """
        drift = self._source.clock_drift_seconds
        moved = drift - self._drift_reported
        self._drift_reported = drift
        if abs(moved) <= elapsed * _DRIFT_PPM_LIMIT / 1e6:
            return
        logger.warning(
            'The receiver and system clocks moved %+.0f ms apart over the last %.0f '
            'seconds, which is more than a crystal explains.  Samples were probably '
            'lost, so levels and grid frequency from this period are suspect.  Check '
            'what else on this machine is taking the CPU.', moved * 1e3, elapsed)

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

"""
File-backed audio source: replays a recorded .wav through the live pipeline.

FilePlaybackPipeline fills the same ring buffer AudioPipeline does, from a thread
that feeds chunks at the file's own sample rate instead of a PortAudio callback.
Everything downstream — analyzer, scope, waterfall, meters — is unchanged and
cannot tell the difference, which is the point: an event captured by the recorder
can be replayed later and analysed exactly as it was live, with the display
running at real speed for a screen recording.

No input device is opened, so a recording can be reviewed on a machine with no
receiver attached.  An output device is opened only while audio is actually being
written to it — not while muted, paused, or stopped at the end of the file — so
muted playback, which is what --mute asks for and what this class does unless told
otherwise, runs exactly the code that ran before playback could be heard at all,
paced by a monotonic deadline rather than by a sound card that nobody is listening
to and that some machines do not have.

Note that the class defaults to muted and the command line does not: building a
pipeline must never open a device the caller did not ask for, but somebody who
typed --playback wants to hear it (see buzz.main).

At the end of the file the feeder simply stops.  The buffer keeps its last few
seconds, so the display holds its final frame rather than going blank, and the
analyzer drops to SEARCHING on its own once the audio stops advancing.
"""

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from buzz.sampler import RingBufferPipeline

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2   # 16-bit PCM, matching what the recorder writes
_INT16_MIN, _INT16_MAX = -32768, 32767


def sample_rate_of(path: Path | str) -> int:
    """The sample rate from a .wav's header, without reading its audio.

    Exists so that a rate this program cannot work at is refused before anything
    expensive runs -- the loudness probe reads the whole recording, and being turned
    away after that has happened is a poor experience.  A header read is a few bytes.
    """
    with wave.open(str(path), 'rb') as wav:
        return wav.getframerate()


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM .wav into (int16 samples, sample rate).

    Any .wav plays, not only ones this program recorded — somebody being sent a file
    by another operator is exactly who this is for.  One property is refused rather
    than coped with: the **sample width**.  The whole signal chain is int16 end to end,
    and silently converting a 24- or 32-bit file would put what is measured here at
    odds with what the same audio measured live — a display and a set of numbers that
    look entirely plausible and are wrong.

    The sample rate is *not* checked here, nor by the pipeline.  Reading a .wav and
    deciding what rates this program will work at are different questions, and the
    second belongs to main -- which is where a file from outside becomes something the
    analyzer and display will be asked to work on.  A caller that only wants the
    samples back, which the tests do at rates chosen for speed, should not be subject
    to a policy about the display's bandwidth.

    Multi-channel files are reduced to channel 0 rather than mixed down, matching what
    the live pipeline does with a stereo input device, and say so in the log — a file
    from elsewhere is quite likely to be stereo, and "half of what you sent was
    ignored" should not have to be inferred from a reading that came out low.
    """
    with wave.open(str(path), 'rb') as wav:
        if wav.getsampwidth() != _BYTES_PER_SAMPLE:
            raise ValueError(
                f'{path}: expected 16-bit PCM audio, got {wav.getsampwidth() * 8}-bit')
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        if channels > 1:
            # Said out loud rather than left to the docstring.  A file from somebody
            # else is quite likely to be stereo, and "we analysed half of what you
            # sent" is not something anyone should have to infer from a number that
            # came out lower than expected.  Channel 0 rather than a downmix, matching
            # what the live pipeline does with a stereo input device -- mixing would
            # average the arc against whatever the other channel happens to hold.
            logger.warning(
                '%s has %d channels; analysing channel 0 only, not a downmix: the '
                'same thing the live monitor does with a stereo input. Extract another '
                'channel to mono first if the interference is not on this one.',
                Path(path).name, channels)
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype='<i2')[::channels]
    # astype() rather than the frombuffer view: that view is read-only and borrows
    # the file's byte order, and consumers expect a plain writable native int16 array.
    return samples.astype(np.int16), sample_rate


def apply_gain(samples: np.ndarray, factor: float) -> np.ndarray:
    """Scale samples by `factor`, stopping at the int16 rails.

    Too much gain therefore distorts, which is what anyone turning something up too
    far expects.  It is worth the clamp to get that: int16 *wraps* on overflow, so
    without one an overflowed sample changes sign, and a passage a few dB too loud
    comes back as noise rather than as a loud passage.  Scaling through float32 keeps
    the multiply out of int16 entirely, where that wrap would happen before anything
    could be clamped.
    """
    if factor == 1.0:
        return samples
    scaled = samples.astype(np.float32) * factor
    return np.clip(scaled, _INT16_MIN, _INT16_MAX).astype(np.int16)


def resolve_playback_path(name: Path | str, directory: Path) -> Path:
    """Resolve a --playback argument to a file path.

    A bare filename is looked up in the recordings directory, so replaying a
    capture is just `--playback event-20260729-143307-0700.wav`; anything with a
    directory component in it is used as given.
    """
    path = Path(name)
    return directory / path if str(name) == path.name else path


class FilePlaybackPipeline(RingBufferPipeline):
    """Replays a .wav into the ring buffer at real-time speed, with transport control.

    The feeder thread appends one CHUNK_SIZE chunk at a time against a deadline
    schedule computed from the file's sample rate.  Deadlines accumulate from an
    origin rather than by sleeping a fixed interval each pass, so the small
    per-chunk overhead of the sleep and the append cannot accumulate into playback
    running progressively slower than the audio it represents.  Pausing and
    restarting re-base that origin, since the schedule is only meaningful for a
    stretch of uninterrupted play.

    pause(), resume() and restart() are called from the Qt thread while the feeder
    runs, so the play position and the origin live behind a Condition the feeder
    waits on.  Appending takes the ring buffer's own lock inside that one, which is
    the only place the two nest, and always in that order.
    """

    def __init__(self, path: Path | str, muted: bool = True,
                 gain_db: float = 0.0) -> None:
        # Loaded before the buffer is sized, because the buffer is sized in seconds
        # and only the file knows what rate those seconds are at.
        samples, sample_rate = load_wav(path)
        super().__init__(sample_rate)
        self._samples, self.sample_rate = samples, sample_rate
        self.path = Path(path)
        # Partial trailing chunks are dropped: consumers read whole chunks, and up
        # to 32 ms of audio at the very end of a file is not worth a special case.
        self._chunks = len(self._samples) // self.CHUNK_SIZE
        self._chunk_period = self.CHUNK_SIZE / self.sample_rate
        # What will actually be played, rather than what the file holds, so that the
        # transport's time index can reach its own total.  position is quantised to
        # whole chunks, so measuring it against the file's full length would leave a
        # finished replay reading 00:39 / 00:40 for the sake of the dropped fraction.
        self.duration = self._chunks * self._chunk_period

        self._state = threading.Condition()
        self._index = 0             # next chunk to feed
        self._paused = False
        self._origin = 0.0          # monotonic time that _origin_index was due at
        self._origin_index = 0
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._rebase()

        # Silent unless asked otherwise, so that building a pipeline never opens an
        # audio device the caller did not ask for — tests and headless replays have
        # no use for one, and some machines have none to open.
        self._muted = muted
        self._output: sd.OutputStream | None = None
        self._output_failed = False
        self._flush_requested = False
        # Gain is for the speakers alone.  Recordings sit at -24 to -34 dBFS, which
        # is quiet on the sort of speakers a laptop has, but the analyzer's whole job
        # is to report the level that was actually there — so this is applied to the
        # copy on its way to the sound card and to nothing else.
        self.gain_db = gain_db
        self._gain = 10 ** (gain_db / 20)
        self.output_available = self._can_open_output()

        self._started = False
        self._thread = threading.Thread(
            target=self._feed, daemon=True, name='playback')

    def start(self) -> None:
        """Begin feeding audio.  Idempotent.

        Constructing a pipeline does not start one.  The caller has a display to
        build first, and starting in the constructor means the opening moments of
        the event are heard before there is a window to see them in — and, worse,
        are fed while widget construction still holds the GIL, so the sound card
        runs dry and the replay opens with a stutter that is not in the recording.
        Opening the file and playing it are separate acts, and only the caller knows
        when the second one should happen.
        """
        if not self._started:
            self._started = True
            # The deadline schedule starts here, not at construction.  _rebase() in
            # __init__ sets an origin, and everything between building the pipeline
            # and starting it — building the window, wiring the analyzer, compiling
            # the FFT kernels on first use — is time the schedule counts as already
            # spent.  The feeder then finds every early chunk overdue and delivers
            # them as fast as the loop runs.
            #
            # Measured before this: 1.5 s between construction and start() put the
            # transport at 1.600 s within 100 ms of starting.  That is a second and a
            # half of a recording pushed through the analyzer at whatever speed the
            # machine manages, which is not what any of it was tuned against, and it
            # defeats the deliberate delay in main.py that waits for the window so
            # the opening of an event is not heard before there is anywhere to see it.
            with self._state:
                self._rebase()
            self._thread.start()

    # ------------------------------------------------------------------ public

    @property
    def finished(self) -> bool:
        """True once the last chunk of the file has been fed into the buffer."""
        return self._finished.is_set()

    @property
    def paused(self) -> bool:
        with self._state:
            return self._paused

    @property
    def position(self) -> float:
        """How far into the file playback has reached, in seconds."""
        with self._state:
            return self._index * self._chunk_period

    @property
    def muted(self) -> bool:
        with self._state:
            return self._muted

    def set_muted(self, muted: bool) -> None:
        """Ask for playback audio to be switched off or on.

        Only records the wish.  The feeder thread is the one that opens and closes
        the stream, because it is also the one writing to it, and closing a stream
        from another thread while a write is in flight is undefined behaviour in
        PortAudio.  Takes effect within one chunk.

        Unmuting clears a previous failure to open the device, so asking again after
        plugging something in is worth a retry; muting never fails.

        Logged here rather than where the stream is actually opened and closed: the
        stream now comes and goes with pausing too, so its lifecycle no longer tracks
        anything the operator asked for.  This is the request, which is the part worth
        a line — and a device that then refuses to open says so itself.
        """
        with self._state:
            self._muted = muted
            if not muted:
                self._output_failed = False
            self._state.notify_all()
        logger.info('Playback audio %s.', 'off' if muted else 'on')

    def toggle_mute(self) -> None:
        self.set_muted(not self.muted)

    def pause(self) -> None:
        """Stop feeding, and release the sound card until playback resumes.

        The stream closes rather than idling — see _sync_output — so a pause left
        running over lunch is not an open device underrunning the whole time.
        """
        with self._state:
            self._paused = True
            self._state.notify_all()

    def resume(self) -> None:
        """Carry on from the current position.  Does nothing at the end of the file."""
        with self._state:
            if self._index >= self._chunks:
                return
            self._paused = False
            self._rebase()
            self._state.notify_all()

    def toggle_pause(self) -> None:
        """Pause, or carry on.  Does nothing once the file has finished.

        Without that guard the space bar can pause a replay that has already ended —
        leaving the transport paused at a position it cannot be resumed from, since
        resume() rightly refuses there, and disagreeing with a Play button that is
        greyed out for exactly this reason.  Restart is the way back from the end.
        """
        if self.finished:
            return
        self.pause() if not self.paused else self.resume()

    def restart(self) -> None:
        """Play again from the beginning, whether paused, playing, or finished.

        Restarting resumes rather than staying paused: a restart button that leaves
        a paused file paused at zero looks like it did nothing at all.

        The ring buffer is emptied too.  It still holds the last several seconds of
        the pass just abandoned, and leaving that in place would hand an analyzer
        that has just been reset a locked-on pulse train to find immediately —
        producing a second pass that opens already locked, which is precisely what
        restarting is meant to avoid.  The cost is a few seconds of empty display
        while the new pass refills it, the same as at startup.

        Audio queued to the sound card is thrown away for the same reason, on the
        feeder thread — see _flush_output.
        """
        with self._state:
            self._index = 0
            self._paused = False
            self._finished.clear()
            self._flush_requested = True
            self._rebase()
            self.clear()
            self._state.notify_all()

    # ----------------------------------------------------------------- internal

    def _rebase(self) -> None:
        """Start the deadline schedule again from now (caller holds _state)."""
        self._origin, self._origin_index = time.monotonic(), self._index

    def _can_open_output(self) -> bool:
        """Whether this machine could play the file, asked once at startup.

        Only so the toolbar can grey its mute button out with a reason, rather than
        offering a control that silently does nothing.
        """
        try:
            sd.check_output_settings(
                samplerate=self.sample_rate, channels=1, dtype='int16')
            return True
        except Exception as exc:
            logger.info('No usable audio output for %d Hz mono playback (%s); '
                        'replay will be silent.', self.sample_rate, exc)
            return False

    def _sync_output(self) -> None:
        """Open or close the output stream to match what the transport is doing.

        Feeder thread only — see set_muted.  A stream is wanted only while audio is
        actually being written to it, which muting, pausing and reaching the end of
        the file each stop being true of.  A stream nobody writes to underruns for as
        long as it is left open, so all three close it: mute already meant the absence
        of a stream rather than a volume of zero, and pausing reaches the device by
        exactly the same route rather than by a second mechanism of its own.

        Both transitions re-base the deadline schedule, which matters most in the
        direction nobody thinks about: on closing, an origin left over from before the
        stream opened is minutes in the past, and the loop would race through the rest
        of the file trying to catch up to it.
        """
        with self._state:
            at_end = self._index >= self._chunks
            wanted = not (self._muted or self._output_failed or self._paused or at_end)
            flush, self._flush_requested = self._flush_requested, False
        if self._output is not None and not wanted:
            self._close_output(drain=at_end)
        elif self._output is not None and flush:
            self._flush_output()
        if wanted and self._output is None:
            self._open_output()

    def _flush_output(self) -> None:
        """Discard audio already queued to the card, and start it playing again.

        A restart rewinds the file, but an output buffer's worth of the pass being
        abandoned is already sitting in the card's queue.  Left there it plays out
        after the click — the old position, heard over the new one — and every later
        chunk trails the display by that same buffer for the rest of the run.

        Aborting and restarting the same stream rather than reopening the device:
        it is faster, and it cannot fail with the device busy and leave the replay
        silent for the sake of a button press.
        """
        self._output.abort()
        self._output.start()

    def _open_output(self) -> None:
        try:
            stream = sd.OutputStream(samplerate=self.sample_rate, channels=1,
                                     dtype='int16', latency='low')
            stream.start()
        except Exception:
            logger.warning('Could not open the audio output — replay stays silent.',
                           exc_info=True)
            with self._state:
                self._output_failed = True
            return
        self._output = stream
        with self._state:
            self._rebase()
        logger.debug('Playback output stream opened.')

    def _close_output(self, drain: bool = False) -> None:
        """Close the output stream, by default throwing away what is still queued.

        abort() rather than stop() wherever somebody asked for silence *now*: stop()
        plays the queue out first, and a mute or pause button that takes a fifth of a
        second to go quiet is a broken one.  The queued audio was already fed to the
        ring buffer, so nothing is lost from the replay itself — only from the speaker.

        The end of the file is the one place that reverses.  Nobody asked for silence
        there, the queue holds the last of the recording, and cutting it off mid-sample
        would end the replay on exactly the step the recorder's fade-out exists to
        prevent.  So that one drains.
        """
        self._output.stop() if drain else self._output.abort()
        self._output.close()
        self._output = None
        with self._state:
            self._rebase()
        logger.debug('Playback output stream closed.')

    def _wait_until_playable(self) -> bool:
        """Block while paused or sitting at the end; False once stopping.

        Reconciles the output stream on every pass, including each time it wakes.
        Being parked here is itself a reason to have no stream — nothing is being
        written while the feeder waits — and it is also the only moment a mute
        arriving during a pause could otherwise be noticed.  Either way the device
        does not sit open and starving while the transport is stopped.
        """
        while not self._stop.is_set():
            self._sync_output()
            with self._state:
                if not (self._paused or self._index >= self._chunks):
                    return True
                self._state.wait()
        return False

    def _feed(self) -> None:
        logger.info('Playing back %s — %.1f s at %d Hz',
                    self.path.name, self.duration, self.sample_rate)
        while self._wait_until_playable():
            with self._state:
                index = self._index
                due = self._origin + (index - self._origin_index) * self._chunk_period
            chunk = self._samples[index * self.CHUNK_SIZE:(index + 1) * self.CHUNK_SIZE]

            # Whichever clock is available paces the loop.  With a stream open the
            # sound card is it — write() returns once the card has room, which is by
            # definition real time — and without one the deadline schedule is, exactly
            # as it was before playback could be heard at all.
            if self._output is not None:
                # The only place gain is applied.  What goes into the ring buffer
                # below is the untouched chunk, so turning a quiet recording up to
                # hear it cannot move a single dB of what the monitor reports.
                self._output.write(apply_gain(chunk, self._gain))
            else:
                delay = due - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    return

            with self._state:
                # A pause or a restart while this chunk was waiting its turn makes
                # it the wrong chunk to play; go back and read the new position.
                if self._paused or self._index != index:
                    continue
                self._append(chunk)
                self._index += 1
                if self._index >= self._chunks:
                    self._finished.set()
                    logger.info('Playback finished: %s', self.path.name)

    def close(self) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()
        if self._started:
            self._thread.join(timeout=1.0)
        # Only once the feeder has actually stopped.  The join above has a timeout,
        # and a feeder still blocked inside write() on a device that stopped draining
        # — one unplugged mid-replay — would outlive it; closing the stream from
        # under that write is the undefined behaviour the rest of this class takes
        # care to avoid.  Leaking a stream on a device that is already misbehaving is
        # the better of the two outcomes, and the process is on its way out anyway.
        if self._output is not None and not self._thread.is_alive():
            self._close_output()
        elif self._output is not None:
            logger.warning('Playback feeder did not stop; leaving the audio output '
                           'open rather than closing it mid-write.')

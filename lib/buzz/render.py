"""Render a playback session to an .mp4 file, through ffmpeg.

Only reached when `--render` is passed.  Nothing here is imported otherwise and the
ffmpeg binary is never looked for, so a monitor that never renders cannot fail — or
even complain — about a program it was never going to run.  The same reasoning as the
recorder's, which creates its directory when recording is armed rather than at
startup.

ffmpeg is driven as a subprocess rather than through a wrapper package.  There is one
command line here, of fixed shape, and a literal argument list can be logged and
pasted straight into a terminal when a render comes out wrong — which is worth more
than any ergonomics a builder API offers.  It also keeps the whole feature free of
install-time weight: `subprocess` and `shutil` are the standard library, so a user
who does not render adds nothing to their environment.

Only the video is piped.  The audio ffmpeg needs is already a file on disk — the .wav
being replayed — so it is passed as a second input and laid on its own timeline.  That
removes the audio path from the sync problem altogether: there is no second stream to
keep aligned, only frames to place correctly against a clock that is already right.
"""

import logging
import math
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# The output grid.  Deliberately 30 and not 29.97: a 1001/1000 rational timebase would
# put every frame time permanently off a round number against an audio timeline
# measured in exact 48000ths, and the NTSC colour-subcarrier problem that produced
# 29.97 in 1953 is not one this file has.
FRAME_RATE = 30

# The display updates at 10 fps (_UPDATE_MS in waterfall.py and scope.py), so two out
# of every three output frames are duplicates.  They cost almost nothing — a P-frame
# with no residual — and the finer grid is what keeps frame placement inside the 32 ms
# quantum that the playback position itself is measured in.  See _FrameGrid.
_KEYFRAME_INTERVAL = FRAME_RATE          # one per second, for scrubbing
_CRF = 18
_PRESET = 'veryslow'                     # this little video encodes in no time
_PROFILE = 'main'
_AUDIO_BITRATE = '192k'
_AUDIO_RATE = 48000                      # 16 kHz plays badly in some browsers

# What the frames arriving here are expected to be: 8 bits each of R, G, B and A, in
# that byte order, with no row padding.  Qt's Format_RGBA8888 is exactly this, and
# saying so explicitly keeps the pipe endian-independent.
PIXEL_FORMAT = 'rgba'
_BYTES_PER_PIXEL = 4


class RenderError(RuntimeError):
    """Rendering could not start, or ffmpeg failed while it ran."""


def find_ffmpeg(configured: str | None = None) -> str:
    """Locate the ffmpeg binary, or say clearly why rendering cannot happen.

    PATH first, then `render.ffmpeg_path` from the config.  That order means a normal
    install needs no configuration at all, and the setting exists for the installs
    that do not land on PATH — a Windows build unzipped into a folder, or winget's
    shim directory in a terminal that has not been restarted since.

    The consequence worth knowing: a copy on PATH wins over a configured one, so the
    setting cannot be used to override a working ffmpeg with a different build.  That
    has not been needed; if it ever is, the order is one line.

    Called once, when a render is asked for.  Failing here costs the operator nothing
    but a message; failing after ten minutes of rendering costs them ten minutes, so
    it happens before the first frame rather than at the first write.
    """
    on_path = shutil.which('ffmpeg')
    if on_path:
        return on_path
    if configured:
        # Accept a directory as well as the binary itself.  Pointing at the folder is
        # the more natural reading of "where ffmpeg is", and getting it wrong would
        # otherwise fail with a message about a file that is plainly right there.
        candidate = Path(configured)
        if candidate.is_dir():
            candidate = candidate / 'ffmpeg.exe' if (candidate / 'ffmpeg.exe').exists() \
                else candidate / 'ffmpeg'
        if candidate.exists():
            return str(candidate)
        raise RenderError(
            f'ffmpeg is not on PATH and render.ffmpeg_path points at {configured}, '
            'where there is no ffmpeg.')
    raise RenderError(
        'ffmpeg was not found on PATH, and --render needs it. Install it (on Windows, '
        '"winget install Gyan.FFmpeg") or set render.ffmpeg_path in the config to '
        'where it lives. ffmpeg is required only for --render; nothing else in the '
        'monitor uses it.')


def ffmpeg_command(ffmpeg: str, output: Path, source: Path, width: int, height: int,
                   gain_db: float = 0.0) -> list[str]:
    """The full argument list for a render, as a value rather than a side effect.

    Kept a plain function of its inputs so the command can be asserted on directly,
    without running anything.  The settings it bakes in were chosen against VLC and
    Firefox as the playback targets:

    - `yuv420p` and Main profile because Firefox will not decode High 4:4:4
      Predictive, which rules out the 4:4:4 that this content — thin scope traces and
      axis text — would otherwise benefit from.  A low CRF buys back some of what
      chroma subsampling costs.
    - `+faststart` because a Firefox target means the file may be served over HTTP,
      and a moov atom at the end forces the whole download before the first frame.
    - `-n` rather than `-y`: an existing output is refused, and refused without
      prompting, since a prompt on a pipe would simply hang.  The caller checks first
      so the operator gets a better message than ffmpeg's; this is the backstop for
      the file appearing in between.
    """
    gain = ('-af', f'volume={gain_db}dB') if gain_db else ()
    return [
        ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n',
        # Input 0: our frames, already conformed to a constant rate.  rawvideo carries
        # no timestamps, which is exactly why the conform happens on our side of the
        # pipe — see _FrameGrid — rather than being left to ffmpeg's -fps_mode.
        '-f', 'rawvideo', '-pix_fmt', PIXEL_FORMAT,
        '-s', f'{width}x{height}', '-r', str(FRAME_RATE), '-i', 'pipe:0',
        # Input 1: the recording itself, untouched, on its own timeline.
        '-i', str(source),
        '-c:v', 'libx264', '-preset', _PRESET, '-crf', str(_CRF),
        '-profile:v', _PROFILE, '-pix_fmt', 'yuv420p', '-g', str(_KEYFRAME_INTERVAL),
        '-c:a', 'aac', '-b:a', _AUDIO_BITRATE, '-ar', str(_AUDIO_RATE), *gain,
        # Carry the recording's own tags into the container: what the event was, the
        # station, when it happened, and the calibration the numbers were measured
        # against.  Input 1 is the .wav, which is where they live; without this the
        # rendered file arrives at whoever is shown it with no idea what it is.
        '-map_metadata', '1',
        # ...but not its chapters.  buzz.wavmeta marks the moment of lock with a `cue`
        # chunk, ffmpeg reads that as a chapter, and the mp4 muxer writes chapters as
        # a *track* -- so a single marker became a third stream of type bin_data
        # sitting alongside the video and audio.  The lock offset is already in the
        # comment tag as lead_in_seconds, so nothing is lost by dropping it.
        '-map_chapters', '-1',
        '-movflags', '+faststart',
        # Guards against the two streams disagreeing about the length by a frame or
        # two; without it the file ends with whichever ran longer, silent or frozen.
        '-shortest',
        str(output),
    ]


def _nearest_slot(seconds: float, frame_rate: int) -> int:
    """Which grid slot a moment belongs to: nearest, and not banker's rounding.

    Truncating instead would be the obvious thing and is wrong twice over.  It biases
    every frame early by half a slot on average — a systematic 17 ms of video leading
    audio, which is precisely the kind of constant offset this design went to trouble
    to avoid — and it is at the mercy of binary representation, where 0.3 * 30 is
    8.999999999999998 and truncates to 8, dropping a frame from a perfectly ordinary
    tenth of a second.  Nearest halves the worst-case error to half a slot and leaves
    it unbiased.

    `floor(x + 0.5)` rather than `round()` because round() rounds half to even, so
    round(0.5) is 0 and round(1.5) is 2 — the same reason the analyzer avoids it when
    staying in phase.
    """
    return math.floor(seconds * frame_rate + 0.5)


class _FrameGrid:
    """Places captured frames onto the constant-rate output grid.

    The display paints when its timer fires, which is neither exactly 10 fps nor
    aligned to anything; the file needs frames at exact multiples of 1/30 s.  Something
    has to reconcile the two, and doing it here rather than handing ffmpeg timestamped
    frames keeps the arithmetic somewhere it can be read and tested — rawvideo over a
    pipe has no timestamps to hand it anyway.

    The rule is that a frame captured at audio position `t` shows what was on screen
    up to `t`, so every grid slot before `t` is filled with the frame *before* it.  A
    slot is therefore never filled with a picture from its own future, which is the
    failure that would show up as the video running ahead of the sound.
    """

    def __init__(self, frame_rate: int = FRAME_RATE) -> None:
        self._frame_rate = frame_rate
        self._filled = 0

    @property
    def filled(self) -> int:
        """Grid slots written so far, which is also the next slot's index."""
        return self._filled

    def slots_before(self, position: float) -> int:
        """How many copies of the current frame precede one captured at `position`.

        Zero is a normal answer: the display runs at 10 fps against a 30 fps grid, so
        roughly two captures in three land in a slot that is already accounted for.
        """
        target = _nearest_slot(position, self._frame_rate)
        if target <= self._filled:
            # Position can fail to advance between two captures, and can go backwards
            # if playback is restarted.  Neither is a reason to write a negative number
            # of frames, and neither should rewind a file already partly written.
            return 0
        count = target - self._filled
        self._filled = target
        return count

    def slots_to_fill(self, duration: float) -> int:
        """Slots still needed for the file to last `duration`, marking them filled.

        The last captured frame is held for whatever is left.  Without this the video
        stops at the final capture and ends slightly short of its own audio.

        Consumes rather than reports, exactly as slots_before() does — the count is
        being written by the caller, so the grid has to know it has been.  Leaving
        `filled` behind here would understate the finished file by its whole tail.
        """
        target = _nearest_slot(duration, self._frame_rate)
        if target <= self._filled:
            return 0
        count = target - self._filled
        self._filled = target
        return count


class RenderSession:
    """An ffmpeg process being fed frames, and the grid deciding how many to feed.

    Frames go in as they are captured, each stamped with where playback had reached
    when its pixels were read.  Reading the two together is what preserves the
    analyzer's lookback — the display genuinely is showing a moment slightly past, and
    a render that corrected for that would no longer match the program.
    """

    def __init__(self, output: Path, source: Path, width: int, height: int,
                 gain_db: float = 0.0, ffmpeg: str | None = None) -> None:
        self.output = Path(output)
        self.source = Path(source)
        self.width, self.height = width, height
        self._expected_bytes = width * height * _BYTES_PER_PIXEL
        self._command = ffmpeg_command(find_ffmpeg(ffmpeg), self.output, self.source,
                                       width, height, gain_db)
        self._grid = _FrameGrid()
        self._process: subprocess.Popen | None = None
        self._previous: bytes | None = None

    @property
    def frames_written(self) -> int:
        return self._grid.filled

    def start(self) -> None:
        """Launch ffmpeg and open the pipe.  Refuses to overwrite an existing file."""
        if self.output.exists():
            raise RenderError(
                f'Cannot start rendering: {self.output} already exists. Renders never '
                'overwrite, because one takes as long as the recording it replays and '
                'silently destroying a previous one would be an expensive mistake. '
                'Delete it or pass a different --render filename.')
        logger.info('Rendering to %s — %dx%d at %d fps', self.output,
                    self.width, self.height, FRAME_RATE)
        # Logged in full, and at info: this is the one thing worth having in the log
        # when a render comes out wrong, because it can be pasted into a terminal
        # and run by hand.
        logger.info('ffmpeg command: %s', ' '.join(self._command))
        try:
            self._process = subprocess.Popen(
                self._command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        except OSError as exc:
            raise RenderError(
                f'Could not start ffmpeg at {self._command[0]}: {exc}. The file was '
                'found but the operating system refused to run it, which usually '
                'means it is not executable, is a broken symlink, or is built for a '
                'different architecture. Try running it directly from a terminal — '
                'the error there is normally more specific than this one.') from exc

    def submit(self, pixels: bytes, position: float) -> None:
        """Offer the frame on screen at playback `position`, in PIXEL_FORMAT bytes."""
        if len(pixels) != self._expected_bytes:
            raise RenderError(
                f'Cannot add a frame to the render: got {len(pixels)} bytes where '
                f'{self._expected_bytes} were expected for a {self.width}x'
                f'{self.height} frame in {PIXEL_FORMAT}. Raw video has no framing of '
                'its own, so a short frame would not be detected by ffmpeg — it would '
                'shift every following row and turn the video into diagonal mush. '
                'The usual cause is the window being a different size from the one '
                'this session was built for, or pixels arriving in a format other '
                f'than {PIXEL_FORMAT}; see MainWindow.frame_bytes().')
        if self._previous is None:
            # Nothing has been on screen before this one, so it stands in for whatever
            # came before it and the slots up to `position` are filled with it.
            #
            # Normally that is no slots at all, because the caller captures once
            # before starting the transport.  It matters when startup was slow enough
            # that playback has already moved -- a cold start compiling the FFT path
            # can hold the event loop for over a second -- and the choice there is
            # between a still opening frame and a video that begins late while its
            # audio begins on time.  Silently losing the first second and desynching
            # everything after it is the worse of the two, and it is exactly what
            # happened before this existed.
            self._previous = pixels
        self._write(self._grid.slots_before(position))
        self._previous = pixels

    def finish(self, duration: float) -> None:
        """Hold the last frame out to `duration`, then close the pipe and wait.

        Waits for ffmpeg rather than abandoning it: the moov atom is written at the
        end, and +faststart then rewrites the file to move it to the front, so a
        process killed here leaves an .mp4 that no player will open.
        """
        if self._process is None:
            return
        self._write(self._grid.slots_to_fill(duration))
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except OSError:
            pass                      # ffmpeg already gone; returncode below explains
        code = self._process.wait()
        self._process = None
        if code != 0:
            raise RenderError(
                f'ffmpeg exited with status {code} while finishing {self.output}, so '
                'the file is either missing or unplayable. ffmpeg logs its own reason '
                'to stderr, which appears above this message and is the thing to read '
                'first. The command that was run is in the log at INFO, and can be '
                'pasted into a terminal to reproduce it by hand.')
        logger.info('Rendered %d frames to %s', self._grid.filled, self.output)

    def _write(self, count: int) -> None:
        """Write `count` copies of the frame currently on screen."""
        if count <= 0 or self._previous is None or self._process is None:
            return
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(self._previous * count)
        except (BrokenPipeError, OSError) as exc:
            raise RenderError(
                f'ffmpeg stopped accepting frames part-way through {self.output}: '
                f'{exc}. This means the encoder exited while the render was still '
                'running — most often because it rejected the arguments it was given, '
                'or ran out of disk. It explains itself on stderr, which appears just '
                'above this message. The partly-written file is not usable and should '
                'be deleted before trying again.') from exc

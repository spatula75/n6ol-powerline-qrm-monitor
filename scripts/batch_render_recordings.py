"""
Render every recording in the recording directory to an .mp4, one file each.

This exists to make a pile of event recordings watchable.  The monitor writes a .wav
per event and there is no way to tell an interesting one from a dull one by its name
or its size, so the practical way to find the interesting ones is to render the lot
and skim the videos.

Usage:
    python scripts/batch_render_recordings.py
    python scripts/batch_render_recordings.py --max-length 20
    python scripts/batch_render_recordings.py --jobs 4
    python scripts/batch_render_recordings.py --recordings /path/to/wavs --limit 5

Each render is one `python -m buzz.main --playback ... --render ...` run, exactly what
an operator would type for a single recording.  Videos are written to a renders/
subdirectory of the recording directory, each named after the .wav it came from.

-------------------------------------------------------------------------------
What this costs, and why it is not faster
-------------------------------------------------------------------------------

A render plays the recording through in real time, because the display it captures is
drawn from audio arriving on a clock.  The batch therefore takes at least as long as
the recordings do end to end, plus about 15 seconds per file for startup, the loudness
probe, and the encoder finishing.  There is no way around the real-time floor for a
single render.

--jobs runs several at once, which does divide the wall time, and defaults to 1 because
the safe number depends on the machine.  Each job is a separate monitor process with
its own analysis thread and its own encoder, so four jobs is roughly four cores' worth
of work.  Somebody with the cores should use them.  Somebody who wants the machine to
stay usable should not.

-------------------------------------------------------------------------------
Interruptions
-------------------------------------------------------------------------------

An .mp4 that already exists is skipped rather than re-rendered, so a batch stopped
part-way can simply be started again and picks up where it left off.  That also means
a failed render must not leave a file behind, or the retry would skip a recording that
never rendered, so a failure deletes whatever the child wrote.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'lib'))

from buzz.config import CONFIG_PATH, BuzzConfig  # noqa: E402

# A subdirectory of the recording directory, rather than the directory itself, so that
# a second batch does not treat its own output as input.
DEFAULT_OUTPUT_NAME = 'renders'

# A render runs in real time, so a recording's own length is the floor.  Several times
# that, plus a fixed allowance, covers the loudness probe, numba's first compile, and a
# slow disk, without leaving a hung child to sit there for the rest of the afternoon.
TIMEOUT_FACTOR = 4.0
TIMEOUT_FLOOR_S = 60.0

# How much of a failed child's output to quote.  Enough for the exception and its
# message, short enough that a batch of failures stays readable.
_FAILURE_TAIL_LINES = 4


@dataclass(frozen=True)
class Job:
    """One recording and the video it is to become."""

    source: Path
    output: Path
    seconds: float

    @property
    def timeout_s(self) -> float:
        return self.seconds * TIMEOUT_FACTOR + TIMEOUT_FLOOR_S


@dataclass(frozen=True)
class Outcome:
    """What became of one job."""

    job: Job
    ok: bool
    elapsed_s: float
    message: str = ''


def recording_seconds(path: Path) -> float:
    """Length of a .wav in seconds, or 0.0 if the header cannot be read.

    A file that cannot be read here is not refused: the renderer itself reports a bad
    file far better than this can, naming what is wrong with it.  Returning zero sends
    it through with the shortest timeout, which is the right treatment for something
    that is going to fail immediately anyway.
    """
    try:
        with wave.open(str(path), 'rb') as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (OSError, wave.Error, ZeroDivisionError):
        return 0.0


def plan(sources: list[Path], output_dir: Path,
         max_length_s: float | None) -> tuple[list[Job], list[tuple[Path, str]]]:
    """Split the recordings into jobs to run and files to skip, with a reason each.

    Separated from running them so that what the batch is about to do can be decided,
    and tested, without starting a single render.  The skip reasons are returned rather
    than printed for the same reason.
    """
    jobs: list[Job] = []
    skipped: list[tuple[Path, str]] = []
    for source in sources:
        output = output_dir / f'{source.stem}.mp4'
        seconds = recording_seconds(source)
        if max_length_s is not None and seconds > max_length_s:
            skipped.append((source, f'{seconds:.0f} s, longer than the {max_length_s:.0f} s limit'))
        elif output.exists():
            skipped.append((source, 'already rendered'))
        else:
            jobs.append(Job(source, output, seconds))
    return jobs, skipped


def render_command(job: Job) -> list[str]:
    """The command line for one render.

    Headless on purpose.  A windowed render is right when somebody is watching one go
    by, but a batch would throw a window in front of whoever started it once per
    recording, and --render --headless paints the same frames offscreen.  It also
    implies --mute, so the batch does not seize the speakers for an hour.
    """
    return [sys.executable, '-m', 'buzz.main',
            '--headless',
            '--playback', str(job.source),
            '--playback-gain', 'auto',
            '--render', str(job.output)]


def child_environment() -> dict[str, str]:
    """The environment a child render runs in.

    PYTHONPATH because lib/ is not an installed package, and the offscreen platform
    because a headless render must not need a display server.  main.py sets that
    itself, and setting it here as well costs nothing and keeps this script honest if
    that ever moves.

    lib/ is prepended to whatever PYTHONPATH the operator already had rather than
    replacing it.  A value they set deliberately would otherwise vanish in every child,
    and the failure would surface as an import error from inside a process whose
    environment nobody chose.
    """
    library_path = str(_REPO_ROOT / 'lib')
    inherited = os.environ.get('PYTHONPATH')
    return {**os.environ,
            'PYTHONPATH': f'{library_path}{os.pathsep}{inherited}' if inherited else library_path,
            'QT_QPA_PLATFORM': 'offscreen'}


def render(job: Job) -> Outcome:
    """Render one recording, and never raise.

    A batch that stops on the first bad file is worse than useless: the operator comes
    back to an hour of nothing.  Every failure therefore becomes an Outcome, and the
    caller decides what to say about it.

    The catch-all is what makes that promise true rather than merely intended.  A
    render can fail in ways that are nobody's fault and nothing to do with the
    recording: an OSError from spawning the child, a PermissionError from cleaning up
    after it.  Any of those escaping here would travel up through the pool and out of
    main() as a traceback, discarding every outcome already collected.
    """
    started = time.monotonic()
    try:
        return _attempt_render(job, started)
    except Exception as exc:
        return Outcome(job, False, time.monotonic() - started,
                       f'The render of {job.source.name} failed with an unexpected '
                       f'error: {exc}.  The usual causes are a missing python '
                       f'interpreter, a recording the batch cannot read, and an output '
                       f'directory it cannot write.  This recording counts as failed '
                       f'and the rest of the batch continues.'
                       + _discard_partial_output(job))


def _attempt_render(job: Job, started: float) -> Outcome:
    """Run one child render through to an exit, a timeout, or a failure.

    The child's output goes to a temporary file rather than a pipe, and that is what
    makes the timeout dependable.  With a pipe, subprocess handles a timeout by killing
    the direct child and then waiting for both pipes to close, and ffmpeg inherits the
    same stderr from the render it runs inside.  A stalled ffmpeg is the likeliest
    reason for a hang in the first place, and killing python does not free it, so the
    wait meant to bound a hung render would never return.  A file has no such wait, so
    kill() and wait() come back at once.
    """
    with tempfile.TemporaryFile() as captured:
        process = subprocess.Popen(render_command(job), cwd=str(_REPO_ROOT),
                                   env=child_environment(),
                                   stdout=captured, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=job.timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return Outcome(job, False, time.monotonic() - started,
                           f'no exit within {job.timeout_s:.0f} s'
                           + _discard_partial_output(job))
        elapsed = time.monotonic() - started
        if returncode == 0 and job.output.exists():
            return Outcome(job, True, elapsed)
        reported = (_output_tail(captured)
                    or f'exit code {returncode}, and no file was written')
        return Outcome(job, False, elapsed, reported + _discard_partial_output(job))


def _output_tail(captured: BinaryIO) -> str:
    """The last few lines the child wrote, for quoting back in a failure."""
    captured.seek(0)
    text = captured.read().decode('utf-8', errors='replace').strip()
    return '\n'.join(text.splitlines()[-_FAILURE_TAIL_LINES:])


def _discard_partial_output(job: Job) -> str:
    """Remove what a failed render wrote, so a later run does not skip the recording.

    Returns a note to append to the failure message when the file could not be removed,
    and never raises.  unlink(missing_ok=True) swallows only FileNotFoundError, so a
    video the operator happens to have open in a player raises PermissionError on
    Windows, as does a read-only output directory anywhere.  Cleanup that throws would
    turn one awkward file into a dead batch.
    """
    try:
        job.output.unlink(missing_ok=True)
        return ''
    except OSError as exc:
        return (f'\n    The partial video {job.output} could not be removed: {exc}.  '
                f'A later run will skip {job.source.name}, because a file for it now '
                f'exists.  Delete that file before the next batch.')


def run_jobs(jobs: list[Job], workers: int,
             announce: Callable[[str], None] = print) -> list[Outcome]:
    """Run the jobs, at most `workers` at a time, reporting each as it finishes.

    Threads rather than processes, because the work itself is already in a child
    process: each thread does nothing but wait on one.

    Results are reported as they complete rather than in order, which is why this
    submits and gathers with as_completed rather than using Executor.map.  map yields
    strictly in submission order, so one long recording at the head of the list holds
    back the report of every short one behind it, and a failure stays invisible until
    its turn arrives.  With --jobs 4 over a directory whose first file is a ten minute
    recording, that meant ten minutes of silence with three renders already done.

    The cancellation is not decoration.  Executor.map closes its generator on the way
    out and drops whatever it has not started, and submitting by hand gives that up, so
    the pool's own shutdown would sit and wait for every queued render instead.
    Cancelling explicitly keeps Ctrl-C meaning what it meant before.
    """
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(render, job) for job in jobs]
        try:
            for future in as_completed(futures):
                outcomes.append(future.result())
                announce(_progress_line(outcomes[-1], len(outcomes), len(jobs)))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return outcomes


def _progress_line(outcome: Outcome, done: int, total: int) -> str:
    name = outcome.job.source.name
    if outcome.ok:
        return f'[{done}/{total}] {name}: rendered in {outcome.elapsed_s:.0f} s'
    return f'[{done}/{total}] {name}: FAILED after {outcome.elapsed_s:.0f} s\n    {outcome.message}'


def summary(outcomes: list[Outcome], skipped: list[tuple[Path, str]],
            output_dir: Path, elapsed_s: float) -> str:
    """The closing report, which is all somebody who walked away will read."""
    failures = [o for o in outcomes if not o.ok]
    lines = [f'\n{len(outcomes) - len(failures)} rendered, {len(skipped)} skipped, '
             f'{len(failures)} failed, in {elapsed_s / 60:.1f} min']
    lines += [f'  {o.job.source.name}: {o.message.splitlines()[0] if o.message else "unknown"}'
              for o in failures]
    lines.append(f'Videos are in {output_dir}')
    return '\n'.join(lines)


def _positive(text: str) -> int:
    """An argparse type for a count that has to be at least one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'{text} is not a count of one or more')
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render every recording to an .mp4 so they can be watched quickly.')
    parser.add_argument('--recordings', type=Path, default=None, metavar='DIR',
                        help='Directory of .wav recordings. Defaults to the recording '
                             'directory this station is configured with.')
    parser.add_argument('--output-dir', type=Path, default=None, metavar='DIR',
                        help=f'Where to write the videos. Defaults to a '
                             f'{DEFAULT_OUTPUT_NAME}/ subdirectory of the recording '
                             f'directory.')
    parser.add_argument('--max-length', type=float, default=None, metavar='SECONDS',
                        help='Skip any recording longer than this. Renders every '
                             'recording whatever its length when not given.')
    parser.add_argument('--jobs', type=_positive, default=1, metavar='N',
                        help='How many renders to run at once. Each one is a separate '
                             'monitor process, so this is about a core apiece.')
    parser.add_argument('--limit', type=_positive, default=None, metavar='N',
                        help='Render at most this many recordings, for a trial run.')
    return parser.parse_args(argv)


def default_recordings_directory() -> Path:
    """The recording directory this station is configured with."""
    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()
    return config.recording.directory_path(config.station)


def _report_progress_promptly() -> None:
    """Line-buffer stdout, so a redirected batch reports as it goes.

    Python line-buffers stdout only when it is a terminal.  To a file or a pipe it
    buffers about 8 kB, and a batch prints one short line per finished render, so
    `... > batch.log` would show nothing for hours and then everything at once.  An
    hours-long batch is exactly the thing somebody redirects and then tails.

    Guarded because stdout is not always a real stream.  A test that captures it, or a
    caller that replaced it, can supply something with no reconfigure at all, and
    losing the buffering is not worth an exception.
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)


def main(argv: list[str] | None = None) -> int:
    _report_progress_promptly()
    args = _parse_args(argv)
    recordings = args.recordings or default_recordings_directory()
    output_dir = args.output_dir or recordings / DEFAULT_OUTPUT_NAME

    sources = sorted(recordings.glob('*.wav'))
    if not sources:
        print(f'No .wav recordings in {recordings}.  Point --recordings at the '
              'directory holding them, or check that the monitor is recording.')
        return 1

    jobs, skipped = plan(sources, output_dir, args.max_length)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    for source, reason in skipped:
        print(f'{source.name}: {reason}')
    if not jobs:
        print(f'Nothing to render.  {len(skipped)} recording(s) were skipped.')
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Rendering {len(jobs)} recording(s) into {output_dir}, {args.jobs} at a time.')
    started = time.monotonic()
    outcomes = run_jobs(jobs, args.jobs)
    print(summary(outcomes, skipped, output_dir, time.monotonic() - started))
    return 1 if any(not o.ok for o in outcomes) else 0


if __name__ == '__main__':  # pragma: no cover
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(1)

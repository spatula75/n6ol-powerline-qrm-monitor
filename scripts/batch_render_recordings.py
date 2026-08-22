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
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
    """
    return {**os.environ,
            'PYTHONPATH': str(_REPO_ROOT / 'lib'),
            'QT_QPA_PLATFORM': 'offscreen'}


def render(job: Job) -> Outcome:
    """Render one recording, and never raise.

    A batch that stops on the first bad file is worse than useless: the operator comes
    back to an hour of nothing.  Every failure therefore becomes an Outcome, and the
    caller decides what to say about it.
    """
    started = time.monotonic()
    try:
        finished = subprocess.run(render_command(job), cwd=str(_REPO_ROOT),
                                  env=child_environment(), capture_output=True,
                                  text=True, timeout=job.timeout_s)
    except subprocess.TimeoutExpired:
        _discard_partial_output(job)
        return Outcome(job, False, time.monotonic() - started,
                       f'no exit within {job.timeout_s:.0f} s')
    elapsed = time.monotonic() - started
    if finished.returncode == 0 and job.output.exists():
        return Outcome(job, True, elapsed)
    _discard_partial_output(job)
    tail = (finished.stderr or finished.stdout or '').strip().splitlines()
    return Outcome(job, False, elapsed,
                   '\n'.join(tail[-_FAILURE_TAIL_LINES:])
                   or f'exit code {finished.returncode}, and no file was written')


def _discard_partial_output(job: Job) -> None:
    """Remove what a failed render wrote, so a later run does not skip the recording."""
    job.output.unlink(missing_ok=True)


def run_jobs(jobs: list[Job], workers: int, announce=print) -> list[Outcome]:
    """Run the jobs, at most `workers` at a time, reporting each as it finishes.

    Threads rather than processes, because the work itself is already in a child
    process: each thread does nothing but wait on one.  Results are reported as they
    complete rather than in order, since with more than one worker there is no order
    to preserve.
    """
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(render, jobs):
            outcomes.append(outcome)
            announce(_progress_line(outcome, len(outcomes), len(jobs)))
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


def main(argv: list[str] | None = None) -> int:
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

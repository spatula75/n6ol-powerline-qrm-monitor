"""
Render a recent recording at every sample rate the monitor supports, and check
that each result actually holds a picture and a sound - not merely a
well-formed container.

Run this as part of preparing a release (see CONTRIBUTING.md's release
procedure). It is deliberately not wired into CI: CI has no live radio and
cannot produce a recording that means anything, so this stays a manual,
hands-on check like the release procedure's other verification against real
hardware.

Usage:
    python tools/release_render_check.py
    python tools/release_render_check.py --source recordings/event-....wav
    python tools/release_render_check.py --max-age-days 3 --clip-seconds 20

-------------------------------------------------------------------------------
What "recent" is for
-------------------------------------------------------------------------------

The point of this check is confidence about what the current build does with a
real capture, not with an arbitrary one. A recording from months ago says
nothing about whether today's receiver, band conditions, or config still
produce something the renderer handles correctly - and its mere existence begs
the question of whether the monitor has actually been recording lately at all,
which is itself worth knowing before a release goes out. So rather than pick
silently, this asks the release engineer when nothing recent enough is on hand,
the same way the monitor asks rather than guesses when a setting is ambiguous.

-------------------------------------------------------------------------------
What gets checked
-------------------------------------------------------------------------------

The chosen recording is trimmed to a short clip and resampled to
buzz.config.MIN_SAMPLE_RATE and MAX_SAMPLE_RATE plus a handful of rates in
between - the same "what a file from another operator's sound card looks
like" set tests/integration/test_render_end_to_end.py's FOREIGN_RATES uses.
Each variant is rendered exactly as --render does it, and the result is
checked four ways, all via ffmpeg/ffprobe filters rather than pixel-by-pixel
inspection - cursory, deliberately: this catches "nothing was painted" and
"the render silently produced a blank or frozen file", not "the waterfall is a
pixel off". Seeing the picture is right still wants a person watching a real
render; see CLAUDE.md's "Don't overdrive your headlights".

  1. blackdetect  - flags a sustained black frame: a blank or frozen render.
  2. volumedetect - flags a silent or near-silent audio track.
  3. frame count  - the file's own decoded frame count against what
                    buzz.render itself logged, so an encoder that quietly
                    dropped or duplicated frames does not pass unnoticed.
  4. signalstats  - per-frame mean luma (YAVG). Its range and stddev over the
                    whole clip distinguish real motion (the waterfall
                    scrolling, the scope tracing, the meter bars moving) from
                    a static or duplicated picture - the specific "frozen
                    opening frame" bug class this project has hit before.
"""

import argparse
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'lib'))

from buzz.config import CONFIG_PATH, BuzzConfig  # noqa: E402
from buzz.ffmpeg import FfmpegError, find_ffmpeg, run  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Rates a file from another operator's sound card is plausibly recorded at -
# the same set tests/integration/test_render_end_to_end.py's FOREIGN_RATES
# uses. Kept as its own copy here rather than imported: a release-readiness
# tool and a unit test are read by different people for different reasons,
# even though the number they need happens to agree.
_FOREIGN_RATES = (8000, 11025, 22050, 44100)

_DEFAULT_MAX_AGE_DAYS = 7.0
_DEFAULT_CLIP_SECONDS = 16.0
# Near-silence threshold for volumedetect's mean_volume, in dB.  Well below any
# recorded event -- these average around -30 dB in practice -- so this only
# flags a track that is effectively dead, not merely a quiet one.
_SILENCE_FLOOR_DB = -60.0
# Minimum acceptable spread in per-frame mean luma (signalstats YAVG) across a
# whole clip.  A real render's waterfall, scope, and meter bars all move over
# 16 s; anything under this is a static or duplicated picture.
_MIN_LUMA_RANGE = 0.5


# --------------------------------------------------------------------- source selection

def newest_recording(directory: Path) -> Path | None:
    """The most recently modified .wav in `directory`, or None if there isn't one."""
    candidates = sorted(directory.glob('*.wav'), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def age_days(path: Path, now: datetime | None = None) -> float:
    """How long ago `path` was last modified, in days."""
    now = now or datetime.now()
    return (now - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 86400.0


def describe_staleness(directory: Path, newest: Path | None, max_age_days: float,
                       now: datetime | None = None) -> str:
    """The explanation shown to the release engineer when nothing recent enough exists."""
    if newest is None:
        return (f'No recordings at all were found in {directory}. This check renders a '
                'real capture at every sample rate the monitor supports, so it needs '
                'one to work with, and there is nothing here to validate against.')
    return (f'The newest recording in {directory} is {newest.name}, '
            f'{age_days(newest, now):.1f} days old, older than the {max_age_days:g}-day '
            'freshness this check asks for. An old recording says nothing about '
            'whether the current receiver, band conditions, and code still produce '
            'something the renderer handles correctly.')


def resolve_source(directory: Path, max_age_days: float, explicit: Path | None,
                   prompt: bool = True) -> Path | None:
    """Choose the recording to validate against.

    An explicit path always wins, matching --audio-rf-conversion-db and the rest of
    this program's convention that something named on the command line beats
    anything guessed from the filesystem.  Otherwise the newest recording is used if
    it is fresh enough; if not, the release engineer is asked rather than the check
    proceeding silently against a recording that may no longer represent anything.

    Returns None if the operator chooses to abort - the caller should exit without
    rendering anything in that case.  `prompt=False` is for tests: it skips input()
    and treats "nothing recent enough" the same as "abort", so the interactive loop
    itself is the only thing not covered by a unit test.
    """
    if explicit is not None:
        return explicit

    newest = newest_recording(directory)
    if newest is not None and age_days(newest) <= max_age_days:
        return newest

    print()
    print(describe_staleness(directory, newest, max_age_days))
    print()
    if not prompt:
        return None

    while True:
        offer_proceed = ', [P]roceed with it anyway' if newest else ''
        raw = input(f'[S]pecify a different file{offer_proceed}, or [A]bort? ').strip().lower()
        if raw in ('a', 'abort', ''):
            return None
        if raw in ('p', 'proceed') and newest is not None:
            return newest
        if raw in ('s', 'specify'):
            candidate = Path(input('Path to the recording to use: ').strip())
            if candidate.is_file():
                return candidate
            print(f'  {candidate} is not a file.')
            continue
        print('  Please enter S, A, or P.' if newest else '  Please enter S or A.')


# --------------------------------------------------------------------- clip preparation

def native_sample_rate(path: Path) -> int:
    """The sample rate a .wav's own header declares."""
    with wave.open(str(path), 'rb') as handle:
        return handle.getframerate()


def make_rate_variants(source: Path, clip_seconds: float, output_dir: Path,
                       ffmpeg: str) -> dict[int, Path]:
    """Trim `source` to `clip_seconds` and resample it to every rate under test.

    The recording's own rate is included, trimmed but not resampled, alongside the
    foreign rates - a regression that only shows up at the native rate would
    otherwise go unchecked.  Every output stays RIFF/WAVE PCM: this feeds
    --playback next, which only reads .wav.
    """
    rates = sorted({native_sample_rate(source), *_FOREIGN_RATES})
    variants = {}
    for rate in rates:
        out = output_dir / f'clip-{rate}hz.wav'
        run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'warning', '-i', str(source),
             '-t', f'{clip_seconds}', '-ar', str(rate), '-c:a', 'pcm_s16le',
             '-f', 'wav', str(out)])
        variants[rate] = out
    return variants


# --------------------------------------------------------------------- rendering

def render_variant(wav_path: Path, output_dir: Path, timeout: float = 120.0) -> tuple[Path, bool, str]:
    """Run --render on `wav_path` exactly as an operator would, headless.

    Not wrapped in buzz.ffmpeg.run: this shells out to `python -m buzz.main`, a
    second process of this program, not to ffmpeg directly.
    """
    mp4 = output_dir / f'{wav_path.stem}.mp4'
    log = output_dir / f'{wav_path.stem}.log'
    environment = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
                   'PYTHONPATH': str(_REPO_ROOT / 'lib'), 'NUMBA_DISABLE_JIT': '0'}
    try:
        finished = subprocess.run(
            [sys.executable, '-m', 'buzz.main', '--playback', str(wav_path),
             '--render', str(mp4), '--headless'],
            capture_output=True, text=True, env=environment, cwd=str(_REPO_ROOT),
            timeout=timeout)
    except subprocess.TimeoutExpired:
        return mp4, False, f'did not finish within {timeout:g}s'
    log.write_text((finished.stdout or '') + (finished.stderr or ''))
    if finished.returncode != 0 or not mp4.exists():
        tail = (finished.stderr or finished.stdout or '')[-500:]
        return mp4, False, f'exit code {finished.returncode}: {tail}'
    return mp4, True, ''


def logged_frame_count(log_path: Path) -> int | None:
    """The frame count buzz.render itself reported, parsed back out of its log."""
    if not log_path.exists():
        return None
    match = re.search(r'Rendered (\d+) frames', log_path.read_text())
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------- content checks

def find_ffprobe(ffmpeg_path: str) -> str:
    """ffprobe, found beside the resolved ffmpeg rather than searched for separately.

    Every normal distribution ships them together, so the one binary buzz.ffmpeg
    already found is the more reliable way to the other than a second PATH search
    that could resolve to a mismatched install.
    """
    sibling = Path(ffmpeg_path).with_name(Path(ffmpeg_path).name.replace('ffmpeg', 'ffprobe'))
    if sibling.exists():
        return str(sibling)
    on_path = shutil.which('ffprobe')
    if on_path:
        return on_path
    raise FfmpegError(
        f'ffprobe was not found beside {ffmpeg_path} or on PATH. It ships alongside '
        'ffmpeg in every normal distribution, so a reinstall of ffmpeg usually '
        'restores it.')


def count_black_segments(mp4: Path, ffmpeg: str) -> int:
    output = run([ffmpeg, '-i', str(mp4), '-vf', 'blackdetect=d=0.1:pic_th=0.98',
                  '-an', '-f', 'null', '-'])
    return len(re.findall(r'black_start:', output))


def audio_levels(mp4: Path, ffmpeg: str) -> tuple[float | None, float | None]:
    output = run([ffmpeg, '-i', str(mp4), '-af', 'volumedetect', '-vn', '-f', 'null', '-'])
    mean = re.search(r'mean_volume:\s*(-?[\d.]+)\s*dB', output)
    peak = re.search(r'max_volume:\s*(-?[\d.]+)\s*dB', output)
    return (float(mean.group(1)) if mean else None,
            float(peak.group(1)) if peak else None)


def decoded_frame_count(mp4: Path, ffprobe: str) -> int:
    output = subprocess.run(
        [ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(mp4)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(output)


def luma_series(mp4: Path, ffmpeg: str) -> list[float]:
    output = run([ffmpeg, '-i', str(mp4), '-vf', 'signalstats,metadata=print:file=-',
                  '-f', 'null', '-'])
    return [float(v) for v in re.findall(r'lavfi\.signalstats\.YAVG=([\d.]+)', output)]


@dataclass(frozen=True)
class RenderCheck:
    """Everything found for one rate: whether it rendered, and what is in the result."""

    rate: int
    render_ok: bool
    render_error: str = ''
    frames_logged: int | None = None
    frames_decoded: int | None = None
    black_segments: int = 0
    mean_db: float | None = None
    peak_db: float | None = None
    luma_lo: float = 0.0
    luma_hi: float = 0.0
    luma_stdev: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.render_ok and not self.flags


def flags_for(black_segments: int, mean_db: float | None, frames_logged: int | None,
             frames_decoded: int, luma_lo: float, luma_hi: float) -> list[str]:
    """Which of the four content checks this render fails, in order of severity."""
    flags = []
    if black_segments:
        flags.append(f'{black_segments} black segment(s)')
    if mean_db is not None and mean_db < _SILENCE_FLOOR_DB:
        flags.append(f'near-silent audio ({mean_db:.1f} dB mean)')
    if frames_logged is not None and frames_decoded != frames_logged:
        flags.append(f'frame count mismatch (decoded {frames_decoded}, logged {frames_logged})')
    if luma_hi - luma_lo < _MIN_LUMA_RANGE:
        flags.append(f'low motion (luma range {luma_hi - luma_lo:.3f})')
    return flags


def check_render(rate: int, wav_path: Path, output_dir: Path, ffmpeg: str,
                 ffprobe: str) -> RenderCheck:
    mp4, render_ok, render_error = render_variant(wav_path, output_dir)
    if not render_ok:
        return RenderCheck(rate=rate, render_ok=False, render_error=render_error,
                           flags=[f'render failed: {render_error}'])

    black = count_black_segments(mp4, ffmpeg)
    mean_db, peak_db = audio_levels(mp4, ffmpeg)
    decoded = decoded_frame_count(mp4, ffprobe)
    logged = logged_frame_count(output_dir / f'{wav_path.stem}.log')
    luma = luma_series(mp4, ffmpeg)
    lo, hi = (min(luma), max(luma)) if luma else (0.0, 0.0)
    stdev = statistics.pstdev(luma) if luma else 0.0

    return RenderCheck(
        rate=rate, render_ok=True, frames_logged=logged, frames_decoded=decoded,
        black_segments=black, mean_db=mean_db, peak_db=peak_db, luma_lo=lo, luma_hi=hi,
        luma_stdev=stdev,
        flags=flags_for(black, mean_db, logged, decoded, lo, hi))


# --------------------------------------------------------------------- reporting

def format_report(source: Path, checks: list[RenderCheck]) -> str:
    lines = [f'Source recording: {source}', '']
    width = max(len(f'{c.rate} Hz') for c in checks)
    header = f"{'rate':{width}}  frames       audio(mean/peak dB)   luma(lo-hi, stdev)      flags"
    lines.append(header)
    lines.append('-' * len(header))
    for c in checks:
        if not c.render_ok:
            lines.append(f'{c.rate} Hz'.ljust(width) + f'  FAILED: {c.render_error}')
            continue
        lines.append(
            f"{f'{c.rate} Hz':{width}}  "
            f"{c.frames_decoded:>4}/{c.frames_logged if c.frames_logged is not None else '?'}"
            f"      {c.mean_db:>6.1f} / {c.peak_db:>6.1f}      "
            f"{c.luma_lo:5.2f}-{c.luma_hi:5.2f} ({c.luma_stdev:5.2f})   "
            f"{', '.join(c.flags) or 'ok'}"
        )
    failed = [c for c in checks if not c.ok]
    lines.append('')
    lines.append(f'{len(checks)} rate(s) checked, {len(failed)} flagged.')
    return '\n'.join(lines)


# --------------------------------------------------------------------- entry point

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=None,
                        help='Use this recording instead of the newest one; skips the '
                             'freshness check entirely.')
    parser.add_argument('--max-age-days', type=float, default=_DEFAULT_MAX_AGE_DAYS,
                        help=f'How old the newest recording may be before this check '
                             f'asks what to do (default {_DEFAULT_MAX_AGE_DAYS:g}).')
    parser.add_argument('--clip-seconds', type=float, default=_DEFAULT_CLIP_SECONDS,
                        help=f'Length of the clip taken from the start of the recording '
                             f'(default {_DEFAULT_CLIP_SECONDS:g}).')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where to write clips, renders, and the report (default: a '
                             'timestamped folder under tmp/).')
    args = parser.parse_args()

    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()

    try:
        ffmpeg = find_ffmpeg(config.render.ffmpeg_path or None)
        ffprobe = find_ffprobe(ffmpeg)
    except FfmpegError as exc:
        print(exc)
        return 1

    directory = config.recording.directory_path(config.station)
    source = resolve_source(directory, args.max_age_days, args.source)
    if source is None:
        print('Aborted: nothing rendered.')
        return 1

    output_dir = args.output_dir or (
        _REPO_ROOT / 'tmp' / 'release_render_check' /
        datetime.now().strftime('%Y%m%d-%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        variants = make_rate_variants(source, args.clip_seconds, output_dir, ffmpeg)
    except FfmpegError as exc:
        print(exc)
        return 1

    checks = [check_render(rate, wav, output_dir, ffmpeg, ffprobe)
             for rate, wav in sorted(variants.items())]

    report = format_report(source, checks)
    print()
    print(report)
    (output_dir / 'report.txt').write_text(report + '\n')
    print()
    print(f'Clips, renders, logs, and this report are in {output_dir}')

    return 0 if all(c.ok for c in checks) else 1


if __name__ == '__main__':  # pragma: no cover
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nCancelled.')
        sys.exit(1)

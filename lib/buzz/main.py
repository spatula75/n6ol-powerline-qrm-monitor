"""
Entry point for the powerline QRM monitor.

Loads configuration, wires up the audio pipeline, collector, weather client,
CSV store, plotter, and publisher, then either launches the Qt waterfall
display (default) or runs headlessly (--headless).  --top keeps the waterfall
window always on top of other windows.

--enable-recording arms the event recorder for the run, without editing the
config file; the toolbar button and the R key toggle it while running.

--playback replaces the live audio device with a recorded .wav (see
buzz.playback), which also suppresses everything that writes measurements: no
collector thread, so no CSV rows, no plots, no uploads, and no recording.
Replaying an old event is a way to look at it again, not a way to add fictional
minutes to the day's data.

In GUI mode the Qt event loop runs on the main thread; the collector runs
on a daemon thread.  Closing the window (or ^C) stops the analyzer, then
the audio pipeline, and lets the daemon thread exit with the process.

In headless mode the main thread blocks until ^C, then stops the analyzer
and the pipeline in that same order.
"""

import argparse
import faulthandler
import logging
import logging.config
import os
import signal
import sys
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from buzz import wavmeta
from buzz.analyzer import ContinuousAnalyzer
from buzz.collector import Collector
from buzz.config import CONFIG_PATH, BuzzConfig, validate_sample_rate
from buzz.csv_store import CsvStore
from buzz.playback import FilePlaybackPipeline, resolve_playback_path, sample_rate_of
from buzz.plotter import Plotter
from buzz.publisher import Publisher
from buzz.recorder import EventRecorder
from buzz.sampler import AudioSampler, RingBufferPipeline
from buzz.weather import (
    CumulusMXWeatherClient,
    NullWeatherClient,
    OpenMeteoWeatherClient,
    WeatherClient,
)

if TYPE_CHECKING:
    # Named for the type hints below and imported nowhere at runtime.  Qt and the
    # render code are both loaded lazily inside the functions that need them -- Qt so
    # that --headless needs no display libraries, the render code so that a monitor
    # nobody asked to render cannot fail on ffmpeg -- and a module-scope import here
    # would undo both.
    from PySide6.QtWidgets import QApplication

    from buzz.waterfall import DisplayRecorder, MainWindow

faulthandler.enable()

ROOT_PACKAGE = 'buzz'
# Named explicitly rather than from __name__, which is what every other module here
# does and would be wrong in this one.  This module is the entry point: run as
# `python -m buzz.main`, its __name__ is '__main__', so getLogger(__name__) returns a
# logger outside the buzz hierarchy — and configure_logging() attaches the console
# handler to `buzz` while leaving root at CRITICAL with no handlers.  Every log line in
# this file was therefore discarded in the one invocation the README documents,
# including _adopt()'s warnings that a recording disagrees with the config, which exist
# precisely so that difference cannot pass unseen.
logger = logging.getLogger(f'{ROOT_PACKAGE}.main')

_T = TypeVar('_T')


def configure_logging() -> None:
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'loggers': {
            ROOT_PACKAGE: {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'CRITICAL',
            'handlers': [],
        },
    }
    logging.config.dictConfig(logging_config)


def make_weather_client(config: BuzzConfig) -> WeatherClient:
    """Build the configured weather client, validating its settings up front.

    A misconfigured source degrades to NullWeatherClient with a single startup
    warning, rather than failing (and logging) on every collection cycle.
    """
    weather_config = config.weather
    if weather_config.source == 'openmeteo':
        if weather_config.latitude is None or weather_config.longitude is None:
            logger.warning('Weather source is openmeteo but latitude/longitude are not set; '
                           'weather data disabled.')
            return NullWeatherClient()
        return OpenMeteoWeatherClient(weather_config.latitude, weather_config.longitude)
    if weather_config.source == 'cumulusmx':
        if not weather_config.url:
            logger.warning('Weather source is cumulusmx but url is not set; '
                           'weather data disabled.')
            return NullWeatherClient()
        return CumulusMXWeatherClient(weather_config.url)
    if weather_config.source != 'none':
        logger.warning(
            "Unknown weather source %r — expected 'cumulusmx', 'openmeteo', or 'none'; "
            'weather data disabled.', weather_config.source)
    return NullWeatherClient()


def _adopt(name: str, filename: str, configured: _T, recorded: _T | None) -> _T:
    """Prefer a value the recording carries over the configured one, saying so.

    Silence would be the dangerous option: every one of these changes what the replay
    measures, and a difference the operator never sees is a difference they will
    read straight off the display as a real one.
    """
    if recorded is None or recorded == configured:
        return configured
    logger.warning('%s was recorded with %s %s rather than the configured %s — '
                   'using the value from the file.', filename, name, recorded, configured)
    return recorded


def check_playback_source(path: Path, config: BuzzConfig) -> None:
    """Refuse a file the rest of the run cannot use, from its header alone.

    Checked here rather than deeper down: this is where a file from outside becomes
    something the analyzer and display will be asked to work on, and the only place
    that knows it came from a person rather than from a test.

    Called twice on the way to a render, and deliberately.  main() calls it first, so
    that a rate this program does not work at is refused before the loudness probe has
    read the whole recording for nothing; open_playback_pipeline calls it as well, so
    the guarantee belongs to the function rather than to the order main happens to do
    things in.  Two header reads is not a cost worth reasoning about.
    """
    try:
        validate_sample_rate(sample_rate_of(path), path.name, config.audio.sample_rate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except (OSError, wave.Error) as exc:
        raise SystemExit(f'Cannot play back {path}: {exc}') from exc


def open_playback_pipeline(config: BuzzConfig, name: str, muted: bool = False,
                           gain_db: float = 0.0,
                           rf_conversion_db: float | None = None) -> FilePlaybackPipeline:
    """Open a .wav as the audio source, taking its recorded settings over the config.

    Three settings decide what the analysis of a replayed file means, and none can be
    recovered from the audio: the sample rate (from the format header), the grid's
    pulse rate, and the dB calibration between audio amplitude and level at the
    receiver (both from the file's metadata — see buzz.wavmeta).

    Trusting the file is the only correct choice for all three.  A mismatched sample
    rate resamples nothing and merely mislabels the audio, putting the pulse grid at
    the wrong spacing and quietly costing several dB; a mismatched pulse rate looks
    for a 120 pps train in a 100 pps recording and finds nothing; a mismatched
    calibration reports the whole event at the wrong absolute level.  A recording
    should measure the same wherever it is replayed.

    A file that cannot be played exits with a one-line message rather than a
    traceback: a mistyped filename is an ordinary thing to do from a command line,
    not a bug in the monitor.

    Returns a pipeline that is not yet playing.  The caller starts it once there is
    somewhere to watch it — see FilePlaybackPipeline.start().
    """
    path = resolve_playback_path(name, config.recording.directory_path(config.station))
    check_playback_source(path, config)
    try:
        pipeline = FilePlaybackPipeline(path, muted=muted, gain_db=gain_db)
    except (OSError, wave.Error, ValueError) as exc:
        raise SystemExit(f'Cannot play back {path}: {exc}') from exc

    settings = wavmeta.read_settings(path)
    pulse_rate = wavmeta.setting(settings, 'pulse_rate', int)
    calibration = wavmeta.setting(settings, 'audio_rf_conversion_db', float)

    audio, station = config.audio, config.station
    audio.sample_rate = _adopt(
        'sample rate', path.name, audio.sample_rate, pipeline.sample_rate)
    audio.pulse_rate = _adopt('pulse rate', path.name, audio.pulse_rate, pulse_rate)
    station.audio_rf_conversion_db = _adopt(
        'level calibration', path.name, station.audio_rf_conversion_db, calibration)

    # Any .wav plays, including one this monitor never made.  It just cannot be
    # analysed with any authority, and the operator is the only one who can judge
    # whether that matters — so say what is being assumed rather than fall back
    # silently and let a plausible-looking dBm reading speak for itself.
    # An explicit figure from the command line beats both, because it is the only one
    # anybody deliberately supplied.  Overriding a value the recording carries is
    # unusual enough to say out loud: the file's own calibration is normally the right
    # one, being the receiver's that made it.
    if rf_conversion_db is not None:
        if calibration is not None and calibration != rf_conversion_db:
            logger.warning('%s records a calibration of %.1f dB; using %.1f dB from '
                           '--audio-rf-conversion-db instead.',
                           path.name, calibration, rf_conversion_db)
        else:
            logger.info('Using %.1f dB from --audio-rf-conversion-db to convert audio '
                        'levels to signal levels for this replay.', rf_conversion_db)
        station.audio_rf_conversion_db = rf_conversion_db

    # Two separate warnings rather than one, because they have different consequences
    # and different remedies.  A wrong pulse rate means nothing locks; a wrong
    # calibration means everything locks and every level is wrong.
    if pulse_rate is None:
        # The other grid, not this one: 120 pps *is* 60 Hz, so warning about 60 Hz
        # while configured for 120 would be telling the operator their own setting is
        # the problem.
        other_pps = 100 if audio.pulse_rate == 120 else 120
        logger.warning(
            '%s does not record its pulse rate; analysing at the configured %d pps. '
            'If nothing locks it may be a %d Hz recording; set audio.pulse_rate to %d.',
            path.name, audio.pulse_rate, other_pps // 2, other_pps)
    if calibration is None and rf_conversion_db is None:
        logger.warning(
            '%s does not record its level calibration; using this station\'s %.1f dB. '
            'dBm and S-unit readings may be wrong by any amount, though lock, phase '
            'and burst shape do not depend on it. Pass --audio-rf-conversion-db if you '
            'know the figure at which it was recorded.',
            path.name, station.audio_rf_conversion_db)
    return pipeline


def _start_playback(pipeline: RingBufferPipeline, playing_back: str | None) -> None:
    """Let a replayed file begin, now that there is something to watch it with."""
    if playing_back:
        pipeline.start()


def _start_collector(config: BuzzConfig, analyzer: ContinuousAnalyzer) -> None:
    """Wire up the measurement side of the monitor and run it on a daemon thread.

    Only live audio gets one.  Everything here writes something durable — a CSV row,
    a plot, an upload — and a replayed recording must not add minutes to a day it did
    not happen on.
    """
    store = CsvStore(config)
    collector = Collector(
        config, analyzer, make_weather_client(config), store, Plotter(config, store),
        Publisher(config) if config.server.enabled else None,
    )
    threading.Thread(
        target=collector.collection_loop, daemon=True, name='collector',
    ).start()


def _wait_until_interrupted(pipeline: RingBufferPipeline, analyzer: ContinuousAnalyzer,
                            recorder: EventRecorder | None = None) -> None:
    """Headless main loop: block until ^C, then stop the analyzer and the audio pipeline.

    Stops the analyzer first, mirroring MainWindow.closeEvent() — otherwise the
    analyzer thread's in-flight tick can end up calling into an already-closed
    audio stream during shutdown.  The recorder is stopped in between, so a
    recording in progress is closed while its audio source is still open.
    """
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        if recorder is not None:
            recorder.stop()
        pipeline.close()


def _start_render(args: argparse.Namespace, config: BuzzConfig, window: 'MainWindow',
                  pipeline: FilePlaybackPipeline,
                  app: 'QApplication') -> 'DisplayRecorder':  # pragma: no cover -- needs a live Qt display
    """Wire an .mp4 render onto a playback session, and quit when it finishes.

    Imported here rather than at module scope so that a monitor run without --render
    never loads the render code and never looks for ffmpeg — the same shape as the
    PySide6 import below, and for the same reason: a feature nobody asked for should
    not be able to fail.

    The window is measured rather than told its size.  It was built without the
    control strip, so it is 734x248 instead of 734x284, and a hard-coded frame size
    here would be a second place to keep that in step.
    """
    from buzz.render import RenderError, RenderSession
    from buzz.waterfall import DisplayRecorder

    # Said before anything starts, because a render takes as long as the recording and
    # otherwise looks like a hang -- particularly headless, where there is not even a
    # window moving to show it is alive.
    #
    # The length comes from the pipeline, which read it out of the .wav header when it
    # opened the file.  Not from the loudness probe: --playback-gain 0 skips that
    # entirely, and the operator still deserves to know what they have committed to.
    logger.info('Rendering in real time: this will take about %s, the length of the '
                'recording. The display is drawn as it plays.',
                _describe_duration(pipeline.duration))

    try:
        session = RenderSession(Path(args.render), pipeline.path,
                                window.width(), window.height(),
                                # The gain the pipeline was actually built with, which
                                # is the resolved figure -- args.playback_gain may still
                                # be the word "auto" or None for "decide for me".
                                gain_db=pipeline.gain_db,
                                ffmpeg=config.render.ffmpeg_path or None)
    except RenderError as exc:
        # Before the event loop starts, so this is a clean exit with a message rather
        # than a window that opens and then dies.
        logger.error('%s', exc)
        sys.exit(2)

    recorder = DisplayRecorder(window, pipeline, session)
    finished = False

    def _done() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        if recorder.error is not None:
            app.exit(1)
        else:
            app.exit(0)

    recorder.finished.connect(_done)
    # Closing the window mid-render finishes the file where it stands rather than
    # abandoning ffmpeg: an .mp4 whose moov atom was never written opens in nothing.
    app.aboutToQuit.connect(recorder.stop)
    return recorder


AUTO_GAIN = 'auto'


def _gain_argument(value: str) -> float | str:
    """Parse --playback-gain, which takes a number of dB or the word "auto"."""
    if value.strip().lower() == AUTO_GAIN:
        return AUTO_GAIN
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{value!r} is neither a number of dB nor "auto". Pass a figure such as '
            '12 or -6 to set the gain yourself, or "auto" to have the recording '
            'measured and the gain chosen for you.') from None


def _describe_duration(seconds: float) -> str:
    """A duration a person can act on: "45 s", or "2 min 10 s" once that reads better."""
    if seconds < 120:
        return f'{seconds:.0f} s'
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f'{minutes} min {remainder} s' if remainder else f'{minutes} min'


def _check_render_output(output: Path) -> None:
    """Refuse an existing output before anything expensive happens.

    RenderSession checks this too, but not until it is built -- by which time the
    loudness probe has read the whole recording for nothing.  One stat here saves that,
    and gives the operator the message before a window opens.

    The refusal itself comes from buzz.render so that both checks say the same thing;
    imported here rather than at module scope for the usual reason, that a run without
    --render should never load the render code at all.
    """
    from buzz.render import refuse_existing_output
    refuse_existing_output(output)


def _resolve_gain(args: argparse.Namespace, config: BuzzConfig, source: Path) -> float:
    """Settle on a playback gain before anything is built that depends on it.

    Deliberately early.  The gain is a constructor argument to the playback pipeline
    and is baked into the ffmpeg command as a filter, so it has to be known before
    either exists -- and doing the measurement here means every way a render can be
    refused happens before a window opens or an output file is created.

    Rendering defaults to measuring, because a rendered event sits around -45 LUFS and
    well below a normal listening level -- the calibration deliberately keeps them
    there -- and a video somebody has to strain at is not worth making. Watching
    does not, because that is a different job: the operator is listening live, has the
    volume control to hand, and did not ask to wait for a measurement.
    """
    requested = args.playback_gain
    if requested is None:
        requested = AUTO_GAIN if args.render else 0.0
    if requested != AUTO_GAIN:
        return float(requested)

    from buzz.ffmpeg import find_ffmpeg
    from buzz.loudness import resolve_gain
    return resolve_gain(source, find_ffmpeg(config.render.ffmpeg_path or None))


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description='N6OL Powerline QRM Monitor')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI waterfall display')
    parser.add_argument('--top', action='store_true',
                        help='Keep the waterfall window always on top of other windows')
    parser.add_argument('--enable-recording', action='store_true',
                        help='Arm event recording at startup, regardless of the config file')
    parser.add_argument('--playback', metavar='FILE',
                        help='Replay a recorded .wav instead of listening to the audio '
                             'device. A bare filename is looked up in the recording '
                             'directory. Suppresses CSV, plots, uploads and recording.')
    parser.add_argument('--mute', action='store_true',
                        help='Start playback silent, opening no audio output device')
    parser.add_argument('--playback-gain', type=_gain_argument, default=None,
                        metavar='DB|auto',
                        help='dB of gain for the playback audio only, to make a quiet '
                             'recording audible on small speakers. Does not affect '
                             'the analysis or anything it reports. "auto" measures the '
                             'recording and picks the gain that reaches -23 LUFS '
                             'without letting true peak past -2 dBTP; it needs ffmpeg. '
                             '--render implies auto, so pass a number there (0 for '
                             'none) to override it.')
    parser.add_argument('--audio-rf-conversion-db', type=float, default=None,
                        metavar='DB',
                        help='dB between audio amplitude and signal level at the '
                             'receiver, for this replay only. Recordings this program '
                             'made carry their own and need no help; a .wav from '
                             'anywhere else is analysed with the figure configured for '
                             'this station, which may be nothing like the one at '
                             'which it was recorded. Supply it here if you know it. '
                             'Playback only.')
    parser.add_argument('--render', metavar='FILE.mp4',
                        help='Render the --playback session to an .mp4 instead of just '
                             'watching it: H.264 video of the display with the '
                             'recording as its soundtrack. Hides the transport '
                             'controls and plays through once. Needs ffmpeg, which is '
                             'used for this and nothing else.')
    args = parser.parse_args()

    if args.render and not args.playback:
        parser.error('--render needs --playback: it records a replay of a recording, '
                     'and there is nothing to replay without one.')
    if args.render and args.headless:
        # Not a contradiction: rendering always draws the real widgets, and --headless
        # asks only that nothing appear on screen.  Qt's offscreen platform paints into
        # memory instead of a window, and since the display font is loaded from a file
        # rather than borrowed from the platform, an offscreen render comes out
        # identical to a windowed one.
        #
        # setdefault, so an operator who has chosen a platform themselves keeps it.
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        # Nobody is watching, so nobody is listening either, and opening an output
        # device achieves nothing -- it can fail outright on a machine that has none.
        # The rendered audio is unaffected either way: ffmpeg reads the recording from
        # disk, so muting only ever decided whether the operator heard it being made.
        args.mute = True

    configure_logging()

    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()

    recorder = None
    if args.playback:
        if args.enable_recording:
            logger.warning('--enable-recording is ignored during playback.')
        source = resolve_playback_path(
            args.playback, config.recording.directory_path(config.station))
        # Everything that can refuse a run happens here, before a window opens or a
        # file is created, and in order of what each one costs: a .wav header read,
        # then a stat, then the measurement that has to read the whole recording.
        # A failure now costs a message; the same failure later costs a half-written
        # .mp4 and a window that appears and dies.
        check_playback_source(source, config)
        try:
            if args.render:
                _check_render_output(Path(args.render))
            gain_db = _resolve_gain(args, config, source)
        except RuntimeError as exc:
            logger.error('%s', exc)
            sys.exit(2)
        pipeline = open_playback_pipeline(config, args.playback, muted=args.mute,
                                          gain_db=gain_db,
                                          rf_conversion_db=args.audio_rf_conversion_db)
        analyzer = ContinuousAnalyzer(pipeline, config)
        analyzer.start()
    else:
        # `is not None` rather than `!= 0.0`: the default is None so that "the
        # operator said nothing" can be told apart from "the operator said zero", and
        # None compares unequal to 0.0, which would warn on every live run.
        if args.mute or args.playback_gain is not None:
            logger.warning('--mute and --playback-gain are ignored outside playback; '
                           'the monitor never sends live audio to an output device.')
        if args.audio_rf_conversion_db is not None:
            logger.warning('--audio-rf-conversion-db is ignored outside playback; live '
                           'audio is calibrated by station.audio_rf_conversion_db in '
                           'the config, which is the figure this station was set up '
                           'with. Change it there rather than per run.')
        pipeline = AudioSampler(config).pipeline
        analyzer = ContinuousAnalyzer(pipeline, config)
        # Built whether or not recording is enabled: `enabled` only decides whether it
        # starts armed, and the toolbar has to be able to arm it mid-run either way.
        config.recording.enabled = config.recording.enabled or args.enable_recording
        recorder = EventRecorder(pipeline, analyzer, config)
        # Started only now, because the recorder's state listener has to be registered
        # before the analyzer thread begins publishing state changes.
        analyzer.start()
        recorder.start()
        _start_collector(config, analyzer)

    # A headless *render* still goes down the Qt path: it has widgets to paint, it just
    # paints them offscreen.  Only a headless run with nothing to render skips it.
    if args.headless and not args.render:
        _start_playback(pipeline, args.playback)
        _wait_until_interrupted(pipeline, analyzer, recorder)
        return

    try:
        from PySide6.QtCore import QTimer  # noqa: I001
        from PySide6.QtWidgets import QApplication
        from buzz.waterfall import MainWindow
    except ImportError:
        logger.warning(
            'PySide6 not installed — falling back to headless mode. '
            'Install PySide6 or run with --headless to suppress this warning.'
        )
        _start_playback(pipeline, args.playback)
        _wait_until_interrupted(pipeline, analyzer, recorder)
        return

    app = QApplication(sys.argv)
    window = MainWindow(pipeline, analyzer, config, always_on_top=args.top,
                        recorder=recorder,
                        playback=pipeline if args.playback else None,
                        show_controls=not args.render)
    window.show()

    recording_display = _start_render(args, config, window, pipeline, app) \
        if args.render else None
    # Not before now, and not merely after show() either: show() returns before Qt
    # has finished creating and laying out the window, and audio started in that gap
    # plays to a screen that is not there yet — then breaks up as widget construction
    # holds the GIL away from the feeder.  A zero-delay timer fires on the first pass
    # of the event loop, by which time the window is up and the interpreter is idle.
    if args.playback and recording_display is None:
        QTimer.singleShot(0, pipeline.start)
    if recording_display is not None:
        # The transport is started by the recorder rather than here, so that the
        # opening frame is captured before playback moves — see DisplayRecorder.start.
        QTimer.singleShot(0, recording_display.start)

    # Allow Ctrl+C to close the window cleanly from the console
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # QTimer keeps the Python interpreter ticking so SIGINT can be delivered
    sigint_keepalive = QTimer()
    sigint_keepalive.timeout.connect(lambda: None)
    sigint_keepalive.start(200)

    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    main()

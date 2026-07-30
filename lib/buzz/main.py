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
import signal
import sys
import threading
import wave
from typing import TypeVar

from buzz import wavmeta
from buzz.analyzer import ContinuousAnalyzer
from buzz.collector import Collector
from buzz.config import CONFIG_PATH, BuzzConfig
from buzz.csv_store import CsvStore
from buzz.playback import FilePlaybackPipeline, resolve_playback_path
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

faulthandler.enable()

ROOT_PACKAGE = 'buzz'
logger = logging.getLogger(__name__)

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


def open_playback_pipeline(config: BuzzConfig, name: str) -> FilePlaybackPipeline:
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
    """
    path = resolve_playback_path(name, config.recording.directory_path(config.station))
    try:
        pipeline = FilePlaybackPipeline(path)
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
    missing = [name for name, value in (('pulse rate', pulse_rate),
                                        ('level calibration', calibration)) if value is None]
    if missing:
        logger.warning('%s does not record its %s — analysing with the configured '
                       'pulse rate of %d Hz and calibration of %.1f dB, which may not '
                       'be what it was recorded with.',
                       path.name, ' or '.join(missing),
                       audio.pulse_rate, station.audio_rf_conversion_db)
    return pipeline


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
    args = parser.parse_args()

    configure_logging()

    config = BuzzConfig.from_toml() if CONFIG_PATH.exists() else BuzzConfig()

    recorder = None
    if args.playback:
        if args.enable_recording:
            logger.warning('--enable-recording is ignored during playback.')
        pipeline = open_playback_pipeline(config, args.playback)
        analyzer = ContinuousAnalyzer(pipeline, config)
        analyzer.start()
    else:
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

    if args.headless:
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
        _wait_until_interrupted(pipeline, analyzer, recorder)
        return

    app = QApplication(sys.argv)
    window = MainWindow(pipeline, analyzer, config, always_on_top=args.top,
                        recorder=recorder,
                        playback=pipeline if args.playback else None)
    window.show()

    # Allow Ctrl+C to close the window cleanly from the console
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # QTimer keeps the Python interpreter ticking so SIGINT can be delivered
    sigint_keepalive = QTimer()
    sigint_keepalive.timeout.connect(lambda: None)
    sigint_keepalive.start(200)

    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    main()

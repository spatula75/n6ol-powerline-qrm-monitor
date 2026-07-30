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


def open_playback_pipeline(config: BuzzConfig, name: str) -> FilePlaybackPipeline:
    """Open a .wav as the audio source, reconciling the sample rate with the config.

    A recording carries its own sample rate, and the analyzer derives its pulse
    spacing from the configured one.  Trusting the file is the only correct choice:
    the alternative resamples nothing and merely mislabels the audio, putting the
    pulse grid at the wrong spacing and quietly costing several dB of measured level.

    A file that cannot be played exits with a one-line message rather than a
    traceback: a mistyped filename is an ordinary thing to do from a command line,
    not a bug in the monitor.
    """
    path = resolve_playback_path(name, config.recording.directory_path(config.station))
    try:
        pipeline = FilePlaybackPipeline(path)
    except (OSError, wave.Error, ValueError) as exc:
        raise SystemExit(f'Cannot play back {path}: {exc}') from exc
    if pipeline.sample_rate != config.audio.sample_rate:
        logger.warning('%s was recorded at %d Hz, not the configured %d Hz — '
                       'analysing at the rate in the file.',
                       path.name, pipeline.sample_rate, config.audio.sample_rate)
        config.audio.sample_rate = pipeline.sample_rate
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
                        playback_name=pipeline.path.name if args.playback else None)
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

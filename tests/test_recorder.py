"""Tests for EventRecorder: lead-in, trailer, length cap, event budget, and filenames."""

import re
import struct
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from buzz import __version__, wavmeta
from buzz.analyzer import AnalysisResult, AnalyzerState
from buzz.config import BuzzConfig
from buzz.playback import load_wav
from buzz.recorder import EventRecorder, event_filename, fade_ramp, unique_path
from buzz.sampler import RingBufferPipeline

CHUNK = RingBufferPipeline.CHUNK_SIZE

# One appended chunk is exactly one second of audio at this rate, so every duration
# in these tests can be counted in chunks.
SAMPLE_RATE = CHUNK


class FakeAnalyzer:
    """Stands in for ContinuousAnalyzer, publishing state changes the same way."""

    def __init__(self, state: AnalyzerState = AnalyzerState.SEARCHING) -> None:
        self.state = state
        self._listeners = []
        self._result: AnalysisResult | None = None

    def latest_result(self) -> AnalysisResult | None:
        return self._result

    def publish(self, snr: float, locked: bool = True) -> None:
        """Publish a measurement, the way the analyzer does on every tick."""
        self._result = AnalysisResult(signal_dbm=-98.0 + snr, noise_dbm=-98.0,
                                      snr=snr, locked=locked)

    def add_state_listener(self, listener) -> None:
        self._listeners.append(listener)

    def set_state(self, state: AnalyzerState) -> None:
        self.state = state
        for listener in self._listeners:
            listener(state)

    def lock(self) -> None:
        self.set_state(AnalyzerState.LOCKED)

    def unlock(self) -> None:
        self.set_state(AnalyzerState.SIGNAL_LOST)


FADE = round(EventRecorder.FADE_SECONDS * SAMPLE_RATE)


def _make_config(tmp_path: Path, sample_rate: int = SAMPLE_RATE, **recording) -> BuzzConfig:
    config = BuzzConfig()
    config.audio.sample_rate = sample_rate
    config.station.timezone = 'America/Los_Angeles'
    # An explicit baseline rather than whatever the shipped defaults happen to be:
    # these tests count in chunks, and a default retuned for real 16 kHz audio comes out
    # somewhere entirely different at this rate.  Length capping is off here so that
    # only the tests actually about the cap have to think about it.
    config.recording.directory = str(tmp_path)
    config.recording.enabled = True
    config.recording.max_events = 1
    config.recording.max_seconds = 0.0
    config.recording.stop_after_seconds = 2.0
    for key, value in recording.items():
        setattr(config.recording, key, value)
    return config


def _make_recorder(tmp_path: Path, sample_rate: int = SAMPLE_RATE, **recording):
    """Return (recorder, pipeline, analyzer), none of them running a thread."""
    pipeline = RingBufferPipeline()
    analyzer = FakeAnalyzer()
    config = _make_config(tmp_path, sample_rate, **recording)
    return EventRecorder(pipeline, analyzer, config), pipeline, analyzer


def _feed(pipeline: RingBufferPipeline, seconds: int, value: int = 1000) -> None:
    """Append `seconds` chunks of constant-valued audio to the ring buffer."""
    for _ in range(seconds):
        pipeline._append(np.full(CHUNK, value, dtype=np.int16))


def _wav_files(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob('*.wav'))


def _eventually(predicate, timeout: float = 1.0) -> bool:
    """Wait for something a background thread is expected to get round to."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _durations(tmp_path: Path) -> list[float]:
    return [len(load_wav(p)[0]) / SAMPLE_RATE for p in _wav_files(tmp_path)]


class FakeClock:
    """Stands in for time.monotonic() so a 24-hour cycle takes no time to test."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += minutes * 60


class TestEventFilename:
    def test_matches_expected_shape(self):
        when = datetime(2026, 7, 29, 14, 33, 7, tzinfo=ZoneInfo('America/Los_Angeles'))
        assert event_filename(when) == 'event-20260729-143307-0700.wav'

    def test_offset_follows_daylight_saving(self):
        when = datetime(2026, 1, 15, 14, 33, 7, tzinfo=ZoneInfo('America/Los_Angeles'))
        assert event_filename(when) == 'event-20260115-143307-0800.wav'

    def test_contains_no_characters_illegal_on_windows(self):
        when = datetime(2026, 7, 29, 14, 33, 7, tzinfo=ZoneInfo('UTC'))
        assert not set(event_filename(when)) & set(':*?"<>|')


class TestUniquePath:
    def test_returns_path_unchanged_when_free(self, tmp_path):
        assert unique_path(tmp_path / 'event.wav') == tmp_path / 'event.wav'

    def test_suffixes_when_taken(self, tmp_path):
        (tmp_path / 'event.wav').touch()
        assert unique_path(tmp_path / 'event.wav') == tmp_path / 'event-2.wav'

    def test_keeps_counting_past_the_first_collision(self, tmp_path):
        (tmp_path / 'event.wav').touch()
        (tmp_path / 'event-2.wav').touch()
        assert unique_path(tmp_path / 'event.wav') == tmp_path / 'event-3.wav'


class TestDisarmed:
    def test_lock_records_nothing_while_disarmed(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, enabled=False)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_starts_disarmed_when_config_disabled(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, enabled=False)
        assert recorder.status().armed is False

    def test_starts_armed_when_config_enabled(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path)
        assert recorder.status().armed is True


class TestRecordingDirectory:
    """Created by arming, so a bad path is found while the operator is still there
    rather than at the end of an unattended day that recorded nothing.

    Recording that is off reaches for nothing at all - no stray directory, and no
    complaint about a path it was never going to use - so the check happens at startup
    when armed there, and otherwise when the Record button is pressed."""

    def _blocked(self, tmp_path: Path) -> Path:
        blocked = tmp_path / 'not-a-directory'
        blocked.touch()                 # a file cannot be turned into a directory
        return blocked

    def test_created_when_armed_at_startup(self, tmp_path):
        target = tmp_path / 'deep' / 'recordings'
        _make_recorder(tmp_path, directory=str(target))
        assert target.is_dir()

    def test_not_created_when_recording_is_off(self, tmp_path):
        target = tmp_path / 'recordings'
        _make_recorder(tmp_path, directory=str(target), enabled=False)
        assert not target.exists()

    def test_created_when_armed_by_hand(self, tmp_path):
        target = tmp_path / 'recordings'
        recorder, _, _ = _make_recorder(tmp_path, directory=str(target), enabled=False)
        recorder.arm()
        assert target.is_dir()

    def test_unusable_directory_does_not_raise_at_startup(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, directory=str(self._blocked(tmp_path)))
        assert recorder.status().recording is False

    def test_unusable_directory_starts_disarmed(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, directory=str(self._blocked(tmp_path)))
        assert recorder.status().armed is False

    def test_unusable_directory_is_reported_at_startup(self, tmp_path, caplog):
        with caplog.at_level('ERROR'):
            _make_recorder(tmp_path, directory=str(self._blocked(tmp_path)))
        assert 'Cannot create the recording directory' in caplog.text

    def test_unusable_directory_refuses_to_arm(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, directory=str(self._blocked(tmp_path)))
        recorder.arm()
        assert recorder.status().armed is False

    def test_unusable_directory_starts_no_rearm_cycle(self, tmp_path):
        recorder, _, _ = _make_recorder(
            tmp_path, directory=str(self._blocked(tmp_path)), rearm_reset_minutes=1440)
        assert recorder.status().rearm_in_seconds is None

    def test_an_impossible_sample_rate_starts_disarmed(self, tmp_path):
        """Every duration here is derived by dividing by the sample rate, and wave
        will not write a file without a valid one."""
        recorder, _, _ = _make_recorder(tmp_path, sample_rate=0)
        assert recorder.status().armed is False

    def test_an_impossible_sample_rate_refuses_to_arm(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, sample_rate=0)
        recorder.arm()
        assert recorder.status().armed is False

    def test_an_impossible_sample_rate_is_reported(self, tmp_path, caplog):
        with caplog.at_level('ERROR'):
            _make_recorder(tmp_path, sample_rate=0)
        assert 'sample rate' in caplog.text

    def test_an_impossible_sample_rate_records_nothing(self, tmp_path):
        """Rather than failing once per poll and leaving a stray file behind each time."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, sample_rate=0)
        _feed(pipeline, 2)
        analyzer.lock()
        for _ in range(3):
            recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_a_writer_that_cannot_be_configured_leaves_nothing_open(self, tmp_path):
        """A failure part-way through opening must not publish a half-built writer
        for the next tick to trip over."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        with patch('buzz.recorder.wave.open', side_effect=RuntimeError('bad frame rate')):
            recorder.tick()
        assert recorder.status().recording is False

    def test_a_writer_that_fails_after_opening_is_closed(self, tmp_path):
        """The file is open by the time the header settings are applied, so a failure
        there has something to clean up - and closing it raises in turn, since it
        never got the settings it needs to write a header."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        writer = MagicMock()
        writer.setframerate.side_effect = RuntimeError('bad frame rate')
        writer.close.side_effect = RuntimeError('sampling rate not specified')
        with patch('buzz.recorder.wave.open', return_value=writer):
            recorder.tick()
        writer.close.assert_called_once()

    def test_a_writer_that_cannot_be_configured_disarms(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        with patch('buzz.recorder.wave.open', side_effect=RuntimeError('bad frame rate')):
            recorder.tick()
        assert recorder.status().armed is False

    def test_arming_succeeds_once_the_path_is_fixed(self, tmp_path):
        blocked = self._blocked(tmp_path)
        recorder, _, _ = _make_recorder(tmp_path, directory=str(blocked))
        blocked.unlink()                # operator clears whatever was in the way
        recorder.arm()
        assert recorder.status().armed is True


class TestStartingARecording:
    def test_lock_creates_one_file(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_filename_matches_event_pattern(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        assert re.fullmatch(r'event-\d{8}-\d{6}[+-]\d{4}\.wav', _wav_files(tmp_path)[0].name)

    def test_lead_in_is_everything_buffered_before_the_lock(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 4)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        assert _durations(tmp_path) == [4.0]

    def test_lead_in_audio_is_the_buffered_audio(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 2, value=777)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        samples = load_wav(_wav_files(tmp_path)[0])[0]
        assert np.all(samples[FADE:-FADE] == 777)     # ends are ramped; see TestFades

    def test_wav_sample_rate_matches_the_pipeline(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        assert load_wav(_wav_files(tmp_path)[0])[1] == SAMPLE_RATE

    def test_wav_is_16_bit_mono(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        with wave.open(str(_wav_files(tmp_path)[0]), 'rb') as wav:
            assert (wav.getnchannels(), wav.getsampwidth()) == (1, 2)

    def test_directory_is_created_if_missing(self, tmp_path):
        target = tmp_path / 'nested' / 'recordings'
        recorder, pipeline, analyzer = _make_recorder(tmp_path, directory=str(target))
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        assert len(_wav_files(target)) == 1

    def test_a_lock_that_ends_between_ticks_still_records(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        analyzer.unlock()       # both edges fall inside one poll interval
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_directory_lost_mid_run_disarms_rather_than_raising(self, tmp_path):
        """The startup check cannot cover a directory that goes away later - a
        removable drive, or a tidy-up script - so opening a file still has to cope."""
        target = tmp_path / 'recordings'
        recorder, pipeline, analyzer = _make_recorder(tmp_path, directory=str(target))
        target.rmdir()
        target.touch()                  # a file where the directory used to be
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        assert recorder.status().armed is False


class TestEndingARecording:
    def test_recording_continues_while_locked(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()
        assert recorder.status().recording is True

    def test_stops_after_the_configured_silence(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        recorder.tick()
        analyzer.unlock()
        _feed(pipeline, 2)
        recorder.tick()
        assert recorder.status().recording is False

    def test_trailer_is_kept_in_the_file(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        analyzer.unlock()
        _feed(pipeline, 2)
        recorder.tick()
        assert _durations(tmp_path) == [3.0]      # 1 s lead-in + 2 s trailer

    def test_signal_returning_inside_the_timeout_keeps_one_file(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        recorder.tick()
        analyzer.unlock()
        _feed(pipeline, 1)
        recorder.tick()
        analyzer.lock()
        _feed(pipeline, 1)
        recorder.tick()
        assert recorder.status().recording is True

    def test_stop_finalizes_a_recording_in_progress(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        assert _durations(tmp_path) == [2.0]

    def test_disarm_closes_a_recording_in_progress(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        recorder.disarm()
        assert recorder.status().recording is False


class TestLengthCap:
    def test_cap_truncates_the_recording_exactly(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_seconds=3.0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 10)
        recorder.tick()
        assert _durations(tmp_path) == [3.0]

    def test_cap_excludes_the_lead_in(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_seconds=3.0)
        _feed(pipeline, 4)                        # lead-in, on top of the 3 s cap
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 10)
        recorder.tick()
        assert _durations(tmp_path) == [7.0]

    def test_zero_means_no_cap(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, max_seconds=0.0, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 30)
        recorder.tick()
        assert recorder.status().recording is True

    def test_capped_event_does_not_immediately_restart(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, max_seconds=2.0, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()                           # hits the cap, still locked
        _feed(pipeline, 5)
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_next_event_records_after_the_signal_returns(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, max_seconds=2.0, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()                           # capped
        analyzer.unlock()
        recorder.tick()                           # signal gone: re-arm for a new event
        analyzer.lock()
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 2


class TestFadeRamp:
    def test_starts_at_exactly_zero(self):
        assert fade_ramp(64)[0] == 0.0

    def test_ends_at_exactly_one(self):
        assert fade_ramp(64)[-1] == 1.0

    def test_has_the_requested_length(self):
        assert len(fade_ramp(64)) == 64

    def test_rises_monotonically(self):
        assert np.all(np.diff(fade_ramp(64)) > 0)

    def test_meets_both_ends_with_zero_slope(self):
        """What a raised cosine buys over a linear ramp: no corner at either join,
        so the fade adds no discontinuity of its own at the ends it exists to fix."""
        slope = np.diff(fade_ramp(64))
        assert slope[0] < slope[len(slope) // 2] > slope[-1]

    def test_is_symmetric(self):
        ramp = fade_ramp(64)
        assert ramp == pytest.approx(1.0 - ramp[::-1])

    def test_empty_ramp_is_allowed(self):
        assert len(fade_ramp(0)) == 0


class TestFades:
    """Recorded at 16 kHz, where the fade is a realistic 80 samples."""

    RATE = 16000
    FADE = round(EventRecorder.FADE_SECONDS * RATE)

    def _record(self, tmp_path, chunks=4, value=1000, **recording):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, sample_rate=self.RATE, **recording)
        _feed(pipeline, chunks, value=value)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        return load_wav(_wav_files(tmp_path)[0])[0]

    def test_file_starts_at_exactly_zero(self, tmp_path):
        assert self._record(tmp_path)[0] == 0

    def test_file_ends_at_exactly_zero(self, tmp_path):
        assert self._record(tmp_path)[-1] == 0

    def test_fade_in_is_monotonic(self, tmp_path):
        samples = self._record(tmp_path)
        assert np.all(np.diff(samples[:self.FADE].astype(int)) >= 0)

    def test_fade_out_is_monotonic(self, tmp_path):
        samples = self._record(tmp_path)
        assert np.all(np.diff(samples[-self.FADE:].astype(int)) <= 0)

    def test_audio_reaches_full_scale_by_the_end_of_the_fade(self, tmp_path):
        assert self._record(tmp_path)[self.FADE - 1] == 1000

    def test_fade_out_starts_from_full_scale(self, tmp_path):
        assert self._record(tmp_path)[-self.FADE] == 1000

    def test_the_body_of_the_recording_is_untouched(self, tmp_path):
        samples = self._record(tmp_path)
        assert np.all(samples[self.FADE:-self.FADE] == 1000)

    def test_no_audio_is_lost_to_the_held_back_tail(self, tmp_path):
        samples = self._record(tmp_path, chunks=4)
        assert len(samples) == 4 * CHUNK

    def test_fade_spans_writes_when_the_lead_in_is_short(self, tmp_path):
        """The fade is applied by position, not per write, so a lead-in shorter than
        the fade still ramps across the spans that follow it."""
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, sample_rate=self.RATE, max_events=0)
        pipeline._append(np.full(8, 1000, dtype=np.int16))   # 8-sample lead-in
        analyzer.lock()
        recorder.tick()
        for _ in range(4):
            _feed(pipeline, 1, value=1000)
            recorder.tick()
        recorder.stop()
        samples = load_wav(_wav_files(tmp_path)[0])[0]
        assert np.all(np.diff(samples[:self.FADE].astype(int)) >= 0)

    def test_a_recording_shorter_than_the_fade_still_ends_at_zero(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, sample_rate=self.RATE, max_seconds=0.001)   # 16 samples
        _feed(pipeline, 1, value=1000)
        analyzer.lock()
        recorder.tick()
        recorder.stop()
        samples = load_wav(_wav_files(tmp_path)[0])[0]
        assert (samples[0], samples[-1]) == (0, 0)


class TestMetadata:
    def _record(self, tmp_path, ended='timeout', **recording):
        """Record one event, ending it the way `ended` names."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, **recording)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        if ended == 'timeout':
            analyzer.unlock()
            _feed(pipeline, 2)
            recorder.tick()
        elif ended == 'capped':
            _feed(pipeline, 20)
            recorder.tick()
        elif ended == 'operator':
            recorder.disarm()
        else:
            recorder.stop()
        return _wav_files(tmp_path)[0]

    def test_records_the_most_lead_in_this_config_could_give(self, tmp_path):
        """Without it, lead_in_seconds is censored data dressed as a measurement.

        The buffer slides while min_lock_seconds is waited out, so the lead-in can
        never exceed capacity less that wait. A file sitting exactly at the bound is
        saying "all there was"; one below it is reporting how long the lock took. From
        the file alone those were indistinguishable, because neither the buffer size
        nor min_lock_seconds is recorded anywhere else.
        """
        settings = wavmeta.read_settings(self._record(tmp_path))
        assert 'lead_in_max_seconds' in settings

    def test_the_bound_shrinks_by_the_min_lock_wait(self, tmp_path):
        """Three seconds spent waiting is three seconds of lead-in lost, because the
        buffer slides while the wait runs. The bound therefore has to move with the
        setting rather than being the raw buffer size."""
        plain, _, _ = _make_recorder(tmp_path)
        delayed, _, _ = _make_recorder(tmp_path, min_lock_seconds=1.0)
        lost = plain._max_lead_in_samples() - delayed._max_lead_in_samples()
        assert lost == pytest.approx(SAMPLE_RATE, abs=CHUNK)

    def test_the_bound_never_goes_negative(self, tmp_path):
        """min_lock_seconds longer than the buffer is already clamped elsewhere; this
        makes sure the arithmetic here cannot report a negative allowance if it ever
        stops being."""
        recorder, _, _ = _make_recorder(tmp_path, min_lock_seconds=3600.0)
        assert recorder._max_lead_in_samples() >= 0

    def test_a_saturated_lead_in_can_be_told_from_a_measured_one(self, tmp_path):
        """The whole point: a reader compares the two numbers and knows which they
        have. This recording locks almost immediately, so it is a measurement."""
        settings = wavmeta.read_settings(self._record(tmp_path))
        assert float(settings['lead_in_seconds']) < float(settings['lead_in_max_seconds'])

    def test_names_the_station(self, tmp_path):
        info = wavmeta.read_info(self._record(tmp_path))
        assert info['IART'] == BuzzConfig().station.callsign

    def test_title_describes_the_recording(self, tmp_path):
        info = wavmeta.read_info(self._record(tmp_path))
        assert 'powerline QRM event' in info['INAM']

    def test_records_the_software_version(self, tmp_path):
        info = wavmeta.read_info(self._record(tmp_path))
        assert info['ISFT'].endswith(__version__)

    def test_creation_date_is_an_iso_timestamp_with_an_offset(self, tmp_path):
        info = wavmeta.read_info(self._record(tmp_path))
        assert datetime.fromisoformat(info['ICRD']).tzinfo is not None

    def test_records_the_pulse_rate(self, tmp_path):
        settings = wavmeta.read_settings(self._record(tmp_path))
        assert settings['pulse_rate'] == str(BuzzConfig().audio.pulse_rate)

    def test_records_the_sample_rate(self, tmp_path):
        settings = wavmeta.read_settings(self._record(tmp_path))
        assert settings['sample_rate'] == str(SAMPLE_RATE)

    def test_records_the_level_calibration(self, tmp_path):
        settings = wavmeta.read_settings(self._record(tmp_path))
        expected = BuzzConfig().station.audio_rf_conversion_db
        assert float(settings['audio_rf_conversion_db']) == pytest.approx(expected)

    def test_records_the_lead_in_length(self, tmp_path):
        settings = wavmeta.read_settings(self._record(tmp_path))
        assert float(settings['lead_in_seconds']) == pytest.approx(2.0)

    @pytest.mark.parametrize('ended', ['timeout', 'operator', 'shutdown'])
    def test_records_why_the_recording_ended(self, tmp_path, ended):
        settings = wavmeta.read_settings(self._record(tmp_path, ended))
        assert settings['ended'] == ended

    def test_a_capped_recording_says_so(self, tmp_path):
        """The one thing metadata can say that the audio cannot: this event was cut
        short by the length limit rather than having actually finished."""
        settings = wavmeta.read_settings(self._record(tmp_path, 'capped', max_seconds=3.0))
        assert settings['ended'] == 'capped'

    def test_cue_marks_the_moment_of_lock(self, tmp_path):
        path = self._record(tmp_path)
        data = path.read_bytes()
        offset = data.index(b'cue ') + 12                 # past id, size, and point count
        position = struct.unpack('<I', data[offset + 4:offset + 8])[0]
        assert position == 2 * CHUNK                      # the 2 s lead-in

    def test_cue_is_labelled(self, tmp_path):
        assert b'LOCK' in self._record(tmp_path).read_bytes()

    def test_audio_is_unaffected_by_tagging(self, tmp_path):
        samples = load_wav(self._record(tmp_path))[0]
        assert len(samples) == 4 * CHUNK                  # 2 s lead-in + 2 s trailer

    def test_a_failed_tagging_does_not_lose_the_recording(self, tmp_path):
        with patch('buzz.recorder.wavmeta.append_metadata', side_effect=OSError('nope')):
            path = self._record(tmp_path)
        assert len(load_wav(path)[0]) == 4 * CHUNK

    def test_a_failed_tagging_is_logged(self, tmp_path, caplog):
        with patch('buzz.recorder.wavmeta.append_metadata', side_effect=OSError('nope')):
            with caplog.at_level('ERROR'):
                self._record(tmp_path)
        assert 'Could not tag' in caplog.text


class TestMinimumLock:
    """A night of two-second blips fills a disk with nothing worth replaying."""

    def test_a_short_lock_records_nothing(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=3)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)              # locked, but only for 2 s
        recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_a_lock_that_holds_records(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=3)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 4)
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_the_wait_is_measured_from_the_lock(self, tmp_path):
        """Audio already buffered before the lock is lead-in, not time served."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=3)
        _feed(pipeline, 10)             # ten seconds of audio, none of it locked
        analyzer.lock()
        recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_the_recording_still_opens_with_its_lead_in(self, tmp_path):
        """Waiting costs lead-in but must not abolish it: the file still opens before
        the lock, just less far before it."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=2)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)
        recorder.tick()
        recorder.stop()
        assert _durations(tmp_path) == [5.0]        # 3 s before the lock, 2 s after

    def test_a_broken_lock_starts_the_wait_again(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=3)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)
        analyzer.unlock()
        recorder.tick()                 # two seconds in, and lost
        analyzer.lock()
        _feed(pipeline, 2)
        recorder.tick()                 # two more, but from scratch
        assert _wav_files(tmp_path) == []

    def test_a_blip_between_polls_never_qualifies(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=3)
        for _ in range(5):
            analyzer.lock()
            analyzer.unlock()
            _feed(pipeline, 2)
            recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_zero_records_every_lock(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=0)
        analyzer.lock()
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_the_wait_is_capped_by_the_ring_buffer(self, tmp_path):
        """Beyond what the buffer holds, a recording would begin after the event it
        is recording, having lost the onset entirely."""
        recorder, pipeline, _ = _make_recorder(tmp_path, min_lock_seconds=10_000)
        assert recorder._min_lock_samples == pipeline.capacity_samples

    def test_an_over_long_wait_is_reported(self, tmp_path, caplog):
        with caplog.at_level('WARNING'):
            _make_recorder(tmp_path, min_lock_seconds=10_000)
        assert 'longer than' in caplog.text

    def test_a_negative_wait_records_every_lock(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, min_lock_seconds=-5)
        assert recorder._min_lock_samples == 0

    def test_the_cue_marks_the_lock_not_the_file_start(self, tmp_path):
        """The seconds spent waiting are part of the event, and sit inside the file
        ahead of the point it was decided to keep it."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=2)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)
        recorder.tick()
        recorder.stop()
        data = _wav_files(tmp_path)[0].read_bytes()
        offset = data.index(b'cue ') + 12
        assert struct.unpack('<I', data[offset + 4:offset + 8])[0] == 3 * CHUNK

    def test_the_metadata_lead_in_excludes_the_wait(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=2)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)
        recorder.tick()
        recorder.stop()
        settings = wavmeta.read_settings(_wav_files(tmp_path)[0])
        assert float(settings['lead_in_seconds']) == pytest.approx(3.0)

    def test_the_wait_counts_against_the_recording_length(self, tmp_path):
        """Ask for a 2 s wait and a 6 s recording and you get six seconds of event,
        not eight: the seconds spent proving the signal are part of it."""
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, min_lock_seconds=2, max_seconds=6)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()                             # waiting out min_lock_seconds
        _feed(pipeline, 2)
        recorder.tick()                             # begins, with 3 s buffered
        _feed(pipeline, 20)
        recorder.tick()                             # the cap trims this
        recorder.stop()
        # 1 s of lead-in before the lock, then 6 s of event: the 2 s already waited
        # through and 4 s more.
        assert _durations(tmp_path) == [7.0]

    def test_the_wait_cannot_be_longer_than_the_recording_length(self, tmp_path):
        """Left alone it would reach back past the start of the file and throw away
        its newest seconds to stay inside an allowance already spent."""
        recorder, _, _ = _make_recorder(tmp_path, min_lock_seconds=8, max_seconds=3)
        assert recorder._min_lock_samples == recorder._max_samples

    def test_a_wait_longer_than_the_recording_length_is_reported(self, tmp_path, caplog):
        with caplog.at_level('WARNING'):
            _make_recorder(tmp_path, min_lock_seconds=8, max_seconds=3)
        assert 'longer than max_seconds' in caplog.text

    def test_an_uncapped_recording_does_not_clamp_the_wait(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, min_lock_seconds=8, max_seconds=0)
        assert recorder._min_lock_samples == 8 * SAMPLE_RATE


class TestMinimumSnr:
    """A signal the analyzer can hear but a person cannot is not worth a file.

    Recording only: the analyzer locks at LOCK_ACQUIRE_SNR regardless, so this
    decides what is kept, never what is heard.
    """

    def _locked_at(self, tmp_path, snr, **recording):
        """A recorder watching a lock of the given SNR, with its window filled.

        Ticked SNR_WINDOW times because that is what the recorder waits for before
        judging anything - one reading decides nothing, by design.
        """
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, min_lock_snr=10.0, **recording)
        _feed(pipeline, 2)
        analyzer.publish(snr)
        analyzer.lock()
        for _ in range(EventRecorder.SNR_WINDOW):
            recorder.tick()
        return recorder, pipeline, analyzer

    def test_a_weak_signal_is_not_recorded(self, tmp_path):
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=7.0)
        for _ in range(5):
            _feed(pipeline, 1)
            recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_a_strong_signal_is_recorded(self, tmp_path):
        recorder, _, _ = self._locked_at(tmp_path, snr=20.0)
        assert len(_wav_files(tmp_path)) == 1

    def test_a_signal_that_builds_is_recorded_when_it_crosses(self, tmp_path):
        """The common shape of a real arc: quiet at first, then loud.  Watched
        rather than abandoned, so the event is caught even though its start is not."""
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=7.0)
        for _ in range(4):
            _feed(pipeline, 1)
            recorder.tick()
        assert _wav_files(tmp_path) == []
        analyzer.publish(25.0)
        for _ in range(5):
            _feed(pipeline, 1)
            recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_starting_late_costs_lead_in(self, tmp_path):
        """The trade this setting makes, and the reason to keep it modest.

        Only visible against a full buffer: until it is discarding, waiting costs
        nothing because nothing has fallen off the far end yet.
        """
        def lead_in(name, weak_seconds):
            directory = tmp_path / name
            recorder, pipeline, analyzer = _make_recorder(
                tmp_path, directory=str(directory), min_lock_snr=10.0)
            _feed(pipeline, 400)                        # buffer full and discarding
            analyzer.publish(2.0)
            analyzer.lock()
            for _ in range(weak_seconds):               # too quiet to record yet
                _feed(pipeline, 1)
                recorder.tick()
            analyzer.publish(30.0)
            for _ in range(EventRecorder.SNR_WINDOW):
                _feed(pipeline, 1)
                recorder.tick()
            recorder.stop()
            return float(wavmeta.read_settings(
                _wav_files(directory)[0])['lead_in_seconds'])

        assert lead_in('late', 20) < lead_in('prompt', 0)

    def test_one_loud_reading_does_not_let_a_weak_event_through(self, tmp_path):
        """Judged over a window, so a single tick cannot carry the decision."""
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=2.0)
        analyzer.publish(30.0)
        _feed(pipeline, 1)
        recorder.tick()
        assert _wav_files(tmp_path) == []

    def test_unlocked_readings_are_not_counted(self, tmp_path):
        """An unlocked result reports zero SNR by convention rather than a measured
        one, and averaging those in would hold off a loud event over nothing."""
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=20.0)
        analyzer.publish(0.0, locked=False)
        assert len(_wav_files(tmp_path)) == 1

    def test_losing_the_lock_forgets_the_readings(self, tmp_path):
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=20.0,
                                                       min_lock_seconds=3)
        analyzer.unlock()
        recorder.tick()
        analyzer.publish(2.0)
        analyzer.lock()
        for _ in range(5):
            _feed(pipeline, 1)
            recorder.tick()
        assert _wav_files(tmp_path) == []

    def _waited_past_the_buffer(self, tmp_path, max_seconds):
        """Watch a signal below the threshold until the lock falls out of the buffer."""
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, min_lock_snr=10.0, max_seconds=max_seconds)
        analyzer.publish(2.0)
        analyzer.lock()
        for _ in range(400):                # more than the buffer holds
            _feed(pipeline, 1)
            recorder.tick()
        analyzer.publish(30.0)
        for _ in range(max_seconds + 5):
            _feed(pipeline, 1)
            recorder.tick()
        recorder.stop()
        return recorder

    def test_a_wait_that_outlives_the_buffer_still_records(self, tmp_path):
        """The lock can fall out of the buffer entirely while the level is watched.

        Timing the cap from the lock then spends it before the file is even opened,
        and the event this setting exists to catch is saved as silence.  Timing it
        from the first buffered sample is no better: the buffer is lead-in that was
        already free, and letting it eat the cap leaves a fraction of a second of the
        loud part everybody was waiting for.
        """
        self._waited_past_the_buffer(tmp_path, max_seconds=5)
        buffered = RingBufferPipeline().capacity_samples / SAMPLE_RATE
        assert _durations(tmp_path) == [buffered + 5.0]

    def test_a_wait_that_outlives_the_buffer_does_not_look_like_falling_behind(
            self, tmp_path, caplog):
        """The write position landing behind itself is what raised that warning, and
        the recorder had not fallen behind anything."""
        with caplog.at_level('WARNING'):
            self._waited_past_the_buffer(tmp_path, max_seconds=5)
        assert 'fell behind' not in caplog.text

    def test_zero_records_whatever_locks(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_snr=0.0)
        analyzer.publish(1.0)
        analyzer.lock()
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_it_waits_for_both_the_level_and_the_time(self, tmp_path):
        recorder, pipeline, analyzer = self._locked_at(tmp_path, snr=20.0,
                                                       min_lock_seconds=4)
        _feed(pipeline, 2)
        recorder.tick()
        assert _wav_files(tmp_path) == []       # loud enough, not yet long enough


class TestSayingWhyItWaited:
    """Why a recording started when it did is not recoverable from the file.

    Working it back from the length took three wrong guesses and a measurement rig
    the first time somebody asked, so the recorder now says so itself.
    """

    def test_a_prompt_start_says_nothing_about_waiting(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 2)
        analyzer.lock()
        with caplog.at_level('INFO'):
            recorder.tick()
        assert 'after the lock' not in caplog.text

    def test_a_wait_for_the_clock_names_min_lock_seconds(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=2)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 3)
        with caplog.at_level('INFO'):
            recorder.tick()
        assert 'waiting out min_lock_seconds' in caplog.text

    def test_a_wait_for_the_level_names_min_lock_snr(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_snr=10.0)
        analyzer.publish(2.0)
        analyzer.lock()
        for _ in range(EventRecorder.SNR_WINDOW):
            _feed(pipeline, 1)
            recorder.tick()
        analyzer.publish(30.0)
        with caplog.at_level('INFO'):
            for _ in range(EventRecorder.SNR_WINDOW):
                _feed(pipeline, 1)
                recorder.tick()
        assert 'waiting for the signal to reach min_lock_snr' in caplog.text

    def test_the_delay_is_reported(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, min_lock_seconds=2)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 3)
        with caplog.at_level('INFO'):
            recorder.tick()
        assert 'started 3.0 s after the lock' in caplog.text

    def test_a_stale_reason_is_not_carried_into_a_new_event(self, tmp_path, caplog):
        """Acquiring a lock forgets it, so an event that starts promptly is never
        described with the reason some earlier one did not."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        recorder._held_back_by = 'waiting out min_lock_seconds'     # left from before
        _feed(pipeline, 2)
        analyzer.lock()
        with caplog.at_level('INFO'):
            recorder.tick()
        assert 'after the lock' not in caplog.text


class TestOddSettings:
    """Settings that contradict each other, or that nobody meant to type.

    None of these should misbehave: a limit that cannot be honoured degrades to the
    nearest thing that can be, and the recorder never ends up in a state that
    duplicates audio or writes to the log five times a second.
    """

    def test_zero_timeout_does_not_end_a_locked_recording(self, tmp_path):
        """A zero timeout used to end the recording on the tick after it began, even
        with the signal still present, because no time at all had passed without lock."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, stop_after_seconds=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 3)
        recorder.tick()
        assert recorder.status().recording is True

    def test_zero_timeout_ends_as_soon_as_the_lock_is_lost(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, stop_after_seconds=0)
        analyzer.lock()
        recorder.tick()
        analyzer.unlock()
        _feed(pipeline, 1)
        recorder.tick()
        assert recorder.status().recording is False

    def test_negative_cap_means_uncapped(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_seconds=-10)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()
        assert recorder.status().recording is True

    def test_negative_cap_does_not_rewind_and_duplicate_audio(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_seconds=-10)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 3)
        recorder.tick()
        recorder.stop()
        assert _durations(tmp_path) == [5.0]

    def test_negative_rearm_period_never_rearms(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, rearm_reset_minutes=-5)
        assert recorder.status().rearm_in_seconds is None

    def test_negative_rearm_period_does_not_flood_the_log(self, tmp_path, caplog):
        recorder, _, _ = _make_recorder(tmp_path, max_events=1, rearm_reset_minutes=-5)
        with caplog.at_level('INFO'):
            for _ in range(10):
                recorder.tick()
        assert 're-armed' not in caplog.text

    def test_timeout_longer_than_the_cap_still_ends_the_recording(self, tmp_path):
        """The cap wins, so the recording ends before the silence timeout can - which
        also means the trailer is only as long as the cap leaves room for."""
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, max_seconds=2.0, stop_after_seconds=30.0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()
        assert recorder.status().recording is False

    def test_timeout_longer_than_the_cap_records_why_it_ended(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(
            tmp_path, max_seconds=2.0, stop_after_seconds=30.0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 5)
        recorder.tick()
        assert wavmeta.read_settings(_wav_files(tmp_path)[0])['ended'] == 'capped'

    def test_rearm_shorter_than_the_cap_does_not_disturb_a_recording(self, tmp_path):
        """The budget can come back round mid-recording; the file in progress is
        none of its business."""
        clock = FakeClock()
        with patch('buzz.recorder.time.monotonic', clock):
            recorder, pipeline, analyzer = _make_recorder(
                tmp_path, max_events=1, max_seconds=0.0, rearm_reset_minutes=1)
            analyzer.lock()
            recorder.tick()
            clock.advance(2)                    # two reset cycles pass mid-recording
            _feed(pipeline, 3)
            recorder.tick()
            recording = recorder.status().recording
            recorder.stop()
        assert recording is True and _durations(tmp_path) == [3.0]


class TestEventBudget:
    def _record_one_event(self, recorder, pipeline, analyzer):
        analyzer.lock()
        recorder.tick()
        analyzer.unlock()
        _feed(pipeline, 2)
        recorder.tick()

    def test_disarms_once_the_budget_is_spent(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        self._record_one_event(recorder, pipeline, analyzer)
        assert recorder.status().armed is False

    def test_the_budget_running_out_says_so(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        with caplog.at_level('INFO'):
            self._record_one_event(recorder, pipeline, analyzer)
        assert 'budget spent' in caplog.text

    def test_stopping_by_hand_does_not_blame_the_budget(self, tmp_path, caplog):
        """The last event's recording is still spent, but the operator is why
        recording is now off - and saying "budget spent" alongside "disarmed" reads
        as the monitor giving two different reasons for the same thing."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        analyzer.lock()
        recorder.tick()
        with caplog.at_level('INFO'):
            recorder.disarm()
        assert 'Recording disarmed' in caplog.text
        assert 'budget spent' not in caplog.text

    def test_shutting_down_does_not_blame_the_budget(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        analyzer.lock()
        recorder.tick()
        with caplog.at_level('INFO'):
            recorder.stop()
        assert 'budget spent' not in caplog.text

    def test_stays_armed_while_the_budget_remains(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=2)
        self._record_one_event(recorder, pipeline, analyzer)
        assert recorder.status().armed is True

    def test_budget_counts_down(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=3)
        self._record_one_event(recorder, pipeline, analyzer)
        assert recorder.status().events_remaining == 2

    def test_records_only_as_many_events_as_budgeted(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=2)
        for _ in range(4):
            self._record_one_event(recorder, pipeline, analyzer)
        assert len(_wav_files(tmp_path)) == 2

    def test_zero_budget_means_unlimited(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=0)
        for _ in range(4):
            self._record_one_event(recorder, pipeline, analyzer)
        assert len(_wav_files(tmp_path)) == 4

    def test_unlimited_budget_reports_no_remaining_count(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, max_events=0)
        assert recorder.status().events_remaining is None

    def test_arm_refills_the_budget(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=2)
        self._record_one_event(recorder, pipeline, analyzer)
        recorder.arm()
        assert recorder.status().events_remaining == 2

    def test_arm_records_again_after_the_budget_was_spent(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        self._record_one_event(recorder, pipeline, analyzer)
        recorder.arm()
        self._record_one_event(recorder, pipeline, analyzer)
        assert len(_wav_files(tmp_path)) == 2

    def test_arm_resumes_during_the_event_it_was_disarmed_in(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=1)
        analyzer.lock()
        recorder.tick()
        recorder.disarm()
        recorder.arm()                            # still locked, no fresh lock needed
        recorder.tick()
        assert recorder.status().recording is True


class TestRearmCycle:
    """max_events as a rate: the budget refills on a fixed cycle, unattended."""

    def _recorder(self, tmp_path, clock, **recording):
        with patch('buzz.recorder.time.monotonic', clock):
            return _make_recorder(tmp_path, **recording)

    def _spend_budget(self, recorder, pipeline, analyzer, clock, events=1):
        with patch('buzz.recorder.time.monotonic', clock):
            for _ in range(events):
                analyzer.lock()
                recorder.tick()
                analyzer.unlock()
                _feed(pipeline, 2)
                recorder.tick()

    def _tick(self, recorder, clock):
        with patch('buzz.recorder.time.monotonic', clock):
            recorder.tick()

    def _status(self, recorder, clock):
        with patch('buzz.recorder.time.monotonic', clock):
            return recorder.status()

    def test_never_rearms_when_the_period_is_zero(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=0)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(60 * 24 * 7)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).armed is False

    def test_rearms_once_the_period_elapses(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(1440)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).armed is True

    def test_stays_disarmed_until_the_period_elapses(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(1439)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).armed is False

    def test_budget_is_refilled(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=3, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock, events=3)
        clock.advance(1440)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).events_remaining == 3

    def test_records_again_after_rearming(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(1440)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        assert len(_wav_files(tmp_path)) == 2

    def test_the_cycle_does_not_drift(self, tmp_path):
        """The point of resetting on a cycle rather than after a delay: a day's
        events taking two hours must not push tomorrow's window two hours later."""
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        clock.advance(120)                       # the event arrives two hours in
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(1440 - 120)                # exactly one period after arming
        self._tick(recorder, clock)
        assert self._status(recorder, clock).armed is True

    def test_a_quiet_period_still_refills(self, tmp_path):
        """A cycle that spent only part of its budget tops back up rather than
        carrying the remainder forward, so the rate is a ceiling, not a quota."""
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=3, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock, events=1)
        clock.advance(1440)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).events_remaining == 3

    def test_a_missed_cycle_does_not_fire_repeatedly(self, tmp_path):
        """After a suspend, the cycle restarts from now rather than catching up on
        windows during which nothing could have been recorded anyway."""
        clock = FakeClock()
        recorder, _, _ = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        clock.advance(1440 * 10)
        self._tick(recorder, clock)
        remaining = self._status(recorder, clock).rearm_in_seconds
        assert remaining == pytest.approx(1440 * 60)

    def test_manual_disarm_cancels_the_cycle(self, tmp_path):
        clock = FakeClock()
        recorder, _, _ = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        recorder.disarm()
        clock.advance(1440 * 2)
        self._tick(recorder, clock)
        assert self._status(recorder, clock).armed is False

    def test_manual_arm_restarts_the_cycle(self, tmp_path):
        clock = FakeClock()
        recorder, _, _ = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        recorder.disarm()
        clock.advance(600)
        with patch('buzz.recorder.time.monotonic', clock):
            recorder.arm()
        assert self._status(recorder, clock).rearm_in_seconds == pytest.approx(1440 * 60)

    def test_status_counts_down_to_the_reset(self, tmp_path):
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, rearm_reset_minutes=1440)
        self._spend_budget(recorder, pipeline, analyzer, clock)
        clock.advance(1080)
        assert self._status(recorder, clock).rearm_in_seconds == pytest.approx(360 * 60)

    def test_no_countdown_without_a_cycle(self, tmp_path):
        clock = FakeClock()
        recorder, _, _ = self._recorder(tmp_path, clock, rearm_reset_minutes=0)
        assert self._status(recorder, clock).rearm_in_seconds is None

    def test_a_reset_does_not_restart_a_capped_event(self, tmp_path):
        """Automatic, so unlike arming by hand it is not an instruction to record
        the event already in progress; one event still means one file."""
        clock = FakeClock()
        recorder, pipeline, analyzer = self._recorder(
            tmp_path, clock, max_events=1, max_seconds=2.0, rearm_reset_minutes=1440)
        with patch('buzz.recorder.time.monotonic', clock):
            analyzer.lock()
            recorder.tick()
            _feed(pipeline, 5)
            recorder.tick()                      # capped, signal still present
            clock.advance(1440)
            _feed(pipeline, 5)
            recorder.tick()                      # budget resets mid-event
        assert len(_wav_files(tmp_path)) == 1


class TestStatusAndToggle:
    def test_toggle_arms_when_off(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path, enabled=False)
        assert recorder.toggle() is True

    def test_toggle_disarms_when_on(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path)
        assert recorder.toggle() is False

    def test_idle_status_has_no_filename(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path)
        assert recorder.status().filename is None

    def test_recording_status_carries_the_filename(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 1)
        analyzer.lock()
        recorder.tick()
        assert recorder.status().filename == _wav_files(tmp_path)[0].name

    def test_elapsed_starts_at_zero_however_long_the_lead_in_was(self, tmp_path):
        """A stopwatch that starts at nine because the ring buffer was full is not
        telling the operator anything they asked about."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 9)
        analyzer.lock()
        recorder.tick()
        assert recorder.status().elapsed_seconds == 0.0

    def test_elapsed_counts_from_the_lock(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 9)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 4)
        recorder.tick()
        assert recorder.status().elapsed_seconds == pytest.approx(4.0)

    def test_elapsed_agrees_with_the_length_cap(self, tmp_path):
        """Both are timed from the lock, so the reading at which a capped recording
        stops is the cap itself rather than the cap plus a lead-in."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_seconds=3.0)
        _feed(pipeline, 9)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 2)
        recorder.tick()
        assert recorder.status().elapsed_seconds == pytest.approx(2.0)

    def test_elapsed_is_zero_when_idle(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path)
        assert recorder.status().elapsed_seconds == 0.0

    def test_elapsed_returns_to_zero_after_a_recording(self, tmp_path):
        """The sample positions it is derived from stay where the recording left
        them, so this is only zero if something says so."""
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 4)
        analyzer.unlock()
        _feed(pipeline, 2)
        recorder.tick()
        assert recorder.status().elapsed_seconds == 0.0

    def test_seeds_lock_state_from_an_already_locked_analyzer(self, tmp_path):
        pipeline = RingBufferPipeline()
        analyzer = FakeAnalyzer(AnalyzerState.LOCKED)
        recorder = EventRecorder(pipeline, analyzer, _make_config(tmp_path))
        _feed(pipeline, 1)
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1


class TestLockStateHandoff:
    """The analyzer pushes lock state and the tick consumes it.

    Reading the live level and clearing the sticky flag are one operation because they
    have to be: a lock and a loss both arriving between two separate steps would clear
    the flag that recorded the lock while leaving the level saying there was none, and
    the brief lock this monitor exists to catch would vanish between two polls.
    """

    def test_reports_a_lock_that_came_and_went(self, tmp_path):
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        analyzer.unlock()
        assert recorder._consume_lock_state()[0] is True

    def test_consuming_clears_the_sticky_flag(self, tmp_path):
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        analyzer.unlock()
        recorder._consume_lock_state()
        assert recorder._consume_lock_state()[0] is False

    def test_a_standing_lock_survives_being_consumed(self, tmp_path):
        """Only the sticky flags are consumed; a signal still present is still present."""
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        recorder._consume_lock_state()
        assert recorder._consume_lock_state()[0] is True

    def test_reports_a_loss_that_came_and_went(self, tmp_path):
        """The edge min_lock_seconds needs: without it a stream of blips is
        indistinguishable from one unbroken lock, since no tick sees the gaps."""
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        analyzer.unlock()
        analyzer.lock()
        assert recorder._consume_lock_state() == (True, True)

    def test_an_unbroken_lock_reports_no_loss(self, tmp_path):
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        assert recorder._consume_lock_state() == (True, False)

    def test_consuming_clears_the_loss_flag(self, tmp_path):
        recorder, _, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        analyzer.unlock()
        recorder._consume_lock_state()
        assert recorder._consume_lock_state()[1] is False

    def test_the_listener_waits_for_a_consume_in_progress(self, tmp_path):
        """The mutual exclusion the flags' own lock exists for.  It is held only across
        two assignments, which is why it is a separate lock from the one tick() holds
        while writing: the analyzer must never end up behind the recorder's disk I/O.
        """
        recorder, _, analyzer = _make_recorder(tmp_path)
        with recorder._state_lock:
            threading.Thread(target=analyzer.lock, daemon=True).start()
            time.sleep(0.05)    # ample time for an unsynchronised listener to arrive
            assert recorder._locked is False
        assert _eventually(lambda: recorder._locked is True)


class TestFallingBehind:
    def test_warns_when_the_ring_buffer_outruns_the_recorder(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 400)                      # more than the buffer can hold
        with caplog.at_level('WARNING'):
            recorder.tick()
        assert 'fell behind' in caplog.text

    def test_keeps_recording_after_falling_behind(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path, max_events=0)
        analyzer.lock()
        recorder.tick()
        _feed(pipeline, 400)
        recorder.tick()
        assert recorder.status().recording is True


class TestThreadedOperation:
    def test_start_and_stop_run_a_full_event(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 2)
        analyzer.lock()
        recorder.start()
        recorder.stop()
        assert len(_wav_files(tmp_path)) == 1

    def test_tick_failure_is_logged_and_does_not_kill_the_thread(self, tmp_path, caplog):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        analyzer.lock()
        with patch.object(EventRecorder, '_tick', side_effect=RuntimeError('boom')):
            with caplog.at_level('ERROR'):
                recorder.start()
                recorder.stop()
        assert 'Recorder tick failed' in caplog.text

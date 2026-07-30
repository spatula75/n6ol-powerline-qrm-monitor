"""Tests for EventRecorder: lead-in, trailer, length cap, event budget, and filenames."""

import re
import struct
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from buzz import __version__, wavmeta
from buzz.analyzer import AnalyzerState
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
    # these tests count in chunks, and a default retuned for real 16 kHz audio lands
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


def _durations(tmp_path: Path) -> list[float]:
    return [len(load_wav(p)[0]) / SAMPLE_RATE for p in _wav_files(tmp_path)]


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
        analyzer.unlock()       # both edges land inside one poll interval
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1

    def test_unwritable_directory_disarms_rather_than_raising(self, tmp_path):
        blocked = tmp_path / 'not-a-directory'
        blocked.touch()
        recorder, pipeline, analyzer = _make_recorder(tmp_path, directory=str(blocked))
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

    def test_elapsed_counts_the_audio_written(self, tmp_path):
        recorder, pipeline, analyzer = _make_recorder(tmp_path)
        _feed(pipeline, 3)
        analyzer.lock()
        recorder.tick()
        assert recorder.status().elapsed_seconds == pytest.approx(3.0)

    def test_elapsed_is_zero_when_idle(self, tmp_path):
        recorder, _, _ = _make_recorder(tmp_path)
        assert recorder.status().elapsed_seconds == 0.0

    def test_seeds_lock_state_from_an_already_locked_analyzer(self, tmp_path):
        pipeline = RingBufferPipeline()
        analyzer = FakeAnalyzer(AnalyzerState.LOCKED)
        recorder = EventRecorder(pipeline, analyzer, _make_config(tmp_path))
        _feed(pipeline, 1)
        recorder.tick()
        assert len(_wav_files(tmp_path)) == 1


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

"""Tests for RIFF metadata: INFO tags, cue markers, and the settings comment."""

import struct
import tracemalloc
import wave
from pathlib import Path

import numpy as np
import pytest

from buzz import wavmeta
from buzz.playback import load_wav

TAGS = {
    'INAM': 'N6OL powerline QRM event 2026-07-29T14:33:07-07:00',
    'IART': 'N6OL',
    'ICMT': 'pulse_rate=120 lead_in_seconds=9.6 ended=timeout',
}


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> Path:
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype('<i2').tobytes())
    return path


def _chunk_names(path: Path) -> list[str]:
    data = path.read_bytes()
    names, pos = [], 12
    while pos + 8 <= len(data):
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        names.append(data[pos:pos + 4].decode('latin-1'))
        pos += 8 + size + (size % 2)
    return names


def _corrupt_list_size(path: Path) -> Path:
    """Overwrite the LIST chunk's declared length with the largest a 32-bit field holds.

    The LIST chunk specifically, because that is the one read_info reads the payload of
    - a bogus length on a chunk it only seeks past costs nothing.
    """
    data = bytearray(path.read_bytes())
    at = data.index(b'LIST')
    data[at + 4:at + 8] = struct.pack('<I', 0xFFFFFFFF)
    path.write_bytes(bytes(data))
    return path


def _tagged(tmp_path: Path, samples=None, cues=None, tags=None) -> Path:
    audio = np.arange(1000, dtype=np.int16) if samples is None else samples
    path = _write_wav(tmp_path / 'event.wav', audio)
    wavmeta.append_metadata(path, TAGS if tags is None else tags, cues)
    return path


class TestChunkFraming:
    def test_even_payload_is_not_padded(self):
        assert wavmeta._chunk(b'test', b'ab') == b'test\x02\x00\x00\x00ab'

    def test_odd_payload_is_padded_to_even_length(self):
        assert wavmeta._chunk(b'test', b'abc') == b'test\x03\x00\x00\x00abc\x00'

    def test_declared_size_excludes_the_pad_byte(self):
        chunk = wavmeta._chunk(b'test', b'abc')
        assert struct.unpack('<I', chunk[4:8])[0] == 3


class TestAppendMetadata:
    def test_audio_survives_byte_for_byte(self, tmp_path):
        audio = np.arange(1000, dtype=np.int16)
        path = _tagged(tmp_path, audio)
        assert np.array_equal(load_wav(path)[0], audio)

    def test_frame_count_is_unchanged(self, tmp_path):
        path = _tagged(tmp_path, np.zeros(1000, dtype=np.int16))
        with wave.open(str(path), 'rb') as wav:
            assert wav.getnframes() == 1000

    def test_chunks_are_appended_after_the_audio(self, tmp_path):
        assert _chunk_names(_tagged(tmp_path)) == ['fmt ', 'data', 'LIST']

    def test_cue_chunks_are_written_when_asked(self, tmp_path):
        path = _tagged(tmp_path, cues={512: 'LOCK'})
        assert _chunk_names(path) == ['fmt ', 'data', 'LIST', 'cue ', 'LIST']

    def test_riff_size_matches_the_new_file_length(self, tmp_path):
        path = _tagged(tmp_path, cues={512: 'LOCK'})
        data = path.read_bytes()
        assert struct.unpack('<I', data[4:8])[0] == len(data) - 8


class TestReadInfo:
    def test_round_trips_every_tag(self, tmp_path):
        assert wavmeta.read_info(_tagged(tmp_path)) == TAGS

    def test_untagged_file_has_no_metadata(self, tmp_path):
        path = _write_wav(tmp_path / 'plain.wav', np.zeros(10, dtype=np.int16))
        assert wavmeta.read_info(path) == {}

    def test_missing_file_reads_as_no_metadata(self, tmp_path):
        assert wavmeta.read_info(tmp_path / 'nope.wav') == {}

    def test_non_riff_file_reads_as_no_metadata(self, tmp_path):
        path = tmp_path / 'notawav.wav'
        path.write_bytes(b'this is not a RIFF file at all')
        assert wavmeta.read_info(path) == {}

    def test_truncated_file_reads_as_no_metadata(self, tmp_path):
        path = _tagged(tmp_path)
        data = path.read_bytes()
        path.write_bytes(data[:len(data) // 2])
        assert wavmeta.read_info(path) == {}

    def test_an_absurd_chunk_length_is_not_allocated(self, tmp_path):
        """A chunk declares its own length in 32 bits, so a truncated transfer or one
        corrupt byte can claim four gigabytes.  read(size) on that really does try to
        allocate all four before finding out the file is two kilobytes - measured here
        rather than assumed - and MemoryError is not the "no metadata" this module
        promises to degrade to.  Sizes are clamped to what is left in the file.
        """
        path = _corrupt_list_size(_tagged(tmp_path))
        tracemalloc.start()
        try:
            wavmeta.read_info(path)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 10 * 1024 * 1024

    def test_an_absurd_chunk_length_still_reads_as_metadata(self, tmp_path):
        """The clamp bounds the read, not the result: the chunk's real bytes are
        untouched by the corruption, so the recovered tags are the exact ones written,
        not merely some dict."""
        assert wavmeta.read_info(_corrupt_list_size(_tagged(tmp_path))) == TAGS

    def test_an_absurd_chunk_length_still_reads_as_settings(self, tmp_path):
        """read_settings walks the same chunks, so it needs the same protection."""
        assert wavmeta.read_settings(_corrupt_list_size(_tagged(tmp_path))) == {
            'pulse_rate': '120', 'lead_in_seconds': '9.6', 'ended': 'timeout',
        }

    def test_odd_length_values_round_trip(self, tmp_path):
        tags = {'ICMT': 'odd'}          # 3 chars + NUL: the pad byte must not leak
        assert wavmeta.read_info(_tagged(tmp_path, tags=tags)) == tags

    def test_reads_tags_written_alongside_cues(self, tmp_path):
        assert wavmeta.read_info(_tagged(tmp_path, cues={512: 'LOCK'})) == TAGS


class TestCueChunk:
    def _cue_payload(self, path: Path) -> bytes:
        data = path.read_bytes()
        pos = 12
        while pos + 8 <= len(data):
            size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
            if data[pos:pos + 4] == b'cue ':
                return data[pos + 8:pos + 8 + size]
            pos += 8 + size + (size % 2)
        raise AssertionError('no cue chunk')

    def test_records_the_number_of_points(self, tmp_path):
        payload = self._cue_payload(_tagged(tmp_path, cues={512: 'LOCK'}))
        assert struct.unpack('<I', payload[:4])[0] == 1

    def test_position_is_the_sample_offset(self, tmp_path):
        payload = self._cue_payload(_tagged(tmp_path, cues={512: 'LOCK'}))
        _, position, chunk, _, _, offset = struct.unpack('<II4sIII', payload[4:28])
        assert (position, offset, chunk) == (512, 512, b'data')

    def test_label_text_is_written(self, tmp_path):
        path = _tagged(tmp_path, cues={512: 'LOCK'})
        assert b'LOCK' in path.read_bytes()

    def test_several_cues_are_numbered_in_order(self, tmp_path):
        payload = self._cue_payload(_tagged(tmp_path, cues={900: 'B', 100: 'A'}))
        first = struct.unpack('<II', payload[4:12])
        assert first == (1, 100)


class TestSettings:
    def test_formats_pairs(self):
        assert wavmeta.format_settings({'a': 1, 'b': 2.5}) == 'a=1 b=2.5'

    def test_round_trips(self):
        values = {'pulse_rate': '120', 'ended': 'timeout'}
        assert wavmeta.parse_settings(wavmeta.format_settings(values)) == values

    def test_ignores_malformed_items(self):
        assert wavmeta.parse_settings('good=1 garbage other=2') == {'good': '1', 'other': '2'}

    def test_empty_comment_parses_to_nothing(self):
        assert wavmeta.parse_settings('') == {}

    def test_values_may_contain_an_equals_sign(self):
        assert wavmeta.parse_settings('a=b=c') == {'a': 'b=c'}

    def test_read_settings_from_a_file(self, tmp_path):
        settings = wavmeta.read_settings(_tagged(tmp_path))
        assert settings['pulse_rate'] == '120'

    def test_read_settings_of_an_untagged_file(self, tmp_path):
        path = _write_wav(tmp_path / 'plain.wav', np.zeros(10, dtype=np.int16))
        assert wavmeta.read_settings(path) == {}


class TestSettingAccessor:
    def test_casts_the_value(self):
        assert wavmeta.setting({'n': '120'}, 'n', int) == 120

    def test_missing_key_is_none(self):
        assert wavmeta.setting({}, 'n', int) is None

    def test_unparsable_value_is_none(self):
        assert wavmeta.setting({'n': 'twelve'}, 'n', int) is None

    def test_floats_are_supported(self):
        assert wavmeta.setting({'db': '-32.5'}, 'db', float) == pytest.approx(-32.5)

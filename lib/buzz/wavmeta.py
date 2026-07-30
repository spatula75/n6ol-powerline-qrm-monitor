"""
RIFF metadata for recorded .wav files: provenance tags and cue markers.

The stdlib wave module writes audio and nothing else — Wave_write has no metadata
API at all, and its setmark() raises outright.  So the chunks are assembled here
and appended after wave has closed the file.

That is safe rather than clever.  A .wav is a RIFF container of independent
chunks, and every reader walks it chunk by chunk; the stdlib's own reader stops
the moment it reaches `data`, so anything after the audio is invisible to it and
the audio itself is untouched.  Appending means updating the RIFF size field at
offset 4, and requires the data chunk to be an even number of bytes, which
16-bit mono always is.

Two kinds of metadata go in:

  LIST/INFO   Free-text tags (INAM, IART, ICRD, ISFT, ICMT) that editors and
              taggers display.  ICMT carries the settings a replay needs in
              key=value form — the sample rate is in the format header, but the
              grid's pulse rate and the station's dB calibration are not, and
              without them a recording measures differently on another machine.

  cue + adtl  Sample-accurate markers with labels.  One is written at the moment
              of lock, so opening a recording in an editor shows where the
              lead-in ends and the event itself begins.

Reading is deliberately forgiving.  Anything unparsable, truncated, or simply
recorded by other software reads as "no metadata", never as an error: playback
of a plain .wav from anywhere else has to keep working.
"""

import logging
import struct
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

# INFO tag text is nominally ASCII.  UTF-8 is the de facto encoding for anything
# beyond it, and everything written here (callsigns, timestamps, numbers) is ASCII
# in practice, so this only matters for an exotic callsign in someone's config.
_ENCODING = 'utf-8'

_RIFF_SIZE_OFFSET = 4       # where the total-size field lives in a RIFF header
_HEADER_SIZE = 12           # 'RIFF' + size + 'WAVE'


# --------------------------------------------------------------------- writing

def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """One RIFF chunk: id, little-endian size, payload, pad byte to an even length."""
    return chunk_id + struct.pack('<I', len(payload)) + payload + (b'\x00' * (len(payload) % 2))


def build_info(tags: dict[str, str]) -> bytes:
    """A LIST/INFO chunk from four-character tag ids to text.

    Values are NUL-terminated, as the INFO convention expects; the pad byte that
    _chunk adds for odd-length values is separate from that terminator.
    """
    body = b'INFO'
    for tag, value in tags.items():
        body += _chunk(tag.encode('ascii'), value.encode(_ENCODING) + b'\x00')
    return _chunk(b'LIST', body)


def build_cues(cues: dict[int, str]) -> bytes:
    """A cue chunk plus the LIST/adtl chunk holding each cue point's label.

    Positions are sample frames from the start of the file.  For uncompressed PCM
    the chunk-start and block-start fields are zero and the play order position and
    sample offset are both simply that frame number.
    """
    points = b''
    labels = b'adtl'
    for cue_id, (position, label) in enumerate(sorted(cues.items()), start=1):
        points += struct.pack('<II4sIII', cue_id, position, b'data', 0, 0, position)
        labels += _chunk(b'labl', struct.pack('<I', cue_id) + label.encode(_ENCODING) + b'\x00')
    return (_chunk(b'cue ', struct.pack('<I', len(cues)) + points)
            + _chunk(b'LIST', labels))


def append_metadata(path: Path | str, tags: dict[str, str],
                    cues: dict[int, str] | None = None) -> None:
    """Append INFO tags and cue markers to a finished .wav, fixing up the RIFF size."""
    blob = build_info(tags) + (build_cues(cues) if cues else b'')
    with open(path, 'r+b') as f:
        f.seek(0, 2)
        f.write(blob)
        total = f.tell() - (_RIFF_SIZE_OFFSET + 4)
        f.seek(_RIFF_SIZE_OFFSET)
        f.write(struct.pack('<I', total))


# --------------------------------------------------------------------- reading

def _iter_chunks(f: Any) -> Any:
    """Yield (chunk id, payload size, payload offset) for each top-level chunk.

    Seeks rather than reading, so inspecting the tags on a long recording does not
    pull its audio into memory.

    Sizes are clamped to what is actually left in the file.  A chunk header declares
    its own length in 32 bits, so a truncated download or a corrupt byte can claim
    four gigabytes; read(size) on that allocates the four gigabytes before finding out
    there is nothing to put in them, and MemoryError is not the "no metadata" this
    module promises to degrade to.  The walk itself was always safe — an overshooting
    seek lands past EOF and the next read comes back short — but a caller holding a
    size it trusts is not.
    """
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(_HEADER_SIZE)
    while True:
        header = f.read(8)
        if len(header) < 8:
            return
        offset = f.tell()
        size = min(struct.unpack('<I', header[4:8])[0], file_size - offset)
        yield header[:4], size, offset
        f.seek(offset + size + (size % 2))


def read_info(path: Path | str) -> dict[str, str]:
    """Return the file's LIST/INFO tags, or {} if it has none or cannot be read."""
    try:
        with open(path, 'rb') as f:
            header = f.read(_HEADER_SIZE)
            if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                return {}
            for name, size, offset in _iter_chunks(f):
                if name != b'LIST':
                    continue
                f.seek(offset)
                payload = f.read(size)
                if payload[:4] == b'INFO':
                    return _parse_info(payload[4:])
    except (OSError, struct.error):
        logger.debug('Could not read metadata from %s', path, exc_info=True)
    return {}


def _parse_info(body: bytes) -> dict[str, str]:
    """Split an INFO chunk body into {tag: text}, stopping at the first bad entry."""
    tags, pos = {}, 0
    while pos + 8 <= len(body):
        tag = body[pos:pos + 4].decode('ascii', 'replace')
        size = struct.unpack('<I', body[pos + 4:pos + 8])[0]
        value = body[pos + 8:pos + 8 + size]
        tags[tag] = value.split(b'\x00')[0].decode(_ENCODING, 'replace')
        pos += 8 + size + (size % 2)
    return tags


# -------------------------------------------------------- settings in a comment

def format_settings(values: dict[str, Any]) -> str:
    """Render settings as `key=value` pairs for the ICMT tag.

    Space-separated, which keeps the tag readable in any editor that shows it while
    staying trivially parsable.  Values must not contain spaces; everything stored
    this way is a number or a short identifier.
    """
    return ' '.join(f'{key}={value}' for key, value in values.items())


def parse_settings(comment: str) -> dict[str, str]:
    """Parse `key=value` pairs back out of a comment, ignoring anything malformed."""
    pairs = (item.split('=', 1) for item in comment.split() if '=' in item)
    return {key: value for key, value in pairs}


def read_settings(path: Path | str) -> dict[str, str]:
    """Return the settings recorded in the file's comment tag, or {}."""
    return parse_settings(read_info(path).get('ICMT', ''))


def setting(settings: dict[str, str], key: str, cast: Callable[[str], _T]) -> _T | None:
    """Return settings[key] converted by `cast`, or None if absent or unparsable.

    A recording made by other software, or by an older version of this one, simply
    has nothing to say about the key — which is not an error, just an absence.
    """
    try:
        return cast(settings[key])
    except (KeyError, ValueError):
        return None

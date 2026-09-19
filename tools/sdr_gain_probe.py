"""Measure receiver levels and overload reports at a series of gain settings.

Run it against the receiver `[audio] source` names:

    PYTHONPATH=lib python tools/sdr_gain_probe.py
    PYTHONPATH=lib python tools/sdr_gain_probe.py --step 10 --seconds 0.2

The `hardware` column shows the receiver's gain readback when available.  The `quiet`
column measures the low percentile of frame levels, as the gain chooser does.
An arc or another signal can change between rows, so the level curve alone cannot
prove whether the receiver applied the gain correctly.

The `rails` column counts delivered I/Q values equal to either format endpoint.
The `values` column gives the denominator for `rail%`, which can round to zero even
when a few values hit an endpoint.  Neither figure measures overload before filtering
or decimation.  The `peak` column uses complex magnitude and can exceed 0 dBFS without
either I or Q reaching an endpoint.

Hardware `overload` reads `active` if the receiver reports overload at the end of the
capture.  It reads `seen` if an overload was present during the interval but cleared.
It reads `none` if no overload was reported, or `-` if the receiver has no indication.
The observation starts after the discard reads and ends after sample collection.
Events describe that interval, not individual samples or a percentage of them.
An interval without a report does not establish headroom for a later arc or transmission.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

from buzz.config import CONFIG_PATH, BuzzConfig  # noqa: E402
from buzz.gain_sweep import BandMeasurement, SweepSource  # noqa: E402
from buzz.sdr_device import OverloadStatus  # noqa: E402

# Match the gain sweep's read size so the probe uses the same settling interval.
BLOCK_SAMPLES = 2048


class ProbeReader(SweepSource, Protocol):
    """A gain-sweep source with optional hardware readback."""

    @property
    def reported_gain_db(self) -> float | None:
        ...

    @property
    def overload_status(self) -> OverloadStatus | None:
        ...


@dataclass(frozen=True)
class GainRow:
    """One gain, and everything measured while the receiver sat at it."""

    asked_db: float
    set_db: float
    # Use None when hardware has no readback, so the column cannot imply zero gain.
    hardware_db: float | None
    quiet_dbfs: float
    peak_dbfs: float
    raw_peak: int
    raw_mean: float
    clipped: int
    raw_values: int
    overload_before: OverloadStatus | None = None
    overload_after: OverloadStatus | None = None

    @property
    def clipped_share(self) -> float:
        """What fraction of delivered I/Q values hit a format endpoint."""
        return self.clipped / self.raw_values if self.raw_values else 0.0

    @property
    def overload_label(self) -> str:
        """Whether hardware reported overload during this capture interval.

        The initial state catches overload carried from an earlier gain.  The count
        catches detection and clearance between snapshots, when both states are clear.
        """
        before, after = self.overload_before, self.overload_after
        if before is None or after is None:
            return '-'
        if after.active:
            return 'active'
        if before.active or after.detections > before.detections:
            return 'seen'
        return 'none'

    def __str__(self) -> str:
        hardware = '       -' if self.hardware_db is None else f'{self.hardware_db:8.1f}'
        return (f'{self.asked_db:8.1f} {self.set_db:8.1f}{hardware} '
                f'{self.quiet_dbfs:9.1f} '
                f'{self.peak_dbfs:9.1f} {self.raw_peak:8d} {self.raw_mean:9.1f} '
                f'{self.clipped:8d} {self.raw_values:9d} '
                f'{self.clipped_share:8.2%} {self.overload_label:>8}')


HEADER = (f'{"asked":>8} {"set":>8} {"hardware":>8} {"quiet":>9} {"peak":>9} '
          f'{"rawpeak":>8} {"rawmean":>9} {"rails":>8} {"values":>9} '
          f'{"rail%":>8} {"overload":>8}')


def ladder(gains: list[float], step_db: float) -> list[float]:
    """The gains to visit: every `step_db` or so, with both ends always in.

    This selects a subset of gains so the table stays readable.  It keeps both ends
    to show the measured level across the receiver's full gain range.

    An empty list of gains gives an empty ladder, because a receiver that reports no
    gains is something for `main` to tell the operator about rather than an IndexError
    from here.
    """
    ordered = sorted(gains)
    if not ordered:
        return []
    picked = [ordered[0]]
    for gain in ordered[1:]:
        if gain - picked[-1] >= step_db:
            picked.append(gain)
    if picked[-1] != ordered[-1]:
        picked.append(ordered[-1])
    return picked


def measure_one(reader: ProbeReader, gain_db: float, seconds: float) -> GainRow:
    """Set one gain and describe what the receiver produced at it.

    Hardware events cover the collection interval after settling.  Keep the initial
    overload state as well as the final count, because an overload can persist across
    gains or start and clear between reads.
    """
    actual = reader.set_gain(gain_db)
    reader.drain()
    for _ in range(reader.blocks_to_discard_after_gain_change):
        reader.read()
    overload_before = reader.overload_status
    parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    clipped = 0
    wanted = int(seconds * reader.iq_sample_rate)
    while sum(len(p) for p in parts) < wanted:
        block = reader.read()
        if block is None:
            break
        parts.append(block.as_complex())
        raw_parts.append(block.raw)
        clipped += block.clipped_samples
    overload_after = reader.overload_status
    samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.complex128)
    raw = np.concatenate(raw_parts) if raw_parts else np.empty(0, dtype=np.int16)
    return GainRow(
        asked_db=gain_db,
        set_db=actual,
        hardware_db=reader.reported_gain_db,
        quiet_dbfs=BandMeasurement.quiet_dbfs(samples, reader.iq_sample_rate),
        peak_dbfs=_peak_dbfs(samples),
        raw_peak=int(np.max(np.abs(raw.astype(np.int64)))) if raw.size else 0,
        raw_mean=float(np.mean(raw)) if raw.size else 0.0,
        clipped=clipped,
        raw_values=int(raw.size),
        overload_before=overload_before,
        overload_after=overload_after)


def verdict(rows: list[GainRow]) -> str:
    """What the table says, in one line, for whoever pasted it into a message.

    The captures need not contain the same signals.  A level span describes the
    observations, but cannot identify a gain fault without further evidence.
    """
    if len(rows) < 2:
        return 'Too few gains to say anything.'
    quiet = [row.quiet_dbfs for row in rows if np.isfinite(row.quiet_dbfs)]
    if len(quiet) < 2:
        return 'The receiver produced no usable samples.'
    asked_span = abs(rows[-1].asked_db - rows[0].asked_db)
    quiet_span = max(quiet) - min(quiet)
    return (f'The level moved {quiet_span:.1f} dB while the gain moved '
            f'{asked_span:.1f} dB.  Changing signals, receiver noise, and overload '
            'can affect this comparison.  Use the hardware readback to check gain changes.')


def _peak_dbfs(samples: np.ndarray) -> float:
    """The loudest sample, in dB relative to full scale."""
    if not samples.size:
        return float('-inf')
    peak = float(np.max(np.abs(samples)))
    return 20.0 * np.log10(peak) if peak > 0 else float('-inf')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--step', type=float, default=10.0,
                        help='dB between the gains to visit')
    parser.add_argument('--seconds', type=float, default=0.25,
                        help='seconds of samples to measure at each gain')
    args = parser.parse_args(argv)

    from buzz.sdr import SweepReader
    from buzz.sdr_device import open_receiver

    config = BuzzConfig.from_toml(args.config)
    settings = config.receiver_settings
    if settings is None:
        print(f'[audio] source is {config.audio.source!r}, and this needs a receiver.')
        return 2
    device = open_receiver(config.audio.source, settings)
    try:
        reader = SweepReader(device, block_samples=BLOCK_SAMPLES)
        gains = ladder(reader.supported_gains_db, args.step)
        if not gains:
            print(f'The {config.audio.source} receiver reported no gain settings, so '
                  'there is nothing to probe.  That usually means the driver opened a '
                  'device it does not recognise.  Check that the receiver is the one '
                  f'[{config.audio.source}] describes, then run this again.')
            return 2
        print(f'{config.audio.source} at {device.tuned_hz / 1e6:.4f} MHz, '
              f'{reader.iq_sample_rate} Hz, {len(gains)} of '
              f'{len(reader.supported_gains_db)} gains')
        print(HEADER)
        rows = []
        for gain in gains:
            row = measure_one(reader, gain, args.seconds)
            rows.append(row)
            print(row, flush=True)
        print()
        print('Overload describes this capture only: active at its end, seen then cleared, '
              'none reported, or - unavailable.')
        print(verdict(rows))
    finally:
        device.close()
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())

"""Tests for tools/sdr_gain_probe.py.

The probe reports levels, exact endpoint hits, and hardware overload independently.
Signals can change during a probe, so its verdict must leave the cause open.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from buzz.config import BuzzConfig
from buzz.sdr_device import RTL_SDR_FORMAT, IqBlock, OverloadStatus
from buzz.sdrplay_device import SDRPLAY_FORMAT
from fake_sdr import FakeSdrDevice

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import sdr_gain_probe as probe_module  # noqa: E402

GAINS = [float(-step) for step in range(20, 121)]


def probe(reader, gains: list[float], seconds: float) -> list[probe_module.GainRow]:
    """Walk a ladder and collect every row, for the cases that need the whole table.

    The tool's own main() prints each row as it is measured rather than collecting
    them, because a probe of a full ladder takes about a minute and an operator
    watching it wants the rows as they arrive.  That loop is three lines, so this
    gathering version lives here rather than in the tool where nothing would call it.
    """
    return [probe_module.measure_one(reader, gain, seconds) for gain in gains]


class NoisyReader:
    """A sweep source whose noise level follows the gain, or ignores it.

    `follows` is the whole point.  A receiver that answers its setting produces a level
    that tracks it, and one whose setting goes nowhere produces the same level however
    often it is asked.
    """

    def __init__(self, *, follows: bool = True, gains: list[float] | None = None,
                 iq_sample_rate: int = 256_000) -> None:
        self.supported_gains_db = list(GAINS if gains is None else gains)
        self.iq_sample_rate = iq_sample_rate
        self.blocks_to_discard_after_gain_change = 1
        self.reported_gain_db: float | None = None
        self.overload_status: OverloadStatus | None = None
        self.follows = follows
        self.gains_set: list[float] = []
        self.drains = 0
        self._gain_db = self.supported_gains_db[0]
        self._random = np.random.default_rng(20260916)

    def set_gain(self, gain_db: float) -> float:
        self.gains_set.append(gain_db)
        self._gain_db = gain_db
        return gain_db

    def drain(self) -> int:
        self.drains += 1
        return 0

    def read(self, timeout: float = 1.0) -> IqBlock | None:
        # A level that tracks the gain, taken against the top of the ladder so that the
        # loudest gain sits just under the rail rather than over it.
        loudest = max(self.supported_gains_db)
        scale = 0.25 * 10 ** ((self._gain_db - loudest) / 20) if self.follows else 0.25
        values = self._random.normal(0.0, scale, 1024).clip(-1.0, 1.0)
        raw = np.round(values * 127.5 + 127.5).astype(np.uint8)
        return IqBlock(raw=raw, fmt=RTL_SDR_FORMAT, arrived_at=0.0, index=1)


class TestTheLadderItWalks:
    def test_it_thins_the_ladder_to_about_the_step_asked_for(self):
        """A hundred rows hide the shape of the curve, which is what this is for."""
        picked = probe_module.ladder(GAINS, 10.0)
        assert len(picked) == 11
        gaps = [b - a for a, b in zip(picked, picked[1:])]
        assert all(gap >= 10.0 for gap in gaps)

    def test_both_ends_are_always_walked(self):
        """The ends are where a curve that does not move shows it least ambiguously."""
        picked = probe_module.ladder(GAINS, 30.0)
        assert picked[0] == min(GAINS)
        assert picked[-1] == max(GAINS)

    def test_an_uneven_ladder_keeps_its_own_rungs(self):
        """It picks from what the receiver reported rather than inventing a grid, so a
        gain it names is always one the receiver can actually be set to.
        """
        v4 = [0.0, 0.9, 1.4, 12.5, 22.9, 33.8, 49.6]
        assert set(probe_module.ladder(v4, 10.0)) <= set(v4)

    def test_a_receiver_with_no_gains_gives_an_empty_ladder(self):
        """It indexed the first rung with no guard, so a receiver reporting nothing
        gave an IndexError where every other path in this tool writes a sentence for
        the operator.  main() has that sentence.
        """
        assert probe_module.ladder([], 10.0) == []


class TestWhatOneGainReports:
    def test_it_reports_the_gain_the_receiver_settled_on(self):
        """An RTL-SDR snaps a request to a step it offers, so the two can differ, and
        the column exists because the difference matters.
        """
        reader = NoisyReader()
        row = probe_module.measure_one(reader, -60.0, 0.002)
        assert row.asked_db == -60.0
        assert row.set_db == -60.0

    def test_it_throws_away_the_stale_blocks_first(self):
        """The same discard the sweep makes, or the row describes the previous gain."""
        reader = NoisyReader()
        probe_module.measure_one(reader, -60.0, 0.002)
        assert reader.drains == 1

    def test_the_raw_figures_come_back_beside_the_scaled_ones(self):
        """A converter delivering something narrower than this program expects shows up
        as a raw peak far short of the rail, where the scaled level alone would only
        read low and not say why.
        """
        reader = NoisyReader()
        row = probe_module.measure_one(reader, -20.0, 0.004)
        assert 0 < row.raw_peak <= 255
        assert row.raw_values > 0

    def test_a_receiver_that_stops_answering_ends_the_row(self):
        reader = NoisyReader()
        reader.read = lambda timeout=1.0: None
        row = probe_module.measure_one(reader, -20.0, 0.01)
        assert row.raw_values == 0
        assert row.quiet_dbfs == float('-inf')


class TestTheVerdict:
    @pytest.mark.parametrize('follows', [True, False])
    def test_neither_a_flat_nor_a_rising_curve_proves_the_gain_arrives(self, follows: bool) -> None:
        reader = NoisyReader(follows=follows)
        rows = probe(reader, probe_module.ladder(GAINS, 20.0), 0.01)
        message = probe_module.verdict(rows)
        assert 'Changing signals, receiver noise, and overload can affect this comparison.' in message
        assert 'reaching the hardware' not in message

    def test_the_verdict_quotes_both_spans(self) -> None:
        rows = [probe_module.GainRow(gain, gain, None, quiet, -3.0, 200, 127.5, 0, 1000)
                for gain, quiet in [(0.0, -60.0), (20.0, -31.0), (40.0, -42.0)]]
        message = probe_module.verdict(rows)
        assert '29.0 dB while the gain moved 40.0 dB' in message

    def test_too_few_rows_say_so_rather_than_guessing(self):
        assert 'Too few gains' in probe_module.verdict([])

    def test_a_receiver_that_gave_nothing_says_that_instead(self):
        """Distinct from a gain that will not move, because the two need different
        things done about them.
        """
        reader = NoisyReader()
        reader.read = lambda timeout=1.0: None
        rows = probe(reader, probe_module.ladder(GAINS, 40.0), 0.01)
        assert 'no usable samples' in probe_module.verdict(rows)


class TestWhatTheHardwareSaysAboutItsOwnGain:
    """The column that separates the two failures this tool exists for.

    A receiver that reports its gain and reports the same figure at every setting is
    one whose gain is not moving.  One that reports nothing at all leaves the column
    blank, which is honest, where a zero would read as a measurement of zero dB.
    """

    def test_a_receiver_that_reports_nothing_leaves_the_column_blank(self):
        reader = NoisyReader()
        row = probe_module.measure_one(reader, -60.0, 0.002)
        assert row.hardware_db is None
        assert str(row).split()[2] == '-'

    def test_a_receiver_that_reports_its_gain_shows_the_figure(self):
        reader = NoisyReader()
        reader.reported_gain_db = -38.5
        row = probe_module.measure_one(reader, -60.0, 0.002)
        assert row.hardware_db == -38.5
        assert '-38.5' in str(row)


class TestRunningIt:
    """`main` opens hardware, so it is driven against a fake device the way the rest of
    the receiver code is.  A tool whose entry point is untested is a tool that breaks
    the first time somebody needs it, which is the worst moment to find out.
    """

    def _config(self, source: str) -> BuzzConfig:
        config = BuzzConfig()
        config.audio.source = source
        return config

    def test_it_prints_a_row_for_every_gain_it_walks(
            self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        device = FakeSdrDevice(gains=[0.0, 20.0, 40.0])
        monkeypatch.setattr(BuzzConfig, 'from_toml',
                            classmethod(lambda cls, path: self._config('rtlsdr')))
        monkeypatch.setattr('buzz.sdr_device.open_receiver',
                            lambda source, settings: device)

        assert probe_module.main(['--step', '10', '--seconds', '0.001']) == 0

        printed = capsys.readouterr().out
        lines = [line for line in printed.splitlines() if line]
        assert probe_module.HEADER in lines
        # The output includes a banner, header, three rows, overload guide, and verdict.
        assert len(lines) == 7
        for gain in ('0.0', '20.0', '40.0'):
            assert any(line.startswith(f'{float(gain):8.1f}') for line in lines), gain
        assert lines[-1].endswith('.')

    def test_it_releases_the_receiver_even_when_a_row_fails(self, monkeypatch):
        """The probe is run while something is already wrong, so a receiver it left
        held would turn one problem into two.
        """
        device = FakeSdrDevice(gains=[0.0, 20.0])
        monkeypatch.setattr(BuzzConfig, 'from_toml',
                            classmethod(lambda cls, path: self._config('rtlsdr')))
        monkeypatch.setattr('buzz.sdr_device.open_receiver',
                            lambda source, settings: device)
        monkeypatch.setattr(probe_module, 'measure_one',
                            lambda *args: (_ for _ in ()).throw(RuntimeError('no')))

        with pytest.raises(RuntimeError):
            probe_module.main([])
        assert device.closed

    def test_a_receiver_reporting_no_gains_is_explained_rather_than_crashing(
            self, capsys, monkeypatch):
        """The driver opens a device it does not recognise and reports an empty gain
        list, which used to reach the operator as an IndexError from ladder().
        """
        device = FakeSdrDevice(gains=[0.0])
        monkeypatch.setattr(BuzzConfig, 'from_toml',
                            classmethod(lambda cls, path: self._config('rtlsdr')))
        monkeypatch.setattr('buzz.sdr_device.open_receiver',
                            lambda source, settings: device)
        # The reader is the boundary here, because FakeSdrDevice needs a gain to snap
        # its own starting figure to and so cannot itself report none.
        monkeypatch.setattr('buzz.sdr.SweepReader',
                            lambda device, block_samples: SimpleNamespace(
                                supported_gains_db=[], iq_sample_rate=256_000))

        assert probe_module.main([]) == 2
        assert 'reported no gain settings' in capsys.readouterr().out
        assert device.closed

    def test_a_sound_card_is_refused_rather_than_opened(self, capsys, monkeypatch):
        """There is no receiver to walk, and a non-zero exit says so to a shell."""
        monkeypatch.setattr(BuzzConfig, 'from_toml',
                            classmethod(lambda cls, path: self._config('soundcard')))

        assert probe_module.main([]) == 2
        assert 'needs a receiver' in capsys.readouterr().out


class TestTheRowItself:
    def test_the_clipped_share_is_of_the_raw_values(self):
        row = probe_module.GainRow(asked_db=-20.0, set_db=-20.0, quiet_dbfs=-40.0,
                                   hardware_db=None, peak_dbfs=-3.0,
                                   raw_peak=255, raw_mean=127.5,
                                   clipped=25, raw_values=1000)
        assert row.clipped_share == pytest.approx(0.025)

    def test_a_row_with_no_values_does_not_divide_by_zero(self):
        row = probe_module.GainRow(asked_db=-20.0, set_db=-20.0, quiet_dbfs=-40.0,
                                   hardware_db=None, peak_dbfs=-3.0,
                                   raw_peak=0, raw_mean=0.0,
                                   clipped=0, raw_values=0)
        assert row.clipped_share == 0.0

    def test_it_prints_as_one_line_under_the_header(self):
        row = probe_module.GainRow(asked_db=-20.0, set_db=-20.0, quiet_dbfs=-40.0,
                                   hardware_db=None, peak_dbfs=-3.0,
                                   raw_peak=255, raw_mean=127.5,
                                   clipped=25, raw_values=1000)
        assert '\n' not in str(row)
        assert len(str(row).split()) == len(probe_module.HEADER.split())


class ScriptedReader(NoisyReader):
    """A source with one block and hardware status per read, including discard reads."""

    def __init__(self, raw: np.ndarray, statuses: list[OverloadStatus | None]) -> None:
        super().__init__()
        self._raw = raw
        self._statuses = iter(statuses)

    def read(self, timeout: float = 1.0) -> IqBlock:
        self.overload_status = next(self._statuses)
        fmt = RTL_SDR_FORMAT if self._raw.dtype == np.uint8 else SDRPLAY_FORMAT
        return IqBlock(raw=self._raw, fmt=fmt, arrived_at=0.0, index=1)


class TestHardwareOverloadDuringCapture:
    @pytest.mark.parametrize(('before', 'after', 'label'), [
        (None, None, '-'),
        (OverloadStatus(False, 0), OverloadStatus(False, 0), 'none'),
        (OverloadStatus(False, 0), OverloadStatus(True, 1), 'active'),
        (OverloadStatus(True, 1), OverloadStatus(True, 1), 'active'),
        (OverloadStatus(True, 1), OverloadStatus(False, 1), 'seen'),
        (OverloadStatus(False, 1), OverloadStatus(False, 2), 'seen'),
        (OverloadStatus(False, 2), OverloadStatus(False, 2), 'none'),
    ])
    def test_reports_the_capture_interval_after_discard(
            self, before: OverloadStatus | None, after: OverloadStatus | None, label: str) -> None:
        reader = ScriptedReader(np.zeros(1024, dtype=np.int16), [before, after])
        row = probe_module.measure_one(reader, -20.0, 0.002)
        assert row.overload_label == label
        assert str(row).split()[-1] == label

    def test_an_overload_can_persist_across_gains_without_another_detection(self) -> None:
        reader = ScriptedReader(np.zeros(1024, dtype=np.int16), [OverloadStatus(True, 1)] * 4)
        rows = probe(reader, [-20.0, -30.0], 0.002)
        assert [row.overload_label for row in rows] == ['active', 'active']

    def test_an_overload_that_clears_during_discard_does_not_describe_the_capture(self) -> None:
        reader = ScriptedReader(np.zeros(1024, dtype=np.int16), [OverloadStatus(False, 1)] * 2)
        reader.overload_status = OverloadStatus(True, 1)
        row = probe_module.measure_one(reader, -20.0, 0.002)
        assert row.overload_label == 'none'


class TestEndpointCounts:
    def test_one_endpoint_hit_remains_visible_when_the_percentage_rounds_to_zero(self) -> None:
        raw = np.zeros(131072, dtype=np.int16)
        raw[0] = -32768
        reader = ScriptedReader(raw, [None, None])
        row = probe_module.measure_one(reader, -20.0, 0.002)
        assert row.clipped == 1
        assert row.raw_values == 131072
        assert row.clipped_share == pytest.approx(1 / 131072)
        assert str(row).split()[7:10] == ['1', '131072', '0.00%']

    def test_hardware_can_report_overload_without_delivered_values_at_an_endpoint(self) -> None:
        raw = np.tile(np.array([-32764, 32764], dtype=np.int16), 512)
        reader = ScriptedReader(raw, [OverloadStatus(False, 0), OverloadStatus(True, 1)])
        row = probe_module.measure_one(reader, -20.0, 0.002)
        assert row.clipped == 0
        assert row.raw_peak == 32764
        assert row.overload_label == 'active'

    def test_unsigned_rtl_endpoints_count_individual_i_and_q_values(self) -> None:
        raw = np.tile(np.array([0, 255, 1, 254], dtype=np.uint8), 256)
        reader = ScriptedReader(raw, [None, None])
        row = probe_module.measure_one(reader, -20.0, 0.002)
        assert row.clipped == 512
        assert row.raw_values == 1024
        assert row.clipped_share == 0.5
        assert row.overload_label == '-'

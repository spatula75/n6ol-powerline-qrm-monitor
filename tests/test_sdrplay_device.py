"""Tests for the SDRplay shim.

These run against `tests/fake_sdrplay.FakeSdrplayApi`, which fills in the real
generated structs, so what passes here also says the bindings are usable.  No receiver
and no SDRplay API are needed.

What none of this can check is whether a real RSP behaves the way the specification
describes, and two things rest on that: `grChanged` marking the block where a gain
change took effect, and where the rails of the decimated stream sit.  Both are named
in `docs-notebook/todo.md`.
"""
import ctypes
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from buzz import sdrplay_api as api
from buzz.config import SdrplayConfig
from buzz.sdr import SweepReader
from buzz.sdr_device import IqBlock, OverloadStatus
from buzz.sdrplay_device import (ESTIMATED_CALIBRATION_INTERCEPT_DB, FLOOR_MARGIN_DB,
                                 HF_CONVERSION_GAIN_DB, HF_LNA_GAIN_REDUCTION_DB,
                                 MAX_GAIN_REDUCTION_DB, MIN_GAIN_REDUCTION_DB,
                                 SDRPLAY_FORMAT, SdrplayDevice, SdrplayLibrary,
                                 _SyncBlocks)
from fake_sdrplay import FakeSdrplayApi

TUNED_HZ = 7_050_000
IQ_SAMPLE_RATE = 256_000


def test_an_sdrplay_estimate_includes_its_measured_output_intercept():
    assert ESTIMATED_CALIBRATION_INTERCEPT_DB == 11.0
    assert SdrplayDevice.estimated_calibration_offset_db(13.0) == -2.0
    assert FLOOR_MARGIN_DB == 10.0
    assert SdrplayDevice.floor_margin_db() == FLOOR_MARGIN_DB == 10.0


class CollectingSink:
    """A `BlockSink` that keeps what it is given, and can refuse on command."""

    def __init__(self, accept: bool = True) -> None:
        self.blocks: list[IqBlock] = []
        self.accept = accept

    def offer(self, block: IqBlock) -> bool:
        if not self.accept:
            return False
        self.blocks.append(block)
        return True


def make_device(library: FakeSdrplayApi | None = None, *, gain_db: float = 40.0,
                tuned_hz: int = TUNED_HZ,
                iq_sample_rate: int = IQ_SAMPLE_RATE) -> tuple[SdrplayDevice,
                                                               FakeSdrplayApi]:
    """A configured device over a fake library, with both handed back."""
    library = library or FakeSdrplayApi()
    device = SdrplayDevice(library, library.device, tuned_hz=tuned_hz,
                           gain_db=gain_db, iq_sample_rate=iq_sample_rate)
    return device, library


class TestConfiguring:
    def test_the_converter_runs_fast_and_the_library_decimates(self):
        """An RSP cannot sample at 256 kHz, so it samples at 2.048 MHz and divides.

        The factor is the smallest that lifts the converter to its own 2 MHz minimum,
        because a faster converter spends USB bandwidth for a band already narrower
        than the result.
        """
        _, library = make_device()
        assert library.fs_hz == 2_048_000
        assert library.control.decimation.decimationFactor == 8
        assert library.control.decimation.enable == 1

    @pytest.mark.parametrize('rate, fs_hz, factor', [
        (2_000_000, 2_000_000, 1),      # already at the minimum, so no decimation
        (500_000, 2_000_000, 4),
        (256_000, 2_048_000, 8),
        (250_000, 2_000_000, 8),
        (192_000, 3_072_000, 16),       # 8 would leave the converter below 2 MHz
    ])
    def test_every_rate_gets_a_converter_rate_the_hardware_admits(self, rate, fs_hz,
                                                                  factor):
        assert SdrplayDevice._rate_plan(rate) == (float(fs_hz), factor)

    def test_a_rate_below_what_decimation_reaches_is_refused(self):
        """32 is the deepest decimation, so 62500 Hz is the floor.

        A rate under it would otherwise be configured as something else entirely, and
        every measurement afterwards would be scaled by a factor nothing stated.
        """
        with pytest.raises(ValueError, match='too low for this receiver'):
            SdrplayDevice._rate_plan(50_000)

    def test_the_filter_is_the_narrowest_that_passes_the_whole_band(self):
        """A filter narrower than the sample rate rolls off inside the waterfall."""
        _, library = make_device()
        assert library.tuner.bwType == api.sdrplay_api_Bw_MHzT.sdrplay_api_BW_0_300

    def test_the_agc_is_turned_off(self):
        """The API turns its AGC on by default, and an AGC riding the impulses would
        compress exactly what this program measures while the noise floor still looks
        healthy.  Same reasoning as the RTL-SDR's digital AGC.
        """
        _, library = make_device()
        assert library.control.agc.enable == (
            api.sdrplay_api_AgcControlT.sdrplay_api_AGC_DISABLE)

    def test_the_receiver_is_tuned_where_it_was_asked(self):
        _, library = make_device()
        assert library.tuner.rfFreq.rfHz == float(TUNED_HZ)
        assert library.tuner.ifType == api.sdrplay_api_If_kHzT.sdrplay_api_IF_Zero

    def test_a_tuning_above_the_vendored_table_is_refused(self):
        """The LNA table this module carries is the row below the first band edge.

        Above it both receivers move to a ten-entry row with different figures.  Using
        the HF row up there would report a gain that is wrong by up to 20 dB, and
        nothing in the program could tell.
        """
        library = FakeSdrplayApi(hw_ver=api.SDRPLAY_RSP1B_ID)
        with pytest.raises(ValueError, match='below 50 MHz'):
            SdrplayDevice(library, library.device, tuned_hz=144_000_000,
                          gain_db=-40, iq_sample_rate=IQ_SAMPLE_RATE)

    def test_an_rsp1a_gets_the_higher_band_edge(self):
        """An RSP1A carries the HF row to 60 MHz where an RSP1B stops at 50."""
        library = FakeSdrplayApi(hw_ver=api.SDRPLAY_RSP1A_ID)
        device, _ = make_device(library, tuned_hz=55_000_000)
        assert device.profile.name == 'SDRplay RSP1A'


class TestTheGainLadder:
    def test_it_runs_from_71_down_to_minus_29_db_without_a_gap(self):
        """Two knobs, and the ranges they reach overlap into one unbroken run.

        The figures state gain rather than the negative of a reduction, which is
        what makes them mean the same thing as an RTL-SDR's and what makes the level
        offset a station calibrates come out on the same scale for either receiver.
        """
        device, _ = make_device()
        ladder = device.supported_gains_db
        assert ladder[0] == 71.0
        assert ladder[-1] == -29.0
        assert ladder == [float(round(HF_CONVERSION_GAIN_DB - total))
                          for total in range(20, 121)]

    def test_the_picker_offers_the_same_ladder_the_device_uses(self):
        """A drift pin.  The classmethod answers a gain picker that has no device open
        and the property answers one that does, and nothing else makes them agree.

        They came apart exactly once: the property moved to real gain and the
        classmethod was left returning the negative of a reduction, so the setup
        program offered -20 through -120 for a setting the monitor then displayed as
        +3.  Every figure on screen was right and no two of them were the same thing.
        """
        device, _ = make_device()
        offered = SdrplayDevice.supported_gains(SdrplayConfig())
        assert offered == device.supported_gains_db
        assert offered[0] > 0, 'the picker went back to quoting reductions'

    def test_the_conversion_gain_is_what_the_receiver_reported(self):
        """A drift pin against hardware.  The figure came from reading gainVals.curr
        at eleven settings on an RSP1B at 3530 kHz and adding the reduction back, which
        gave 91.0 to 91.6 across every LNA state.

        It is pinned because nothing else in the program would notice it moving, and a
        wrong conversion gain shifts every level the station reports by the difference
        while the sweep and the display carry on looking healthy.
        """
        assert 91.0 <= HF_CONVERSION_GAIN_DB <= 91.6

    def test_the_least_lna_reduction_that_fits_is_the_one_used(self):
        """Reduction at the front end costs noise figure where baseband reduction does
        not, so a total reachable two ways takes the quieter pair.

        The pairs here are the ones an RSP1B actually held: each was set through the
        real library and the gain it reported back agreed with the table to within half
        a decibel.  See tools/sdr_gain_probe.
        """
        assert SdrplayDevice._knobs_for(71.0) == (0, 20)
        assert SdrplayDevice._knobs_for(40.0) == (0, 51)
        # Past what LNA state 0 reaches, so the next state takes over.
        assert SdrplayDevice._knobs_for(31.0) == (1, 54)
        assert SdrplayDevice._knobs_for(-29.0) == (6, 59)

    def test_every_rung_of_the_ladder_splits_into_knobs_the_hardware_admits(self):
        """The ladder and the splitter are written apart, so nothing makes them agree.

        A rung that cannot be split would raise out of set_gain_db part way through a
        sweep, which is the one place it cannot be handled.
        """
        device, _ = make_device()
        for gain_db in device.supported_gains_db:
            lna_state, baseband = SdrplayDevice._knobs_for(gain_db)
            assert 0 <= lna_state < len(HF_LNA_GAIN_REDUCTION_DB)
            assert MIN_GAIN_REDUCTION_DB <= baseband <= MAX_GAIN_REDUCTION_DB
            total = HF_LNA_GAIN_REDUCTION_DB[lna_state] + baseband
            assert round(HF_CONVERSION_GAIN_DB - total) == gain_db

    def test_a_gain_outside_the_ladder_is_refused_by_name(self):
        with pytest.raises(ValueError, match='runs from -29 to 71 dB'):
            SdrplayDevice._knobs_for(-200)

    def test_a_request_is_snapped_to_the_ladder(self):
        """The ladder is whole decibels, so a fractional request has to move."""
        device, library = make_device()
        assert device.set_gain_db(40.4) == 40.0
        assert library.gain.gRdB == 51


class TestTheConversionGainComesFromTheReceiver:
    """Where the hundred decibels of ladder sit depends on the unit and the band.

    The figure measured on one RSP1B at 3530 kHz is a fallback, not the answer.  Asking
    the hardware is what stops a band change quietly relabelling every rung, which is
    the one thing about this that could go wrong without anything noticing.
    """

    def test_it_asks_the_hardware_rather_than_using_the_fallback(self):
        """The receiver here has 10 dB less conversion gain than the one the fallback
        came from, so every rung of its ladder sits 10 dB lower.
        """
        silent = FakeSdrplayApi()
        device, _ = make_device(silent)
        assert device.supported_gains_db[0] == 71.0    # the fallback, so far

        talkative = FakeSdrplayApi(conversion_gain_db=81.4)
        device, _ = make_device(talkative)
        assert device._conversion_gain_db == pytest.approx(81.4)
        assert device.supported_gains_db[0] == 61.0
        assert device.supported_gains_db[-1] == -39.0

    def test_the_gain_asked_for_is_the_gain_the_ladder_ends_at(self):
        """The first write goes out against the fallback, because the library reports
        nothing until a gain has been set.  So the device writes again once it knows
        what the rungs mean, or it would sit wherever the fallback pointed.
        """
        library = FakeSdrplayApi(conversion_gain_db=81.4)
        device, _ = make_device(library, gain_db=40.0)
        assert device.gain_db == 40.0
        assert device.gain_db in device.supported_gains_db
        # 81.4 less the 41 dB of reduction that a 40 dB gain needs on this receiver.
        # Written while the library is stopped, so it takes effect at the next init
        # rather than through an update.
        assert library.gain.gRdB == 41
        assert library.gain.LNAstate == 0

    def test_a_library_that_reports_nothing_keeps_the_measured_fallback(self, caplog):
        """The one case where nobody can do better, so it says so and carries on."""
        library = FakeSdrplayApi()
        with caplog.at_level(logging.INFO):
            device, _ = make_device(library)
        assert device._conversion_gain_db == HF_CONVERSION_GAIN_DB
        assert 'did not report its own gain' in caplog.text

    def test_the_ladder_is_still_a_hundred_and_one_whole_decibels(self):
        """Learning moves where the ladder sits and not what it is made of, because a
        sweep walks the rungs and a shifting spacing would change what it measured.
        """
        library = FakeSdrplayApi(conversion_gain_db=81.4)
        device, _ = make_device(library)
        ladder = device.supported_gains_db
        assert len(ladder) == 101
        assert all(a - b == 1.0 for a, b in zip(ladder, ladder[1:]))

    def test_every_rung_still_splits_into_knobs_the_hardware_admits(self):
        """The ladder and the splitter both take the learned figure, and nothing else
        makes them agree.  A rung that would not split raises out of set_gain_db part
        way through a sweep, which is the one place it cannot be handled.
        """
        library = FakeSdrplayApi(conversion_gain_db=81.4)
        device, _ = make_device(library)
        for gain_db in device.supported_gains_db:
            lna_state, baseband = SdrplayDevice._knobs_for(gain_db,
                                                           device._conversion_gain_db)
            assert 0 <= lna_state < len(HF_LNA_GAIN_REDUCTION_DB)
            assert MIN_GAIN_REDUCTION_DB <= baseband <= MAX_GAIN_REDUCTION_DB


class TestWhatTheReceiverSaysItsGainIs:
    def test_the_figure_returned_is_always_one_from_the_ladder(self):
        """The fault that made a sweep answer with the gain it started at.

        A sweep files each reading under whatever set_gain_db returns and then looks
        those keys up in supported_gains_db.  Returning the hardware's own figure put
        every reading under a key that list does not contain, so every gain but the
        first was dropped and the chooser had one measurement to work with.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.set_reported_gain_db(38.5)
        assert device.set_gain_db(40.0) == 40.0
        assert device.gain_db in device.supported_gains_db

    def test_the_hardware_figure_is_still_available_on_its_own(self):
        """For tools/sdr_gain_probe, which is what told the two apart."""
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.set_reported_gain_db(38.5)
        device.set_gain_db(40.0)
        assert device.reported_gain_db == 38.5

    def test_a_disagreement_is_reported_once_rather_than_every_time(self, caplog):
        """A stale table would otherwise shift every measurement with nothing to
        notice.  Once, because this runs on every gain step of a sweep.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.set_reported_gain_db(-38.5)
        with caplog.at_level(logging.WARNING):
            device.set_gain_db(40.0)
            device.set_gain_db(41.0)
        warnings = [r for r in caplog.records if 'gain table' in r.message]
        assert len(warnings) == 1

    def test_a_small_disagreement_is_left_alone(self, caplog):
        """The conversion gain came from one receiver at one frequency and is not flat
        across HF, so a decibel of difference elsewhere in the band is expected rather
        than a fault worth warning about.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.set_reported_gain_db(41.0)
        with caplog.at_level(logging.WARNING):
            device.set_gain_db(40.0)
        assert not [r for r in caplog.records if 'gain table' in r.message]

    def test_a_zero_reading_is_read_as_not_yet_applied(self):
        """The receiver fills the figure in as it applies the change, so a zero means
        the change has not taken effect and there is nothing to compare against.
        """
        device, library = make_device()
        library.set_reported_gain_db(0.0)
        device.start_stream(CollectingSink(), 4)
        assert device.set_gain_db(-25.0) == -25.0


class TestStreaming:
    def test_the_two_arrays_are_interleaved_into_one(self):
        """The whole reason this shim exists.  The library hands over separate I and Q
        where SampleFormat assumes interleaved, which is also what a .wav frame is.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 4)
        library.deliver([1, 2, 3, 4], [-1, -2, -3, -4])
        assert sink.blocks[0].raw.tolist() == [1, -1, 2, -2, 3, -3, 4, -4]

    def test_a_block_is_assembled_across_several_deliveries(self):
        """The library picks how many samples to deliver and it is not the block size
        anybody asked for, so the shim accumulates.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 4)
        library.deliver([1, 2], [1, 2])
        assert sink.blocks == []
        library.deliver([3, 4], [3, 4])
        assert len(sink.blocks) == 1
        assert sink.blocks[0].raw.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]

    def test_one_delivery_can_make_several_blocks(self):
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver(list(range(6)), list(range(6)))
        assert [block.raw.tolist() for block in sink.blocks] == [
            [0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]]

    def test_the_leftover_of_a_delivery_starts_the_next_block(self):
        """Samples are not dropped at a block boundary, which a naive split would do."""
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([1, 2, 3], [1, 2, 3])
        library.deliver([4], [4])
        assert [block.raw.tolist() for block in sink.blocks] == [
            [1, 1, 2, 2], [3, 3, 4, 4]]

    def test_the_samples_read_as_the_format_says(self):
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 1)
        library.deliver([16384], [-16384])
        assert sink.blocks[0].fmt is SDRPLAY_FORMAT
        assert sink.blocks[0].as_complex()[0] == pytest.approx(0.5 - 0.5j)

    def test_a_refused_block_is_counted_rather_than_raised(self):
        """This runs on the library's own thread, where an exception has nowhere to
        go, so a full sink is a counter rather than a failure.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(accept=False), 2)
        library.deliver([1, 2], [1, 2])
        assert device.blocks_refused == 1

    def test_a_second_stream_is_refused(self):
        """Two sinks on one device would silently drop the first one's reference."""
        device, _ = make_device()
        device.start_stream(CollectingSink(), 4)
        with pytest.raises(RuntimeError, match='already streaming'):
            device.start_stream(CollectingSink(), 4)

    def test_streaming_after_close_is_refused(self):
        device, _ = make_device()
        device.close()
        with pytest.raises(RuntimeError, match='because it is closed'):
            device.start_stream(CollectingSink(), 4)

    def test_stopping_a_stream_that_never_started_is_not_an_error(self):
        """close() calls it unconditionally, so it has to be answerable at any time."""
        device, _ = make_device()
        assert device.stop_stream() is True


class TestWhatTheDeviceReports:
    def test_it_reports_the_rate_and_the_tuning_it_was_given(self):
        """The rate is the decimated one rather than the converter's, because that is
        what a consumer reads samples at.  Nothing above this knows about decimation.
        """
        device, _ = make_device()
        assert device.iq_sample_rate == IQ_SAMPLE_RATE
        assert device.tuned_hz == TUNED_HZ

    def test_the_gain_cannot_move_after_close(self):
        device, _ = make_device()
        device.close()
        with pytest.raises(RuntimeError, match='because the device is closed'):
            device.set_gain_db(-50.0)

    def test_an_empty_delivery_is_ignored(self):
        """The library may call with nothing, and a zero-length copy would leave an
        empty array in the pending list for every one of them.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([], [])
        assert sink.blocks == []
        assert device.blocks_refused == 0

    def test_a_stream_can_be_stopped_and_started_again(self):
        """Stopping uninitializes the library, so starting again has to initialize it
        rather than assume it is still running.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(), 2)
        assert device.stop_stream() is True
        assert device.is_streaming is False
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([5, 5], [5, 5])
        assert len(sink.blocks) == 1


class TestTheSynchronousQueue:
    def test_it_refuses_rather_than_growing_when_the_reader_is_behind(self):
        """A synchronous reader wants the newest samples, so a deep queue would hand it
        a backlog to work through instead of what the receiver is hearing now.
        """
        blocks = _SyncBlocks(2)
        made = [IqBlock(raw=np.zeros(2, dtype=np.int16), fmt=SDRPLAY_FORMAT,
                        arrived_at=0.0, index=index) for index in range(3)]
        assert [blocks.offer(block) for block in made] == [True, True, False]

    def test_it_gives_up_rather_than_waiting_forever(self):
        """Nothing arriving means the library has stopped, and a reader blocked on a
        queue would hang the sweep rather than end it.
        """
        assert _SyncBlocks(1).take(0.01) is None


class TestDiscardingAfterAGainChange:
    def test_blocks_before_the_grchanged_marker_are_dropped(self):
        """The hardware says where the change took effect, so nothing has to be
        counted.  An RTL-SDR has no such marker and discards a fixed number.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([1, 1], [1, 1])
        device.set_gain_db(-60.0)
        library.deliver([9, 9], [9, 9])                      # stale
        library.deliver([2, 2], [2, 2], gr_changed=True)     # the boundary
        library.deliver([3, 3], [3, 3])                      # the new gain
        assert [block.raw.tolist() for block in sink.blocks] == [
            [1, 1, 1, 1], [3, 3, 3, 3]]

    def test_a_dropped_block_leaves_a_gap_in_the_index(self):
        """index counts what the receiver produced rather than what survived, so a
        consumer can tell that something went missing and how much.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([1, 1], [1, 1])
        device.set_gain_db(-60.0)
        library.deliver([9, 9], [9, 9])
        library.deliver([2, 2], [2, 2], gr_changed=True)
        library.deliver([3, 3], [3, 3])
        assert [block.index for block in sink.blocks] == [1, 3]

    def test_the_drop_gives_up_rather_than_waiting_for_a_marker_forever(self):
        """The fault that made a gain sweep report the lowest gain on the ladder.

        Nobody has confirmed that a real RSP sets `grChanged`, and without a bound a
        library that never sets it drops every block from the first gain change
        onwards.  The receiver then goes silent, every gain after the first measures
        nothing, and the only gain with readings is the one the sweep started at.
        """
        device, library = make_device()
        sink = CollectingSink()
        device.start_stream(sink, 2)
        device.set_gain_db(-60.0)
        for _ in range(device._settle_blocks + 1):
            library.deliver([4, 4], [4, 4])      # never marked
        assert sink.blocks, 'the stream never recovered from a gain change'

    def test_it_says_so_once_when_the_marker_never_comes(self, caplog):
        """A sweep moves the gain hundreds of times, so this cannot warn per change."""
        device, library = make_device()
        device.start_stream(CollectingSink(), 2)
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                device.set_gain_db(-60.0)
                for _ in range(device._settle_blocks + 1):
                    library.deliver([4, 4], [4, 4])
        assert len([r for r in caplog.records if 'marking one as changed' in r.message]) == 1

    def test_a_queued_block_from_before_the_change_is_thrown_away(self):
        """The drop covers what the library has yet to deliver and cannot reach a block
        already sitting in the reader's queue.

        A sweep reads one block after moving the gain, so a stale one left in the queue
        is not a delay: it is the measurement, taken at the previous gain.
        """
        device, library = make_device()
        timer = _once_reading(device, lambda: library.deliver([1, 1], [1, 1]))
        device.read_block(2)
        timer.join()
        library.deliver([9, 9], [9, 9])          # queued at the old gain
        device.set_gain_db(-60.0)
        for _ in range(device._settle_blocks + 1):
            library.deliver([5, 5], [5, 5])
        block = device.read_block(2)
        assert block is not None
        assert block.raw.tolist() == [5, 5, 5, 5], 'a pre-change block was served'

    def test_the_settling_ceiling_follows_the_rate_and_the_block_size(self):
        """Derived rather than fixed, so a different block size does not silently
        change how long the device waits.
        """
        device, _ = make_device()
        device.start_stream(CollectingSink(), 2048)
        assert device._settle_blocks == 63       # 0.5 s of 256 kHz, in 2048s

    def test_a_gain_change_before_any_consumer_still_clears_itself(self):
        """The library runs from the moment the receiver opens, so a gain change here
        does send an update and does arm the discard.

        What must not happen is the arming outliving the change.  Before the ceiling
        existed this was the shape that dropped every block of the stream that
        followed, with nothing able to clear it.
        """
        device, library = make_device()
        device.set_gain_db(31.0)
        sink = CollectingSink()
        device.start_stream(sink, 2)
        for _ in range(device._settle_blocks + 1):
            library.deliver([1, 1], [1, 1])
        assert sink.blocks, 'the discard outlived the gain change'

    def test_a_marked_block_clears_it_without_waiting_for_the_ceiling(self):
        device, library = make_device()
        device.set_gain_db(31.0)
        sink = CollectingSink()
        device.start_stream(sink, 2)
        library.deliver([2, 2], [2, 2], gr_changed=True)
        library.deliver([3, 3], [3, 3])
        assert [block.raw.tolist() for block in sink.blocks] == [[3, 3, 3, 3]]

    def test_the_gain_moves_through_update_rather_than_being_refused(self):
        """The API supports a gain change during a stream, so this device answers the
        opposite way to an RTL-SDR, which wedges if you try it.
        """
        device, library = make_device()
        assert device.profile.gain_changes_while_streaming is True
        device.start_stream(CollectingSink(), 4)
        device.set_gain_db(-60.0)
        assert library.updates[-1][2] == (
            api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Tuner_Gr)


class TestSynchronousReads:
    def test_a_read_runs_a_stream_of_its_own(self):
        """There is no synchronous call in this API at all, so the only way to serve
        one is to run a stream and take from it.
        """
        device, library = make_device()

        def deliver_once() -> None:
            library.deliver([7, 8], [7, 8])

        timer = _once_reading(device, deliver_once)
        block = device.read_block(2)
        timer.join()
        assert block is not None
        assert block.raw.tolist() == [7, 7, 8, 8]

    def test_the_public_stream_flag_stays_false_through_one(self):
        """is_streaming answers whether a caller asked for a stream, not whether the
        library is delivering.  A sweep reads synchronously and nothing above it should
        conclude that a capture is running.
        """
        device, library = make_device()
        timer = _once_reading(device, lambda: library.deliver([1, 1], [1, 1]))
        device.read_block(2)
        timer.join()
        assert device.is_streaming is False

    def test_a_stream_while_reading_synchronously_is_refused(self):
        """Both use the one library callback, so the second would take the first's
        deliveries and the first would simply stop receiving any.
        """
        device, library = make_device()
        timer = _once_reading(device, lambda: library.deliver([1, 1], [1, 1]))
        device.read_block(2)
        timer.join()
        with pytest.raises(RuntimeError, match='serving synchronous reads'):
            device.start_stream(CollectingSink(), 2)

    def test_a_read_that_gets_nothing_says_so_and_gives_up(self, monkeypatch, caplog):
        """A sweep reads in a loop, so a read that blocks forever would hang it rather
        than end it, and the operator would see a dialog that never finishes.
        """
        monkeypatch.setattr('buzz.sdrplay_device._SYNC_READ_TIMEOUT_SECONDS', 0.01)
        device, _ = make_device()
        with caplog.at_level(logging.WARNING):
            assert device.read_block(2) is None
        assert 'produced no samples' in caplog.text

    def test_a_read_while_streaming_is_refused(self):
        device, _ = make_device()
        device.start_stream(CollectingSink(), 4)
        with pytest.raises(RuntimeError, match='while it is streaming'):
            device.read_block(4)

    def test_a_read_after_close_gives_nothing_rather_than_raising(self):
        """A sweep reads in a loop, and the end of the device is not an error there."""
        device, _ = make_device()
        device.close()
        assert device.read_block(4) is None


class TestEvents:
    def test_shutdown_events_cannot_spoil_the_next_capture(self) -> None:
        class ClearanceDuringShutdown(FakeSdrplayApi):
            def uninit(self, handle: int) -> None:
                super().uninit(handle)
                self.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                                 api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Corrected)

        device, library = make_device(ClearanceDuringShutdown())
        assert device.overload_status == OverloadStatus(False, 0)
        assert library.updates == []
        assert library.calls.count('update') == 0
        device.start_stream(CollectingSink(), 4)
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        assert device.overload_status == OverloadStatus(True, 1)
        device.stop_stream()
        device.start_stream(CollectingSink(), 4)
        assert device.overload_status == OverloadStatus(False, 1)
        assert len(library.updates) == 1
        device.close()
        assert library.calls.count('update') == 1

    def test_a_failed_start_does_not_accept_further_overload_events(self) -> None:
        device, library = make_device()
        library.fail_on['init'] = RuntimeError('startup failed')
        with pytest.raises(RuntimeError, match='startup failed'):
            device.start_stream(CollectingSink(), 4)
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        assert device.overload_status == OverloadStatus(False, 0)
        assert library.calls.count('update') == 0

    def test_events_inside_init_are_acknowledged_before_init_returns(self) -> None:
        class DetectionDuringStartup(FakeSdrplayApi):
            def init(self, handle: int, callbacks: api.sdrplay_api_CallbackFnsT) -> None:
                super().init(handle, callbacks)
                self.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                                 api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)

        device, library = make_device(DetectionDuringStartup())
        assert device.overload_status == OverloadStatus(True, 1)
        assert library.updates == [
            (int(library.device.dev), api.sdrplay_api_TunerSelectT.sdrplay_api_Tuner_A,
             api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Ctrl_OverloadMsgAck)]

    def test_shutdown_during_acknowledgement_does_not_latch_a_capture_error(self) -> None:
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)

        def shutdown_in_update(handle: int, tuner: int, reason: int) -> None:
            device.stop_stream()
            raise RuntimeError('API stopped during acknowledgement')

        library.update = shutdown_in_update
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        device.start_stream(CollectingSink(), 4)
        assert device.overload_status == OverloadStatus(False, 1)

    def test_an_overload_is_counted(self) -> None:
        """Hardware reports overload independently of delivered sample endpoints."""
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.raise_event(
            api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        assert device.overloads == 1

    def test_the_correction_of_an_overload_is_not_counted_as_another(self):
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.raise_event(
            api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Corrected)
        assert device.overloads == 0

    def test_another_event_is_ignored(self) -> None:
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_DeviceRemoved)
        assert device.overloads == 0
        assert library.updates == []

    def test_reader_retains_state_and_counts_across_detection_and_clearance(self) -> None:
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        reader = SweepReader(device)
        initial = reader.overload_status
        assert initial == OverloadStatus(False, 0)

        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        detected = reader.overload_status
        assert detected == OverloadStatus(True, 1)

        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Corrected)
        assert reader.overload_status == OverloadStatus(False, 1)
        assert initial == OverloadStatus(False, 0)
        assert detected == OverloadStatus(True, 1)

        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        assert reader.overload_status == OverloadStatus(True, 2)

    @pytest.mark.parametrize('event', [
        api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected,
        api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Corrected,
    ])
    @pytest.mark.parametrize('tuner', [api.sdrplay_api_TunerSelectT.sdrplay_api_Tuner_A,
                                      api.sdrplay_api_TunerSelectT.sdrplay_api_Tuner_B])
    def test_both_overload_events_acknowledge_the_tuner_from_the_callback(
            self, event: int, tuner: int) -> None:
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        params = api.sdrplay_api_EventParamsT()
        params.powerOverloadParams.powerOverloadChangeType = event
        library.callbacks.EventCbFn(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                                    tuner, ctypes.pointer(params), None)
        assert library.updates == [
            (int(library.device.dev), tuner,
             api.sdrplay_api_ReasonForUpdateT.sdrplay_api_Update_Ctrl_OverloadMsgAck)]

    @pytest.mark.parametrize('message', ['service disconnected', 'service disconnected.'])
    def test_acknowledgement_failure_reaches_the_reader_without_escaping_the_callback(
            self, capsys: pytest.CaptureFixture[str], message: str) -> None:
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        library.fail_on['update'] = RuntimeError(message)
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Detected)
        assert capsys.readouterr().err == ''
        with pytest.raises(RuntimeError, match='service disconnected.*unreliable') as failure:
            _ = SweepReader(device).overload_status
        assert 'disconnected.  Hardware' in str(failure.value)

        library.fail_on.clear()
        library.raise_event(api.sdrplay_api_EventT.sdrplay_api_PowerOverloadChange,
                            api.sdrplay_api_PowerOverloadCbEventIdT.sdrplay_api_Overload_Corrected)
        with pytest.raises(RuntimeError, match='restart the probe'):
            _ = device.overload_status


class TestClosing:
    def test_the_receiver_and_the_session_are_both_given_back(self):
        """A session left open keeps the service holding the receiver, so the next run
        finds a device that is present and cannot be selected.
        """
        device, library = make_device()
        assert device.close() is True
        assert library.calls[-2:] == ['release', 'close']
        assert library.session_open is False

    def test_a_receiver_that_streamed_is_stopped_before_it_is_released(self):
        """Releasing a device the library is still delivering from is the same hazard
        as closing an RTL-SDR handle a thread is reading through.
        """
        device, library = make_device()
        device.start_stream(CollectingSink(), 4)
        assert device.close() is True
        assert library.calls[-3:] == ['uninit', 'release', 'close']

    def test_a_receiver_is_always_stopped_because_it_is_always_started(self):
        """The library is initialized at open rather than at the first read, so that it
        has filled in gainVals.curr before anything asks for the gain ladder.  Every
        device therefore has something to stop by the time it closes.
        """
        device, library = make_device()
        assert 'init' in library.calls
        device.close()
        assert library.calls.index('uninit') > library.calls.index('init')

    def test_it_is_safe_to_call_twice(self):
        """Shutdown calls it and the atexit hook fires afterwards regardless."""
        device, library = make_device()
        assert device.close() is True
        before = len(library.calls)
        assert device.close() is True
        assert len(library.calls) == before

    def test_a_library_that_will_not_let_go_is_reported_rather_than_waited_on(self):
        """The API waits on a background service, so a call can hang.  Waiting forever
        at shutdown would hang the monitor instead of the one call.
        """
        device, library = make_device()
        library.fail_on['uninit'] = RuntimeError('the service stopped answering')
        assert device.close() is True     # the failure is swallowed, not raised


class TestOpening:
    def test_the_api_lock_is_held_across_listing_and_selecting(self):
        """The service is shared with every other program on the machine, and the
        window between listing and selecting is where two of them collide.  The fake
        asserts the lock is held, so this fails rather than merely looking right.
        """
        library = FakeSdrplayApi()
        SdrplayDevice._select(library, 0)
        assert library.calls[:4] == ['lock', 'devices', 'select', 'unlock']

    def test_a_receiver_that_is_not_an_rsp1_is_not_offered(self):
        """Every other RSP needs a different LNA table, and two embed parameter structs
        this device never writes, so admitting one would claim untested support.
        """
        library = FakeSdrplayApi(hw_ver=3)     # an RSPduo
        with pytest.raises(RuntimeError, match='No RSP1A or RSP1B was found'):
            SdrplayDevice._select(library, 0)

    def test_an_empty_list_says_something_different_from_a_wrong_device(self):
        """Nothing plugged in and the wrong thing plugged in need different answers."""
        library = FakeSdrplayApi(devices=0)
        with pytest.raises(RuntimeError, match='SDRplayAPIService is not running'):
            SdrplayDevice._select(library, 0)

    def test_the_lock_is_given_back_even_when_selecting_fails(self):
        """A lock left held stops every other program using any SDRplay receiver."""
        library = FakeSdrplayApi()
        library.fail_on['select'] = RuntimeError('taken')
        with pytest.raises(RuntimeError):
            SdrplayDevice._select(library, 0)
        assert library.locked is False


class TestTheLibraryLoader:
    def test_the_setting_is_tried_before_anything_else(self):
        """An operator with two API versions installed has to be able to say which."""
        candidates = SdrplayLibrary._candidates('/opt/sdrplay/libsdrplay_api.so')
        assert candidates[0] == Path('/opt/sdrplay/libsdrplay_api.so')

    def test_both_linux_spellings_are_tried(self):
        """A runtime-only install has libsdrplay_api.so.3 and no development symlink."""
        names = {path.name for path in SdrplayLibrary._candidates(None)}
        assert {'sdrplay_api.dll', 'libsdrplay_api.so',
                'libsdrplay_api.so.3'} <= names

    def test_a_plain_name_comes_before_a_guess_about_where_it_lives(self):
        """A library already on the search path is the one the operator installed."""
        candidates = [str(path) for path in SdrplayLibrary._candidates(None)]
        assert candidates.index('sdrplay_api.dll') < min(
            index for index, path in enumerate(candidates) if 'Program Files' in path)

    def test_no_candidate_is_tried_twice(self):
        candidates = [str(path) for path in SdrplayLibrary._candidates(None)]
        assert len(candidates) == len(set(candidates))

    def test_a_failure_with_a_setting_blames_the_setting(self):
        """Somebody who pointed the program at a path wants to hear about that path,
        not about where the program would have looked otherwise.
        """
        message = SdrplayLibrary._why_the_library_would_not_load('/nowhere/libx.so')
        assert '/nowhere/libx.so' in message
        assert 'api_path' in message

    def test_a_failure_with_no_setting_says_where_to_get_the_api(self):
        """SDRconnect installs the driver and not the API, and having it working is
        what makes an operator sure the API must be there.
        """
        message = SdrplayLibrary._why_the_library_would_not_load(None)
        assert 'sdrplay.com/hardware-api' in message
        assert 'SDRconnect' in message


class TestTheSampleFormat:
    def test_it_describes_signed_16_bit_pairs(self):
        assert SDRPLAY_FORMAT.dtype == np.dtype(np.int16)
        assert SDRPLAY_FORMAT.bytes_per_frame == 4
        assert SDRPLAY_FORMAT.rail_low == -32768
        assert SDRPLAY_FORMAT.rail_high == 32767

    def test_full_scale_reads_as_about_plus_and_minus_one(self):
        """A signed range is one step short at the top, which SampleFormat documents.
        The reading has to come out on the same scale as an RTL-SDR's all the same.
        """
        block = IqBlock(raw=np.array([-32768, 32767], dtype=np.int16),
                        fmt=SDRPLAY_FORMAT, arrived_at=0.0, index=1)
        value = block.as_complex()[0]
        assert value.real == pytest.approx(-1.0)
        assert value.imag == pytest.approx(1.0, abs=1e-4)

    def test_a_sample_at_a_rail_counts_as_clipped(self):
        block = IqBlock(raw=np.array([-32768, 0, 32767, 5], dtype=np.int16),
                        fmt=SDRPLAY_FORMAT, arrived_at=0.0, index=1)
        assert block.clipped_samples == 2


def _once_reading(device: SdrplayDevice, deliver: object) -> object:
    """Run `deliver` once `read_block` has somewhere to put what arrives.

    `read_block` attaches its queue and then waits, so a delivery has to come from
    another thread.  Waiting on the library being initialized is not enough any more,
    because that happens at open: a block delivered before the queue exists is
    discarded, which is right for the device and leaves the reader waiting forever.

    It waits on the condition rather than sleeping, for the reason `CLAUDE.md` gives
    about `pilot.pause`: a sleep has to suit the slowest machine that will ever run it.
    """
    def run() -> None:
        deadline = time.monotonic() + 5.0
        while device._sync_sink is None and time.monotonic() < deadline:
            time.sleep(0.001)
        deliver()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_the_fake_fills_in_a_real_device_struct():
    """The fake is only useful if it exercises the generated layout.

    A serial that reads back as text out of a 64-byte array is what says the struct is
    laid out the way the C compiler lays it out, which is the same check the hardware
    made when these bindings were first written.
    """
    library = FakeSdrplayApi(serial='2405203460')
    assert bytes(library.device.SerNo).rstrip(b'\x00') == b'2405203460'
    assert library.device.hwVer == api.SDRPLAY_RSP1B_ID
    assert ctypes.sizeof(library.device) == 96

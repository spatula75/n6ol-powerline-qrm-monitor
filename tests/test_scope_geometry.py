"""How the scope cuts its trace out of the audio, at any sample rate and either grid.

Separate from test_scope_math.py because this is one idea rather than a collection of
helpers: the display should say the same thing whatever the audio arrives at, and
everything here is a consequence of that.

Before this, one sweep was the pulse grid's *phase* period -- rate // gcd(rate,
pulse_rate) -- which is number-theoretic rather than proportional.  It landed on 1 pulse
period at 48 kHz and 8 at 11.025 kHz, so the time base followed a gcd instead of the
signal: the same interference read 0.83 ms/div in one file and 6.67 in another.
"""

from math import floor

import numpy as np
import pytest

from buzz.scope import (
    _PHOSPHOR_PULSES,
    _SWEEP_PULSES,
    extract_sweeps,
    n_complete_sweeps,
    sweep_geometry,
)

# Rates a recording can plausibly arrive at, and both grid frequencies: 120 pps for a
# 60 Hz grid, 100 pps for 50 Hz.
RATES = (8000, 11025, 16000, 22050, 32000, 44100, 48000)
PULSE_RATES = (120, 100)
COMBINATIONS = [(rate, pulse_rate) for pulse_rate in PULSE_RATES for rate in RATES]


class TestTheTraceShowsTheSamePicture:

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_sweep_is_always_the_same_number_of_pulses(self, rate, pulse_rate):
        geometry = sweep_geometry(rate, pulse_rate)
        pulses = geometry.sweep_samples / (rate / pulse_rate)
        assert pulses == pytest.approx(_SWEEP_PULSES, abs=0.01), (
            f'At {rate} Hz / {pulse_rate} pps the trace shows {pulses:.2f} pulse '
            f'periods rather than {_SWEEP_PULSES}. The sweep width is meant to be '
            'counted in pulse periods, not taken from the phase period.')

    @pytest.mark.parametrize('rate', RATES)
    def test_time_per_division_is_constant_for_a_grid(self, rate):
        """The number the operator reads off the header, which used to depend on the
        sample rate of the file they happened to open."""
        assert sweep_geometry(rate, 120).ms_per_division == pytest.approx(2.5, abs=0.02)

    @pytest.mark.parametrize('rate', RATES)
    def test_a_fifty_hertz_grid_gets_a_longer_division(self, rate):
        """3.00 ms/div at 100 pps, which is correct rather than a compromise: a 50 Hz
        grid has a longer pulse period, so three cycles take longer. Forcing 2.50 would
        show two and a half cycles -- the same picture stretched."""
        assert sweep_geometry(rate, 100).ms_per_division == pytest.approx(3.0, abs=0.02)

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_phosphor_holds_the_same_number_of_pulses(self, rate, pulse_rate):
        geometry = sweep_geometry(rate, pulse_rate)
        pulses = geometry.phosphor_seconds * pulse_rate
        assert pulses == pytest.approx(_PHOSPHOR_PULSES, abs=0.5), (
            f'At {rate} Hz / {pulse_rate} pps the phosphor holds {pulses:.1f} pulse '
            f'periods rather than {_PHOSPHOR_PULSES}.')

    @pytest.mark.parametrize('rate', RATES)
    def test_the_phosphor_lasts_the_same_time_for_a_grid(self, rate):
        assert sweep_geometry(rate, 120).phosphor_seconds == pytest.approx(0.2, abs=0.005)

    def test_sixteen_kilohertz_is_unchanged(self):
        """The rate this program records at, where the phase period happened to be
        exactly three pulse periods -- so none of this moves the familiar display."""
        geometry = sweep_geometry(16000, 120)
        assert geometry.sweep_samples == 400
        assert geometry.stride_samples == 400
        assert geometry.ms_per_division == pytest.approx(2.5)


class TestSweepsSuperimpose:
    """Successive sweeps must start a whole phase period apart.

    That is the only interval after which the pulse grid repeats against the sample
    clock.  Step by anything else and each sweep samples the pulses at a different
    sub-sample offset, so they no longer overlay and the trace smears -- quietly
    undoing the drift tracking the scope exists to show.
    """

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_stride_is_a_whole_number_of_phase_periods(self, rate, pulse_rate):
        geometry = sweep_geometry(rate, pulse_rate)
        assert geometry.stride_samples % geometry.phase_period == 0, (
            f'At {rate} Hz / {pulse_rate} pps the stride is {geometry.stride_samples} '
            f'against a phase period of {geometry.phase_period}, so consecutive sweeps '
            'sample the grid at different offsets and will not superimpose.')

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_stride_covers_the_sweep(self, rate, pulse_rate):
        """Sweeps may sit further apart than the trace is wide -- a real scope
        re-triggers on a running signal -- but a stride shorter than the trace would
        draw the same pulses twice at different screen positions."""
        geometry = sweep_geometry(rate, pulse_rate)
        assert geometry.stride_samples >= geometry.sweep_samples

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_pulses_land_identically_in_every_sweep(self, rate, pulse_rate):
        """What all of the above exists to produce, checked on real samples rather than
        on the arithmetic: build a pulse train, cut it up the way the widget does, and
        every sweep must carry its pulses in exactly the same columns."""
        geometry = sweep_geometry(rate, pulse_rate)
        count = geometry.capture_samples + geometry.phase_period
        samples = np.zeros(count, dtype=np.float32)
        spacing = rate / pulse_rate
        for index in range(int(count / spacing)):
            # floor(x + 0.5), not round(): Python rounds half to even, so at a spacing
            # like 110.25 the offset 220.5 goes to 220 while 661.5 goes to 662 -- the
            # same fractional position landing differently by the parity of its integer
            # part, which breaks the grid's periodicity in the fixture rather than in
            # the code.  The project avoids banker's rounding wherever phase matters
            # for exactly this reason; a test standing in for the pulse grid has to
            # follow the same rule.
            at = floor(index * spacing + 0.5)
            if at < count:
                samples[at] = 1.0
        sweeps = extract_sweeps(samples, 0, geometry.sweep_samples,
                                geometry.stride_samples, geometry.n_sweeps)
        assert len(sweeps) == geometry.n_sweeps
        patterns = {tuple(np.flatnonzero(sweep > 0.5)) for sweep in sweeps}
        assert len(patterns) == 1, (
            f'At {rate} Hz / {pulse_rate} pps the pulses fall in {len(patterns)} '
            'different places from sweep to sweep, so the traces will not superimpose.')


class TestTheGeometryIsAlwaysUsable:

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_the_capture_supplies_its_sweeps(self, rate, pulse_rate):
        geometry = sweep_geometry(rate, pulse_rate)
        assert n_complete_sweeps(geometry.capture_samples, 0, geometry.sweep_samples,
                                 geometry.stride_samples) >= geometry.n_sweeps

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_it_survives_the_worst_trigger_offset(self, rate, pulse_rate):
        """The trigger can push the first sweep almost a whole phase period in, which
        is why the capture carries that much slack at the head."""
        geometry = sweep_geometry(rate, pulse_rate)
        remaining = n_complete_sweeps(
            geometry.capture_samples, geometry.phase_period - 1,
            geometry.sweep_samples, geometry.stride_samples)
        assert remaining >= geometry.n_sweeps - 1

    @pytest.mark.parametrize('rate,pulse_rate', COMBINATIONS)
    def test_every_figure_is_sane(self, rate, pulse_rate):
        geometry = sweep_geometry(rate, pulse_rate)
        assert geometry.sweep_samples > 0
        assert geometry.stride_samples > 0
        assert geometry.n_sweeps >= 1
        assert 0 <= geometry.pretrigger < geometry.sweep_samples

    def test_an_awkward_rate_degrades_rather_than_breaking(self):
        """8001 Hz repeats only every 40 pulse periods, so 24 cannot divide it evenly.
        The phosphor holds fewer, longer sweeps instead of producing nonsense, and the
        time base is still right."""
        geometry = sweep_geometry(8001, 120)
        assert geometry.n_sweeps >= 1
        assert geometry.stride_samples % geometry.phase_period == 0
        assert geometry.ms_per_division == pytest.approx(2.5, abs=0.02)


class TestStridedExtraction:
    """extract_sweeps and n_complete_sweeps once stride and width are separate."""

    def test_a_stride_equal_to_the_width_still_tiles(self):
        """The behaviour from when the two were necessarily the same."""
        samples = np.arange(40, dtype=np.float32)
        assert extract_sweeps(samples, 0, 10).shape == (4, 10)
        assert n_complete_sweeps(40, 0, 10) == 4

    def test_a_wider_stride_skips_audio_between_sweeps(self):
        sweeps = extract_sweeps(np.arange(100, dtype=np.float32), 0, 10, stride=25)
        assert sweeps.shape == (4, 10)
        assert [int(sweep[0]) for sweep in sweeps] == [0, 25, 50, 75]

    def test_it_starts_where_it_is_told(self):
        assert int(extract_sweeps(np.arange(100, dtype=np.float32),
                                  7, 10, stride=25)[0][0]) == 7

    def test_the_limit_caps_the_count(self):
        samples = np.arange(100, dtype=np.float32)
        assert extract_sweeps(samples, 0, 10, stride=25, limit=2).shape == (2, 10)

    def test_a_partial_final_sweep_is_dropped(self):
        """A half-drawn trace running into a flat line would read as signal."""
        # 35 samples is exactly two: the second sweep occupies 25..34.
        assert n_complete_sweeps(35, 0, 10, 25) == 2
        assert extract_sweeps(np.arange(35, dtype=np.float32), 0, 10, 25).shape == (2, 10)
        # One fewer and the second sweep cannot complete.
        assert n_complete_sweeps(34, 0, 10, 25) == 1

    def test_nothing_fitting_returns_an_empty_stack(self):
        empty = extract_sweeps(np.arange(5, dtype=np.float32), 0, 10, 25)
        assert empty.shape == (0, 10)
        assert n_complete_sweeps(5, 0, 10, 25) == 0

    def test_a_start_past_the_end_returns_nothing(self):
        assert n_complete_sweeps(20, 50, 10, 25) == 0

    def test_overlapping_strides_are_allowed(self):
        """Not what the widget asks for, but the arithmetic should not depend on it."""
        sweeps = extract_sweeps(np.arange(30, dtype=np.float32), 0, 10, stride=5)
        assert [int(sweep[0]) for sweep in sweeps] == [0, 5, 10, 15, 20]

# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Phase-synchronised oscilloscope panel above the waterfall (`buzz.scope`). The
  sweep is triggered from the analyzer's tracked pulse phase rather than an
  amplitude threshold, so a 120 pps arc renders as a standing wave instead of
  sliding across the screen. CRT-style phosphor persistence makes pulse-to-pulse
  jitter visible as a halo around the trace. Press `A` to switch between the raw
  bipolar view and a coherently-averaged rectified envelope. The vertical scale
  auto-ranges from the signal and is reported in dBFS; the trigger indicator
  reads `LOCK`, `HOLD` (extrapolating through a fade) or `FREE` (never locked).
- `ContinuousAnalyzer.trigger_phase()` — thread-safe accessor returning the
  predicted pulse phase plus a `TriggerSync` confidence level, for display sync.

### Changed
- Main window is now 734×248 (was 726×224). The scope and waterfall each occupy
  120 px, with 8 px of padding between them and before the meter column.
  Waterfall history is 4.8 s (48 rows), down from 10 s.

### Fixed
- `ContinuousAnalyzer._record_phase_measurement()` now writes the phase pair and
  its measurement timestamp as one locked group. They are read together by
  `trigger_phase()` on the Qt thread, where a read landing between the two writes
  would project a fresh phase across a stale interval and mis-place the trigger.

## [1.1.0] — 2026-07-27

Continuous live analysis and display replace the old once-a-minute sampling
loop, plus a DSP correctness pass and utility-line drift tracking on top of it.

### Added
- Continuously-running audio pipeline (`AudioPipeline`) and a
  `ContinuousAnalyzer` state machine (`SEARCHING` → `LOCKED` → `SIGNAL_LOST`,
  with tiered re-acquisition) replacing per-minute FFT sampling — sub-second
  signal/noise readings instead of once-a-minute snapshots.
- Live PySide6 waterfall display and S-band meter panel (run without
  `--headless` to open it); `--top` keeps the window always on top.
- Utility-line drift tracking: the analyzer estimates how fast grid frequency
  is drifting and uses it to predict the pulse phase forward between refines
  and correct the sample spacing within an analysis window, removing 5–7 dB
  of systematic level bias that existed at ordinary drift rates.
  `phase_drift_rate()` and `grid_frequency_hz()` expose the estimate; grid
  frequency and phase drift are now logged to the daily CSV (columns inserted
  after Signal Lock Status, so old files still parse unchanged).
- `tools/pulse_probe.py` — diagnostic that measures the real pulse shape,
  drift rate, and number of active mains phases from live audio, for future
  tuning of `PULSE_WIDTH_SAMPLES`.
- `level_meter.py` — live text S-meter for receiver gain calibration.  Displays
  a continuously-updating 21-char bar (S1–S9 linear, then +20/+40/+60 sections
  with 3 ticks each) plus dBm and S-unit readout.  Uses a persistent
  callback-based PortAudio stream (DirectSound blocking I/O is unreliable on
  Windows) at 20 ms per frame.  Flicker-free: each refresh overwrites in place
  without an intermediate clear.
- `AudioSampler.level_stream()` / `LevelStream` in `sampler.py` — persistent
  callback-driven input stream; `.read()` blocks on a `threading.Event` until
  the next hardware buffer fires.

### Changed
- Extracted the pulse-train DSP core into `buzz.dsp` (kernel builder, FFT fit
  array, Numba amplitude averaging, dBFS conversion, and a shared
  `analyze_window()`); `sampler.py` is now pure audio I/O and `analyzer.py`
  pure state machine.
- Centralised `ContinuousAnalyzer` state transitions: tier methods return the
  state they propose and `_transition()` owns all bookkeeping (debounce reset,
  phase validation, refine-timer stamp); per-state tick methods own cadence.
- Pulse kernel and amplitude averager now agree exactly on pulse positions
  (both round to the nearest sample, where the kernel previously truncated);
  phase reduction uses the exact integer repeat period instead of a
  fractional modulus that could misround near a period boundary.
- DSP amplitudes and FFT fit scores kept as floating point instead of being
  floor-divided to integers, preserving resolution near the noise floor.
- DC offset removal switched from mean to median before rectification — the
  receiver runs LSB, and a mean gets dragged by the pulse train itself,
  injecting an error that grows with signal strength.
- `LOCK_LOSE_SNR` raised 2.0 → 3.0 dB and `FAST_SCAN_SNR` raised 4.0 → 8.0 dB,
  both retuned against measured pure-noise statistics rather than guesses.
- Waterfall FFT frames now overlap 75% (was non-overlapping) so Hann satisfies
  COLA in power, not just amplitude, and averages power instead of magnitude;
  corrected a noise-floor anchor that was 4.4 dB off.  The colour scale now
  auto-ranges continuously off the live spectrum (10th/98th percentile floor
  and ceiling, with headroom reserved for transients) instead of a fixed
  calibration that goes stale with receiver or band conditions, floored at a
  minimum dynamic range so a genuinely quiet band can't paint itself warm
  from measurement noise alone.
- Weather fetches now have a 10 s timeout, and a failed fetch degrades to
  blank weather fields instead of dropping the minute's CSV row.
- CSV row parsing moved into `CsvStore.read_rows()`; the plotter consumes
  typed rows instead of parsing files itself.
- Waterfall display derives its bin geometry and frequency axis from the
  configured sample rate instead of a hardcoded 16 kHz.
- SCP uploads now verify the server host key against known_hosts
  (`~/.ssh/known_hosts` plus optional `~/.buzz/known_hosts`) instead of
  auto-accepting any key — closes a man-in-the-middle vector; add new hosts
  with `ssh-keyscan <host> >> ~/.buzz/known_hosts`.
- Collector uploads are gated on the publisher's presence rather than
  re-checking `server.enabled`.
- Unknown `[weather] source` values now log a warning instead of silently
  disabling weather.
- `ContinuousAnalyzer._run()` now catches and logs a tick failure and retries,
  instead of an uncaught exception silently killing analysis for the rest of
  the session.
- Coverage measurement no longer excludes all of `waterfall.py`; only the
  three Qt widget classes are marked uncovered, so the module's pure
  functions count toward the total.  Coverage gate raised 90% → 96%.
- Minimum supported Python raised to 3.12 (3.11 dropped from CI and packaging
  metadata).
- Window title corrected to "N6OL Powerline QRM Monitor".

### Fixed
- `generate_summary_graph()` leaked a matplotlib figure when there was no data
  to plot (up to three figures per hour on a quiet station).
- `analyze_window()` returns None instead of crashing when the audio window is
  shorter than the scan kernel.
- `phase_drift_rate()` / `grid_frequency_hz()` read analyzer state with no
  synchronization; now locked consistently with the rest of the class's
  cross-thread reads.
- Headless mode didn't stop the analyzer thread before closing the audio
  pipeline, risking a race against an already-closed stream during shutdown;
  now stops the analyzer first, matching the GUI shutdown path.
- CI: a hardcoded `--cov-fail-under` flag was silently overriding
  `pyproject.toml`'s coverage gate; removed so the config file is the single
  source of truth.  Also fixed PySide6 failing to import on the GitHub Actions
  runner (missing Qt/EGL system libraries) and moved off `actions/checkout`
  and `actions/setup-python` versions running on a deprecated Node.js runtime.

### Removed
- Legacy `AudioSampler.take_sample()` path (superseded by the continuous
  analyzer ring buffer) and the now-unused `duration` and
  `measurements_to_take` config fields.

## [1.0.0] — 2026-06-10

First stable, well-documented release.  The core detection algorithm has been
in continuous operation at N6OL since 2024-05-15.

### Added
- Comprehensive unit test suite (228 tests, 93 % line coverage) with
  deterministic synthetic-audio golden files that lock in DSP behavior.
- Python `logging` throughout — timestamped log lines replace bare `print()` calls.
- `pytest-cov` added to requirements; `pyproject.toml` enforces ≥ 90 % coverage.
- Ruff lint configuration in `pyproject.toml`; all lint errors resolved.
- Open-Meteo weather provider as an alternative to CumulusMX.
- Optional server-upload mode — set `[server] enabled = false` to run locally.

### Changed
- Configuration migrated from flat key-value to nested TOML sections
  (`[audio]`, `[station]`, `[weather]`, `[server]`).
- CSV column headers corrected: signal and noise floor now labelled `(dBm)`;
  SNR left unitless (it is a dimensionless ratio).
- `signals_adjusted` renamed `source_power_estimate` for clarity; removed from
  the dead `min_y` axis-scaling calculation.
- Extracted `_PULSE_WIDTH_SAMPLES = 3` constant; replaces all bare `3` literals
  in the DSP core.
- Extracted `_S9_DBM = -73` named constant (IARU reference level).
- `signals_adjusted` → `source_power_estimate` in plotter; clarified it is
  axis-scaling only, not a plotted series.
- Error and warning messages now follow context → problem → remedy structure.

### Fixed
- `generate_graph_from_csv` y-axis lower bound was accidentally including
  `source_power_estimate` in `min_y`, which had no effect (adjusted values
  are always ≥ originals) but was misleading.
- CSV parsing switched from manual `split(',')` to `csv.reader` to handle
  quoted fields correctly.

---

## [0.4.0] — 2025 (approximate)

### Added
- Full type annotations across all modules.
- Module-level and method docstrings throughout.
- `_bar_color()` helper extracted from inline comprehension in plotter.
- `zip(*[...])` sampling pattern in collector for cleaner multi-value averaging.

### Changed
- All internal-only methods renamed to `_private` convention.
- Daily graph layout fixed to exact 1600×640 px output with pixel-accurate margins.

---

## [0.3.0] — 2025 (approximate)

### Added
- Interactive audio device configurator (`configure.py`) with live signal-level
  display; writes device index and name into `~/.buzz/config.toml`.
- TOML configuration support (`tomllib` / `tomli`).
- Jinja2 HTML index template with per-minute auto-refresh.
- 23:59 no-refresh logic to avoid midnight page flip to an empty graph.

---

## [0.2.0] — 2024 (approximate)

### Added
- Numba JIT-compiled `_average_pulse_amplitude` for ~10× faster pulse summation.
- Pre-computed pulse kernel (built once at init, not per sample).
- `pulse_rate` config parameter — supports both 60 Hz (120 pps) and 50 Hz (100 pps) grids.
- Probability summary graphs: all-time, 7-day, 30-day.
- CumulusMX weather integration.
- SCP upload via Paramiko.

---

## [0.1.0] — 2024-05-15

Initial working implementation deployed at N6OL.

### Added
- FFT-based pulse-train correlation detector using `scipy.signal.fftconvolve`.
- Symmetric (palindrome) pulse kernel so convolution equals cross-correlation.
- Per-minute CSV logging with timestamp, SNR, signal, and noise floor.
- Daily signal-vs-noise-floor PNG plot with S9, threshold, and noise-floor
  reference lines.
- Mean-absolute-amplitude measurement (not RMS) — deliberately chosen for
  impulsive noise to avoid understating peak arc amplitude.
- Minimum-correlation phase used as noise reference to exclude arc bursts from
  the floor measurement.

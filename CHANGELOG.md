# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `--render FILE.mp4` renders a `--playback` session to video: H.264 of the display
  with the recording as its soundtrack, so an arc heard at two in the morning becomes
  something that can be shown to somebody. The transport controls are removed rather
  than hidden, since a render is a fixed pass over a file and there is nothing to
  operate. Frames are captured at the display's own 10 fps and placed on a 30 fps grid
  by the position playback had reached when the pixels were read - so the video carries
  the analyzer's real lookback rather than an idealised version of it, and a slow frame
  makes the render slow rather than out of sync. The audio is the recording itself,
  handed to ffmpeg as a second input and never piped, which removes it from the sync
  problem altogether. `--playback-gain` applies to the rendered audio, unlike live
  playback where gain reaches the speakers alone; nothing measured is involved in a
  rendered file, and a recording set deliberately low for measurement is awkward to
  show anyone at the level it was captured.
- Recordings keep their identity in the video. The `.wav`'s LIST/INFO tags - the event,
  the station, the moment, and the calibration behind the numbers - are carried into the
  MP4 container. Its cue marker is not: ffmpeg reads that as a chapter and MP4 chapters
  are a *track*, so a single marker arrived as a third data stream. The lock offset
  survives in the comment as `lead_in_seconds`.
- `--playback-gain auto` measures the recording and works the figure out, instead of
  the operator guessing and trying again. It takes whichever is smaller of the gain
  that reaches −23 LUFS (the EBU R128 broadcast reference) and the gain that leaves
  true peak at −2 dBTP, so it gets as close to a standard listening level as it can
  without letting anything clip; a recording whose bursts sit high above its noise
  floor reaches the peak ceiling first and comes out slightly quieter, which is the
  right way round. The result is one fixed gain, applied as a plain volume change.
  Nothing is compressed and nothing is limited: both reshape the pulse envelope that
  carries how bad the interference is, and ffmpeg's own `loudnorm` was observed
  choosing exactly that on these recordings, which is why it measures here but does
  not apply. `--render` implies `auto`, because a recorded event sits around −45 LUFS,
  well below a normal listening level - the calibration process keeps it deliberately
  there, which suits measuring impulsive noise and not showing it to anyone. Passing a
  figure, including `0`, overrides that; watching a replay does not imply it at all.
  The gain used is written into the video's metadata as `render_gain_db`, so the file
  says how far its audio was raised.
- `--audio-rf-conversion-db DB` supplies the level calibration for a replay, for the
  runs where the file cannot. A `.wav` this program recorded carries its own and needs
  no help; one from another operator is otherwise analysed with the figure configured
  for this station, which may be nothing like the receiver that made it - so every dBm
  and S-unit reading is wrong by an unknown amount while looking entirely plausible.
  Supply the sending station's figure and it is used for that run alone. It takes
  precedence over one recorded in the file, and says so when it does, since the
  recording's own is normally the right one.
- `[render] ffmpeg_path` for installs that do not appear on PATH. ffmpeg is needed for
  `--render` and the loudness probe that feeds it, and for nothing else; a monitor
  that never renders never looks for it, so leaving this empty costs nothing.
- Recordings now note `lead_in_max_seconds` beside `lead_in_seconds`, so the latter can
  be read honestly. The ring buffer keeps sliding while `min_lock_seconds` is waited
  out, so the lead-in can never exceed the buffer's capacity less that wait - and a
  recording sitting at that bound is saying "everything there was", not "the lock took
  this long". The two were indistinguishable from the file, since neither the buffer
  size nor `min_lock_seconds` was recorded anywhere in it. Noticed while checking a
  rendered video: eight of fourteen recordings clustered at 6.56–6.59 s against a 9.6 s
  buffer and a 3 s wait, which is the bound to within a poll, and nothing said so.

### Fixed
- Playback no longer sprints through the opening of a recording. The deadline schedule
  took its origin when the pipeline was *built* rather than when it was started, so
  everything in between - building the window, wiring the analyzer, compiling the DSP
  kernels - became a backlog the feeder delivered as fast as it could. Measured at 1.5 s
  of startup putting the transport 1.6 s in within 100 ms of pressing play. The analyzer
  was being shown the first seconds of every replay at whatever speed the machine
  managed, and the deliberate wait for the window in `main.py` was itself the backlog.
- The display's labels are monospace again, and exist at all when headless. They asked
  for the `Monospace` family, which is a fontconfig generic: on Windows it resolved to
  Tahoma, which is proportional, so columns of frequencies and S-units never lined up;
  and under Qt's offscreen platform, which reports no fonts whatsoever, every label drew
  as an empty box. The face is now loaded from a file into the application's own font
  database, where neither the platform nor the machine's installed fonts can reach it.
  DejaVu Sans Mono comes with matplotlib, already a dependency, so nothing new ships.
- The waterfall keeps its size and its frequency resolution at any sample rate. Its FFT
  window is a fixed span of *time* now - 32 ms, which is what 512 samples meant at
  16 kHz - rather than a fixed number of samples, so the bin count works out at
  `4000 Hz × 32 ms = 128` whatever the audio arrives at and the rate cancels out
  entirely. Before this a 44.1 kHz file was analysed 512 samples at a time, which is
  86 Hz per bin against 16 kHz's 31, and covered 0–4 kHz in 46 bins - a waterfall 230 px
  wide instead of 640. The dB references move with the window, as they always did:
  both had the window length in their formulas already, so this changes where N comes
  from rather than the arithmetic. Verified at every rate from 8 to 48 kHz - a
  full-scale tone still reads 0 dBFS and broadband noise still sits on its anchor.
- Sample rates outside **8–48 kHz** are refused with an explanation rather than
  analysed. 8 kHz is exactly Nyquist for the 4 kHz the display and the analysis look
  at, so below it the top of the waterfall is empty band; far above it the fixed-size
  ring buffer holds too little history to acquire the way the analyzer was tuned to,
  and a powerline arc has nothing to say up there anyway.
- A stereo file now says in the log that only channel 0 is being analysed. It always
  was - matching what the live monitor does with a stereo input device, since mixing
  would average the arc against whatever the other channel holds - but "half of what
  you sent was ignored" should not have to be inferred from a reading that came out
  low. A render takes channel 0 too, so the video's audio is the audio that was
  measured rather than both channels beside a picture of one.

### Changed
- The JIT-compiled DSP helpers declare their signatures, so Numba compiles them at
  import instead of on first call. Lazy compilation ran on whichever thread arrived
  first - in the GUI, the Qt thread part-way through a paint, freezing the window for
  about a second while audio carried on arriving. `cache=True` alongside means only the
  first run after a change pays at all.
- The messages that report trouble say what to do about it. A file that is not 16-bit
  now gives the ffmpeg command that converts it; a failed weather fetch says which
  columns went blank and that the noise figures did not; `configure.py` cancelled says
  the config was left alone. Each carries what was being attempted, the likely cause,
  and a next step, which is the standard the newer code was already written to.
- Dropped audio from the input device is reported accurately and at most once a minute.
  The device captured faster than the monitor collected and the driver discarded the
  difference, which leaves no gap and no silence: the callback still receives a full
  block, so what arrives is a splice of two runs that were never adjacent. The audio
  clock counts only what it was handed, so the pulse train moved through samples nobody
  recorded and the phase jumps; the drift fit reads that step as drift, which makes the
  grid frequency the reading to distrust rather than the levels. It is logged rather
  than raised, because a monitor that exits on a transient overflow loses every later
  measurement to protect one polluted minute, and the analyzer re-acquires by itself.

## [1.3.0] - 2026-07-30

Event recording and playback arrive together: the monitor writes each locked event to
its own `.wav` and can replay one back through the whole pipeline, so an arc heard at
two in the morning can be listened to, measured, and shown to somebody afterwards.

Underneath, the analyzer publishes state changes to listeners instead of being polled,
which is what makes a recorder possible at all; and the test suite grew a tier that
runs the real components over real threads at real speed, because every bug that
reached a running program lived in exactly that gap.

### Added
- Automatic event recording (`buzz.recorder`). While armed, the recorder writes
  each locked event to its own 16-bit mono `.wav` in the configured directory,
  named for the moment of lock in station local time with the UTC offset attached
  (`event-20260729-143307-0700.wav`; ISO 8601's colons are illegal in Windows
  filenames). Configured under `[recording]`: how many of the next events to
  record before disarming, an optional cap on a single recording, and how long a
  signal may be gone before the file is closed. Armed at startup with
  `--enable-recording`, and toggled while running from the toolbar or the `R` key.
- Recordings carry a lead-in and a trailer. The ring buffer is already holding the
  last several seconds of audio when lock happens, so the recorder starts from its
  oldest surviving sample rather than the live tail and the file opens with the
  run-up to the event; the audio captured while waiting out `stop_after_seconds`
  is written as it arrives, so the trailer is already in the file by the time the
  timeout expires. A signal returning inside that window continues the same
  recording rather than starting a second one.
- Recordings are faded in and out over 5 ms, so every file starts and ends at
  exactly zero and cannot click - including at the seams when files are played
  back to back. Both ends can really be a cut through full-scale audio: an arc
  already buzzing when the monitor starts is locked onto within a second or two,
  making the lead-in a live pulse train from its first sample, and `max_seconds`
  ends a file mid-event the same way. A sound card's DC offset would step at both
  ends even in silence. The ramp is a raised cosine rather than an exponential
  (which approaches zero without reaching it, so truncating it reinstates the step
  the fade exists to remove) and meets both ends with zero slope, worth 6 dB/octave
  of splatter rolloff over a linear ramp's corner. 5 ms because a fade of duration
  T spreads the transition over roughly 1/T of bandwidth: a few samples would smear
  a click across the whole audio band rather than removing it. The recorder holds
  back a fade's worth of the newest audio so the fade-out can be applied to
  whichever samples turn out to be last, which is only known after the fact.
- The recording directory is created when recording is armed rather than at the
  first event - at startup when it is armed there, and otherwise when the Record
  button is pressed - so a mistyped path or a permissions problem is reported while
  the operator is still watching, not discovered at the end of an unattended day
  from an empty folder that explains nothing. Recording that is off reaches for
  nothing at all, so a run without it leaves no stray directory behind and raises
  no complaint about a path it was never going to use. A directory that
  cannot be created switches recording off and is logged as an error; the monitor
  carries on measuring and logging, since the failure should cost recordings
  rather than the day's data. Arming re-checks, so fixing the path and pressing
  Record retries. Opening a file still copes with a directory that disappears
  mid-run, which no startup check can cover.
- Each recording's length is now accounted for in the log when it closes - total,
  lead-in, and seconds from the lock - because the total is not a number any
  setting names: `max_seconds` measures the event and the lead-in and trailer sit
  outside it, so a file is always somewhat longer than the cap. A lead-in cut short
  because the monitor had not yet filled its buffer says so too, rather than
  looking like a setting nobody chose. The documentation now states plainly that
  `max_seconds` buys that many seconds of actual 120 pps noise rather than that
  many seconds of file.
- `min_lock_snr` keeps events too faint to hear off the disk, without making the
  monitor any less sensitive: it gates recording alone, never locking, measurement,
  logging or the display. The analyzer locks at 6 dB SNR - a constant in the code,
  not a setting - so anything at or below that is a no-op, and the documentation
  says so. A signal that starts quiet and builds is not skipped but watched:
  recording begins the moment it crosses, which catches the event at the cost of
  its opening seconds, since the wait comes out of the lead-in. Judged over about a
  second of readings rather than one, both so a single loud tick cannot carry a weak
  event through and because levels read several dB low until the drift tracker
  converges - thresholding on the first reading after a lock would reject events
  that actually qualify. Time spent below the threshold is paid for out of the
  lead-in and the lead-in runs out, so a signal that loiters near the bar for longer
  than the buffer holds loses its onset entirely; that is documented, and is the one
  way this is sharper-edged than `min_lock_seconds`, whose wait is capped at the
  buffer and so can never cost the beginning of an event.
- `max_seconds` is counted from the moment recording starts, less any
  `min_lock_seconds` spent waiting: those seconds are part of the event and are
  kept, so asking for a 3 s wait and a 10 s recording buys ten seconds of event
  rather than thirteen. `min_lock_seconds` is therefore clamped to `max_seconds` as
  well as to the buffer, since a wait longer than the whole allowance would reach
  back past the start of the file and discard its newest seconds to stay inside a
  limit already spent. A `min_lock_snr` wait is not counted, being open-ended:
  charging one that can run to minutes would spend the allowance before the file was
  opened. Whatever the buffer holds beyond the deliberate wait is lead-in and comes
  free on top. Measuring it from the lock broke down as soon
  as anything delayed that start: a wait longer than the cap spent it before the
  file was opened, saving a real event as a nought-second recording that reported,
  wrongly, having fallen behind the ring buffer. Measuring instead from the first
  buffered sample fixed that but traded it for a subtler fault - free lead-in ate
  the cap, so a threshold crossing produced a file of exactly `max_seconds` that was
  almost entirely the quiet approach and barely any of the loud part it had been
  waiting for.
- `min_lock_seconds` holds a recording off until the interference has been present
  that long, so a night of two-second blips no longer fills the directory with
  files too short to be worth replaying - or spends the event budget on them. A
  lock that drops and returns starts the count again rather than adding up, which
  needs the *loss* edge and not just the acquisition: no tick ever observes the
  gaps in a stream of blips, because the next one has set the flag again before the
  tick runs, so a run of them would otherwise look exactly like one long lock.
  Capped at the ring buffer's own length, since the wait is paid for out of the
  lead-in and waiting longer than the buffer holds would start the recording after
  the onset of the event it exists to capture. Keep it to a few seconds; the docs
  say so and say why.
- `rearm_reset_minutes` turns the event budget into a rate rather than a one-off:
  `max_events = 10` with `rearm_reset_minutes = 1440` records up to ten events a
  day, every day, unattended, without being able to fill a disk. The cycle runs
  from when the budget was last reset rather than from when it ran out, so it
  keeps its time of day instead of sliding later by however long each day's events
  took to arrive; a missed cycle (a suspended machine) restarts from now rather
  than firing repeatedly to catch up on windows nothing could have been recorded
  in. Unused events are not carried forward. `0` never re-arms, and switching
  recording off by hand cancels the cycle - off has to mean off. The toolbar shows
  the countdown while the budget is spent.
- Recordings carry RIFF metadata (`buzz.wavmeta`): LIST/INFO tags naming the
  station, software version and moment of lock, a comment holding the settings a
  replay needs, and a labelled cue marker at the exact sample where the analyzer
  locked, so an editor shows where the lead-in ends. The stdlib `wave` module has
  no metadata API at all - `Wave_write.setmark()` raises - so the chunks are
  appended after it closes the file, which is safe because a `.wav` is a chain of
  independent chunks and the stdlib reader stops at `data`. Tagging can never fail
  a recording: the audio is closed and safe before it is attempted.
- `--playback` now adopts the pulse rate and level calibration recorded in the
  file, as it already did the sample rate. These are the settings that decide what
  a replay measures and none can be recovered from the audio: a 100 pps recording
  analysed as 120 pps never locks, and a mismatched calibration reports the whole
  event at the wrong absolute level. A file without them still plays, warning that
  it is being analysed with the local configuration instead.
- Playback starts with the window rather than with the process. Opening a file no
  longer starts it playing; `main` does that from the first pass of the Qt event
  loop, once the window is up. Audio started at construction was heard before there
  was anything to see it in, and then broke up - the feeder was competing for the
  GIL with widget construction, so the sound card ran dry and the replay opened
  with a stutter that was not in the recording.
- `--playback-gain DB` turns a quiet replay up for the speakers and for nothing
  else. Powerline noise is typically recorded around −24 to −34 dBFS, which is hard
  to hear on a laptop; the gain is applied to the copy on its way to the sound card,
  so it cannot move a dB of what the analyzer measures or the meters read. Asking
  for more than the headroom allows clips at the int16 rails and distorts, which is
  what turning something up too far should do - the clamp is there because int16
  *wraps* on overflow, and without it a passage a few dB too loud would come back as
  noise rather than as a loud passage.
- Playback can be heard: unmuting sends the audio to the default output device, so
  an event can be listened to while it is watched. `--mute` starts silent, `M` and
  a toolbar button toggle it, and the button greys out with a reason on a machine
  with no usable output. Muting is the absence of an output stream rather than a
  volume of zero, so silent replay runs exactly the code that ran before playback
  could be heard - no device is opened, and the monotonic deadline paces it as
  before. Pausing and reaching the end of the file release the device by the same
  route, since nothing is being written to it then either and a stream nobody
  writes to underruns for as long as it is left open. The end of the file drains
  what is queued rather than discarding it: nobody asked for silence there, and
  cutting the last of the recording off mid-sample would end the replay on exactly
  the step the recorder's fade-out exists to prevent. Whichever exists is the clock: with a stream open the sound card decides
  when the next chunk is wanted, which is what keeps audio and display together
  without a second clock to drift against. Both transitions re-base the schedule,
  since an origin left over from before a stream opened is minutes in the past and
  would send the loop racing through the rest of the file to catch up. Restart
  discards audio already queued to the card, which would otherwise play out the
  abandoned pass after the click and leave every later chunk trailing the display
  by an output buffer. The stream is opened, flushed and closed on the feeder
  thread, never from the Qt thread, since closing one mid-write is undefined
  behaviour in PortAudio.
- Restarting playback resets the analyzer, so the second pass is a cold start
  rather than one that opens already locked at a drift rate learned from the pass
  before - watching the monitor acquire a signal is usually the point of replaying
  an event. The ring buffer is emptied with it, since several seconds of the
  abandoned pass would otherwise be sitting there for a freshly-reset analyzer to
  lock onto immediately. `ContinuousAnalyzer.reset()` splits the work by deadline:
  tracker state (drift, fitted history, DC estimate, tier timers) is left for the
  analysis thread, which owns it, while the fields the displays read are cleared
  synchronously under the lock `trigger_phase()` uses - a tick can sit in
  `wait_for_data` for a second, and a second is long enough for the replayed audio
  to re-lock and make the restart look like it did nothing.
- Playback has a transport: play/pause and restart, with a running time index
  (`▶ 00:12 / 00:39 - event-....wav`). The toolbar carries it instead of the record
  button, which means nothing when there is no live audio to record. `Space` pauses
  and resumes without moving the mouse across a window being screen-recorded, and
  restart replays the file from wherever it has got to, finished or not. Pausing
  and restarting re-base the feeder's deadline schedule, since it is only
  meaningful across a stretch of uninterrupted play; the feeder blocks on a
  condition while paused or sitting at the end rather than polling for work.
- `--playback FILE` replays a recorded `.wav` through the whole live pipeline
  (`buzz.playback`), at the file's own sample rate, so an event can be analysed
  again at real speed for a screen recording. A bare filename resolves against the
  recording directory. No audio device is opened, and the collector is not started
  at all - no CSV rows, plots, uploads or recording, since reviewing an old event
  must not add minutes to a day it did not happen on.
- `ContinuousAnalyzer.add_state_listener()` publishes state changes to registered
  listeners from inside `_transition()`. Lock is an event, not a level: a consumer
  polling for it can only infer the event by watching for the level to differ from
  last time, which makes a brief lock between two polls invisible - precisely the
  intermittent signals this monitor exists to catch. Listeners run on the analyzer
  thread and are isolated from each other, so a failing one cannot abort a
  transition or stop analysis.
- Toolbar strip across the top of the display window, with a recording control that
  names the state rather than the action - `Record` when off, `Armed` once on, dimmed
  to match, since a lit button reading "Record" during a recording invites an action
  already taken. It stays clickable, being also the only way to switch recording off
  with the mouse. Beside it a status line (armed and events remaining, elapsed time
  and filename while recording, or the file being replayed during playback). The bar
  spans the full window
  width rather than sitting in the left-hand stack, which keeps the meter panel
  aligned with the displays - its segment geometry is derived from the window
  height and does not survive being stretched.
- An integration suite under `tests/integration/`, run with `pytest -m integration
  --no-cov` and deselected from a plain `pytest` so the fast feedback loop stays
  fast. It drives real components over real threads at real speed - the combination
  every costly bug in this project has lived in, and the one thing the unit suite
  cannot exercise by construction. Three groups: recording an event whose level
  crosses `min_lock_snr` mid-arc; replaying one, including that Restart really does
  make the analyzer acquire from scratch rather than opening already locked; and a
  Qt offscreen render that asserts on pixels, covering the two display bugs that
  reached a running program unnoticed (a toolbar drawn in the desktop's grey, and a
  Record button that stayed lit once armed). CI runs it as its own job, alongside
  the unit suite rather than after it.
- A release workflow. Pushing a version tag runs everything CI runs plus the
  integration suite, checks that `lib/buzz/__init__.py`, `pyproject.toml` and the
  tag all agree on the version and that the changelog has a section for it, then
  builds `.tar.gz` and `.zip` archives of the tagged tree with `git archive`,
  attaches a `SHA256SUMS` beside them, and publishes the GitHub release with that
  version's changelog section as its notes. GitHub's own "Source code" archives are
  generated on demand and their checksums have changed before now; these are built
  once and stay true.

### Changed
- The ring buffer moved out of `AudioPipeline` into a `RingBufferPipeline` base
  class, so live capture and `.wav` playback are two ways of filling the same
  buffer and every consumer downstream is unchanged. Added `read_from(position)`
  alongside `get_snapshot()`: displays want the most recent N samples and do not
  care what they skipped, while a recorder needs each sample exactly once, in
  order, with any loss visible rather than silent.

### Fixed
- `pip install .` no longer produces a monitor that silently falls back to headless:
  PySide6 was listed in `requirements.txt` but missing from the dependencies in
  `pyproject.toml`.

## [1.2.0] - 2026-07-28

A phase-synchronised oscilloscope display joins the waterfall, backed by a
least-squares drift tracker precise enough to hold its trace still.

### Added
- Phase-synchronised oscilloscope panel above the waterfall (`buzz.scope`). The
  sweep is triggered from the analyzer's tracked pulse phase rather than an
  amplitude threshold, so a 120 pps arc renders as a standing wave instead of
  sliding across the screen. CRT-style phosphor persistence makes pulse-to-pulse
  jitter visible as a halo around the trace. Press `A` to switch between the raw
  bipolar view and a coherently-averaged rectified envelope. The vertical scale
  auto-ranges from the signal and is reported in dBFS; the trigger indicator
  reads `LOCK`, `HOLD` (extrapolating through a fade) or `FREE` (never locked).
- `ContinuousAnalyzer.trigger_phase()` - thread-safe accessor returning the
  predicted pulse phase plus a `TriggerSync` confidence level, for display sync.

### Changed
- `scope.accumulate_trace()` is now JIT-compiled. It runs once per sweep, about
  fifty times a second, and the vectorised difference-and-cumsum formulation it
  replaced allocated a quarter-megabyte temporary every call and integrated all of
  it regardless of how few cells needed touching. Measured on a 96×640 buffer at
  five sweeps per frame: 2088 µs → 50 µs, giving back ~1.8% of a core continuously.
- Tests now run with `NUMBA_DISABLE_JIT=1` (set in `conftest.py`), so coverage can
  see inside JIT-compiled functions - machine code executes no bytecode to trace,
  and `dsp.py` had been reporting 82% for that reason alone. CI additionally runs
  the whole suite a second time with the JIT enabled, since the two paths are not
  automatically equivalent.
- `SIGNAL_LOST` now falls back to `SEARCHING` once the stored phase pair is older
  than `PHASE_HOLD_TIMEOUT` (60 s of captured audio), clearing `_phases_valid`.
  Beyond that age the extrapolated phase is far outside `PHASE_SEARCH_RADIUS`, so
  the cheap re-acquisition tiers cannot succeed anyway - and it keeps the scope's
  trigger indicator honest, since `HOLD` is reported purely on having a valid phase
  pair and would otherwise claim a synchronised sweep all night if the arc quit at
  dusk. Aged on the audio clock, so a stalled sound device doesn't expire phases
  that are still good.
- Drift-rate estimation now fits a least-squares line through the last
  `DRIFT_FIT_POINTS` phase measurements, rather than dividing a single prediction
  error by a single refine interval. Phases are measured to whole samples, so the
  old estimator was pinned at `0.5 / REFINE_INTERVAL` ≈ 0.83 samples/s of error with
  nothing to average against; the fit spans a 5.4 s baseline over which zero-mean
  quantisation error largely cancels. Measured against synthetic audio at known
  drift rates, steady-state error falls from 0.38 to under 0.01 samples/s *and*
  settles faster after a step change, so it costs no responsiveness. On the scope
  this is the difference between the trace creeping a division a minute and standing
  still. `DRIFT_LEARNING_RATE` and `MIN_DRIFT_UPDATE_INTERVAL` are removed; the
  latter existed only to stop `error / elapsed` exploding, and there is no such
  division now.
- The scope's triggering pulse now sits 1.5 divisions from the left edge instead of
  1.0. The trigger is the pulse's *peak*, but the pulse begins before that and rings
  on after it through the receiver's audio passband, so its leading flank needs room
  or it falls off the frame. The half-division also keeps the peak clear of a
  graticule line rather than drawn underneath one.
- Main window is now 734×248 (was 726×224). The scope and waterfall each occupy
  120 px, with 8 px of padding between them and before the meter column.
  Waterfall history is 4.8 s (48 rows), down from 10 s.

### Fixed
- `average_pulse_amplitude()` accumulated at single precision when called from
  interpreted Python. Callers pass float32 audio, and under NumPy 2's NEP 50
  promotion `python_float + np.float32` yields `np.float32`, so the running total
  degraded after its first addition - drifting ~1e-4 from the float64 result
  `calculate_pps_fit_array()` computes for the same data. Numba types the
  accumulator as float64 regardless, so the JIT'd and interpreted paths returned
  different answers for identical input. An explicit `float()` cast pins both to
  double precision.
- `ContinuousAnalyzer._record_phase_measurement()` now writes the phase pair and
  its measurement timestamp as one locked group. They are read together by
  `trigger_phase()` on the Qt thread, where a read landing between the two writes
  would project a fresh phase across a stale interval and mis-place the trigger.

## [1.1.0] - 2026-07-27

Continuous live analysis and display replace the old once-a-minute sampling
loop, plus a DSP correctness pass and utility-line drift tracking on top of it.

### Added
- Continuously-running audio pipeline (`AudioPipeline`) and a
  `ContinuousAnalyzer` state machine (`SEARCHING` → `LOCKED` → `SIGNAL_LOST`,
  with tiered re-acquisition) replacing per-minute FFT sampling - sub-second
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
- `tools/pulse_probe.py` - diagnostic that measures the real pulse shape,
  drift rate, and number of active mains phases from live audio, for future
  tuning of `PULSE_WIDTH_SAMPLES`.
- `level_meter.py` - live text S-meter for receiver gain calibration.  Displays
  a continuously-updating 21-char bar (S1–S9 linear, then +20/+40/+60 sections
  with 3 ticks each) plus dBm and S-unit readout.  Uses a persistent
  callback-based PortAudio stream (DirectSound blocking I/O is unreliable on
  Windows) at 20 ms per frame.  Flicker-free: each refresh overwrites in place
  without an intermediate clear.
- `AudioSampler.level_stream()` / `LevelStream` in `sampler.py` - persistent
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
- DC offset removal switched from mean to median before rectification - the
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
  minimum dynamic range so a truly quiet band can't paint itself warm
  from measurement noise alone.
- Weather fetches now have a 10 s timeout, and a failed fetch degrades to
  blank weather fields instead of dropping the minute's CSV row.
- CSV row parsing moved into `CsvStore.read_rows()`; the plotter consumes
  typed rows instead of parsing files itself.
- Waterfall display derives its bin geometry and frequency axis from the
  configured sample rate instead of a hardcoded 16 kHz.
- SCP uploads now verify the server host key against known_hosts
  (`~/.ssh/known_hosts` plus optional `~/.buzz/known_hosts`) instead of
  auto-accepting any key - closes a man-in-the-middle vector; add new hosts
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

## [1.0.0] - 2026-06-10

First stable, well-documented release.  The core detection algorithm has been
in continuous operation at N6OL since 2024-05-15.

### Added
- Comprehensive unit test suite (228 tests, 93 % line coverage) with
  deterministic synthetic-audio golden files that lock in DSP behavior.
- Python `logging` throughout - timestamped log lines replace bare `print()` calls.
- `pytest-cov` added to requirements; `pyproject.toml` enforces ≥ 90 % coverage.
- Ruff lint configuration in `pyproject.toml`; all lint errors resolved.
- Open-Meteo weather provider as an alternative to CumulusMX.
- Optional server-upload mode - set `[server] enabled = false` to run locally.

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

## [0.4.0] - 2025 (approximate)

### Added
- Full type annotations across all modules.
- Module-level and method docstrings throughout.
- `_bar_color()` helper extracted from inline comprehension in plotter.
- `zip(*[...])` sampling pattern in collector for cleaner multi-value averaging.

### Changed
- All internal-only methods renamed to `_private` convention.
- Daily graph layout fixed to exact 1600×640 px output with pixel-accurate margins.

---

## [0.3.0] - 2025 (approximate)

### Added
- Interactive audio device configurator (`configure.py`) with live signal-level
  display; writes device index and name into `~/.buzz/config.toml`.
- TOML configuration support (`tomllib` / `tomli`).
- Jinja2 HTML index template with per-minute auto-refresh.
- 23:59 no-refresh logic to avoid midnight page flip to an empty graph.

---

## [0.2.0] - 2024 (approximate)

### Added
- Numba JIT-compiled `_average_pulse_amplitude` for ~10× faster pulse summation.
- Pre-computed pulse kernel (built once at init, not per sample).
- `pulse_rate` config parameter - supports both 60 Hz (120 pps) and 50 Hz (100 pps) grids.
- Probability summary graphs: all-time, 7-day, 30-day.
- CumulusMX weather integration.
- SCP upload via Paramiko.

---

## [0.1.0] - 2024-05-15

Initial working implementation deployed at N6OL.

### Added
- FFT-based pulse-train correlation detector using `scipy.signal.fftconvolve`.
- Symmetric (palindrome) pulse kernel so convolution equals cross-correlation.
- Per-minute CSV logging with timestamp, SNR, signal, and noise floor.
- Daily signal-vs-noise-floor PNG plot with S9, threshold, and noise-floor
  reference lines.
- Mean-absolute-amplitude measurement (not RMS) - deliberately chosen for
  impulsive noise to avoid understating peak arc amplitude.
- Minimum-correlation phase used as noise reference to exclude arc bursts from
  the floor measurement.

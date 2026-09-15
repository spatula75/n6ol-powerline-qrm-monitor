# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `[recording] min_free_disk_percent`, a share of the disk to leave free. Recording is
  held off while the disk is below it and starts again on its own once there is room,
  so a station that fills its disk stops recording rather than taking the machine down
  with it. Ten percent by default, and 0 records until the disk is full. Checked
  before each event rather than before each write, so being wrong about it costs at
  most one recording.
- `[recording] record_iq`, which writes a second `.wav` of raw IQ beside each event
  recording: stereo, I on the left channel and Q on the right, at the receiver's own
  sample rate, in the device's own sample format. Nothing is scaled, levelled or
  faded, so the file is the measurement rather than this program's reading of it. It
  is for handing the raw data to somebody with their own tools; nothing here reads one
  back. Off by default, and a receiver only.

  Its metadata carries what the samples cannot: the frequency the hardware was tuned
  to, which is where DC sits in the file, the listening frequency and the offset
  between them, the tuner gain, the level calibration, and the grid's pulse rate. The
  cue marker sits at the moment of lock, as it does in an audio recording.

  Turning it on costs memory whether or not an event is ever recorded, because the
  monitor has to hold the last several seconds of raw IQ at all times for a recording
  to have the same run-up its audio gets. That is 4.7 MB at the default sample rate
  and 44 MB at the highest the hardware takes.
- `lib/buzz/iq.py`, the conversion from an SDR's IQ stream to the mono audio the
  rest of the program already expects. It mixes the frequency of interest down to
  zero, filters to one sideband, decimates by a whole number, takes the real part
  and scales to int16. Producing audio rather than an envelope is what lets the
  analyzer, the recorder, playback, rendering and the display all work unchanged:
  the analyzer treats the median of its window as a DC offset, which holds for
  audio and not for an envelope.

  The filter has complex coefficients so that it keeps one side of the tuned
  frequency and rejects the other. A plain lowpass keeps both, and measured on a
  recorded arc that reads 3.78 dB high on signal while quietly doubling what the
  bandwidth setting means.

  The converter holds its filter state and the position of its mixing sinusoid
  between calls, so a stream arriving in separate blocks gives the same samples as
  one pass over the whole signal. `tests/test_iq_to_audio.py` asserts that at six
  block sizes, along with alias rejection, sideband rejection, DC-spike rejection,
  and clipping rather than wrapping at the int16 rail.

- `lib/buzz/sdr.py`, the hardware half of the same path. It opens the receiver,
  configures it, and hands blocks of raw bytes to the thread that converts them. The
  callback copies its block, timestamps it and returns, because the receiver's own
  FIFO holds 3.67 ms at 256 kHz and nothing anywhere reports an overflow of it.
  `RtlSdrPipeline` fills the same ring buffer a sound card fills, so the analyzer,
  the recorder, the display and the collector needed no changes.

  The tuner gain is snapped to a step the hardware offers and then remembered, since
  an RTL-SDR Blog V4 cannot report its own gain. Measured on that hardware, the
  setter works and the getter returns 0.0 whatever is set. Raw values sitting at the
  converter's rail are counted and reported, because a clipped arc reads smaller
  than it truly is.

- `[audio] source`, which selects `soundcard` or `rtlsdr`. It defaults to
  `soundcard`, so an existing station behaves exactly as before. With `rtlsdr` the
  new `[rtlsdr]` section supplies the frequency, tuner gain, IQ sample rate,
  decimation, bandwidth, tuning offset, sideband and device index. Any other value
  is refused at startup rather than treated as a sound card, since `[rtlsdr]` has no
  setup screen yet and reaches the file by hand.

  Two limits apply to that section, and the monitor names both when it refuses one.
  The IQ rate divided by the decimation has to fall between 8000 Hz and 48000 Hz,
  which is the band the rest of the program already works in: 2400000 Hz at the
  default decimation of 16 would otherwise give 150 kHz of audio, a ring buffer
  holding one second instead of 9.6, and recordings at a rate `--playback` refuses.
  The bandwidth has to fit that audio rate with room for the filter skirt, which is
  6400 Hz at the default settings rather than the 8000 Hz half the rate suggests. At
  8000 Hz the top of the band folds back onto the bottom 6 dB down, and a broadband
  arc has energy exactly there.

  `[rtlsdr] audio_rf_conversion_db` does for a receiver what
  `[station] audio_rf_conversion_db` does for a sound card. It sits in its own
  section because the figure depends on the tuner gain above it. Left unset, it is
  estimated as the negative of that gain. Measured on one receiver, the estimate
  moves 3.0 dB across the gain range anybody would use, so it is a place to start rather
  than a substitute for calibrating.

  The work was developed and measured against an RTL-SDR Blog V4. Other receivers
  are untested, and a V3 reaches HF only through direct sampling, which this does
  not enable.

  `pyrtlsdr[lib]` is what talks to the hardware. `requirements.txt` installs it, and
  a packaged install asks for it with `pip install .[rtlsdr]`. A sound-card station
  never loads it, because the import sits inside the function that opens the device.

- `tools/ste_lint.py --fragments`, an advisory pass for sentence fragments. It is
  off by default, and deliberately so. Spotting a clause with no finite verb means
  knowing which words are verbs, and measured over this repo the pass flagged 16
  sentences of which about 9 were fragments. That is a good trade when somebody
  chose to look and a bad one in a gate that blocks a commit.
- The setup program knows which audio source it is configuring. The source moved to
  the main menu, above the sections, because it decides which of them apply.
  `[rtlsdr]` hides for a sound-card station, and the sound card device and rate hide
  for a receiver.
- The RTL-SDR section lists the steps of its procedure in order and nothing else:
  the listening frequency, the tuner gain, the level calibration, and which receiver.
  An uncalibrated level shows the figure the monitor will actually use and says it is
  an estimate, so a borrowed number and a measured one no longer look alike.
- `[rtlsdr] calibrated_at_gain_db`, written by the calibration tool rather than set by
  hand. Changing the tuner gain afterwards leaves the calibration wrong by roughly the
  difference, and nothing else would notice.
- `docs-notebook/`, an engineering notebook for why a constant holds the value it
  does, what an experiment ruled out, and how the hardware behaved when tested. It is
  committed, unlike `tmp/`, and unpublished, unlike `docs/`. Its first three documents
  cover the gain calibration, the receiver's measured artifacts, and the shape of the
  setup screens.
- `tools/ste_lint.py` requires every Markdown file under `docs-notebook/` to open with
  an attribution line. A reader judging a measurement needs to know what produced it,
  and the rule was broken within minutes of being written, in the README that states
  it.
- `lib/buzz/gain_sweep.py`, which chooses the receiver's tuner gain by measuring the
  band rather than asking anyone to guess. The receiver section's "Auto-calibrate
  gain..." row runs it, and takes the answer as the gain, the level offset, and the
  calibration mark together.

  Neither measurement it makes depends on a signal being present, because nobody can
  promise an arc is running when the tool is opened. The level between bursts comes
  from a low percentile of per-frame RMS, and the antenna's share of the noise floor
  from the shape of the whole sweep. It walks every gain five times, alternating
  direction, and takes the median floor and the worst peak, because an arc that comes
  and goes makes a single pass measure every step in a different world.

  It picks the tuner step whose reported noise floor comes closest to reading 3.01 dB
  high, which is the knee where the antenna and the converter contribute equally.
  Nearest to a target rather than lowest inside a budget: a budget is a bar, so half a
  decibel of drift in the fit moves a step across it, and one station's answer moved
  between 20.7 and 25.4 dB on repeated runs of the same sweep. A target also bounds
  what the rule can spend, where a budget spends whatever the next step down happens
  to cost.

  The dialog works out how long the sweep will take from the number of gains the tuner
  reports, rather than quoting a figure measured on one model of receiver.

  It reports rather than guesses when the two bounds leave nothing: an antenna too
  quiet to beat the receiver at any gain, a band loud enough to clip at every gain,
  and the two crossing each get their own wording. A quiet station is told what its
  antenna is doing instead of handed a number.
- The setup program's level meters open the source the config actually names. An
  RTL-SDR station reaching either meter used to open whatever sound card was named in
  `[audio]`, meter that, and let an offset be calibrated against a device the monitor
  was never going to use.
- `[rtlsdr] gain_db` is chosen from a list of the steps the tuner reports rather than
  typed. A tuner accepts a fixed set and snaps anything else to the nearest, so a
  typed 41.0 became 40.2 with nothing said. Choosing a gain also moves the level
  calibration with it, by the difference, so the reported dBm does not change and the
  calibration stays describing the gain in use.
- The level meter's offset moves by 0.1 dB on Up and Down and 1 dB on PageUp and
  PageDown. Half a decibel could not reach the figure that matches a gain of 40.2,
  which is the case anybody calibrating a receiver is in.
- `BuzzConfig.level_offset_db` resolves the one level offset that applies, from the
  playback override, then `[rtlsdr]`, then `[station]`. Everything that converts a
  level reads it.
- `_load_section` names any config key it does not recognize instead of dropping it in
  silence. It is still ignored rather than fatal, so a file from another build starts,
  but a misspelled setting no longer reverts to its default without a word.
- Startup warns when the tuner gain has moved since the level calibration was
  measured. `[rtlsdr] calibrated_at_gain_db` recorded that gain and nothing read it,
  while the documentation said startup compared the two.

- `[station] enable_frequency_chart`, which publishes `current_frequency_estimate.png`,
  a chart of the estimated grid frequency against time. It covers the current day and
  is redrawn every cycle and overwritten, alongside the daily charts rather than with
  the hourly summaries. The horizontal axis runs from the station's midnight to the
  moment it was drawn, so the newest reading is always at the right-hand end. Off by
  default.

  The vertical axis is fixed at the nominal grid frequency plus or minus 0.1 Hz, which
  is half the pulse rate: 60 Hz on a 120 pps grid, 50 Hz on a 100 pps one. Holding the
  scale rather than fitting it to the data is what lets one hour's chart be compared
  with the next. A reading outside that band is drawn at the edge it passed, with a
  marker and a count in the legend, so an excursion cannot be mistaken for a gap.

  A minute with no lock on the pulse train has no frequency to report, and the trace
  breaks there rather than joining across it.

  Startup reports a chart left behind by turning the setting off, as it already does
  for the all-time summary. The name says the chart is current whatever its age, so
  nothing else would say otherwise.

  The layout follows the "System Frequency WECC/USA West" panel at
  `kestrelgrid.com/static/WECC.png`, so a station's own reading can be put beside a
  reference drawn from the grid operators' data: the same 1861 by 579 pixel axes box
  in a 1999 pixel figure, the same margins, monospace labels, and hour marks every two
  hours rotated clear of each other. Grid lines are light gray, every 0.025 Hz and
  every two hours. The padding inside the axes is 5% of the span at each end, which is
  what the reference leaves matplotlib's default margin at, so the two stay padded
  alike whatever the hour.

### Changed
- `EventRecorder` is split into `RecordingTrigger`, which decides when an event is
  worth a file and when that file ends, and `AbstractEventRecorder`, which handles the
  mechanics of writing one. `AudioEventRecorder` is the first subclass. The trigger
  reads no audio and opens no file, so a second format needs no second copy of the
  lock gating, the event budget, or the rearm cycle, and one event counts once against
  the budget however many files it produced. Nothing an operator sets or sees changes.
  `docs-notebook/iq-recording-design.md` records why, and raw IQ capture is what it is
  for.
- `[rtlsdr]` frequencies are given in kHz: `frequency_khz`, `bandwidth_khz` and
  `tuning_offset_khz` replace the Hz-denominated keys. Nobody wants to type three
  zeroes on the end of every frequency.
- `[rtlsdr] audio_rf_conversion_db` is now `calibrated_offset_db`. It was never the
  same setting as `[station] audio_rf_conversion_db`, only the same name, and sharing
  one read as a single setting stored in two places.
- `[rtlsdr] gain_db` ships as 28.0 rather than 40.2, which is what the automatic
  calibration measured on the antenna this was developed against. It gives up some
  noise-floor accuracy for headroom, which is the right way round: a clipped arc
  cannot be recovered.
- The receiver section sits directly below the audio section in the setup program and
  second in `config.example.toml`, next to the source that selects it.
- `tools/ste_lint.py` applies the wordy-word substitutions to strict text only, as
  `docs/ste-writing.md` has always specified. Flavored prose gets the sentence,
  active-voice and plain-verb rules; the vocabulary restrictions were never meant to
  reach it. The tool had been checking them everywhere, so docstrings, comments and
  tutorial prose were being held to a rule the specification exempts.
- Every distinct match on a line is now reported rather than the first. Fixing one
  fault used to reveal the next only on the following run, so a clean result after an
  edit proved less than it appeared to.
- `ensure` and `acquire` are no longer substituted. `docs/ste-writing.md` records both,
  and the reasons differ: `ensure` is a disagreement with the source, and `acquire` is
  domain vocabulary here, since this program acquires a lock rather than obtaining an
  object.
- `[rtlsdr] frequency_hz` defaults to 3.588 MHz rather than 7.074. The receiver tunes
  50 kHz above it, so the whole 256 kHz span falls inside 80m and clear of the CW DX
  window. Powerline noise is generally worse low in HF. An existing config keeps
  whatever it already says.
- Settings that exist but that nobody should meet in a menu are marked `x-file-only`
  in the schema. They stay documented in `config.example.toml`, which is the only way
  anybody could edit them by hand. The five are the receiver's sample rate,
  decimation, bandwidth, tuning offset and sideband.

### Fixed
- A disk that was already full when an event started dropped a stray `.wav` on every
  poll for as long as the signal lasted. The opening write sat outside the guard that
  covers the file being opened, so it escaped into the trigger, which logs what a
  listener raises and carries on. No filename came back, and the trigger read the
  absence as a recorder that writes nothing rather than as one that failed, so it
  stayed armed and opened another file on the next poll. It now counts the recorders
  that answered rather than reading only the answers that arrived, and a lead-in write
  that fails gives up on the file the same way a later one does.
- A recording whose closing write failed left the recorder believing a file was still
  open for the rest of the run, so the next event replaced the writer without closing
  it. `wave` flushes the data and patches the header sizes when the file is closed,
  which is where a disk that filled during the event refuses. The close is inside the
  same guard as the rest of the writing now.
- The message refusing an impossible sample rate named `[audio] sample_rate` whichever
  recorder refused, so an IQ recording's bad rate sent the operator to edit a setting
  that was not the one at fault. Each recorder names the section its own rate came
  from.
- `[recording] record_iq` was ignored without a word on a station reading a sound
  card. A sound card has no IQ to record, and the setup program does not offer the
  setting there, but a hand-edited config file could still turn it on and get no
  files and no explanation. The monitor says so at startup, and `BuzzConfig.record_iq`
  now answers the question in one place rather than at each reader.
- An IQ recording wrote the requested receiver sample rate into its `.wav` header
  rather than the rate the hardware settled on, which is the rate the samples in the
  file are actually at. A 28.8 MHz divider cannot hit every request, so the two can
  differ. The audio rate was already read back off the device for the same reason.
- A recording whose disk filled part way through went on failing and saying so on
  every poll for the rest of the event, and never closed the file it could no longer
  write to. `wave` writes data and header sizes lazily, so a writer left open leaves
  the file in whatever state buffering happened to put it. The recorder now closes
  what it has, reports the failure once, and declines the rest of that event.
- The gain calibration dialog could replace its own answer with a stale progress line.
  Progress crosses from the sweep's thread by `call_soon_threadsafe`, which queues
  rather than runs, so one posted just before the sweep finished could arrive after
  the measured gain was on screen and overwrite it, leaving the operator looking at a
  step counter for a sweep that had already answered.
- A recording that could not open its file still counted against the event budget, so
  an operator with a full disk or an unwritable directory paid for files they never
  got, and recording ran out of budget having written nothing. The budget is now spent
  only by an event that produced a file.
- The setup program's Finish screen offered only Back when there was nothing to save,
  so somebody who opened it to leave the program was told there were no changes and
  sent to the menu they came from. It offers Exit as well, and says whether a config
  file exists, since with none the monitor runs on its built-in defaults.
- `# latitude =` and `# longitude =` in `config.example.toml` had nothing after the
  equals sign, so uncommenting either was a TOML syntax error. Both carry an example
  now, and the example is the Holmdel horn antenna.
- The setup program's level meters said nothing when audio stopped arriving. A read
  now gives up after a second and the meter says "no audio" instead of holding the
  last number it saw, which an operator had no way to tell from a live reading. The
  stall line is built to the width of a reading, so the block does not shift.
- The level meter's DC estimate used a weight that was correct for one block size and
  one sample rate, and quietly meant a different time constant for any other. It is
  computed from both now. The sound card's figure is unchanged.
- The analyzer's Tier-3a screening window was a fixed 4000 samples while the kernel it
  has to hold grows with the sample rate, so above about 34 kHz at 120 pps the kernel
  no longer fitted. `fftconvolve` accepts that and returns the swapped arrangement
  rather than raising, so re-acquisition was gated on a meaningless number over a
  third of the supported rate band. The window is counted in pulse periods now and
  comes to the same 4000 samples at 16 kHz.
- The analyzer's DC estimate weight is derived from the tick cadence and the time
  constant rather than written as 0.02, which was the answer for one cadence. The
  value is unchanged.
- The setup program froze the whole interface on the way out of the level meter and
  the gain sweep. Closing a receiver joins two threads with five second timeouts, and
  that ran on the event loop. Both close on a worker thread now.
- The gain sweep measured the receiver's DC offset along with the band. The monitor
  tunes away from that spike and filters it out, so the sweep was sizing the gain
  against something nothing downstream hears: an offset of 0.04 read 7.07 dB high, and
  worst at low gain. The quiet level now removes it and the peak still counts it,
  since clipping happens at the converter before any filtering.
- `[station] audio_rf_conversion_db` is no longer written to an RTL-SDR station's
  config file, and no longer overwritten at startup. It described a sound card, and a
  receiver's file carried it looking live while the monitor ignored it.
- Four descriptions in the setup program named no subject: "Snapped to the nearest
  step the tuner offers" left nothing doing the snapping. `ste_lint --fragments`
  reports the construct now, over Python and `schema.json`.
- A finished gain sweep offered no way to decline the figure it measured except
  Escape. It has a Cancel button beside "Use this gain" now, and the arrow keys move
  between the two, which they did not.
- The gain sweep's noise-floor measurement is thrown off by a running arc far less
  than it was. Its frames were 4 ms, the same order as a 120 pps burst, so nearly
  every frame straddled one and there was no quiet frame for the percentile to find:
  a 6 ms burst 25 dB over the floor read 21 dB high. A frame is 1 ms now, derived from
  the receiver's sample rate, which fits inside the 2.3 ms gap between bursts and
  brings that to 0.01 dB. It costs 0.17 dB on a clean band, and the same at every
  gain, so it leaves the gain the sweep picks unmoved.

  An arc dense enough to leave no gap is still measured as the floor, correctly: there
  is nothing else there to measure. Calibrate when the band is quiet.
- The gain sweep says the reserve leaves "at least" the headroom figure. The chosen
  gain is the lowest one where the antenna dominates, which is usually well below the
  highest the reserve allows, so the real margin is commonly a good deal more.
- The gain sweep uses the clipping it observes, which it recorded and then ignored. A
  gain that clipped during the sweep is not a prediction about arcs but one that
  happened, so it rules that gain out and every gain above it. The five passes are
  what make it worth consulting: an intermittent arc firing during any one of them is
  caught. On the station this was developed against that is the difference between
  25.4 and 22.9 dB, which is the step its operator had been taking by hand.

  The bar is the same 4 parts per million the monitor reports clipping at, rather than
  a single value at a rail. Five passes of a quarter second at 256 kHz collect 640,000
  raw values, so it takes three of them. The tuner's own DC offset puts the occasional
  sample at a rail with nothing arcing, and counting one of those capped the gain a
  step or more low with nothing said about why.
- The gain sweep gives up noise-floor accuracy rather than headroom when the two
  cannot both be had, and says how much it gave up. It used to refuse outright, and a
  station near the crossing then got no gain at all and set one by hand anyway,
  making that trade without the figures to make it on. Clipping is nonlinear and
  cannot be undone; a floor that reads high is wrong by a known amount in a known
  direction.
- The monitor no longer warns about every clipped sample. It reports above 4 parts
  per million, about 123 values a minute at 256 kHz, and says to run the calibration
  rather than to lower the gain a step. A handful a minute comes from the first burst
  of an intermittent arc, moves an averaged burst amplitude by eight millionths of a
  decibel, and acting on it costs a gain step, which below the knee costs one to three
  decibels on every noise floor reported afterwards.
- The clipping message offers the cheaper remedies before the expensive one: a higher
  band, where powerline noise is weaker, or a frequency further from where the antenna
  is resonant.
- The gain sweep reads the receiver synchronously on one thread rather than streaming
  it. Changing gain during an async stream is two threads touching one device: the
  capture thread sits inside librtlsdr driving libusb's event loop while the gain goes
  out as control transfers from somewhere else. Twice in a few dozen sweeps a transfer
  never completed, after which closing the receiver never returned and the program
  hung. There is no second thread now, so there is nothing to race, and no outstanding
  transfer for the close to wait on.

  A sweep can afford it: it throws away most of what it reads and measures a
  statistical property of noise, so samples missed between reads cost it nothing. The
  monitor still streams, because it cannot miss a sample.

  A synchronous read has to be a whole number of 512-byte USB packets, which pyrtlsdr
  documents as a FIXME and does not enforce. A bad size closes the device and raises a
  libusb error that says nothing about sizes, so it is refused up front instead.
- The gain sweep discards both buffers between the tuner and a measurement rather
  than one. The counted discard covers librtlsdr's transfer pool; the receiver's own
  queue was not covered, so the count spent itself on stale entries and let that many
  post-change blocks through in their place. Worst at the first step of a sweep, where
  the queue has been filling since the device opened.
- The gain sweep could finish its last step, show nothing, and leave the receiver
  held. The release happened in the worker's `finally` as an awaited call, so the
  event loop decided whether it ran, and anything raised after the sweep died inside
  the worker where Textual reports it nowhere the operator can see. The release now
  happens in the sweep's own thread, which no task cancellation can skip, and a
  failure past that point reaches the screen. A receiver the program could not release
  is said on screen too, since the consequence otherwise falls on the next run as
  LIBUSB_ERROR_ACCESS: a permissions error that is nothing of the sort.

  Opening it moved into that same thread, because the open had a thread of its own and
  the receiver it produced belonged to a task that cancellation could take away.
  Escape during the opening second, which takes about 0.72 s on this hardware, left
  the device held for the rest of the session and the next attempt reading as the same
  permissions error. A receiver that opens and then refuses to configure is released
  too, where the constructor's own exit hook had not been registered yet.
- Exiting the setup program could hang after a gain sweep. Progress crossed back to
  the interface with `App.call_from_thread`, which waits until the loop has run the
  callback, and a loop that is shutting down never runs it. CPython joins every
  thread-pool worker at interpreter exit, so one waiting thread hung the process
  rather than the dialog that orphaned it.
- `tools/ste_lint.py --changed` checks files git has not seen yet. A new file does not
  appear in `git diff`, so the gate read it as nothing to check and reported clean.
  Three findings sat in two new files through several green runs and surfaced only
  once the files were committed, which is the wrong moment.
- `GainSweep` rounds an even number of passes up to an odd one. The floor is combined
  with a median, and numpy's median of an even count averages the two middle values
  instead of picking one, which gives up the outlier rejection the passes exist for.
  Measured against a simulated arc: three, five and seven passes each recovered the
  arc-free answer 25 times out of 25, and two passes recovered it in none of them.
- `tools/ste_lint.py` exits 2 instead of reporting `clean` when it has checked nothing.
  A bare invocation with no paths and no `--changed`, or any named path that does not
  exist, used to print a clean line and exit 0. A mandatory gate could be skipped by
  misspelling its own argument.
- The attribution link in `docs/ste-writing.md` pointed at a path that no longer
  exists. The MIT notice has to travel with the work, so a dead pointer weakens it.

## [1.5.3] - 2026-08-24

### Added
- `[station] enable_all_time_summary`, which publishes a probability summary over the
  whole data set alongside the 7-day and 30-day ones. It is off by default, which is a
  change in behavior: that chart used to be published for everyone. It averages every
  day since `summary_start_date_iso`, which sets where the chart begins and now applies
  only when this is on.

  Turning it off leaves the chart already written where it is, here and on the web
  server. Startup names the file once and says it will no longer be updated.
- `tools/slow_workers.py`, a pytest plugin that delays every `asyncio.to_thread` call
  so a test racing a background worker fails reliably rather than intermittently. Load
  it with `PYTHONPATH=tools pytest tests/test_setup_app.py -p slow_workers --no-cov`.
  It found ten timing races beyond the one CI had reported.

- `scripts/`, for programs an operator runs rather than a contributor. `tools/` holds
  checks that answer a question about the code, and this is a different kind of thing.
- `scripts/batch_render_recordings.py`, which renders every recording in the recording
  directory to an `.mp4` under `renders/`, one per `.wav`, so a pile of events can be
  skimmed for the interesting ones. Each render is one `buzz.main --playback --render`
  run, and a render plays in real time, so the batch takes at least as long as the
  recordings do. `--jobs` runs several at once and defaults to 1, since each one is a
  whole monitor process with its own analysis thread and encoder. `--max-length` skips
  anything longer than a given number of seconds and renders everything when it is not
  given, `--limit` caps the count for a trial run, and `--recordings` and `--output-dir`
  override the directories. An existing `.mp4` is skipped rather than re-rendered, so an
  interrupted batch resumes, and a failed render deletes whatever it wrote so the retry
  does not mistake it for finished work.
- `[server] current_chart`, choosing how the fixed `data/current.png` address is
  published. `copy` uploads the chart a second time under that name and works on any
  web server, which is the default. `symlink` points the name at the day's dated chart
  instead and saves the upload, but Apache serves it only with `FollowSymLinks` on the
  upload directory, which some shared hosts refuse. An unrecognized value is refused at
  startup rather than once per minute in the log.

### Changed
- The published page updates the chart in place once per minute instead of reloading
  itself with a `<meta http-equiv="refresh">`. It fetches one fixed address,
  `data/current.png`, and reads the update time from that response's `Last-Modified`
  header, so it shows the moment the chart reached the web rather than a timestamp
  rendered into the page. The time is displayed in the reader's own timezone.
- The page no longer names a dated chart or carries a timestamp of its own, so it is
  the same document all day and changes only when the callsign, pulse rate, or station
  timezone changes.
- At the station's midnight the page stops updating and says so, rather than following
  the new day's chart down to its single data point. It detects this by comparing the
  station-local date of each `Last-Modified` against the one it saw on load, so a
  reader in another timezone sees the same behavior as one beside the receiver. A
  refresh resumes it. Neither half of this can stop the page updating: a browser whose
  ICU build does not know the station's timezone loses the pause alone rather than the
  whole script, since `Intl.DateTimeFormat` is built inside a `try` at the top of the
  script where a throw would otherwise end it before the first fetch, and the
  `response.body.cancel()` that discards the unread chart at the pause is skipped when
  a response carries no body, which would otherwise throw into a `catch` that says
  nothing and leave the reader polling a frozen page.
- The page stops polling an hour after the last chart it was given, and says so: "No
  update since 12:04 PM PDT. This page has stopped checking. Refresh this page to
  resume." Stopping the requests after a bounded time is what the midnight pause was
  for, and the pause alone could not deliver it, because it fires only when a fetched
  `Last-Modified` carries a new station-local date. A station that stops uploading at
  noon serves the same header forever, so the pause never came and a tab left open
  polled once a minute indefinitely. Every poll that brings no new chart counts toward
  the hour, including a server error, a `Last-Modified` that does not parse, and a
  fetch that never completes, so a server that is down is given up on as well. The
  limit is written as a duration divided by the poll interval rather than as a count
  of 60, so changing how often the page polls cannot quietly change how long it waits.
- An unchanged chart is no longer read or repainted. The poll costs a 304 already, and
  the page was still reading the body and swapping in a new object URL for bytes that
  were identical to the ones on screen.
- The 23:59 collection no longer suppresses the page's auto-refresh, which the browser
  now decides for itself.
- A reader with JavaScript turned off no longer sees the page promise an update it
  cannot make. Dropping the `<meta http-equiv="refresh">` left the status line claiming
  a minute-by-minute update that only the script can perform, and the update time comes
  from a `Last-Modified` header the script reads, so neither the refresh nor the
  timestamp happens without it. A `<noscript>` in the head hides the status line with a
  style rule, and a `<noscript>` beside it explains what the page cannot do. It also
  says the chart may be an old copy held by the browser and gives the forced reload that
  fetches the current one, because the chart is served without `Cache-Control` and a
  plain reload can paint a cached one. The style rule keeps the sentence in the markup,
  where somebody editing the wording will look for it, rather than in a
  `document.write` string.
- The page's text scales with the screen, holding 16 px from about 711 px wide upward
  and easing down to 13 px on a phone, so the wrapped paragraphs stop taking vertical
  space the chart wants. The size mixes `rem` with `vw` rather than using `vw` alone,
  which keeps the reader's own font-size setting in effect.

### Fixed
- Eleven setup-program tests raced the workers that fill their dialogs. They paused the
  app once and asserted, which passed on a developer's machine and failed on a loaded
  runner, and one of them duly failed on `main` after merging. They now wait for the
  condition they depend on, with a bounded timeout that says what never happened.
- A recording too short to hold one whole chunk of audio no longer hangs playback
  forever. `FilePlaybackPipeline` drops the partial trailing chunk, so a file under
  about 32 ms at the 16 kHz default has no chunks at all and the feeder starts at the
  end of it. The finished flag was set only just after a chunk was consumed, so such a
  file never set it: the feeder parked, `finished` stayed false, and `--render` waited
  on a replay that could neither start nor end, with only an outside timeout to stop
  it. The end of the file is now published from the condition itself rather than from
  one of the two routes to it. A recording that short is also refused up front, naming
  its length and the length playback reads at a time, because there is nothing useful
  to render from it and an empty `.mp4` is a poor way to find that out.
- The published page fits a narrow screen. Three faults combined: no viewport meta tag,
  so a phone laid the page out at a notional desktop width and scaled it down; no
  `max-width` on the 1600 px chart, so it drew at full size whatever the screen; and a
  centered flex item that overflows spills off both edges at once, leaving the left half
  in negative scroll space that neither scrolling nor zooming can reach. That last one
  is why the sides stayed cut off however far you scrolled. `width: 100vw` also counted
  the scrollbar gutter and forced a horizontal scrollbar on desktop.
- A `[server] remote_path` without its trailing slash no longer misplaces every
  upload. `/var/www/html/noise` was concatenated straight onto the first filename, so
  the index went to `/var/www/html/noiseindex.html` and the data to
  `/var/www/html/noisedata/`. Nothing failed: the transfer succeeded, nothing was
  logged, and the page was simply never where the web server looked. The trailing
  slash is added when it is missing. The publishing guide's own example omitted it,
  which is now corrected.
- The page's link to the data archive is relative rather than `/noise/data/`, which
  was one station's own URL layout and gave everyone else a link to nothing.
- Uploads write to a staging name and are renamed over the target, so a browser that
  fetches a file mid-upload can no longer read a truncated one. `sftp.put()` overwrites
  in place, which left a window of tens of milliseconds per file where a chart would
  draw half-painted. The staging name is one reused `.uploading` per directory, so a
  transfer abandoned by a crash is overwritten by the next upload rather than left
  behind, and its leading dot keeps it out of the directory listing. It is also removed
  before each upload rather than merely overwritten, because a staging *symlink* left by
  a dropped connection would otherwise be followed by the next `put()`, which would write
  that file's bytes into the dated chart the link pointed at.

  Where the server has no `posix-rename` extension, the plain rename is tried first and
  the target is removed only if that fails. paramiko reports every failed operation as
  `IOError` and gives an errno to just two of them, so a missing extension and a
  read-only directory look alike. Removing first would let a permission or quota error
  delete a good chart and then fail to replace it, which leaves the page showing a
  broken image until a later cycle succeeds. A failure carrying `EACCES` or `ENOENT` is
  now re-raised rather than treated as a missing extension.

  The staging path is derived with `rpartition` rather than `rsplit`, which returns the
  whole string when it finds no separator. A target with no directory, which is what
  `index.html` is when `[server] remote_path` is unset, gave the staging path
  `index.html/.uploading`. The upload failed on that every cycle and took the rest of
  the cycle down with it, so `current.png` was never published and the page never
  updated.

## [1.5.2] - 2026-08-20

### Added
- `tools/ste_lint.py`, which checks prose against the writing rules in `CLAUDE.md`
  and `docs/ste-writing.md`. It reads comments, docstrings, the messages of `raise`,
  `assert` and `logger.*`, Markdown outside code fences, and the operator-facing text
  in `schema.json`, so identifiers and command syntax are never mistaken for
  sentences. `--changed` limits it to the lines a diff added, and it exits 1 on a
  finding so it can run beside `ruff check .`. It cannot check the three rules that
  need a reader: sentence fragments, passive voice, and an `-ing` form used as the
  main verb.

### Changed
- The setup program edits a yes/no setting with a labeled pair of radio buttons
  rather than a switch. The switch was an unlabeled square that slid between two
  ends, and the words "on" and "off" appeared nowhere in the dialog, so reading a
  setting meant remembering which side had meant on. The radio pair names both
  choices and marks the one in force, and the mark stays put while the arrow keys
  move the cursor over it.
- The recording section shows every setting whatever "Arm recording at startup" is
  set to. That setting only decides whether the monitor starts armed. The Record
  button, the R key and `--enable-recording` all arm a run that started disarmed,
  and the monitor honors the directory, the event budget and the lock gates when
  they do, so hiding those seven settings kept an operator from choosing values that
  were going to be used anyway. Publishing still hides its own settings while it is
  off, because the monitor builds no uploader at all in that state.
- The scope's horizontal graticule now divides the pulse period into thirds rather
  than the sweep into tenths, giving 2.78 ms/div at 120 pps and 3.33 ms/div at 100
  pps. A division is then the spacing between the bursts of two arcing phases of one
  distribution circuit, so each phase occupies a cell of its own and the number of
  arcing phases can be counted by eye. The lines closing a whole pulse period are
  drawn brighter than the phase slots inside it. The division count comes from the
  configured pulse rate, not from the analyzer's measured grid frequency, so the
  graticule stands still.
- The scope's bright vertical rule at the center of the screen is gone. With nine
  divisions the center falls inside a cell rather than on a line, so the rule would
  have marked nothing.
- The scope graticule is about 10% brighter.
- The display panel is 648 px wide rather than 640, so the scope's nine divisions
  come out at exactly 72 px each instead of eight cells of 71 and one of 72. The
  waterfall keeps its own 640 px width, since that is set by the frequency scale, and
  is centered in the panel against a black margin 4 px either side. The window is
  therefore 742 px wide rather than 734, and a rendered `.mp4` is 742x248.
  `waterfall.panel_width()` rounds to a multiple of `lcm(H_DIVISIONS, 2)`, so the
  frame width stays even at every admitted sample rate and x264 keeps accepting it.

## [1.5.1] - 2026-08-08

### Added
- A Diataxis-organized documentation site under `docs/`, built with MkDocs Material
  and published to GitHub Pages at
  <https://spatula75.github.io/n6ol-powerline-qrm-monitor/>: tutorials, how-to guides,
  a command-line and configuration reference, and a page on how the analyzer works
  internally.

### Changed
- `README.md` is now a short summary that points to the documentation site, rather
  than duplicating it.

### Removed
- `README-analysis.md`, superseded by `docs/concepts/how-it-works.md`.

## [1.5.0] - 2026-08-06

### Added
- `tools/release_render_check.py`, a step in the release procedure (see
  `CONTRIBUTING.md`) that renders a recent recording at both ends of the
  sample-rate band the monitor admits, and at several rates in between, then
  checks each result for a real picture and a real sound, not just a well-formed
  container: no sustained black frame, audio above a near-silence floor, a
  decoded frame count matching what the renderer itself logged, and enough
  per-frame luma variation to rule out a static or duplicated picture. Asks the
  release engineer what to do when nothing in the recordings directory is recent
  enough rather than validating against a stale file or skipping the check
  silently.

- `lib/buzz/setup/`, the beginnings of a guided setup path for operators who would
  rather not hand-edit a TOML file. `schema.json` describes every setting the
  monitor has - type, default, legal values, and what it is for - and is the single
  source three things now read instead of repeating one another: validation of
  `~/.buzz/config.toml`, the generator that writes `config.example.toml`, and the
  setup program's own screens, below. A test pins every schema default against the
  dataclass default it describes, so the two cannot drift.

- The terminal setup program itself, run with `python -m buzz.setup` or via
  `setup.bat` / `setup.sh`.  A full-screen menu, styled after `raspi-config`, built
  entirely from `schema.json`.  A main menu lists every config section and checks
  one off once you have looked at it.  Entering a section lists its
  currently-visible fields and opens a dialog sized to the field's type to edit one
  (free text, a switch, or a labeled list for an `enum`), navigable with either the
  arrow keys or Tab.  Escape or Q always confirms before exiting, even with nothing
  changed.  Finish shows exactly what changed before writing anything, and Save
  backs up an existing `~/.buzz/config.toml` to a timestamped `.bak` file first - if
  the backup cannot be written, the config is left alone rather than overwritten.
  Black screen, cyan text, in the same phosphor color the oscilloscope display
  uses, built from the 16-color ANSI palette rather than RGB hex so it renders
  consistently across terminals rather than however each one happens to
  approximate an arbitrary color.  Runs on Windows, Linux, macOS, and BSD with no
  extra system dependency, via the new `textual` requirement.  `setup.bat` and
  `setup.sh` bootstrap the environment first: reuse `.venv` if it already exists
  and is Python 3.12 or later, otherwise find a system Python that is and create
  it, then install `requirements.txt` before launching - so a first-time user
  never has to know any of that happened before the guided setup starts.

- Device selection and level calibration, wired into the setup program.  The
  Audio section's input device field opens a picker instead of a text box: a
  one-shot probe of every input device, a level bar per device, and a disabled
  row for one that cannot open at the configured sample rate, with R to rescan.
  Rows show the device name alone; the value actually saved, and matched
  against at every later startup, still carries its host API too, since that
  is what tells apart the same physical device listed once per API.  Audio
  also gets a Calibration meter action: a live, read-only S-meter for matching
  a receiver's own S-meter by adjusting its RF and AF gain.  The Station
  section's audio-to-RF offset field opens a second, related dialog with the
  same meter instead of a plain number box, for the minority of receivers with
  no separate AF gain to adjust - an internal sound device, for instance -
  where the offset itself is the only thing left to calibrate: Up and Down
  nudge it, Space resets it to the default, and Enter confirms, all against
  the same live reading, updating without reopening the audio stream on every
  nudge.  The S-meter rendering (the dBm-to-S-unit string and the ASCII bar)
  lives in `buzz.setup.smeter`, so both new dialogs draw the same meter from
  one implementation.

- A timezone picker for the Station section's timezone field, so nobody has to
  type an IANA name like `America/Chicago` from memory.  Typing a few letters
  of a region or city filters tzdata's roughly 340 real zones, the same
  database the monitor resolves at runtime, so a name this dialog offers can
  never be one it later rejects.  Deprecated backward-compatibility aliases
  such as `UTC` and `Zulu` are left out - both are the same zone as `Etc/UTC`
  under an older name, and tzdata's own compiled source says so outright.
  Each row also shows its current UTC offset, read off the system clock so it
  already reflects whichever of daylight or standard time the zone is in
  today.

- `run.bat` and `run.sh`, matching `setup.bat`/`setup.sh`: start the monitor with
  no special arguments, using the `.venv` setup already created.  Each checks for
  that `.venv` and for `~/.buzz/config.toml` before doing anything else, and says
  plainly which one is missing and to run setup first, rather than starting with
  defaults nobody chose.  The file, not just the `~/.buzz` directory: `FinishScreen`
  creates that directory right before writing the config into it, so a setup run
  killed between those two steps would otherwise leave the directory behind with
  nothing in it, and pass a check that only looked for the directory.

### Changed
- Coverage measurement now covers `tools/` as well as `lib/buzz`.
  The release check is part of the release procedure now, so leaving it outside the
  gate meant 429 lines of it counted for nothing. Both tools reach 100%, and the
  total moved 99.19% → 99.29%.
- `config.example.toml` is generated from `lib/buzz/setup/schema.json` rather than
  hand-maintained, and a test fails if the committed copy stops matching what the
  schema would produce. It had been a third place every setting was described, after
  the dataclass comments and the setup program's own labels; the sample now cannot disagree
  with the code about a default, and six settings that had been showing example
  values as though they were defaults no longer do.

### Removed
- `level_meter.py`. The setup program's own Calibration meter action (see Added,
  above) draws the same live S-meter, so the standalone script had nothing left
  to do that the guided path did not already cover.
- `configure.py`. The setup program's Audio section device picker (see Added,
  above) covers the same ground - probing every input device, showing a level
  bar, and saving the chosen `input_device_name` - so the standalone console
  configurator had nothing left to do either. The interactive console flow it
  used (`select_device()` and the functions it called) went with it.
- `[station] distance_attenuation`. It was only ever added back onto qualifying
  signal levels to pad the daily chart's y-axis upper bound, for an estimated
  source-power series that was never drawn. Charts scale to the signal, noise, and
  audio-level anchor now, so they will generally be a little tighter than before.
- `[audio] device_index`. Nothing at runtime read it: the device is resolved by
  `input_device_name` at every startup, deliberately, because names survive a reboot
  and PortAudio indices do not. Its only remaining use was marking the current
  device in the device picker, which now matches on the name instead - so an
  index that has gone stale can no longer point the marker at the wrong device.
- Both are simply ignored if present in an existing `~/.buzz/config.toml`; unknown
  keys have always been dropped on load, so no config file needs editing.

## [1.4.0] - 2026-08-01

A recorded event can now be rendered to video with `--render FILE.mp4`: the display
and the recording's own audio, muxed together, so an arc heard at two in the morning
can be shown to somebody instead of just described to them. The waterfall stopped
depending on the sample rate along the way, so the display reads the same at 8 kHz and
48 kHz rather than only at the 16 kHz it was tuned against.

The rest is groundwork underneath that: shared constants replace six modules' worth of
copied numbers, error messages were audited to say what to do rather than just what
went wrong, and every docstring was checked against the code it describes rather than
trusted on sight.

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
- The device-setup level bar now shows 6 dB per segment, matching the S-meters
  elsewhere in the program, instead of 4.75. The bar spans 1 LSB to 16-bit full scale,
  which is a fixed 90.31 dB, so its width and its dB per segment are the same number
  said two ways. The width was a bare 19 from the configurator's first commit, and the
  caption beside it claimed "≈ 6 dB" in that same commit: the two never agreed, rather
  than drifting apart. 90.31 / 6 gives 15, and the width is derived from it now.
- `configure.py` no longer discards a customised `[recording]` or `[render]` section
  every time it saves. It loads the full six-section config but only ever wrote four
  of them back, so an event budget, `max_seconds`, or `ffmpeg_path` set by hand
  silently reverted to its default the next time a device was selected, with nothing
  in the output to say so.

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
- Constants two or more modules depend on now live in `buzz/constants.py` instead of
  each module keeping its own copy: 16-bit full scale, S9's dBm reference, and the
  6 dB-per-S-unit convention. `dsp.py`, `scope.py`, `waterfall.py`, `plotter.py`,
  `device_setup.py`, and `level_meter.py` all read from it, and `device_setup.py`'s
  bar width is now derived from the shared constants rather than a hand-computed
  literal, which is what the 4.75 dB fix above depends on to stay fixed.

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

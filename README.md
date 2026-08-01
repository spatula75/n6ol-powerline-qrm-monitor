# N6OL Powerline QRM Monitor

By [Nicklas Johnson, N6OL](https://n6ol.us/) — BSD 2-Clause License

A tool for ham radio operators to continuously monitor, log, and publish measurements
of powerline interference (QRM). It records audio from a radio receiver, detects the
characteristic pulse-train signature of powerline noise, and produces time-series plots
and time-of-day probability charts that are automatically uploaded to a web server.

Useful for documenting interference patterns when working with a power company to locate
and fix a problem source, or simply for understanding when the noise is worst.

Imported as a full-fledged repo from the original 
[gist](https://gist.github.com/spatula75/e6c654262e420aecf85ba7493a92ec31), first posted
to GitHub on May 12, 2024.  This project actually began on April 16, 2024 in an effort to 
track QRM produced by PG&E equipment near my station.  See [HISTORY.md](HISTORY.md) for
more detail.

## Sample output

Daily signal vs. noise floor (6-minute moving average):

![Daily plot](docs/sample_daily_plot.png)

30-day time-of-day interference probability:

![30-day probability summary](docs/sample_summary_plot.png)

---

## Requirements

- Python 3.12 or later
- A radio receiver with an audio output connected to a sound card line input
- An SSH-accessible web server for publishing output (optional but expected)
- A [CumulusMX](https://cumulusmx.com/) weather station or Open-Meteo API access for weather data (optional)

**Platform support:** Developed and tested on Windows. Linux and macOS should
work without code changes — the core DSP and collection code is fully
cross-platform and the CI runs on Linux. FreeBSD should also work, but `numba`
must be installed via the ports collection (`devel/py-numba`) rather than pip,
since pip does not ship FreeBSD binary wheels for numba.

---

## Installation

```
git clone https://github.com/spatula75/n6ol-powerline-qrm-monitor.git
cd n6ol-powerline-qrm-monitor
python -m venv .venv
```

Activate the virtual environment:

```
# Windows
.venv\Scripts\activate

# Linux / macOS / BSD
source .venv/bin/activate
```

```
pip install -r requirements.txt
```

**About virtual environments:** The `.venv` folder contains an isolated copy of
Python with its own installed packages, preventing conflicts with anything else
on your system. You need to activate it each time you open a new terminal before
running the monitor or configure script. PyCharm and VS Code can detect and
activate it automatically.

### Configuration

Copy the example config and edit it to match your setup:

```
# Windows
copy config.example.toml %USERPROFILE%\.buzz\config.toml

# Linux / macOS / BSD
mkdir -p ~/.buzz && cp config.example.toml ~/.buzz/config.toml
```

Open `~/.buzz/config.toml` in a text editor. Each setting has a comment
explaining what it does. At minimum you'll need to update the `[station]` path
and timezone, and the `[server]` section if you want uploads enabled.

Then run the audio device configurator — it will scan your input devices, show
live signal levels, and write the correct device settings into your config
automatically:

```
python configure.py
```

---

## Web server publishing (optional)

The monitor can upload its output files to a public web server after each
collection cycle, making the plots accessible from anywhere.  This requires
an SSH-accessible server and a passwordless SSH key pair.

### What gets uploaded

Each cycle the following are written locally and, when publishing is enabled,
uploaded to `remote_path/data/` on the server:

- `noise_data.YYYY-MM-DD.csv` — the day's raw measurements
- `noise_plot.YYYY-MM-DD.png` — the raw daily signal trace
- `noise_plot_movavg.YYYY-MM-DD.png` — the smoothed daily trace

On the hour, the three probability summary graphs are also uploaded.  After
every cycle `index.html` is rendered and uploaded to `remote_path/` (one level
above the data files) so it can serve as the site root.

### Generating an SSH key pair

The monitor authenticates with a dedicated private key so it can upload without
a password.  Generate one with:

```
# Linux / macOS / BSD / Git Bash on Windows
ssh-keygen -t ed25519 -f ~/.buzz/buzz.pem -N ""
```

This creates `~/.buzz/buzz.pem` (private key) and `~/.buzz/buzz.pem.pub`
(public key).  The `-N ""` sets an empty passphrase so the monitor can
authenticate unattended.

### Installing the public key on the server

Append the public key to the `~/.ssh/authorized_keys` file of the SSH user on
your web server:

```
cat ~/.buzz/buzz.pem.pub | ssh yourname@yourserver "cat >> ~/.ssh/authorized_keys"
```

Or if `ssh-copy-id` is available:

```
ssh-copy-id -i ~/.buzz/buzz.pem.pub yourname@yourserver
```

Test that key authentication works before enabling uploads:

```
ssh -i ~/.buzz/buzz.pem yourname@yourserver
```

### Enabling uploads in the config

Edit `~/.buzz/config.toml` and set the `[server]` section:

```toml
[server]
enabled = true
host = "yourserver.example.com"
username = "yourname"
remote_path = "/var/www/html/noise/"   # must end with /
key_path = "/home/yourname/.buzz/buzz.pem"
```

On Windows use forward slashes or double-backslashes in `key_path`:

```toml
key_path = "C:/Users/yourname/.buzz/buzz.pem"
```

---

## Radio Setup and Calibration

The monitor works by listening to a fixed audio level from your receiver.
Getting this right is the most important step — wrong AF gain means wrong dB readings.

### Choosing a frequency

This application performs no rig control — you set the frequency on the radio
manually, and the monitor simply listens to whatever audio the receiver produces.
Once you have settled on a frequency, enable your radio's **control lock** to
prevent accidentally nudging the VFO during a long monitoring run.

**Choose a frequency where you already suspect powerline interference is a
problem.**  The **80-meter band (3.5–4.0 MHz)** has worked well in practice,
but any band where you've noticed interference is a valid choice.

For best results, **bypass your antenna tuner** and tune to a frequency where
your antenna is naturally resonant.

### Receiver settings

- **AGC: OFF.** AGC will chase the noise and flatten everything to the same level,
  making it impossible to measure actual signal strength.
- **Filter: as wide as possible.** Powerline noise is broadband; a wide filter
  captures more of it and gives a stronger, more consistent reading.
- **Mode: LSB or USB.** Either works.
- **RF gain: start at 0 dB** (maximum — no attenuation).  If the interference
  is so strong it is overloading or saturating the receiver front end, reduce RF
  gain until the signal is clean, then increase AF gain to compensate.  This
  manual trade-off is a direct consequence of running with AGC off.
- **Preamp: off. Attenuator: off.**
- **Sound card input level: 0 dB** — no attenuation and no software amplification,
  just a straight pass-through of the signal on the line input.
- **AF (audio) gain: start low** and increase slowly.  Use the live level meter
  script to get a real-time reading from the program while you adjust:

  ```
  python level_meter.py
  ```

  This displays a continuously-updating text S-meter using the same amplitude
  calculation as the monitor.  Adjust RF and AF gain until the reading here
  matches your radio's S-meter, using **S9 = −73 dBm**, each S-unit = **6 dB**.

Once you have the RF gain, AF gain, and sound card input level set, **write them
down.**  These three settings form your calibration baseline; if you ever need to
reconnect the receiver or reinstall drivers, you will want to restore them exactly.

Once you have the AF gain set, leave it there. This is your calibration point.
The `audio_rf_conversion_db` setting in the config file fine-tunes the dB offset
between what the sound card measures and the actual RF level at your receiver input.

---

## Running the Monitor

```
python -m buzz.main
```

The monitor runs continuous audio analysis in the background, processing
frames approximately every 200 ms and accumulating results in a rolling
buffer.  At the top of each minute it averages the last full minute of
results, appends a row to today's CSV, regenerates the plots, and uploads
everything to the web server.  At the top of each hour it also regenerates
the three probability summary graphs (all-time, 7-day, 30-day).

To suppress the display window and run headlessly (useful when running as a
system service or on a machine without a monitor):

```
python -m buzz.main --headless
```

To keep the display window pinned on top of other windows:

```
python -m buzz.main --top
```

To arm event recording for this run, without editing the config file:

```
python -m buzz.main --enable-recording
```

---

## Recording Events

The monitor can save interference events to `.wav` files as they happen, and
replay them later through the same displays.  This is what to reach for when you
want to show somebody the noise rather than describe it: catch the event once,
then replay it as often as you like, at real speed, with a screen recorder
running — no receiver required on the machine doing the replaying.

Recording is configured in the `[recording]` section of the config file, armed at
startup with `--enable-recording`, and toggled while running with the toolbar
button or the **R** key.  Everything about it is off by default.

The button names the state rather than the action, so it reads **Record** when
recording is off and **Armed** once it is on, dimming at the same time — there is
nothing left to ask it for.  It stays clickable either way: dimmed is not
disabled, and it is also how you switch recording back off.

### What ends up in the file

```
|<-- lead-in -->|<---------- event ---------->|<-- trailer -->|
 already buffered   locked onto the pulse train  stop_after_seconds
 when lock happened                              with no lock
```

A recording begins when the analyzer locks onto the pulse train, but it does not
begin *at* that moment: the monitor is always holding the last several seconds of
audio in memory, and all of it goes into the file.  So the recording opens with
the run-up to the event instead of dropping you into the middle of it.  The end
works the same way — the audio recorded while waiting out `stop_after_seconds` is
kept, so every file ends with the noise floor the event faded into.

A signal that flickers stays one recording, as long as it comes back inside
`stop_after_seconds`.  Files are 16-bit mono PCM at the configured sample rate —
the same format the analysis runs on, with no conversion anywhere.

Each file is faded in and out over 5 ms, so it begins and ends at exactly zero
and never clicks — playing several back to back gives no pop at the seams.  A
file can genuinely begin or end on full-scale audio: if the arc is already
buzzing when the monitor starts, the lead-in is a live pulse train from its first
sample, and `max_seconds` ends a recording mid-event the same way.  Sound cards
also carry a small DC offset, which would step at both ends even in silence.  The
fade is a raised cosine and costs less than one pulse out of the 120 per second.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Arm recording at startup.  `--enable-recording` does the same for one run. |
| `directory` | `recordings/` under the station path | Where files are written, and where `--playback` looks for a bare filename.  Created at startup if missing. |
| `max_events` | `10` | How many of the next events to record before disarming.  `0` records every event. |
| `rearm_reset_minutes` | `0` | Minutes between resets of that budget.  `0` never re-arms. |
| `max_seconds` | `120` | How much to record once a recording starts.  The file is always longer — the lead-in already in the buffer, and the trailer, sit outside it.  `0` is uncapped. |
| `stop_after_seconds` | `10` | Silence before a recording is closed — and therefore how long the trailer is. |
| `min_lock_seconds` | `0` | How long the signal must hold before a recording starts.  Keep it to 5 s or less. |
| `min_lock_snr` | `0` | How strong the signal must be before a recording starts, in dB SNR.  Recording only. |

Recording disarms itself once `max_events` events have been captured, so the
defaults take the next ten events, at up to two minutes each, and then leave the
disk alone.  Pressing **Record** again starts a fresh count.  A recording stopped
by the length cap is not continued in a second file: the rest of that event is
skipped, and the next event starts the next recording.

### Ignoring events too short to be worth keeping

Not every lock is worth a file.  A night of two-second blips leaves a directory
full of recordings too short to sit and watch, and they count against
`max_events` just as a real event would.  `min_lock_seconds` holds off until the
interference has been present that long:

```toml
min_lock_seconds = 3
```

A signal that never lasts that long is never recorded at all, and a lock that
drops and returns starts the count again rather than adding up.

**The wait counts against `max_seconds`.**  Those seconds are part of the event —
the monitor has them buffered and keeps them — so `min_lock_seconds = 3` with
`max_seconds = 10` gives ten seconds of event, three of which you waited through
and seven recorded after.  Not thirteen.

**Keep it short — 5 seconds or less.**  It is also paid for out of the lead-in,
which comes from a sliding buffer only a few seconds long, so a recording that
waits two seconds opens two seconds later relative to the event than one that does
not.  Two ceilings apply, and the value is clamped to the lower with a warning
saying what was used instead: the buffer's length, beyond which the file would
begin *after* the event started and miss the onset entirely; and `max_seconds`,
which the wait cannot exceed without spending an allowance it is counted against.

### Ignoring events too faint to be worth keeping

The analyzer is sensitive enough to lock onto interference you can barely hear.
`min_lock_snr` keeps those off the disk:

```toml
min_lock_snr = 12
```

**This affects recording only.**  It does not change when a signal is locked,
measured, logged to the CSV, or drawn on the display — the monitor stays exactly
as sensitive as it was.  Locking happens at **6 dB SNR**, which is a constant in
the analyzer rather than a setting here, so any value at or below 6 does nothing
at all.

A signal that starts quiet and builds — which is how many arcs behave — is **not
skipped**.  Recording begins the moment it crosses the threshold, so the event is
caught even when its opening seconds are not.

**Time spent below the threshold is paid for out of the buffer, and the buffer runs
out.**  Only the last few seconds of audio are ever held, so:

```
lead-in kept  =  buffer length  −  time spent below the threshold
```

Sit at 6 dB for 3 seconds before crossing a threshold of 10 and you keep 6.6 of the
usual 9.6 seconds of run-up.  Sit there for 30 seconds and the run-up is gone
entirely — the file opens roughly 20 seconds *into* the event, having lost its
onset along with everything before it.

You still get a whole recording when that happens, and a full-length one.  Whatever
the buffer holds when the level crosses is lead-in and is free, however long the
wait made it; `max_seconds` then buys that much again on top.  So a long wait costs
you the run-up, never the recording.

This is the one way `min_lock_snr` is sharper-edged than `min_lock_seconds`.  That
setting is capped at the buffer's length, so it can never cost you the beginning of
an event; this one's wait depends on the signal, so it cannot be capped.  Set it
only as high as it needs to be to reject what you do not want.

The level is judged over about a second of readings rather than a single one, both
so one loud moment cannot carry a weak event through and because the first readings
after a lock are the least trustworthy — the analyzer's drift tracker has not
converged yet, and levels read several dB low until it does.

### Recording to a schedule

`rearm_reset_minutes` turns `max_events` into a rate rather than a one-off, which
is what makes unattended running practical.  With `max_events = 10` and
`rearm_reset_minutes = 1440`, the monitor records up to ten events a day, every
day, and cannot fill the disk while you are away.

The cycle is measured from when the budget was last reset rather than from when it
ran out, so it keeps its time of day.  A day whose ten events all arrive before
noon still gets its next ten at the same hour tomorrow, instead of sliding later
and later.  Unused events are not carried forward — a quiet day does not earn you
twenty events the next.

While the budget is spent, the toolbar shows when it comes back:

```
Recording off — re-arms in 23h 47m
```

Switching recording off with the **Record** button cancels the cycle as well.  Off
means off: a monitor that re-armed itself overnight because it happened to be
turned off mid-cycle would be a nasty surprise on your return.

The recording directory is created when recording is armed, not when the first
event arrives — at startup with `enabled = true` or `--enable-recording`, and
otherwise the moment you press **Record**.  Either way a mistyped path or a
permissions problem is reported there and then, rather than at the end of an
unattended night from an empty folder:

```
ERROR  buzz.recorder: Cannot create the recording directory D:\captures — recording
is off.  Check the directory setting in the [recording] section of the config, and
permissions on that path.
```

Recording is switched off in that case, but the monitor carries on measuring and
logging as usual — a directory nobody can write to should cost you your
recordings, not the day's data.  Fix the path and press **Record** to retry.

Files are named for the moment of lock, in station local time with the UTC offset
attached:

```
event-20260729-143307-0700.wav
```

### What the file remembers

Recordings carry standard RIFF metadata, which any audio editor or tagger can
read.  Tags name the station, the software version, and the moment of lock; a
comment records the settings needed to interpret the audio; and a cue marker sits
at the exact sample where the analyzer locked, so opening the file in an editor
*shows* you where the lead-in ends and the event begins.

```
INAM  N6OL powerline QRM event 2026-07-29T14:33:07-07:00
IART  N6OL
ICRD  2026-07-29T14:33:07-07:00
ISFT  n6ol-powerline-qrm-monitor 1.1.0
ICMT  sample_rate=16000 pulse_rate=120 audio_rf_conversion_db=-32.0
      lead_in_seconds=9.6 lead_in_max_seconds=9.6 ended=timeout
cue   sample 153600 → "LOCK"
```

`ended` is the one thing the audio itself cannot tell you: whether the recording
stopped because the event finished (`timeout`), because the length cap cut it
short (`capped`), or because you stopped it (`operator`, `shutdown`).

`lead_in_max_seconds` is there so `lead_in_seconds` can be read honestly.  The
buffer is a sliding window and it keeps sliding while `min_lock_seconds` is waited
out, so the lead-in can never exceed the buffer's capacity less that wait.  A file
sitting **at** the bound is telling you it kept everything it had — the arc was
already running and how long the lock really took is unknowable.  A file **below**
it is reporting a genuine measurement.  Without the bound written down the two look
identical, and neither the buffer size nor `min_lock_seconds` is recorded anywhere
else in the file.

With a 9.6 s buffer and `min_lock_seconds = 3`, every saturated recording reads
`lead_in_seconds` of about 6.6 rather than 9.6, which is the wait coming out of the
front.

### How long a file ends up

**Always somewhat longer than `max_seconds`.**  That setting measures the event,
not the file: it is how much is recorded from the moment recording starts.
The lead-in and the trailer sit outside it.

```
file length  =  whatever the buffer held  +  up to max_seconds  +  trailer
```

So `max_seconds = 10` with a full buffer gives a 9.6 + 10 = 19.6 s file, plus
whatever trailer the timeout adds.

Waiting changes which seconds those are, not how many.  `min_lock_seconds` is
counted against the allowance — you waited through that part of the event, and it
is kept — so three seconds of waiting means seven more recorded, not ten.  A
`min_lock_snr` wait is not counted, because it is open-ended: charging a wait that
can run to minutes would spend the whole allowance before the file was opened.

Set it by how much of the noise you want to study, not by how big you want the
files.

**Expect an overrun of up to 200 ms.**  The recorder works on a poll rather than
watching continuously, so a `min_lock_seconds` wait is noticed up to one poll after
it has actually elapsed, and only the configured value is charged against the
allowance — the remainder lands on top.  A 10 s setting can therefore produce
10.1 s of event.  The audio itself is trimmed to the sample; it is the moment the
allowance starts from that is quantised.

The log spells the sum out when each recording closes, because the total is not a
number any setting names:

```
Recorded event-20260730-080714-0700.wav — 12.6 s: 2.6 s lead-in + 10.0 s from the
lock (reached the 10 s limit)
```

A lead-in shorter than expected usually means the monitor had not been running
long enough to fill its buffer — an arc already buzzing at startup is locked onto
within a second or two, well before there is a full run-up to keep.  The log says
so when that is the reason.

### Replaying a recording

```
python -m buzz.main --playback event-20260729-143307-0700.wav
```

A bare filename is looked up in the recording directory; anything with a path in
it is used as given.  Playback runs at the file's own sample rate, so the
displays move at the speed the event actually happened.

### Files from somebody else

Any 16-bit PCM `.wav` plays, not only ones this program recorded — being sent a
capture by another operator and wanting to know whether it locks at 120 pps is
exactly the case this handles.  A file from a random sound card is likely to be
44.1 kHz and stereo, and both are fine:

- **Sample rate** may be anything from **8 kHz to 48 kHz**, and the display looks
  the same at all of them.  The FFT window is a fixed span of *time* rather than a
  fixed number of samples, so the waterfall keeps its 31 Hz per bin and its width
  whatever the audio arrives at.  The buffer behind it is sized in seconds too, so
  the analyzer always has the same 9.6 seconds of history to work with, and the
  scope always shows the same three pulse periods.  8 kHz is the floor because the
  display shows 0–4 kHz and that is exactly Nyquist there; anything outside the
  range is refused with a message rather than analysed into plausible nonsense.
- **Stereo** is reduced to **channel 0**, not mixed down — the same thing the live
  monitor does with a stereo input device — and it says so in the log.  Mixing
  would average the arc against whatever the other channel happens to hold.  If
  the interference is only on the right channel, extract that channel first.
- **Sample width** must be 16-bit.  The signal chain is int16 end to end, and
  silently converting a 24- or 32-bit file would put what is measured here at odds
  with what the same audio measured live.

What a foreign file cannot bring with it is the pulse rate and the level
calibration, since those live in metadata this program writes.  Your own configured
values are used instead and a warning says so, because a dBm reading taken against
someone else's receiver settings looks entirely plausible and means nothing.

The distinction is worth holding on to.  **Levels are suspect; the rest is not.**
Whether the pulse train locks at all, how steady its phase is, the line frequency
it reports, and the shape of the burst on the scope are properties of the
interference rather than of the receiver, and survive the trip intact.  Those are
what tell an arcing gap from somebody's switching supply.

If you know the calibration the sending station used, supply it for the replay:

```
python -m buzz.main --playback from-a-ham.wav --audio-rf-conversion-db -28.5
```

That figure is used in place of your own for this run only, and takes precedence
over one recorded in the file — the recording's own is normally right, being the
receiver that made it, so overriding it is reported.  If nothing locks at all,
suspect the grid instead: a European recording carries 100 pps, and against the
configured 120 the analyzer will never acquire.  Set `audio.pulse_rate` to 100 and
replay it.

The toolbar carries a transport instead of the record button, since there is
nothing to record and every reason to want to stop on an interesting moment:

```
Pause  Restart  Mute    ▶ 00:12 / 00:39 — event-20260729-184450-0700.wav
```

The first button is named for what clicking it does, so it reads **Pause** while
playing and **Play** while paused; **Space** does the same thing without moving
the mouse across the window you are recording.  **Restart** plays the file again
from the beginning, from wherever you are and whether or not it has finished.  At
the end of the file the time index turns to ■ and Play greys out, since Restart is
the only thing left to do.

### Hearing the replay

Playback sends the audio to your default output device, so you can hear the buzz
while you watch it; **Mute** (or **M**) silences it without stopping the replay.
To start silent instead — on a machine with no sound card, or when you only want
the displays — use `--mute`:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --mute
```

Muting is the absence of an output stream rather than a volume of zero, which is
what makes it safe on a machine that has no sound card at all: muted replay runs
exactly the code that ran before playback could be heard, paced by the monitor's
own clock.  Unmuted, the sound card becomes the clock instead — it is the one
that decides when the next chunk is actually wanted.

Pausing and reaching the end of the file release the device the same way, so a
replay left paused is not holding a sound card open with nothing to send it.

Powerline noise is usually recorded well below full scale — around −24 to −34 dBFS
is typical — which is quiet on laptop speakers.  `--playback-gain` turns it up:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --playback-gain 10
```

The gain is applied to the audio on its way to the sound card and to nothing else,
so it cannot move a single dB of what the analyzer measures, what the meters read,
or what any of it would have written to a CSV.  It is there to make the buzz
audible, not to change it.

Ask for more than the headroom allows and the loud parts will simply hit the rails
and **distort** — 10 dB on a −24 dBFS recording is comfortable, 30 dB will not be.
Turn it back down if it sounds crunchy; nothing about the analysis is affected
either way.

Rather than guessing, `--playback-gain auto` measures the recording and works the
figure out:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --playback-gain auto
```

It takes whichever is smaller of the gain that reaches −23 LUFS (the EBU R128
broadcast reference) and the gain that leaves true peak at −2 dBTP, so it gets as
close to a standard listening level as it can without letting anything clip.  A
recording whose bursts sit high above its noise floor hits the peak ceiling first
and comes out a little quieter, which is the right way round.  The result is a
single fixed gain — no compression and no limiting, because both would reshape the
pulse envelope that carries how bad the interference actually is.

`auto` needs ffmpeg, and it reads the whole file before playback starts, so there
is a pause of a second or two on a long recording.  The log says what it measured
and what it chose.

Switching between the two never loses your place in the file, and neither does
**Restart**, which throws away the fraction of a second already queued to the card
so the audio jumps back to the top with the display instead of trailing it.  If no
output device is available the button greys out and says so.

Restart resets the analyzer too, so the second pass is a genuine cold start:
lock indicator to FREE, meters to silence, drift rate and phase forgotten, and the
pulse train found again from nothing.  Watching the monitor acquire a signal is
usually the point of replaying an event, and an analyzer that still remembered
finding it the first time would open the second pass already locked.

It also takes the pulse rate and level calibration from the file's metadata, so a
recording measures the same wherever it is replayed — the sample rate is in the
`.wav` header, but nothing else about how to read the audio is, and a 100 pps
recording analysed as 120 pps simply never locks.  Any mismatch with your own
config is logged.  A `.wav` from anywhere else still plays; it just warns that it
is being analysed with your settings, which may not be the ones with which it was
made.

### Rendering a replay to video

`--render` records the replay to an `.mp4` — the display exactly as it appears,
with the recording as its soundtrack — so an arc heard at two in the morning
becomes something you can show somebody:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --render arc.mp4
```

The window opens without its transport controls, plays through once, and the
program exits when the file is complete.  A render happens in real time, so a
40-second recording takes 40 seconds.  It will not overwrite an existing file.

Add `--headless` to render without a window at all — useful on a machine with no
display, or simply to avoid one stealing focus for two minutes.  The video is
identical either way, because the display font is carried with the program rather
than borrowed from the desktop.  A headless render is silent as well: nobody is
watching, so nothing is sent to the sound card.  That does not change the video's
audio, which ffmpeg takes from the recording on disk.

The video is H.264 at 30 fps with AAC audio, chosen to play in VLC and in Firefox
without any persuasion.  The display genuinely updates ten times a second, so two
frames in three are duplicates; they cost almost nothing, and the finer grid is
what keeps picture and sound together.  Each frame is timed by where playback had
reached when its pixels were read, which means the analyzer's normal lag is
preserved rather than quietly corrected — the video shows what the monitor showed,
not an idealised version of it.

The recording's own metadata travels with it: what the event was, which station
heard it, when, and the calibration behind the numbers.

**`--render` implies `--playback-gain auto`**, because a rendered event sits around
−45 LUFS, well below a normal listening level.  That is by design: the
calibration process deliberately keeps the audio low, which is right for measuring
impulsive noise and awkward for showing it to somebody.  To override it, pass a figure — including zero for no gain at all:

```
python -m buzz.main --playback event.wav --render arc.mp4 --playback-gain 0
```

Whatever gain is used is written into the video's metadata as `render_gain_db`, so
the file says how far its audio was raised.  Watching a replay does *not* imply
auto gain — that is a different job, with the volume control to hand and nobody
waiting on a measurement.

Rendering needs **ffmpeg**, and nothing else in the monitor does.  If it is not on
your PATH, set `ffmpeg_path` in the `[render]` section of the config to the folder
holding it.  On Windows, `winget install Gyan.FFmpeg` puts it on PATH for any new
terminal.

### Short or weak recordings may not lock on replay

A replay is analysed exactly as live audio is, so it is subject to the same
acquisition behaviour — and a short file gives that behaviour very few chances.

While searching, the analyzer examines one second of audio at a time, once a
second.  Since the window and the interval are the same length, that is very nearly
continuous: measured across a replay, about 98% of the timeline is examined.  Two
things still work against a short, weak recording.

**The opening seconds are barely examined.**  The first search is made before a
full second has even been buffered, and the next lands about two seconds in.  A
three-second file therefore gets one or two real attempts at it, not thirty.

**The window averages.**  A burst shorter than a second is measured across the
whole second, so half a second of pulse train reads about 6 dB weaker than it
actually is — and the threshold for locking is 6 dB SNR.

Together these mean a brief, marginal event that locked when it was captured may
not lock when replayed.  Nothing is wrong with the recording; there is simply less
of it to work with.  **Restart** resets the analyzer and refills the buffer, which
gives it an independent second go — noise differs from pass to pass, and a
borderline signal can fail one attempt and pass the next.  Capturing more of an
event in the first place (`max_seconds`, or a longer `stop_after_seconds`) is the
better fix.

Replay is analysis only.  No CSV rows, no plots, no uploads, and no recording —
looking at an old event again must not add minutes to a day on which it did not
happen.
No audio *input* device is opened at all, so recordings can be reviewed on a
machine with no receiver attached to it.  When the file runs out, playback stops
and the displays hold their last frame.

---

## Display Window

When started without `--headless`, the monitor opens a live display window
with three panels and a toolbar:

```
+--------------------------------------+
|  Record    Recording off             |
+-----------------------------+--------+
|  Oscilloscope               |        |
+-----------------------------+ NF SIG |
|  Waterfall                  |        |
+-----------------------------+--------+
```

- **Toolbar** — across the top: arms recording, and shows what it is doing.
- **Oscilloscope** — top left, a synchronized view of the raw audio waveform.
- **Waterfall** — below it, a scrolling spectrogram.
- **S-meters** — the right-hand column, running the full height of the displays.

Four keys work anywhere in the window: **A** switches the scope between its raw
and averaged views, **R** arms or disarms recording, **Space** pauses or resumes
playback, and **M** mutes or unmutes it.

![Display window](docs/sample_waterfall_display.png)

The three bursts on the scope above are one powerline arc, caught three times in
a row — the sweep spans exactly three pulse periods, so the same event is drawn
at the same place on every pass.

### Oscilloscope

The scope shows the actual audio waveform, swept in sync with the interference
so that a repeating pulse train appears to stand still — the same effect as
setting the sync control on a bench oscilloscope to match the signal under
observation.

Each sweep covers **25 ms**, which is exactly three pulse periods at 120 pps, at
**2.5 ms per division**.  Rather than triggering on signal amplitude, the sweep
is synchronized to the pulse phase the analyzer is already tracking, so a brief
noise spike cannot false-trigger it and the picture holds steady even as utility
frequency drifts.

The trace has a **phosphor persistence** effect, like an analog storage scope.
This carries real information: the bright core is where successive sweeps agree,
and the dimmer halo around it is where they disagree.  The width of that halo is
a direct readout of how much the interference varies from cycle to cycle.

What powerline arcing typically looks like here may surprise you — not a sharp
spike, but a **symmetric burst of broadband noise a few milliseconds wide**.  A
gap discharge fires continuously for as long as the line voltage stays above the
gap's breakdown threshold, which is a substantial slice of each half-cycle rather
than an instant.  A wider burst generally means a worse fault, and unlike
amplitude that reading does not depend on your gain, antenna, or propagation.

The vertical scale auto-ranges from the signal, so the trace stays usefully sized
regardless of receiver gain.

**Press `A`** to switch between two views:

- **RAW** (default) — the bipolar waveform as received, noise and all.
- **AVG** — the rectified envelope, averaged over many sweeps.  Averaging pulls
  the pulse *shape* out of the noise (about 22 dB of improvement), revealing
  structure a single sweep buries.

#### Scope header

| Indicator | Meaning |
|---|---|
| **◆ LOCK** (green) | Sweep is synchronized to a live, tracked pulse train. |
| **◇ HOLD** (amber) | Signal has faded, but the sweep is still synchronized using the last known phase and drift rate.  A returning signal often becomes visible here before the analyzer formally re-locks. |
| **○ FREE** (grey) | No pulse train to synchronize to.  The sweep free-runs and its horizontal position is arbitrary. |
| `59.98 Hz` | Measured utility line frequency, derived from how fast the pulse phase is drifting.  Shown only when there is a phase to measure. |
| `RAW` / `AVG` | Which view is active — see `A` above. |
| `2.50 ms/div` | Horizontal timebase.  The trace is always three pulse periods wide, so this reads 2.50 ms/div on a 60 Hz grid and 3.00 on a 50 Hz one, whatever the sample rate. |
| `FS −24.3 dBFS` | Vertical full scale, as headroom below digital clipping.  This is the auto-range's current setting. |
| **CLIP** (red) | The input is hitting the converter's limit — reduce AF gain. |

`HOLD` does not persist indefinitely.  If the signal stays away long enough that
the extrapolated phase is no longer trustworthy, the analyzer gives up on it and
the indicator drops to `FREE`, rather than continuing to claim a synchronization
it cannot deliver.

### Waterfall

Below the scope is a scrolling waterfall spectrogram.  Each horizontal strip
represents one short audio frame; newer frames scroll in from the top.
Brighter colors indicate higher energy.  Powerline interference appears as a
repeating pattern of bright bands spaced evenly at the harmonic frequencies of
the configured pulse rate.

The color scale auto-ranges based on recent activity, so it stays readable
regardless of receiver gain or band conditions. It may take a little time to
settle into the live range after a cold start.

### S-meters

The right-hand column shows two signal-strength bars that update in real time.

- **NF** (noise floor) — average amplitude at the between-pulse positions,
  representing background noise.
- **SIG** (signal) — average amplitude at the pulse positions, representing
  the powerline interference.

Both bars use the standard ham radio scale: S9 = −73 dBm, each S-unit = 6 dB.
The difference between SIG and NF is the SNR.

Above each bar is a thin line showing the phase offset applied by
the most recent internal correction step.  A dot at center means no correction
was needed; a line extending left or right shows the direction and relative
magnitude of the correction.  This is a diagnostic indicator — most users can
safely ignore it, but it confirms the analyzer is actively tracking the pulse
train's phase as propagation conditions, sound-card clock variation, and
scheduling jitter cause gradual drift.

---

## How It Works

### Signal model

Powerline interference has a distinctive structure: arcing or corona discharge
on a power line fires at twice the AC line frequency — 120 pulses per second on
a 60 Hz grid, 100 pps on a 50 Hz grid. Each pulse is very short and broadband.
The monitor exploits this periodicity to separate the interference from background
noise.

### Continuous analysis pipeline

The analyzer processes audio continuously, running the following steps
approximately every 200 ms:

1. **Record audio.** `sounddevice` captures a short mono 16-bit PCM frame
   from the configured input device.

2. **Build a pulse-train kernel.** A sparse coefficient array is constructed
   with groups of three non-zero samples placed at the expected pulse positions
   for half a second's worth of pulses. The kernel is symmetric (a palindrome),
   so FFT convolution is mathematically equivalent to cross-correlation — no
   separate correlation step needed.

3. **FFT convolution.** `scipy.signal.fftconvolve` slides the kernel across the
   frame in O((N+M) log(N+M)) time. The output is a score at every sample
   position reflecting how well a pulse train starting there fits the data.
   The position with the highest score is the pulse phase; the position with the
   lowest score is the noise floor phase (halfway between pulses).

4. **Sum pulse trains.** A Numba-JIT compiled function sums the amplitude values
   at the actual pulse positions (peak phase) and at the midpoint positions (noise
   phase). Dividing by the count gives average peak and noise amplitudes.

5. **Convert to dB and compute SNR.** Both amplitudes are converted to dBFS
   (dB relative to full scale), then `audio_rf_conversion_db` is applied to get
   dBm. SNR is the difference between the two.

Results are stored in a rolling buffer covering approximately the last 72 seconds.
Every two seconds the analyzer also runs a phase-refinement step, scanning a
small window around the current pulse phase to keep the kernel aligned as
propagation conditions, sound-card clock imprecision, and processing jitter
cause gradual drift.

At each minute boundary the collector reads the buffer and averages the last
full minute of results.  Signal and SNR values are averaged only from frames
where the analyzer held a confirmed lock on the pulse train; noise floor is
averaged across all frames regardless of lock status.

### Output

- **Daily CSV** (`noise_data.YYYY-MM-DD.csv`) — one row per minute with
  timestamp, SNR, signal level (dBm), noise floor (dBm), Signal Lock Status,
  grid frequency, phase drift, and weather data.  All dBm values are averages
  over the last full minute of continuous analysis.  **Signal Lock Status** is
  `full` when the analyzer held a confirmed lock on the pulse train for the
  entire minute, `partial` when lock was held for part of the minute, or `none`
  when no lock was established (e.g., the interference was absent or too weak
  to acquire).  When the status is `none`, the signal level equals the noise
  floor and SNR is 0; the daily chart omits the red signal line for those
  intervals and draws only the green noise floor.

  **Grid frequency (Hz)** and **Phase drift (samples/s)** come from the
  analyzer's phase tracker and are blank for any minute with no lock, since the
  tracker has nothing current to report then.  Grid frequency is logged to three
  decimals, which is one digit past what the *absolute* accuracy supports: the
  whole reading is scaled by the sound card's sample-clock error, typically
  50–100 ppm, or 0.003–0.006 Hz at 60 Hz.  Treat the third decimal as meaningful
  for how the frequency **changes** — that error is a fixed scale factor and
  cancels out of any comparison — but read the absolute value as good to about
  ±0.01 Hz unless you have calibrated the sound card.  Because the error is a
  single multiplicative constant, calibrating later lets you correct the entire
  logged history by one scale factor; the raw phase drift is logged alongside so
  the underlying measurement is preserved rather than only the derived figure.

  These two columns sit immediately after Signal Lock Status.  Anything that
  parses these files by column position and reads past index 4 (the weather
  fields) needs updating for files written by this version onward; the monitor's
  own reader stops at index 4 and is unaffected, so older files still load.
- **Daily plots** — a 1600×640 px chart of signal vs noise floor over the day,
  plus a 6-point moving average version.
- **Probability summary graphs** — bar charts showing the normalized probability
  of interference at each 15-minute interval of the day, covering all time,
  the last 7 days, and the last 30 days.
- **index.html** — a simple page that embeds the moving-average plot and
  auto-refreshes every minute (except at 23:59 to avoid a midnight flip).

All output files are uploaded to the configured web server via a single SSH/SCP
connection per collection cycle.

---

## Potential improvements

**Directionality.**  The current design uses a single receiver and antenna and
can only measure signal strength — it cannot determine which direction a noise
source lies.  A set of inexpensive fixed magnetic loop antennas oriented in
different compass directions could potentially provide a rough initial bearing
by comparing signal intensities across the loops.  This idea is theoretical and
unexplored; even a vague heading would be useful for narrowing down which span
of power line to inspect.

**SDR integration.**  The monitor currently relies on a conventional receiver
feeding a PC sound card.  Inexpensive software-defined radio (SDR) dongles
(RTL-SDR, HackRF, etc.) could be driven directly from Python using libraries
such as `pyrtlsdr` or GNU Radio, eliminating the analog audio path and the
sound card calibration step.  An SDR approach would also make it straightforward
to monitor several frequencies simultaneously from a single device.

---

## Further reading

**IEEE Std 1897-2024 — IEEE Standard for Describing and Measuring Power-Line
Noise for Power-Line Communications**
([https://standards.ieee.org/ieee/1897/6837/](https://standards.ieee.org/ieee/1897/6837/))
*Purchase or IEEE subscription required.*

This standard defines rigorous methodologies for characterizing and locating
gap-type power-line interference sources — the same arcing and corona discharge
phenomena this monitor detects.  It covers measurement techniques, signal models,
and recommended practices.  Included here as a technical reference.

**"IEEE Recommendations for Locating Power-Line Gap Interference Sources,"**
*QST*, February 2026.

Written for a ham radio audience and published in the ARRL's flagship journal,
this article covers the history behind the development of IEEE Std 1897-2024
and the standardization effort that produced it.  Useful background for
understanding the context and motivation behind the standard before reading
the specification itself.

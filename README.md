# N6OL Powerline QRM Monitor

By [Nicklas Johnson, N6OL](https://n6ol.us/) - BSD 2-Clause License

A tool for ham radio operators. It monitors, logs, and publishes measurements of
powerline interference (QRM) continuously. It records audio from a radio receiver,
detects the pulse-train signature of powerline noise, and produces time-series plots
and time-of-day probability charts. It uploads those charts to a web server for you.

Use it to document interference patterns when you work with a power company to find
and fix a source, or to learn when the noise is worst.

This started as a
[gist](https://gist.github.com/spatula75/e6c654262e420aecf85ba7493a92ec31), first
posted to GitHub on May 12, 2024, and became a full repository. The project began on
April 16, 2024, to track QRM from PG&E equipment near my station. See
[HISTORY.md](HISTORY.md) for more.

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

**Platform support:** I develop and test on Windows. Linux and macOS should work with
no code changes. The core DSP and collection code is cross-platform, and CI runs on
Linux. FreeBSD should also work, but install `numba` from the ports collection
(`devel/py-numba`) rather than pip, because pip ships no FreeBSD binary wheels for it.

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

**About virtual environments:** The `.venv` folder holds an isolated copy of Python
with its own packages. It prevents conflicts with anything else on your system.
Activate it each time you open a new terminal, before you run the monitor or the
configure script. PyCharm and VS Code can find and activate it for you.

### Configuration

Copy the example config and edit it to match your setup:

```
# Windows
copy config.example.toml %USERPROFILE%\.buzz\config.toml

# Linux / macOS / BSD
mkdir -p ~/.buzz && cp config.example.toml ~/.buzz/config.toml
```

Open `~/.buzz/config.toml` in a text editor. A comment above each setting explains
what it does. At minimum, update the `[station]` path and timezone. Update the
`[server]` section too if you want uploads.

Then run the audio device configurator. It scans your input devices, shows live signal
levels, and writes the correct device settings into your config:

```
python configure.py
```

---

## Web server publishing (optional)

The monitor can upload its output files to a public web server after each collection
cycle, which makes the plots readable from anywhere. This needs an SSH-accessible
server and an SSH key pair with no passphrase.

### What gets uploaded

The monitor writes these locally each cycle. When publishing is on, it also uploads
them to `remote_path/data/` on the server:

- `noise_data.YYYY-MM-DD.csv` - the day's raw measurements
- `noise_plot.YYYY-MM-DD.png` - the raw daily signal trace
- `noise_plot_movavg.YYYY-MM-DD.png` - the smoothed daily trace

On the hour it also uploads the three probability summary graphs. After every cycle it
renders `index.html` and uploads it to `remote_path/`, one level above the data files,
so that it can serve as the site root.

### Generating an SSH key pair

The monitor authenticates with a dedicated private key, so it can upload without a
password. Generate one with:

```
# Linux / macOS / BSD / Git Bash on Windows
ssh-keygen -t ed25519 -f ~/.buzz/buzz.pem -N ""
```

This creates `~/.buzz/buzz.pem`, the private key, and `~/.buzz/buzz.pem.pub`, the
public key. The `-N ""` sets an empty passphrase, so the monitor can authenticate
without an operator.

### Installing the public key on the server

Append the public key to the `~/.ssh/authorized_keys` file of the SSH user on your web
server:

```
cat ~/.buzz/buzz.pem.pub | ssh yourname@yourserver "cat >> ~/.ssh/authorized_keys"
```

Or use `ssh-copy-id` if you have it:

```
ssh-copy-id -i ~/.buzz/buzz.pem.pub yourname@yourserver
```

Test that key authentication works before you switch uploads on:

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

On Windows, use forward slashes or double backslashes in `key_path`:

```toml
key_path = "C:/Users/yourname/.buzz/buzz.pem"
```

---

## Radio Setup and Calibration

The monitor listens to a fixed audio level from your receiver. Get this right first.
It is the most important step. Wrong AF gain gives wrong dB readings.

### Choosing a frequency

This application does not control the rig. Set the frequency on the radio yourself.
The monitor listens to whatever audio the receiver produces. Once you settle on a
frequency, switch on the radio's **control lock**, so that you cannot nudge the VFO
during a long run.

**Choose a frequency where you already suspect powerline interference.** The
**80-meter band (3.5–4.0 MHz)** has worked well here. Any band where you have noticed
interference is a valid choice.

For best results, **bypass your antenna tuner** and tune where your antenna is
naturally resonant.

### Receiver settings

- **AGC: OFF.** AGC chases the noise and flattens everything to one level. You then
  cannot measure signal strength at all.
- **Filter: as wide as possible.** Powerline noise is broadband. A wide filter catches
  more of it and reads stronger and steadier.
- **Mode: LSB or USB.** Either works.
- **RF gain: start at 0 dB**, the maximum, with no attenuation. If the interference is
  strong enough to overload or saturate the receiver front end, reduce RF gain until
  the signal is clean. Then raise AF gain to compensate. Running with AGC off is what
  makes this trade-off yours to manage.
- **Preamp: off. Attenuator: off.**
- **Sound card input level: 0 dB.** No attenuation and no software amplification. Pass
  the line input straight through.
- **AF (audio) gain: start low** and raise it slowly. Run the live level meter script
  to read the program's own figure while you adjust:

  ```
  python level_meter.py
  ```

  This shows a text S-meter that updates continuously. It uses the same amplitude
  calculation as the monitor. Adjust RF and AF gain until the reading matches your
  radio's S-meter. **S9 is −73 dBm**, and each S-unit is **6 dB**.

Once you have set the RF gain, the AF gain, and the sound card input level, **write
them down.** Those three settings are your calibration baseline. Restore them exactly
if you ever reconnect the receiver or reinstall drivers.

Once you set the AF gain, leave it alone. It is your calibration point. The
`audio_rf_conversion_db` setting in the config file trims the dB offset between what
the sound card measures and the real RF level at your receiver input.

---

## Running the Monitor

```
python -m buzz.main
```

The monitor analyzes audio continuously in the background. It processes a frame about
every 200 ms and collects results in a rolling buffer. At the top of each minute it
averages the last full minute, appends a row to today's CSV, regenerates the plots,
and uploads everything to the web server. At the top of each hour it also regenerates
the three probability summary graphs: all-time, 7-day, and 30-day.

To hide the display window and run headless, which suits a system service or a machine
with no monitor:

```
python -m buzz.main --headless
```

To pin the display window above other windows:

```
python -m buzz.main --top
```

To arm event recording for one run, without editing the config file:

```
python -m buzz.main --enable-recording
```

---

## Recording Events

The monitor can save interference events to `.wav` files as they happen, and replay
them later through the same displays. Reach for this when you want to show somebody
the noise rather than describe it. Catch the event once, then replay it as often as
you like, at real speed, with a screen recorder running. The machine that replays it
needs no receiver.

Configure recording in the `[recording]` section of the config file. Arm it at startup
with `--enable-recording`, and toggle it while the monitor runs with the toolbar button
or the **R** key. Everything about it is off by default.

The button names the state rather than the action. It reads **Record** when recording
is off, and **Armed** once it is on, and it dims at the same time, because there is
nothing left to ask it for. It stays clickable either way. Dimmed is not disabled, and
it is also how you switch recording off.

### What ends up in the file

```
|<-- lead-in -->|<---------- event ---------->|<-- trailer -->|
 already buffered   locked onto the pulse train  stop_after_seconds
 when lock happened                              with no lock
```

A recording begins when the analyzer locks onto the pulse train, but not *at* that
moment. The monitor always holds the last several seconds of audio in memory, and all
of it goes into the file. The recording therefore opens with the run-up to the event
instead of dropping you into the middle. The end works the same way. The monitor keeps
the audio it records while it waits out `stop_after_seconds`, so every file ends with
the noise floor the event faded into.

A signal that flickers stays one recording, as long as it returns inside
`stop_after_seconds`. Files are 16-bit mono PCM at the configured sample rate, which
is the format the analysis runs on. Nothing converts anywhere.

The monitor fades each file in and out over 5 ms, so it begins and ends at exactly zero
and never clicks. Play several back to back and you hear no pop at the seams. A file
really can begin or end on full-scale audio. If the arc is already buzzing when the
monitor starts, the lead-in is a live pulse train from its first sample, and
`max_seconds` ends a recording mid-event the same way. Sound cards also carry a small
DC offset, which would step at both ends even in silence. The fade is a raised cosine
and costs less than one pulse out of the 120 per second.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Arm recording at startup.  `--enable-recording` does the same for one run. |
| `directory` | `recordings/` under the station path | Where files are written, and where `--playback` looks for a bare filename.  Created at startup if missing. |
| `max_events` | `10` | How many of the next events to record before disarming.  `0` records every event. |
| `rearm_reset_minutes` | `0` | Minutes between resets of that budget.  `0` never re-arms. |
| `max_seconds` | `120` | How much to record once a recording starts.  The file is always longer - the lead-in already in the buffer, and the trailer, sit outside it.  `0` is uncapped. |
| `stop_after_seconds` | `10` | Silence before a recording is closed - and therefore how long the trailer is. |
| `min_lock_seconds` | `0` | How long the signal must hold before a recording starts.  Keep it to 5 s or less. |
| `min_lock_snr` | `0` | How strong the signal must be before a recording starts, in dB SNR.  Recording only. |

Recording disarms itself once it has captured `max_events` events. The defaults
therefore take the next ten events, at up to two minutes each, and then leave the disk
alone. Press **Record** again to start a fresh count. A recording that the length cap
stops does not continue in a second file. The monitor skips the rest of that event, and
the next event starts the next recording.

### Ignoring events too short to be worth keeping

Not every lock deserves a file. A night of two-second blips leaves a directory full of
recordings too short to sit and watch, and each one counts against `max_events` as a
real event would. `min_lock_seconds` holds off until the interference has been present
that long:

```toml
min_lock_seconds = 3
```

The monitor never records a signal that never lasts that long. A lock that drops and
returns starts the count again rather than adding up.

**The wait counts against `max_seconds`.** Those seconds are part of the event. The
monitor has them buffered and keeps them. So `min_lock_seconds = 3` with
`max_seconds = 10` gives ten seconds of event: three you waited through and seven
recorded after. Not thirteen.

**Keep it short, at 5 seconds or less.** The lead-in also pays for it. That audio comes
from a sliding buffer only a few seconds long, so a recording that waits two seconds
opens two seconds later against the event than one that does not. Two ceilings apply,
and the monitor clamps the value to the lower one and warns you which it used. The
first is the buffer's length, beyond which the file would begin *after* the event
started and miss the onset. The second is `max_seconds`, which the wait cannot exceed
without spending an allowance it counts against.

### Ignoring events too faint to be worth keeping

The analyzer is sensitive enough to lock onto interference you can barely hear.
`min_lock_snr` keeps those off the disk:

```toml
min_lock_snr = 12
```

**This affects recording only.** It does not change when the monitor locks a signal,
measures it, logs it to the CSV, or draws it. The monitor stays exactly as sensitive as
it was. Locking happens at **6 dB SNR**, which is a constant in the analyzer rather
than a setting here, so any value at or below 6 does nothing at all.

The monitor does **not** skip a signal that starts quiet and builds, which is how many
arcs behave. Recording begins the moment it crosses the threshold, so the monitor
catches the event even when it misses the opening seconds.

**The buffer pays for time spent below the threshold, and the buffer runs out.** The
monitor only ever holds the last few seconds of audio, so:

```
lead-in kept  =  buffer length  −  time spent below the threshold
```

Sit at 6 dB for 3 seconds before crossing a threshold of 10 and you keep 6.6 of the
usual 9.6 seconds of run-up. Sit there for 30 seconds and the run-up is gone. The file
then opens roughly 20 seconds *into* the event, and has lost the onset along with
everything before it.

You still get a whole recording when that happens, and a full-length one. Whatever the
buffer holds when the level crosses is lead-in, and it is free however long the wait
made it. `max_seconds` then buys that much again on top. A long wait therefore costs
you the run-up, never the recording.

This is the one way `min_lock_snr` cuts sharper than `min_lock_seconds`. The buffer's
length caps that setting, so it can never cost you the beginning of an event. This
one's wait depends on the signal, so nothing can cap it. Set it only as high as it
needs to be to reject what you do not want.

The monitor judges the level over about a second of readings rather than a single one.
That stops one loud moment from carrying a weak event through. It also avoids the first
readings after a lock, which are the least trustworthy: the drift tracker has not
converged yet, and levels read several dB low until it does.

### Recording to a schedule

`rearm_reset_minutes` turns `max_events` into a rate rather than a one-off, which is
what makes unattended running practical. With `max_events = 10` and
`rearm_reset_minutes = 1440`, the monitor records up to ten events a day, every day,
and cannot fill the disk while you are away.

The cycle runs from the last reset rather than from the moment the budget ran out, so
it keeps its time of day. A day whose ten events all arrive before noon still gets its
next ten at the same hour tomorrow, instead of sliding later and later. The monitor
does not carry unused events forward. A quiet day does not earn you twenty events the
next.

While the budget is spent, the toolbar shows when it returns:

```
Recording off - re-arms in 23h 47m
```

Switching recording off with the **Record** button also cancels the cycle. Off means
off. A monitor that re-armed itself overnight, because it happened to be turned off
mid-cycle, would be a nasty surprise on your return.

The monitor creates the recording directory when you arm recording, not when the first
event arrives. That means at startup with `enabled = true` or `--enable-recording`, and
otherwise the moment you press **Record**. Either way it reports a mistyped path or a
permissions problem there and then, rather than at the end of an unattended night from
an empty folder:

```
ERROR  buzz.recorder: Cannot create the recording directory D:\captures - recording
is off.  Check the directory setting in the [recording] section of the config, and
permissions on that path.
```

Recording switches off in that case, but the monitor carries on measuring and logging.
A directory nobody can write to should cost you your recordings, not the day's data.
Fix the path and press **Record** to retry.

The monitor names files for the moment of lock, in station local time with the UTC
offset attached:

```
event-20260729-143307-0700.wav
```

### What the file remembers

Recordings carry standard RIFF metadata, which any audio editor or tagger can read.
Tags name the station, the software version, and the moment of lock. A comment records
the settings you need to interpret the audio. A cue marker sits at the exact sample
where the analyzer locked, so opening the file in an editor *shows* you where the
lead-in ends and the event begins.

```
INAM  N6OL powerline QRM event 2026-07-29T14:33:07-07:00
IART  N6OL
ICRD  2026-07-29T14:33:07-07:00
ISFT  n6ol-powerline-qrm-monitor 1.1.0
ICMT  sample_rate=16000 pulse_rate=120 audio_rf_conversion_db=-32.0
      lead_in_seconds=9.6 lead_in_max_seconds=9.6 ended=timeout
cue   sample 153600 → "LOCK"
```

`ended` is the one thing the audio itself cannot tell you. It says whether the
recording stopped because the event finished (`timeout`), because the length cap cut it
short (`capped`), or because you stopped it (`operator`, `shutdown`).

`lead_in_max_seconds` is there so that you can read `lead_in_seconds` honestly. The
buffer is a sliding window and it keeps sliding while the monitor waits out
`min_lock_seconds`, so the lead-in can never exceed the buffer's capacity less that
wait. A file **at** the bound is telling you it kept everything it had. The arc was
already running, and how long the lock really took is unknowable. A file **below** it
is reporting a true measurement. Without the bound written down the two look identical,
and the file records neither the buffer size nor `min_lock_seconds` anywhere else.

With a 9.6 s buffer and `min_lock_seconds = 3`, every saturated recording reads
`lead_in_seconds` of about 6.6 rather than 9.6, which is the wait coming out of the
front.

### How long a file ends up

**Always somewhat longer than `max_seconds`.** That setting measures the event, not the
file. It is how much the monitor records from the moment recording starts. The lead-in
and the trailer sit outside it.

```
file length  =  whatever the buffer held  +  up to max_seconds  +  trailer
```

So `max_seconds = 10` with a full buffer gives a 9.6 + 10 = 19.6 s file, plus whatever
trailer the timeout adds.

Waiting changes which seconds those are, not how many. `min_lock_seconds` counts
against the allowance, because you waited through that part of the event and the
monitor keeps it. Three seconds of waiting therefore means seven more recorded, not
ten. A `min_lock_snr` wait does not count, because it is open-ended. Charging a wait
that can run to minutes would spend the whole allowance before the file opened.

Set it by how much of the noise you want to study, not by how big you want the files.

**Expect an overrun of up to 200 ms.** The recorder polls rather than watching
continuously, so it notices a `min_lock_seconds` wait up to one poll after the wait has
elapsed. It charges only the configured value against the allowance, and the remainder
sits on top. A 10 s setting can therefore produce 10.1 s of event. The monitor trims
the audio itself to the sample. What is quantized is the moment the allowance starts
from.

The log spells the sum out when each recording closes, because no setting names the
total:

```
Recorded event-20260730-080714-0700.wav - 12.6 s: 2.6 s lead-in + 10.0 s from the
lock (reached the 10 s limit)
```

A lead-in shorter than you expect usually means the monitor had not run long enough to
fill its buffer. It locks onto an arc that is already buzzing at startup within a
second or two, well before there is a full run-up to keep. The log says so when that is
the reason.

### Replaying a recording

```
python -m buzz.main --playback event-20260729-143307-0700.wav
```

The monitor looks a bare filename up in the recording directory, and uses anything with
a path in it as given. Playback runs at the file's own sample rate, so the displays
move at the speed the event happened.

### Files from somebody else

Any 16-bit PCM `.wav` plays, not only the ones this program recorded. Another operator
sends you a capture and you want to know whether it locks at 120 pps: that is exactly
the case this handles. A file from a random sound card is likely to be 44.1 kHz and
stereo, and both are fine.

- **Sample rate** may be anything from **8 kHz to 48 kHz**, and the display looks the
  same at all of them. The FFT window is a fixed span of *time* rather than a fixed
  number of samples, so the waterfall keeps its 31 Hz per bin and its width whatever
  the audio arrives at. The buffer behind it is sized in seconds too, so the analyzer
  always has the same 9.6 seconds of history, and the scope always shows the same three
  pulse periods. 8 kHz is the floor because the display shows 0–4 kHz, which is exactly
  Nyquist there. The monitor refuses anything outside the range with a message rather
  than analyzing it into plausible nonsense.
- **Stereo** reduces to **channel 0**. It is not mixed down, which is what the live
  monitor does with a stereo input device, and the log says so. Mixing would average the
  arc against whatever the other channel holds. If the interference is only on the right
  channel, extract that channel first.
- **Sample width** must be 16-bit. The signal chain is int16 end to end. Converting a
  24- or 32-bit file silently would put what you measure here at odds with what the
  same audio measures live.

A foreign file cannot bring the pulse rate and the level calibration with it, because
those live in metadata this program writes. The monitor uses your own configured values
instead and warns you, because a dBm reading taken against someone else's receiver
settings looks entirely plausible and means nothing.

The distinction is worth holding on to. **Levels are suspect. The rest is not.**
Whether the pulse train locks at all, how steady its phase is, the line frequency it
reports, and the shape of the burst on the scope are properties of the interference
rather than of the receiver. They survive the trip intact, and they are what tell an
arcing gap from somebody's switching supply.

If you know the calibration the sending station used, supply it for the replay:

```
python -m buzz.main --playback from-a-ham.wav --audio-rf-conversion-db -28.5
```

The monitor uses that figure in place of your own for this run only. It takes
precedence over one recorded in the file, and the monitor reports the override, because
the recording's own figure is normally right: it came from the receiver that made it.
If nothing locks at all, suspect the grid instead. A European recording carries 100 pps,
and the analyzer will never acquire it against the configured 120. Set
`audio.pulse_rate` to 100 and replay it.

The toolbar carries a transport instead of the record button. There is nothing to
record, and every reason to want to stop on an interesting moment:

```
Pause  Restart  Mute    ▶ 00:12 / 00:39 - event-20260729-184450-0700.wav
```

The first button is named for what clicking it does, so it reads **Pause** while
playing and **Play** while paused. **Space** does the same thing without moving the
mouse across the window you are recording. **Restart** plays the file again from the
beginning, from wherever you are and whether or not it has finished. At the end of the
file the time index turns to ■ and Play grays out, because Restart is the only thing
left to do.

### Hearing the replay

Playback sends the audio to your default output device, so you can hear the buzz while
you watch it. **Mute**, or **M**, silences it without stopping the replay. To start
silent instead, on a machine with no sound card or when you only want the displays, use
`--mute`:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --mute
```

Muting removes the output stream rather than setting a volume of zero, which is what
makes it safe on a machine with no sound card at all. A muted replay runs exactly the
code that ran before playback could be heard, paced by the monitor's own clock. Unmuted,
the sound card becomes the clock instead, because it decides when the next chunk is
wanted.

Pausing and reaching the end of the file release the device the same way. A replay left
paused therefore holds no sound card open with nothing to send it.

Powerline noise is usually recorded well below full scale, typically around −24 to
−34 dBFS, which is quiet on laptop speakers. `--playback-gain` turns it up:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --playback-gain 10
```

The monitor applies the gain to the audio on its way to the sound card and to nothing
else. It cannot move a single dB of what the analyzer measures, what the meters read,
or what any of it would have written to a CSV. It exists to make the buzz audible, not
to change it.

Ask for more than the headroom allows and the loud parts hit the rails and **distort**.
10 dB on a −24 dBFS recording is comfortable. 30 dB will not be. Turn it back down if
it sounds crunchy. Nothing about the analysis changes either way.

Rather than guess, use `--playback-gain auto`. It measures the recording and works the
figure out:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --playback-gain auto
```

It takes whichever is smaller: the gain that reaches −23 LUFS, the EBU R128 broadcast
reference, or the gain that leaves true peak at −2 dBTP. It therefore gets as close to
a standard listening level as it can without letting anything clip. A recording whose
bursts sit high above its noise floor hits the peak ceiling first and comes out a little
quieter, which is the right way round. The result is one fixed gain. Nothing is
compressed and nothing is limited, because both would reshape the pulse envelope that
carries how bad the interference is.

`auto` needs ffmpeg, and it reads the whole file before playback starts, so a long
recording pauses for a second or two. The log says what it measured and what it chose.

Switching between the two never loses your place in the file. Neither does **Restart**,
which throws away the fraction of a second already queued to the card, so the audio
jumps back to the top with the display instead of trailing it. If no output device is
available, the button grays out and says so.

Restart resets the analyzer too, so the second pass is a true cold start. The lock
indicator returns to FREE, the meters to silence, and the drift rate and phase are
forgotten. The analyzer finds the pulse train again from nothing. Watching the monitor
acquire a signal is usually the point of replaying an event, and an analyzer that still
remembered finding it the first time would open the second pass already locked.

It also takes the pulse rate and level calibration from the file's metadata, so a
recording measures the same wherever you replay it. The sample rate is in the `.wav`
header, but nothing else about how to read the audio is, and the analyzer simply never
locks a 100 pps recording that it analyzes as 120 pps. The monitor logs any mismatch
with your own config. A `.wav` from anywhere else still plays. It only warns that it is
being analyzed with your settings, which may not be the ones that made it.

### Rendering a replay to video

`--render` records the replay to an `.mp4`: the display exactly as it appears, with the
recording as its soundtrack. An arc heard at two in the morning becomes something you
can show somebody:

```
python -m buzz.main --playback event-20260729-184450-0700.wav --render arc.mp4
```

The window opens without its transport controls, plays through once, and the program
exits when the file is complete. A render happens in real time, so a 40-second
recording takes 40 seconds. It will not overwrite an existing file.

Add `--headless` to render without a window at all. That suits a machine with no
display, or simply avoids one stealing focus for two minutes. The video is identical
either way, because the program carries the display font rather than borrowing it from
the desktop. A headless render is silent as well: nobody is watching, so nothing goes
to the sound card. That does not change the video's audio, which ffmpeg takes from the
recording on disk.

The video is H.264 at 30 fps with AAC audio, chosen to play in VLC and in Firefox
without any persuasion. The display really does update ten times a second, so two frames
in three are duplicates. They cost almost nothing, and the finer grid is what keeps
picture and sound together. Each frame is timed by where playback had reached when the
program read its pixels, which preserves the analyzer's normal lag rather than quietly
correcting it. The video shows what the monitor showed, not an idealized version.

The recording's own metadata travels with it: what the event was, which station heard
it, when, and the calibration behind the numbers.

**`--render` implies `--playback-gain auto`**, because a rendered event sits around
−45 LUFS, well below a normal listening level. That is by design. The calibration
process deliberately keeps the audio low, which is right for measuring impulsive noise
and awkward for showing it to somebody. To override it, pass a figure, including zero
for no gain at all:

```
python -m buzz.main --playback event.wav --render arc.mp4 --playback-gain 0
```

The monitor writes whatever gain it uses into the video's metadata as `render_gain_db`,
so the file says how far its audio was raised. Watching a replay does *not* imply auto
gain. That is a different job, with the volume control to hand and nobody waiting on a
measurement.

Rendering needs **ffmpeg**, and nothing else in the monitor does. If it is not on your
PATH, set `ffmpeg_path` in the `[render]` section of the config to the folder that holds
it. On Windows, `winget install Gyan.FFmpeg` puts it on PATH for any new terminal.

### Short or weak recordings may not lock on replay

The monitor analyzes a replay exactly as it analyzes live audio, so the same
acquisition behavior applies. A short file gives that behavior very few chances.

While searching, the analyzer examines one second of audio at a time, once a second.
The window and the interval are the same length, so that is very nearly continuous:
measured across a replay, it examines about 98% of the timeline. Two things still work
against a short, weak recording.

**The opening seconds are barely examined.** The analyzer makes its first search before
a full second has even been buffered, and the next arrives about two seconds in. A
three-second file therefore gets one or two real attempts, not thirty.

**The window averages.** It measures a burst shorter than a second across the whole
second, so half a second of pulse train reads about 6 dB weaker than it is. The
threshold for locking is 6 dB SNR.

Together these mean a brief, marginal event that locked when you captured it may not
lock when you replay it. Nothing is wrong with the recording. There is simply less of it
to work with. **Restart** resets the analyzer and refills the buffer, which gives it an
independent second go. Noise differs from pass to pass, and a borderline signal can fail
one attempt and pass the next. Capturing more of an event in the first place, with
`max_seconds` or a longer `stop_after_seconds`, is the better fix.

Replay is analysis only. No CSV rows, no plots, no uploads, and no recording. Looking at
an old event again must not add minutes to a day on which it did not happen. The monitor
opens no audio *input* device at all, so you can review recordings on a machine with no
receiver attached. When the file runs out, playback stops and the displays hold their
last frame.

---

## Display Window

Started without `--headless`, the monitor opens a live display window with three panels
and a toolbar:

```
+--------------------------------------+
|  Record    Recording off             |
+-----------------------------+--------+
|  Oscilloscope               |        |
+-----------------------------+ NF SIG |
|  Waterfall                  |        |
+-----------------------------+--------+
```

- **Toolbar** - across the top: arms recording, and shows what it is doing.
- **Oscilloscope** - top left, a synchronized view of the raw audio waveform.
- **Waterfall** - below it, a scrolling spectrogram.
- **S-meters** - the right-hand column, running the full height of the displays.

Four keys work anywhere in the window. **A** switches the scope between its raw and
averaged views. **R** arms or disarms recording. **Space** pauses or resumes playback.
**M** mutes or unmutes it.

![Display window](docs/sample_waterfall_display.png)

The three bursts on the scope above are one powerline arc, caught three times in a row.
The sweep spans exactly three pulse periods, so every pass draws the same event at the
same place.

### Oscilloscope

The scope shows the audio waveform itself, swept in sync with the interference so that a
repeating pulse train appears to stand still. It is the same effect as setting the sync
control on a bench oscilloscope to match the signal you are watching.

Each sweep covers **25 ms**, which is exactly three pulse periods at 120 pps, at
**2.5 ms per division**. The sweep does not trigger on signal amplitude. It synchronizes
to the pulse phase the analyzer already tracks, so a brief noise spike cannot
false-trigger it, and the picture holds steady even as the utility frequency drifts.

The trace has a **phosphor persistence** effect, like an analog storage scope. This
carries real information. The bright core is where successive sweeps agree, and the
dimmer halo around it is where they disagree. The width of that halo reads out directly
how much the interference varies from cycle to cycle.

What powerline arcing looks like here may surprise you. It is not a sharp spike, but a
**symmetric burst of broadband noise a few milliseconds wide**. A gap discharge fires
continuously for as long as the line voltage stays above the gap's breakdown threshold,
which is a substantial slice of each half-cycle rather than an instant. A wider burst
generally means a worse fault, and unlike amplitude, that reading does not depend on
your gain, antenna, or propagation.

The vertical scale auto-ranges from the signal, so the trace stays usefully sized
whatever the receiver gain.

**Press `A`** to switch between two views:

- **RAW** (default) - the bipolar waveform as received, noise and all.
- **AVG** - the rectified envelope, averaged over many sweeps.  Averaging pulls the
  pulse *shape* out of the noise, about 22 dB of improvement, and reveals structure that
  a single sweep buries.

#### Scope header

| Indicator | Meaning |
|---|---|
| **◆ LOCK** (green) | Sweep is synchronized to a live, tracked pulse train. |
| **◇ HOLD** (amber) | Signal has faded, but the sweep is still synchronized using the last known phase and drift rate.  A returning signal often becomes visible here before the analyzer formally re-locks. |
| **○ FREE** (gray) | No pulse train to synchronize to.  The sweep free-runs and its horizontal position is arbitrary. |
| `59.98 Hz` | Measured utility line frequency, derived from how fast the pulse phase is drifting.  Shown only when there is a phase to measure. |
| `RAW` / `AVG` | Which view is active - see `A` above. |
| `2.50 ms/div` | Horizontal timebase.  The trace is always three pulse periods wide, so this reads 2.50 ms/div on a 60 Hz grid and 3.00 on a 50 Hz one, whatever the sample rate. |
| `FS −24.3 dBFS` | Vertical full scale, as headroom below digital clipping.  This is the auto-range's current setting. |
| **CLIP** (red) | The input is hitting the converter's limit - reduce AF gain. |

`HOLD` does not persist forever. If the signal stays away long enough that the
extrapolated phase is no longer trustworthy, the analyzer gives up and the indicator
drops to `FREE`, rather than claiming a synchronization it cannot deliver.

### Waterfall

Below the scope is a scrolling waterfall spectrogram. Each horizontal strip is one short
audio frame, and newer frames scroll in from the top. Brighter colors mean higher
energy. Powerline interference appears as a repeating pattern of bright bands, spaced
evenly at the harmonic frequencies of the configured pulse rate.

The color scale auto-ranges from recent activity, so it stays readable whatever the
receiver gain or the band conditions. It may take a little time to settle into the live
range after a cold start.

### S-meters

The right-hand column shows two signal-strength bars that update in real time.

- **NF** (noise floor) - average amplitude at the between-pulse positions,
  representing background noise.
- **SIG** (signal) - average amplitude at the pulse positions, representing
  the powerline interference.

Both bars use the standard ham radio scale: S9 is −73 dBm, and each S-unit is 6 dB. The
difference between SIG and NF is the SNR.

Above each bar is a thin line that shows the phase offset from the most recent internal
correction step. A dot at center means the analyzer needed no correction. A line
extending left or right shows the direction and relative size of the correction. This is
a diagnostic indicator, and most users can ignore it. It confirms that the analyzer is
tracking the pulse train's phase as propagation conditions, sound-card clock variation,
and scheduling jitter cause gradual drift.

---

## How It Works

### Signal model

Powerline interference has a distinctive structure. Arcing or corona discharge on a
power line fires at twice the AC line frequency: 120 pulses per second on a 60 Hz grid,
and 100 pps on a 50 Hz grid. Each pulse is very short and broadband. The monitor
exploits this periodicity to separate the interference from background noise.

### Continuous analysis pipeline

The analyzer processes audio continuously. It runs these steps about every 200 ms.

1. **Record audio.** `sounddevice` captures a short mono 16-bit PCM frame from the
   configured input device.

2. **Build a pulse-train kernel.** The analyzer builds a sparse coefficient array with
   groups of three non-zero samples at the expected pulse positions, covering half a
   second's worth of pulses. The kernel is symmetric, a palindrome, so FFT convolution
   is mathematically equivalent to cross-correlation. No separate correlation step is
   needed.

3. **FFT convolution.** `scipy.signal.fftconvolve` slides the kernel across the frame in
   O((N+M) log(N+M)) time. The output scores every sample position by how well a pulse
   train starting there fits the data. The highest-scoring position is the pulse phase.
   The lowest-scoring position is the noise floor phase, halfway between pulses.

4. **Sum pulse trains.** A Numba-JIT compiled function sums the amplitude values at the
   pulse positions, the peak phase, and at the midpoint positions, the noise phase.
   Dividing by the count gives average peak and noise amplitudes.

5. **Convert to dB and compute SNR.** The analyzer converts both amplitudes to dBFS, dB
   relative to full scale, then applies `audio_rf_conversion_db` to get dBm. SNR is the
   difference between the two.

The analyzer stores results in a rolling buffer that covers about the last 72 seconds.
Every two seconds it also runs a phase-refinement step. That scans a small window around
the current pulse phase to keep the kernel aligned as propagation conditions, sound-card
clock imprecision, and processing jitter cause gradual drift.

At each minute boundary the collector reads the buffer and averages the last full minute
of results. It averages signal and SNR only from frames where the analyzer held a
confirmed lock on the pulse train. It averages noise floor across all frames, whatever
the lock status.

### Output

- **Daily CSV** (`noise_data.YYYY-MM-DD.csv`) - one row per minute with timestamp, SNR,
  signal level (dBm), noise floor (dBm), Signal Lock Status, grid frequency, phase
  drift, and weather data.  All dBm values are averages over the last full minute of
  continuous analysis.  **Signal Lock Status** reads `full` when the analyzer held a
  confirmed lock for the entire minute, `partial` when it held lock for part of the
  minute, or `none` when it established no lock at all, which means the interference was
  absent or too weak to acquire.  When the status is `none`, the signal level equals the
  noise floor and SNR is 0.  The daily chart then omits the red signal line for those
  intervals and draws only the green noise floor.

  **Grid frequency (Hz)** and **Phase drift (samples/s)** come from the analyzer's phase
  tracker.  Both are blank for any minute with no lock, because the tracker has nothing
  current to report.  Grid frequency is logged to three decimals, which is one digit past
  what the *absolute* accuracy supports.  The sound card's sample-clock error scales the
  whole reading, typically by 50–100 ppm, or 0.003–0.006 Hz at 60 Hz.  Treat the third
  decimal as meaningful for how the frequency **changes**, because that error is a fixed
  scale factor and cancels out of any comparison.  Read the absolute value as good to
  about ±0.01 Hz unless you have calibrated the sound card.  The error is a single
  multiplicative constant, so calibrating later lets you correct the entire logged
  history by one scale factor.  The raw phase drift is logged alongside, which preserves
  the underlying measurement rather than only the derived figure.

  These two columns sit immediately after Signal Lock Status.  Anything that parses these
  files by column position and reads past index 4, the weather fields, needs updating for
  files this version onward writes.  The monitor's own reader stops at index 4 and is
  unaffected, so older files still load.
- **Daily plots** - a 1600×640 px chart of signal vs noise floor over the day, plus a
  6-point moving average version.
- **Probability summary graphs** - bar charts showing the normalized probability of
  interference at each 15-minute interval of the day, covering all time, the last 7
  days, and the last 30 days.
- **index.html** - a simple page that embeds the moving-average plot and auto-refreshes
  every minute, except at 23:59 to avoid a midnight flip.

The monitor uploads all output files to the configured web server over a single SSH/SCP
connection per collection cycle.

---

## Potential improvements

**Directionality.** The current design uses one receiver and one antenna, so it can
measure signal strength alone. It cannot tell which direction a noise source lies in. A
set of inexpensive fixed magnetic loop antennas, oriented in different compass
directions, could give a rough initial bearing by comparing signal intensities across
the loops. This idea is theoretical and unexplored. Even a vague heading would help
narrow down which span of power line to inspect.

**SDR integration.** The monitor relies on a conventional receiver feeding a PC sound
card. Inexpensive software-defined radio (SDR) dongles such as RTL-SDR and HackRF could
be driven directly from Python, using libraries such as `pyrtlsdr` or GNU Radio. That
would remove the analog audio path and the sound card calibration step. An SDR approach
would also make it straightforward to monitor several frequencies at once from one
device.

---

## Further reading

**IEEE Std 1897-2024 - IEEE Standard for Describing and Measuring Power-Line
Noise for Power-Line Communications**
([https://standards.ieee.org/ieee/1897/6837/](https://standards.ieee.org/ieee/1897/6837/))
*Purchase or IEEE subscription required.*

This standard defines rigorous methods for characterizing and locating gap-type
power-line interference sources, which are the same arcing and corona discharge
phenomena this monitor detects. It covers measurement techniques, signal models, and
recommended practices. It appears here as a technical reference.

**"IEEE Recommendations for Locating Power-Line Gap Interference Sources,"**
*QST*, February 2026.

Written for a ham radio audience and published in the ARRL's flagship journal, this
article covers the history behind IEEE Std 1897-2024 and the standardization effort that
produced it. It is useful background for understanding the context and motivation behind
the standard before you read the specification itself.

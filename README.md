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

- Python 3.11 or later
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

The monitor wakes up at the top of each minute, takes `measurements_to_take`
recordings, appends a row to today's CSV, regenerates the plots, and uploads
everything to the web server. At the top of each hour it also regenerates the
three probability summary graphs (all-time, 7-day, 30-day).

---

## How It Works

### Signal model

Powerline interference has a distinctive structure: arcing or corona discharge
on a power line fires at twice the AC line frequency — 120 pulses per second on
a 60 Hz grid, 100 pps on a 50 Hz grid. Each pulse is very short and broadband.
The monitor exploits this periodicity to separate the interference from background
noise.

### Measurement pipeline (once per minute)

1. **Record audio.** `sounddevice` captures a mono 16-bit PCM recording of
   `duration` seconds from the configured input device.

2. **Build a pulse-train kernel.** A sparse coefficient array is constructed
   with groups of three non-zero samples placed at the expected pulse positions
   for half a second's worth of pulses. The kernel is symmetric (a palindrome),
   so FFT convolution is mathematically equivalent to cross-correlation — no
   separate correlation step needed.

3. **FFT convolution.** `scipy.signal.fftconvolve` slides the kernel across the
   recording in O((N+M) log(N+M)) time. The output is a score at every sample
   position reflecting how well a pulse train starting there fits the data.
   The position with the highest score is the pulse phase; the position with the
   lowest score is the noise floor phase (halfway between pulses).

4. **Sum pulse trains.** A Numba-JIT compiled function sums the amplitude values
   at the actual pulse positions (peak phase) and at the midpoint positions (noise
   phase). Dividing by the count gives average peak and noise amplitudes.

5. **Convert to dB and compute SNR.** Both amplitudes are converted to dBFS
   (dB relative to full scale), then `audio_rf_conversion_db` is applied to get
   dBm. SNR is the difference between the two.

6. **Average.** Steps 1–5 repeat `measurements_to_take` times and the results
   are averaged before being written to the CSV.

### Output

- **Daily CSV** (`noise_data.YYYY-MM-DD.csv`) — one row per minute with
  timestamp, SNR, signal level, noise floor, and weather data.
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

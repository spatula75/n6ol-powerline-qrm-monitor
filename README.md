# N6OL Powerline QRM Monitor

By [Nicklas Johnson, N6OL](https://n6ol.us/) — BSD 2-Clause License

A tool for ham radio operators to continuously monitor, log, and publish measurements
of powerline interference (QRM). It records audio from a radio receiver, detects the
characteristic pulse-train signature of powerline noise, and produces time-series plots
and time-of-day probability charts that are automatically uploaded to a web server.

Useful for documenting interference patterns when working with a power company to locate
and fix a problem source, or simply for understanding when the noise is worst.

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

## Radio Setup and Calibration

The monitor works by listening to a fixed audio level from your receiver.
Getting this right is the most important step — wrong AF gain means wrong dB readings.

**Receiver settings:**
- **AGC: OFF.** AGC will chase the noise and flatten everything to the same level,
  making it impossible to measure actual signal strength.
- **Filter: as wide as possible.** Powerline noise is broadband; a wide filter
  captures more of it and gives a stronger, more consistent reading.
- **Mode: LSB or USB.** Either works.
- **RF gain: 0 dB** (or maximum, depending on your radio's convention — no attenuation).
- **Preamp: off. Attenuator: off.**
- **AF (audio) gain: start low** and increase slowly until the signal strength reported
  by the program roughly matches your S-meter reading, using the correspondence
  **S9 = −73 dBm** with each S-unit equal to **6 dB**
  (S8 = −79, S7 = −85, S6 = −91, etc.).

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

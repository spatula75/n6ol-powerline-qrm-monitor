# Command line reference

## Launching

The N6OL Powerline QRM Monitor is designed to run from a Python virtual environment, and creates one called `.venv` the first time the setup program is run.  The `run.bat` and `run.sh` scripts simply launch the program with no arguments using the virtual environment.  If you wish to run with different arguments on the command line, be sure to activate the venv first.  On Windows: `.venv\Scripts\activate` and on Linux, FreeBSD, and macOS: `source .venv/bin/activate`.

To run the main program, use `python -m buzz.main` followed by optional arguments.  To run the setup program, use `python -m buzz.setup`.

## Optional command line arguments

`--headless` - run without the Qt display, suitable for a 24x7 monitoring process or if there's no value in keeping a display active.

`--top` - keeps the display on top of all other windows (always visible).  Has no effect with `--headless`, since there is no window to keep on top.

`--enable-recording` - even if recording is not enabled at startup in the configuration file, this arms recording on startup.

`--playback <filename>` - replay a saved recording (.wav file). If the file is found in the default recordings directory, a full path is not required. May be used with `--mute`, `--playback-gain`, `--audio-rf-conversion-db`, and `--render`.

`--mute` - in playback, mutes the audio output, so the display is still rendered, but without any sound.

`--playback-gain <dB value or 'auto'>` - either a gain in decibels to apply to the recorded audio when playing back, or the word `auto` to normalize the audio to -23 dB LUFS.

`--audio-rf-conversion-db <dB value>` - useful when playing back a .wav file the monitor did not generate, if you know the signal strength the receiver reported when it was recorded.  This number represents the offset between the full-scale audio amplitude and the dBm value reported by the receiver.  It is used to make the S meters on the display accurate.  For .wav files produced by the monitor, this value is carried in metadata and applied automatically.

`--render <filename>` - render the playback to a video file, in real time. Can be used with `--headless`.

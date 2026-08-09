# Command line reference

## Launching

The N6OL Powerline QRM Monitor is designed to run from a Python virtual environment, and creates one called `.venv` the first time the setup program is run.  The `run.bat` and `run.sh` scripts simply launch the program with no arguments using the virtual environment.  If you wish to run with different arguments on the command line, be sure to activate the venv first.  On Windows: `.venv\Scripts\activate` and on Linux, FreeBSD, and macOS: `source .venv/bin/activate`.

To run the main program, use `python -m buzz.main` followed by optional arguments.  To run the setup program, use `python -m buzz.setup`.

## Optional command line arguments

`--headless` - run without the Qt display, suitable for a 24x7 monitoring process or if there's no value in keeping a display active.

`--top` - incompatible with `--headless`, when running with the display active, keeps the display on top of all other windows (always visible)

`--enable-recording` - even if recording is not enabled at startup in the configuration file, this arms recording on startup.

`--playback <filename>` - replay a saved recording (.wav file). If the file is found in the default recordings directory, a full path is not required. May be used with `--mute`, `--playback-gain`, `--audio-rf-conversion-db`, and `--render`.

`--mute` - in playback, mutes the audio output, so the display is still rendered, but without any sound.

`--playback-gain <dB value or 'auto'>` - either a gain in decibels to apply to the recorded audio when playing back, or the word `auto` to normalize the audio to -23 dB LUFS

`--audio-rf-conversion-db <dB value>` - useful when playing a .wav file that was not generated with the analyzer when the signal strength of the receiver at the time the recording was made is known.  This number represents the offset between the full-scale audio amplitude and the dBm value reported by the receiver.  It is used to make the S meters on the display accurate.  For .wav files produced by the monitor, this value is carried in metadata and applied automatically.

`--render <filename>` - realtime render the playback to a video file. Can be used with `--headless`.

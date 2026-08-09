# Replay and render recordings

## Introduction

In addition to being able to analyze powerline noise from an audio source like your radio, the N6OL Powerline QRM Monitor can use an audio recording as a source to analyze.

When the monitor creates a recording, it saves metadata to the recording to preserve the calibration offset for replay (see the [Calibration tutorial](../tutorials/calibration.md) for details about audio-to-RF-level calibration), so the analysis should look nearly identical (with some small timing variation) to the way it first appeared on the monitor when it was recorded.

When analyzing .wav files *not* created by the monitor, the S-meter readings may not be accurate, but the analysis itself should perform similarly as though the audio were received off-the-air.

The analysis front-end can also be rendered real-time to a video file, suitable for sending to others.

## Activating the Python virtual environment

When running the main module, it is necessary to ensure you are running in the Python virtual environment created during [initial setup](../tutorials/getting-started.md).  To do this on Windows, run `.venv\Scripts\activate` and on macOS, Linux, and FreeBSD use `source .venv/bin/activate`.  To disable the virtual environment later for all operating systems simply use `deactivate`.

## Playback

In the simplest form, use `python -m buzz.main --playback <filename>` where the filename matches the name of the recording you'd like to play back.  If you are playing a recording already in your recordings directory, you can specify just the name of the file itself.  For any other directory, give the full path.

Logging, charting, and recording are disabled during playback.

If you are playing back a file that wasn't recorded using the monitor, you can specify `--audio-rf-conversion-db -<value>` to provide the offset between the audio level in the recording and the RF dBm level (shown on the S meters in the main display).  This generally requires that whoever provided the file tells you what the RF level was at the time the recording was made; then you can choose a value accordingly.  For example, `--audio-rf-conversion-db -28.5`

Should you not wish to hear any audio during playback, you can add `--mute` to the command line.  If, on the other hand, you wish to apply gain to hear the audio better (as files are often recorded at a low amplitude), add `--playback-gain` followed by either a dB gain you would like applied, or the word `auto` to set the gain to -23dB LUFS (the EBU R128 broadcast reference) with a peak amplitude of no more than -2dB.

## Rendering to video

Occasionally it is helpful to be able to provide the full visual analysis of a recording to someone or to use in a video production.  The monitor can render to video in realtime by adding `--render <filename.mp4>` to the playback options.  This can be combined with the `--headless` option if you do not wish the UI to become visible during rendering.

Rendering is performed in real-time just as though it had been screen recorded during a playback, and thus can be susceptible to the same types of local interruptions that can happen with the UI in general (e.g., high CPU utilization causing lag or dropped frames).  Thus it is advised to close any other programs which may cause high CPU or disk utilization during rendering, and it may work better to render to a local file rather than to a network storage device.

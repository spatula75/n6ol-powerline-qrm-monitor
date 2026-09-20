# Set up an SDRPlay RSP1-series Receiver

## Introduction

The SDRPlay RSP1 series of receivers are relatively inexpensive but good quality SDR devices which are very capable
of monitoring powerline noise with a very low noise floor.  Though more expensive than RTLSDR devices by about a
factor of 3 at the time of this writing, this also gets you 14-bit resolution (15 with oversampling), a more
rugged build, and in this author's opinion, a higher quality experience overall.

Support is realized in the monitor by directly wrapping the SDRPlay Hardware API library with generated Python code,
as at the time of this writing, no simple, straightforward means of accessing SDRPlay devices existed.  (Some 
libraries like SoapySDR do exist; however, I felt the installation was way too fiddly.)

This has been tested on Windows only at this time, but if you have success with other operating systems,
please submit what you discover so that it can be included here.

## Installation

The monitor expects to find the SDRPlay libraries in `C:\Program Files\SDRplay\API\x64` on Windows and in 
`/usr/local/lib` or `/usr/lib` on Linux.  The shared objects are not installed with the SDR Connect product, and
need to be downloaded separately from the [SDRPlay Hardware API site](https://sdrplay.com/hardware-api/) and
then installed on your machine.

Should you need to point to a different directory for the library files, you can edit your `config.toml` file
and supply an `api_path` setting with the correct full path to the directory containing the library files.

The Python API wrappers need to be tied to a particular version of the API library, because any structural changes
can damage the space-time continuum.  Currently, the monitor is expecting version 3.15 of the SDRPlay Hardware API.
If a different version is found during startup, the monitor will log a complaint and exit.

## Notes

This code was tested against the RSP1B device, and measurements and estimates found in the code are based on this
device only.  You may find with other versions of the device that you need to make manual changes to your
configuration.  In particular, the Level calibration setting may need to be adjusted one direction or another to
match a known dB or S-level reading from your antenna.

# Set up an RTL-SDR.com v4 Receiver

## Introduction

The RTL-SDR.com v4 receiver is an inexpensive and functional SDR device which may be perfectly suitable for 
monitoring powerline noise, freeing up your expensive radio gear for more lofty pursuits.

Support in this monitoring tool is realized through the use of the `pyrtlsdr` Python package, which in turn
wraps the free and open source `librtlsdr` library.

This has been tested on Windows only at this time, but if you have success with other operating systems,
please submit what you discover so that it can be included here.

## General Notes
The RTL-SDR.com documentation may still suggest replacing a shared library or building a shared library from 
source in order to use the v4 receiver.  This is not necessary when using `pyrtlsdr` and will in fact break 
functionality if you follow this advice.  `librtlsdr` already knows how to access the necessary feature set of
the v4 receiver, so no library replacement or compilation is required.

## Windows Notes

On Windows in particular, the device must be configured within the operating system to be accessible using the
WinUSB driver instead of any native device drivers in order to be accessible to `librtlsdr`.  This is typically
achieved by running the [Zadig](https://zadig.akeo.ie/) utility.

After launching Zadig, select Options | List all devices.  Then from the drop-down list of devices, select 
"Bulk-In, Interface (Interface 0)."  You'll likely see the Driver reported as `(NONE)` with `WinUSB` selectable as
the replacement value.  Keep this default and click `Install Driver`.  This should make your RTL-SDR device 
available to `librtlsdr`.

Once this step has been completed, you should be able to follow the remaining setup instructions found 
in the [Getting Started](../tutorials/getting-started.md) and [Calibration](../tutorials/calibration.md) guides.

## Caveats

The RTL-2832 analog-to-digital converter found in the RTL-SDR.com receiver is a somewhat limited 8-bit ADC.
An 8-bit ADC has a maximum dynamic range of only around 48 dB, and also suffers from considerable quantization
noise.  Consequently, some effort must be spent in finding a "sweet spot" where RF amplification is sufficient to
match the device's own intrinsic noise level with the noise seen by the antenna, while still leaving enough room
for the powerline noise to be measurable without clipping/saturation rendering the measurements invalid.

This is what the `Auto-calibrate gain` option in the setup tool is attempting to achieve, but in the presence of
extremely strong powerline noise, it may not always be possible.  A very strong signal could completely saturate
the ADC and the measurement will flatten when charted (because measurement above the maximum level the ADC can
report is not possible - everything above the maximum value just becomes the maximum value).

Should you find yourself in this predicament, one thing you can try is switching to a higher band.  Powerline noise
amplitude tends to have an inverse relationship with frequency, so if, for example, the ADC is unavoidably saturated
on 80m, you may have an easier time on 60m or 40m.  Note that this also implies you cannot directly compare 
measurements taken on one band with measurements taken on another band.  In other words, the actual signal level
you see on 80m might be -67 dBm, but on 40m it might be only -72 dBm, but with a similar noise floor.

If you were to compare charts or CSV files from before and after switching bands, it might appear as though the 
amplitude of the noise changed, but what actually changed was just your switching bands and consequently seeing
less powerline noise.

In practical terms, so long as you bear in mind that measurements are relative, this is of no important consequence.
The real value in the data logging and charting is not to precisely measure decibel levels of received noise, but
rather to track trends of noise occurrence and to know when you might want to go out direction finding.

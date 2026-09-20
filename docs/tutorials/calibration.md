# Radio setup and calibration

## Introduction

The CSV files and charts produced by the monitor, and the bar graph display in the main window, all want to display
the amplitude signals received at your station in terms of dBm.  Signals reaching the analyzer have been demodulated
to an audio pass-band, either by your radio in sound-card mode or by baseband conversion in SDR mode,
and thus some calibration is required to report RF signal levels accurately rather than AF signal levels.

## Sound Card Calibration

Since the monitor is not reading the demodulated signal from your radio directly, but rather the audio-amplified
version of that signal, it is necessary to calibrate the level of the audio signal so that it can be directly
correlated with the amplitude of the actual signal.

Also, because the application is sampling in 16 bits, audio which exactly matched the signal level would lose
information below about -98dB, the lower limit of 16-bit audio read from a sound device.  Because our radios can be
sensitive to much lower levels, there's also value in exploiting this difference.

By default, the monitor is configured for a 32dB conversion offset between the two; thus, a -98dB audio reading on
the sound card corresponds to -130dBm on the radio.

If you're using a sound card with a radio, proceed with this section.  If, however, you're using an SDR receiver,
skip to the next section, "SDR Calibration".

### Method 1: AF Gain

If you can control the AF Gain from your radio to the monitor program, either by making an adjustment on the radio,
or by adjusting the gain of the input device in your operating system, this is the simpler approach.
[Start up the configuration tool](getting-started.md#run-setup), enter the Audio configuration, and select
"Calibration" at the bottom of the list.

This opens a live S meter relating what the monitor would currently see as the signal strength in dBm based on the
current audio amplitude and the active offset (default -32dB).  Simply adjust the audio level using either your radio
controls or your operating system gain controls until the bar graph in the monitor agrees with the S meter on your
radio.  Make a note of your settings.  That's it!

### Method 2: Offset adjustment

If you *cannot* control the AF gain, such as may be the case with rigs that have built-in sound devices without gain
adjustments, which also can't have their gain adjusted in the operating system, you will need to calibrate the offset
instead.

[Launch the configuration tool](getting-started.md#run-setup), navigate to the Station settings, and choose
"Audio-to-RF offset." Here you can use your up and down arrows on your keyboard to adjust the offset until the
S meter on your screen agrees with the S meter on your radio.  Press `ENTER` when you're done.

## SDR Calibration

SDR use requires calibrating two settings: hardware gain and level calibration.  Hardware gain is applied by the
receiver itself to the incoming RF signal, before sampling, and a value must be carefully chosen to minimize noise 
introduced by the hardware itself, while also allowing for maximum dynamic range so that capturing
strong signals does not overload the ADC circuitry and cause digital clipping.

It is also necessary to determine an appropriate offset to convert between the amplitude of the signals to which gain
was applied and the approximate amplitude of the RF signal seen at the receiver input, in order to correctly estimate
signal levels for logging and the on-screen S meters.

### RTL-SDR notes

Unfortunately, inexpensive 8-bit analog-to-digital converters such as those found in RTL-SDR devices do not offer
tremendous dynamic range - only about 48 dB - so any measurements taken are by necessity a compromise of some kind.

By making gain adjustments, we can decide where in the RF envelope that ~48dB of dynamic range sits.  Below the
envelope nothing can be known, and above it, everything becomes digitally clipped.

See the [RTL-SDR setup](../how-to-guides/rtl-sdr-v4-setup.md) notes for more details and notes about the RTL-SDR v4.

### SDRPlay notes

The 14-bit ADC in the SDRPlay, combined with the receiver's internal oversampling, in contrast to the RTL-SDR
provide a comfortably low noise floor and wide dynamic range.

See the [SDRPlay setup](../how-to-guides/sdrplay-rsp1-setup.md) notes for more details and notes about the SDRPlay
RSP1 series of receivers.

### Calibration Steps

The monitor defaults to using receiver number 0.  For most users this should be fine. 
If you are using multiple receivers, ensure the correct receiver number is selected first, before proceeding.

The first step to take is to run the `Auto-calibrate gain` tool with your antenna connected to your SDR device.
This will measure the response of the device to each of its supported gain values multiple times in an effort to
find the gain setting at which the noise floor contribution of the SDR device itself comes closest to matching 
the contribution from the antenna.  This permits getting a reasonably good measurement of the
true value of the noise floor while allowing for the maximum dynamic range for any received arc noise.

After applying gain in hardware, we want to reverse that operation to convert the decoded audio level back to
an approximate RF level for reporting and for display on a virtual S meter.  This is the `Level calibration` setting,
and after running `Auto-calibrate gain`, it defaults to the negative of the gain setting.  If you have an actual
reference to use for comparison, this value can be adjusted to match, as experience has shown these inexpensive
USB devices don't always apply the gain exactly as-advertised; there is some non-linearity in the gain adjustment.
The automatic value here should still be a good starting value.

Should excessive digital clipping occur during monitoring, this will be logged in warning messages to the monitor's 
output.  In that case, you can choose a different, lower `Tuner gain` setting.  Doing so will also shift the
`Level calibration` value by the difference, because the two need to be adjusted together.

If you consistently get values from `Auto-calibrate gain` which result in excessive clipping, you can also consider
adjusting the `arc_headroom_db` setting in your `config.toml` file.  The default value can be found in
`config.example.toml`.  Increasing it has the general effect of the auto calibration picking a lower `Tuner gain`
value.

If you're completely unable to set an adequate minimal gain while avoiding clipping, you may want to choose a
different band for monitoring.  The amplitude of powerline noise generally decreases with increasing frequency,
so if, for example, the signal is just too strong on the 80m band, try 60m or 40m instead, while still trying
to choose a frequency where your antenna is resonant, so you're making a fair and valid comparison between the
band's noise level and the powerline noise.

If you do change bands, be sure to re-run `Auto-calibrate gain`, because the ideal gain can depend on the band.

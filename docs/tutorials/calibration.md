# Radio setup and calibration

## Introduction

The CSV files and charts produced by the monitor, and the bar graph display in the main window, all want to display the amplitude signals received at your station in terms of dBm.  Since the monitor is not reading the demodulated signal from your radio directly, but rather the audio-amplified version of that signal, it is necessary to calibrate the level of the audio signal so that it can be directly correlated with the amplitude of the actual signal.

Also, because the application is sampling in 16 bits, audio which exactly matched the signal level would lose information below about -98dB, the lower limit of 16-bit audio read from a sound device.  Because our radios can be sensitive to much lower levels, there's also value in exploiting this difference.

By default, the monitor is configured for a 32dB conversion offset between the two; thus, a -98dB audio reading on the sound card corresponds to -130dBm on the radio.

## Method 1: AF Gain

If you can control the AF Gain from your radio to the monitor program, either by making an adjustment on the radio, or by adjusting the gain of the input device in your operating system, this is the simpler approach.  [Start up the configuration tool](getting-started.md#run-setup), enter the Audio configuration, and select "Calibration" at the bottom of the list.

This opens a live S meter relating what the monitor would currently see as the signal strength in dBm based on the current audio amplitude and the active offset (default -32dB).  Simply adjust the audio level using either your radio controls or your operating system gain controls until the bar graph in the monitor agrees with the S meter on your radio.  Make a note of your settings.  That's it!

## Method 2: Offset adjustment

If you *cannot* control the AF gain, such as may be the case with rigs that have built-in sound devices without gain adjustments, which also can't have their gain adjusted in the operating system, you will need to calibrate the offset instead.

[Launch the configuration tool](getting-started.md#run-setup), navigate to the Station settings, and choose "Audio-to-RF offset." Here you can use your up and down arrows on your keyboard to adjust the offset until the S meter on your screen agrees with the S meter on your radio.  Press `ENTER` when you're done.

# Record events

## Introduction

The N6OL Powerline QRM Monitor can automatically record to .wav files when it detects the onset of powerline QRM.  There are a number of configuration options to help you avoid recording short, transient or minimally disruptive events and to limit how many events you record and how much of each event to record to avoid blowing up your hard drive.

Important note: when limiting the duration of recordings, expect that the actual recording duration is almost always somewhat longer than the duration you specify.  This is because at the start of the recording, everything that already existed in the buffer prior to the start of the event is included in the recording in addition to the event itself.

## Setup Options

Launch the setup program either by running setup.bat or ./setup.sh, and navigate to "Event recording."  All of the recording options are listed there whether or not recording starts armed.

If you'd like the monitor to start with recording already armed, select "Arm recording at startup," use the arrow keys to move to "On," press `SPACE` or `ENTER` to select it, then use `TAB` to select "OK" and `ENTER`.  You don't have to turn this on to record events; the Record button, the `R` key, and the `--enable-recording` flag all arm a monitor that started up disarmed.

### Recording Directory

By default, the monitor will use a `recordings` directory under your station's logging/charting output path, but you can specify a different directory here if you'd like.

### Budgeting Options

To avoid recording every event that ever happens, you can configure a recording "budget" which records a fixed maximum number of recordings over a fixed period of time.

#### Events to Record

This works with the next option, "Budget reset (minutes)" to limit the number of recordings the monitor will make over the duration specified.  Note that the budget also resets if you restart the monitor; this check is performed entirely in-memory.  The actual number of files on disk is not checked.

#### Budget reset (minutes)

Starting at the time you start up the monitor, your "Events to Record" budget will reset every time this many minutes passes.

#### Examples

If you wanted to record the first 10 events every hour, you could set "Events to record" to 10, and "Budget reset" to 60.  Then every time 60 minutes passes, the number of remaining events to record will reset to 10.  Or if you wanted 5 events per day, you can set "Events to record" to 5 and "Budget reset" to 1440.

### Longest recording

This specifies the maximum number of seconds of an event that will be recorded, with the caveat that when a recording begins, the entire contents of the buffer are included in the file, so much of the time, the actual duration of the recording can be expected to be somewhat longer than the value specified here.

### Stop after silence

Should you wish to record a few seconds of audio after the end of an event (if it stops within the duration specified by "Longest Recording"), you can specify that duration here. You should probably keep this somewhat short - 1-10 seconds is probably reasonable.

### Minimum lock before recording

To avoid recording very short, transient events, you may want to specify a minimum duration for the monitor to hold a lock prior to the recording starting.  If you keep this to a value less than the length of the ring buffer (9.6 seconds) the monitor should always still record the beginning of the event even after the minimum lock duration has passed.

### Minimum SNR before recording

Should you wish to avoid recording very quiet events, here you can specify a minimum amplitude over the noise floor estimate that an event must reach before it is automatically recorded.

## Arming

Should you exhaust your record budget and wish to record further, you can also click the "Arm" button in the UI to manually re-enable automatic recording.

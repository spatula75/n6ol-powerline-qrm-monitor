"""Physical and domain constants shared by more than one module.

Each of these was previously defined independently in two or three places, which is
how device_setup.py's level bar drifted to 4.75 dB/segment while every S-meter in the
program stayed at 6: nothing tied the two together, so one could move without the
other noticing.  A value belongs here once a second module needs it too, not before -
see CLAUDE.md's note on this file for the rule.  buzz.dsp.SILENCE_DBFS and
buzz.dsp.PULSE_WIDTH_SAMPLES are already exactly this kind of shared constant and stay
in dsp.py rather than moving here, since dsp is itself imported everywhere that needs
them and a re-export would only add a second name for the same thing.
"""

# 16-bit signed full scale, i.e. 2**15.  The 0 dBFS reference for every amplitude-to-dB
# conversion in the program: dsp.DB_REFERENCE, scope.py's vertical scale, and
# device_setup.py's level bar all measure against this.  Not the same thing as int16's
# actual positive rail (32767, one less) - that is a hardware/format limit, this is a
# calibration reference point, and the two are related but not interchangeable.
FULL_SCALE_COUNTS = 32768.0

# IARU HF standard: S9 is exactly -73 dBm at the receiver input.
S9_DBM = -73.0

# The ham radio S-meter convention: 6 dB separates one S-unit from the next, S1 through
# S9.  Not negotiable - it is what "S-unit" means - so anything that draws an S-meter,
# or anything meant to read the same way as one, derives its scale from this rather
# than picking its own number of dB and calling the result S-units.  IARU's convention
# changes above S9: S9+10, S9+20 and so on are 10 dB apart, a different spacing kept
# local to waterfall.py rather than generalised here, since nothing else needs it.
DB_PER_S_UNIT = 6.0

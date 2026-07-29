import os

# Run Numba-decorated functions as plain Python under test.
#
# @njit compiles a function to machine code, so its body never executes Python
# bytecode and coverage.py cannot see it — the lines report as unhit even when the
# tests exercise them thoroughly.  Without this, every JIT'd function silently
# subtracts from the coverage gate and, worse, looks untested when it is not:
# dsp.average_pulse_amplitude sat at "82% covered, lines 115-124 missing" for exactly
# this reason, which is indistinguishable at a glance from nobody having tested it.
#
# Must be set before numba is first imported, which is why it is at the top of
# conftest.py rather than anywhere else — pytest loads this file before any test
# module, and therefore before anything imports buzz.dsp or buzz.scope.
#
# setdefault rather than a plain assignment, so the compiled path stays testable:
#
#     NUMBA_DISABLE_JIT=0 pytest tests/
#
# runs the same suite against the machine code production actually executes.  That
# matters more than it sounds.  The two paths are not automatically equivalent — the
# interpreter applies NumPy's NEP 50 promotion rules to mixed float32/float64
# arithmetic while Numba types the expression itself, and average_pulse_amplitude
# silently returned different answers under the two until an explicit cast was added
# (see its accumulator loop).  Coverage runs interpreted; correctness deserves both.
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'lib'))
sys.path.insert(0, str(_ROOT))

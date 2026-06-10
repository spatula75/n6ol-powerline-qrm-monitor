import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'lib'))
sys.path.insert(0, str(_ROOT))

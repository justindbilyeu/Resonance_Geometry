"""Documentation shim for :mod:`src.analysis.ringing_threshold`.

The canonical implementation now lives in ``src.analysis`` so that
both the library code and the published documentation can stay in
sync.  Downstream notebooks that previously imported from this docs
path can keep working by relying on this re-export layer.
"""

from src.analysis.ringing_threshold import *  # noqa: F401,F403

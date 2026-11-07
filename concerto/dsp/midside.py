"""Mid-Side decomposition.

M carries common information; S carries differential information.
In the Milestone-1 baseline these are analysis features; S additionally feeds
the (gated, band-limited, decorrelated) surround paths.
"""

from __future__ import annotations

import numpy as np


def mid_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, S) as float64 arrays: M=(L+R)/2, S=(L-R)/2."""
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    m = 0.5 * (l + r)
    s = 0.5 * (l - r)
    return m, s

"""First-order all-pass decorrelation banks.

Section:  y[n] = a*x[n] + x[n-1] - a*y[n-1],  |a| < 1.

Cascaded sections with independent coefficient sets per output path. Filter
state persists across the entire signal — never reset at analysis-window
boundaries. All-pass sections change phase only; magnitude response is
unity (within float roundoff).
"""

from __future__ import annotations

import numpy as np


def _allpass_1d(x: np.ndarray, a: float, state: tuple[float, float]) -> tuple[np.ndarray, tuple[float, float]]:
    """Process one channel through one first-order all-pass section.

    `state` is (x_prev, y_prev). Returns (y, new_state). Accumulates in
    float64 for numerical stability; caller casts back to float32.
    """
    xd = np.asarray(x, dtype=np.float64)
    x_prev, y_prev = state
    y = np.empty_like(xd)
    for i in range(xd.shape[0]):
        xi = xd[i]
        yi = a * xi + x_prev - a * y_prev
        y[i] = yi
        x_prev = xi
        y_prev = yi
    return y, (x_prev, y_prev)


class AllPassBank:
    """Cascaded first-order all-pass sections for one output path."""

    def __init__(self, coefficients: list[float]) -> None:
        if not coefficients:
            raise ValueError("at least one all-pass section is required")
        for a in coefficients:
            if not -1.0 < a < 1.0:
                raise ValueError(f"unstable all-pass coefficient a={a} (|a| must be < 1)")
        self.coefficients = list(coefficients)
        # One (x_prev, y_prev) state pair per section; persists for lifetime.
        self._states: list[tuple[float, float]] = [(0.0, 0.0)] * len(coefficients)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        for i, a in enumerate(self.coefficients):
            y, self._states[i] = _allpass_1d(y, a, self._states[i])
        return y.astype(np.float32)

    def reset(self) -> None:
        self._states = [(0.0, 0.0)] * len(self.coefficients)

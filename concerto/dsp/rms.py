"""Smooth band-limiting filters and RMS-based surround energy control.

Band-limiting uses Butterworth SOS sections (smooth slopes, not brick-wall).
The gain controller compares windowed surround RMS against stereo RMS and
steers a slow one-pole gain envelope toward a target ratio. It is explicitly
NOT a fast compressor: time constants are on the order of seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

EPS = 1e-12


def bandlimit_sos(sample_rate: int, highpass_hz: float, lowpass_hz: float, order: int = 4):
    """Return (sos_hp, sos_lp, zi_hp, zi_lp) with steady-state initial conditions."""
    sos_hp = butter(order, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
    sos_lp = butter(order, lowpass_hz, btype="lowpass", fs=sample_rate, output="sos")
    zi_hp = sosfilt_zi(sos_hp) * 0.0
    zi_lp = sosfilt_zi(sos_lp) * 0.0
    return sos_hp, sos_lp, zi_hp, zi_lp


def bandlimit(
    x: np.ndarray,
    sos: tuple,
    apply: bool = True,
) -> tuple[np.ndarray, tuple]:
    """Apply band-limiting to x with persistent state. Returns (y, new_sos)."""
    if not apply:
        return np.asarray(x, dtype=np.float32), sos
    sos_hp, sos_lp, zi_hp, zi_lp = sos
    y, zi_hp2 = sosfilt(sos_hp, x.astype(np.float64), zi=zi_hp, axis=0)
    y, zi_lp2 = sosfilt(sos_lp, y, zi=zi_lp, axis=0)
    return y.astype(np.float32), (sos_hp, sos_lp, zi_hp2, zi_lp2)


def rms(x: np.ndarray) -> float:
    """Windowed RMS accumulated in float64."""
    xd = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(xd * xd) + EPS)) if xd.size else 0.0


class GainController:
    """Slow one-pole gain envelope driving surround RMS toward a target ratio.

    p_t = alpha*p_target + (1-alpha)*p_{t-1},  alpha = 1 - exp(-dt/tau).
    """

    def __init__(
        self,
        sample_rate: int,
        window_s: float = 0.4,
        tau_s: float = 2.0,
        target_ratio: float = 0.35,
        gain_min: float = 0.0,
        gain_max: float = 2.0,
    ) -> None:
        self.window = max(1, int(sample_rate * window_s))
        self.alpha = 1.0 - np.exp(-self.window / max(tau_s * sample_rate, 1e-9))
        self.target_ratio = target_ratio
        self.gain_min = gain_min
        self.gain_max = gain_max
        self._gain = 1.0
        self._env = 0.0  # RMS envelope of reference

    def process(self, surround: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Scale `surround` chunk so its RMS tracks target_ratio * reference RMS."""
        ref_rms = rms(reference)
        # One-pole smoothing of the reference envelope (no pumping).
        self._env = self.alpha * ref_rms + (1.0 - self.alpha) * self._env
        sur_rms = rms(surround)
        desired = self.target_ratio * self._env
        target_gain = desired / (sur_rms + EPS) if sur_rms > 0 else self._gain
        target_gain = float(np.clip(target_gain, self.gain_min, self.gain_max))
        self._gain = self.alpha * target_gain + (1.0 - self.alpha) * self._gain
        return (np.asarray(surround, dtype=np.float64) * self._gain).astype(np.float32)

    @property
    def gain(self) -> float:
        return self._gain

"""Vocal centre rendering (AI-gated, Milestone 4).

The vocal centre is applied as a post-pass on the rendered 7.1 buffer: the
native fast path and the front-preservation guarantees are untouched, and the
feature degrades gracefully (centre gate stays closed when V is low).

Policy hook (SPEC.md section 6): "High vocal-centre confidence permits
controlled centre extraction" — the gate IS that rule, smoothed and bounded.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..audio.buffer import MultichannelBuffer
from ..audio.layouts import INTERNAL_7_1


def apply_vocal_centre(buf: MultichannelBuffer,
                       vocals_l: np.ndarray, vocals_r: np.ndarray,
                       v_schedule: np.ndarray,
                       gain_db: float = -6.0,
                       tau_s: float = 0.25,
                       chunk: int = 16384,
                       progress: Callable[[int, int], None] | None = None
                       ) -> MultichannelBuffer:
    """Replace the (silent) FC channel with the gated vocal mid signal.

    FC = gate(t) * gain * (vocals_L + vocals_R) / 2, where gate is the
    smoothed per-chunk vocal-presence confidence V from the harness. All
    other channels are untouched; FL/FR remain bit-identical to the source.
    """
    n = buf.frame_count
    idx = {label: i for i, label in enumerate(INTERNAL_7_1.order)}
    fc = np.zeros(n, dtype=np.float32)
    vm = 0.5 * (np.asarray(vocals_l[:n], dtype=np.float64)
                + np.asarray(vocals_r[:n], dtype=np.float64))
    lin_gain = 10.0 ** (gain_db / 20.0)

    dt = chunk / buf.sample_rate
    alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))
    gate = 0.0
    bar_update = progress((n + chunk - 1) // chunk) if progress else None
    for start in range(0, n, chunk):
        ci = start // chunk
        if ci < len(v_schedule):
            target = float(np.clip(v_schedule[ci], 0.0, 1.0))
            gate = alpha * target + (1.0 - alpha) * gate
        end = min(start + chunk, n)
        fc[start:end] = (lin_gain * gate * vm[start:end]).astype(np.float32)
        if bar_update:
            bar_update()

    channels = list(buf.channels)
    channels[idx["FC"]] = fc
    return MultichannelBuffer(channels, buf.sample_rate, buf.layout_name)

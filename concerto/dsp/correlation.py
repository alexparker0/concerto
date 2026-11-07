"""STFT analysis and frequency-dependent correlation / coherence.

Analysis runs in float64. STFT: N=4096, hop=1024, Hann window, 75% overlap.

Two distinct measurements are produced per frequency bin, accumulated
energy-weighted over time:

- rho(f): signed inter-channel correlation, Re(P_LR)/sqrt(P_LL*P_RR + eps)
- gamma^2(f): magnitude-squared coherence, |P_LR|^2/(P_LL*P_RR + eps)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class AnalysisSettings:
    fft_size: int = 4096
    hop_size: int = 1024
    window: str = "hann"


@dataclass
class FrequencyReport:
    freqs: np.ndarray          # float64 [bins], Hz
    rho: np.ndarray            # float64 [bins], signed correlation
    coherence: np.ndarray      # float64 [bins], gamma^2
    mid_energy: np.ndarray     # float64 [bins]
    side_energy: np.ndarray    # float64 [bins]
    directness: np.ndarray     # float64 [bins], covariance directness d(f)


def stft(x: np.ndarray, settings: AnalysisSettings) -> np.ndarray:
    """Manual STFT -> complex64/128 array [bins, frames] with Hann window."""
    n = settings.fft_size
    hop = settings.hop_size
    window = np.hanning(n) if settings.window == "hann" else np.ones(n)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < n:
        x = np.pad(x, (0, n - x.shape[0]))
    n_frames = 1 + (x.shape[0] - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * window[None, :]
    return np.fft.rfft(frames, axis=1).T  # [bins, frames]


def correlation_report(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings | None = None,
) -> FrequencyReport:
    """Per-bin correlation, coherence, and mid/side energies."""
    settings = settings or AnalysisSettings()
    xl = stft(left, settings)
    xr = stft(right, settings)

    p_lr = np.mean(xl * np.conj(xr), axis=1)            # cross spectrum
    p_ll = np.mean(np.abs(xl) ** 2, axis=1)
    p_rr = np.mean(np.abs(xr) ** 2, axis=1)

    rho = np.real(p_lr) / (np.sqrt(p_ll * p_rr) + EPS)
    coh = np.abs(p_lr) ** 2 / (p_ll * p_rr + EPS)
    # Covariance-based directness: normalized dominant-eigenvalue separation of
    # the 2x2 stereo covariance matrix (frontier spec). ~1 for hard-panned or
    # dual-mono content, ~0 for diffuse/uncorrelated content.
    direct = np.sqrt((p_ll - p_rr) ** 2 + 4.0 * np.abs(p_lr) ** 2) / (p_ll + p_rr + EPS)

    xm = 0.5 * (xl + xr)
    xs = 0.5 * (xl - xr)
    mid_e = np.mean(np.abs(xm) ** 2, axis=1)
    side_e = np.mean(np.abs(xs) ** 2, axis=1)

    freqs = np.fft.rfftfreq(settings.fft_size, d=1.0 / sample_rate)
    return FrequencyReport(freqs, rho, coh, mid_e, side_e, direct)


def band_average(
    report: FrequencyReport,
    edges: tuple[float, ...] = (100.0, 1000.0, 8000.0, 16000.0),
) -> list[tuple[str, float, float]]:
    """Energy-weighted average rho and gamma^2 per frequency band."""
    out = []
    prev = 0.0
    for edge in edges:
        mask = (report.freqs >= prev) & (report.freqs < edge)
        prev = edge
        if not np.any(mask):
            out.append((f"{prev:.0f}Hz", 0.0, 0.0))
            continue
        w = report.mid_energy[mask] + report.side_energy[mask] + EPS
        rho = float(np.sum(w * report.rho[mask]) / np.sum(w))
        coh = float(np.sum(w * report.coherence[mask]) / np.sum(w))
        out.append((f"{prev / 1000:.1f}kHz" if prev >= 1000 else f"{prev:.0f}Hz", rho, coh))
    return out

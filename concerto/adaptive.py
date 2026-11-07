"""Adaptive parameter controllers (Milestone 3).

Deterministic, analysis-driven parameter schedules. No neural networks.

Signal-adaptive stereo_beta
---------------------------
The stereo-difference surround feed is SL = L - beta*R, SR = R - beta*L.
The cross-feed leakage beta trades directional fidelity (low beta: 1/beta
contrast, but surround correlates with the ipsilateral front) against
anti-phase cancellation safety and envelopment (high beta: near-mono
difference feed for diffuse content).

The per-block beta target is steered by the covariance directness metric
d(f) of the L/R pair, weighted by side dominance w(f) = max(0, side_e -
mid_e) so that centre-dominant (dual-mono) content does not mask the
measurement of the differential content:

    d_weighted = sum_f w(f)*d(f) / sum_f w(f)      (0 = diffuse, 1 = direct)
    beta_target = beta_max - (beta_max - beta_min) * d_weighted

Beta transitions are exponentially smoothed per analysis update:

    p_t = alpha*p_target + (1-alpha)*p_{t-1},  alpha = 1 - exp(-dt/tau)

Bounds (from the measured drawbacks):
    beta_min = 0.15  directional floor (~6.7x ipsilateral contrast)
    beta_max = 0.90  near-mono ceiling for diffuse ambience

Blocks with no side energy (dual mono) steer to beta_max: the surrounds
rightly carry almost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dsp.midside import mid_side
from .dsp.upmix import CHUNK

EPS = 1e-12


@dataclass
class AdaptiveBetaConfig:
    enabled: bool = True
    beta_min: float = 0.15
    beta_max: float = 0.90
    tau_s: float = 1.5
    fft_size: int = 4096
    hop_size: int = 1024


def block_directness(left: np.ndarray, right: np.ndarray, sample_rate: int,
                     cfg: AdaptiveBetaConfig) -> float:
    """Side-dominance-weighted directness of one analysis block (float)."""
    n = cfg.fft_size
    hop = cfg.hop_size
    window = np.hanning(n)
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    if l.shape[0] < n:
        l = np.pad(l, (0, n - l.shape[0]))
        r = np.pad(r, (0, n - r.shape[0]))
    n_frames = 1 + (l.shape[0] - n) // hop
    if n_frames <= 0:
        return 0.0

    bins = n // 2 + 1
    # Time-averaged power/cross-spectra (single-frame covariances are always
    # "fully correlated" — directness is only meaningful after averaging).
    p_ll = np.zeros(bins)
    p_rr = np.zeros(bins)
    p_lr = np.zeros(bins, dtype=complex)
    side_e = np.zeros(bins)
    idx = np.arange(n)[None, :] + hop * np.arange(n_frames)[:, None]
    for f in range(n_frames):
        off = f * hop
        xl = np.fft.rfft(l[off:off + n] * window)
        xr = np.fft.rfft(r[off:off + n] * window)
        xs = np.fft.rfft(0.5 * (l[off:off + n] - r[off:off + n]) * window)
        p_ll += np.abs(xl) ** 2
        p_rr += np.abs(xr) ** 2
        p_lr += xl * np.conj(xr)
        side_e += np.abs(xs) ** 2
    p_ll /= n_frames
    p_rr /= n_frames
    p_lr /= n_frames
    side_e /= n_frames

    # Per-bin directness of the stereo pair (time-averaged).
    d = np.sqrt((p_ll - p_rr) ** 2 + 4.0 * np.abs(p_lr) ** 2) / (p_ll + p_rr + EPS)
    # Finite-sample floor: with n_eff effectively independent frames, pure
    # noise still yields E[d] ~ 1/sqrt(n_eff). Remove the floor and rescale
    # so diffuse content maps to ~0 and fully directional to ~1.
    n_eff = max(n_frames / 4.0, 1.0)  # 75% overlap -> ~4 frames per window
    floor = 1.0 / np.sqrt(n_eff)
    denom = max(1.0 - floor, 1e-6)
    d = np.clip((d - floor) / denom, 0.0, 1.0)
    # Weight by side energy: only bins where differential content lives.
    # (Dual mono has side_e ~ 0 -> no steering evidence, as intended.)
    den = float(np.sum(side_e))
    if den <= EPS:
        return 0.0
    return float(np.sum(side_e * d) / den)


def compute_beta_schedule(left: np.ndarray, right: np.ndarray, sample_rate: int,
                          cfg: AdaptiveBetaConfig, chunk: int = CHUNK
                          ) -> np.ndarray:
    """Per-chunk smoothed beta schedule for the full track (float64)."""
    n = left.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    dt = chunk / sample_rate
    alpha = 1.0 - np.exp(-dt / max(cfg.tau_s, 1e-6))

    schedule = np.empty(n_chunks, dtype=np.float64)
    beta = 0.5 * (cfg.beta_min + cfg.beta_max)
    for i in range(n_chunks):
        lo, hi = i * chunk, min((i + 1) * chunk, n)
        d = block_directness(left[lo:hi], right[lo:hi], sample_rate, cfg)
        target = cfg.beta_max - (cfg.beta_max - cfg.beta_min) * d
        beta = alpha * target + (1.0 - alpha) * beta
        schedule[i] = beta
    return np.clip(schedule, cfg.beta_min, cfg.beta_max)


# ===========================================================================
# Milestone 3: deterministic rule-based controller
#
# Feature vector per analysis block (SPEC.md section 6):
#   rho       signed stereo correlation (band-averaged, energy-weighted)
#   gamma2    mean magnitude-squared coherence
#   W         stereo width (side/mid RMS ratio, compressed to 0..1)
#   D         diffuse-energy proportion (1 - side-weighted directness)
#   T         transient density (spectral-flux peak rate, 0..1)
#   A         ambience estimate (energy fraction with low coherence)
#   V         vocal-presence confidence -- NOT estimated without AI
#             (Milestone 4); centre contribution therefore stays disabled.
#
# Rule policy (conservative, from the spec's starting table):
#   Strong transients      -> reduce decorrelation intensity (fast timescale)
#   High existing ambience -> do not add more decorrelated energy
#   Low ambience           -> modest surround gain lift (add envelopment)
#   High directness        -> protect front image (modest gain reduction)
#
# All policy outputs are multipliers on the configured base parameters,
# clamped, and smoothed: slow tau for broad spatial parameters, fast tau
# (transient_tau_s) for transient protection.
# ===========================================================================


@dataclass
class FeatureConfig:
    fft_size: int = 4096
    hop_size: int = 1024
    # Transient detection: spectral flux normalised by block median.
    transient_threshold: float = 2.5
    # Coherence threshold below which energy counts as "ambient".
    ambience_coherence_max: float = 0.5


@dataclass
class BlockFeatures:
    rho: float
    gamma2: float
    width: float
    diffuse: float
    transients: float
    ambience: float
    directness: float = 0.0


def extract_block_features(left: np.ndarray, right: np.ndarray,
                           sample_rate: int, cfg: FeatureConfig
                           ) -> BlockFeatures:
    """Compute the normalised feature vector for one analysis block."""
    n = cfg.fft_size
    hop = cfg.hop_size
    window = np.hanning(n)
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    if l.shape[0] < n:
        l = np.pad(l, (0, n - l.shape[0]))
        r = np.pad(r, (0, n - r.shape[0]))
    n_frames = 1 + (l.shape[0] - n) // hop
    if n_frames <= 0:
        return BlockFeatures(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)

    bins = n // 2 + 1
    p_ll = np.zeros(bins)
    p_rr = np.zeros(bins)
    p_lr = np.zeros(bins, dtype=complex)
    mid_e = np.zeros(bins)
    side_e = np.zeros(bins)
    frame_energy = np.zeros(n_frames)

    for f in range(n_frames):
        off = f * hop
        xl = np.fft.rfft(l[off:off + n] * window)
        xr = np.fft.rfft(r[off:off + n] * window)
        p_ll += np.abs(xl) ** 2
        p_rr += np.abs(xr) ** 2
        p_lr += xl * np.conj(xr)
        xm = 0.5 * (xl + xr)
        xs = 0.5 * (xl - xr)
        mid_e += np.abs(xm) ** 2
        side_e += np.abs(xs) ** 2
        frame_energy[f] = float(np.sum(l[off:off + n] ** 2 + r[off:off + n] ** 2))
    p_ll /= n_frames
    p_rr /= n_frames
    p_lr /= n_frames
    mid_e /= n_frames
    side_e /= n_frames

    # rho and gamma2, energy-weighted over bins. Bins where one channel's
    # energy is negligible (hard-panned content) are directional by
    # convention: gamma2 = 1, rho = +1. Without this, a silent channel makes
    # 0/eps read as "fully incoherent" and corrupts the ambience estimate.
    w = p_ll + p_rr + EPS
    rel = np.minimum(p_ll, p_rr) / (np.maximum(p_ll, p_rr) + EPS)
    balanced = rel > 1e-3
    gamma_bins = np.where(balanced,
                          np.abs(p_lr) ** 2 / (p_ll * p_rr + EPS), 1.0)
    rho_bins = np.where(balanced,
                        np.real(p_lr) / (np.sqrt(p_ll * p_rr) + EPS), 1.0)
    rho = float(np.sum(w * rho_bins) / np.sum(w))
    gamma2 = float(np.sum(w * gamma_bins) / np.sum(w))

    # Width: side/mid RMS ratio, log-compressed to 0..1 (1 = side == mid).
    m_rms = float(np.sqrt(np.sum(mid_e) + EPS))
    s_rms = float(np.sqrt(np.sum(side_e) + EPS))
    width = float(np.clip(np.log1p(s_rms / (m_rms + EPS)) /
                          np.log1p(1.0) / 2.0, 0.0, 1.0))

    # Diffuse proportion: 1 - side-weighted directness (finite-sample floor
    # corrected, as in block_directness).
    d = np.sqrt((p_ll - p_rr) ** 2 + 4.0 * np.abs(p_lr) ** 2) / (p_ll + p_rr + EPS)
    n_eff = max(n_frames / 4.0, 1.0)
    floor = 1.0 / np.sqrt(n_eff)
    denom = max(1.0 - floor, 1e-6)
    d = np.clip((d - floor) / denom, 0.0, 1.0)
    sd = float(np.sum(side_e * d) / (np.sum(side_e) + EPS))
    diffuse = float(np.clip(1.0 - sd, 0.0, 1.0))

    # Ambience: energy fraction in low-coherence bins.
    amb_w = w * (gamma_bins < cfg.ambience_coherence_max)
    ambience = float(np.clip(np.sum(amb_w) / np.sum(w), 0.0, 1.0))

    # Transient density: amplitude-envelope peaks relative to the block's
    # median envelope. (Spectral-flux thresholds collapse to ~0 for
    # stationary signals, where the flux median is numerical noise.)
    env = np.sqrt(frame_energy + EPS)
    base = float(np.median(env)) + EPS
    rising = env[1:] > env[:-1]
    peaks = int(np.sum((env[1:] > 1.8 * base) & rising))
    transients = float(np.clip(peaks / max(n_frames - 1, 1) / 0.25, 0.0, 1.0))

    return BlockFeatures(rho, gamma2, width, diffuse, transients, ambience, directness=sd)


@dataclass
class PolicyConfig:
    """Bounds and smoothing for the rule-based controller."""
    gain_min_mult: float = 0.6
    gain_max_mult: float = 1.4
    decorr_min: float = 0.25      # decorrelation intensity floor
    decorr_max: float = 1.0
    tau_s: float = 1.5            # slow timescale (broad spatial parameters)
    transient_tau_s: float = 0.05  # fast timescale (transient protection)
    transient_threshold: float = 0.5  # feature T above which protection kicks in


def policy_targets(f: BlockFeatures, p: PolicyConfig) -> tuple[float, float]:
    """Rule policy -> (gain multiplier, decorrelation intensity) targets.

    Deterministic rules from the spec's starting table; multipliers apply to
    the configured base surround gain. Decorrelation intensity is the mix
    between the raw band-limited feed (0) and the fully decorrelated output
    (1). Centre contribution is policy-disabled (V unavailable without AI).
    """
    gain_mult = 1.0
    decorr = 1.0

    # Strong transients: reduce decorrelation (avoid smearing attacks).
    if f.transients > p.transient_threshold:
        decorr *= 0.4
        gain_mult *= 0.9
    # High existing ambience: do not add more decorrelated energy.
    if f.ambience > 0.5:
        decorr *= 0.6
        gain_mult *= 0.9
    # Low ambience: modest lift to add envelopment.
    elif f.ambience < 0.2:
        gain_mult *= 1.15
    # High directness (discrete sources, hard-panned): protect front image.
    if f.diffuse < 0.25 and f.width < 0.3:
        gain_mult *= 0.9

    gain_mult = float(np.clip(gain_mult, p.gain_min_mult, p.gain_max_mult))
    decorr = float(np.clip(decorr, p.decorr_min, p.decorr_max))
    return gain_mult, decorr


def compute_adaptive_schedules(left: np.ndarray, right: np.ndarray,
                               sample_rate: int,
                               beta_cfg: AdaptiveBetaConfig,
                               policy_cfg: PolicyConfig,
                               feature_cfg: FeatureConfig | None = None,
                               chunk: int = CHUNK,
                               progress: Callable[[int, int], None] | None = None
                               ) -> dict[str, np.ndarray]:
    """Per-chunk schedules: beta, gain multiplier, decorrelation intensity.

    Broad parameters (beta, gain, decorr) smooth on the slow timescale. A
    separate fast envelope reduces decorrelation around transients and
    recovers on the transient_tau_s timescale (transient protection).
    """
    feature_cfg = feature_cfg or FeatureConfig()
    n = left.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    dt = chunk / sample_rate
    alpha_slow = 1.0 - np.exp(-dt / max(policy_cfg.tau_s, 1e-6))
    alpha_fast = 1.0 - np.exp(-dt / max(policy_cfg.transient_tau_s, 1e-6))
    alpha_beta = 1.0 - np.exp(-dt / max(beta_cfg.tau_s, 1e-6))

    beta_s = np.empty(n_chunks)
    gain_s = np.empty(n_chunks)
    decorr_s = np.empty(n_chunks)

    beta = 0.5 * (beta_cfg.beta_min + beta_cfg.beta_max)
    gain = 1.0
    decorr = 1.0
    protect = 0.0  # fast transient-protection envelope (1 = protected)

    bar_update = progress(n_chunks) if progress else None
    for i in range(n_chunks):
        lo, hi = i * chunk, min((i + 1) * chunk, n)
        feats = extract_block_features(left[lo:hi], right[lo:hi], sample_rate,
                                       feature_cfg)
        d = feats.directness
        beta_t = beta_cfg.beta_max - (beta_cfg.beta_max - beta_cfg.beta_min) * d
        beta = alpha_beta * beta_t + (1.0 - alpha_beta) * beta
        beta_s[i] = float(np.clip(beta, beta_cfg.beta_min, beta_cfg.beta_max))

        g_t, dc_t = policy_targets(feats, policy_cfg)
        gain = alpha_slow * g_t + (1.0 - alpha_slow) * gain
        decorr = alpha_slow * dc_t + (1.0 - alpha_slow) * decorr

        # Fast transient protection: dive toward minimum decorrelation and
        # recover quickly after the transient passes.
        protect_t = 1.0 if feats.transients > policy_cfg.transient_threshold else 0.0
        protect = alpha_fast * protect_t + (1.0 - alpha_fast) * protect
        decorr_eff = decorr * (1.0 - protect) + policy_cfg.decorr_min * protect

        gain_s[i] = float(np.clip(gain, policy_cfg.gain_min_mult, policy_cfg.gain_max_mult))
        decorr_s[i] = float(np.clip(decorr_eff, 0.0, policy_cfg.decorr_max))
        if bar_update:
            bar_update()

    return {"beta": beta_s, "gain": gain_s, "decorr": decorr_s}

def compute_adaptive_schedules_stream(reader: Any,
                                      sample_rate: int,
                                      beta_cfg: AdaptiveBetaConfig,
                                      policy_cfg: PolicyConfig,
                                      feature_cfg: FeatureConfig | None = None,
                                      progress: Callable[[int, int], None] | None = None
                                      ) -> dict[str, np.ndarray]:
    """Compute adaptive schedules by streaming blocks through reader.

    Runs in O(chunk_size) memory without buffering audio.
    Rewinds reader to frame 0 upon completion.
    """
    feature_cfg = feature_cfg or FeatureConfig()
    chunk = getattr(reader, "chunk_size", CHUNK)
    frame_count = getattr(reader, "frame_count", 0)
    n_chunks = (frame_count + chunk - 1) // chunk if frame_count > 0 else 1
    dt = chunk / sample_rate
    alpha_slow = 1.0 - np.exp(-dt / max(policy_cfg.tau_s, 1e-6))
    alpha_fast = 1.0 - np.exp(-dt / max(policy_cfg.transient_tau_s, 1e-6))
    alpha_beta = 1.0 - np.exp(-dt / max(beta_cfg.tau_s, 1e-6))

    beta_list = []
    gain_list = []
    decorr_list = []

    beta = 0.5 * (beta_cfg.beta_min + beta_cfg.beta_max)
    gain = 1.0
    decorr = 1.0
    protect = 0.0

    reader.seek(0)
    bar_update = progress(n_chunks) if progress else None
    for l_chunk, r_chunk in reader.blocks():
        feats = extract_block_features(l_chunk, r_chunk, sample_rate, feature_cfg)
        d = feats.directness
        beta_t = beta_cfg.beta_max - (beta_cfg.beta_max - beta_cfg.beta_min) * d
        beta = alpha_beta * beta_t + (1.0 - alpha_beta) * beta
        beta_list.append(float(np.clip(beta, beta_cfg.beta_min, beta_cfg.beta_max)))

        g_t, dc_t = policy_targets(feats, policy_cfg)
        gain = alpha_slow * g_t + (1.0 - alpha_slow) * gain
        decorr = alpha_slow * dc_t + (1.0 - alpha_slow) * decorr

        protect_t = 1.0 if feats.transients > policy_cfg.transient_threshold else 0.0
        protect = alpha_fast * protect_t + (1.0 - alpha_fast) * protect
        decorr_eff = decorr * (1.0 - protect) + policy_cfg.decorr_min * protect

        gain_list.append(float(np.clip(gain, policy_cfg.gain_min_mult, policy_cfg.gain_max_mult)))
        decorr_list.append(float(np.clip(decorr_eff, 0.0, policy_cfg.decorr_max)))
        if bar_update:
            bar_update()

    reader.seek(0)
    return {
        "beta": np.array(beta_list, dtype=np.float64),
        "gain": np.array(gain_list, dtype=np.float64),
        "decorr": np.array(decorr_list, dtype=np.float64),
    }

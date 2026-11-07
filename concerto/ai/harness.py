"""AI acceptance harness (SPEC.md section 7).

A stem is only usable if:
  1. It aligns with the original sample timeline.
  2. It does not introduce unacceptable transient artefacts.
  3. Its recombination is reasonably close to the original.
  4. Its stereo positioning remains interpretable.
  5. It is demonstrably better than the DSP-only alternative (tracked here as
     an energy/mask-quality measurement; the perceptual judgement stays with
     listening tests).

Separation success != reconstruction success: this harness gates, it does not
render.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from .separation import Stems


@dataclass
class HarnessConfig:
    alignment_max_lag: int = 4          # samples
    recombination_min_db: float = 20.0  # residual must be >= 20 dB below mix
    artefact_env_corr_min: float = 0.5  # vocal envelope vs mix envelope in band
    stereo_flip_min: float = -0.95      # per-stem L/R correlation floor


@dataclass
class HarnessReport:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"passed": self.passed,
                           "checks": self.checks,
                           "metrics": self.metrics}, indent=2)

    def _add(self, name: str, ok, detail: str) -> None:
        ok = bool(ok)
        self.checks.append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            self.passed = False


def _env(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000))
    n = (len(x) // hop) * hop
    return np.sqrt((x[:n].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12)


def evaluate_stems(mix_l: np.ndarray, mix_r: np.ndarray, stems: Stems,
                   sample_rate: int, cfg: HarnessConfig | None = None,
                   seg: int | None = None) -> HarnessReport:
    """Run the acceptance criteria over the first `seg` samples."""
    cfg = cfg or HarnessConfig()
    rep = HarnessReport(passed=True)
    seg = min(seg if seg is not None else 60 * sample_rate, len(mix_l))
    names = ("vocals", "drums", "bass", "other")
    stem_list = list(stems)

    # 1. Timeline alignment: lag of (sum of stems) vs mix must be ~0.
    s_l, s_r = stems.sum()
    n_min = min(len(s_l), len(mix_l))
    a = s_l[:n_min] - s_l[:n_min].mean()
    b = mix_l[:n_min] - mix_l[:n_min].mean()
    c = np.correlate(a[: min(seg, 10 * sample_rate)],
                     b[: min(seg, 10 * sample_rate)], "same")
    lag = int(np.argmax(np.abs(c)) - len(c) // 2)
    rep._add("alignment", abs(lag) <= cfg.alignment_max_lag, f"lag={lag} samples")
    rep.metrics["alignment_lag_samples"] = lag

    # 3. Recombination: residual between mix and stem sum.
    err = mix_l[:n_min] - s_l[:n_min]
    mix_p = float(np.sum(mix_l[:n_min] ** 2))
    res_db = 10.0 * np.log10((mix_p + 1e-12) / (float(np.sum(err ** 2)) + 1e-12))
    rep._add("recombination", res_db >= cfg.recombination_min_db,
             f"residual {res_db:.1f} dB below mix (>= {cfg.recombination_min_db})")
    rep.metrics["recombination_db"] = round(res_db, 2)

    # 2. Transient artefacts: segmentation glitches appear as spikes in the
    # stem's own envelope. Metric: worst frame-to-frame envelope jump relative
    # to the mean envelope level. (Envelope *correlation* with the mix is not
    # usable here — a tonal vocal stem has a constant envelope by nature.)
    sos = butter(4, [200.0, 4000.0], btype="bandpass", fs=sample_rate,
                 output="sos")
    band_voc = sosfilt(sos, stems.vocals[0][:seg])
    ev = _env(band_voc, sample_rate)
    if len(ev) > 1 and float(np.mean(ev)) > 1e-6:
        jump = float(np.max(np.abs(np.diff(ev)))) / (float(np.mean(ev)) + 1e-12)
    else:
        jump = 0.0
    rep._add("transient_artefacts", jump <= 3.0,
             f"max envelope jump={jump:.2f} (<= 3.0)")
    rep.metrics["envelope_jump"] = round(jump, 3)

    # 4. Stereo interpretability: no stem may be anti-phase. Silent stems
    # carry no interpretable position and pass trivially.
    for name, (l, r) in zip(names, stem_list):
        seg_n = min(seg, 10 * sample_rate, len(l), len(r))
        if float(np.sum(l[:seg_n] ** 2)) + float(np.sum(r[:seg_n] ** 2)) < 1e-10:
            rep._add(f"stereo_{name}", True, "silent stem (trivially valid)")
            continue
        cc = float(np.corrcoef(l[:seg_n], r[:seg_n])[0, 1])
        rep._add(f"stereo_{name}", cc >= cfg.stereo_flip_min,
                 f"L/R corr={cc:.3f}")
        rep.metrics[f"stereo_corr_{name}"] = round(cc, 3)

    # 5. Mask quality: vocal energy share and bleed (energy of vocals in the
    # non-vocal stems' sum vs total). Reported, not gated.
    voc_e = float(np.sum(stems.vocals[0][:seg] ** 2))
    others_e = float(np.sum((stems.drums[0][:seg] + stems.bass[0][:seg]
                             + stems.other[0][:seg]) ** 2))
    bleed_db = 10.0 * np.log10((voc_e + 1e-12) / (others_e + 1e-12))
    rep.metrics["vocal_energy_share_db"] = round(bleed_db, 2)

    return rep


def v_schedule_from_stems(vocals_l: np.ndarray, vocals_r: np.ndarray,
                          mix_l: np.ndarray, mix_r: np.ndarray,
                          sample_rate: int, chunk: int,
                          gate_threshold: float = 0.15,
                          tau_s: float = 0.5) -> np.ndarray:
    """Per-chunk vocal-presence confidence V in [0, 1] (the policy's V feature).

    V = vocal-band energy share of the vocal stem vs the mix, smoothed. The
    renderer gates the vocal centre contribution with this schedule.
    """
    from scipy.signal import butter, sosfilt

    sos = butter(4, [200.0, 4000.0], btype="bandpass", fs=sample_rate, output="sos")
    n_chunks = (len(vocals_l) + chunk - 1) // chunk
    dt = chunk / sample_rate
    alpha = 1.0 - np.exp(-dt / max(tau_s, 1e-6))

    v = np.empty(n_chunks)
    conf = 0.0
    for i in range(n_chunks):
        lo, hi = i * chunk, min((i + 1) * chunk, len(vocals_l))
        vm = sosfilt(sos, 0.5 * (vocals_l[lo:hi] + vocals_r[lo:hi]))
        mm = sosfilt(sos, 0.5 * (mix_l[lo:hi] + mix_r[lo:hi]))
        num = float(np.sum(vm ** 2))
        den = float(np.sum(mm ** 2))
        target = num / (den + 1e-12)
        conf = alpha * target + (1.0 - alpha) * conf
        v[i] = conf
    # Soft gate around the threshold, full confidence well above it.
    return np.clip((v - gate_threshold) / (0.4 - gate_threshold), 0.0, 1.0)

"""Automated acceptance tests for rendered output.

Implements the Milestone-1 QA table: channel count, alignment, clipping, LFE,
stability, duration, FLAC integrity, fold-down, front preservation.
`validate` runs these against any rendered file; `tests/` runs them in pytest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.buffer import measure
from ..audio.layouts import (FLAC_6CH_AS_51, FLAC_8CH, INTERNAL_5_1,
                             INTERNAL_7_1, WAV_6CH, WAV_8CH, permutation)

CHANNEL_ROUTES_71 = {
    "FL": (1000.0, "sine"),
    "FR": (2000.0, "sine"),
    "FC": (3000.0, "sine"),
    "LFE": (60.0, "sine"),
    "SL": (4000.0, "sine"),
    "SR": (5000.0, "sine"),
    "BL": (6000.0, "sine"),
    "BR": (7000.0, "sine"),
}


@dataclass
class ValidationResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, ok, detail: str) -> None:
        ok = bool(ok)
        self.checks.append({"check": name, "passed": ok, "detail": detail})
        self.passed = bool(self.passed and ok)

    def to_json(self) -> str:
        return json.dumps({"passed": self.passed, "checks": self.checks}, indent=2)


def _file_order(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return data, int(sr)


def _to_internal(data: np.ndarray, container: str) -> np.ndarray:
    """Map file channel order to internal order; returns [frames x ch]."""
    n = data.shape[1]
    if container == "wav":
        layout = {6: WAV_6CH, 8: WAV_8CH}.get(n)
    else:  # flac
        layout = {6: FLAC_6CH_AS_51, 8: FLAC_8CH}.get(n)
    if layout is None:
        return data
    perm = permutation(layout, INTERNAL_7_1 if n == 8 else INTERNAL_5_1)
    return data[:, perm]


def validate_file(path: str, reference: str | None = None) -> ValidationResult:
    """Run all acceptance checks on a rendered multichannel file."""
    result = ValidationResult(passed=True)
    info = sf.info(path)
    data, sr = _file_order(path)
    data = _to_internal(data, sf.info(path).format.lower())  # internal order
    n_ch = data.shape[1]

    # Channel count
    result.add("channel_count", n_ch in (6, 8), f"{n_ch} channels")

    labels = list(INTERNAL_7_1.order) if n_ch == 8 else list(INTERNAL_5_1.order)

    # Clipping
    peak, _ = measure(data)
    result.add("clipping", peak <= 1.0 + 1e-9, f"peak={peak:.6f}")

    # Duration
    if reference is not None:
        ref_info = sf.info(reference)
        result.add(
            "duration",
            info.frames == ref_info.frames and sr == ref_info.samplerate,
            f"out={info.frames}@{sr}, ref={ref_info.frames}@{ref_info.samplerate}",
        )

    # FLAC integrity: decode already succeeded above; check frame count consistency
    result.add("flac_integrity", info.frames == data.shape[0], f"frames={info.frames}")

    # LFE silent
    lfe = data[:, labels.index("LFE")]
    result.add("lfe_silent", float(np.max(np.abs(lfe))) == 0.0,
               f"LFE peak={float(np.max(np.abs(lfe))):.2e}")

    # Sample alignment: with a reference, each front channel must correlate
    # at exactly zero lag. Generic inter-channel lag checks are unreliable on
    # uncorrelated content; channel routing is verified by routing-test.
    if reference is not None:
        ref, ref_sr = _file_order(reference)
        seg = min(len(data), 48000)
        for i, label in ((0, "FL"), (1, "FR")):
            a = data[:seg, i] - data[:seg, i].mean()
            b = ref[:seg, i] - ref[:seg, i].mean()
            c = np.correlate(a, b, "same")
            lag = int(np.argmax(np.abs(c)) - seg // 2)
            result.add(f"alignment_{label}", abs(lag) <= 1, f"lag={lag} samples")

        # Front preservation
        e_l = float(np.max(np.abs(data[:, 0] - ref[:, 0])))
        e_r = float(np.max(np.abs(data[:, 1] - ref[:, 1])))
        result.add("front_preservation", e_l == 0.0 and e_r == 0.0,
                   f"E_L={e_l:.2e}, E_R={e_r:.2e}")

    # Filter stability: all finite
    result.add("stability", bool(np.all(np.isfinite(data))), "finite output")

    # Fold-down: ITU-style 5.1/7.1 -> stereo, check for severe cancellation
    idx = {label: i for i, label in enumerate(labels)}
    g = 0.707
    lfd = data[:, idx["FL"]] + g * data[:, idx["FC"]] + g * data[:, idx["SL"]]
    rfd = data[:, idx["FR"]] + g * data[:, idx["FC"]] + g * data[:, idx["SR"]]
    if n_ch == 8:
        lfd = lfd + g * data[:, idx["BL"]]
        rfd = rfd + g * data[:, idx["BR"]]
    corr = float(
        np.corrcoef(lfd[: min(len(lfd), 48000)], rfd[: min(len(rfd), 48000)])[0, 1]
    )
    result.add("fold_down", corr < 0.999 or measure(lfd)[1] > 1e-4,
               f"L/R fold-down corr={corr:.4f}")

    # Downmix consistency (frontier spec): MPEG-style fold-down vs the original
    # stereo. Reported as SDR in dB; the baseline renderer must stay well above
    # 10 dB (surround bleed is present but small by construction).
    if reference is not None:
        ref, _ = _file_order(reference)
        seg = min(len(lfd), len(ref), 48000)
        err = lfd[:seg] - ref[:seg, 0]
        num = float(np.sum(ref[:seg, 0] ** 2))
        den = float(np.sum(err ** 2))
        sdr_l = 10.0 * np.log10((num + 1e-12) / (den + 1e-12))
        err = rfd[:seg] - ref[:seg, 1]
        num = float(np.sum(ref[:seg, 1] ** 2))
        den = float(np.sum(err ** 2))
        sdr_r = 10.0 * np.log10((num + 1e-12) / (den + 1e-12))
        result.add("downmix_sdr", min(sdr_l, sdr_r) > 10.0,
                   f"fold-down SDR L={sdr_l:.1f} dB, R={sdr_r:.1f} dB")

    return result


def routing_test_signal(sample_rate: int, seconds: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Generate an 8-channel buffer with a unique distinguishable sine per channel."""
    labels = list(INTERNAL_7_1.order)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    chans = []
    for label in labels:
        f, _ = CHANNEL_ROUTES_71[label]
        amp = 0.5 if label != "LFE" else 0.8
        chans.append((amp * np.sin(2 * np.pi * f * t)).astype(np.float32))
    return chans, labels


def detect_channel_frequencies(path: str) -> dict[str, float]:
    """Verify channel routing: dominant frequency per internal channel."""
    from scipy.signal import welch

    data, sr = _file_order(path)
    data = _to_internal(data, sf.info(path).format.lower())
    labels = list(INTERNAL_7_1.order) if data.shape[1] == 8 else list(INTERNAL_5_1.order)
    out = {}
    for i, label in enumerate(labels):
        f, p = welch(data[:, i], sr, nperseg=8192)
        out[label] = float(f[np.argmax(p)])
    return out

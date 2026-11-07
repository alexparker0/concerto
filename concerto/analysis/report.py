"""Analysis reports: per-bin CSV + summary JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..audio.buffer import SourceMetadata
from ..dsp.correlation import AnalysisSettings, band_average, correlation_report
from ..dsp.midside import mid_side
from ..audio.buffer import measure


def write_reports(
    left: np.ndarray,
    right: np.ndarray,
    meta: SourceMetadata,
    out_dir: str | Path,
    settings: AnalysisSettings | None = None,
) -> tuple[Path, Path]:
    """Generate <stem>.csv (per-bin) and <stem>.json (summary) reports."""
    settings = settings or AnalysisSettings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(meta.path).stem

    report = correlation_report(left, right, meta.sample_rate, settings)

    csv_path = out_dir / f"{stem}_analysis.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "rho", "coherence", "mid_energy", "side_energy", "directness"])
        for f, r, c, m, s, d in zip(report.freqs, report.rho, report.coherence,
                                    report.mid_energy, report.side_energy, report.directness):
            w.writerow([f"{f:.2f}", f"{r:.6f}", f"{c:.6f}", f"{m:.8e}", f"{s:.8e}", f"{d:.6f}"])

    bands = band_average(report)
    m, s = mid_side(left, right)
    _, m_rms = measure(m)
    _, s_rms = measure(s)
    width = s_rms / (m_rms + 1e-12)  # simple width proxy: side/mid RMS ratio
    json_path = out_dir / f"{stem}_analysis.json"
    summary = {
        "source": {
            "path": meta.path,
            "format": meta.format,
            "subtype": meta.subtype,
            "sample_rate": meta.sample_rate,
            "frame_count": meta.frame_count,
            "duration_s": round(meta.frame_count / meta.sample_rate, 3),
            "channels": meta.channels,
            "peak": round(meta.peak, 6),
            "rms": round(meta.rms, 6),
        },
        "analysis": {
            "fft_size": settings.fft_size,
            "hop_size": settings.hop_size,
            "window": settings.window,
            "stereo_width": round(width, 6),
            "mid_rms": round(m_rms, 6),
            "side_rms": round(s_rms, 6),
            "bands": [
                {"band": b, "rho": round(r, 6), "coherence": round(c, 6)}
                for b, r, c in bands
            ],
        },
        "recommended_parameters": {
            "surround_gain": 0.25,
            "highpass_hz": 150.0,
            "lowpass_hz": 8000.0,
            "target_rms_ratio": 0.35,
            "centre_mode": "disabled",
            "decorrelation": "independent_allpass",
        },
        "version": "0.1.0",
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return csv_path, json_path

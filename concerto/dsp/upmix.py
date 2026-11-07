"""Matrix-based conservative upmix renderer.

7.1 is the primary spatial representation:

    FL = L,  FR = R                      (preservation: bit-identical)
    FC = 0 (default) or mid_fill_gain*M
    LFE = 0                              (disabled by default)
    SL = g_s * BP(allpass_A(S)) * gc
    SR = g_s * BP(allpass_B(S)) * gc
    BL = g_s * BP(allpass_C(S)) * gc
    BR = g_s * BP(allpass_D(S)) * gc

where BP is the smooth 150 Hz–8 kHz band-limit and gc is the slow RMS gain
controller driving surround RMS toward a configurable ratio of stereo RMS.
5.1 output is derived by combining side and back pairs with 0.707.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.buffer import MultichannelBuffer
from ..audio.layouts import INTERNAL_5_1, INTERNAL_7_1
from .allpass import AllPassBank
from .midside import mid_side
from .rms import GainController, bandlimit, bandlimit_sos

CHUNK = 1 << 14  # 16384 samples per processing block (~0.34 s @ 48 kHz)


@dataclass
class RenderConfig:
    layout: str = "7.1"
    bit_depth: int = 24
    preserve_front: bool = True
    surround_gain: float = 0.25
    highpass_hz: float = 150.0
    lowpass_hz: float = 8000.0
    filter_order: int = 4
    target_rms_ratio: float = 0.35
    rms_window_s: float = 0.4
    gain_tau_s: float = 2.0
    gain_min: float = 0.0
    gain_max: float = 2.0
    decorrelate: bool = True
    centre_mode: str = "disabled"
    mid_fill_gain: float = 0.3548


def load_config(path: str | Path) -> dict:
    """Load a TOML config file into a flat dict of render parameters."""
    import sys
    p = Path(path)
    if not p.exists() and getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            candidates = [
                Path(sys._MEIPASS) / path,
                Path(sys._MEIPASS) / "configs" / p.name,
                Path(sys._MEIPASS) / p.name,
            ]
            for c in candidates:
                if c.exists():
                    p = c
                    break
        if not p.exists():
            exe_dir = Path(sys.executable).parent
            candidates = [
                exe_dir / path,
                exe_dir / "configs" / p.name,
                exe_dir / p.name,
            ]
            for c in candidates:
                if c.exists():
                    p = c
                    break
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    return {
        "layout": out.get("layout", "7.1"),
        "bit_depth": out.get("bit_depth", 24),
        "preserve_front": out.get("preserve_front", True),
        "surround_gain": sur.get("gain", 0.25),
        "extraction": sur.get("extraction", "stereo_difference"),
        "stereo_beta": sur.get("stereo_beta", 0.25),
        "highpass_hz": sur.get("highpass_hz", 150.0),
        "lowpass_hz": sur.get("lowpass_hz", 8000.0),
        "filter_order": sur.get("filter_order", 4),
        "target_rms_ratio": sur.get("target_rms_ratio", 0.35),
        "rms_window_s": sur.get("rms_window_s", 0.4),
        "gain_tau_s": sur.get("gain_tau_s", 2.0),
        "gain_min": sur.get("gain_min", 0.0),
        "gain_max": sur.get("gain_max", 2.0),
        "decorrelate": dec.get("enabled", True),
        "coeffs": {
            "SL": dec.get("a_left", [0.68, -0.41, 0.23]),
            "SR": dec.get("a_right", [-0.55, 0.37, -0.19]),
            "BL": dec.get("a_back_l", [0.31, -0.62, 0.44]),
            "BR": dec.get("a_back_r", [-0.29, 0.48, -0.57]),
        },
        "centre_mode": cen.get("mode", "disabled"),
        "mid_fill_gain": cen.get("mid_fill_gain", 0.3548),
        "vocal_gain_db": cen.get("vocal_gain_db", -6.0),
        "adaptive": adv.get("enabled", False),
        "adaptive_beta": adv.get("beta_adaptive", False),
        "beta_min": adv.get("beta_min", 0.15),
        "beta_max": adv.get("beta_max", 0.90),
        "beta_tau_s": adv.get("beta_tau_s", 1.5),
    }


class UpmixRenderer:
    """Stateful renderer; filter and gain state persist across all blocks."""

    def __init__(self, sample_rate: int, config: dict) -> None:
        self.sr = sample_rate
        self.cfg = config
        # 7.1 is always the primary internal representation; 5.1 is derived
        # afterwards via derive_5_1_from_7_1 (see SPEC.md section 8).
        self.layout = INTERNAL_7_1
        self.base_gain = float(config["surround_gain"])
        self.surround_gain = self.base_gain       # live gain (multiplier applied by schedules)
        self.decorr_mix = 1.0                     # live decorrelation intensity (0..1)
        self.extraction = config.get("extraction", "stereo_difference")
        self.stereo_beta = float(config.get("stereo_beta", 0.25))
        self.banks: dict[str, AllPassBank] = {}
        if config["decorrelate"]:
            for label, coeffs in config["coeffs"].items():
                self.banks[label] = AllPassBank(coeffs)
        else:
            # Undecorrelated reference mode: identical unity banks
            self.banks = {k: AllPassBank([0.0]) for k in ("SL", "SR", "BL", "BR")}
        # Band-limit chains: one persistent state per feed. Mono difference
        # uses one shared chain; stereo difference keeps a directional feed
        # per side (L-beta*R for the left surrounds, R-beta*L for the right).
        if self.extraction == "stereo_difference":
            self.sos = {
                "SL": bandlimit_sos(sample_rate, config["highpass_hz"],
                                    config["lowpass_hz"], config["filter_order"]),
                "SR": bandlimit_sos(sample_rate, config["highpass_hz"],
                                    config["lowpass_hz"], config["filter_order"]),
            }
        else:
            self.sos = {
                "mono": bandlimit_sos(sample_rate, config["highpass_hz"],
                                      config["lowpass_hz"], config["filter_order"]),
            }
        gc_params = {
            "sample_rate": sample_rate,
            "window_s": config["rms_window_s"],
            "tau_s": config["gain_tau_s"],
            "target_ratio": config["target_rms_ratio"],
            "gain_min": config["gain_min"],
            "gain_max": config["gain_max"],
        }
        self.gain_ctl_sl = GainController(**gc_params)
        self.gain_ctl_sr = GainController(**gc_params)
        self.gain_ctl_bl = GainController(**gc_params)
        self.gain_ctl_br = GainController(**gc_params)
        self.gain_ctl = self.gain_ctl_sl
        self.centre_mode = config["centre_mode"]
        self.mid_fill_gain = float(config["mid_fill_gain"])

    @staticmethod
    def _mix(feed: np.ndarray, processed: np.ndarray, k: float) -> np.ndarray:
        """Decorrelation intensity: (1-k)*feed + k*allpassed(feed)."""
        if k >= 0.999:
            return processed
        if k <= 0.001:
            return feed
        return ((1.0 - k) * feed.astype(np.float64)
                + k * processed.astype(np.float64)).astype(np.float32)

    def render_block(self, l: np.ndarray, r: np.ndarray,
                     beta: float | None = None,
                     gain: float | None = None,
                     decorr: float | None = None) -> list[np.ndarray]:
        """Render a single block of stereo audio into multichannel channels."""
        if beta is not None:
            self.stereo_beta = float(np.clip(beta, 0.05, 0.98))
        if gain is not None:
            self.surround_gain = self.base_gain * float(np.clip(gain, 0.25, 4.0))
        if decorr is not None:
            self.decorr_mix = float(np.clip(decorr, 0.0, 1.0))

        n = l.shape[0]
        fl = l.copy()
        fr = r.copy()
        fc = np.zeros(n, dtype=np.float32)
        lfe = np.zeros(n, dtype=np.float32)

        if self.extraction == "stereo_difference":
            sl_feed = (l.astype(np.float64) - self.stereo_beta * r).astype(np.float32)
            sr_feed = (r.astype(np.float64) - self.stereo_beta * l).astype(np.float32)
            sl_bp, self.sos["SL"] = bandlimit(sl_feed, self.sos["SL"], apply=True)
            sr_bp, self.sos["SR"] = bandlimit(sr_feed, self.sos["SR"], apply=True)
            sl_bp = sl_bp * self.surround_gain
            sr_bp = sr_bp * self.surround_gain
        else:
            _, s = mid_side(l, r)
            s_bp, self.sos["mono"] = bandlimit(
                s.astype(np.float32), self.sos["mono"], apply=True)
            s_bp = s_bp * self.surround_gain
            sl_bp = sr_bp = s_bp

        sl = self.banks["SL"].process(sl_bp) if self.banks else sl_bp
        sr = self.banks["SR"].process(sr_bp) if self.banks else sr_bp
        sl = self._mix(sl_bp, sl, self.decorr_mix)
        sr = self._mix(sr_bp, sr, self.decorr_mix)
        sl = self.gain_ctl_sl.process(sl, l)
        sr = self.gain_ctl_sr.process(sr, r)

        is71 = self.layout.name == INTERNAL_7_1.name
        if is71:
            bl = self.banks["BL"].process(sl_bp) if self.banks else sl_bp
            br = self.banks["BR"].process(sr_bp) if self.banks else sr_bp
            bl = self._mix(sl_bp, bl, self.decorr_mix)
            br = self._mix(sr_bp, br, self.decorr_mix)
            bl = self.gain_ctl_bl.process(bl, l)
            br = self.gain_ctl_br.process(br, r)
        else:
            bl = np.zeros(n, dtype=np.float32)
            br = np.zeros(n, dtype=np.float32)

        if self.centre_mode == "mid_fill":
            m, _ = mid_side(l, r)
            fc = (self.mid_fill_gain * m).astype(np.float32)

        if is71:
            return [fl, fr, fc, lfe, sl, sr, bl, br]
        return [fl, fr, fc, lfe, sl, sr]

    def render(self, left: np.ndarray, right: np.ndarray,
               beta_schedule: np.ndarray | None = None,
               gain_schedule: np.ndarray | None = None,
               decorr_schedule: np.ndarray | None = None,
               progress: Callable[[int, int], None] | None = None
               ) -> MultichannelBuffer:
        """Render full-length stereo into an 8-channel primary buffer."""
        assert left.shape == right.shape
        assert left.dtype == np.float32
        n = left.shape[0]
        is71 = self.layout.name == INTERNAL_7_1.name
        order = list(INTERNAL_7_1.order) if is71 else list(INTERNAL_5_1.order)
        ch = {k: np.empty(n, dtype=np.float32) for k in order}

        n_chunks = (n + CHUNK - 1) // CHUNK
        bar_update = progress(n_chunks) if progress else None
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            ci = start // CHUNK
            b = beta_schedule[ci] if beta_schedule is not None else None
            g = gain_schedule[ci] if gain_schedule is not None else None
            d = decorr_schedule[ci] if decorr_schedule is not None else None
            blk = self.render_block(left[start:end], right[start:end], beta=b, gain=g, decorr=d)
            for k, label in enumerate(order):
                ch[label][start:end] = blk[k]
            if bar_update:
                bar_update()

        return MultichannelBuffer([ch[k] for k in order], self.sr, self.layout.name)

def derive_5_1_from_7_1(buf: MultichannelBuffer) -> MultichannelBuffer:
    """Combine side+back pairs: S_5.1 = 0.707*(SL+BL). Front unchanged."""
    idx = {label: i for i, label in enumerate(INTERNAL_7_1.order)}
    fl = buf.channels[idx["FL"]]
    fr = buf.channels[idx["FR"]]
    fc = buf.channels[idx["FC"]]
    lfe = buf.channels[idx["LFE"]]
    sl = 0.707 * (buf.channels[idx["SL"]] + buf.channels[idx["BL"]])
    sr = 0.707 * (buf.channels[idx["SR"]] + buf.channels[idx["BR"]])
    return MultichannelBuffer([fl, fr, fc, lfe, sl, sr], buf.sample_rate, INTERNAL_5_1.name)

def derive_5_1_from_7_1_block(channels: list[np.ndarray]) -> list[np.ndarray]:
    """Combine side+back pairs for a single block: S_5.1 = 0.707*(SL+BL)."""
    # channels is [FL, FR, FC, LFE, SL, SR, BL, BR]
    fl, fr, fc, lfe, sl, sr, bl, br = channels
    sl_51 = (0.707 * (sl.astype(np.float64) + bl.astype(np.float64))).astype(np.float32)
    sr_51 = (0.707 * (sr.astype(np.float64) + br.astype(np.float64))).astype(np.float32)
    return [fl, fr, fc, lfe, sl_51, sr_51]

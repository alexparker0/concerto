"""ctypes bridge to the native C++20 DSP core (native/build/bin/concerto_core).

The Python implementations remain the behavioural reference; this bridge is a
drop-in accelerator validated by equivalence tests (tests/test_native.py).
All file I/O stays in Python (libsndfile) — the native core is pure DSP.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

from .audio.buffer import MultichannelBuffer
from .audio.layouts import INTERNAL_7_1
from .dsp.upmix import CHUNK, UpmixRenderer  # Python reference + chunk size

_CHUNK = 1 << 14  # 16384 samples per processing block (~0.34 s @ 48 kHz)
assert _CHUNK == CHUNK


class NativeCoreUnavailable(RuntimeError):
    pass


def _find_library() -> Path | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "native" / "build" / "bin",
        Path(__file__).resolve().parent.parent / "native" / "build",
    ]
    for d in candidates:
        for name in ("concerto_core.dll", "libconcerto_core.dll", "libconcerto_core.so",
                     "libconcerto_core.dylib"):
            p = d / name
            if p.exists():
                return p
    return None


class _CcRendererParams(ct.Structure):
    _fields_ = [
        ("sample_rate", ct.c_int32),
        ("surround_gain", ct.c_double),
        ("n_sos_sections", ct.c_int32),
        ("sos", ct.c_double * (4 * 6)),
        ("a_sl", ct.c_double * 3),
        ("a_sr", ct.c_double * 3),
        ("a_bl", ct.c_double * 3),
        ("a_br", ct.c_double * 3),
        ("decorrelate", ct.c_int32),
        ("extraction", ct.c_int32),
        ("stereo_beta", ct.c_double),
        ("target_ratio", ct.c_double),
        ("rms_window_s", ct.c_double),
        ("gain_tau_s", ct.c_double),
        ("gain_min", ct.c_double),
        ("gain_max", ct.c_double),
        ("centre_mode", ct.c_int32),
        ("mid_fill_gain", ct.c_double),
        ("adaptive_beta", ct.c_int32),
    ]


def library_available() -> bool:
    return _find_library() is not None


class NativeUpmixRenderer:
    """Native-core renderer with the same interface as dsp.upmix.UpmixRenderer."""

    def __init__(self, sample_rate: int, config: dict) -> None:
        path = _find_library()
        if path is None:
            raise NativeCoreUnavailable("native concerto_core library not found")
        self._lib = ct.CDLL(str(path))
        lib = self._lib
        lib.cc_renderer_new.restype = ct.c_void_p
        lib.cc_renderer_new.argtypes = [ct.POINTER(_CcRendererParams)]
        lib.cc_renderer_free.argtypes = [ct.c_void_p]
        lib.cc_renderer_set_stereo_beta.argtypes = [ct.c_void_p, ct.c_double]
        lib.cc_renderer_set_dsp_params.argtypes = [ct.c_void_p, ct.c_double, ct.c_double]
        lib.cc_renderer_process.restype = ct.c_int
        lib.cc_renderer_process.argtypes = [ct.c_void_p] + [ct.POINTER(ct.c_float)] * 10 + [ct.c_int32]
        lib.cc_analyze_stereo.restype = ct.c_int

        sos = config.pop("_sos_flat", None)
        if sos is None:
            # Build the band-limit SOS here via scipy (identical coefficients
            # to the Python reference path).
            from scipy.signal import butter

            sos_hp = butter(config["filter_order"], config["highpass_hz"],
                            btype="highpass", fs=sample_rate, output="sos")
            sos_lp = butter(config["filter_order"], config["lowpass_hz"],
                            btype="lowpass", fs=sample_rate, output="sos")
            sos = [c for sec in (*sos_hp, *sos_lp) for c in sec]

        params = _CcRendererParams(
            sample_rate=sample_rate,
            surround_gain=float(config["surround_gain"]),
            n_sos_sections=len(sos) // 6,
            decorrelate=1 if config["decorrelate"] else 0,
            extraction=1 if config.get("extraction", "stereo_difference") == "stereo_difference" else 0,
            stereo_beta=float(config.get("stereo_beta", 0.25)),
            target_ratio=float(config["target_rms_ratio"]),
            rms_window_s=float(config["rms_window_s"]),
            gain_tau_s=float(config["gain_tau_s"]),
            gain_min=float(config["gain_min"]),
            gain_max=float(config["gain_max"]),
            centre_mode=1 if config["centre_mode"] == "mid_fill" else 0,
            mid_fill_gain=float(config["mid_fill_gain"]),
            # live_beta_ == static stereo_beta until set_stereo_beta() is
            # called, so enabling it unconditionally is behaviour-preserving.
            adaptive_beta=1 if config.get("extraction", "stereo_difference") == "stereo_difference" else 0,
        )
        for i, v in enumerate(sos):
            params.sos[i] = float(v)
        for key, field in (("SL", "a_sl"), ("SR", "a_sr"), ("BL", "a_bl"), ("BR", "a_br")):
            coeffs = config["coeffs"][key][:3]
            while len(coeffs) < 3:
                coeffs = coeffs + [0.0]
            for i, v in enumerate(coeffs):
                getattr(params, field)[i] = float(v)

        self._handle = lib.cc_renderer_new(ct.byref(params))
        if not self._handle:
            raise NativeCoreUnavailable("cc_renderer_new failed (bad params?)")
        self.sr = sample_rate
        self.layout = INTERNAL_7_1
        self.layout_name = INTERNAL_7_1.name
        self._lib_ref = lib  # keep alive

    def render_block(self, left: np.ndarray, right: np.ndarray,
                     beta: float | None = None,
                     gain: float | None = None,
                     decorr: float | None = None) -> list[np.ndarray]:
        lib = self._lib_ref
        l = np.ascontiguousarray(left, dtype=np.float32)
        r = np.ascontiguousarray(right, dtype=np.float32)
        n = l.shape[0]

        if beta is not None:
            lib.cc_renderer_set_stereo_beta(
                ct.c_void_p(self._handle),
                float(np.clip(beta, 0.05, 0.98)))
        if gain is not None or decorr is not None:
            lib.cc_renderer_set_dsp_params(
                ct.c_void_p(self._handle),
                float(np.clip(gain, 0.25, 4.0)) if gain is not None else 1.0,
                float(np.clip(decorr, 0.0, 1.0)) if decorr is not None else 1.0)

        outs = [np.empty(n, dtype=np.float32) for _ in range(8)]
        rc = lib.cc_renderer_process(
            ct.c_void_p(self._handle),
            l.ctypes.data_as(ct.POINTER(ct.c_float)),
            r.ctypes.data_as(ct.POINTER(ct.c_float)),
            *[ch.ctypes.data_as(ct.POINTER(ct.c_float)) for ch in outs],
            n,
        )
        if rc != 0:
            raise NativeCoreUnavailable(f"cc_renderer_process failed: {rc}")
        return outs

    def render(self, left: np.ndarray, right: np.ndarray,
               beta_schedule: np.ndarray | None = None,
               gain_schedule: np.ndarray | None = None,
               decorr_schedule: np.ndarray | None = None,
               progress: Callable[[int, int], None] | None = None) -> MultichannelBuffer:
        n_total = left.shape[0]
        outs = {k: np.empty(n_total, dtype=np.float32)
                for k in ("FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR")}

        n_chunks = (n_total + _CHUNK - 1) // _CHUNK
        bar_update = progress(n_chunks) if progress else None
        for start in range(0, n_total, _CHUNK):
            end = min(start + _CHUNK, n_total)
            ci = start // _CHUNK
            b = beta_schedule[ci] if beta_schedule is not None else None
            g = gain_schedule[ci] if gain_schedule is not None else None
            d = decorr_schedule[ci] if decorr_schedule is not None else None
            blk = self.render_block(left[start:end], right[start:end], beta=b, gain=g, decorr=d)
            for k, key in enumerate(("FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR")):
                outs[key][start:end] = blk[k]
            if bar_update:
                bar_update()

        order = list(INTERNAL_7_1.order)
        return MultichannelBuffer([outs[k] for k in order], self.sr, self.layout_name)

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            if getattr(self, "_handle", None):
                self._lib_ref.cc_renderer_free(ct.c_void_p(self._handle))
        except Exception:
            pass


def native_analyze_stereo(left: np.ndarray, right: np.ndarray, sample_rate: int,
                          fft_size: int = 4096, hop_size: int = 1024
                          ) -> dict[str, np.ndarray]:
    """Run the native STFT analysis; returns float64 arrays keyed by metric."""
    path = _find_library()
    if path is None:
        raise NativeCoreUnavailable("native concerto_core library not found")
    lib = ct.CDLL(str(path))
    lib.cc_analyze_stereo.restype = ct.c_int
    lib.cc_analyze_stereo.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                      ct.c_int64, ct.c_int32, ct.c_int32, ct.c_int32] + \
                                     [ct.POINTER(ct.c_double)] * 6
    l = np.ascontiguousarray(left, dtype=np.float64)
    r = np.ascontiguousarray(right, dtype=np.float64)
    bins = fft_size // 2 + 1
    out = {k: np.empty(bins, dtype=np.float64)
           for k in ("freqs", "rho", "coherence", "mid_energy", "side_energy", "directness")}
    rc = lib.cc_analyze_stereo(
        l.ctypes.data_as(ct.POINTER(ct.c_double)),
        r.ctypes.data_as(ct.POINTER(ct.c_double)),
        l.shape[0], fft_size, hop_size, sample_rate,
        *[out[k].ctypes.data_as(ct.POINTER(ct.c_double))
          for k in ("freqs", "rho", "coherence", "mid_energy", "side_energy", "directness")],
    )
    if rc != 0:
        raise NativeCoreUnavailable(f"cc_analyze_stereo failed: {rc}")
    return out


def make_renderer(sample_rate: int, config: dict) -> UpmixRenderer | NativeUpmixRenderer:
    """Native renderer when the core is built; Python reference otherwise."""
    if library_available():
        try:
            return NativeUpmixRenderer(sample_rate, dict(config))
        except NativeCoreUnavailable:
            pass
    return UpmixRenderer(sample_rate, config)

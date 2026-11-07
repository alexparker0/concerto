"""Stem separation via ONNX Runtime (multi-vendor GPU: Intel Arc, DirectML, CUDA) or Demucs.

Supports cross-vendor hardware acceleration:
  - Intel Arc GPUs and NPUs via OpenVINOExecutionProvider & DmlExecutionProvider
  - Windows DirectX 12 GPUs via DmlExecutionProvider (Intel, AMD, NVIDIA, Qualcomm)
  - NVIDIA GPUs via CUDAExecutionProvider
  - Offline CPU fallback via CPUExecutionProvider
Includes content-addressed disk caching to avoid redundant separation on re-renders.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class AIDependencyMissing(RuntimeError):
    pass


@dataclass
class Stems:
    """Four stems on the original sample timeline, float32, stereo."""

    vocals: tuple[np.ndarray, np.ndarray]
    drums: tuple[np.ndarray, np.ndarray]
    bass: tuple[np.ndarray, np.ndarray]
    other: tuple[np.ndarray, np.ndarray]

    def __iter__(self):
        return iter((self.vocals, self.drums, self.bass, self.other))

    def sum(self) -> tuple[np.ndarray, np.ndarray]:
        """Sum of all stems (Demucs' training target is mix == sum)."""
        l = self.vocals[0] + self.drums[0] + self.bass[0] + self.other[0]
        r = self.vocals[1] + self.drums[1] + self.bass[1] + self.other[1]
        return l, r

def get_available_execution_providers() -> list[Any]:
    """Return prioritized execution providers for cross-vendor GPU acceleration.

    Prioritizes Intel Arc GPU/NPU (OpenVINO), Windows DirectML (Intel Arc, AMD, NVIDIA),
    and CUDA before falling back to CPU.
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]

    providers: list[Any] = []
    # 1. Intel Arc / OpenVINO (optimal for Intel Arc discrete & integrated GPUs and NPUs)
    if "OpenVINOExecutionProvider" in available:
        providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"}))
    # 2. DirectML (universal cross-vendor Windows GPU acceleration: Intel Arc, AMD, NVIDIA)
    if "DmlExecutionProvider" in available:
        providers.append(("DmlExecutionProvider", {"device_id": 0}))
    # 3. CUDA / TensorRT
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    if "TensorrtExecutionProvider" in available:
        providers.append(("TensorrtExecutionProvider", {"device_id": 0}))
    # 4. AMD ROCm
    if "ROCMExecutionProvider" in available:
        providers.append(("ROCMExecutionProvider", {"device_id": 0}))
    # 5. Always include CPU
    providers.append("CPUExecutionProvider")
    return providers


def _compute_audio_hash(left: np.ndarray, right: np.ndarray, sample_rate: int,
                        model_name: str) -> str:
    h = hashlib.sha256()
    h.update(str(sample_rate).encode())
    h.update(model_name.encode())
    h.update(str(len(left)).encode())
    h.update(np.ascontiguousarray(left[:min(16384, len(left))]).tobytes())
    h.update(np.ascontiguousarray(right[:min(16384, len(right))]).tobytes())
    if len(left) > 16384:
        h.update(np.ascontiguousarray(left[-16384:]).tobytes())
        h.update(np.ascontiguousarray(right[-16384:]).tobytes())
    return h.hexdigest()[:24]


def _cache_path(audio_hash: str, cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir else Path(".concerto_cache") / "stems"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{audio_hash}.npz"


def _load_cached_stems(path: Path) -> Stems | None:
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return Stems(
            vocals=(data["vl"], data["vr"]),
            drums=(data["dl"], data["dr"]),
            bass=(data["bl"], data["br"]),
            other=(data["ol"], data["or_r"]),
        )
    except Exception:
        return None


def _save_cached_stems(path: Path, stems: Stems) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vl=stems.vocals[0], vr=stems.vocals[1],
            dl=stems.drums[0], dr=stems.drums[1],
            bl=stems.bass[0], br=stems.bass[1],
            ol=stems.other[0], or_r=stems.other[1],
        )
    except Exception:
        pass


def _roformer_hann_periodic(win_length: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_length) / win_length))


def _roformer_stft(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 441) -> np.ndarray:
    window = _roformer_hann_periodic(n_fft)
    padded = np.pad(audio, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=1)[:, ::hop_length, :]
    spec = np.fft.rfft(frames * window, axis=2)
    return np.ascontiguousarray(spec.transpose(0, 2, 1))


def _roformer_istft(spec: np.ndarray, n_fft: int = 2048, hop_length: int = 441, length: int = 352800) -> np.ndarray:
    window = _roformer_hann_periodic(n_fft)
    n_frames = spec.shape[1]
    expected = n_fft + hop_length * (n_frames - 1)
    frames = np.fft.irfft(spec, n=n_fft, axis=0) * window[:, None]
    signal = np.zeros(expected)
    envelope = np.zeros(expected)
    w2 = window ** 2
    for t in range(n_frames):
        start = t * hop_length
        signal[start : start + n_fft] += frames[:, t]
        envelope[start : start + n_fft] += w2
    start = n_fft // 2
    stop = min(start + length, expected)
    out = signal[start:stop] / np.maximum(envelope[start:stop], 1e-12)
    if out.size < length:
        out = np.pad(out, (0, length - out.size))
    return out


def _roformer_stft_bands_last(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 441) -> np.ndarray:
    spec = _roformer_stft(audio, n_fft, hop_length)
    real = np.stack([spec.real, spec.imag], axis=-1)
    interleaved = real.transpose(1, 0, 2, 3)
    flat = interleaved.reshape(-1, spec.shape[2], 2)
    return np.ascontiguousarray(flat[None].astype(np.float32))


def _roformer_istft_bands_last(spec: np.ndarray, channels: int = 2, n_fft: int = 2048,
                              hop_length: int = 441, length: int = 352800) -> np.ndarray:
    n_freqs = spec.shape[0] // channels
    per_channel = spec.reshape(n_freqs, channels, spec.shape[1], 2).transpose(1, 0, 2, 3)
    complex_spec = per_channel[..., 0] + 1j * per_channel[..., 1]
    return np.stack([_roformer_istft(complex_spec[c], n_fft, hop_length, length) for c in range(channels)])


def _complex_multiply(spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ar, ai = spec[..., 0], spec[..., 1]
    br, bi = mask[..., 0], mask[..., 1]
    return np.stack([ar * br - ai * bi, ar * bi + ai * br], axis=-1)


def separate_stems_onnx(left: np.ndarray, right: np.ndarray, sample_rate: int,
                        model_path: str | Path, providers: list[Any] | None = None,
                        progress: bool = False) -> Stems:
    """Run stem separation using ONNX Runtime with cross-vendor GPU execution.

    Automatically supports:
      1. RoFormer / Mel-Band RoFormer (rank-4 complex spectrogram models)
      2. Standard waveform separation models (rank-3 audio tensor models)
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise AIDependencyMissing(
            "ONNX inference requires onnxruntime: pip install onnxruntime-directml (or onnxruntime)"
        ) from exc

    providers = providers or get_available_execution_providers()
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    model_sr = 44100
    l = np.asarray(left, dtype=np.float32)
    r = np.asarray(right, dtype=np.float32)
    if sample_rate != model_sr:
        from scipy.signal import resample_poly
        g = np.gcd(sample_rate, model_sr)
        l = resample_poly(l, model_sr // g, sample_rate // g).astype(np.float32)
        r = resample_poly(r, model_sr // g, sample_rate // g).astype(np.float32)

    mix = np.stack([l, r])  # [2, samples]
    total_len = mix.shape[1]

    def back(x: np.ndarray) -> np.ndarray:
        if sample_rate != model_sr:
            from scipy.signal import resample_poly
            g = np.gcd(sample_rate, model_sr)
            x = resample_poly(x, sample_rate // g, model_sr // g).astype(np.float32)
        return x[: len(left)]

    def _run(feed: dict) -> list:
        nonlocal session
        try:
            return session.run(None, feed)
        except Exception:
            # DML / GPU provider may crash on certain graphs; fall back to CPU
            session = ort.InferenceSession(str(model_path),
                                          providers=["CPUExecutionProvider"])
            return session.run(None, feed)

    # Case 1: RoFormer / Mel-Band RoFormer (Rank 4: [batch, F*C, T, 2])
    if len(input_shape) == 4 or input_name == "spec":
        n_fft = 2048
        hop_length = 441
        spec_frames = input_shape[2] if isinstance(input_shape[2], int) else 801
        chunk_size = (spec_frames - 1) * hop_length
        step = chunk_size // 2
        fade_size = chunk_size // 10
        border = chunk_size - step

        window = np.ones(chunk_size, dtype=np.float32)
        window[:fade_size] = np.linspace(0, 1, fade_size)
        window[-fade_size:] = np.linspace(1, 0, fade_size)

        pad_needed = (step - (total_len % step)) % step
        padded_mix = np.pad(mix, ((0, 0), (border, border + pad_needed + chunk_size)), mode="reflect")

        accum = None
        weight_accum = np.zeros(padded_mix.shape[1], dtype=np.float32)

        n_chunks = len(range(0, total_len + pad_needed, step))
        for ci, start in enumerate(range(0, total_len + pad_needed, step)):
            seg = padded_mix[:, start : start + chunk_size]
            spec_in = _roformer_stft_bands_last(seg, n_fft, hop_length)
            mask_out = _run({input_name: spec_in})[0]
            # mask_out: [1, n_stems, F*C, T, 2]
            masked = _complex_multiply(spec_in[:, None], mask_out)[0]
            masked[..., :2, :, :] = 0.0  # zero DC bin

            n_stems = masked.shape[0]
            if accum is None:
                accum = np.zeros((n_stems, 2, padded_mix.shape[1]), dtype=np.float32)

            for s in range(n_stems):
                time_domain = _roformer_istft_bands_last(
                    masked[s], channels=2, n_fft=n_fft, hop_length=hop_length, length=chunk_size
                )
                accum[s, :, start : start + chunk_size] += time_domain * window

            weight_accum[start : start + chunk_size] += window

        weight_accum = np.maximum(weight_accum, 1e-12)
        accum /= weight_accum[None, None, :]
        final_stems = accum[:, :, border : border + total_len]

        if final_stems.shape[0] == 1:
            # Single-stem vocal model: residual = mix - vocals
            voc_l, voc_r = final_stems[0, 0], final_stems[0, 1]
            res_l = mix[0] - voc_l
            res_r = mix[1] - voc_r
            zeros = np.zeros_like(voc_l)
            return Stems(
                vocals=(back(voc_l), back(voc_r)),
                drums=(back(zeros), back(zeros)),
                bass=(back(zeros), back(zeros)),
                other=(back(res_l), back(res_r)),
            )
        else:
            # 4-stem RoFormer: drums (0), bass (1), other (2), vocals (3)
            return Stems(
                drums=(back(final_stems[0, 0]), back(final_stems[0, 1])),
                bass=(back(final_stems[1, 0]), back(final_stems[1, 1])),
                other=(back(final_stems[2, 0]), back(final_stems[2, 1])),
                vocals=(back(final_stems[3, 0]), back(final_stems[3, 1])),
            )

    # Case 2: Waveform models (Rank 3: [1, 2, samples])
    inp = mix[np.newaxis, ...]  # [1, 2, samples]
    out = _run({input_name: inp})[0]
    out = out[0]
    return Stems(
        drums=(back(out[0, 0]), back(out[0, 1])),
        bass=(back(out[1, 0]), back(out[1, 1])),
        other=(back(out[2, 0]), back(out[2, 1])),
        vocals=(back(out[3, 0]), back(out[3, 1])),
    )


def separate_stems(left: np.ndarray, right: np.ndarray, sample_rate: int,
                   model_name: str = "htdemucs", segment: int | None = None,
                   progress: bool = False, use_cache: bool = True,
                   cache_dir: str | Path | None = None,
                   onnx_model_path: str | Path | None = None) -> Stems:
    """Separate stereo mix into stems via ONNX Runtime or Demucs.

    Supports disk caching and cross-vendor GPU execution (Intel Arc, DirectML, CUDA).
    """
    if use_cache:
        ahash = _compute_audio_hash(left, right, sample_rate,
                                    f"{model_name}:{onnx_model_path or 'torch'}")
        cpath = _cache_path(ahash, cache_dir)
        cached = _load_cached_stems(cpath)
        if cached is not None:
            return cached

    if onnx_model_path and Path(onnx_model_path).exists():
        stems = separate_stems_onnx(left, right, sample_rate, onnx_model_path,
                                    progress=progress)
        if use_cache:
            _save_cached_stems(cpath, stems)
        return stems

    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AIDependencyMissing(
            "AI features require torch and demucs: pip install torch demucs (or onnxruntime)"
        ) from exc

    demucs_sr = 44100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name)
    model.to(device)
    model.eval()

    l = np.asarray(left, dtype=np.float32)
    r = np.asarray(right, dtype=np.float32)
    if sample_rate != demucs_sr:
        from scipy.signal import resample_poly
        g = np.gcd(sample_rate, demucs_sr)
        l = resample_poly(l, demucs_sr // g, sample_rate // g).astype(np.float32)
        r = resample_poly(r, demucs_sr // g, sample_rate // g).astype(np.float32)

    wav = torch.from_numpy(np.stack([np.stack([l, r])])).to(device)
    kwargs = {}
    if segment is not None:
        kwargs["segment"] = segment
    with torch.no_grad():
        out = apply_model(model, wav, device=device, split=True,
                          overlap=0.25, progress=progress, **kwargs)
    out = out[0].cpu().numpy().astype(np.float32)
    names = model.sources
    stems_dict = {name: (out[i, 0], out[i, 1]) for i, name in enumerate(names)}

    def back(x: np.ndarray) -> np.ndarray:
        if sample_rate != demucs_sr:
            from scipy.signal import resample_poly
            g = np.gcd(sample_rate, demucs_sr)
            x = resample_poly(x, sample_rate // g, demucs_sr // g).astype(np.float32)
        return x[: len(left)]

    result = Stems(
        vocals=(back(stems_dict["vocals"][0]), back(stems_dict["vocals"][1])),
        drums=(back(stems_dict["drums"][0]), back(stems_dict["drums"][1])),
        bass=(back(stems_dict["bass"][0]), back(stems_dict["bass"][1])),
        other=(back(stems_dict["other"][0]), back(stems_dict["other"][1])),
    )
    if use_cache:
        _save_cached_stems(cpath, result)
    return result

"""FFmpeg fallback I/O for formats libsndfile cannot handle.

Opus, MP3, M4A/AAC, WavPack, ALAC etc. are decoded through the ffmpeg CLI
(already a documented dependency) into float64 PCM. Metadata comes from
ffprobe. The original file is never modified.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np


class FfmpegUnavailable(RuntimeError):
    pass


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FfmpegUnavailable(f"{name} not found on PATH")
    return path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: str) -> dict:
    """Return the ffprobe JSON stream info for an audio file."""
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def ffmpeg_decode_stereo(path: str) -> tuple[np.ndarray, int, dict]:
    """Decode any ffmpeg-supported file to float64 stereo [frames x 2].

    Returns (data, sample_rate, metadata). Non-stereo sources are downmixed
    to stereo by ffmpeg's default pan (matching typical player behaviour);
    the source metadata records the original channel count.
    """
    info = probe(path)
    streams = [s for s in info.get("streams", [])
               if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError(f"no audio stream found: {path}")
    a = streams[0]
    src_sr = int(a["sample_rate"])
    src_channels = int(a["channels"])

    proc = subprocess.Popen(
        [_bin("ffmpeg"), "-v", "error", "-i", path,
         "-vn", "-ac", "2", "-ar", str(src_sr),
         "-f", "f64le", "-c:a", "pcm_f64le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate()
    if proc.returncode != 0:
        err_msg = stderr_data.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed (code {proc.returncode}): {err_msg}")
    data = np.frombuffer(stdout_data, dtype=np.float64).reshape(-1, 2)

    meta = {
        "format": a.get("codec_name", "unknown"),
        "subtype": f"{int(a.get('bits_per_raw_sample') or 0)}-bit"
        if a.get("bits_per_raw_sample") else a.get("profile", "lossy"),
        "sample_rate": src_sr,
        "channels": src_channels,
        "frame_count": int(a.get("duration_ts") or 0) or len(data),
        "duration_s": float(a.get("duration") or 0.0),
        "container": info.get("format", {}).get("format_name", "unknown"),
    }
    return data, src_sr, meta

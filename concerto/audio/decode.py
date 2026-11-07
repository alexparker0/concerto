"""Decode stereo FLAC/WAV via libsndfile (soundfile), with an FFmpeg
fallback for formats libsndfile cannot read (Opus, MP3, M4A/AAC, WavPack...).

The original file is never modified; only decoded float32 PCM leaves this
module, together with complete source metadata.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from .buffer import SourceMetadata, StereoBuffer, measure
from .ffmpeg_io import ffmpeg_available, ffmpeg_decode_stereo


def _meta_from_ffmpeg(path: str, data: np.ndarray, sr: int, meta: dict) -> SourceMetadata:
    peak, rms = measure(data)
    return SourceMetadata(
        path=str(path),
        format=str(meta.get("format", "unknown")),
        subtype=str(meta.get("subtype", "unknown")),
        sample_rate=int(sr),
        frame_count=int(data.shape[0]),
        channels=2,
        peak=peak,
        rms=rms,
    )


def decode_stereo(path: str) -> tuple[StereoBuffer, SourceMetadata]:
    """Decode a stereo audio file to float32 PCM.

    FLAC/WAV/AIFF and other libsndfile formats decode natively; everything
    else (Opus, MP3, M4A/AAC, ...) falls back to FFmpeg. Sample rate is
    preserved verbatim — no resampling happens anywhere in the pipeline.
    For lossy sources the decoded rate is the source's decoder rate (Opus is
    always 48 kHz), which is preserved like any other rate.
    """
    try:
        info = sf.info(path)
    except (RuntimeError, OSError):
        info = None

    if info is not None:
        if info.channels != 2:
            raise ValueError(
                f"expected stereo input, got {info.channels} channels: {path}")
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        left = np.ascontiguousarray(data[:, 0], dtype=np.float32)
        right = np.ascontiguousarray(data[:, 1], dtype=np.float32)
        peak, rms = measure(data)
        meta = SourceMetadata(
            path=str(path),
            format=info.format,
            subtype=info.subtype,
            sample_rate=int(sr),
            frame_count=int(info.frames),
            channels=int(info.channels),
            peak=peak,
            rms=rms,
        )
        return StereoBuffer(left, right, int(sr), int(info.frames)), meta

    # FFmpeg fallback for formats soundfile cannot open.
    if not ffmpeg_available():
        raise ValueError(
            f"unsupported or unreadable audio file: {path} "
            f"(libsndfile failed and ffmpeg is not available)")
    data, sr, fmeta = ffmpeg_decode_stereo(path)
    meta = _meta_from_ffmpeg(path, data, sr, fmeta)
    left = np.ascontiguousarray(data[:, 0], dtype=np.float32)
    right = np.ascontiguousarray(data[:, 1], dtype=np.float32)
    return StereoBuffer(left, right, int(sr), data.shape[0]), meta


class StereoStreamReader:
    """Chunked streaming decoder with O(chunk_size) memory footprint.

    Streams stereo float32 PCM block-by-block without loading the entire
    recording into RAM.
    """

    def __init__(self, path: str, chunk_size: int = 16384) -> None:
        self.path = str(path)
        self.chunk_size = int(chunk_size)
        self._sf: sf.SoundFile | None = None
        self._ffmpeg_proc = None
        self._is_ffmpeg = False

        try:
            info = sf.info(self.path)
        except (RuntimeError, OSError):
            info = None

        if info is not None:
            if info.channels != 2:
                raise ValueError(
                    f"expected stereo input, got {info.channels} channels: {self.path}")
            self.format = str(info.format)
            self.subtype = str(info.subtype)
            self.sample_rate = int(info.samplerate)
            self.frame_count = int(info.frames)
            self.channels = 2
            self._sf = sf.SoundFile(self.path, mode="r")
        else:
            if not ffmpeg_available():
                raise ValueError(
                    f"unsupported or unreadable audio file: {self.path} "
                    f"(libsndfile failed and ffmpeg is not available)")
            from .ffmpeg_io import probe
            meta = probe(self.path)
            streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
            if not streams:
                raise ValueError(f"no audio stream found: {self.path}")
            s = streams[0]
            self.sample_rate = int(s.get("sample_rate", 44100))
            dur = float(s.get("duration", 0.0)) or float(meta.get("format", {}).get("duration", 0.0))
            self.frame_count = int(dur * self.sample_rate)
            self.format = str(meta.get("format", {}).get("format_name", "ffmpeg"))
            self.subtype = str(s.get("codec_name", "unknown"))
            self.channels = 2
            self._is_ffmpeg = True
            self._start_ffmpeg()

    def _start_ffmpeg(self) -> None:
        import subprocess
        from .ffmpeg_io import _bin
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.kill()
            except Exception:
                pass
        cmd = [
            _bin("ffmpeg"), "-v", "error", "-i", self.path,
            "-vn", "-ac", "2", "-ar", str(self.sample_rate),
            "-f", "f32le", "-c:a", "pcm_f32le", "-"
        ]
        self._ffmpeg_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def blocks(self):
        """Yield (left_chunk, right_chunk) arrays of float32."""
        if self._sf is not None:
            for block in self._sf.blocks(blocksize=self.chunk_size, dtype="float32", always_2d=True):
                yield np.ascontiguousarray(block[:, 0]), np.ascontiguousarray(block[:, 1])
        else:
            bytes_per_frame = 2 * 4  # 2 channels * float32
            while True:
                buf = self._ffmpeg_proc.stdout.read(self.chunk_size * bytes_per_frame)
                if not buf:
                    break
                arr = np.frombuffer(buf, dtype=np.float32).reshape(-1, 2)
                yield np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1])

    def seek(self, frame: int = 0) -> None:
        """Rewind to target frame (default 0)."""
        if self._sf is not None:
            self._sf.seek(frame)
        else:
            self._start_ffmpeg()

    def close(self) -> None:
        if self._sf is not None:
            self._sf.close()
            self._sf = None
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.kill()
            except Exception:
                pass
            self._ffmpeg_proc = None

    def __enter__(self) -> "StereoStreamReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def open_stereo_stream(path: str, chunk_size: int = 16384) -> StereoStreamReader:
    """Open a streaming stereo reader for path."""
    return StereoStreamReader(path, chunk_size=chunk_size)

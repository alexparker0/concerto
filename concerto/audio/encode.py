"""Multichannel output encoding via libsndfile, with explicit layout mapping.

Output format is selected by file extension: FLAC (default) or WAV.
Every conversion between the application-internal channel order and the
file's native order goes through concerto.audio.layouts — never raw copies.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from .layouts import (
    FLAC_6CH_AS_51, FLAC_8CH, INTERNAL_5_1, INTERNAL_7_1,
    WAV_6CH, WAV_8CH, permutation,
)

_SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}

def _dither_channels(channels: list[np.ndarray], bit_depth: int) -> list[np.ndarray]:
    """Apply TPDF dither to processed surround channels, preserving FL/FR and silent channels."""
    if bit_depth >= 32:
        return channels
    out = []
    lsb = 1.0 / (1 << (bit_depth - 1))
    rng = np.random.default_rng()
    for i, ch in enumerate(channels):
        # Preserve FL (0), FR (1), and silent channels (like LFE)
        if i in (0, 1) or np.max(np.abs(ch)) == 0.0:
            out.append(ch)
        else:
            d = (rng.uniform(-0.5, 0.5, size=ch.shape) +
                 rng.uniform(-0.5, 0.5, size=ch.shape)) * lsb
            out.append(np.clip(ch + d, -1.0, 1.0).astype(ch.dtype))
    return out


def _to_file_order(channels: list[np.ndarray], file_layout, internal_layout):
    """Interleave an internal-order channel-major buffer into file order."""
    perm = permutation(internal_layout, file_layout)
    file_channels = [channels[i] for i in perm]
    return np.stack(file_channels, axis=1).astype(np.float64)


def _file_layout(internal_name: str, container: str):
    if internal_name == INTERNAL_5_1.name:
        return (FLAC_6CH_AS_51 if container == "flac" else WAV_6CH), INTERNAL_5_1
    if internal_name == INTERNAL_7_1.name:
        return (FLAC_8CH if container == "flac" else WAV_8CH), INTERNAL_7_1
    raise ValueError(f"not an internal layout: {internal_name!r}")


def encode_flac(
    channels: list[np.ndarray],
    sample_rate: int,
    layout_name: str,
    path: str,
    bit_depth: int = 24,
    dither: bool = False,
) -> str:
    """Write internal-order channels to a multichannel FLAC file."""
    if bit_depth not in _SUBTYPES:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    file_layout, internal = _file_layout(layout_name, "flac")
    if dither:
        channels = _dither_channels(channels, bit_depth)
    data = _to_file_order(channels, file_layout, internal)
    sf.write(path, data, sample_rate, format="FLAC", subtype=_SUBTYPES[bit_depth])
    return path


def encode_wav(
    channels: list[np.ndarray],
    sample_rate: int,
    layout_name: str,
    path: str,
    bit_depth: int = 24,
    dither: bool = False,
) -> str:
    """Write internal-order channels to a multichannel WAVE file.

    WAVE 8ch orders the back pair before the side pair (the opposite of
    FLAC) — the mapping table handles this. 32-bit output is float32.
    """
    if bit_depth == 32:
        subtype, dtype = "FLOAT", np.float32
    elif bit_depth in _SUBTYPES:
        subtype, dtype = _SUBTYPES[bit_depth], np.float64
    else:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    file_layout, internal = _file_layout(layout_name, "wav")
    if dither and bit_depth < 32:
        channels = _dither_channels(channels, bit_depth)
    data = _to_file_order(channels, file_layout, internal)
    sf.write(path, data.astype(dtype), sample_rate, format="WAV", subtype=subtype)
    return path


def encode_output(channels: list[np.ndarray], sample_rate: int,
                  layout_name: str, path: str, bit_depth: int = 24,
                  dither: bool = False) -> str:
    """Write internal-order channels, choosing the container by extension."""
    lower = str(path).lower()
    if lower.endswith(".flac"):
        return encode_flac(channels, sample_rate, layout_name, path, bit_depth, dither=dither)
    if lower.endswith(".wav"):
        return encode_wav(channels, sample_rate, layout_name, path, bit_depth, dither=dither)
    raise ValueError(
        f"unsupported output format: {path} (supported: .flac, .wav)")



class MultichannelStreamWriter:
    """Chunked streaming writer for multichannel FLAC/WAV with O(chunk_size) memory."""

    def __init__(self, path: str, sample_rate: int, layout_name: str,
                 bit_depth: int = 24, dither: bool = False) -> None:
        self.path = str(path)
        self.sample_rate = int(sample_rate)
        self.layout_name = str(layout_name)
        self.bit_depth = int(bit_depth)
        self.dither = bool(dither)

        lower = self.path.lower()
        if lower.endswith(".flac"):
            self.container = "flac"
            if self.bit_depth not in _SUBTYPES:
                raise ValueError(f"unsupported bit depth: {self.bit_depth}")
            self.subtype = _SUBTYPES[self.bit_depth]
            self.format = "FLAC"
            self.dtype = np.float64
        elif lower.endswith(".wav"):
            self.container = "wav"
            if self.bit_depth == 32:
                self.subtype, self.dtype = "FLOAT", np.float32
            elif self.bit_depth in _SUBTYPES:
                self.subtype, self.dtype = _SUBTYPES[self.bit_depth], np.float64
            else:
                raise ValueError(f"unsupported bit depth: {self.bit_depth}")
            self.format = "WAV"
        else:
            raise ValueError(f"unsupported output format: {self.path} (supported: .flac, .wav)")

        self.file_layout, self.internal = _file_layout(self.layout_name, self.container)
        self.n_channels = len(self.file_layout.order)
        self._sf = sf.SoundFile(
            self.path,
            mode="w",
            samplerate=self.sample_rate,
            channels=self.n_channels,
            format=self.format,
            subtype=self.subtype,
        )
        self._peak = 0.0
        self._frames_written = 0

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def write_block(self, channels: list[np.ndarray]) -> None:
        """Write one chunk of internal-order channels directly to disk."""
        if not channels:
            return
        for c in channels:
            p = float(np.max(np.abs(c))) if c.size else 0.0
            if p > self._peak:
                self._peak = p

        dithered = channels
        if self.dither and (self.container == "flac" or self.bit_depth < 32):
            dithered = _dither_channels(channels, self.bit_depth)

        data = _to_file_order(dithered, self.file_layout, self.internal)
        self._sf.write(data.astype(self.dtype))
        self._frames_written += channels[0].shape[0]

    def close(self) -> float:
        if self._sf is not None:
            self._sf.close()
            self._sf = None
        return self._peak

    def __enter__(self) -> "MultichannelStreamWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def open_multichannel_stream(path: str, sample_rate: int, layout_name: str,
                             bit_depth: int = 24, dither: bool = False
                             ) -> MultichannelStreamWriter:
    """Open a streaming multichannel writer for path."""
    return MultichannelStreamWriter(path, sample_rate, layout_name,
                                    bit_depth=bit_depth, dither=dither)

def read_channels(path: str) -> tuple[np.ndarray, int, int]:
    """Read a multichannel file back as (frames x channels) float64."""
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    return data, int(sr), data.shape[1]

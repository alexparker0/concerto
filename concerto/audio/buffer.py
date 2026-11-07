"""Audio buffers and source metadata preservation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourceMetadata:
    """Everything needed to preserve the original recording's identity."""

    path: str
    format: str
    subtype: str            # e.g. "PCM_24" — original bit-depth metadata
    sample_rate: int
    frame_count: int
    channels: int
    peak: float             # original absolute peak
    rms: float              # original overall RMS


@dataclass
class StereoBuffer:
    """Float32 interleaved-free stereo representation."""

    left: np.ndarray        # float32 [n]
    right: np.ndarray       # float32 [n]
    sample_rate: int
    frame_count: int

    def __post_init__(self) -> None:
        if self.left.dtype != np.float32:
            self.left = self.left.astype(np.float32)
        if self.right.dtype != np.float32:
            self.right = self.right.astype(np.float32)
        assert self.left.shape == self.right.shape
        self.frame_count = int(self.left.shape[0])


@dataclass
class MultichannelBuffer:
    """Channel-major multichannel buffer in internal channel order."""

    channels: list[np.ndarray]   # float32 [n] each
    sample_rate: int
    layout_name: str

    def __post_init__(self) -> None:
        self.channels = [
            c.astype(np.float32) if c.dtype != np.float32 else c for c in self.channels
        ]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def frame_count(self) -> int:
        return int(self.channels[0].shape[0])

    def peak(self) -> float:
        return max(float(np.max(np.abs(c))) for c in self.channels)


def measure(arr: np.ndarray) -> tuple[float, float]:
    """Return (peak, rms) of a float array, accumulated in float64."""
    a = np.asarray(arr, dtype=np.float64)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    return peak, rms

class StereoRingBuffer:
    """Fixed-capacity circular buffer for 2-channel float32 audio."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf_l = np.zeros(self.capacity, dtype=np.float32)
        self._buf_r = np.zeros(self.capacity, dtype=np.float32)
        self._r_pos = 0
        self._w_pos = 0
        self._count = 0

    @property
    def available(self) -> int:
        """Number of readable samples currently buffered."""
        return self._count

    @property
    def space(self) -> int:
        """Remaining capacity for writes."""
        return self.capacity - self._count

    def write(self, left: np.ndarray, right: np.ndarray) -> int:
        """Write stereo samples into the ring buffer. Returns number of samples written."""
        l = np.asarray(left, dtype=np.float32)
        r = np.asarray(right, dtype=np.float32)
        assert l.shape == r.shape
        to_write = min(l.shape[0], self.space)
        if to_write == 0:
            return 0

        first_part = min(to_write, self.capacity - self._w_pos)
        second_part = to_write - first_part

        self._buf_l[self._w_pos : self._w_pos + first_part] = l[:first_part]
        self._buf_r[self._w_pos : self._w_pos + first_part] = r[:first_part]

        if second_part > 0:
            self._buf_l[:second_part] = l[first_part:to_write]
            self._buf_r[:second_part] = r[first_part:to_write]

        self._w_pos = (self._w_pos + to_write) % self.capacity
        self._count += to_write
        return to_write

    def read(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Consume and return up to `count` stereo samples."""
        to_read = min(int(count), self._count)
        if to_read == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        first_part = min(to_read, self.capacity - self._r_pos)
        second_part = to_read - first_part

        if second_part == 0:
            out_l = self._buf_l[self._r_pos : self._r_pos + first_part].copy()
            out_r = self._buf_r[self._r_pos : self._r_pos + first_part].copy()
        else:
            out_l = np.concatenate([
                self._buf_l[self._r_pos : self._r_pos + first_part],
                self._buf_l[:second_part],
            ])
            out_r = np.concatenate([
                self._buf_r[self._r_pos : self._r_pos + first_part],
                self._buf_r[:second_part],
            ])

        self._r_pos = (self._r_pos + to_read) % self.capacity
        self._count -= to_read
        return out_l, out_r

    def peek(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return up to `count` samples without advancing read pointer."""
        to_read = min(int(count), self._count)
        if to_read == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        first_part = min(to_read, self.capacity - self._r_pos)
        second_part = to_read - first_part

        if second_part == 0:
            out_l = self._buf_l[self._r_pos : self._r_pos + first_part].copy()
            out_r = self._buf_r[self._r_pos : self._r_pos + first_part].copy()
        else:
            out_l = np.concatenate([
                self._buf_l[self._r_pos : self._r_pos + first_part],
                self._buf_l[:second_part],
            ])
            out_r = np.concatenate([
                self._buf_r[self._r_pos : self._r_pos + first_part],
                self._buf_r[:second_part],
            ])
        return out_l, out_r

    def clear(self) -> None:
        """Reset buffer state."""
        self._r_pos = 0
        self._w_pos = 0
        self._count = 0

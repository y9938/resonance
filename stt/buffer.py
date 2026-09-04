from __future__ import annotations

import numpy as np


class AudioMemoryBuffer:
    """
    Zero-I/O growable contiguous in-RAM PCM buffer.
    Backing storage: C-level bytearray with zero-copy numpy projection.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        # Invariant: Stores float32 little-endian PCM bytes (4 bytes per sample).
        self._data = bytearray()

    def append(self, chunk: np.ndarray) -> None:
        """
        Caller obligation: chunk must be 1D float32 array or contiguous buffer.
        """
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk, dtype=np.float32)
        elif chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # C-level amortized O(1) append into contiguous buffer
        self._data.extend(chunk.tobytes())

    def as_ndarray(self) -> np.ndarray:
        """
        Returns a 1D float32 numpy view directly referencing the backing bytearray without copying.
        """
        if not self._data:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(self._data, dtype=np.float32)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        """Number of float32 samples in the buffer."""
        return len(self._data) // 4

    @property
    def duration_sec(self) -> float:
        return len(self) / self.sample_rate

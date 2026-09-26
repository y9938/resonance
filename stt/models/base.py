from __future__ import annotations

import abc
from typing import Any

import numpy as np


class STTModelAdapter(abc.ABC):
    """Abstract interface for STT model inference adapters."""

    supports_native_batching: bool = False

    @abc.abstractmethod
    def transcribe(self, audio: Any, **kwargs: Any) -> str:
        """Transcribe PCM audio tensor or file to text."""

    def transcribe_batch(self, chunks: tuple[np.ndarray, ...], **kwargs: Any) -> list[str]:
        """Process one chunk until this adapter proves native batching."""
        if len(chunks) != 1:
            raise ValueError("This model accepts exactly one chunk per inference call")
        return [self.transcribe(chunks[0], **kwargs)]


def safe_resolve_device(device: str | None = None) -> str:
    import os
    target = (device or os.getenv("DEVICE", "cpu")).lower().strip()
    if target.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            _ = torch.cuda.device_count()
        except Exception:
            return "cpu"
    return target

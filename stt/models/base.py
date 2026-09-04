from __future__ import annotations

import abc
from typing import Any


class STTModelAdapter(abc.ABC):
    """Abstract interface for STT model inference adapters."""

    @abc.abstractmethod
    def transcribe(self, audio: Any, **kwargs: Any) -> str:
        """Transcribe PCM audio tensor or file to text."""


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

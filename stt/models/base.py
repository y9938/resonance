from __future__ import annotations

import abc
from typing import Any


class STTModelAdapter(abc.ABC):
    """Abstract interface for STT model inference adapters."""

    @abc.abstractmethod
    def transcribe(self, audio: Any, **kwargs: Any) -> str:
        """Transcribe PCM audio tensor or file to text."""

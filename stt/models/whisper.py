from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from stt.models.base import STTModelAdapter

log = logging.getLogger("resonance.server")


class WhisperAdapter(STTModelAdapter):
    """Wraps WhisperModel to match GigaAM's transcribe(path) -> str interface."""

    def __init__(self, model: Any, beam_size: int = 5) -> None:
        self._model = model
        self._beam_size = beam_size

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> str:
        segments, _ = self._model.transcribe(audio, beam_size=self._beam_size)
        return " ".join(seg.text.strip() for seg in segments).strip()


def load_whisper(device: str | None = None) -> WhisperAdapter:
    log.info("Loading STT model (Distil-Whisper-v3)...")
    from faster_whisper import WhisperModel

    target_device = device or os.getenv("DEVICE", "cpu")
    if target_device.startswith("cuda"):
        ct2_device = "cuda"
        device_index = 0
        if ":" in target_device:
            try:
                device_index = int(target_device.split(":")[1])
            except ValueError:
                pass
        compute_type = "float16"
    else:
        ct2_device = "cpu"
        device_index = 0
        compute_type = "int8"

    kwargs: dict[str, Any] = {
        "device": ct2_device,
        "compute_type": compute_type,
    }
    if device_index > 0:
        kwargs["device_index"] = device_index

    model = WhisperModel(
        "Systran/faster-distil-whisper-large-v3",
        **kwargs,
    )
    log.info(f"Whisper model loaded: device={ct2_device}, device_index={device_index}, compute_type={compute_type}")
    return WhisperAdapter(model)

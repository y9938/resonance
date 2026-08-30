from __future__ import annotations

import logging
import os
from typing import Any

from stt.models.base import STTModelAdapter

log = logging.getLogger("resonance.server")


class GigaAMAdapter(STTModelAdapter):
    """Wraps GigaAM for True Zero-I/O in-RAM tensor inference without disk operations."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(self, audio: Any, **kwargs: Any) -> str:
        import numpy as np
        import torch

        # Domain Invariant: True Zero-I/O RAM pipeline bypassing file creation and wrapper threshold
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).to(self._model._device).to(self._model._dtype)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            length = torch.tensor([wav.shape[-1]], device=self._model._device)
        else:
            wav, length = self._model.prepare_wav(str(audio))

        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(wav, length)
            text, _ = self._model._decode(encoded, encoded_len, length, False)[0]
        return text


def load_gigaam(device: str | None = None) -> GigaAMAdapter:
    log.info("Loading STT model (GigaAM-v3)...")
    import gigaam

    target_device = device or os.getenv("DEVICE", "cpu")
    model = gigaam.load_model("v3_e2e_ctc", device=target_device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"STT model loaded: {params:.1f}M parameters")
    return GigaAMAdapter(model)

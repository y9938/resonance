from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from stt.models.base import STTModelAdapter

log = logging.getLogger("resonance.server")


class GigaAMAdapter(STTModelAdapter):
    """Runs GigaAM inference on in-memory PCM without creating media files."""

    supports_native_batching = True

    @classmethod
    def supported_languages(cls) -> frozenset[str]:
        return frozenset({"ru"})

    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(self, audio: Any, *, language: str | None = None, **kwargs: Any) -> str:
        if isinstance(audio, np.ndarray):
            return self.transcribe_batch((audio,))[0]

        wav, length = self._model.prepare_wav(str(audio))
        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(wav, length)
            return self._model._decode(encoded, encoded_len, length, False)[0][0]

    def transcribe_batch(self, chunks: tuple[np.ndarray, ...], *, language: str | None = None, **kwargs: Any) -> list[str]:
        if not chunks:
            raise ValueError("Inference batch must not be empty")
        if any(chunk.ndim != 1 or not len(chunk) for chunk in chunks):
            raise ValueError("GigaAM requires nonempty mono PCM chunks")

        device, dtype = self._model._device, self._model._dtype
        waveforms = [torch.from_numpy(chunk).to(device=device, dtype=dtype) for chunk in chunks]
        lengths = torch.tensor([len(chunk) for chunk in chunks], device=device)
        padded = pad_sequence(waveforms, batch_first=True)
        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(padded, lengths)
            decoded = self._model._decode(encoded, encoded_len, lengths, False)
        return [text for text, _ in decoded]


def load_gigaam(device: str | None = None) -> GigaAMAdapter:
    log.info("Loading STT model (GigaAM-v3)...")
    import gigaam

    from stt.models.base import safe_resolve_device

    target_device = safe_resolve_device(device)
    cache_dir = os.getenv("GIGAAM_CACHE_DIR")
    download_root = os.path.expanduser(cache_dir) if cache_dir else None
    model = gigaam.load_model(
        "v3_e2e_ctc", device=target_device, download_root=download_root
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"STT model loaded: {params:.1f}M parameters")
    return GigaAMAdapter(model)

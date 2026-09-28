from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np

from stt.models.base import STTModelAdapter, safe_resolve_device

log = logging.getLogger("resonance.server")
WHISPER_NUM_LANGUAGES = 100


@lru_cache(maxsize=1)
def _turbo_languages() -> frozenset[str]:
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(multilingual=True, num_languages=WHISPER_NUM_LANGUAGES)
    return frozenset(tokenizer.all_language_codes)


class WhisperAdapter(STTModelAdapter):
    """Transcribe Resonance PCM chunks with the official Whisper checkpoint."""

    @classmethod
    def supported_languages(cls) -> frozenset[str]:
        return _turbo_languages()

    def __init__(self, model: Any, beam_size: int = 5) -> None:
        self._model = model
        self._beam_size = beam_size

    def transcribe(self, audio: np.ndarray, *, language: str | None = None, **kwargs: Any) -> str:
        if not isinstance(language, str) or language not in self.supported_languages():
            raise ValueError("Whisper requires an explicit language")
        return self._decode(audio, language=language).text.strip()

    def transcribe_and_detect_language(self, audio: np.ndarray) -> tuple[str, str]:
        result = self._decode(audio, language=None)
        return result.text.strip(), result.language

    def _decode(self, audio: np.ndarray, *, language: str | None) -> Any:
        import torch
        import whisper

        pcm = np.asarray(audio, dtype=np.float32)
        if pcm.ndim != 1 or not len(pcm):
            raise ValueError("Whisper requires nonempty mono PCM")
        if len(pcm) > whisper.audio.N_SAMPLES:
            raise ValueError("Whisper chunk exceeds its audio context")

        padded = whisper.pad_or_trim(pcm)
        mel = whisper.log_mel_spectrogram(padded, n_mels=self._model.dims.n_mels).to(self._model.device)
        options = whisper.DecodingOptions(
            task="transcribe", language=language, beam_size=self._beam_size,
            without_timestamps=False, fp16=self._model.device.type == "cuda",
        )
        with torch.inference_mode():
            return whisper.decode(self._model, mel, options)


def load_whisper(device: str | None = None) -> WhisperAdapter:
    log.info("Loading STT model (Whisper Turbo)...")
    import whisper

    target_device = safe_resolve_device(device or os.getenv("DEVICE", "cpu"))
    model = whisper.load_model("turbo", device=target_device).eval()
    if model.num_languages != WHISPER_NUM_LANGUAGES:
        raise RuntimeError("Whisper Turbo language vocabulary does not match the configured checkpoint")
    log.info("Whisper Turbo loaded: device=%s", model.device)
    return WhisperAdapter(model)

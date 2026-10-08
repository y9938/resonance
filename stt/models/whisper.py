from __future__ import annotations

import logging
import os
import time
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

    # TODO: Revisit native batching after upgrading openai-whisper. In 20250625,
    # batched beam search expands tokens by n_group but not audio_features.
    # Keep beam=5; validate text quality, result order, and offline B=2 latency
    # before declaring batching support or changing the production B=1 policy.

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
    started = time.monotonic()
    log.info("Loading STT model (Whisper Turbo)...")
    import whisper

    target_device = safe_resolve_device(device or os.getenv("DEVICE", "cpu"))
    model = whisper.load_model("turbo", device=target_device).eval()
    if model.num_languages != WHISPER_NUM_LANGUAGES:
        raise RuntimeError("Whisper Turbo language vocabulary does not match the configured checkpoint")
    log.info("Whisper Turbo loaded: device=%s; %.1fs", model.device, time.monotonic() - started)
    return WhisperAdapter(model)

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from stt.models.base import STTModelAdapter
from stt.models.gigaam import GigaAMAdapter, load_gigaam
from stt.models.granite import GraniteAdapter, load_granite
from stt.models.whisper import WhisperAdapter, load_whisper

log = logging.getLogger("resonance.server")


def load_silero_tts(device: str | None = None) -> Any:
    started = time.monotonic()
    log.info("Loading TTS model (Silero v5_cis_base)...")
    import torch

    target_device = device or os.getenv("DEVICE", "cpu")
    model, _ = torch.hub.load(
        "snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_cis_base",
        trust_repo=True,
    )
    model.to(torch.device(target_device))
    log.info("Silero TTS loaded: device=%s; %.1fs", target_device, time.monotonic() - started)
    return model


class ModelManager:
    """Lazy model loader. Assumes: single-threaded init, thread-safe after."""

    def __init__(
        self,
        gigaam_loader: Callable[[], GigaAMAdapter] | None = None,
        whisper_loader: Callable[[], WhisperAdapter] | None = None,
        granite_loader: Callable[[], GraniteAdapter] | None = None,
        tts_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._gigaam_loader = gigaam_loader
        self._whisper_loader = whisper_loader
        self._granite_loader = granite_loader
        self._tts_loader = tts_loader

        self._stt_gigaam: GigaAMAdapter | None = None
        self._stt_whisper: WhisperAdapter | None = None
        self._stt_granite: GraniteAdapter | None = None
        self._tts: Any | None = None
        self._lock = threading.Lock()

    @property
    def stt_gigaam_loaded(self) -> bool:
        return self._stt_gigaam is not None

    @property
    def stt_whisper_loaded(self) -> bool:
        return self._stt_whisper is not None

    @property
    def stt_granite_loaded(self) -> bool:
        return self._stt_granite is not None

    @property
    def tts_loaded(self) -> bool:
        return self._tts is not None

    def stt_gigaam(self) -> GigaAMAdapter:
        with self._lock:
            if self._stt_gigaam is None:
                loader = self._gigaam_loader or load_gigaam
                self._stt_gigaam = loader()
            return self._stt_gigaam

    def stt_whisper(self) -> WhisperAdapter:
        with self._lock:
            if self._stt_whisper is None:
                loader = self._whisper_loader or load_whisper
                self._stt_whisper = loader()
            return self._stt_whisper

    def stt_granite(self) -> GraniteAdapter:
        with self._lock:
            if self._stt_granite is None:
                loader = self._granite_loader or load_granite
                self._stt_granite = loader()
            return self._stt_granite

    def tts(self) -> Any:
        with self._lock:
            if self._tts is None:
                loader = self._tts_loader or load_silero_tts
                self._tts = loader()
            return self._tts

    def get_stt_model(self, model_name: str) -> STTModelAdapter:
        """Resolve STT adapter by model identifier."""
        name = model_name.lower().strip()
        if name == "granite":
            return self.stt_granite()
        elif name == "whisper":
            return self.stt_whisper()
        elif name == "gigaam":
            return self.stt_gigaam()
        raise ValueError(f"Unknown STT model: {model_name}")


STT_MODEL_TYPES = {"gigaam": GigaAMAdapter, "whisper": WhisperAdapter, "granite": GraniteAdapter}


def resolve_stt_model(language: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Validate and resolve (language, model_name) for STT routing."""
    norm_model = model.strip().lower() if model is not None else None
    resolved_lang = language.strip().lower() if language is not None else None
    if norm_model is not None and norm_model not in STT_MODEL_TYPES:
        raise ValueError(f"Unsupported model: {model}")
    if resolved_lang is None:
        if norm_model is None:
            return "ru", "gigaam"
        languages = STT_MODEL_TYPES[norm_model].supported_languages()
        if len(languages) != 1:
            raise ValueError(f"Language is required for multilingual model: {norm_model}")
        return next(iter(languages)), norm_model
    if resolved_lang == "auto":
        raise ValueError("Automatic language detection is not supported")
    if norm_model is None:
        if resolved_lang == "ru":
            return "ru", "gigaam"
        if resolved_lang in WhisperAdapter.supported_languages():
            return resolved_lang, "whisper"
        raise ValueError(f"Unsupported language: {resolved_lang}")
    if resolved_lang not in STT_MODEL_TYPES[norm_model].supported_languages():
        raise ValueError(f"Model {norm_model} does not support language: {resolved_lang}")
    return resolved_lang, norm_model

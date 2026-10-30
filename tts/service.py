from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from fastapi import HTTPException
from scipy.io import wavfile

from core.process_metrics import format_process_resources, process_snapshot


@dataclass(frozen=True)
class TtsVoice:
    voice_id: str
    language: str
    backend_id: str


@dataclass(frozen=True)
class TtsLanguage:
    language_id: str
    default_voice_id: str
    voice_ids: tuple[str, ...]


@dataclass(frozen=True)
class KokoroVoiceSpec:
    voice_id: str
    lang_code: str


@dataclass
class TtsSynthesisResult:
    audio: torch.Tensor
    sample_rate: int
    chunks: int


class TtsBackend:
    backend_id: str = ""
    name: str = ""

    @property
    def loaded(self) -> bool:
        raise NotImplementedError

    @property
    def loading(self) -> bool:
        return False

    def estimate_chunks(self, text: str) -> int:
        raise NotImplementedError

    def synthesize(self, text: str, voice_id: str) -> TtsSynthesisResult:
        raise NotImplementedError


def clean_tts_text(text: str) -> str:
    text = text.removeprefix("\ufeff")

    lines = [ln for ln in text.split("\n") if not re.search(r"https?://", ln)]
    text = "\n".join(lines)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    return text.strip()


def _split_long_token(token: str, max_len: int) -> list[str]:
    return [token[i : i + max_len] for i in range(0, len(token), max_len)]


def _split_by_words(text: str, max_len: int) -> list[str]:
    words = text.split()
    result: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_len:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_long_token(word, max_len))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= max_len:
            current = current + " " + word
        else:
            result.append(current)
            current = word
    if current:
        result.append(current)
    return result


def _split_long_sentence(sent: str, max_len: int) -> list[str]:
    clauses = re.split(r"(?<=[,;---])\s+", sent)
    result: list[str] = []
    current = ""

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_len:
            if not current:
                current = clause
            elif len(current) + 1 + len(clause) <= max_len:
                current = current + " " + clause
            else:
                result.append(current)
                current = clause
        else:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_by_words(clause, max_len))
    if current:
        result.append(current)
    return result


def split_tts_text(text: str, max_len: int) -> list[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def append_text(chunk_text: str) -> None:
        nonlocal current
        if not current:
            current = chunk_text
        elif len(current) + 1 + len(chunk_text) <= max_len:
            current = current + " " + chunk_text
        else:
            flush()
            current = chunk_text

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            append_text(sent)
            continue

        flush()
        for chunk in _split_long_sentence(sent, max_len):
            if len(chunk) <= max_len:
                append_text(chunk)
            else:
                flush()
                chunks.extend(_split_long_token(chunk, max_len))
    flush()

    return chunks if chunks else [text[:max_len]]


class SileroRuTtsBackend(TtsBackend):
    backend_id = "silero_ru"
    name = "Silero v5_cis_base"

    def __init__(
        self,
        *,
        get_model: Callable[[], Any],
        is_loaded: Callable[[], bool],
        is_loading: Callable[[], bool] | None = None,
        sample_rate: int,
        max_chars: int,
    ) -> None:
        self._get_model = get_model
        self._is_loaded = is_loaded
        self._is_loading = is_loading or (lambda: False)
        self._sample_rate = sample_rate
        self._max_chars = max_chars

    @property
    def loaded(self) -> bool:
        return self._is_loaded()

    @property
    def loading(self) -> bool:
        return self._is_loading()

    def estimate_chunks(self, text: str) -> int:
        clean = clean_tts_text(text)
        return len(split_tts_text(clean, self._max_chars))

    def synthesize(self, text: str, voice_id: str) -> TtsSynthesisResult:
        model = self._get_model()
        clean = clean_tts_text(text)
        chunks = split_tts_text(clean, self._max_chars)
        audio_parts: list[torch.Tensor] = []

        for chunk_text in chunks:
            audio = model.apply_tts(
                text=chunk_text,
                speaker=voice_id,
                sample_rate=self._sample_rate,
            )
            audio_parts.append(audio.detach().cpu())

        if not audio_parts:
            raise RuntimeError("All TTS chunks failed")

        return TtsSynthesisResult(
            audio=torch.cat(audio_parts, dim=0),
            sample_rate=self._sample_rate,
            chunks=len(chunks),
        )


class KokoroEnTtsBackend(TtsBackend):
    backend_id = "kokoro_en"
    name = "Kokoro English"
    sample_rate = 24000
    _voice_specs: ClassVar[dict[str, KokoroVoiceSpec]] = {
        "af_heart": KokoroVoiceSpec(voice_id="af_heart", lang_code="a"),
        "af_alloy": KokoroVoiceSpec(voice_id="af_alloy", lang_code="a"),
        "af_aoede": KokoroVoiceSpec(voice_id="af_aoede", lang_code="a"),
        "af_bella": KokoroVoiceSpec(voice_id="af_bella", lang_code="a"),
        "af_jessica": KokoroVoiceSpec(voice_id="af_jessica", lang_code="a"),
        "af_kore": KokoroVoiceSpec(voice_id="af_kore", lang_code="a"),
        "af_nicole": KokoroVoiceSpec(voice_id="af_nicole", lang_code="a"),
        "af_nova": KokoroVoiceSpec(voice_id="af_nova", lang_code="a"),
        "af_river": KokoroVoiceSpec(voice_id="af_river", lang_code="a"),
        "af_sarah": KokoroVoiceSpec(voice_id="af_sarah", lang_code="a"),
        "af_sky": KokoroVoiceSpec(voice_id="af_sky", lang_code="a"),
        "am_adam": KokoroVoiceSpec(voice_id="am_adam", lang_code="a"),
        "am_echo": KokoroVoiceSpec(voice_id="am_echo", lang_code="a"),
        "am_eric": KokoroVoiceSpec(voice_id="am_eric", lang_code="a"),
        "am_fenrir": KokoroVoiceSpec(voice_id="am_fenrir", lang_code="a"),
        "am_liam": KokoroVoiceSpec(voice_id="am_liam", lang_code="a"),
        "am_michael": KokoroVoiceSpec(voice_id="am_michael", lang_code="a"),
        "am_onyx": KokoroVoiceSpec(voice_id="am_onyx", lang_code="a"),
        "am_puck": KokoroVoiceSpec(voice_id="am_puck", lang_code="a"),
        "am_santa": KokoroVoiceSpec(voice_id="am_santa", lang_code="a"),
        "bf_alice": KokoroVoiceSpec(voice_id="bf_alice", lang_code="b"),
        "bf_emma": KokoroVoiceSpec(voice_id="bf_emma", lang_code="b"),
        "bf_isabella": KokoroVoiceSpec(voice_id="bf_isabella", lang_code="b"),
        "bf_lily": KokoroVoiceSpec(voice_id="bf_lily", lang_code="b"),
        "bm_daniel": KokoroVoiceSpec(voice_id="bm_daniel", lang_code="b"),
        "bm_fable": KokoroVoiceSpec(voice_id="bm_fable", lang_code="b"),
        "bm_george": KokoroVoiceSpec(voice_id="bm_george", lang_code="b"),
        "bm_lewis": KokoroVoiceSpec(voice_id="bm_lewis", lang_code="b"),
    }

    def __init__(self, *, log: Any) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._model: Any | None = None
        self._loading = False
        self._pipelines: dict[str, Any] = {}
        self._log = log

    @property
    def loaded(self) -> bool:
        with self._state_lock:
            return self._model is not None

    @property
    def loading(self) -> bool:
        with self._state_lock:
            return self._loading

    def estimate_chunks(self, text: str) -> int:
        return 1

    def synthesize(self, text: str, voice_id: str) -> TtsSynthesisResult:
        with self._lock:
            spec = self._get_voice_spec(voice_id)
            pipeline = self._get_pipeline(spec.lang_code)
            audio_parts: list[torch.Tensor] = []
            for result in pipeline(clean_tts_text(text), voice=spec.voice_id):
                if getattr(result, "audio", None) is None:
                    continue
                audio_parts.append(torch.as_tensor(result.audio).detach().cpu())
        if not audio_parts:
            raise RuntimeError("Kokoro returned no audio chunks")
        return TtsSynthesisResult(
            audio=torch.cat(audio_parts, dim=-1),
            sample_rate=self.sample_rate,
            chunks=len(audio_parts),
        )

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from kokoro import KModel
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro dependency is not installed. Add 'kokoro' to the environment."
                ) from exc
            started = time.monotonic()
            resources_before = process_snapshot() if self._log.isEnabledFor(logging.DEBUG) else None
            device = os.getenv("DEVICE", "cpu")
            self._log.info(f"Loading TTS model (Kokoro English) on {device}...")
            with self._state_lock:
                self._loading = True
            try:
                model = KModel(repo_id="hexgrad/Kokoro-82M").to(torch.device(device)).eval()
            except BaseException:
                with self._state_lock:
                    self._loading = False
                if resources_before is not None:
                    resources_after = process_snapshot()
                    if resources_after is not None:
                        self._log.debug(
                            "Model load process resources: scope=process observed_during_model_id=kokoro_en "
                            "result=failed elapsed=%.1fs %s",
                            time.monotonic() - started,
                            format_process_resources(resources_before, resources_after),
                        )
                raise
            with self._state_lock:
                self._model = model
                self._loading = False
            self._log.info("Kokoro English loaded: device=%s; %.1fs", device, time.monotonic() - started)
            if resources_before is not None:
                resources_after = process_snapshot()
                if resources_after is not None:
                    self._log.debug(
                        "Model load process resources: scope=process observed_during_model_id=kokoro_en result=ready elapsed=%.1fs %s",
                        time.monotonic() - started,
                        format_process_resources(resources_before, resources_after),
                    )
        return self._model

    def _get_pipeline(self, lang_code: str) -> Any:
        if lang_code not in self._pipelines:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro dependency is not installed. Add 'kokoro' to the environment."
                ) from exc
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                model=self._get_model(),
                repo_id="hexgrad/Kokoro-82M",
            )
        return self._pipelines[lang_code]

    def _get_voice_spec(self, voice_id: str) -> KokoroVoiceSpec:
        spec = self._voice_specs.get(voice_id)
        if spec is None:
            raise RuntimeError(f"Unsupported Kokoro voice: {voice_id}")
        return spec


class TtsService:
    def __init__(
        self,
        *,
        config: Any,
        get_model: Callable[[], Any],
        is_model_loaded: Callable[[], bool],
        is_model_loading: Callable[[], bool],
        log: Any,
        output_dir: Path,
    ) -> None:
        self._config = config
        self._log = log
        self.output_dir = output_dir
        self.backends: dict[str, TtsBackend] = {
            SileroRuTtsBackend.backend_id: SileroRuTtsBackend(
                get_model=get_model,
                is_loaded=is_model_loaded,
                is_loading=is_model_loading,
                sample_rate=config.TTS_SR,
                max_chars=config.TTS_MAX_CHARS,
            ),
            KokoroEnTtsBackend.backend_id: KokoroEnTtsBackend(log=log),
        }
        self.voices = self._build_tts_voice_catalog()
        self.languages = self._build_tts_language_catalog()

    def _build_tts_voice_catalog(self) -> dict[str, TtsVoice]:
        ru_voices = (
            "ru_alexandr",
            "ru_alfia",
            "ru_alfia2",
            "ru_bogdan",
            "ru_dmitriy",
            "ru_ekaterina",
            "ru_vika",
            "ru_gamat",
            "ru_igor",
            "ru_karina",
            "ru_kejilgan",
            "ru_kermen",
            "ru_marat",
            "ru_miyau",
            "ru_nurgul",
            "ru_oksana",
            "ru_onaoy",
            "ru_ramilia",
            "ru_roman",
            "ru_safarhuja",
            "ru_saida",
            "ru_sibday",
            "ru_zara",
            "ru_zhadyra",
            "ru_zhazira",
            "ru_zinaida",
            "ru_eduard",
        )
        en_voices = tuple(KokoroEnTtsBackend._voice_specs)
        catalog = {
            voice_id: TtsVoice(
                voice_id=voice_id,
                language="ru",
                backend_id=SileroRuTtsBackend.backend_id,
            )
            for voice_id in ru_voices
        }
        for voice_id in en_voices:
            catalog[voice_id] = TtsVoice(
                voice_id=voice_id,
                language="en",
                backend_id=KokoroEnTtsBackend.backend_id,
            )
        return catalog

    def _build_tts_language_catalog(self) -> dict[str, TtsLanguage]:
        return {
            "ru": TtsLanguage(
                language_id="ru",
                default_voice_id="ru_roman",
                voice_ids=tuple(
                    voice_id
                    for voice_id, voice in self.voices.items()
                    if voice.language == "ru"
                ),
            ),
            "en": TtsLanguage(
                language_id="en",
                default_voice_id="af_heart",
                voice_ids=tuple(
                    voice_id
                    for voice_id, voice in self.voices.items()
                    if voice.language == "en"
                ),
            ),
        }

    def list_voice_ids(self) -> list[str]:
        return list(self.voices)

    def list_languages(self) -> list[str]:
        return list(self.languages)

    def default_voice_id(self) -> str:
        configured = self._config.TTS_VOICE_ID
        if configured in self.voices:
            return configured
        fallback = next(iter(self.voices))
        self._log.warning(
            f"Invalid default TTS voice_id '{configured}', using '{fallback}'"
        )
        return fallback

    def get_voice_or_400(self, voice_id: str) -> TtsVoice:
        voice = self.voices.get(voice_id)
        if voice is None:
            raise HTTPException(400, f"Invalid voice_id. Use: {self.list_voice_ids()}")
        return voice

    def get_language_or_400(self, language: str) -> TtsLanguage:
        entry = self.languages.get(language)
        if entry is None:
            raise HTTPException(400, f"Invalid language. Use: {self.list_languages()}")
        return entry

    def get_backend_for_voice(self, voice_id: str) -> tuple[TtsVoice, TtsBackend]:
        voice = self.get_voice_or_400(voice_id)
        return voice, self.backends[voice.backend_id]

    def validate_language_voice(self, language: str, voice_id: str) -> TtsVoice:
        self.get_language_or_400(language)
        voice = self.get_voice_or_400(voice_id)
        if voice.language != language:
            raise HTTPException(
                400,
                f"voice_id '{voice_id}' does not belong to language '{language}'",
            )
        return voice

    def default_language(self) -> str:
        return self.get_voice_or_400(self.default_voice_id()).language

    def serialize_catalog(self) -> dict[str, Any]:
        return {
            "default_language": self.default_language(),
            "languages": [
                {
                    "id": language.language_id,
                    "default_voice_id": language.default_voice_id,
                    "voices": [
                        {
                            "id": voice_id,
                            "backend_id": self.voices[voice_id].backend_id,
                        }
                        for voice_id in language.voice_ids
                    ],
                }
                for language in self.languages.values()
            ],
        }

    def run_job(
        self,
        *,
        job_id: str,
        text: str,
        voice_id: str,
        jobs: Any,
        filename: str | None = None,
    ) -> None:
        start_time = time.monotonic()
        resources_start = process_snapshot() if self._log.isEnabledFor(logging.DEBUG) else None
        outcome = "running"

        def cancel_requested() -> bool:
            nonlocal outcome
            if jobs.is_cancelled(job_id):
                outcome = "cancelled"
                elapsed = time.monotonic() - start_time
                self._log.info(f"TTS cancelled: job_id={job_id}; {elapsed:.2f}s")
                jobs.update_event(job_id, "cancelled", {})
                return True
            return False

        try:
            _, backend = self.get_backend_for_voice(voice_id)
            estimated_chunks = backend.estimate_chunks(text)

            jobs.update_event(job_id, "start", {"total": estimated_chunks})
            result = backend.synthesize(text, voice_id)

            if cancel_requested():
                return
            jobs.update_event(
                job_id,
                "progress",
                {"current": result.chunks, "total": result.chunks},
            )

            full_audio = result.audio
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{secrets.token_urlsafe(24)}.wav"
            output_path = str(self.output_dir / out_name)
            wavfile.write(
                output_path,
                result.sample_rate,
                (full_audio.detach().cpu().numpy() * 32767).astype("int16"),
            )

            download_url = f"/api/stream/download?p={out_name}"
            if filename:
                download_url += f"&filename={filename}"

            if cancel_requested():
                return
            jobs.update_event(
                job_id,
                "complete",
                {
                    "download_url": download_url,
                    "duration": len(full_audio) / result.sample_rate,
                    "chunks": result.chunks,
                    "filename": filename,
                },
            )
            outcome = "completed"
            elapsed = time.monotonic() - start_time
            self._log.info(f"TTS completed: job_id={job_id}; {elapsed:.2f}s")
        except Exception as exc:
            outcome = "failed"
            elapsed = time.monotonic() - start_time
            self._log.exception(f"TTS failed: job_id={job_id}; {exc} ({elapsed:.2f}s)")
            jobs.update_event(job_id, "error", {"message": str(exc)})
        finally:
            if resources_start is not None:
                resources_end = process_snapshot()
                if resources_end is not None:
                    self._log.debug(
                        "TTS process resources: scope=process observed_during_job_id=%s outcome=%s elapsed=%.1fs %s",
                        job_id, outcome, time.monotonic() - resources_start.monotonic,
                        format_process_resources(resources_start, resources_end),
                    )

    def sweep_stale_files(self, max_age_sec: int) -> None:
        if not self.output_dir.is_dir():
            return
        now = time.time()
        for path in self.output_dir.iterdir():
            if not path.is_file() or path.suffix.lower() != ".wav":
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age > max_age_sec:
                try:
                    path.unlink()
                except OSError:
                    pass

    async def run_file_sweeper(self, interval_sec: int, max_age_sec: int) -> None:
        while True:
            await asyncio.sleep(interval_sec)
            await asyncio.to_thread(self.sweep_stale_files, max_age_sec)

"""Regression tests for internal TTS voice catalog and backend routing."""

import pytest
import torch
from fastapi import HTTPException

import server
from server import Config, get_config, list_models
from tts.service import KokoroEnTtsBackend, SileroRuTtsBackend, TtsSynthesisResult

KOKORO_EN_VOICE_IDS = {
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
}


def test_ru_voice_routes_to_silero_backend() -> None:
    voice, backend = server.tts_service.get_backend_for_voice("ru_roman")

    assert voice.voice_id == "ru_roman"
    assert voice.backend_id == SileroRuTtsBackend.backend_id
    assert backend.backend_id == SileroRuTtsBackend.backend_id


def test_unknown_tts_voice_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        server.tts_service.get_voice_or_400("en_missing")

    assert exc.value.status_code == 400
    assert "Invalid voice_id" in str(exc.value.detail)


def test_invalid_default_tts_voice_falls_back_to_first_catalog_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "TTS_VOICE_ID", "invalid_voice")

    assert server.tts_service.default_voice_id() == server.tts_service.list_voice_ids()[0]


@pytest.mark.asyncio
async def test_public_config_uses_voice_catalog_and_default_resolution() -> None:
    config = await get_config()

    assert config["tts"]["default_language"] == "ru"
    assert [language["id"] for language in config["tts"]["languages"]] == ["ru", "en"]
    assert config["tts"]["languages"][0]["default_voice_id"] == server.tts_service.default_voice_id()


def test_en_voice_routes_to_kokoro_backend() -> None:
    voice, backend = server.tts_service.get_backend_for_voice("af_heart")

    assert voice.voice_id == "af_heart"
    assert voice.backend_id == KokoroEnTtsBackend.backend_id
    assert backend.backend_id == KokoroEnTtsBackend.backend_id


def test_kokoro_backend_exposes_all_documented_english_voices() -> None:
    assert set(KokoroEnTtsBackend._voice_specs) == KOKORO_EN_VOICE_IDS


@pytest.mark.asyncio
async def test_public_models_expose_all_tts_backends() -> None:
    payload = await list_models()

    backends = {row["id"]: row for row in payload["tts_backends"]}
    assert SileroRuTtsBackend.backend_id in backends
    assert KokoroEnTtsBackend.backend_id in backends
    assert payload["tts_catalog"]["languages"][1]["voices"][0]["id"] == "af_heart"


@pytest.mark.asyncio
async def test_public_config_exposes_voice_ids_per_language() -> None:
    config = await get_config()

    languages = {entry["id"]: entry for entry in config["tts"]["languages"]}
    assert {voice["id"] for voice in languages["ru"]["voices"]} >= {"ru_roman", "ru_oksana"}
    assert {voice["id"] for voice in languages["en"]["voices"]} == KOKORO_EN_VOICE_IDS


@pytest.mark.asyncio
async def test_public_config_exposes_backend_id_per_voice() -> None:
    config = await get_config()

    languages = {entry["id"]: entry for entry in config["tts"]["languages"]}
    ru_voices = {voice["id"]: voice for voice in languages["ru"]["voices"]}
    en_voices = {voice["id"]: voice for voice in languages["en"]["voices"]}

    assert ru_voices["ru_roman"]["backend_id"] == SileroRuTtsBackend.backend_id
    assert en_voices["af_heart"]["backend_id"] == KokoroEnTtsBackend.backend_id


def test_validate_tts_language_voice_rejects_voice_from_another_language() -> None:
    with pytest.raises(HTTPException) as exc:
        server.tts_service.validate_language_voice("ru", "af_heart")

    assert exc.value.status_code == 400
    assert "does not belong to language" in str(exc.value.detail)


def test_tts_worker_logs_when_job_is_cancelled_after_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def estimate_chunks(self, text: str) -> int:
            return 1

        def synthesize(self, text: str, voice_id: str) -> TtsSynthesisResult:
            return TtsSynthesisResult(audio=torch.zeros(1), sample_rate=24000, chunks=1)

    class FakeJobs:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def update_event(self, job_id: str, event_type: str, data: dict) -> None:
            self.events.append((event_type, data))

        def is_cancelled(self, job_id: str) -> bool:
            return True

    class FakeLog:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str) -> None:
            self.messages.append(message)

        def error(self, message: str) -> None:
            self.messages.append(message)

    jobs = FakeJobs()
    log = FakeLog()

    monkeypatch.setattr(
        server.tts_service,
        "get_backend_for_voice",
        lambda voice_id: (None, FakeBackend()),
    )
    monkeypatch.setattr(server.tts_service, "_log", log)

    server.tts_service.run_job(
        job_id="job-1",
        text="hello",
        voice_id="bm_george",
        jobs=jobs,
    )

    assert [event for event, _ in jobs.events] == ["start", "cancelled"]
    assert any(message.startswith("TTS cancelled: ") for message in log.messages)

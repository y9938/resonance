"""Regression tests for internal TTS voice catalog and backend routing."""

import pytest
from fastapi import HTTPException

from server import (
    Config,
    KokoroEnTtsBackend,
    SileroRuTtsBackend,
    default_tts_voice_id,
    get_config,
    list_models,
    get_tts_backend_for_voice,
    validate_tts_language_voice,
    get_tts_voice_or_400,
    list_tts_voice_ids,
)

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
    voice, backend = get_tts_backend_for_voice("ru_roman")

    assert voice.voice_id == "ru_roman"
    assert voice.backend_id == SileroRuTtsBackend.backend_id
    assert backend.backend_id == SileroRuTtsBackend.backend_id


def test_unknown_tts_voice_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        get_tts_voice_or_400("en_missing")

    assert exc.value.status_code == 400
    assert "Invalid voice_id" in str(exc.value.detail)


def test_invalid_default_tts_voice_falls_back_to_first_catalog_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "TTS_VOICE_ID", "invalid_voice")

    assert default_tts_voice_id() == list_tts_voice_ids()[0]


@pytest.mark.asyncio
async def test_public_config_uses_voice_catalog_and_default_resolution() -> None:
    config = await get_config()

    assert config["tts"]["default_language"] == "ru"
    assert [language["id"] for language in config["tts"]["languages"]] == ["ru", "en"]
    assert config["tts"]["languages"][0]["default_voice_id"] == default_tts_voice_id()


def test_en_voice_routes_to_kokoro_backend() -> None:
    voice, backend = get_tts_backend_for_voice("af_heart")

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
        validate_tts_language_voice("ru", "af_heart")

    assert exc.value.status_code == 400
    assert "does not belong to language" in str(exc.value.detail)

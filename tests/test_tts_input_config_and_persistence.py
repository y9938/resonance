"""Regression tests for TTS input config wiring and draft persistence."""

from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "public" / "index.html"


def test_tts_ui_does_not_hardcode_50000_limit() -> None:
    source = INDEX_HTML.read_text()

    assert "len > 50000" not in source
    assert "text.length > 50000" not in source
    assert "max 50000 characters" not in source
    assert "макс. 50 000 символов" not in source


def test_tts_file_load_uses_same_draft_persistence_path_as_text_input() -> None:
    source = INDEX_HTML.read_text()

    assert "function syncTtsDraftStorage()" in source
    assert "reader.onload = (e) => {" in source
    assert "els.ttsInput.value = e.target.result;" in source
    assert "syncTtsDraftStorage();" in source


def test_tts_limit_logic_uses_runtime_config() -> None:
    source = INDEX_HTML.read_text()

    assert "function getTtsInputLimit()" in source
    assert "const inputLimit = getTtsInputLimit();" in source
    assert "if (inputLimit > 0 && text.length > inputLimit)" in source


def test_tts_ui_uses_language_and_voice_selects_instead_of_single_speaker() -> None:
    source = INDEX_HTML.read_text()

    assert "id=\"ttsLanguage\"" in source
    assert "id=\"ttsVoice\"" in source
    assert "id=\"ttsSpeaker\"" not in source


def test_tts_request_sends_language_and_voice_id() -> None:
    source = INDEX_HTML.read_text()

    assert "language=' + encodeURIComponent(els.ttsLanguage.value)" in source
    assert "voice_id=' + encodeURIComponent(els.ttsVoice.value)" in source
    assert "&speaker=" not in source


def test_tts_config_consumes_structured_catalog() -> None:
    source = INDEX_HTML.read_text()

    assert "CONFIG.tts.languages" in source
    assert "TTS_LANGUAGE_KEY" in source
    assert "TTS_VOICE_BY_LANGUAGE_KEY" in source
    assert "CONFIG.tts_speakers" not in source


def test_tts_ui_uses_voice_groups_instead_of_flat_voice_map() -> None:
    source = INDEX_HTML.read_text()

    assert "voiceGroups" in source
    assert "entry.tts.voiceGroups" in source
    assert "entry.tts.voices" not in source


def test_tts_voice_persistence_is_scoped_per_language() -> None:
    source = INDEX_HTML.read_text()

    assert "TTS_VOICE_BY_LANGUAGE_KEY" in source
    assert "function getSavedTtsVoiceByLanguage()" in source
    assert "function persistTtsVoiceSelection(languageId, voiceId)" in source
    assert "savedVoices[els.ttsLanguage.value]" in source
    assert "localStorage.setItem(TTS_VOICE_KEY" not in source

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

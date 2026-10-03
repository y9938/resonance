from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stt.models.gigaam import GigaAMAdapter, load_gigaam
from stt.models.granite import GraniteAdapter
from stt.models.manager import ModelManager
from stt.models.whisper import WhisperAdapter


@pytest.mark.parametrize(
    ("cache_dir", "expected"),
    [
        (None, None),
        ("", None),
        ("/models/gigaam", "/models/gigaam"),
        ("~/models/gigaam", str(Path.home() / "models/gigaam")),
        ("models/gigaam", "models/gigaam"),
    ],
)
def test_gigaam_cache_override(monkeypatch, cache_dir, expected):
    import gigaam

    if cache_dir is None:
        monkeypatch.delenv("GIGAAM_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("GIGAAM_CACHE_DIR", cache_dir)
    upstream_loader = MagicMock()
    monkeypatch.setattr(gigaam, "load_model", upstream_loader)

    load_gigaam("cpu")

    assert upstream_loader.call_args.kwargs["download_root"] == expected


def test_model_manager_get_stt_model():
    mock_gigaam = MagicMock(spec=GigaAMAdapter)
    mock_whisper = MagicMock(spec=WhisperAdapter)
    mock_granite = MagicMock(spec=GraniteAdapter)

    mgr = ModelManager(
        gigaam_loader=lambda: mock_gigaam,
        whisper_loader=lambda: mock_whisper,
        granite_loader=lambda: mock_granite,
    )

    assert mgr.get_stt_model("gigaam") is mock_gigaam
    assert mgr.get_stt_model("whisper") is mock_whisper
    assert mgr.get_stt_model("granite") is mock_granite
    assert mgr.get_stt_model("  WHISPER  ") is mock_whisper


def test_model_manager_unknown_model_raises():
    mgr = ModelManager()
    with pytest.raises(ValueError):
        mgr.get_stt_model("non_existent")

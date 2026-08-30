from unittest.mock import MagicMock

import pytest

from stt.models.base import STTModelAdapter
from stt.models.gigaam import GigaAMAdapter
from stt.models.granite import GraniteAdapter
from stt.models.manager import ModelManager
from stt.models.whisper import WhisperAdapter


def test_stt_model_adapter_inheritance():
    assert issubclass(GigaAMAdapter, STTModelAdapter)
    assert issubclass(WhisperAdapter, STTModelAdapter)
    assert issubclass(GraniteAdapter, STTModelAdapter)


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
    with pytest.raises(ValueError, match="Unknown STT model: non_existent"):
        mgr.get_stt_model("non_existent")

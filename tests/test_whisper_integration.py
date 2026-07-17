import os
from unittest.mock import patch, MagicMock
import pytest
from server import _load_whisper, WhisperAdapter, ModelManager, list_models


def test_load_whisper_cuda():
    with patch.dict(os.environ, {"DEVICE": "cuda"}):
        with patch("faster_whisper.WhisperModel") as MockModel:
            _load_whisper()
            MockModel.assert_called_once_with(
                "Systran/faster-distil-whisper-large-v3",
                device="cuda",
                compute_type="float16",
            )


def test_load_whisper_cpu():
    with patch.dict(os.environ, {"DEVICE": "cpu"}):
        with patch("faster_whisper.WhisperModel") as MockModel:
            _load_whisper()
            MockModel.assert_called_once_with(
                "Systran/faster-distil-whisper-large-v3",
                device="cpu",
                compute_type="int8",
            )


def test_load_whisper_mps_falls_back_to_cpu():
    """MPS is not supported by CTranslate2; must be remapped to cpu+int8."""
    with patch.dict(os.environ, {"DEVICE": "mps"}):
        with patch("faster_whisper.WhisperModel") as MockModel:
            _load_whisper()
            MockModel.assert_called_once_with(
                "Systran/faster-distil-whisper-large-v3",
                device="cpu",
                compute_type="int8",
            )


def test_whisper_adapter_joins_segments():
    mock_model = MagicMock()
    seg1 = MagicMock()
    seg1.text = "Hello"
    seg2 = MagicMock()
    seg2.text = " world"
    mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())

    adapter = WhisperAdapter(mock_model, beam_size=5)
    result = adapter.transcribe("audio.wav")

    assert result == "Hello world"
    mock_model.transcribe.assert_called_once_with("audio.wav", beam_size=5)


def test_whisper_adapter_empty_segments():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    adapter = WhisperAdapter(mock_model, beam_size=5)
    result = adapter.transcribe("audio.wav")

    assert result == ""


def test_model_manager_stt_whisper_loads_once():
    """stt_whisper() should only call _load_whisper once (lazy singleton)."""
    with patch("server._load_whisper") as mock_load:
        mock_load.return_value = MagicMock()
        mgr = ModelManager()
        assert mgr.stt_whisper_loaded is False

        m1 = mgr.stt_whisper()
        m2 = mgr.stt_whisper()

        assert m1 is m2
        assert mock_load.call_count == 1
        assert mgr.stt_whisper_loaded is True


@pytest.mark.asyncio
async def test_list_models_returns_nested_stt():
    payload = await list_models()
    assert "stt" in payload
    assert "gigaam" in payload["stt"]
    assert "whisper" in payload["stt"]
    assert "name" in payload["stt"]["gigaam"]
    assert "loaded" in payload["stt"]["gigaam"]
    assert payload["stt"]["whisper"]["name"] == "Distil-Whisper-v3"

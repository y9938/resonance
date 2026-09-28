import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from server import list_models
from stt.models import ModelManager, WhisperAdapter, load_whisper


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_load_whisper_uses_requested_torch_device(device):
    with patch.dict(os.environ, {"DEVICE": device}), patch("whisper.load_model") as load_model:
        with patch("stt.models.whisper.safe_resolve_device", return_value=device):
            load_model.return_value.eval.return_value.num_languages = 100
            load_whisper()
        load_model.assert_called_once_with("turbo", device=device)
        load_model.return_value.eval.assert_called_once_with()


def test_whisper_adapter_decodes_pcm_with_model_mel_dimensions_and_english():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dims.n_mels = 128
    mel = MagicMock()
    mel.to.return_value = mel
    result = MagicMock(text=" Hello world ")
    pcm = np.ones(16000, dtype=np.float32)

    with (
        patch("whisper.log_mel_spectrogram", return_value=mel) as spectrogram,
        patch("whisper.decode", return_value=result) as decode,
    ):
        text = WhisperAdapter(model, beam_size=5).transcribe(pcm, language="en")

    assert text == "Hello world"
    assert spectrogram.call_args.kwargs == {"n_mels": 128}
    padded = spectrogram.call_args.args[0]
    assert len(padded) == 30 * 16000
    np.testing.assert_array_equal(padded[:len(pcm)], pcm)
    options = decode.call_args.args[2]
    assert options.language == "en"
    assert options.task == "transcribe"
    assert options.beam_size == 5
    assert options.without_timestamps is False
    assert options.fp16 is False


def test_whisper_adapter_rejects_invalid_pcm_and_auto_language():
    model = MagicMock()
    model.device = torch.device("cpu")
    adapter = WhisperAdapter(model)
    with pytest.raises(ValueError, match="explicit language"):
        adapter.transcribe(np.ones(16000, dtype=np.float32), language="auto")
    with pytest.raises(ValueError, match="nonempty mono"):
        adapter.transcribe(np.empty(0, dtype=np.float32), language="en")
    with pytest.raises(ValueError, match="audio context"):
        adapter.transcribe(np.ones(30 * 16000 + 1, dtype=np.float32), language="en")


def test_whisper_adapter_detects_and_transcribes_first_chunk_once():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dims.n_mels = 128
    pcm = np.ones(16000, dtype=np.float32)
    with patch("whisper.decode", return_value=MagicMock(text=" Hallo ", language="de")) as decode:
        assert WhisperAdapter(model).transcribe_and_detect_language(pcm) == ("Hallo", "de")
    decode.assert_called_once()
    assert decode.call_args.args[2].language is None


def test_model_manager_stt_whisper_loads_once():
    """stt_whisper() should only call _load_whisper once (lazy singleton)."""
    with patch("stt.models.manager.load_whisper") as mock_load:
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
    models = {item["id"]: item for item in payload["stt"]["models"]}
    assert isinstance(models["gigaam"]["loaded"], bool)
    assert models["whisper"]["name"] == "Whisper Turbo"

from unittest.mock import MagicMock

import numpy as np
import torch

from stt.models import GraniteAdapter

def test_granite_adapter_transcribe_standard():
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "Hello world"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")
    dummy_audio = np.zeros((16000,), dtype=np.float32)
    res = adapter.transcribe(dummy_audio, diarization=False)

    assert res == "Hello world"
    mock_processor.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "<|audio|> can you transcribe the speech into a written format?"}],
        tokenize=False,
        add_generation_prompt=True
    )
    mock_processor.assert_called_once()
    mock_model.generate.assert_called_once()


def test_granite_adapter_transcribe_diarization():
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "[Speaker 1]: Hello [Speaker 2]: Hi"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")
    dummy_audio = np.zeros((16000,), dtype=np.float32)
    res = adapter.transcribe(dummy_audio, diarization=True)

    assert res == "[Speaker 1]: Hello [Speaker 2]: Hi"
    mock_processor.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."}],
        tokenize=False,
        add_generation_prompt=True
    )


def test_granite_adapter_stereo_to_mono():
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "Hello"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")
    stereo_audio = np.zeros((2, 16000), dtype=np.float32)
    res = adapter.transcribe(stereo_audio, diarization=False)

    assert res == "Hello"
    mock_processor.assert_called_once()


def test_granite_adapter_transcribe_slices_prompt():
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.side_effect = lambda tokens, **kwargs: "Hello world" if list(tokens) == [3, 4, 5] else "Fail"
    mock_processor.return_value = {
        "input_features": torch.zeros((1, 80, 3000)),
        "input_ids": torch.zeros((1, 2))
    }

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")
    dummy_audio = np.zeros((16000,), dtype=np.float32)
    res = adapter.transcribe(dummy_audio, diarization=False)

    assert res == "Hello world"


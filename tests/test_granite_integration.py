from unittest.mock import MagicMock

import numpy as np
import torch

from stt.models import GraniteAdapter


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

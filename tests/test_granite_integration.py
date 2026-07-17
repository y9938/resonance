import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from server import GraniteAdapter, _load_granite

def test_granite_adapter_transcribe_standard():
    # Mock model and processor
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "Hello world"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")

    # Mock soundfile.read
    dummy_audio = np.zeros((16000,))
    with patch("soundfile.read", return_value=(dummy_audio, 16000)):
        res = adapter.transcribe("dummy.wav", diarization=False)

    assert res == "Hello world"

    # Verify prompt construction
    mock_processor.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "<|audio|> can you transcribe the speech into a written format?"}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Verify input preparation and generation
    mock_processor.assert_called_once()
    mock_model.generate.assert_called_once()


def test_granite_adapter_transcribe_diarization():
    # Mock model and processor
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "[Speaker 1]: Hello [Speaker 2]: Hi"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")

    # Mock soundfile.read
    dummy_audio = np.zeros((16000,))
    with patch("soundfile.read", return_value=(dummy_audio, 16000)):
        res = adapter.transcribe("dummy.wav", diarization=True)

    assert res == "[Speaker 1]: Hello [Speaker 2]: Hi"

    # Verify diarization prompt construction
    mock_processor.tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."}],
        tokenize=False,
        add_generation_prompt=True
    )


def test_granite_adapter_resampling_and_mono():
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_processor = MagicMock()
    mock_processor.tokenizer.apply_chat_template.return_value = "templated prompt"
    mock_processor.tokenizer.decode.return_value = "Hello"
    mock_processor.return_value = {"input_features": torch.zeros((1, 80, 3000))}

    adapter = GraniteAdapter(mock_model, mock_processor, "cpu")

    # Test stereo audio load (2 channels, 44.1kHz)
    stereo_waveform = torch.zeros((2, 44100))
    
    with patch("soundfile.read", return_value=(stereo_waveform.numpy().T, 44100)), \
         patch("torchaudio.transforms.Resample") as mock_resample_cls:
        
        mock_resampler = MagicMock()
        mock_resample_cls.return_value = mock_resampler
        # resampler returns a mock tensor at 16kHz
        mock_resampler.return_value = torch.zeros((2, 16000))
        
        adapter.transcribe("dummy.wav", diarization=False)

        # Check resampling initialization and call
        mock_resample_cls.assert_called_once_with(orig_freq=44100, new_freq=16000)
        assert mock_resampler.call_count == 1
        called_arg = mock_resampler.call_args[0][0]
        assert torch.equal(called_arg, stereo_waveform)


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

    dummy_audio = np.zeros((16000,))
    with patch("soundfile.read", return_value=(dummy_audio, 16000)):
        res = adapter.transcribe("dummy.wav", diarization=False)

    assert res == "Hello world"


from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from stt.models.base import STTModelAdapter

log = logging.getLogger("resonance.server")


class GraniteAdapter(STTModelAdapter):
    """Wraps ibm-granite/granite-speech-4.1-2b-plus model for inference."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def transcribe(self, audio: np.ndarray, diarization: bool = False, **kwargs: Any) -> str:
        import torch

        waveform = torch.from_numpy(audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_input = waveform.squeeze().numpy()

        if diarization:
            prompt = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
        else:
            prompt = "<|audio|> can you transcribe the speech into a written format?"

        chat = [{"role": "user", "content": prompt}]
        prompt_text = self._processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(
            text=prompt_text,
            audio=audio_input,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self._model.dtype)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2000,
            )

        if "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_len:]
        else:
            new_tokens = generated_ids[0]

        transcription = self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return transcription.strip()


def load_granite(device: str | None = None) -> GraniteAdapter:
    log.info("Loading STT model (IBM Granite Speech 4.1 Plus)...")
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    target_device = device or os.getenv("DEVICE", "cpu")
    model_id = "ibm-granite/granite-speech-4.1-2b-plus"

    processor = AutoProcessor.from_pretrained(model_id)

    if target_device.startswith("cuda"):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch_dtype,
    ).to(target_device)

    log.info(f"Granite model loaded: device={target_device}, dtype={torch_dtype}")
    return GraniteAdapter(model, processor, target_device)

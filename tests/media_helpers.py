"""Small encoded-media builders shared by STT tests."""

import io
import wave

import numpy as np

from stt.media import EncodedMedia


def silent_wav(samples: int, *, sample_rate: int = 16000) -> EncodedMedia:
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return EncodedMedia(encoded.getvalue(), "silent.wav")

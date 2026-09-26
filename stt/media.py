"""Seekable encoded input and bounded, sequential PCM frames for batch STT."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

import av
import numpy as np


@dataclass(frozen=True)
class EncodedMedia:
    data: bytes
    filename: str | None = None


def media_duration(media: EncodedMedia) -> float:
    with av.open(io.BytesIO(media.data)) as container:
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio is None:
            raise ValueError("No audio stream found")
        if audio.duration is not None and audio.time_base is not None:
            seconds = float(audio.duration * audio.time_base)
        elif container.duration is not None:
            seconds = container.duration / av.time_base
        else:
            raise ValueError("Audio duration is unavailable")
        if seconds <= 0:
            raise ValueError("Audio duration must be positive")
        return seconds


def iter_media_frames(
    media: EncodedMedia, *, sample_rate: int = 16000, frame_samples: int = 512
) -> Iterator[np.ndarray]:
    """Decode once, resample continuously, and discard incomplete final VAD frame."""
    with av.open(io.BytesIO(media.data)) as container:
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio is None:
            raise ValueError("No audio stream found")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        carry = np.empty(0, dtype=np.float32)

        def frames_from(decoded: av.AudioFrame | None) -> Iterator[np.ndarray]:
            nonlocal carry
            for converted in resampler.resample(decoded):
                samples = converted.to_ndarray().reshape(-1)
                if len(carry):
                    samples = np.concatenate((carry, samples))
                complete = (len(samples) // frame_samples) * frame_samples
                for offset in range(0, complete, frame_samples):
                    yield samples[offset : offset + frame_samples]
                carry = samples[complete:]

        for decoded in container.decode(audio):
            yield from frames_from(decoded)
        yield from frames_from(None)

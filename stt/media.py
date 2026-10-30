"""Seekable encoded input and bounded, sequential PCM frames for batch STT."""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from dataclasses import dataclass

import av
import numpy as np


@dataclass(frozen=True)
class EncodedMedia:
    data: bytes
    filename: str | None = None


@dataclass
class DecodeStats:
    """PCM produced by one resampler pass.

    Includes an over-limit chunk rejected before yield.
    """

    decoded_samples: int = 0
    collect_signal: bool = False
    finite_samples: int = 0
    sum_squares: float = 0.0
    peak_abs: float = 0.0
    nonfinite_samples: int = 0

    def add(self, samples: np.ndarray) -> None:
        self.decoded_samples += len(samples)
        if not self.collect_signal or not len(samples):
            return
        # Bound temporary masks/copies when the input is an hours-long in-memory PCM array.
        block_samples = 1_048_576
        for offset in range(0, len(samples), block_samples):
            block = samples[offset : offset + block_samples]
            finite = np.isfinite(block)
            self.nonfinite_samples += int(len(block) - np.count_nonzero(finite))
            if not np.any(finite):
                continue
            values = block[finite]
            absolute = np.abs(values)
            self.finite_samples += len(values)
            self.sum_squares += float(np.dot(values, values))
            self.peak_abs = max(self.peak_abs, float(np.max(absolute)))

    @property
    def rms(self) -> float:
        if not self.finite_samples:
            return 0.0
        return (self.sum_squares / self.finite_samples) ** 0.5


class DecodedSampleLimitExceeded(Exception):
    """Resampled PCM exceeded the caller's sample budget."""


def media_duration(media: EncodedMedia) -> float | None:
    with av.open(io.BytesIO(media.data)) as container:
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio is None:
            raise ValueError("No audio stream found")
        if audio.duration is not None and audio.time_base is not None:
            seconds = float(audio.duration * audio.time_base)
            if math.isfinite(seconds) and seconds > 0:
                return seconds
        if container.duration is not None:
            seconds = container.duration / av.time_base
            if math.isfinite(seconds) and seconds > 0:
                return seconds
        return None


def iter_media_pcm(
    media: EncodedMedia,
    *,
    sample_rate: int = 16000,
    max_samples: int | None = None,
    stats: DecodeStats | None = None,
) -> Iterator[np.ndarray]:
    """Yield resampled mono PCM, enforcing a limit before each chunk leaves the decoder."""
    if stats is None:
        stats = DecodeStats()
    with av.open(io.BytesIO(media.data)) as container:
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio is None:
            raise ValueError("No audio stream found")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

        def converted_pcm(decoded: av.AudioFrame | None) -> Iterator[np.ndarray]:
            for converted in resampler.resample(decoded):
                samples = converted.to_ndarray().reshape(-1)
                stats.add(samples)
                if max_samples is not None and stats.decoded_samples > max_samples:
                    raise DecodedSampleLimitExceeded()
                yield samples

        for decoded in container.decode(audio):
            yield from converted_pcm(decoded)
        yield from converted_pcm(None)


def iter_media_frames(
    media: EncodedMedia,
    *,
    sample_rate: int = 16000,
    frame_samples: int = 512,
    max_samples: int | None = None,
    stats: DecodeStats | None = None,
) -> Iterator[np.ndarray]:
    """Decode once, resample continuously, and discard incomplete final VAD frame."""
    carry = np.empty(0, dtype=np.float32)
    for samples in iter_media_pcm(
        media, sample_rate=sample_rate, max_samples=max_samples, stats=stats,
    ):
        if len(carry):
            samples = np.concatenate((carry, samples))
        complete = (len(samples) // frame_samples) * frame_samples
        for offset in range(0, complete, frame_samples):
            yield samples[offset : offset + frame_samples]
        carry = samples[complete:]

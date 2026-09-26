"""Bounded Silero sequence inference with state scoped to one batch source."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator
from importlib import metadata

import numpy as np
import onnxruntime as ort

_FRAME_SAMPLES = 512
_CONTEXT_SAMPLES = 64
# 64 keeps lookahead and working memory small; measured throughput matched larger blocks.
_DEFAULT_BLOCK_FRAMES = 64
# Resonance resource bound, not a fundamental limit of the Silero model.
_MAX_BLOCK_FRAMES = 512


class SequenceVADEngine:
    """Share only the immutable ONNX Runtime session between audio streams."""

    def __init__(self) -> None:
        # Resolve packaged weights without importing silero_vad: its package import
        # changes torch's process-wide thread count.
        model = metadata.distribution("silero-vad").locate_file(
            "silero_vad/data/silero_vad_16k_sequence.onnx"
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model), providers=["CPUExecutionProvider"], sess_options=options
        )

    def new_stream(
        self, *, block_frames: int = _DEFAULT_BLOCK_FRAMES
    ) -> SequenceVADStream:
        if block_frames < 1 or block_frames > _MAX_BLOCK_FRAMES:
            raise ValueError(f"block_frames must be in [1, {_MAX_BLOCK_FRAMES}]")
        return SequenceVADStream(self._session, block_frames)


class SequenceVADStream:
    """One-shot probability stream; recurrent state cannot cross recording boundaries."""

    def __init__(self, session: ort.InferenceSession, block_frames: int) -> None:
        self._session = session
        self._block_frames = block_frames
        self._used = False
        self._hidden = np.zeros((1, 1, 128), dtype=np.float32)
        self._cell = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def score_frames(
        self, frames: Iterator[np.ndarray]
    ) -> Iterator[tuple[np.ndarray, float]]:
        if self._used:
            raise RuntimeError("SequenceVADStream belongs to exactly one audio stream")
        self._used = True

        def generate() -> Iterator[tuple[np.ndarray, float]]:
            source = iter(frames)

            while block := list(itertools.islice(source, self._block_frames)):
                audio = np.stack(block)
                if (
                    audio.ndim != 2
                    or audio.shape[1] != _FRAME_SAMPLES
                    or audio.dtype != np.float32
                ):
                    raise ValueError("Sequence VAD requires 512-sample float32 frames")
                contexts = np.empty((len(block), _CONTEXT_SAMPLES), dtype=np.float32)
                contexts[0] = self._context
                if len(block) > 1:
                    contexts[1:] = audio[:-1, -_CONTEXT_SAMPLES:]
                self._context = audio[-1, -_CONTEXT_SAMPLES:].copy()

                probabilities, self._hidden, self._cell = self._session.run(
                    ["speech_probs", "hn", "cn"],
                    {
                        "input": np.concatenate((contexts, audio), axis=1),
                        "h": self._hidden,
                        "c": self._cell,
                    },
                )
                for frame, probability in zip(block, probabilities, strict=True):
                    yield frame, float(probability)

        return generate()


_ENGINE: SequenceVADEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_sequence_vad_engine() -> SequenceVADEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = SequenceVADEngine()
    return _ENGINE

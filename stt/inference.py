"""One process-wide boundary for model inference.

Model adapters do not promise re-entrant inference.  Batch workers and every
live source therefore share this small synchronous gate; callers already run
the blocking work in their respective worker threads.
"""

from __future__ import annotations

import threading
from typing import Any


class InferenceGate:
    """One active model call; waiting live calls precede the next batch call."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._live_waiters = 0

    def call(self, fn: Any, *, live: bool) -> Any:
        with self._condition:
            if live:
                self._live_waiters += 1
                self._condition.notify_all()
            try:
                self._condition.wait_for(
                    lambda: not self._active and (live or self._live_waiters == 0)
                )
                self._active = True
            finally:
                if live:
                    self._live_waiters -= 1
        try:
            return fn()
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


_INFERENCE_GATE = InferenceGate()


def transcription_text(result: Any) -> str:
    """Normalize a model adapter result to plain transcript text."""
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


def transcribe_serialized(model: Any, audio: Any, **kwargs: Any) -> Any:
    return _INFERENCE_GATE.call(lambda: model.transcribe(audio, **kwargs), live=True)


def transcribe_batch_serialized(model: Any, chunks: tuple[Any, ...], **kwargs: Any) -> list[str]:
    if not chunks:
        raise ValueError("Inference batch must not be empty")
    if len(chunks) > 1 and getattr(model, "supports_native_batching", False) is not True:
        raise ValueError("Model has not declared native batching support")

    def run() -> list[str]:
        if hasattr(model, "transcribe_batch"):
            result = model.transcribe_batch(chunks, **kwargs)
        else:
            result = [model.transcribe(chunks[0], **kwargs)]
        if len(result) != len(chunks):
            raise RuntimeError("Model returned an incorrect number of transcripts")
        return result

    return _INFERENCE_GATE.call(run, live=False)

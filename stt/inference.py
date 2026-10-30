"""One process-wide boundary for model inference.

Model adapters do not promise re-entrant inference.  Batch workers and every
live source therefore share this small synchronous gate; callers already run
the blocking work in their respective worker threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceTiming:
    wait_seconds: float = 0.0
    run_seconds: float = 0.0


class InferenceGate:
    """One active model call; waiting live calls precede the next batch call."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._live_waiters = 0

    def call(self, fn: Any, *, live: bool, timing: InferenceTiming | None = None) -> Any:
        wait_started = time.perf_counter() if timing is not None else None
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
        run_started = time.perf_counter() if timing is not None else None
        if timing is not None:
            assert wait_started is not None and run_started is not None
            timing.wait_seconds += run_started - wait_started
        try:
            return fn()
        finally:
            if timing is not None and run_started is not None:
                timing.run_seconds += time.perf_counter() - run_started
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


def transcribe_serialized(
    model: Any, audio: Any, *, timing: InferenceTiming | None = None, **kwargs: Any,
) -> Any:
    return _INFERENCE_GATE.call(
        lambda: model.transcribe(audio, **kwargs), live=True, timing=timing,
    )


def transcribe_batch_serialized(
    model: Any, chunks: tuple[Any, ...], *, timing: InferenceTiming | None = None,
    **kwargs: Any,
) -> list[str]:
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

    return _INFERENCE_GATE.call(run, live=False, timing=timing)


def transcribe_and_detect_language_serialized(
    model: Any, audio: Any, *, timing: InferenceTiming | None = None,
) -> tuple[str, str]:
    return _INFERENCE_GATE.call(
        lambda: model.transcribe_and_detect_language(audio), live=False, timing=timing,
    )

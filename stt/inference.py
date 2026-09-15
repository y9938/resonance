"""One process-wide boundary for model inference.

Model adapters do not promise re-entrant inference.  Batch workers and every
live source therefore share this small synchronous gate; callers already run
the blocking work in their respective worker threads.
"""

from __future__ import annotations

import threading
from typing import Any

_INFERENCE_GATE = threading.Lock()


def transcribe_serialized(model: Any, audio: Any, **kwargs: Any) -> Any:
    with _INFERENCE_GATE:
        return model.transcribe(audio, **kwargs)

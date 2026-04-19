"""STT SSE: transcribe result must become JSON-serializable primitives."""

import json

from server import StreamEvent
from stt.pipeline import _segment_text_from_transcribe


class FakeTranscriptionResult:
    def __init__(self, text: str) -> None:
        self.text = text


def test_segment_text_from_transcribe_extracts_text_attribute() -> None:
    assert _segment_text_from_transcribe(FakeTranscriptionResult("hello")) == "hello"
    assert _segment_text_from_transcribe("plain") == "plain"


def test_progress_event_with_normalized_segment_serializes_to_json() -> None:
    event = StreamEvent(
        "progress",
        {
            "current": 1,
            "total": 2,
            "segment": {
                "start": 0.0,
                "end": 1.0,
                "text": _segment_text_from_transcribe(
                    FakeTranscriptionResult("segment one")
                ),
            },
        },
    )
    line = event.to_sse().strip()
    assert line.startswith("data: ")
    payload = json.loads(line.removeprefix("data: "))
    assert payload["type"] == "progress"
    assert payload["segment"]["text"] == "segment one"

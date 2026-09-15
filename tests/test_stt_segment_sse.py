"""STT SSE: transcribe result must become JSON-serializable primitives."""

import json

from core.jobs import StreamEvent
from stt.inference import transcription_text


class FakeTranscriptionResult:
    def __init__(self, text: str) -> None:
        self.text = text


def test_transcription_text_extracts_text_attribute() -> None:
    assert transcription_text(FakeTranscriptionResult("hello")) == "hello"
    assert transcription_text("plain") == "plain"


def test_progress_event_with_normalized_segment_serializes_to_json() -> None:
    event = StreamEvent(
        "progress",
        {
            "current": 1,
            "total": 2,
            "segment": {
                "start": 0.0,
                "end": 1.0,
                "text": transcription_text(
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

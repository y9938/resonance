"""Unit tests for long-form STT pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from stt import pipeline
from stt.pipeline import MediaInfo, run_stt_job, save_upload_to_path

class FakeUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._buffer = b"".join(chunks)

    async def read(self, size: int = -1) -> bytes:
        if not self._buffer:
            return b""
        if size <= 0:
            chunk = self._buffer
            self._buffer = b""
            return chunk
        chunk = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return chunk


@pytest.mark.asyncio
async def test_save_upload_to_path_streams_to_disk(tmp_path: Path) -> None:
    destination = tmp_path / "input.bin"
    upload = FakeUpload([b"hello", b" ", b"world"])

    size = await save_upload_to_path(upload, destination, chunk_size=2, max_bytes=0)

    assert size == 11
    assert destination.read_bytes() == b"hello world"


@pytest.mark.asyncio
async def test_save_upload_to_path_enforces_max_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "input.bin"
    upload = FakeUpload([b"abc", b"def"])

    with pytest.raises(ValueError, match="File too large"):
        await save_upload_to_path(upload, destination, chunk_size=2, max_bytes=5)


class FakeJobs:
    def __init__(self, *, cancel_after_first_progress: bool = False) -> None:
        self.events: list[tuple[str, dict]] = []
        self.cancelled = False
        self.cancel_after_first_progress = cancel_after_first_progress

    def update_event(self, job_id: str, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))
        if self.cancel_after_first_progress and event_type == "progress":
            self.cancelled = True

    def is_cancelled(self, job_id: str) -> bool:
        return self.cancelled


class FakeModel:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def transcribe(self, array) -> str:
        self.paths.append("array")
        return "text:array"


class FakeLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)


def test_run_stt_job_processes_segments_sequentially(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted: list[int] = []

    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=45.0, codec_name="opus", sample_rate=48000, channels=2, size_bytes=123)

    def fake_stream_vad_chunks(input_path, *args, **kwargs):
        # Yield two dummy chunks with timestamps
        extracted.append(0)
        yield 0.0, 20.0, np.zeros(16000 * 20, dtype=np.float32)
        extracted.append(1)
        yield 18.0, 38.0, np.zeros(16000 * 20, dtype=np.float32)

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "stream_vad_chunks", fake_stream_vad_chunks)

    jobs = FakeJobs()
    model = FakeModel()
    log = FakeLog()
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    input_path = upload_root / "input.opus"
    input_path.write_bytes(b"audio")

    run_stt_job(
        job_id="job-1",
        input_path=str(input_path),
        upload_root=str(upload_root),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "progress", "complete"]
    assert len(model.paths) == 2  # The fake model will just track the arrays in paths now
    assert len(extracted) == 2
    assert not upload_root.exists()


def test_run_stt_job_stops_before_next_segment_when_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted: list[int] = []

    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=45.0, codec_name="opus", sample_rate=48000, channels=2, size_bytes=123)

    def fake_stream_vad_chunks(input_path, *args, **kwargs):
        extracted.append(0)
        yield 0.0, 20.0, np.zeros(16000 * 20, dtype=np.float32)
        extracted.append(1)
        yield 18.0, 38.0, np.zeros(16000 * 20, dtype=np.float32)

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "stream_vad_chunks", fake_stream_vad_chunks)

    jobs = FakeJobs(cancel_after_first_progress=True)
    model = FakeModel()
    log = FakeLog()
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    input_path = upload_root / "input.opus"
    input_path.write_bytes(b"audio")

    run_stt_job(
        job_id="job-1",
        input_path=str(input_path),
        upload_root=str(upload_root),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "cancelled"]
    assert extracted == [0, 1]  # The generator might yield the second chunk before checking cancel
    assert len(model.paths) == 1
    assert any(message.startswith("STT cancelled: ") for message in log.messages)

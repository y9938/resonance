"""Unit tests for long-form STT pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from stt import pipeline
from stt.pipeline import MediaInfo, SegmentSpec, plan_segments, run_stt_job, save_upload_to_path


def test_plan_segments_covers_entire_duration_with_overlap() -> None:
    segments = plan_segments(duration_sec=45.0, chunk_sec=20, overlap_sec=2)

    assert [(round(seg.start_sec, 3), round(seg.end_sec, 3)) for seg in segments] == [
        (0.0, 20.0),
        (18.0, 38.0),
        (36.0, 45.0),
    ]


def test_plan_segments_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        plan_segments(duration_sec=10.0, chunk_sec=20, overlap_sec=20)


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
        if (
            self.cancel_after_first_progress
            and event_type == "progress"
            and data.get("current") == 1
        ):
            self.cancelled = True

    def is_cancelled(self, job_id: str) -> bool:
        return self.cancelled


class FakeModel:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def transcribe(self, path: str) -> str:
        self.paths.append(path)
        return f"text:{Path(path).name}"


class FakeLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)


def test_run_stt_job_processes_segments_sequentially(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted: list[tuple[float, float, str]] = []

    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=45.0, codec_name="opus", sample_rate=48000, channels=2, size_bytes=123)

    def fake_plan_segments(**kwargs) -> list[SegmentSpec]:
        return [
            SegmentSpec(index=1, start_sec=0.0, end_sec=20.0),
            SegmentSpec(index=2, start_sec=18.0, end_sec=38.0),
        ]

    def fake_extract_segment_ffmpeg(input_path: str, segment: SegmentSpec, output_path: str, *, sample_rate: int) -> None:
        extracted.append((segment.start_sec, segment.end_sec, output_path))
        Path(output_path).write_bytes(b"segment")

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "plan_segments", fake_plan_segments)
    monkeypatch.setattr(pipeline, "extract_segment_ffmpeg", fake_extract_segment_ffmpeg)

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
        overlap_sec=2,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "progress", "complete"]
    assert len(model.paths) == 2
    assert len(extracted) == 2
    assert not upload_root.exists()


def test_run_stt_job_stops_before_next_segment_when_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted: list[int] = []

    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=45.0, codec_name="opus", sample_rate=48000, channels=2, size_bytes=123)

    def fake_plan_segments(**kwargs) -> list[SegmentSpec]:
        return [
            SegmentSpec(index=1, start_sec=0.0, end_sec=20.0),
            SegmentSpec(index=2, start_sec=18.0, end_sec=38.0),
        ]

    def fake_extract_segment_ffmpeg(input_path: str, segment: SegmentSpec, output_path: str, *, sample_rate: int) -> None:
        extracted.append(segment.index)
        Path(output_path).write_bytes(b"segment")

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "plan_segments", fake_plan_segments)
    monkeypatch.setattr(pipeline, "extract_segment_ffmpeg", fake_extract_segment_ffmpeg)

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
        overlap_sec=2,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "cancelled"]
    assert extracted == [1]
    assert len(model.paths) == 1
    assert any(message.startswith("STT cancelled: ") for message in log.messages)

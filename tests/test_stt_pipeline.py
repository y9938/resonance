from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stt import pipeline
from stt.pipeline import MediaInfo, run_stt_job


class FakeJobs:
    def __init__(self, *, cancel_after_first_progress: bool = False) -> None:
        self.events: list[tuple[str, dict]] = []
        self.cancelled = False
        self.cancel_after_first_progress = cancel_after_first_progress

    def update_event(self, job_id: str, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))
        if self.cancel_after_first_progress and event_type == "progress":
            self.cancelled = True

    def mark_cancelled(self, job_id: str) -> bool:
        self.cancelled = True
        return True

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
        input_paths=str(input_path),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "progress", "complete"]
    assert len(model.paths) == 2
    assert len(extracted) == 2


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
        input_paths=str(input_path),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
    )

    assert [event for event, _ in jobs.events] == ["start", "progress", "cancelled"]
    assert jobs.events[0][1].get("stage") == "transcription"
    assert extracted == [0, 1]
    assert len(model.paths) == 1
    assert any(message.startswith("STT cancelled: ") for message in log.messages)


def test_run_stt_job_diarization_stage_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=10.0, codec_name="opus", sample_rate=16000, channels=1, size_bytes=100)

    def fake_stream_vad_chunks(input_path, *args, **kwargs):
        yield 0.0, 5.0, np.zeros(16000 * 5, dtype=np.float32)

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "stream_vad_chunks", fake_stream_vad_chunks)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: np.zeros(16000 * 10, dtype=np.int16).tobytes())
    monkeypatch.setattr("stt.diarization.diarize_audio", lambda audio: [])

    jobs = FakeJobs()
    model = FakeModel()
    log = FakeLog()
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    input_path = upload_root / "input.opus"
    input_path.write_bytes(b"audio")

    run_stt_job(
        job_id="job-diarize",
        input_paths=str(input_path),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
        diarization=True,
    )

    assert jobs.events[0][0] == "start"
    assert jobs.events[0][1].get("stage") == "diarization"


def test_run_stt_job_diarization_cancelled_midway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=10.0, codec_name="opus", sample_rate=16000, channels=1, size_bytes=100)

    jobs = FakeJobs()
    model = FakeModel()
    log = FakeLog()
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    input_path = upload_root / "input.opus"
    input_path.write_bytes(b"audio")

    def fake_diarize(audio, cancel_check=None):
        jobs.mark_cancelled("job-cancel-midway")
        if cancel_check and cancel_check():
            raise RuntimeError("STT job cancelled")
        return []

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: np.zeros(16000 * 10, dtype=np.int16).tobytes())
    monkeypatch.setattr("stt.diarization.diarize_audio", fake_diarize)

    run_stt_job(
        job_id="job-cancel-midway",
        input_paths=str(input_path),
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
        diarization=True,
    )

    cancelled_logs = [msg for msg in log.messages if "STT cancelled:" in msg]
    assert len(cancelled_logs) == 1, f"Expected exactly 1 cancel log, got {len(cancelled_logs)}"
    assert jobs.events[-1][0] == "cancelled"


def test_run_stt_job_dual_stream_tagging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_probe_media(input_path: str) -> MediaInfo:
        return MediaInfo(duration_sec=10.0, codec_name="wav", sample_rate=16000, channels=1, size_bytes=100)

    def fake_stream_vad_chunks(input_path, *args, **kwargs):
        if "mic.wav" in input_path:
            yield 0.0, 4.0, np.zeros(16000 * 4, dtype=np.float32)
        else:
            yield 4.0, 10.0, np.zeros(16000 * 6, dtype=np.float32)

    class MockEchoModel:
        def transcribe(self, chunk, **kwargs):
            return "Sample transcribed text"

    monkeypatch.setattr(pipeline, "probe_media", fake_probe_media)
    monkeypatch.setattr(pipeline, "stream_vad_chunks", fake_stream_vad_chunks)

    jobs = FakeJobs()
    model = MockEchoModel()
    log = FakeLog()
    upload_root = tmp_path / "upload"
    upload_root.mkdir()
    sys_path = upload_root / "sys.wav"
    mic_path = upload_root / "mic.wav"
    sys_path.write_bytes(b"wav")
    mic_path.write_bytes(b"wav")

    run_stt_job(
        job_id="job-dual",
        input_paths={"sys": str(sys_path), "mic": str(mic_path)},
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=16000,
        chunk_sec=20,
        diarization=False,
    )

    progress_events = [data for event, data in jobs.events if event == "progress"]
    assert len(progress_events) == 2
    assert progress_events[0]["segment"]["text"] == "[SOURCE:MIC]: Sample transcribed text"
    assert progress_events[0]["segment"]["source"] == "mic"
    assert progress_events[1]["segment"]["text"] == "[SOURCE:SYS]: Sample transcribed text"
    assert progress_events[1]["segment"]["source"] == "sys"

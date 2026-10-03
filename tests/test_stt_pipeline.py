from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from stt import pipeline
from stt.pipeline import MediaInfo, run_stt_job
from tests.media_helpers import silent_wav


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


class SilentVAD:
    def new_stream(self) -> SilentVAD:
        return self

    def score_frames(self, frames):
        return ((frame, 0.0) for frame in frames)


class InitialSpeechVAD(SilentVAD):
    def score_frames(self, frames):
        return ((frame, 1.0 if index < 10 else 0.0) for index, frame in enumerate(frames))


def test_encoded_unknown_duration_completes_with_actual_pcm_duration(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "media_duration", lambda media: None)
    monkeypatch.setattr(pipeline, "get_sequence_vad_engine", lambda: SilentVAD())
    jobs = FakeJobs()
    run_stt_job(
        job_id="unknown", input_paths=silent_wav(16001), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=2,
    )

    assert [event for event, _ in jobs.events] == ["start", "complete"]
    assert jobs.events[0][1]["duration"] == jobs.events[0][1]["total"] == 0
    assert jobs.events[-1][1]["duration"] == 16001 / 16000


def test_encoded_unknown_duration_progress_has_unknown_total(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "media_duration", lambda media: None)
    monkeypatch.setattr(pipeline, "get_sequence_vad_engine", lambda: InitialSpeechVAD())
    jobs = FakeJobs()
    run_stt_job(
        job_id="unknown-with-speech", input_paths=silent_wav(32000), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=2,
    )

    progress = [data for event, data in jobs.events if event == "progress"]
    assert len(progress) == 1
    assert progress[0]["total"] == 0
    assert progress[0]["current"] < 2
    assert jobs.events[-1] == ("complete", {"duration": 2.0})


@pytest.mark.parametrize("hint", [None, 0.5])
def test_encoded_actual_samples_override_duration_hint(monkeypatch, hint) -> None:
    monkeypatch.setattr(pipeline, "media_duration", lambda media: hint)
    monkeypatch.setattr(pipeline, "get_sequence_vad_engine", lambda: SilentVAD())
    jobs = FakeJobs()
    run_stt_job(
        job_id="over-limit", input_paths=silent_wav(16001), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=1,
    )

    assert jobs.events[-1] == ("error", {"message": "Audio too long (max 1s)"})
    assert all(event != "complete" for event, _ in jobs.events)


def test_encoded_known_over_limit_fails_before_decode(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "media_duration", lambda media: 2.0)
    monkeypatch.setattr(
        pipeline, "iter_media_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded")),
    )
    jobs = FakeJobs()
    run_stt_job(
        job_id="known-long", input_paths=silent_wav(1), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=1,
    )

    assert jobs.events == [("error", {"message": "Audio too long (max 1s)"})]


@pytest.mark.parametrize("hint", [None, 1.0])
def test_encoded_zero_pcm_fails_regardless_of_duration_hint(monkeypatch, hint) -> None:
    monkeypatch.setattr(pipeline, "media_duration", lambda media: hint)
    monkeypatch.setattr(pipeline, "get_sequence_vad_engine", lambda: SilentVAD())
    jobs = FakeJobs()
    run_stt_job(
        job_id="empty-pcm", input_paths=silent_wav(0), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=1,
    )

    assert jobs.events[-1] == ("error", {"message": "Audio too short or empty after decoding"})


def test_diarization_rejects_excess_pcm_before_accumulation(monkeypatch) -> None:
    from stt import diarization

    monkeypatch.setattr(pipeline, "media_duration", lambda media: None)

    def limited_pcm(*args, **kwargs):
        assert kwargs["max_samples"] == 16000
        yield np.zeros(16000, dtype=np.float32)
        raise pipeline.DecodedSampleLimitExceeded

    monkeypatch.setattr(pipeline, "iter_media_pcm", limited_pcm)
    diarize = MagicMock()
    monkeypatch.setattr(diarization, "diarize_audio", diarize)
    appended_samples = 0
    original_append = pipeline.AudioMemoryBuffer.append

    def tracked_append(buffer, pcm):
        nonlocal appended_samples
        appended_samples += len(pcm)
        assert appended_samples <= 16000
        return original_append(buffer, pcm)

    monkeypatch.setattr(pipeline.AudioMemoryBuffer, "append", tracked_append)
    jobs = FakeJobs()
    run_stt_job(
        job_id="diarization-over-limit", input_paths=silent_wav(16001), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=1, diarization=True,
    )

    assert appended_samples == 16000
    diarize.assert_not_called()
    assert jobs.events[-1] == ("error", {"message": "Audio too long (max 1s)"})


def test_diarization_and_stt_use_separate_sample_budgets(monkeypatch) -> None:
    from stt import diarization

    monkeypatch.setattr(pipeline, "media_duration", lambda media: None)
    monkeypatch.setattr(pipeline, "get_sequence_vad_engine", lambda: SilentVAD())
    diarize = MagicMock(return_value=[])
    monkeypatch.setattr(diarization, "diarize_audio", diarize)
    jobs = FakeJobs()
    run_stt_job(
        job_id="diarization-at-limit", input_paths=silent_wav(16000), jobs=jobs,
        model=FakeModel(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, max_duration_sec=1, diarization=True,
    )

    diarize.assert_called_once()
    assert len(diarize.call_args.args[0]) == 16000
    assert jobs.events[-1] == ("complete", {"duration": 1.0})


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

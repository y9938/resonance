import wave
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import server
from core.jobs import JobRegistry
from server import active_live_sessions, app, jobs
from stt.live import LiveSTTSession
from stt.stream_vad import _VAD_WINDOW_SAMPLES

client = TestClient(app)


def _use_fresh_live_registry(monkeypatch) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    server.active_live_sessions.clear()
    return registry


def _start_live_job(owner: TestClient, model: MagicMock) -> str:
    with patch("server.models.get_stt_model", return_value=model):
        response = owner.post("/api/jobs/live/start?language=ru&model=gigaam")
    assert response.status_code == 200
    return response.json()["job_id"]


def _speech_then_silence(silence_windows: int = 10) -> np.ndarray:
    return np.concatenate(
        [
            np.ones(_VAD_WINDOW_SAMPLES * 7, dtype=np.float32),
            np.zeros(_VAD_WINDOW_SAMPLES * silence_windows, dtype=np.float32),
        ]
    )


def _load_real_speech_samples(sample_rate: int = 16000) -> np.ndarray:
    with wave.open("tests/fixtures/ru_audio.wav", "rb") as wf:
        n_frames = min(wf.getnframes(), sample_rate * 3)  # first 3 seconds
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples


def test_live_stt_session_vad_and_transcribe():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = "Привет мир"
    mock_jobs = MagicMock()

    session = LiveSTTSession(
        job_id="test_live_job",
        session_id="test_sess",
        model=mock_model,
        jobs=mock_jobs,
        sample_rate=16000,
        source="mic",
    )

    speech = _load_real_speech_samples()
    silence = np.zeros(16000, dtype=np.float32)

    emitted1 = session.process_pcm_chunk(speech, source="mic")
    emitted2 = session.process_pcm_chunk(silence, source="mic")
    emitted = emitted1 + emitted2 + session.flush()

    assert len(emitted) >= 1
    assert "Привет мир" in emitted[0]["text"]
    assert not emitted[0]["text"].startswith("[SOURCE:MIC]")
    assert emitted[0]["source"] == "mic"


class FakeVad:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)

    def process_frame(self, _state, _window) -> float:
        return next(self._probabilities)


def test_live_session_discards_short_speech_before_model() -> None:
    model = MagicMock()
    session = LiveSTTSession(
        job_id="short-speech",
        session_id="test_sess",
        model=model,
        jobs=MagicMock(),
    )
    session._vad_engine = FakeVad([1.0] + [0.0] * 12)

    audio = np.concatenate(
        [
            np.ones(_VAD_WINDOW_SAMPLES, dtype=np.float32),
            np.zeros(_VAD_WINDOW_SAMPLES * 12, dtype=np.float32),
        ]
    )
    assert session.process_pcm_chunk(audio) == []
    model.transcribe.assert_not_called()


def test_live_session_flushes_final_speech_and_tracks_duration() -> None:
    model = MagicMock()
    model.transcribe.return_value = "Последняя фраза"
    session = LiveSTTSession(
        job_id="final-speech",
        session_id="test_sess",
        model=model,
        jobs=MagicMock(),
    )
    session._vad_engine = FakeVad([1.0] * 7)

    session.process_pcm_chunk(np.ones(_VAD_WINDOW_SAMPLES * 7, dtype=np.float32))
    emitted = session.flush()

    assert len(emitted) == 1
    assert model.transcribe.call_count == 1
    assert session.duration_sec == 7 * _VAD_WINDOW_SAMPLES / 16000


def test_live_endpoints_lifecycle():
    with patch("server.models.get_stt_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "Тест живого распознавания"
        mock_get_model.return_value = mock_model

        # 1. Start live job
        resp_start = client.post("/api/jobs/live/start?language=ru&model=gigaam")
        assert resp_start.status_code == 200
        job_id = resp_start.json()["job_id"]
        assert job_id is not None

        status = jobs.get_status(job_id)
        assert status is not None
        assert status["state"] == "running"

        # 2. Post a decoded live chunk without requiring native FFmpeg libraries.
        decoded_buffer = MagicMock()
        decoded_buffer.as_ndarray.return_value = np.zeros(16000, dtype=np.float32)
        with patch("server.decode_media_bytes", return_value=decoded_buffer):
            resp_chunk = client.post(
                f"/api/jobs/live/{job_id}/chunk?sequence=1",
                files={"file": ("chunk.wav", b"valid-for-mocked-decoder", "audio/wav")},
            )
        assert resp_chunk.status_code == 200

        # 3. Stop live job
        resp_stop = client.post(f"/api/jobs/live/{job_id}/stop")
        assert resp_stop.status_code == 200
        assert resp_stop.json()["job_id"] == job_id

        final_status = jobs.get_status(job_id)
        assert final_status["state"] == "completed"
        assert final_status["result"]["duration"] == 1.0


def test_live_endpoints_reject_other_session_and_cancel_cleans_up() -> None:
    owner = TestClient(app)
    other = TestClient(app)
    with patch("server.models.get_stt_model", return_value=MagicMock()):
        start = owner.post("/api/jobs/live/start?language=ru&model=gigaam")
        job_id = start.json()["job_id"]

        assert other.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"audio", "audio/wav")},
        ).status_code == 404
        assert other.post(f"/api/jobs/live/{job_id}/stop").status_code == 404

        assert owner.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        assert job_id not in active_live_sessions
        assert owner.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"audio", "audio/wav")},
        ).status_code == 404


def test_live_chunk_empty_or_invalid_media_keeps_session_active(monkeypatch) -> None:
    registry = _use_fresh_live_registry(monkeypatch)
    owner = TestClient(app)
    job_id = _start_live_job(owner, MagicMock())

    empty = owner.post(
        f"/api/jobs/live/{job_id}/chunk?sequence=1",
        files={"file": ("chunk.wav", b"", "audio/wav")},
    )
    assert empty.status_code == 400
    assert job_id in server.active_live_sessions
    assert registry.get_status(job_id)["state"] == "running"

    with patch("server.decode_media_bytes", side_effect=ValueError("bad wav")):
        invalid = owner.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"not-a-wav", "audio/wav")},
        )
    assert invalid.status_code == 400
    assert job_id in server.active_live_sessions
    assert registry.get_status(job_id)["state"] == "running"
    assert owner.post(f"/api/jobs/live/{job_id}/stop").status_code == 200


def test_live_chunk_silence_is_success_without_emitted_segment(monkeypatch) -> None:
    registry = _use_fresh_live_registry(monkeypatch)
    owner = TestClient(app)
    job_id = _start_live_job(owner, MagicMock())
    server.active_live_sessions[job_id].session._vad_engine = FakeVad([0.0] * 40)
    decoded = MagicMock()
    decoded.as_ndarray.return_value = np.zeros(_VAD_WINDOW_SAMPLES * 20, dtype=np.float32)

    with patch("server.decode_media_bytes", return_value=decoded):
        response = owner.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {"emitted": 0, "ack_sequence": 1}
    assert job_id in server.active_live_sessions
    assert registry.get_status(job_id)["state"] == "running"
    assert owner.post(f"/api/jobs/live/{job_id}/stop").status_code == 200


def test_live_chunk_processing_failure_fails_and_cleans_up_session(monkeypatch) -> None:
    registry = _use_fresh_live_registry(monkeypatch)
    owner = TestClient(app)
    failing_model = MagicMock()
    failing_model.transcribe.side_effect = RuntimeError("ASR unavailable")
    job_id = _start_live_job(owner, failing_model)
    server.active_live_sessions[job_id].session._vad_engine = FakeVad([1.0] * 7 + [0.0] * 35)
    decoded = MagicMock()
    decoded.as_ndarray.return_value = _speech_then_silence(35)

    with patch("server.decode_media_bytes", return_value=decoded):
        response = owner.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert response.status_code == 500
    assert job_id not in server.active_live_sessions
    assert registry.get_status(job_id)["state"] == "failed"
    assert owner.post(
        f"/api/jobs/live/{job_id}/chunk?sequence=1",
        files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
    ).status_code == 404
    assert owner.post(f"/api/jobs/live/{job_id}/stop").status_code == 404

    replacement_job_id = _start_live_job(owner, MagicMock())
    assert replacement_job_id in server.active_live_sessions
    assert owner.post(f"/api/jobs/live/{replacement_job_id}/stop").status_code == 200


def test_live_chunk_progress_publication_failure_is_fatal(monkeypatch) -> None:
    registry = _use_fresh_live_registry(monkeypatch)
    owner = TestClient(app)
    model = MagicMock()
    model.transcribe.return_value = "Готовый сегмент"
    job_id = _start_live_job(owner, model)
    server.active_live_sessions[job_id].session._vad_engine = FakeVad([1.0] * 7 + [0.0] * 35)
    decoded = MagicMock()
    decoded.as_ndarray.return_value = _speech_then_silence(35)

    original_update_event = registry.update_event

    def fail_progress(job_id: str, event_type: str, data: dict) -> bool:
        if event_type == "progress":
            raise RuntimeError("event store unavailable")
        return original_update_event(job_id, event_type, data)

    monkeypatch.setattr(registry, "update_event", fail_progress)
    with patch("server.decode_media_bytes", return_value=decoded):
        response = owner.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert response.status_code == 500
    assert job_id not in server.active_live_sessions
    assert registry.get_status(job_id)["state"] == "failed"

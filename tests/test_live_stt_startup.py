import threading
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import server
from core.jobs import JobRegistry


def _isolated_registry(monkeypatch) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    server.active_live_sessions.clear()
    server.active_system_captures.clear()
    return registry


def _only_job_status(registry: JobRegistry) -> dict:
    assert len(registry._jobs) == 1
    job_id = next(iter(registry._jobs))
    status = registry.get_status(job_id)
    assert status is not None
    return status


def test_live_start_model_failure_creates_no_job_or_session(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(side_effect=RuntimeError("model unavailable")))

    response = TestClient(server.app).post("/api/jobs/live/start?language=ru&model=gigaam")

    assert response.status_code == 500
    assert registry._jobs == {}
    assert server.active_live_sessions == {}


def test_live_start_session_failure_marks_job_failed_and_retry_succeeds(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    client = TestClient(server.app)

    with monkeypatch.context() as failed_start:
        failed_start.setattr(server, "LiveSTTSession", MagicMock(side_effect=RuntimeError("VAD unavailable")))
        response = client.post("/api/jobs/live/start?language=ru&model=gigaam")

    assert response.status_code == 500
    assert _only_job_status(registry)["state"] == "failed"
    assert server.active_live_sessions == {}

    response = client.post("/api/jobs/live/start?language=ru&model=gigaam")

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert registry.get_status(job_id)["state"] == "running"
    assert job_id in server.active_live_sessions
    assert client.post(f"/api/jobs/live/{job_id}/stop").status_code == 200


def test_system_audio_factory_failure_creates_no_job_or_capture(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        server,
        "get_system_audio_capture",
        MagicMock(side_effect=RuntimeError("capture unavailable")),
    )

    response = TestClient(server.app).post("/api/system-audio/start?language=ru&model=gigaam")

    assert response.status_code == 500
    assert registry._jobs == {}
    assert server.active_system_captures == {}


def test_system_audio_model_failure_creates_no_job_or_capture(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(side_effect=RuntimeError("model unavailable")))
    capture_factory = MagicMock()
    monkeypatch.setattr(server, "get_system_audio_capture", capture_factory)

    response = TestClient(server.app).post("/api/system-audio/start?language=ru&model=gigaam")

    assert response.status_code == 500
    capture_factory.assert_not_called()
    assert registry._jobs == {}
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_start_failure_marks_job_failed_and_retry_succeeds(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    class RetryCapture:
        attempts = 0

        def __init__(self, **_kwargs) -> None:
            self.stop_calls = 0
            self.stopped = threading.Event()

        def start_capture(self) -> None:
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("driver busy")

        def stop_capture(self) -> None:
            self.stop_calls += 1
            self.stopped.set()

        def get_audio_stream(self):
            self.stopped.wait()
            if False:
                yield ("sys", None)

    monkeypatch.setattr(server, "get_system_audio_capture", RetryCapture)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        failed = await client.post("/api/system-audio/start?language=ru&model=gigaam")

        assert failed.status_code == 500
        assert _only_job_status(registry)["state"] == "failed"
        assert server.active_system_captures == {}

        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")

        assert started.status_code == 200
        job_id = started.json()["job_id"]
        assert registry.get_status(job_id)["state"] == "running"
        assert job_id in server.active_system_captures
        assert (await client.post(f"/api/system-audio/stop?job_id={job_id}")).status_code == 200


def test_system_audio_session_failure_marks_job_failed_without_starting_capture(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    capture = MagicMock()
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))
    monkeypatch.setattr(server, "LiveSTTSession", MagicMock(side_effect=RuntimeError("VAD unavailable")))

    response = TestClient(server.app).post("/api/system-audio/start?language=ru&model=gigaam")

    assert response.status_code == 500
    capture.start_capture.assert_not_called()
    capture.stop_capture.assert_not_called()
    assert _only_job_status(registry)["state"] == "failed"
    assert server.active_system_captures == {}


def test_system_audio_rolls_back_started_engine_when_task_registration_fails(monkeypatch) -> None:
    registry = _isolated_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    capture = MagicMock()
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))
    monkeypatch.setattr(server.asyncio, "to_thread", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(server.asyncio, "create_task", MagicMock(side_effect=RuntimeError("task registration failed")))

    response = TestClient(server.app).post("/api/system-audio/start?language=ru&model=gigaam")

    assert response.status_code == 500
    capture.start_capture.assert_called_once_with()
    capture.stop_capture.assert_called_once_with()
    assert _only_job_status(registry)["state"] == "failed"
    assert server.active_system_captures == {}

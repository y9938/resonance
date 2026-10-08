import asyncio
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest

import server
from core.jobs import JobRegistry


def _isolated_system_registry(monkeypatch) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    server.active_system_captures.clear()
    return registry


async def _wait_for_state(registry: JobRegistry, job_id: str, state: str) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        status = registry.get_status(job_id)
        if status and status["state"] == state:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"Job {job_id} did not reach {state}")


class CooperativeCapture:
    def __init__(self, **_kwargs) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def start_capture(self) -> None:
        self.started.set()

    def stop_capture(self) -> None:
        self.stopped.set()

    def get_audio_stream(self):
        self.stopped.wait()
        if False:
            yield ("sys", None)


@pytest.mark.asyncio
async def test_system_audio_cooperative_stop_flushes_then_completes(monkeypatch, local_host_app) -> None:
    registry = _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        config = (await client.get("/api/config")).json()
        assert config["local_files_enabled"] and config["system_audio_enabled"]
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert stopped.status_code == 200
    assert capture.stopped.is_set()
    assert registry.get_status(job_id)["state"] == "completed"
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_cannot_reattach_while_stopping(monkeypatch, local_host_app) -> None:
    _isolated_system_registry(monkeypatch)

    class DelayedCapture(CooperativeCapture):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def get_audio_stream(self):
            self.release.wait()
            if False:
                yield ("sys", None)

    capture = DelayedCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", lambda **kwargs: capture)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        stop = asyncio.create_task(client.post(f"/api/system-audio/stop?job_id={started.json()['job_id']}"))
        try:
            assert await asyncio.to_thread(capture.stopped.wait, 1)
            resumed = await client.post("/api/system-audio/start?language=ru&model=gigaam")
            assert resumed.status_code == 409
        finally:
            capture.release.set()
            stopped = await stop
        assert stopped.status_code == 200


@pytest.mark.asyncio
async def test_system_audio_producer_exception_fails_and_cleans_registry(monkeypatch, local_host_app) -> None:
    registry = _isolated_system_registry(monkeypatch)

    class CrashingCapture(CooperativeCapture):
        def get_audio_stream(self):
            raise RuntimeError("driver disconnected")

    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    capture = CrashingCapture()
    monkeypatch.setattr(server, "get_system_audio_capture", lambda **kwargs: capture)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        await _wait_for_state(registry, job_id, "failed")
        stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert stopped.status_code == 404
    assert capture.stopped.is_set()
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_stop_timeout_fails_without_premature_completion(monkeypatch, local_host_app) -> None:
    registry = _isolated_system_registry(monkeypatch)

    class HungCapture(CooperativeCapture):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.release = threading.Event()

        def stop_capture(self) -> None:
            self.stopped.set()

        def get_audio_stream(self):
            self.release.wait()
            if False:
                yield ("sys", None)

    capture = HungCapture()
    monkeypatch.setattr(server.Config, "SYSTEM_CAPTURE_STOP_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        timed_out = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    capture.release.set()
    await asyncio.sleep(0.02)
    assert timed_out.status_code == 504
    assert registry.get_status(job_id)["state"] == "failed"
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_cancel_keeps_cancelled_terminal_and_cleans_registry(monkeypatch, local_host_app) -> None:
    registry = _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")

    await asyncio.sleep(0.02)
    assert cancelled.status_code == 200
    assert capture.stopped.is_set()
    assert registry.get_status(job_id)["state"] == "cancelled"
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_cannot_return_or_stop_another_sessions_capture(monkeypatch, local_host_app) -> None:
    registry = _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    factory = MagicMock(return_value=capture)
    monkeypatch.setattr(server, "get_system_audio_capture", factory)

    transport = httpx.ASGITransport(app=server.app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as owner,
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as other,
    ):
        started = await owner.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        try:
            response = await other.post("/api/system-audio/start?language=ru&model=gigaam")
            assert response.status_code == 409
            assert "job_id" not in response.json()
            assert (await other.get(f"/api/jobs/{job_id}")).status_code == 404
            assert (await other.post(f"/api/system-audio/stop?job_id={job_id}")).status_code == 404
            assert not capture.stopped.is_set()
            factory.assert_called_once()
            assert registry.get_status(job_id)["state"] == "running"
            resumed = await owner.post("/api/system-audio/start?language=ru&model=gigaam")
            assert resumed.json()["job_id"] == job_id
        finally:
            assert (await owner.post(f"/api/system-audio/stop?job_id={job_id}")).status_code == 200


@pytest.mark.asyncio
async def test_system_audio_reattach_requires_identical_config(monkeypatch, local_host_app) -> None:
    _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://localhost"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        try:
            resumed = await client.post("/api/system-audio/start?language=ru&model=gigaam")
            different = await client.post("/api/system-audio/start?language=ru&model=gigaam&include_microphone=true")
            snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
            assert snapshot["result"]["capture_config"] == started.json()["capture_config"]
            assert resumed.json()["capture_config"] == {"language": "ru", "model": "gigaam",
                                                       "include_microphone": False}
        finally:
            stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert resumed.status_code == 200
    assert resumed.json()["job_id"] == job_id
    assert resumed.json()["resumed"] is True
    assert resumed.json()["started_at"] == started.json()["started_at"]
    assert different.status_code == 409
    assert stopped.status_code == 200

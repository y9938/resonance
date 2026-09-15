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
async def test_system_audio_cooperative_stop_flushes_then_completes(monkeypatch) -> None:
    registry = _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert stopped.status_code == 200
    assert capture.stopped.is_set()
    assert registry.get_status(job_id)["state"] == "completed"
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_producer_exception_fails_and_cleans_registry(monkeypatch) -> None:
    registry = _isolated_system_registry(monkeypatch)

    class CrashingCapture(CooperativeCapture):
        def get_audio_stream(self):
            raise RuntimeError("driver disconnected")

    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", CrashingCapture)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        await _wait_for_state(registry, job_id, "failed")
        stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert stopped.status_code == 404
    assert server.active_system_captures == {}


@pytest.mark.asyncio
async def test_system_audio_stop_timeout_fails_without_premature_completion(monkeypatch) -> None:
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
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
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
async def test_system_audio_cancel_keeps_cancelled_terminal_and_cleans_registry(monkeypatch) -> None:
    registry = _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
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
async def test_system_audio_reattach_requires_identical_config(monkeypatch) -> None:
    _isolated_system_registry(monkeypatch)
    capture = CooperativeCapture()
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "get_system_audio_capture", MagicMock(return_value=capture))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        started = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        job_id = started.json()["job_id"]
        resumed = await client.post("/api/system-audio/start?language=ru&model=gigaam")
        different = await client.post("/api/system-audio/start?language=ru&model=gigaam&include_microphone=true")
        stopped = await client.post(f"/api/system-audio/stop?job_id={job_id}")

    assert resumed.status_code == 200
    assert resumed.json()["job_id"] == job_id
    assert resumed.json()["resumed"] is True
    assert resumed.json()["started_at"] == started.json()["started_at"]
    assert different.status_code == 409
    assert stopped.status_code == 200

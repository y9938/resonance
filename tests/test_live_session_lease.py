import asyncio
import threading
import time
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

import server
from core.jobs import JobRegistry


def _isolated_live_registry(monkeypatch, timeout: float) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    monkeypatch.setattr(server.Config, "LIVE_STT_IDLE_TIMEOUT_SEC", timeout)
    server.active_live_sessions.clear()
    return registry


async def _wait_for_state(registry: JobRegistry, job_id: str, state: str) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        status = registry.get_status(job_id)
        if status and status["state"] == state:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"Job {job_id} did not reach {state}")


async def _wait_for_handle_removal(job_id: str) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if job_id not in server.active_live_sessions:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"Live handle {job_id} was not removed")


async def _start_live_job(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/jobs/live/start?language=ru&model=gigaam")
    assert response.status_code == 200
    return response.json()["job_id"]


def _decoded_pcm() -> MagicMock:
    decoded = MagicMock()
    decoded.as_ndarray.return_value = np.zeros(512, dtype=np.float32)
    return decoded


@pytest.mark.asyncio
async def test_idle_live_job_completes_as_partial_and_rejects_later_requests(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await _wait_for_state(registry, job_id, "completed")
        await _wait_for_handle_removal(job_id)

        status = registry.get_status(job_id)
        assert status["result"]["completion_reason"] == "idle_timeout"
        assert status["result"]["partial"] is True
        assert (await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )).status_code == 404
        assert (await client.post(f"/api/jobs/live/{job_id}/stop")).status_code == 404


@pytest.mark.asyncio
async def test_nonempty_chunk_extends_idle_deadline(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.05)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=_decoded_pcm()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await asyncio.sleep(0.03)
        response = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        assert response.status_code == 200

        await asyncio.sleep(0.03)
        assert registry.get_status(job_id)["state"] == "running"
        await _wait_for_state(registry, job_id, "completed")


@pytest.mark.asyncio
async def test_timeout_waits_for_inflight_chunk_before_flushing(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=_decoded_pcm()))

    entered = threading.Event()
    release = threading.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session
        session.flush = MagicMock(return_value=[])

        def blocked_process(_pcm, _source):
            entered.set()
            release.wait()
            return []

        session.process_pcm_chunk = blocked_process
        chunk = asyncio.create_task(client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        ))
        assert await asyncio.to_thread(entered.wait, 0.5)
        await asyncio.sleep(0.04)

        rejected = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        assert rejected.status_code == 404
        session.flush.assert_not_called()

        release.set()
        assert (await chunk).status_code == 200
        await _wait_for_state(registry, job_id, "completed")

    session.flush.assert_called_once_with("mic")


@pytest.mark.asyncio
async def test_timeout_and_manual_stop_have_one_terminal_owner(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await asyncio.sleep(0.02)
        stopped = await client.post(f"/api/jobs/live/{job_id}/stop")
        assert stopped.status_code in {200, 404}
        await _wait_for_state(registry, job_id, "completed")
        await asyncio.sleep(0.03)

    events = registry.events_after(job_id, 0)
    assert [event["type"] for event in events].count("complete") == 1


@pytest.mark.asyncio
async def test_timeout_and_cancel_have_one_terminal_owner(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await asyncio.sleep(0.02)
        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code in {200, 404}
        await asyncio.sleep(0.04)

    events = registry.events_after(job_id, 0)
    terminal_events = [event["type"] for event in events if event["type"] in {"complete", "cancelled"}]
    assert len(terminal_events) == 1
    assert registry.get_status(job_id)["state"] in {"completed", "cancelled"}


@pytest.mark.asyncio
async def test_decode_error_still_extends_the_idle_lease(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.05)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(side_effect=ValueError("bad wav")))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await asyncio.sleep(0.03)
        invalid = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"invalid-wav", "audio/wav")},
        )
        assert invalid.status_code == 400

        await asyncio.sleep(0.03)
        assert registry.get_status(job_id)["state"] == "running"
        await _wait_for_state(registry, job_id, "completed")


@pytest.mark.asyncio
async def test_manual_stop_cancels_expiry_task(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        stopped = await client.post(f"/api/jobs/live/{job_id}/stop")
        assert stopped.status_code == 200
        await asyncio.sleep(0.04)

    events = registry.events_after(job_id, 0)
    assert [event["type"] for event in events].count("complete") == 1
    assert registry.get_status(job_id)["result"]["completion_reason"] == "user_stop"
    assert registry.get_status(job_id)["result"]["partial"] is False


@pytest.mark.asyncio
async def test_cancel_cancels_expiry_task(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0.02)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        await asyncio.sleep(0.04)

    events = registry.events_after(job_id, 0)
    assert [event["type"] for event in events].count("cancelled") == 1
    assert [event["type"] for event in events].count("complete") == 0


@pytest.mark.asyncio
async def test_zero_idle_timeout_disables_lease(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch, timeout=0)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        await asyncio.sleep(0.04)
        assert registry.get_status(job_id)["state"] == "running"
        assert job_id in server.active_live_sessions
        assert (await client.post(f"/api/jobs/live/{job_id}/stop")).status_code == 200

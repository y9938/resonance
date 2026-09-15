import asyncio
import threading
from unittest.mock import MagicMock

import httpx
import pytest

import server
from core.jobs import JobRegistry


def _isolated_live_registry(monkeypatch) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    server.active_live_sessions.clear()
    return registry


async def _start_live_job(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/jobs/live/start?language=ru&model=gigaam")
    assert response.status_code == 200
    return response.json()["job_id"]


def _decoded_pcm() -> MagicMock:
    decoded = MagicMock()
    decoded.as_ndarray.return_value = MagicMock()
    return decoded


@pytest.mark.asyncio
async def test_stop_waits_for_claimed_chunk_before_flushing(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decoded = _decoded_pcm()
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=decoded))

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
        chunk = asyncio.create_task(
            client.post(
                f"/api/jobs/live/{job_id}/chunk?sequence=1",
                files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
            )
        )
        assert await asyncio.to_thread(entered.wait, 0.5)

        stopping = asyncio.create_task(client.post(f"/api/jobs/live/{job_id}/stop"))
        await asyncio.sleep(0.02)
        assert not stopping.done()
        session.flush.assert_not_called()

        release.set()
        assert (await chunk).status_code == 200
        assert (await stopping).status_code == 200

    session.flush.assert_called_once_with("mic")
    assert registry.get_status(job_id)["state"] == "completed"
    assert server.active_live_sessions == {}


@pytest.mark.asyncio
async def test_finishing_handle_rejects_new_chunk_without_processing(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=_decoded_pcm()))

    entered = threading.Event()
    release = threading.Event()
    calls = 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session

        def blocked_process(_pcm, _source):
            nonlocal calls
            calls += 1
            entered.set()
            release.wait()
            return []

        session.process_pcm_chunk = blocked_process
        first_chunk = asyncio.create_task(
            client.post(
                f"/api/jobs/live/{job_id}/chunk?sequence=1",
                files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
            )
        )
        assert await asyncio.to_thread(entered.wait, 0.5)
        stopping = asyncio.create_task(client.post(f"/api/jobs/live/{job_id}/stop"))
        await asyncio.sleep(0.02)

        rejected = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        assert rejected.status_code == 404
        assert calls == 1

        release.set()
        assert (await first_chunk).status_code == 200
        assert (await stopping).status_code == 200


@pytest.mark.asyncio
async def test_only_one_stop_owns_flush_and_completion(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))

    entered = threading.Event()
    release = threading.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session

        def blocked_flush(_source):
            entered.set()
            release.wait()
            return []

        session.flush = MagicMock(side_effect=blocked_flush)
        first_stop = asyncio.create_task(client.post(f"/api/jobs/live/{job_id}/stop"))
        assert await asyncio.to_thread(entered.wait, 0.5)
        second_stop = await client.post(f"/api/jobs/live/{job_id}/stop")
        assert second_stop.status_code == 404
        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 404

        release.set()
        assert (await first_stop).status_code == 200

    session.flush.assert_called_once_with("mic")
    events = registry.events_after(job_id, 0)
    assert [event["type"] for event in events].count("complete") == 1


@pytest.mark.asyncio
async def test_claim_context_releases_after_processing_exception() -> None:
    session = MagicMock()
    handle = server.LiveSessionHandle(session, idle_timeout_sec=0)

    with pytest.raises(RuntimeError, match="processing failed"):
        async with handle.claim_chunk():
            raise RuntimeError("processing failed")

    assert await handle.claim_finish() is session
    await asyncio.wait_for(handle.wait_for_drain(), timeout=0.1)


@pytest.mark.asyncio
async def test_cancel_wins_against_inflight_chunk_without_late_progress(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=_decoded_pcm()))

    entered = threading.Event()
    release = threading.Event()
    progress_acceptance: list[bool] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session

        def blocked_process(_pcm, _source):
            entered.set()
            release.wait()
            progress_acceptance.append(
                registry.update_event(job_id, "progress", {"current": 1, "total": 1})
            )
            return []

        session.process_pcm_chunk = blocked_process
        chunk = asyncio.create_task(
            client.post(
                f"/api/jobs/live/{job_id}/chunk?sequence=1",
                files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
            )
        )
        assert await asyncio.to_thread(entered.wait, 0.5)

        cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        release.set()
        assert (await chunk).status_code == 200

    assert progress_acceptance == [False]
    assert registry.get_status(job_id)["state"] == "cancelled"
    assert server.active_live_sessions == {}

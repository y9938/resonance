import asyncio
import threading
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

import server
from core.jobs import JobRegistry


def _isolated_live_registry(monkeypatch) -> JobRegistry:
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    monkeypatch.setattr(server.Config, "LIVE_STT_IDLE_TIMEOUT_SEC", 0)
    server.active_live_sessions.clear()
    return registry


async def _start_live_job(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/jobs/live/start?language=ru&model=gigaam")
    assert response.status_code == 200
    return response.json()["job_id"]


def _decoded_pcm() -> MagicMock:
    decoded = MagicMock()
    decoded.as_ndarray.return_value = np.zeros(512, dtype=np.float32)
    return decoded


@pytest.mark.asyncio
async def test_sequence_success_commits_ack_and_duplicate_has_no_side_effects(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decode = MagicMock(return_value=_decoded_pcm())
    monkeypatch.setattr(server, "decode_media_bytes", decode)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session
        process = MagicMock(return_value=[])
        session.process_pcm_chunk = process

        first = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        duplicate = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert first.json() == {"emitted": 0, "ack_sequence": 1}
    assert duplicate.json() == {"emitted": 0, "ack_sequence": 1}
    decode.assert_called_once()
    process.assert_called_once()


@pytest.mark.asyncio
async def test_retry_after_lost_response_publishes_progress_once(monkeypatch) -> None:
    registry = _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "decode_media_bytes", MagicMock(return_value=_decoded_pcm()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session

        def emit_one_progress(_pcm, _source):
            registry.update_event(
                job_id,
                "progress",
                {"current": 1, "total": 1, "segment": {"text": "once"}},
            )
            return [{"text": "once"}]

        session.process_pcm_chunk = MagicMock(side_effect=emit_one_progress)
        await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        retry = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert retry.json()["ack_sequence"] == 1
    assert [event["type"] for event in registry.events_after(job_id, 0)].count("progress") == 1


@pytest.mark.asyncio
async def test_future_sequence_is_rejected_without_decode_or_processing(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decode = MagicMock(return_value=_decoded_pcm())
    monkeypatch.setattr(server, "decode_media_bytes", decode)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session
        process = MagicMock(return_value=[])
        session.process_pcm_chunk = process
        await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        decode.reset_mock()
        process.reset_mock()
        future = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=3",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert future.status_code == 409
    assert future.json() == {"expected_sequence": 2}
    decode.assert_not_called()
    process.assert_not_called()


@pytest.mark.asyncio
async def test_corrupt_expected_sequence_does_not_commit_it(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decode = MagicMock(return_value=_decoded_pcm())
    monkeypatch.setattr(server, "decode_media_bytes", decode)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session
        session.process_pcm_chunk = MagicMock(return_value=[])
        await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        decode.side_effect = ValueError("bad wav")
        invalid = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=2",
            files={"file": ("chunk.wav", b"invalid-wav", "audio/wav")},
        )
        decode.side_effect = None
        valid = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=2",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert invalid.status_code == 400
    assert valid.json()["ack_sequence"] == 2


@pytest.mark.asyncio
async def test_concurrent_duplicate_sequence_processes_once(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decode = MagicMock(return_value=_decoded_pcm())
    monkeypatch.setattr(server, "decode_media_bytes", decode)

    entered = threading.Event()
    release = threading.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session

        def blocked_process(_pcm, _source):
            entered.set()
            release.wait()
            return []

        process = MagicMock(side_effect=blocked_process)
        session.process_pcm_chunk = process
        first = asyncio.create_task(client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        ))
        assert await asyncio.to_thread(entered.wait, 0.5)
        second = asyncio.create_task(client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        ))
        release.set()
        responses = await asyncio.gather(first, second)

    assert [response.json()["ack_sequence"] for response in responses] == [1, 1]
    process.assert_called_once()
    decode.assert_called_once()


@pytest.mark.asyncio
async def test_invalid_sequence_boundary_and_retry_after_stop(monkeypatch) -> None:
    _isolated_live_registry(monkeypatch)
    monkeypatch.setattr(server.models, "get_stt_model", MagicMock(return_value=MagicMock()))
    decode = MagicMock(return_value=_decoded_pcm())
    monkeypatch.setattr(server, "decode_media_bytes", decode)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    ) as client:
        job_id = await _start_live_job(client)
        session = server.active_live_sessions[job_id].session
        process = MagicMock(return_value=[])
        session.process_pcm_chunk = process
        for suffix in ("", "?sequence=0", "?sequence=-1"):
            response = await client.post(
                f"/api/jobs/live/{job_id}/chunk{suffix}",
                files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
            )
            assert response.status_code == 422
        decode.assert_not_called()
        process.assert_not_called()

        accepted = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )
        assert accepted.status_code == 200
        assert (await client.post(f"/api/jobs/live/{job_id}/stop")).status_code == 200
        retry = await client.post(
            f"/api/jobs/live/{job_id}/chunk?sequence=1",
            files={"file": ("chunk.wav", b"valid-wav", "audio/wav")},
        )

    assert retry.status_code == 404

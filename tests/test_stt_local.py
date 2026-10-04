"""Local batch API: filesystem authority, shared lifecycle and direct path decoding."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

import server
from core.jobs import JobRegistry
from tests.media_helpers import silent_wav


@pytest.fixture
def local_server(monkeypatch, local_host_app):
    monkeypatch.setattr(server, "jobs", JobRegistry())
    monkeypatch.setattr(server.models, "get_stt_model", lambda name: MagicMock(spec=server.WhisperAdapter))
    monkeypatch.setattr(server.Config, "STT_MAX_DURATION_SEC", 0)
    return httpx.ASGITransport(app=server.app, client=("127.0.0.1", 12345))


async def terminal(client, job_id):
    for _ in range(500):
        status = (await client.get(f"/api/jobs/{job_id}")).json()
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.01)
    pytest.fail(f"Job did not finish: {status}")


@pytest.mark.asyncio
async def test_local_file_uses_path_and_restores_batch_without_encoded_copy(local_server, tmp_path, monkeypatch):
    source = tmp_path / "private" / "clip.wav"
    source.parent.mkdir()
    source.write_bytes(silent_wav(16001).data)
    second_source = source.with_name("clip-2.wav")
    second_source.write_bytes(silent_wav(16000).data)
    seen = []
    worker = server.run_stt_worker

    def record_source(**kwargs):
        seen.append(kwargs["audio_path"])
        worker(**kwargs)

    monkeypatch.setattr(server, "run_stt_worker", record_source)
    monkeypatch.setattr(Path, "read_bytes", MagicMock(side_effect=AssertionError("full file read")))
    monkeypatch.setattr(server, "EncodedMedia", MagicMock(side_effect=AssertionError("encoded copy")))
    async with httpx.AsyncClient(transport=local_server, base_url="http://localhost") as client:
        assert (await client.get("/api/config")).json()["local_files_enabled"]
        responses = await asyncio.gather(*(
            client.post(
                f"/api/jobs/stt/local?batch_id=local-batch&batch_index={index}&batch_total=2&detect_language=true",
                json={"path": str(path)}, headers={"Origin": "http://localhost"},
            ) for index, path in enumerate((source, second_source), 1)
        ))
        response = responses[0]
        assert (await terminal(client, responses[1].json()["job_id"]))["state"] == "completed"
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        status = await terminal(client, job_id)
        assert status["state"] == "completed", status
        assert status["result"]["duration"] == pytest.approx(16001 / 16000)
        assert status["model"] == "whisper"
        assert status["language"] is None
        async with httpx.AsyncClient(
            transport=local_server, base_url="http://localhost", cookies=client.cookies,
        ) as restored:
            history = (await restored.get("/api/jobs")).json()["jobs"]
            assert len(history) == 2
            history.sort(key=lambda job: job["batch_index"])
            assert history[0]["filename"] == "clip.wav"
            assert (history[0]["batch_id"], history[0]["batch_index"], history[0]["batch_total"]) == ("local-batch", 1, 2)
            assert (await restored.get(f"/api/jobs/{job_id}")).json() == status
            assert str(source.parent) not in str(history) + str(status)
    assert sorted(seen) == sorted([str(source), str(second_source)])


@pytest.mark.asyncio
@pytest.mark.parametrize("peer,host,headers,enabled", [
    ("192.0.2.1", "localhost", {}, True),
    ("192.0.2.1", "localhost", {"X-Forwarded-For": "127.0.0.1"}, True),
    ("127.0.0.1", "evil.example", {}, True),
    ("127.0.0.1", "localhost", {"Origin": "https://evil.example"}, True),
    ("127.0.0.1", "localhost", {"Origin": "null"}, True),
    ("127.0.0.1", "localhost", {"Origin": "http://localhost:9999"}, True),
    ("127.0.0.1", "localhost", {"Sec-Fetch-Site": "cross-site"}, True),
    ("127.0.0.1", "localhost", {"Forwarded": "for=192.0.2.1"}, True),
    ("127.0.0.1", "localhost", {}, False),
])
async def test_remote_cross_origin_and_default_server_cannot_use_host_capabilities(local_server, monkeypatch, peer, host, headers, enabled):
    monkeypatch.setattr(server.app.state, "local_host", enabled)
    worker = MagicMock()
    monkeypatch.setattr(server, "run_stt_worker", worker)
    capture = MagicMock()
    monkeypatch.setattr(server, "get_system_audio_capture", capture)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=(peer, 12345)), base_url=f"http://{host}",
    ) as client:
        response = await client.post("/api/jobs/stt/local", json={"path": "/etc/passwd"}, headers=headers)
        assert response.status_code == 403
        for endpoint in ("/api/system-audio/start", "/api/system-audio/stop?job_id=unavailable"):
            assert (await client.post(endpoint, headers=headers)).status_code == 403
        config = (await client.get("/api/config", headers=headers)).json()
        assert not config["local_files_enabled"]
        assert not config["system_audio_enabled"]
    worker.assert_not_called()
    capture.assert_not_called()


@pytest.mark.asyncio
async def test_system_audio_preference_does_not_disable_local_files(local_server, monkeypatch):
    monkeypatch.setattr(server.Config, "ENABLE_SYSTEM_AUDIO", False)
    async with httpx.AsyncClient(transport=local_server, base_url="http://localhost") as client:
        config = (await client.get("/api/config")).json()
        assert config["local_files_enabled"]
        assert not config["system_audio_enabled"]
        for endpoint in ("/api/system-audio/start", "/api/system-audio/stop?job_id=unavailable"):
            assert (await client.post(endpoint)).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "directory", "unreadable", "invalid"])
async def test_bad_local_source_is_a_failed_job_without_path_leak(local_server, tmp_path, kind):
    source = tmp_path / "private-media"
    if kind == "directory":
        source.mkdir()
    elif kind in {"invalid", "unreadable"}:
        source.write_bytes(b"invalid media")
        if kind == "unreadable":
            source.chmod(0)
    try:
        async with httpx.AsyncClient(transport=local_server, base_url="http://localhost") as client:
            response = await client.post("/api/jobs/stt/local", json={"path": str(source)})
            assert response.status_code == 200
            status = await terminal(client, response.json()["job_id"])
            assert status["state"] == "failed"
            assert status["error"]
            assert str(tmp_path) not in str(status)
    finally:
        if kind == "unreadable":
            source.chmod(0o600)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_queued_source_lifetime_and_cancel_before_open(local_server, tmp_path, monkeypatch, cancel):
    source = tmp_path / "queued.wav"
    source.write_bytes(silent_wav(16000).data)
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    monkeypatch.setattr(server, "STT_WORKER_SEMAPHORE", semaphore)
    finished = threading.Event()
    worker = server.run_stt_worker

    def notify_finished(**kwargs):
        try:
            worker(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(server, "run_stt_worker", notify_finished)
    async with httpx.AsyncClient(transport=local_server, base_url="http://localhost") as client:
        response = await client.post("/api/jobs/stt/local", json={"path": str(source)})
        job_id = response.json()["job_id"]
        try:
            assert (await client.get(f"/api/jobs/{job_id}")).json()["state"] == "queued"
            source.unlink()
            if cancel:
                assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
        finally:
            semaphore.release()
        assert await asyncio.to_thread(finished.wait, 5)
        status = await terminal(client, job_id)
        assert status["state"] == ("cancelled" if cancel else "failed")
        if cancel:
            assert status["error"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("diarization", [False, True])
async def test_local_api_enforces_decoded_limit(local_server, tmp_path, monkeypatch, diarization):
    from stt import pipeline

    source = tmp_path / "over-limit.wav"
    source.write_bytes(silent_wav(16001).data)
    monkeypatch.setattr(server.Config, "STT_MAX_DURATION_SEC", 1)
    probe = pipeline.probe_media
    monkeypatch.setattr(pipeline, "probe_media", lambda path: pipeline.MediaInfo(0, None, 16000, 1, probe(path).size_bytes))
    async with httpx.AsyncClient(transport=local_server, base_url="http://localhost") as client:
        response = await client.post(
            f"/api/jobs/stt/local?diarization={str(diarization).lower()}", json={"path": str(source)},
        )
        status = await terminal(client, response.json()["job_id"])
        assert status["state"] == "failed"
        assert status["error"] == "Audio too long (max 1s)"

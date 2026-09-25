import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

import server
from core.context import SessionContextManager
from core.jobs import JobRegistry
from stt.live import LiveSTTSession
from stt.live_transport import LiveSessionHandle


@pytest.mark.asyncio
async def test_transport_expiry_does_not_clear_context(monkeypatch):
    manager = SessionContextManager()
    monkeypatch.setattr("stt.live.session_context_manager", manager)
    monkeypatch.setattr("core.context.session_context_manager", manager)
    registry = JobRegistry()
    monkeypatch.setattr(server, "jobs", registry)
    monkeypatch.setattr(server, "active_live_sessions", {})

    session_id = "session"
    record = registry.create("stt", session_id, {"source": "mic_live"})
    live_session = LiveSTTSession(
        job_id=record.job_id,
        session_id=session_id,
        model=MagicMock(),
        jobs=registry,
        sample_rate=16000,
        source="mic",
        preview_broker=server.live_preview_broker,
    )
    handle = LiveSessionHandle(live_session, idle_timeout_sec=0.15)
    server.active_live_sessions[record.job_id] = handle
    manager.append_live(session_id, "committed speech")

    handle.start_expiry_task(lambda s: server._expire_live_session(record.job_id, handle, s))
    await asyncio.sleep(0.35)

    assert record.job_id not in server.active_live_sessions
    assert manager.get_session_tail(session_id) == ["committed speech"]
    assert manager.get_ipc_tail() == ["committed speech"]


@pytest.mark.asyncio
async def test_http_context_tail_isolated_and_validated(monkeypatch):
    manager = SessionContextManager()
    monkeypatch.setattr("core.context.session_context_manager", manager)
    for index in range(3):
        manager.append_session("session-a", f"line {index}")
    manager.append_session("session-b", "B only")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://test",
        cookies={server.SESSION_COOKIE: "session-a"},
    ) as client:
        response = await client.get("/api/context/tail?lines=2")
        assert response.status_code == 200
        assert response.json()["lines"] == ["line 1", "line 2"]
        assert (await client.get("/api/context/tail?lines=0")).status_code == 422

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://test",
        cookies={server.SESSION_COOKIE: "session-b"},
    ) as client:
        response = await client.get("/api/context/tail")
        assert response.json()["lines"] == ["B only"]

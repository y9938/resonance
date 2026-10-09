"""Production assets and development document origin are served by FastAPI."""

import re
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from core.frontend import mount_frontend
from server import local_host_allowed

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_development_rejects_missing_entry_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("RESONANCE_FRONTEND_DEV", "1")
    frontend = tmp_path / "src/web"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text('<html><body></body></html>')
    app = FastAPI()
    mount_frontend(app, tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000",
    ) as client:
        with pytest.raises(RuntimeError, match="Frontend entry marker is missing"):
            await client.get("/")


@pytest.mark.asyncio
@pytest.mark.parametrize("development", [False, True])
async def test_frontend_document_and_local_capability(monkeypatch, development):
    monkeypatch.setenv("RESONANCE_FRONTEND_DEV", "1" if development else "0")
    monkeypatch.setenv("RESONANCE_VITE_ORIGIN", "http://127.0.0.1:49321")
    app = FastAPI()
    app.state.local_host = True

    @app.get("/api/capability")
    def capability(request: Request):
        return {"allowed": local_host_allowed(request)}

    mount_frontend(app, ROOT)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        page = await client.get("/")
        assert page.status_code == 200
        if development:
            assert "http://127.0.0.1:49321/@vite/client" in page.text
            assert "http://127.0.0.1:49321/main.ts" in page.text
        else:
            assert "http://127.0.0.1:49321" not in page.text
            scripts = re.findall(r'<script[^>]+src="([^"]+)"', page.text)
            assert scripts
            for url in scripts:
                asset = await client.get(url)
                assert asset.status_code == 200
                assert "immutable" in asset.headers["cache-control"]
            assert "immutable" not in page.headers["cache-control"]
            license_response = await client.get("/third-party/licenses.md")
            assert license_response.status_code == 200
            assert license_response.text.strip()
        assert (await client.get("/api/capability", headers={"origin": "http://localhost:8000"})).json()["allowed"]
        assert not (await client.get("/api/capability", headers={"origin": "http://localhost:5173"})).json()["allowed"]
        assert not (await client.get("/api/capability", headers={"x-forwarded-for": "127.0.0.1"})).json()["allowed"]
        app.state.local_host = False
        assert not (await client.get("/api/capability")).json()["allowed"]

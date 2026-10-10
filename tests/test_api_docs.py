"""Public documentation must not grant access to host capture or local files."""

import pytest
from fastapi.testclient import TestClient

import server


@pytest.mark.parametrize(
    ("enabled", "address", "request_url", "headers"),
    [
        (False, "127.0.0.1", "http://localhost", {}),
        (True, "203.0.113.1", "http://localhost", {}),
        (True, "127.0.0.1", "http://example.com", {}),
        (True, "127.0.0.1", "http://localhost", {"origin": "https://example.com"}),
        (True, "127.0.0.1", "http://localhost", {"forwarded": "for=127.0.0.1"}),
        (True, "127.0.0.1", "http://localhost", {"x-forwarded-for": "127.0.0.1"}),
        (True, "127.0.0.1", "http://localhost", {"sec-fetch-site": "cross-site"}),
    ],
)
def test_public_docs_preserve_host_feature_restrictions(
    monkeypatch, enabled, address, request_url, headers,
):
    monkeypatch.setattr(server.app.state, "local_host", enabled, raising=False)
    client = TestClient(server.app, base_url=request_url, client=(address, 12345))
    assert client.get("/docs", headers=headers).status_code == 200
    assert client.get("/openapi.json", headers=headers).status_code == 200
    for path, body in (
        ("/api/system-audio/start", None),
        ("/api/system-audio/stop?job_id=test", None),
        ("/api/jobs/stt/local", {"path": "/private/audio.wav"}),
    ):
        assert client.post(path, json=body, headers=headers).status_code == 403


def test_local_docs_generate_current_schema(local_host_app):
    client = TestClient(local_host_app, base_url="http://localhost", client=("127.0.0.1", 12345))
    page = client.get("/docs")
    assert page.status_code == 200
    assert "SwaggerUIBundle" in page.text
    assert "/openapi.json" in page.text
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["openapi"].startswith("3.")
    assert "post" in schema["paths"]["/api/jobs/stt"]
    assert "post" in schema["paths"]["/api/system-audio/start"]
    assert "/docs" not in schema["paths"]
    assert "/openapi.json" not in schema["paths"]

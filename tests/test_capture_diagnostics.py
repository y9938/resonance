import logging

from fastapi.testclient import TestClient

import server

PAYLOAD = {"event": "mic-failed", "mode": "dictation", "stage": "permission", "error": "NotAllowedError"}


def test_local_microphone_permission_failure_is_logged(local_host_app, monkeypatch, caplog):
    monkeypatch.setattr(server.log, "propagate", True)
    client = TestClient(local_host_app, base_url="http://localhost", client=("127.0.0.1", 12345))
    with caplog.at_level(logging.INFO):
        response = client.post("/api/diagnostics/capture", json=PAYLOAD,
                               headers={"user-agent": "Safari\nforged log entry"})
    assert response.status_code == 204
    assert not response.content
    assert "NotAllowedError" in caplog.text
    assert "stage=permission" in caplog.text
    # Untrusted browser metadata stays on one line.
    assert "Safari\\nforged log entry" in caplog.text
    assert len([record for record in caplog.records if record.name == "resonance.server"]) == 1


def test_capture_diagnostics_reject_remote_origin_and_unbounded_fields(local_host_app):
    client = TestClient(local_host_app, base_url="http://localhost", client=("127.0.0.1", 12345))
    assert client.post("/api/diagnostics/capture", json=PAYLOAD,
                       headers={"origin": "https://unrelated.example"}).status_code == 403
    assert client.post("/api/diagnostics/capture", json={**PAYLOAD, "error": "arbitrary text"}).status_code == 422
    assert client.post("/api/diagnostics/capture", json={**PAYLOAD, "event": "x" * 10000}).status_code == 422
    remote = TestClient(local_host_app, base_url="http://localhost", client=("203.0.113.1", 12345))
    assert remote.post("/api/diagnostics/capture", json=PAYLOAD).status_code == 403

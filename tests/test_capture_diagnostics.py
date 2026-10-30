import logging

from fastapi.testclient import TestClient

import server

PAYLOAD = {"event": "mic-failed", "mode": "dictation", "stage": "permission", "error": "NotAllowedError"}


def test_local_microphone_permission_failure_is_logged(local_host_app, monkeypatch, caplog):
    monkeypatch.setattr(server.log, "propagate", True)
    monkeypatch.setattr(logging.getLogger("resonance"), "propagate", True)
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


def test_local_dictation_metrics_are_bounded_and_correlated_to_the_stt_job(
    local_host_app, monkeypatch, caplog,
):
    monkeypatch.setattr(server.log, "propagate", True)
    monkeypatch.setattr(logging.getLogger("resonance"), "propagate", True)
    payload = {
        "event": "encoded",
        "capture_id": "79edaf14-46d8-4a44-bbe0-90fcd0d93899",
        "capture_wall_ms": 61_000,
        "audio_duration_s": 60,
        "sample_rate_hz": 48_000,
        "pcm_buffer_bytes": 11_520_000,
        "pcm_chunk_count": 704,
        "merge_wall_ms": 14,
        "wav_encode_wall_ms": 900,
        "wav_bytes": 5_760_044,
    }
    client = TestClient(local_host_app, base_url="http://localhost", client=("127.0.0.1", 12345))

    with caplog.at_level(logging.DEBUG, logger="resonance.server"):
        encoded = client.post("/api/diagnostics/capture", json=payload)
        submitted = client.post(
            "/api/diagnostics/capture",
            json={
                **payload,
                "event": "submission-finished",
                "job_id": "test-job_123",
                "job_id_returned": True,
                "send_to_job_response_ms": 1500,
            },
        )

    assert encoded.status_code == submitted.status_code == 204
    assert "event=encoded" in caplog.text
    assert "pcm_buffer_mib=10.99" in caplog.text
    assert "event=submission-finished" in caplog.text
    assert "job_id=test-job_123" in caplog.text
    assert "send_to_job_response_ms=1500.0" in caplog.text
    assert client.post(
        "/api/diagnostics/capture", json={**payload, "pcm_chunk_count": 10_000_001},
    ).status_code == 422
    assert client.post(
        "/api/diagnostics/capture",
        json={**payload, "job_id": "unexpected", "job_id_returned": True,
              "send_to_job_response_ms": 10},
    ).status_code == 422
    assert client.post(
        "/api/diagnostics/capture",
        json={**payload, "event": "submission-finished", "job_id": "job-1",
              "job_id_returned": False, "send_to_job_response_ms": 10},
    ).status_code == 422

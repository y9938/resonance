import io
import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server import JobRegistry, active_live_sessions, app, jobs
from stt.models.manager import resolve_stt_model

client = TestClient(app)


def _make_minimal_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 320)
    return buf.getvalue()


VALID_WAV_BYTES = _make_minimal_wav_bytes()


def test_api_models_includes_languages_routing():
    with patch("whisper.load_model") as load_weights:
        response = client.get("/api/models")
    load_weights.assert_not_called()
    assert response.status_code == 200
    payload = response.json()
    assert "stt" in payload
    models = {entry["id"]: entry for entry in payload["stt"]["models"]}
    assert {"de", "en", "ru"} <= set(models["whisper"]["languages"])
    assert isinstance(models["whisper"]["loaded"], bool)
    assert set(payload["stt"]["language_names"]) == set(models["whisper"]["languages"])


def test_stt_routing_unsupported_language():
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post("/api/jobs/stt?language=xx-unknown", files=files)
    assert response.status_code == 400


@patch("server.run_stt_worker")
@patch("server.models.stt_gigaam")
def test_stt_routing_default_language(mock_stt_gigaam, mock_run_stt_worker):
    mock_stt_gigaam.return_value = MagicMock()
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post("/api/jobs/stt", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "ru"
    assert status["model"] == "gigaam"


@patch("server.run_stt_worker")
@patch("server.models.stt_whisper")
def test_stt_routing_english(mock_stt_whisper, mock_run_stt_worker):
    mock_stt_whisper.return_value = MagicMock()
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post("/api/jobs/stt?language=en", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "whisper"


@patch("server.run_stt_worker")
@patch("server.models.stt_whisper")
def test_batch_auto_uses_whisper_without_explicit_language(mock_stt_whisper, worker):
    mock_stt_whisper.return_value = MagicMock()
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post("/api/jobs/stt?detect_language=true", files=files)
    assert response.status_code == 200
    status = jobs.get_status(response.json()["job_id"])
    assert status["model"] == "whisper"
    assert status["language"] is None
    assert worker.call_args.kwargs["detect_language"] is True


@pytest.mark.parametrize("query", [
    "detect_language=true&language=de", "detect_language=true&model=gigaam",
    "detect_language=true&model=granite",
])
def test_batch_auto_rejects_conflicting_routing(query):
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post(f"/api/jobs/stt?{query}", files=files)
    assert response.status_code == 400


@patch("server.run_stt_worker")
@patch("server.models.stt_whisper")
@pytest.mark.parametrize("language", ["de", "ru"])
def test_explicit_whisper_supports_multilingual_batch(mock_stt_whisper, mock_run_stt_worker, language):
    mock_stt_whisper.return_value = MagicMock()
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post(f"/api/jobs/stt?language={language}&model=whisper", files=files)
    assert response.status_code == 200
    status = jobs.get_status(response.json()["job_id"])
    assert status["language"] == language
    assert status["model"] == "whisper"


def test_stt_job_registry_fields():
    registry = JobRegistry()
    rec = registry.create("stt", "session-1", language="en", model="whisper")
    
    status = registry.get_status(rec.job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "whisper"
    
    list_payload = registry.list_for_session("session-1", limit=10)
    assert len(list_payload["jobs"]) == 1
    job_item = list_payload["jobs"][0]
    assert job_item["language"] == "en"
    assert job_item["model"] == "whisper"

    rec_none = registry.create("stt", "session-1")
    list_payload_none = registry.list_for_session("session-1", limit=10)
    job_items_by_id = {j["job_id"]: j for j in list_payload_none["jobs"]}
    
    job_none_item = job_items_by_id[rec_none.job_id]
    assert "language" not in job_none_item
    assert "model" not in job_none_item
    
    status_none = registry.get_status(rec_none.job_id)
    assert status_none is not None
    assert status_none["language"] is None
    assert status_none["model"] is None


@patch("server.run_stt_worker")
@patch("server.models.stt_granite")
def test_stt_routing_granite(mock_stt_granite, mock_run_stt_worker):
    mock_stt_granite.return_value = MagicMock()
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}

    response = client.post("/api/jobs/stt?language=en&model=granite", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "granite"
    assert status["result"].get("diarization") is not True

    response = client.post("/api/jobs/stt?language=en&model=granite&diarization=true", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "granite"
    assert status["result"].get("diarization") is True


def test_stt_routing_invalid_combinations():
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}

    response = client.post("/api/jobs/stt?language=en&model=invalid_model", files=files)
    assert response.status_code == 400

    response = client.post("/api/jobs/stt?language=en&model=gigaam", files=files)
    assert response.status_code == 400

    response = client.post("/api/jobs/stt?model=whisper", files=files)
    assert response.status_code == 400

    response = client.post("/api/jobs/stt?language=en&model=whisper&diarization=true", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status["result"].get("diarization") is True


@patch("server.models.get_stt_model")
def test_live_route_retains_resolved_language(mock_get_model):
    mock_get_model.return_value = MagicMock()
    response = client.post("/api/jobs/live/start?language=de&model=whisper")
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert active_live_sessions[job_id].session.language == "de"
    assert client.post(f"/api/jobs/live/{job_id}/stop").status_code == 200


@pytest.mark.parametrize(("language", "model", "expected"), [
    (None, None, ("ru", "gigaam")),
    ("ru", None, ("ru", "gigaam")),
    ("en", None, ("en", "whisper")),
    ("de", None, ("de", "whisper")),
    ("ru", "whisper", ("ru", "whisper")),
    (None, "gigaam", ("ru", "gigaam")),
    (None, "granite", ("en", "granite")),
])
def test_resolve_multilingual_stt(language, model, expected):
    assert resolve_stt_model(language, model) == expected


@pytest.mark.parametrize(("language", "model"), [
    (None, "whisper"), ("auto", None), ("en-US", None),
    ("en", "gigaam"), ("de", "granite"),
])
def test_reject_ambiguous_or_unsupported_language(language, model):
    with pytest.raises(ValueError):
        resolve_stt_model(language, model)

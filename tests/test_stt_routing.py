import io
import wave
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from server import JobRegistry, app, jobs

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
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert "stt" in payload
    assert "languages" in payload["stt"]
    assert "granite" in payload["stt"]
    assert payload["stt"]["granite"]["name"] == "IBM Granite Speech 4.1 Plus"
    assert payload["stt"]["languages"] == {"ru": "gigaam", "en": "whisper"}


def test_stt_routing_unsupported_language():
    files = {"file": ("test.wav", VALID_WAV_BYTES, "audio/wav")}
    response = client.post("/api/jobs/stt?language=fr", files=files)
    assert response.status_code == 400
    assert "Unsupported language" in response.json()["detail"]


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
    assert "Unsupported model" in response.json()["detail"]

    response = client.post("/api/jobs/stt?language=ru&model=whisper", files=files)
    assert response.status_code == 400
    assert "Russian language only supports gigaam model" in response.json()["detail"]

    response = client.post("/api/jobs/stt?language=en&model=gigaam", files=files)
    assert response.status_code == 400
    assert "English language does not support gigaam model" in response.json()["detail"]

    response = client.post("/api/jobs/stt?language=en&model=whisper&diarization=true", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status["result"].get("diarization") is True

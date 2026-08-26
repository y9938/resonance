from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from server import app, jobs, JobRegistry

client = TestClient(app)

def test_api_models_includes_languages_routing():
    # 1. test_api_models_includes_languages_routing: Get /api/models and assert "languages" exists nested under "stt".
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert "stt" in payload
    assert "languages" in payload["stt"]
    assert "granite" in payload["stt"]
    assert payload["stt"]["granite"]["name"] == "IBM Granite Speech 4.1 Plus"
    assert payload["stt"]["languages"] == {"ru": "gigaam", "en": "whisper"}


def test_stt_routing_unsupported_language():
    # 2. test_stt_routing_unsupported_language: Post to /api/jobs/stt?language=fr and assert status 400.
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/api/jobs/stt?language=fr", files=files)
    assert response.status_code == 400
    assert "Unsupported language" in response.json()["detail"]


@patch("server.run_stt_worker")
@patch("server.models.stt_gigaam")
def test_stt_routing_default_language(mock_stt_gigaam, mock_run_stt_worker):
    # 3. test_stt_routing_default_language: Post to /api/jobs/stt with no language.
    # Verify language="ru" and model="gigaam" are set on the job.
    mock_stt_gigaam.return_value = MagicMock()
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
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
    # 4. test_stt_routing_english: Post to /api/jobs/stt?language=en.
    # Verify language="en" and model="whisper" are set on the job.
    mock_stt_whisper.return_value = MagicMock()
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    response = client.post("/api/jobs/stt?language=en", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "whisper"


def test_stt_job_registry_fields():
    # 5. test_stt_job_registry_fields: Instantiate JobRegistry, create job, and verify fields.
    registry = JobRegistry()
    rec = registry.create("stt", "session-1", language="en", model="whisper")
    
    # get_status must always contain language and model
    status = registry.get_status(rec.job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "whisper"
    
    # list_for_session must contain language and model ONLY if they are not None
    list_payload = registry.list_for_session("session-1", limit=10)
    assert len(list_payload["jobs"]) == 1
    job_item = list_payload["jobs"][0]
    assert job_item["language"] == "en"
    assert job_item["model"] == "whisper"

    # Verify that a job with None language and model doesn't return those keys in list_for_session
    rec_none = registry.create("stt", "session-1")
    list_payload_none = registry.list_for_session("session-1", limit=10)
    job_items_by_id = {j["job_id"]: j for j in list_payload_none["jobs"]}
    
    job_none_item = job_items_by_id[rec_none.job_id]
    assert "language" not in job_none_item
    assert "model" not in job_none_item
    
    # get_status of the None job should still contain them as None
    status_none = registry.get_status(rec_none.job_id)
    assert status_none is not None
    assert status_none["language"] is None
    assert status_none["model"] is None


@patch("server.run_stt_worker")
@patch("server.models.stt_granite")
def test_stt_routing_granite(mock_stt_granite, mock_run_stt_worker):
    mock_stt_granite.return_value = MagicMock()
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}

    # Without diarization
    response = client.post("/api/jobs/stt?language=en&model=granite", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "granite"
    assert status["result"].get("diarization") is not True

    # With diarization
    response = client.post("/api/jobs/stt?language=en&model=granite&diarization=true", files=files)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = jobs.get_status(job_id)
    assert status is not None
    assert status["language"] == "en"
    assert status["model"] == "granite"
    assert status["result"].get("diarization") is True


def test_stt_routing_invalid_combinations():
    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}

    # Invalid model name
    response = client.post("/api/jobs/stt?language=en&model=invalid_model", files=files)
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]

    # Russian language with whisper model
    response = client.post("/api/jobs/stt?language=ru&model=whisper", files=files)
    assert response.status_code == 400
    assert "Russian language only supports gigaam model" in response.json()["detail"]

    # English language with gigaam model
    response = client.post("/api/jobs/stt?language=en&model=gigaam", files=files)
    assert response.status_code == 400
    assert "English language does not support gigaam model" in response.json()["detail"]

    # Diarization with Whisper
    response = client.post("/api/jobs/stt?language=en&model=whisper&diarization=true", files=files)
    assert response.status_code == 400
    assert "Diarization is only supported by the granite model" in response.json()["detail"]

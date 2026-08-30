from core.jobs import JobRegistry


def test_job_registry_lifecycle():
    reg = JobRegistry()
    rec = reg.create("stt", "session_123", language="ru", model="gigaam")

    assert rec.job_id is not None
    assert rec.session_id == "session_123"
    assert rec.job_type == "stt"
    assert rec.state == "queued"
    assert rec.language == "ru"
    assert rec.model == "gigaam"
    assert reg.exists(rec.job_id) is True
    assert reg.belongs_to_session(rec.job_id, "session_123") is True
    assert reg.belongs_to_session(rec.job_id, "session_other") is False

    # Start event
    reg.update_event(rec.job_id, "start", {"total": 100, "duration": 15.5})
    st = reg.get_status(rec.job_id)
    assert st is not None
    assert st["state"] == "running"
    assert st["progress_total"] == 100
    assert st["progress_current"] == 0
    assert st["result"]["duration"] == 15.5

    # Progress event
    reg.update_event(rec.job_id, "progress", {"current": 50, "total": 100, "segment": {"text": "hello", "start": 0.0, "end": 1.5}})
    st = reg.get_status(rec.job_id)
    assert st["progress_current"] == 50
    assert len(st["result"]["segments"]) == 1

    # Complete event
    reg.update_event(rec.job_id, "complete", {"duration": 20.0})
    st = reg.get_status(rec.job_id)
    assert st["state"] == "completed"
    assert st["progress_current"] == 100
    assert st["result"]["duration"] == 20.0


def test_job_registry_cancellation():
    reg = JobRegistry()
    rec = reg.create("stt", "session_1")

    assert reg.is_cancelled(rec.job_id) is False
    assert reg.mark_cancelled(rec.job_id) is True
    assert reg.is_cancelled(rec.job_id) is True

    st = reg.get_status(rec.job_id)
    assert st["state"] == "cancelled"

    # Cancel all
    rec2 = reg.create("stt", "session_1")
    rec3 = reg.create("tts", "session_2")
    reg.cancel_all()
    assert reg.is_cancelled(rec2.job_id) is True
    assert reg.is_cancelled(rec3.job_id) is True


def test_job_registry_events_after():
    reg = JobRegistry()
    rec = reg.create("stt", "session_1")

    reg.update_event(rec.job_id, "start", {"total": 10})
    reg.update_event(rec.job_id, "progress", {"current": 5})
    reg.update_event(rec.job_id, "complete", {})

    evs_all = reg.events_after(rec.job_id, 0)
    assert len(evs_all) == 3
    assert evs_all[0]["seq"] == 1
    assert evs_all[1]["seq"] == 2
    assert evs_all[2]["seq"] == 3

    evs_after_1 = reg.events_after(rec.job_id, 1)
    assert len(evs_after_1) == 2
    assert evs_after_1[0]["seq"] == 2


def test_job_registry_list_for_session():
    reg = JobRegistry()
    rec1 = reg.create("stt", "session_a")
    reg.update_event(rec1.job_id, "progress", {"segment": {"start": 0.0, "end": 5.0, "text": "test"}})

    rec2 = reg.create("tts", "session_a")
    reg.update_event(rec2.job_id, "complete", {"download_url": "/api/stream/download?p=test.wav", "duration": 3.0})

    # rec for another session
    reg.create("stt", "session_b")

    page = reg.list_for_session("session_a", limit=10, offset=0)
    assert len(page["jobs"]) == 2
    assert page["has_more"] is False
    assert page["next_offset"] == 2

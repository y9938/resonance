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
    assert reg.update_event(rec.job_id, "start", {"total": 100, "duration": 15.5}) is True
    st = reg.get_status(rec.job_id)
    assert st is not None
    assert st["state"] == "running"
    assert st["progress_total"] == 100
    assert st["progress_current"] == 0
    assert st["result"]["duration"] == 15.5

    # Progress event
    assert reg.update_event(rec.job_id, "progress", {"current": 50, "total": 100, "segment": {"text": "hello", "start": 0.0, "end": 1.5}}) is True
    st = reg.get_status(rec.job_id)
    assert st["progress_current"] == 50
    assert len(st["result"]["segments"]) == 1

    # Complete event
    assert reg.update_event(rec.job_id, "complete", {"duration": 20.0}) is True
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


def test_job_status_includes_durable_event_cursor() -> None:
    reg = JobRegistry()
    rec = reg.create("stt", "session", {})
    assert reg.get_status(rec.job_id)["last_event_seq"] == 0
    reg.update_event(rec.job_id, "start", {"total": 0})
    reg.update_event(rec.job_id, "progress", {"current": 1, "total": 1})
    assert reg.get_status(rec.job_id)["last_event_seq"] == 2


def test_job_started_at_is_set_once_by_first_start_event() -> None:
    reg = JobRegistry()
    rec = reg.create("stt", "session", {})
    assert reg.get_status(rec.job_id)["started_at"] is None
    reg.update_event(rec.job_id, "start", {"total": 0})
    started_at = reg.get_status(rec.job_id)["started_at"]
    assert isinstance(started_at, float)
    reg.update_event(rec.job_id, "start", {"total": 0})
    assert reg.get_status(rec.job_id)["started_at"] == started_at


def test_job_registry_terminal_states_reject_all_later_events() -> None:
    cases = [
        ("complete", {"duration": 1.0}, "progress", {"current": 2, "segment": {"text": "late"}}),
        ("complete", {"duration": 1.0}, "start", {"total": 10}),
        ("error", {"message": "failed"}, "progress", {"current": 2}),
        ("error", {"message": "failed"}, "complete", {"duration": 2.0}),
        ("cancelled", {}, "progress", {"current": 2}),
        ("cancelled", {}, "complete", {"duration": 2.0}),
    ]

    for terminal_event, terminal_data, rejected_event, rejected_data in cases:
        reg = JobRegistry()
        rec = reg.create("stt", "session_1")
        assert reg.update_event(rec.job_id, "start", {"total": 10}) is True
        assert reg.update_event(rec.job_id, terminal_event, terminal_data) is True

        before_status = reg.get_status(rec.job_id)
        before_events = reg.events_after(rec.job_id, 0)
        assert before_status is not None

        assert reg.update_event(rec.job_id, rejected_event, rejected_data) is False
        assert reg.get_status(rec.job_id) == before_status
        assert reg.events_after(rec.job_id, 0) == before_events


def test_job_registry_terminal_states_reject_cancellation_without_mutation() -> None:
    terminal_events = [
        ("complete", {"duration": 1.0}),
        ("error", {"message": "failed"}),
        ("cancelled", {}),
    ]

    for event_type, event_data in terminal_events:
        reg = JobRegistry()
        rec = reg.create("stt", "session_1")
        assert reg.update_event(rec.job_id, event_type, event_data) is True

        before_status = reg.get_status(rec.job_id)
        before_events = reg.events_after(rec.job_id, 0)
        assert before_status is not None

        assert reg.mark_cancelled(rec.job_id) is False
        assert reg.get_status(rec.job_id) == before_status
        assert reg.events_after(rec.job_id, 0) == before_events


def test_job_registry_cancel_all_preserves_terminal_jobs() -> None:
    reg = JobRegistry()
    active = reg.create("stt", "session_1")
    completed = reg.create("stt", "session_1")
    assert reg.update_event(completed.job_id, "complete", {}) is True
    completed_before = reg.get_status(completed.job_id)
    completed_events = reg.events_after(completed.job_id, 0)

    reg.cancel_all()

    assert reg.is_cancelled(active.job_id) is True
    assert reg.get_status(completed.job_id) == completed_before
    assert reg.events_after(completed.job_id, 0) == completed_events


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

from server import JobRegistry


def test_stt_job_list_summary_includes_batch_metadata() -> None:
    jobs = JobRegistry()
    rec = jobs.create(
        "stt",
        "session-1",
        {
            "filename": "clip-02.wav",
            "batch_id": "batch-abc",
            "batch_index": 2,
            "batch_total": 10,
        },
    )

    payload = jobs.list_for_session("session-1", limit=10)

    assert payload["jobs"] == [
        {
            "job_id": rec.job_id,
            "job_type": "stt",
            "state": "queued",
            "progress_current": 0,
            "progress_total": 0,
            "error": None,
            "duration": None,
            "filename": "clip-02.wav",
            "batch_id": "batch-abc",
            "batch_index": 2,
            "batch_total": 10,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
    ]

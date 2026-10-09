from core.jobs import JobRegistry


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

    assert len(payload["jobs"]) == 1
    summary = payload["jobs"][0]
    assert summary["job_id"] == rec.job_id
    assert summary["filename"] == "clip-02.wav"
    assert summary["batch_id"] == "batch-abc"
    assert summary["batch_index"] == 2
    assert summary["batch_total"] == 10

"""Regression tests for jobs list refresh responsibilities in the client."""

from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "public" / "index.html"


def test_jobs_list_refresh_is_guarded_by_drawer_visibility() -> None:
    source = INDEX_HTML.read_text()

    assert "function refreshJobsListIfOpen()" in source
    assert "if (!els.jobsDrawer.classList.contains('open')) {" in source
    assert "refreshJobsListIfOpen();\n        }" in source


def test_active_job_lifecycle_does_not_call_jobs_list_directly() -> None:
    source = INDEX_HTML.read_text()

    forbidden_snippets = [
        "resetUI(type, false);\n            loadJobsList();",
        "setActiveJobId('STT', myJobId);\n                loadJobsList();",
        "setActiveJobId('TTS', payload.job_id);\n                loadJobsList();",
        "closeJobStreams('STT');\n                        loadJobsList();",
        "closeJobStreams('TTS');\n                        loadJobsList();",
    ]

    for snippet in forbidden_snippets:
        assert snippet not in source

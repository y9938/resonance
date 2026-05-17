from pathlib import Path


INDEX_HTML = Path(__file__).resolve().parent.parent / "public" / "index.html"
ZH_LOCALE = Path(__file__).resolve().parent.parent / "public" / "locales" / "zh-CN.js"


def test_stt_batch_frontend_contract_exists() -> None:
    html = INDEX_HTML.read_text()

    assert '<input type="file" id="sttFileInput" accept="audio/*,video/*" multiple>' in html
    assert 'id="sttBatchPanel"' in html
    assert 'id="sttBatchList"' in html
    assert 'function handleSttFiles(files)' in html
    assert 'function groupJobsForDisplay(jobs)' in html
    assert 'batch_id' in html


def test_stt_batch_ui_copy_and_shape_regressions() -> None:
    html = INDEX_HTML.read_text()
    zh = ZH_LOCALE.read_text()

    assert "jobsTypeStt: 'STT'" in html
    assert "jobsTypeTts: 'TTS'" in html
    assert "jobsBatchSummary: '{done} / {total}'" in html
    assert "jobsTypeStt: 'STT'" in zh
    assert "jobsTypeTts: 'TTS'" in zh
    assert "sttBatchTitle" in zh
    assert "jobsBatchSummary: '{done} / {total}'" in zh
    assert "badge.className = 'stt-batch-state ' + job.state" not in html
    assert "summary.append(titleBox, badge);" not in html
    assert "function clearSttBatchState()" in html
    assert "restoreJob(next.job_id, 'stt', { preserveScroll: true, preserveSttLayout: true });" in html


def test_stt_batch_followup_ui_contracts() -> None:
    html = INDEX_HTML.read_text()

    assert 'id="sttBatchCancelCurrent"' in html
    assert 'id="sttBatchDownloadAll"' in html
    assert "function cancelCurrentSttBatchJob()" in html
    assert "function downloadAllSttBatchTranscriptions()" in html
    assert "function openJobsBatchOnSttPage(batchId, jobs)" in html
    assert "jobs-batch-open" in html
    assert "restoreJob(job.job_id, 'stt', { preserveScroll: true, preserveSttLayout: true });" in html
    assert "sttResultText.style.minHeight" in html
    assert ".locale-option.active" in html
    assert "box-shadow: inset 0 0 0 1px" in html
    assert "target.scrollIntoView({ block: 'start', behavior: 'smooth' });" not in html
    assert "els.sttBatchNext.hidden" in html
    assert "els.sttBatchDownloadAll.hidden" in html
    assert "clearSttBatchState();" in html

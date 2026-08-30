from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "public" / "index.html"
RU_LOCALE = Path(__file__).resolve().parent.parent / "public" / "locales" / "ru.js"
ZH_LOCALE = Path(__file__).resolve().parent.parent / "public" / "locales" / "zh-CN.js"


def test_stt_batch_frontend_contract_exists() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert '<input type="file" id="sttFileInput" accept="audio/*,video/*" multiple>' in html
    assert 'id="sttBatchPanel"' in html
    assert 'id="sttBatchList"' in html
    assert 'function handleSttFiles(files)' in html
    assert 'function groupJobsForDisplay(jobs)' in html
    assert 'batch_id' in html


def test_stt_batch_ui_copy_and_shape_regressions() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    zh = ZH_LOCALE.read_text(encoding="utf-8")

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
    html = INDEX_HTML.read_text(encoding="utf-8")

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


def test_stt_language_dropdown_static() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    ru = RU_LOCALE.read_text(encoding="utf-8")
    zh = ZH_LOCALE.read_text(encoding="utf-8")

    assert 'id="sttLanguage"' in html
    assert 'sttLanguageLabel' in html
    assert 'sttLangRu' in html
    assert 'sttLangEn' in html

    # Verify keys exist in ru.js
    assert 'sttLanguageLabel' in ru
    assert 'sttLangRu' in ru
    assert 'sttLangEn' in ru

    # Verify keys exist in zh-CN.js
    assert 'sttLanguageLabel' in zh
    assert 'sttLangRu' in zh
    assert 'sttLangEn' in zh


def test_stt_language_js_contracts() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'resonance_sttLanguage' in html
    assert 'els.sttLanguage' in html
    assert 'language' in html


def test_system_audio_locale_contracts() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    ru = RU_LOCALE.read_text(encoding="utf-8")
    zh = ZH_LOCALE.read_text(encoding="utf-8")

    sys_keys = [
        'sttSysTitle',
        'sttSysHintIdle',
        'sttSysHintCapturing',
        'sttSysHintProcessing',
        'sttSysStart',
        'sttSysIncludeMic',
        'speakerMic',
        'speakerSys',
    ]

    for key in sys_keys:
        assert key in html, f"Key {key} missing from index.html fallback dictionary"
        assert key in ru, f"Key {key} missing from ru.js"
        assert key in zh, f"Key {key} missing from zh-CN.js"

    assert "Processing system audio..." not in html
    assert "setSttMicHint('sttSysHintProcessing')" in html
    assert 'id="sttSysIncludeMic"' in html
    assert 'id="sttSysIncludeMicContainer"' in html
    assert 'include_microphone' in html

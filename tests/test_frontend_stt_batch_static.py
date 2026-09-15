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


def test_live_preview_uses_separate_mutable_tail_contract() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'provisionalTail: new Map()' in html
    assert 'closedPreviewGenerations: new Map()' in html
    assert "case 'transcript_preview'" in html
    assert 'generation <= closed' in html
    assert 'provisionalTail.set(source' in html
    assert 'segments.push(event.segment)' in html
    assert "[committedBlocks, provisional].filter(Boolean).join('\\n\\n')" in html


def test_live_stt_hides_generic_controls_by_source_metadata() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function isLiveSttJob(status)" in html
    assert "source === 'mic_live' || source === 'system_audio'" in html
    assert "function updateSttJobControls(isLive)" in html
    assert "els.sttProgress.classList.remove('active');" in html
    assert "updateSttJobControls(true);" in html
    assert "updateSttJobControls(isLiveSttJob(status));" in html


def test_system_live_reattaches_from_durable_cursor_without_start_command() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function reattachSystemAudioUi(jobId, startedAtSec, filename = 'System Audio Capture.wav')" in html
    assert "state.stt.segments = state.stt.segments || [];" in html
    assert "els.sttResult.classList.add('active');" in html
    assert "fetchJobStatusForRestore('STT', sttJobId)" in html
    assert "const retryDelaysMs = [0, 250, 750];" in html
    assert "status.last_event_seq || 0" in html
    assert "function subscribeSttJobEvents(jobId, segments, updateDisplay, after = 0)" in html
    assert "events?after=" in html
    assert "state.stt.lastEventSeq = Math.max" in html


def test_system_live_timer_uses_backend_started_at_on_reattach() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function reattachSystemAudioUi(jobId, startedAtSec, filename = 'System Audio Capture.wav')" in html
    assert "startSttMicTimer(startedAtMs);" in html
    assert "status.started_at," in html
    assert "function startSttMicTimer(startedAtMs = Date.now())" in html
    assert "sttMicState.recordingStartTime = startedAtMs;" in html


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

    live_keys = [
        'sttModeDictation',
        'sttModeLive',
        'sttMicHintLive',
        'errLiveBackpressure',
    ]
    for key in live_keys:
        assert f'data-i18n="{key}"' in html or f"'{key}'" in html or f'"{key}"' in html, f"Key {key} missing from index.html"
        assert key in ru, f"Key {key} missing from ru.js"
        assert key in zh, f"Key {key} missing from zh-CN.js"

    assert "Processing system audio..." not in html
    assert "setSttMicHint('sttSysHintProcessing')" in html
    assert 'id="sttSysIncludeMic"' in html
    assert 'id="sttSysIncludeMicContainer"' in html
    assert 'include_microphone' in html


def test_live_microphone_transport_uses_server_vad_and_ordered_uploads() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "function createLiveTransport(jobId, sampleRate)" in html
    assert "function queueLiveChunk(transport)" in html
    assert "function pumpLiveTransport(transport)" in html
    assert "function sendPendingLiveChunk(transport, pending)" in html
    assert "pendingChunks: []," in html
    assert "pendingDurationSec: 0," in html
    assert "LIVE_PENDING_DURATION_LIMIT_SEC = 10" in html
    assert "LIVE_RETRY_DELAYS_MS = [0, 250, 750, 1500]" in html
    assert "liveTransport.sampleCount >= liveTransport.sampleRate" in html
    assert "await queueLiveChunk(transport);" in html
    assert "failed: false," in html
    assert "function failLiveTransport(transport, error)" in html
    assert "nextSequence: 1," in html
    assert "sequence: transport.nextSequence," in html
    assert "?sequence=${pending.sequence}" in html
    assert "payload.ack_sequence !== pending.sequence" in html
    assert "transport.pendingChunks.shift();" in html
    assert "transport.pendingDurationSec = Math.max" in html
    assert "response.status >= 500" in html
    assert "stopLiveCaptureForBackpressure(transport);" in html
    assert "if (!transport.failed && !transport.captureStoppedForBackpressure)" in html
    assert ".catch((err) => console.debug('Live chunk upload failed:', err))" not in html
    assert "consecutiveSilentBuffers" not in html
    assert "const rms =" not in html


def test_live_microphone_stop_releases_capture_guard_for_source_switching() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    start = html.index("async function stopSttMicRecording()")
    end = html.index("function closeJobStreams", start)
    stop_handler = html[start:end]
    assert "if (isLive) {\n                sttMicState.recordingStartTime = null;" in stop_handler
    assert html.count("sttMicState.recordingStartTime = null;") >= 4

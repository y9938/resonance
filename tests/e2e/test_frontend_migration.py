"""Behavior at the browser/API boundary after the Svelte migration."""

from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import expect


@pytest.fixture
def frontend_api(page, base_url):
    state = {"jobs": [], "held": [], "hold_upload": False, "hold_status": None, "stopped": []}
    page.add_init_script("""
        localStorage.setItem('resonance_locale', 'en');
        window.streams = [];
        window.EventSource = class {
            constructor(url) { this.url = url; this.closed = false; window.streams.push(this); }
            close() { this.closed = true; }
        };
    """)

    def route_api(route):
        parsed = urlparse(route.request.url)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/config":
            route.fulfill(json={"local_files_enabled": True, "system_audio_enabled": True, "tts_max_chars": 1000,
                                "tts": {"default_language": "en", "languages": [{"id": "en", "default_voice_id": "af_heart", "voices": [{"id": "af_heart", "backend_id": "kokoro_en"}]}]}})
        elif path == "/api/models":
            route.fulfill(json={"stt": {"models": [{"id": "whisper", "name": "Whisper", "languages": ["en", "ru"]}]}})
        elif path == "/api/jobs":
            route.fulfill(json={"jobs": state["jobs"], "has_more": False, "next_offset": len(state["jobs"])})
        elif path in {"/api/jobs/stt", "/api/jobs/tts"}:
            index = len(state["jobs"]) + 1
            job = {"job_id": f"job-{index}", "job_type": path.rsplit("/", 1)[1], "state": "running",
                   "filename": f"clip-{index}.wav", "progress_current": 0, "progress_total": 0,
                   "result": {"filename": f"clip-{index}.wav", "segments": []}}
            if "batch_id" in query:
                job.update(batch_id=query["batch_id"][0], batch_index=int(query["batch_index"][0]))
                job["result"]["batch_id"] = job["batch_id"]
            state["jobs"].append(job)
            if state["hold_upload"]:
                state["held"].append((route, job))
            else:
                route.fulfill(json={"job_id": job["job_id"]})
        elif path.endswith("/cancel") or path == "/api/system-audio/stop":
            state["stopped"].append(path)
            route.fulfill(json={"ok": True})
        elif path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[1]
            job = next((j for j in state["jobs"] if j["job_id"] == job_id), None)
            if job_id == state["hold_status"]:
                state["held"].append((route, job))
            elif job:
                route.fulfill(json=job)
            else:
                route.fulfill(status=404, json={})
        else:
            route.fulfill(status=404, json={})

    page.route("**/api/**", route_api)
    state["open"] = lambda: page.goto(base_url)
    return state


def emit(page, event):
    page.evaluate("event => window.streams.at(-1).onmessage({data: JSON.stringify(event)})", event)


def test_pending_batch_rows_and_focus_survive_snapshots(page, frontend_api):
    state = frontend_api
    state["hold_upload"] = True
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    state["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator("#sttFileInput").set_input_files([
        {"name": "first.wav", "mimeType": "audio/wav", "buffer": b"first"},
        {"name": "second.wav", "mimeType": "audio/wav", "buffer": b"second"},
    ])
    rows = page.locator("#sttBatchList .stt-batch-row")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_be_disabled()
    expect(rows.nth(1)).to_be_disabled()
    expect(rows.nth(0)).to_contain_text("first.wav")
    expect(rows.nth(1)).to_contain_text("second.wav")
    for _ in range(100):
        if len(state["held"]) == 2:
            break
        page.wait_for_timeout(20)
    assert len(state["held"]) == 2
    for route, job in state["held"]:
        route.fulfill(json={"job_id": job["job_id"]})
    expect(rows.nth(1)).to_be_enabled()
    rows.nth(1).click()
    expect(rows.nth(1)).to_have_attribute("aria-pressed", "true")
    state["jobs"][1].update(progress_current=37, progress_total=100)
    state["jobs"].reverse()  # Wire order must not change stable batch slot identity.
    expect(rows.nth(1).locator(".stt-batch-detail")).to_have_text("37%")
    expect(rows.nth(1)).to_be_focused()
    expect(rows.nth(1)).to_have_attribute("aria-pressed", "true")
    page.reload()
    expect(rows).to_have_count(2)
    expect(rows.nth(1)).to_have_attribute("aria-pressed", "true")
    page.locator("#jobsMenuBtn").click()
    page.locator("#jobsList summary").click()
    expect(page.locator("#jobsList details")).to_have_attribute("open", "")
    state["jobs"][0]["progress_current"] = 50
    expect(rows.nth(1).locator(".stt-batch-detail")).to_have_text("50%")
    expect(page.locator("#jobsList details")).to_have_attribute("open", "")
    assert not errors


def test_late_restore_cannot_replace_newer_selection(page, frontend_api):
    state = frontend_api
    state["jobs"] = [
        {"job_id": name, "job_type": "stt", "state": "completed", "batch_id": "batch", "batch_index": index,
         "progress_current": 0, "progress_total": 0, "filename": name + ".wav",
         "result": {"filename": name + ".wav", "batch_id": "batch", "segments": [{"start": 0, "end": 1, "text": name}]}}
        for index, name in enumerate(["older", "newer"], 1)
    ]
    page.add_init_script("localStorage.setItem('resonance_stt_active_batch_id','batch')")
    state["open"]()
    rows = page.locator("#sttBatchList .stt-batch-row")
    expect(rows).to_have_count(2)
    state["hold_status"] = "older"
    rows.nth(0).click()
    rows.nth(1).click()
    expect(page.locator("#sttResultText")).to_have_value("[00:00-00:01] newer")
    for route, job in state["held"]:
        route.fulfill(json=job)
    page.wait_for_timeout(100)
    expect(page.locator("#sttResultText")).to_have_value("[00:00-00:01] newer")
    expect(rows.nth(1)).to_have_attribute("aria-pressed", "true")


@pytest.mark.parametrize("source", ["mic_live", "system_audio"])
def test_live_f5_replay_and_preview_ownership(page, frontend_api, source):
    job = {"job_id": "live", "job_type": "stt", "state": "running", "started_at": 100,
           "progress_current": 1, "progress_total": 0, "last_event_seq": 5,
           "result": {"source": source, "filename": "live.wav", "segments": [{"start": 0, "end": 1, "text": "durable"}]}}
    frontend_api["jobs"] = [job]
    page.add_init_script("localStorage.setItem('resonance_stt_active_job_id','live')")
    frontend_api["open"]()
    page.wait_for_function("window.streams.length === 1")
    assert page.evaluate("window.streams[0].url").endswith("after=5")
    expect(page.locator("#sttProgress")).not_to_be_visible()
    if source == "system_audio":
        expect(page.locator("#sttMicStop")).to_be_visible()
    else:
        expect(page.locator("#sttMicStart")).to_be_visible()
    emit(page, {"type": "transcript_preview", "seq": 6, "generation": 2, "text": "provisional"})
    expect(page.locator("#sttResultText")).to_have_value("[00:00-00:01] durable\n\nprovisional")
    event = {"type": "progress", "seq": 7, "current": 2, "total": 0,
             "segment": {"start": 1, "end": 2, "text": "confirmed", "generation": 2}}
    emit(page, event)
    emit(page, event)
    expect(page.locator("#sttMeta")).to_have_text("2 segments")
    expect(page.locator("#sttResultText")).to_have_value("[00:00-00:02] durable confirmed")
    job["result"]["segments"].append(event["segment"])
    job["last_event_seq"] = 7
    page.evaluate("window.streams.at(-1).onerror()")
    page.wait_for_function("window.streams.length === 2")
    assert page.evaluate("window.streams.at(-1).url").endswith("after=7")
    expect(page.locator("#sttMeta")).to_have_text("2 segments")
    page.reload()
    page.wait_for_function("window.streams.length === 1")
    expect(page.locator("#sttMeta")).to_have_text("2 segments")
    if source == "system_audio":
        page.locator("#sttMicStop").click()
        expect(page.locator("#sttMicStart")).to_be_visible()
        assert frontend_api["stopped"] == ["/api/system-audio/stop"]


def test_tts_and_stt_error_cancel_restart(page, frontend_api):
    frontend_api["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator('[data-tab="tts"]').click()
    page.locator("#ttsInput").fill("A short text")
    page.locator("#ttsSubmit").click()
    page.wait_for_function("window.streams.length === 1")
    emit(page, {"type": "start", "total": 2})
    emit(page, {"type": "progress", "current": 1, "total": 2})
    expect(page.locator("#ttsProgressText")).to_have_text("Chunk 1 / 2")
    emit(page, {"type": "complete", "chunks": 2, "duration": 3, "download_url": "/api/audio/test.wav"})
    expect(page.locator("#ttsResult")).to_be_visible()
    expect(page.locator("#ttsDownload")).to_be_visible()
    expect(page.locator("#ttsAudio")).to_have_attribute("src", "/api/audio/test.wav")
    page.locator('[data-tab="stt"]').click()
    audio = {"name": "audio.wav", "mimeType": "audio/wav", "buffer": b"audio"}
    page.locator("#sttFileInput").set_input_files(audio)
    page.wait_for_function("window.streams.length === 2")
    emit(page, {"type": "progress", "current": 1, "total": 0, "segment": {"start": 0, "end": 1, "text": "partial"}})
    emit(page, {"type": "error", "message": "Audio too long"})
    expect(page.locator("#sttResult")).not_to_be_visible()
    expect(page.locator("#sttResultText")).to_have_value("")
    page.locator("#sttFileInput").set_input_files(audio)
    page.wait_for_function("window.streams.length === 3")
    page.locator("#sttCancel").click()
    expect(page.locator("#sttProgress")).not_to_be_visible()
    page.locator("#sttFileInput").set_input_files(audio)
    page.wait_for_function("window.streams.length === 4")
    emit(page, {"type": "progress", "current": 3, "total": 10})
    expect(page.locator("#sttProgressText")).to_have_text("30%")


def mock_microphone(page):
    page.add_init_script("""
        window.stoppedTracks = 0;
        Object.defineProperty(navigator, 'mediaDevices', {value: {
            getUserMedia: async () => ({getTracks: () => [{stop() { window.stoppedTracks++; }}]})
        }});
        window.AudioContext = class {
            sampleRate = 8000;
            destination = {};
            resume() { return Promise.resolve(); }
            close() { return Promise.resolve(); }
            createMediaStreamSource() { return {connect(){},disconnect(){}}; }
            createGain() { return {gain:{value:0},connect(){},disconnect(){}}; }
            createScriptProcessor() {
                window.processor = {connect(){},disconnect(){},onaudioprocess:null};
                return window.processor;
            }
        };
        window.recordSeconds = seconds => {
            for (let i=0; i<seconds; i++) window.processor.onaudioprocess({
                inputBuffer: {getChannelData: () => new Float32Array(8000)}
            });
        };
    """)


def start_mock_live(page, frontend_api):
    mock_microphone(page)
    page.route("**/api/jobs/live/start?*", lambda route: route.fulfill(json={"job_id": "live"}))
    frontend_api["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator("#sttMicModeLive").click()
    page.locator("#sttMicStart").click()
    page.wait_for_function("Boolean(window.processor && window.processor.onaudioprocess)")


def test_live_transport_retries_same_sequence_before_next_chunk(page, frontend_api):
    sent = []

    def chunk(route):
        sequence = int(parse_qs(urlparse(route.request.url).query)["sequence"][0])
        sent.append(sequence)
        if len(sent) == 1:
            route.fulfill(status=503, body="temporary failure")
        else:
            route.fulfill(json={"ack_sequence": sequence})

    page.route("**/api/jobs/live/live/chunk?*", chunk)
    page.route("**/api/jobs/live/live/stop", lambda route: route.fulfill(json={"ok": True}))
    start_mock_live(page, frontend_api)
    page.evaluate("recordSeconds(2)")
    page.locator("#sttMicStop").click()
    page.wait_for_function("window.stoppedTracks === 1")
    for _ in range(50):
        if len(sent) >= 3:
            break
        page.wait_for_timeout(50)
    assert sent == [1, 1, 2]
    expect(page.locator("#sttError")).not_to_be_visible()


def test_live_transport_backpressure_stops_capture_without_successful_stop(page, frontend_api):
    held = []
    page.route("**/api/jobs/live/live/chunk?*", lambda route: held.append(route))
    start_mock_live(page, frontend_api)
    page.evaluate("recordSeconds(11)")
    expect(page.locator("#sttError")).to_be_visible()
    expect(page.locator("#sttMicStart")).to_be_visible()
    assert page.evaluate("window.stoppedTracks") == 1
    assert not frontend_api["stopped"]
    for route in held:
        route.abort()


def test_dictation_preview_can_be_discarded_and_sent_as_batch_audio(page, frontend_api):
    mock_microphone(page)
    frontend_api["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator("#sttMicStart").click()
    page.wait_for_function("Boolean(window.processor && window.processor.onaudioprocess)")
    page.evaluate("recordSeconds(1)")
    page.locator("#sttMicStop").click()
    expect(page.locator("#sttMicPreview")).to_be_visible()
    page.locator("#sttMicDiscard").click()
    expect(page.locator("#sttMicPreview")).not_to_be_visible()
    page.locator("#sttMicStart").click()
    page.wait_for_function("Boolean(window.processor && window.processor.onaudioprocess)")
    page.evaluate("recordSeconds(1)")
    page.locator("#sttMicStop").click()
    page.locator("#sttMicSend").click()
    page.wait_for_function("window.streams.length === 1")
    expect(page.locator("#sttMicPreview")).not_to_be_visible()
    assert len(frontend_api["jobs"]) == 1


def test_system_audio_include_microphone_and_stop(page, frontend_api):
    starts = []

    def start(route):
        starts.append(parse_qs(urlparse(route.request.url).query))
        route.fulfill(json={"job_id": "system", "started_at": 100})

    page.route("**/api/system-audio/start?*", start)
    frontend_api["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator("#tabSys").click()
    page.locator("#sttSysIncludeMic").check()
    page.locator("#sttMicStart").click()
    expect(page.locator("#sttMicStop")).to_be_visible()
    expect(page.locator("#sttSysIncludeMic")).to_be_disabled()
    assert starts[0]["include_microphone"] == ["true"]
    assert starts[0]["language"] == ["en"]
    assert page.evaluate("localStorage.getItem('resonance_sttSysIncludeMic')") == "true"
    page.locator("#sttMicStop").click()
    expect(page.locator("#sttMicStart")).to_be_visible()
    assert frontend_api["stopped"] == ["/api/system-audio/stop"]


def test_star_field_preserves_geometry_and_interactions(page, frontend_api):
    page.set_viewport_size({"width": 1440, "height": 900})
    frontend_api["open"]()
    field = page.locator(".star-field")
    expect(field).to_be_visible()
    expect(field).to_have_attribute("aria-hidden", "true")
    page.wait_for_timeout(250)  # Let the existing panel entrance finish.
    assert page.evaluate("""() => {
        const main = document.querySelector('main').getBoundingClientRect();
        return [...document.querySelectorAll('.star-field svg')].every(star => {
            const rect = star.getBoundingClientRect();
            return (rect.right <= main.left || rect.left >= main.right)
                && rect.top >= main.top && rect.bottom <= main.bottom;
        });
    }""")

    geometry = """() => {
        const r = document.querySelector('main').getBoundingClientRect();
        return [r.x, r.y, r.width, r.height,
                document.documentElement.scrollWidth, document.documentElement.scrollHeight];
    }"""
    for width in [1440, 390]:
        page.set_viewport_size({"width": width, "height": 900})
        before = page.evaluate(geometry)
        field.evaluate("e => e.style.display = 'none'")
        assert page.evaluate(geometry) == before
        field.evaluate("e => e.style.removeProperty('display')")

    page.set_viewport_size({"width": 1440, "height": 900})
    assert page.evaluate("document.documentElement.scrollWidth === document.documentElement.clientWidth")
    page.locator('[data-tab="tts"]').click()
    page.locator("#ttsInput").fill("The decorative layer leaves forms usable.")
    expect(page.locator("#ttsInput")).to_be_focused()
    page.locator("#jobsMenuBtn").click()
    expect(page.locator("#jobsMenuBtn")).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Escape")
    expect(page.locator("#jobsMenuBtn")).to_have_attribute("aria-expanded", "false")


def test_star_field_motion_preferences_and_resize(page, frontend_api):
    page.set_viewport_size({"width": 1440, "height": 900})
    page.emulate_media(reduced_motion="no-preference")
    frontend_api["open"]()
    field = page.locator(".star-field")
    expect(field).to_be_visible()
    snapshot = """() => [...document.querySelectorAll('.star-field *')].map(e => {
        const style = getComputedStyle(e);
        return [style.opacity, style.transform];
    })"""
    initial = page.evaluate(snapshot)
    changed = f"initial => JSON.stringify(({snapshot})()) !== JSON.stringify(initial)"
    page.wait_for_function(changed, arg=initial)

    for motion, width in [("reduce", 1440), ("no-preference", 390)]:
        page.emulate_media(reduced_motion=motion)
        page.set_viewport_size({"width": width, "height": 900})
        if width == 390:
            expect(field).not_to_be_visible()
        page.wait_for_timeout(100)
        stopped = page.evaluate(snapshot)
        page.wait_for_timeout(250)
        assert page.evaluate(snapshot) == stopped


@pytest.mark.parametrize("event", [
    {"type": "error", "message": "Microphone unavailable"},
    {"type": "complete"},
    {"type": "cancelled"},
])
def test_system_capture_terminal_event_unlocks_controls(page, frontend_api, event):
    page.route("**/api/system-audio/start?*", lambda route: route.fulfill(json={"job_id": "system"}))
    frontend_api["open"]()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator("#tabSys").click()
    page.locator("#sttMicStart").click()
    expect(page.locator("#sttMicStop")).to_be_visible()
    emit(page, event)
    expect(page.locator("#sttMicStart")).to_be_enabled()
    expect(page.locator("#sttMicStop")).not_to_be_visible()
    expect(page.locator("#sttSysIncludeMic")).to_be_enabled()
    if event["type"] == "error":
        expect(page.locator("#sttError")).to_contain_text(event["message"])

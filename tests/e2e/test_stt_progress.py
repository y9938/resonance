"""Browser contracts for batch STT progress and status restore."""

import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Page, expect

PUBLIC_ROOT = Path(__file__).resolve().parents[2] / "dist" / "web"


def open_stt_page(page: Page, base_url: str, status: dict | None = None, *, local_files: bool = False) -> None:
    page.add_init_script("""
        localStorage.setItem('resonance_locale', 'en');
        window.__sttEventSources = [];
        window.EventSource = class {
            constructor(url) { this.url = url; window.__sttEventSources.push(this); }
            close() {}
        };
    """)

    def handle_request(route):
        path = urlparse(route.request.url).path
        if path == "/api/config":
            payload = {"upload_limit_mb": 50, "local_files_enabled": local_files, "tts": {"languages": []}}
        elif path == "/api/models":
            payload = {
                "stt": {
                    "models": [{
                        "id": "whisper", "name": "Whisper", "languages": ["en"], "loaded": False,
                    }],
                },
            }
        elif path == "/api/jobs/job-1" and status is not None:
            payload = status
        elif path == "/api/jobs":
            payload = {"jobs": [status] if status else [], "has_more": False, "next_offset": 1 if status else 0}
        elif path == "/api/jobs/stt":
            payload = {"job_id": "job-1"}
        elif path.startswith("/api/"):
            route.fulfill(status=404, body="")
            return
        else:
            asset = PUBLIC_ROOT / (path.lstrip("/") or "index.html")
            if asset.is_file():
                route.fulfill(path=str(asset))
            else:
                route.fulfill(status=404, body="")
            return
        route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

    page.route("**/*", handle_request)
    page.goto(base_url)
    page.wait_for_selector("#sttLanguage:not([disabled])")


def change_locale(page: Page, code: str) -> None:
    page.locator("#localePicker .ts-control").click()
    page.locator(f'#localePicker .ts-dropdown [data-value="{code}"]').click()


def test_windows_clipboard_path_and_submission_error(page: Page, base_url: str) -> None:
    open_stt_page(page, base_url, local_files=True)
    paths = []
    error = "The selected media file cannot be opened"

    def reject_submission(route):
        paths.append(route.request.post_data_json["path"])
        route.fulfill(status=400, content_type="application/json", body=json.dumps({"detail": error}))

    page.route("**/api/jobs/stt/local?*", reject_submission)
    path = r"Z:\Легендарные и пугающие часы за 15 баксов： от Осамы до Обамы.mp4"
    page.locator("#sttLocalFiles summary").click()
    page.locator("#sttLocalPaths").fill(f'"{path}"')
    page.locator("#sttLocalStart").click()
    expect(page.locator("#sttError")).to_be_visible()
    expect(page.locator("#sttError")).to_have_text(error)
    expect(page.locator("#sttLocalStart")).to_be_enabled()
    assert paths == [path]


def test_stt_sse_progress_uses_known_or_unknown_total(page: Page, base_url: str) -> None:
    open_stt_page(page, base_url)
    page.locator("#sttFileInput").set_input_files({"name": "audio.wav", "mimeType": "audio/wav", "buffer": b"audio"})
    page.wait_for_function("window.__sttEventSources.length > 0")

    def send(event):
        return page.evaluate("""async event => {
            const source = window.__sttEventSources.at(-1);
            source.onmessage({ data: JSON.stringify(event) });
            await new Promise(requestAnimationFrame);
            return {
                text: document.getElementById('sttProgressText').textContent,
                fill: document.getElementById('sttProgressFill').style.width,
            };
        }""", event)

    assert send({"type": "start", "total": 0, "stage": "diarization"}) == {
        "text": "Diarizing speakers…", "fill": "0%",
    }
    assert send({"type": "progress", "current": 90, "total": 0}) == {
        "text": "Processing…", "fill": "0%",
    }
    assert send({"type": "progress", "current": 1, "total": 4}) == {
        "text": "25%", "fill": "25%",
    }
    assert send({"type": "progress", "current": 4, "total": 4}) == {
        "text": "100%", "fill": "100%",
    }
    change_locale(page, "ru")
    change_locale(page, "en")
    expect(page.locator("#sttProgressText")).to_have_text("100%")
    assert send({"type": "complete", "duration": 4}) == {
        "text": "Complete", "fill": "100%",
    }


def test_stt_restore_unknown_total_and_empty_completed_transcript(page: Page, base_url: str) -> None:
    status = {
        "job_id": "job-1",
        "job_type": "stt",
        "state": "running",
        "progress_current": 30,
        "progress_total": 0,
        "last_event_seq": 0,
        "result": {
            "filename": "silent.wav",
            "segments": [{"start": 0, "end": 1, "text": "Earlier speech"}],
        },
    }
    page.add_init_script("localStorage.setItem('resonance_stt_active_job_id', 'job-1')")
    open_stt_page(page, base_url, status)

    expect(page.locator("#sttProgressText")).to_have_text("Processing…")
    expect(page.locator("#sttResult")).to_be_visible()
    assert page.locator("#sttProgressFill").evaluate("el => el.style.width") == "0%"
    change_locale(page, "ru")
    expect(page.locator("#sttProgressText")).to_have_text("Обработка…")

    status["state"] = "completed"
    status["progress_current"] = 0
    status["result"]["segments"] = []
    page.reload()

    expect(page.locator("#sttProgressText")).to_have_text("Complete")
    assert page.locator("#sttProgressFill").evaluate("el => el.style.width") == "100%"
    assert page.locator("#sttResultText").input_value() == ""
    assert page.locator("#sttResultText").get_attribute("data-filename") == "silent.wav"
    expect(page.locator("#sttResult")).to_be_visible()

    status["progress_current"] = 50
    status["progress_total"] = 60
    page.locator("#jobsMenuBtn").click()
    page.locator("#jobsList button").first.click()
    change_locale(page, "ru")
    expect(page.locator("#sttProgressText")).to_have_text("Готово")


def test_stt_batch_rows_show_queued_unknown_and_known_progress(page: Page, base_url: str) -> None:
    jobs = [
        {"job_id": "queued", "job_type": "stt", "state": "queued", "filename": "a.wav", "progress_current": 0, "progress_total": 0},
        {"job_id": "unknown", "job_type": "stt", "state": "running", "filename": "b.wav", "progress_current": 9, "progress_total": 0},
        {"job_id": "known", "job_type": "stt", "state": "running", "filename": "c.wav", "progress_current": 37, "progress_total": 100},
    ]
    page.add_init_script("localStorage.setItem('resonance_stt_active_batch_id', 'batch-1')")
    for index, job in enumerate(jobs):
        job.update(batch_id="batch-1", batch_index=index + 1)
    open_stt_page(page, base_url)
    page.route("**/api/jobs?*", lambda route: route.fulfill(json={"jobs": jobs, "has_more": False, "next_offset": 3}))
    page.reload()
    expect(page.locator("#sttBatchList .stt-batch-detail")).to_have_count(3)

    assert page.locator("#sttBatchList .stt-batch-detail").all_text_contents() == [
        "Queued",
        "Processing…",
        "37%",
    ]


def test_failed_local_job_error_survives_reload_and_history_open(page: Page, base_url: str) -> None:
    status = {
        "job_id": "job-1", "job_type": "stt", "state": "failed",
        "error": "The selected file is no longer accessible",
        "result": {"filename": "lecture.mp4"},
    }
    page.add_init_script("localStorage.setItem('resonance_stt_active_job_id', 'job-1')")
    open_stt_page(page, base_url, status)
    expect(page.locator("#sttError")).to_be_visible()
    expect(page.locator("#sttError")).to_have_text(status["error"])
    page.locator("#jobsMenuBtn").click()
    page.locator("#jobsList button").first.click()
    expect(page.locator("#sttError")).to_be_visible()
    expect(page.locator("#sttError")).to_have_text(status["error"])

"""Browser contracts for batch STT progress and status restore."""

import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Page, expect

PUBLIC_ROOT = Path(__file__).resolve().parents[2] / "public"


def open_stt_page(page: Page, base_url: str, status: dict | None = None) -> None:
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
            payload = {"upload_limit_mb": 50, "tts": {"languages": []}}
        elif path == "/api/models":
            payload = {
                "stt": {
                    "default_language": "en",
                    "models": [{
                        "id": "whisper", "name": "Whisper", "languages": ["en"], "loaded": False,
                    }],
                },
            }
        elif path == "/api/jobs/job-1" and status is not None:
            payload = status
        elif path == "/api/jobs":
            payload = {"jobs": [], "has_more": False, "next_offset": 0}
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


def test_stt_sse_progress_uses_known_or_unknown_total(page: Page, base_url: str) -> None:
    open_stt_page(page, base_url)
    page.evaluate("""() => {
        document.getElementById('sttProgress').classList.add('active');
        subscribeSttJobEvents('job-1', [], () => {});
    }""")

    def send(event):
        return page.evaluate("""event => {
            const source = window.__sttEventSources.at(-1);
            source.onmessage({ data: JSON.stringify(event) });
            return {
                text: document.getElementById('sttProgressText').textContent,
                fill: document.getElementById('sttProgressFill').style.width,
            };
        }""", event)

    assert send({"type": "start", "total": 0, "stage": "diarization"}) == {
        "text": page.evaluate("t('progressDiarizing')"), "fill": "0%",
    }
    assert send({"type": "progress", "current": 90, "total": 0}) == {
        "text": page.evaluate("t('progressProcessing')"), "fill": "0%",
    }
    assert send({"type": "progress", "current": 1, "total": 4}) == {
        "text": "25%", "fill": "25%",
    }
    assert send({"type": "progress", "current": 4, "total": 4}) == {
        "text": "100%", "fill": "100%",
    }
    page.evaluate("refreshActiveProgressLabels()")
    expect(page.locator("#sttProgressText")).to_have_text("100%")
    assert send({"type": "complete", "duration": 4}) == {
        "text": page.evaluate("t('progressComplete')"), "fill": "100%",
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
    assert page.evaluate("setLocale('ru')") is True
    expect(page.locator("#sttProgressText")).to_have_text(page.evaluate("t('progressProcessing')"))

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
    page.evaluate("restoreJob('job-1', 'stt')")
    assert page.evaluate("setLocale('ru')") is True
    expect(page.locator("#sttProgressText")).to_have_text(page.evaluate("t('progressComplete')"))


def test_stt_batch_rows_show_queued_unknown_and_known_progress(page: Page, base_url: str) -> None:
    open_stt_page(page, base_url)
    jobs = [
        {"job_id": "queued", "job_type": "stt", "state": "queued", "filename": "a.wav", "progress_current": 0, "progress_total": 0},
        {"job_id": "unknown", "job_type": "stt", "state": "running", "filename": "b.wav", "progress_current": 9, "progress_total": 0},
        {"job_id": "known", "job_type": "stt", "state": "running", "filename": "c.wav", "progress_current": 37, "progress_total": 100},
    ]
    page.evaluate("jobs => setSttBatchState('batch-1', jobs)", jobs)

    assert page.locator("#sttBatchList .stt-batch-detail").all_text_contents() == [
        page.evaluate("t('jobsStateQueued')"),
        page.evaluate("t('progressProcessing')"),
        "37%",
    ]

"""STT batch upload frontend smoke test.

The API is mocked here: this test verifies the browser-side batch UX without
requiring an actual STT model run.
"""

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "ru_audio.wav"
STT_CATALOG = {
    "stt": {
        "models": [
            {"id": "gigaam", "name": "GigaAM-v3", "languages": ["ru"], "loaded": False},
            {"id": "whisper", "name": "Whisper Turbo", "languages": ["de", "en", "ru"], "loaded": False},
            {"id": "granite", "name": "IBM Granite", "languages": ["en"], "loaded": False},
        ],
    }
}


@pytest.mark.parametrize("local_paths", [False, True])
def test_stt_batch_upload_creates_one_visible_queue(page: Page, base_url: str, local_paths):
    started: list[dict[str, list[str]]] = []
    jobs: list[dict[str, object]] = []
    cancelled: list[str] = []
    paths: list[str] = []

    def fulfill_json(route, payload):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    def handle_api(route):
        parsed = urlparse(route.request.url)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/config":
            fulfill_json(route, {"upload_limit_mb": 50, "local_files_enabled": local_paths, "tts": {"languages": []}})
            return

        if parsed.path == "/api/models":
            fulfill_json(route, STT_CATALOG)
            return

        if parsed.path in {"/api/jobs/stt", "/api/jobs/stt/local"}:
            if parsed.path.endswith("/local"):
                paths.append(route.request.post_data_json["path"])
            job_id = f"job-{len(started) + 1}"
            started.append(qs)
            job = {
                "job_id": job_id,
                "job_type": "stt",
                "state": "queued",
                "progress_current": 0,
                "progress_total": 0,
                "filename": f"clip-{len(started)}.wav",
                "created_at": 1000 + len(started),
                "updated_at": 1000 + len(started),
            }
            if "batch_id" in qs:
                job.update(
                    {
                        "batch_id": qs["batch_id"][0],
                        "batch_index": int(qs["batch_index"][0]),
                        "batch_total": int(qs["batch_total"][0]),
                    }
                )
            jobs.append(job)
            fulfill_json(route, {"job_id": job_id})
            return

        if parsed.path == "/api/jobs":
            fulfill_json(route, {"jobs": jobs, "has_more": False, "next_offset": len(jobs)})
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/events"):
            route.fulfill(status=200, content_type="text/event-stream", body="")
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path.split("/")[-2]
            cancelled.append(job_id)
            job = next((item for item in jobs if item["job_id"] == job_id), None)
            if job is not None:
                job["state"] = "cancelled"
            fulfill_json(route, {"ok": True})
            return

        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = next((item for item in jobs if item["job_id"] == job_id), None)
            if job is None:
                route.fulfill(status=404, content_type="application/json", body="{}")
                return
            fulfill_json(
                route,
                {
                    **job,
                    "result": {
                        "filename": job["filename"],
                        "batch_id": job.get("batch_id"),
                        "segments": [{"start": 0, "end": 1, "text": str(job["filename"])}],
                    },
                },
            )
            return

        route.continue_()

    page.route("**/api/**", handle_api)
    page.goto(f"{base_url}")
    page.wait_for_selector("#sttLanguage:not([disabled])")

    audio_bytes = AUDIO_FILE.read_bytes()
    page.evaluate(
        f"""
        const bytes = new Uint8Array({list(audio_bytes)});
        const blob = new Blob([bytes], {{ type: 'audio/wav' }});
        window.batchFiles = [
            new File([blob], 'clip-1.wav', {{ type: 'audio/wav' }}),
            new File([blob], 'clip-2.wav', {{ type: 'audio/wav' }}),
        ];
    """
    )

    if local_paths:
        page.locator("#sttLocalFiles summary").click()
        page.locator("#sttLocalPaths").fill("/media/clip-1.wav\n/media/clip-2.wav")
        page.locator("#sttLocalStart").click()
    else:
        page.evaluate(
            """
            const input = document.getElementById('sttFileInput');
            const dataTransfer = new DataTransfer();
            for (const file of window.batchFiles) dataTransfer.items.add(file);
            input.files = dataTransfer.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        """
        )

    page.wait_for_selector("#sttBatchPanel.active")
    expect(page.locator("#sttBatchList .stt-batch-row")).to_have_count(2)

    for _ in range(20):
        if len(started) == 2:
            break
        page.wait_for_timeout(100)

    assert len(started) == 2
    assert started[0]["batch_id"] == started[1]["batch_id"]
    assert started[0]["batch_total"] == ["2"]
    assert started[1]["batch_total"] == ["2"]

    expect(page.locator("#sttResultText")).to_have_value(re.compile("clip-1.wav"))
    page.evaluate(
        """
        document.body.style.minHeight = '2200px';
        window.scrollTo(0, 500);
        document.querySelector('#sttBatchList .stt-batch-row').click();
    """
    )
    page.wait_for_timeout(200)
    assert abs(page.evaluate("window.scrollY") - 500) < 5
    assert page.eval_on_selector("#sttResultText", "el => getComputedStyle(el).minHeight") == "300px"

    page.evaluate("document.getElementById('jobsMenuBtn').click()")
    page.wait_for_selector("#jobsDrawer.open")
    expect(page.locator("#jobsList details")).to_have_count(1)
    expect(page.locator("#jobsList .jobs-child-row")).to_have_count(2)
    expect(page.locator("#jobsList details summary")).to_contain_text("STT batch · 0 / 2")

    page.click("#jobsList .jobs-batch-open")
    page.wait_for_selector("#jobsDrawer", state="attached")
    page.wait_for_function("() => !document.getElementById('jobsDrawer').classList.contains('open')")
    page.wait_for_selector("#sttBatchPanel.active")
    assert abs(page.evaluate("window.scrollY") - 500) < 5

    if local_paths:
        assert paths == ["/media/clip-1.wav", "/media/clip-2.wav"]
        page.evaluate("localStorage.setItem('resonance_locale', 'ru')")
        page.reload()
        expect(page.locator("#sttBatchList .stt-batch-row")).to_have_count(2)
        expect(page.locator("#sttLocalFiles summary")).to_contain_text("Локальные файлы")
        expect(page.locator("#sttLocalPaths")).to_have_value("")
        assert "/media/" not in page.evaluate("JSON.stringify(localStorage)")
        page.locator("#sttBatchList .stt-batch-row").first.click()
        # Files selected through the browser remain an upload fallback after restore.
        page.evaluate("window.batchFiles = [new File(['audio'], 'clip-1.wav', {type: 'audio/wav'})]")

    page.click("#sttBatchCancelCurrent")
    for _ in range(20):
        if cancelled == ["job-1"]:
            break
        page.wait_for_timeout(100)
    assert cancelled == ["job-1"]

    page.evaluate(
        """
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(window.batchFiles[0]);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """
    )
    page.wait_for_function("() => !document.getElementById('sttBatchPanel').classList.contains('active')")

    for _ in range(20):
        if len(started) == 3:
            break
        page.wait_for_timeout(100)

    assert len(started) == 3
    assert "batch_id" not in started[2]


def test_stt_language_dropdown_routing(page: Page, base_url: str):
    started: list[dict[str, list[str]]] = []
    jobs: list[dict[str, object]] = []

    def fulfill_json(route, payload):
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    def handle_api(route):
        parsed = urlparse(route.request.url)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/config":
            fulfill_json(route, {"upload_limit_mb": 50, "tts": {"languages": []}})
            return

        if parsed.path == "/api/models":
            fulfill_json(route, STT_CATALOG)
            return

        if parsed.path == "/api/jobs/stt":
            job_id = f"job-{len(started) + 1}"
            started.append(qs)
            job = {
                "job_id": job_id,
                "job_type": "stt",
                "state": "queued",
                "progress_current": 0,
                "progress_total": 0,
                "filename": f"clip-{len(started)}.wav",
                "created_at": 1000 + len(started),
                "updated_at": 1000 + len(started),
            }
            jobs.append(job)
            fulfill_json(route, {"job_id": job_id})
            return

        if parsed.path == "/api/jobs":
            fulfill_json(route, {"jobs": jobs, "has_more": False, "next_offset": len(jobs)})
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/events"):
            route.fulfill(status=200, content_type="text/event-stream", body="")
            return

        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = next((item for item in jobs if item["job_id"] == job_id), None)
            if job is None:
                route.fulfill(status=404, content_type="application/json", body="{}")
                return
            fulfill_json(
                route,
                {
                    **job,
                    "result": {
                        "filename": job["filename"],
                        "segments": [{"start": 0, "end": 1, "text": str(job["filename"])}],
                    },
                },
            )
            return

        route.continue_()

    page.route("**/api/**", handle_api)
    page.add_init_script("localStorage.setItem('resonance_locale', 'ru'); localStorage.setItem('resonance_sttModel', 'whisper')")
    page.goto(f"{base_url}")
    page.wait_for_selector("#sttLanguage:not([disabled])")

    # Assert dropdown exists and defaults to 'ru'
    dropdown = page.locator("#sttLanguage")
    expect(page.locator(".stt-language-picker .ts-control")).to_be_visible()
    assert dropdown.evaluate("el => el.value") == "ru"
    assert page.locator("#sttModel").input_value() == ""

    # Create dummy files for upload
    audio_bytes = AUDIO_FILE.read_bytes()
    page.evaluate(
        f"""
        const bytes = new Uint8Array({list(audio_bytes)});
        const blob = new Blob([bytes], {{ type: 'audio/wav' }});
        window.testFiles = [
            new File([blob], 'ru_test.wav', {{ type: 'audio/wav' }}),
            new File([blob], 'en_test.wav', {{ type: 'audio/wav' }}),
        ];
    """
    )

    # 1. Upload in 'ru' (default)
    page.evaluate(
        """
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(window.testFiles[0]);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """
    )

    for _ in range(20):
        if len(started) == 1:
            break
        page.wait_for_timeout(100)

    assert len(started) == 1
    assert started[0]["language"] == ["ru"]

    # 2. Select 'en' from the dropdown
    page.locator(".stt-language-picker .ts-control").click()
    page.locator(".stt-language-picker .ts-control input").fill("English")
    page.locator(".stt-language-picker .ts-dropdown .option").filter(has_text="English").first.click()
    assert dropdown.evaluate("el => el.value") == "en"

    # Upload in 'en'
    page.evaluate(
        """
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(window.testFiles[1]);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """
    )

    for _ in range(20):
        if len(started) == 2:
            break
        page.wait_for_timeout(100)

    assert len(started) == 2
    assert started[1]["language"] == ["en"]
    assert "model" not in started[1]

    page.locator(".stt-language-picker .ts-control").click()
    page.locator(".stt-language-picker .ts-control input").fill("Deutsch")
    page.locator(".stt-language-picker .ts-dropdown .option").filter(has_text="Deutsch").first.click()
    assert page.locator("#sttModel").input_value() == ""
    assert not page.locator("#sttModelContainer").is_visible()

    page.locator(".stt-language-picker .ts-control").click()
    page.locator(".stt-language-picker .ts-control input").fill("English")
    page.locator(".stt-language-picker .ts-dropdown .option").filter(has_text="English").first.click()
    page.locator("#sttModel").select_option("granite")
    page.locator("#sttAutoDetect").check()
    assert page.locator("#sttAutoDetect").evaluate("el => !!el.closest('.input-footer')")
    expect(page.locator("#sttAutoModelEffective")).to_contain_text("Whisper Turbo")
    assert not page.locator("#sttModelContainer").is_visible()
    assert dropdown.evaluate("el => el.value") == "en"

    page.locator("#sttMicModeLive").click()
    expect(page.locator("#sttAutoDetect")).to_be_visible()
    expect(page.locator("#sttModelContainer")).to_be_visible()
    assert page.locator("#sttModel").input_value() == "granite"

    page.evaluate(
        """
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(window.testFiles[0]);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """
    )
    for _ in range(20):
        if len(started) == 3:
            break
        page.wait_for_timeout(100)
    assert started[2]["detect_language"] == ["true"]
    assert "language" not in started[2]
    assert "model" not in started[2]
    page.locator("#sttAutoDetect").uncheck()
    assert not page.locator("#sttAutoModelEffective").is_visible()
    assert dropdown.evaluate("el => el.value") == "en"
    assert page.locator("#sttModel").input_value() == "granite"

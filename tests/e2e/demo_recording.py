"""Demo recording for README GIF.

Records: STT upload → TTS upload → both complete.
"""

from pathlib import Path

from playwright.sync_api import Page, expect

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "en_audio.wav"
TEXT_FILE = Path(__file__).parent.parent / "fixtures" / "ru_text.txt"


def test_demo_flow(page: Page, base_url: str):
    """STT and TTS processing."""
    page.goto(f"{base_url}")
    
    page.evaluate("document.body.style.zoom = '1.2'")
    
    page.wait_for_selector("#sttDropzone")
    page.wait_for_timeout(800)

    page.click("#sttLocalFiles summary")
    page.wait_for_timeout(1800)
    page.click("#sttLocalFiles summary")
    page.wait_for_timeout(600)

    # Demonstrate System Audio tab
    page.click("#tabSys")
    page.wait_for_timeout(1200)
    page.click("#tabMic")
    page.wait_for_timeout(800)

    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.click(".stt-language-picker .ts-control")
    page.locator(".stt-language-picker .ts-control input").fill("English")
    page.locator('.stt-language-picker .ts-dropdown .option[data-value="en"]').click()
    page.wait_for_timeout(800)

    audio_bytes = AUDIO_FILE.read_bytes()
    text_content = TEXT_FILE.read_text()

    page.evaluate(
        f"""
        const bytes = new Uint8Array({list(audio_bytes)});
        const blob = new Blob([bytes], {{ type: 'audio/wav' }});
        const file = new File([blob], 'audio.wav', {{ type: 'audio/wav' }});
        window.testFile = file;
    """
    )

    # Upload STT
    page.evaluate("""
        const file = window.testFile;
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """)

    expect(page.locator("#sttProgressText")).to_contain_text("Uploading")
    page.wait_for_timeout(600)

    # Scroll early so both the progress bar at the top and the live text below are in viewport
    page.wait_for_selector("#sttResult.active", timeout=20000)
    page.evaluate(
        "document.getElementById('sttProgress').scrollIntoView({block: 'start', behavior: 'smooth'})"
    )
    page.wait_for_timeout(800)

    # Observe live streaming segments appearing in the textarea
    page.wait_for_function(
        "() => document.getElementById('sttResultText').value.length > 0",
        timeout=30000,
    )
    page.wait_for_timeout(700)

    # Wait for STT complete
    page.wait_for_function(
        """() => {
            const text = document.getElementById('sttProgressText').textContent;
            return text.includes('Complete') || text.includes('100%');
        }""",
        timeout=120000,
    )
    page.wait_for_timeout(600)

    # Toggle transcription view
    page.click("#sttViewContinuous")
    page.wait_for_timeout(900)

    # Switch interface language
    old_locale = page.eval_on_selector("#localeInput", "el => el.value")
    next_locale = "ru" if old_locale != "ru" else "en"
    page.click("#localePicker .ts-control")
    page.wait_for_selector("#localePicker .ts-wrapper.dropdown-active")
    page.wait_for_timeout(700)
    page.wait_for_selector("#localePicker .ts-dropdown .option")
    page.wait_for_timeout(600)
    page.locator(f'#localePicker .ts-dropdown .option[data-value="{next_locale}"]').click()
    page.wait_for_function(
        "(oldVal) => document.getElementById('localeInput').value !== oldVal",
        arg=old_locale,
        timeout=5000,
    )
    page.wait_for_timeout(1200)

    # Switch to TTS
    page.click('.tab[data-tab="tts"]')
    page.wait_for_selector("#ttsInput", state="visible")
    page.wait_for_timeout(500)

    # Upload TTS
    page.fill("#ttsInput", text_content)
    page.wait_for_timeout(400)
    page.click("#ttsSubmit")
    page.wait_for_timeout(500)

    # Wait for TTS complete
    page.wait_for_function(
        """() => {
            const result = document.getElementById('ttsResult');
            const audio = document.getElementById('ttsAudio');
            if (!result || !audio) return false;
            const src = audio.getAttribute('src') || '';
            return result.classList.contains('active') && src.length > 0;
        }""",
        timeout=60000,
    )
    page.wait_for_timeout(1200)

    # Demonstrate F5 refresh and restore previous job from the job list
    page.evaluate("showToast('F5')")
    page.wait_for_timeout(1200)
    page.reload(wait_until="domcontentloaded")
    page.evaluate("document.body.style.zoom = '1.2'")
    page.wait_for_selector("#jobsMenuBtn")
    page.wait_for_timeout(1500)

    # Ensure TTS result is gone after reload
    page.wait_for_function(
        """() => {
            const el = document.getElementById('ttsResult');
            return el && !el.classList.contains('active');
        }""",
        timeout=10000,
    )
    page.wait_for_timeout(700)

    page.click("#jobsMenuBtn")
    page.wait_for_selector("#jobsDrawer.open")
    page.wait_for_selector("#jobsList button")
    page.wait_for_timeout(1400)

    # Restore latest job
    page.locator("#jobsList button").first.click()
    page.wait_for_selector("#ttsResult.active")
    page.wait_for_timeout(1500)

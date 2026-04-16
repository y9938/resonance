"""Demo recording for README GIF.

Records: STT upload → TTS upload → both complete.
"""

from pathlib import Path

from playwright.sync_api import Page, expect

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "ru_audio.wav"
TEXT_FILE = Path(__file__).parent.parent / "fixtures" / "ru_text.txt"


def test_demo_flow(page: Page, base_url: str):
    """STT and TTS processing."""
    page.goto(f"{base_url}")
    
    page.evaluate("document.body.style.zoom = '1.2'")
    
    page.wait_for_selector("#sttDropzone")
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

    # Wait for STT complete
    page.wait_for_function(
        """() => {
            const text = document.getElementById('sttProgressText').textContent;
            return text.includes('Complete');
        }""",
        timeout=120000,
    )
    page.wait_for_timeout(1600)

    # Scroll to show transcribed text
    page.evaluate(
        "document.getElementById('sttResult').scrollIntoView({block: 'center', behavior: 'smooth'})"
    )
    page.wait_for_timeout(2200)

    # Toggle transcription view
    page.click("#sttViewContinuous")
    page.wait_for_timeout(900)

    # Switch interface language to the second option
    old_locale = page.eval_on_selector("#localeInput", "el => el.value")
    page.click("#localeInput")
    page.wait_for_selector("#localePicker.open")
    page.wait_for_timeout(700)
    page.wait_for_selector("#localeListbox .locale-option")
    page.wait_for_timeout(600)
    page.locator("#localeListbox .locale-option").nth(1).click()
    page.wait_for_function(
        "(oldVal) => document.getElementById('localeInput').value !== oldVal",
        arg=old_locale,
        timeout=5000,
    )
    page.wait_for_timeout(1200)

    # Switch to TTS
    page.click('.tab[data-tab="tts"]')
    page.wait_for_selector("#ttsInput", state="visible")
    page.wait_for_timeout(900)

    # Upload TTS
    page.fill("#ttsInput", text_content)
    page.wait_for_timeout(600)
    page.click("#ttsSubmit")
    page.wait_for_timeout(900)

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
    page.wait_for_timeout(2000)

    # Demonstrate F5 refresh and restore previous job from the job list
    page.wait_for_timeout(900)
    page.evaluate("showToast('F5')")
    page.wait_for_timeout(1800)
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
    page.wait_for_timeout(3000)

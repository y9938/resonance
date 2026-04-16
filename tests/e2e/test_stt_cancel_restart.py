"""STT cancel + restart regression test.

Verifies: Cancel cleans up internal state. Restart after cancel
must process normally (not stuck on "Uploading...").
"""

from pathlib import Path

from playwright.sync_api import Page, expect

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "ru_audio.wav"
TEXT_FILE = Path(__file__).parent.parent / "fixtures" / "ru_text.txt"


def test_cancel_restart_shows_progress(page: Page, base_url: str):
    """STT cancel + TTS parallel → STT restart must complete."""
    page.goto(f"{base_url}")
    page.wait_for_selector("#sttDropzone")

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

    # Start STT
    page.evaluate("""
        const file = window.testFile;
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """)

    stt_progress = page.locator("#sttProgressText")
    expect(stt_progress).to_contain_text("Uploading")

    # Wait for Cancel button to appear
    page.wait_for_selector("#sttCancel", state="visible")

    # Start TTS (parallel job)
    page.click('.tab[data-tab="tts"]')
    page.wait_for_selector("#ttsInput", state="visible")
    page.fill("#ttsInput", text_content)
    page.click("#ttsSubmit")

    # Switch back to STT tab
    page.click('.tab[data-tab="stt"]')
    page.wait_for_selector("#sttCancel", state="visible")

    # Cancel STT
    page.click("#sttCancel")

    # Restart STT with same file
    page.evaluate("""
        const file = window.testFile;
        const input = document.getElementById('sttFileInput');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    """)

    # Must progress beyond "Uploading..."
    page.wait_for_function(
        """() => {
            const text = document.getElementById('sttProgressText').textContent;
            return text.includes('Processing') || text.includes('Chunk');
        }"""
    )

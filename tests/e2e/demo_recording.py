"""Demo recording for README GIF.

Records: STT upload → TTS upload → both complete.
"""

from pathlib import Path

from playwright.sync_api import Page, expect

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "audio.wav"
TEXT_FILE = Path(__file__).parent.parent / "fixtures" / "text.txt"


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

    # Switch to TTS
    page.click('.tab[data-tab="tts"]')
    page.wait_for_selector("#ttsInput", state="visible")
    page.wait_for_timeout(600)

    # Upload TTS
    page.fill("#ttsInput", text_content)
    page.wait_for_timeout(400)
    page.click("#ttsSubmit")
    page.wait_for_timeout(600)

    # Wait for TTS complete
    page.wait_for_function(
        """() => {
            const text = document.getElementById('ttsProgressText').textContent;
            return text.includes('Complete');
        }""",
        timeout=60000,
    )
    page.wait_for_timeout(2800)

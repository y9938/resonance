"""Language picker tests cover selection, localization, and viewport placement."""

from playwright.sync_api import Page, expect


def test_language_pickers_search_and_relabel(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_selector("#sttLanguage:not([disabled])")

    locale = page.locator("#localePicker")
    locale.locator(".ts-control").click()
    locale.locator(".ts-control input").fill("Русский")
    locale.locator(".ts-dropdown .option").filter(has_text="Русский").first.click()

    language = page.locator(".stt-language-picker")
    language.locator(".ts-control").click()
    language.locator(".ts-control input").fill("Deutsch")
    language.locator(".ts-dropdown .option").filter(has_text="Deutsch").first.click()
    assert page.locator("#sttLanguage").input_value() == "de"
    assert "немецкий" in language.locator(".ts-control").inner_text().casefold()

    locale.locator(".ts-control").click()
    locale.locator(".ts-control input").fill("English")
    locale.locator(".ts-dropdown .option").filter(has_text="English").first.click()
    assert page.locator("#sttLanguage").input_value() == "de"
    expect(language.locator(".ts-control")).to_contain_text("German")
    page.reload()
    page.wait_for_selector("#sttLanguage:not([disabled])")
    assert page.locator("#sttLanguage").input_value() == "de"


def test_language_picker_fallback_and_keyboard(page: Page, base_url: str) -> None:
    page.add_init_script("localStorage.setItem('resonance_locale', 'ru')")
    page.goto(base_url)
    page.wait_for_selector("#sttLanguage:not([disabled])")
    language = page.locator(".stt-language-picker")
    search = language.locator(".ts-control input")
    language.locator(".ts-control").click()
    search.fill("Bashkir")
    language.locator(".ts-dropdown .option").filter(has_text="Bashkir").first.click()
    assert page.locator("#sttLanguage").input_value() == "ba"
    expect(language.locator(".ts-control")).to_contain_text("Bashkir")
    language.locator(".ts-control").click()
    search.fill("no-such-language")
    expect(language.locator(".locale-picker-empty")).to_contain_text("Подходящие языки не найдены")
    search.fill("Deutsch")
    search.press("ArrowDown")
    search.press("Enter")
    assert page.locator("#sttLanguage").input_value() == "de"


def test_stt_language_dropdown_uses_available_viewport_space(page: Page, base_url: str) -> None:
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(base_url)
    page.wait_for_selector("#sttLanguage:not([disabled])")
    page.locator(".stt-language-picker .ts-control").click()
    page.wait_for_function("""() => {
        const picker = document.querySelector('.stt-language-picker');
        const control = picker.querySelector('.ts-control').getBoundingClientRect();
        const menu = picker.querySelector('.ts-dropdown').getBoundingClientRect();
        return menu.height > 20 && menu.top >= control.bottom && menu.bottom <= innerHeight;
    }""")

    page.set_viewport_size({"width": 1280, "height": 560})
    page.wait_for_function("""() => {
        const picker = document.querySelector('.stt-language-picker');
        const control = picker.querySelector('.ts-control').getBoundingClientRect();
        const menu = picker.querySelector('.ts-dropdown').getBoundingClientRect();
        return menu.height > 20 && menu.bottom <= control.top && menu.top >= 0;
    }""")

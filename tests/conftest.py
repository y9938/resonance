"""Pytest configuration for e2e tests."""

import pytest


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Set viewport and video recording for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
        "record_video_size": {"width": 1920, "height": 1080},
    }


def pytest_collection_modifyitems(session, config, items) -> None:
    """Run Playwright-driven tests last.

    pytest-playwright uses Playwright's *sync* API, which spins up an asyncio loop in
    the main thread. If such a test errors/xfails, a "running loop" can remain set
    for the thread, and pytest-asyncio (strict) refuses to run async tests after
    that. Reordering avoids mixing these two loop lifecycles in one process.
    """

    def uses_playwright(item) -> bool:
        if "e2e" in str(item.fspath):
            return True
        fixture_names = getattr(item, "fixturenames", ())
        return any(
            name in fixture_names
            for name in ("page", "context", "browser", "playwright", "new_context")
        )

    items.sort(key=uses_playwright)

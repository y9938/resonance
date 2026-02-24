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

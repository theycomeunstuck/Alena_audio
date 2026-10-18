"""One Playwright lifetime shared by every browser acceptance module."""
import pytest


@pytest.fixture(scope='session')
def chromium():
    playwright_api = pytest.importorskip('playwright.sync_api')
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except playwright_api.Error as exc:
            pytest.skip(f'Playwright Chromium is not installed: {exc}')
        yield browser
        browser.close()

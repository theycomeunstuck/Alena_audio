"""Test fixtures for the isolated journal and browser acceptance tests."""
import os
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def isolated_worker_response_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MEMORY_RESPONSE_JOURNAL", str(tmp_path / "worker-responses.sqlite3"))
    monkeypatch.setenv("LOCAL_CHAT_RESPONSE_JOURNAL", str(tmp_path / "chat-responses.sqlite3"))


@pytest.fixture
def disposable_store():
    """A Store on a fresh disposable loopback PostgreSQL database (audit F-017).

    Mirrors the flow_store pattern in tests/test_memory_api_flow.py:
      - skips cleanly when TEACHCOPILOT_TEST_DSN is unset (no Store(None));
      - refuses non-loopback DSNs and validates the disposable-name contract
        via scripts/load_memory.require_disposable_dsn logic;
      - creates a per-test disposable database (teachcopilot_memory_verify_ +
        12 hex chars), migrates it, and drops it afterwards, so the database
        named in TEACHCOPILOT_TEST_DSN itself is never migrated or mutated.
    """
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        pytest.skip("set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    from scripts.load_memory import require_disposable_dsn
    from local_app.store import Store

    require_local_dsn(dsn)
    name = DATABASE_PREFIX + uuid4().hex[:12]
    # Self-check of the same guard the smoke scripts use: the per-test DSN must
    # be loopback and carry the disposable name pattern.
    require_disposable_dsn(make_dsn(dsn, dbname=name))
    admin = psycopg2.connect(make_dsn(dsn, dbname="postgres"))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        store = Store(make_dsn(dsn, dbname=name))
        store.migrate()
        yield store
    finally:
        with admin.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        admin.close()


@pytest.fixture(scope='module')
def chromium():
    playwright_api = pytest.importorskip('playwright.sync_api')
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except playwright_api.Error as exc:
            pytest.skip(f'Playwright Chromium is not installed: {exc}')
        yield browser
        browser.close()

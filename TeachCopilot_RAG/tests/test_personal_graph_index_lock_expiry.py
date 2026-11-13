"""Lease fencing under real PostgreSQL lock waits, using the disposable flow DB."""
import concurrent.futures
import time
from uuid import uuid4

import psycopg2
import pytest

from local_app import personal_graph_index as index
from tests.test_memory_api_flow import flow_store  # noqa: F401


def _wait_for_lock(cur, app_name):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        # The locker remains in one transaction, so refresh cached statistics.
        cur.execute('SELECT pg_stat_clear_snapshot()')
        cur.execute("""SELECT EXISTS(SELECT 1 FROM pg_stat_activity
            WHERE application_name=%s AND wait_event_type='Lock')""", (app_name,))
        if cur.fetchone()[0]:
            return
        time.sleep(0.05)
    pytest.fail('registry operation did not reach the PostgreSQL lock')


@pytest.mark.parametrize('operation', ['renew', 'finish'])
def test_expiry_during_row_lock_wait_fences_owner(flow_store, monkeypatch, operation):
    dsn = flow_store.dsn
    key = ('db', 'lock-test-' + uuid4().hex, 'snapshot', 'lessons', 'config')
    owner = index._claim(dsn, key, 'job', 60)
    app_name = 'pgi_lock_' + uuid4().hex
    monkeypatch.setattr(index, '_registry_connect',
                        lambda target: psycopg2.connect(target, connect_timeout=10, application_name=app_name))

    # Hold an uncommitted deadline change so the blocked operation sees the
    # previously live row first, then must recheck after the lock is released.
    with psycopg2.connect(dsn) as locker, locker.cursor() as cur:
        cur.execute("""UPDATE teachcopilot.personal_graph_indexes
            SET lease_until=clock_timestamp()+interval '1 second' WHERE run_id=%s RETURNING lease_until""",
                    (owner['run_id'],))
        deadline = cur.fetchone()[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            if operation == 'renew':
                pending = pool.submit(index._renew, dsn, owner['run_id'], owner['token'], 60)
            else:
                pending = pool.submit(index._finish, dsn, owner['run_id'], owner['token'], 'ready', {})
            try:
                _wait_for_lock(cur, app_name)
                while True:
                    cur.execute('SELECT clock_timestamp() >= %s', (deadline,))
                    if cur.fetchone()[0]:
                        break
                    time.sleep(0.05)
                locker.commit()
            finally:
                # A failed assertion must also release the lock before the
                # executor waits for the blocked operation to exit.
                locker.rollback()
            assert pending.result(timeout=10) is False

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT state, lease_until <= clock_timestamp() FROM teachcopilot.personal_graph_indexes WHERE run_id=%s",
                    (owner['run_id'],))
        assert cur.fetchone() == ('building', True)

    replacement = index._claim(dsn, key, 'replacement', 60)
    assert replacement['state'] == 'claimed'
    assert replacement['run_id'] != owner['run_id']
    assert index._renew(dsn, replacement['run_id'], replacement['token'], 60)
    assert index._finish(dsn, replacement['run_id'], replacement['token'], 'ready', {})


def test_claim_uses_current_time_after_advisory_lock_wait(flow_store, monkeypatch):
    dsn = flow_store.dsn
    key = ('db', 'lock-test-' + uuid4().hex, 'snapshot', 'lessons', 'config')
    app_name = 'pgi_claim_' + uuid4().hex
    monkeypatch.setattr(index, '_registry_connect',
                        lambda target: psycopg2.connect(target, connect_timeout=10, application_name=app_name))
    with psycopg2.connect(dsn) as locker, locker.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (index._canonical(key),))
        cur.execute("SELECT clock_timestamp()+interval '2 seconds'")
        deadline = cur.fetchone()[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(index._claim, dsn, key, 'job', 1)
            try:
                _wait_for_lock(cur, app_name)
                while True:
                    cur.execute('SELECT clock_timestamp() >= %s', (deadline,))
                    if cur.fetchone()[0]:
                        break
                    time.sleep(0.05)
                locker.commit()
            finally:
                locker.rollback()
            owner = pending.result(timeout=10)
    assert owner['state'] == 'claimed'
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT lease_until > clock_timestamp() FROM teachcopilot.personal_graph_indexes WHERE run_id=%s',
                    (owner['run_id'],))
        assert cur.fetchone()[0] is True

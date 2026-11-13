"""PostgreSQL lease and concurrent claim checks; set TEACHCOPILOT_TEST_DSN explicitly."""
import concurrent.futures
from uuid import uuid4

import pytest

from local_app import personal_graph_index as index

from tests.test_memory_api_flow import flow_store  # noqa: F401


@pytest.fixture
def key(flow_store):
    dsn = flow_store.dsn
    value = ('test-db', 'test-' + uuid4().hex, 'snapshot', 'lessons', 'config')
    yield dsn, value
    with index._registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute('DELETE FROM teachcopilot.personal_graph_indexes WHERE learner_id=%s', (value[1],))


def test_concurrent_claim_has_one_owner(key):
    dsn, key = key
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda n: index._claim(dsn, key, str(n), 60), range(6)))
    assert sum(x['state'] == 'claimed' for x in results) == 1
    assert len({x['run_id'] for x in results}) == 1


def test_expired_claim_gets_new_scope_and_old_owner_is_fenced(key):
    dsn, key = key
    first = index._claim(dsn, key, 'one', 60)
    with index._registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.personal_graph_indexes SET lease_until=now()-interval '1 second' WHERE run_id=%s", (first['run_id'],))
    second = index._claim(dsn, key, 'two', 60)
    assert first['run_id'] != second['run_id']
    assert not index._finish(dsn, first['run_id'], first['token'], 'ready', {})
    assert index._finish(dsn, second['run_id'], second['token'], 'ready', {})

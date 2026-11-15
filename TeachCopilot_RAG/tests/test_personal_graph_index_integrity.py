"""Integrity checks with disposable PostgreSQL and a source-scoped graph stub."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from local_app import personal_graph_index as index
from local_app.memory_ported.personal_memory_lab.archive import Archive
from local_app.memory_ported.personal_memory_lab.backends import scope
from tests.test_memory_api_flow import flow_store  # noqa: F401


LESSON = {'lesson_id': 'lesson', 'observed_on': '2025-01-01', 'turns': [
    {'turn_id': 'source', 'role': 'student', 'text': 'synthetic source text'}]}


class GraphDriver:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def execute_query(self, query, **kwargs):
        group = kwargs['group_id']
        assert kwargs['routing_'] == 'r'
        if 'labels(n)' in query:
            assert 'n {group_id: $group_id}' in query
            return deepcopy(self.nodes.get(group, [])), None, None
        assert 'r {group_id: $group_id}' in query and 'CASE WHEN endpoints_in_scope' in query
        rows = []
        for edge in self.edges.get(group, []):
            scoped = edge['source_group'] == group and edge['target_group'] == group
            rows.append({'endpoints_in_scope': scoped,
                         **{key: deepcopy(edge[key]) if scoped else None
                            for key in ('source_uuid', 'target_uuid', 'rel_type', 'props')}})
        return rows, None, None


class Adapter:
    def __init__(self, archive, driver):
        self.archive, self.driver = archive, driver
        self.calls = 0

    async def ingest(self, run_id, learner_id, lesson):
        self.calls += 1
        assert self.archive.claim(run_id, learner_id, lesson['lesson_id'], 'graphiti')
        group = scope(run_id, learner_id)
        ep, entity_a, entity_b = group + '-ep', group + '-a', group + '-b'
        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.driver.nodes[group] = [
            {'labels': ['Episodic'], 'props': {'uuid': ep, 'group_id': group,
                                               'content': 'synthetic episode', 'created_at': created}},
            {'labels': ['Entity', 'Node'], 'props': {'uuid': entity_a, 'group_id': group,
                                                     'name': 'first', 'name_embedding': [0.1, 0.2]}},
            {'labels': ['Entity'], 'props': {'uuid': entity_b, 'group_id': group, 'name': 'second'}},
        ]
        self.driver.edges[group] = [
            {'source_uuid': ep, 'source_group': group, 'target_uuid': entity_a, 'target_group': group,
             'rel_type': 'MENTIONS', 'props': {'uuid': group + '-m', 'group_id': group}},
            {'source_uuid': entity_a, 'source_group': group, 'target_uuid': entity_b, 'target_group': group,
             'rel_type': 'RELATES_TO', 'props': {'uuid': group + '-r', 'group_id': group,
                                                 'fact': 'synthetic fact', 'fact_embedding': [0.2, 0.3]}},
        ]
        self.archive.finish(run_id, learner_id, lesson['lesson_id'], 'graphiti',
                            metadata={'episode_uuids': [ep]})


class Runtime:
    model_identities = {'chat': {'digest': 'bound'}, 'embedding': {'digest': 'bound'}}
    drift = False

    def embed_many(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def check_model_identities(self):
        if self.drift:
            raise ValueError('model tag changed')


def _args(store, learner, archive, graph, adapter):
    return dict(dsn=store.dsn, learner_id=learner, job_id='job', snapshot_fingerprint='snapshot',
                lessons=[LESSON], runtime=Runtime(), archive=archive, graph=graph, adapter=adapter,
                config_fingerprint='config', current_snapshot=lambda: 'snapshot')


def _ensure(args):
    return asyncio.run(index.ensure_personal_graph_index(**args))


def _health(args, result):
    return asyncio.run(index._health(args['archive'], args['graph'], result['run_id'], args['learner_id'],
                                     args['lessons'], result['manifest'], dsn=args['dsn']))


def _replace_vector(dsn, run_id, learner_id):
    literal = '[' + ','.join(['0', '1'] + ['0'] * 1022) + ']'
    with index._registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""UPDATE nirs_personal_memory.source_chunks SET embedding=%s::vector
            WHERE run_id=%s AND learner_id=%s""", (literal, run_id, learner_id))
        assert cur.rowcount == 1


def test_changed_vector_and_deleted_graph_edge_rebuild_in_new_scopes(flow_store):
    store = flow_store
    learner = 'integrity-' + uuid4().hex
    archive = Archive(store.dsn)
    archive.initialize()
    driver = GraphDriver()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    args = _args(store, learner, archive, graph, adapter)

    first = _ensure(args)
    assert not first['reused'] and _ensure(args)['reused']
    assert first['manifest']['source_vector_count'] == 1
    assert first['manifest']['graph_node_count'] == 3
    assert first['manifest']['graph_edge_count'] == 2
    assert 'synthetic source' not in str(first['manifest'])
    assert 'synthetic fact' not in str(first['manifest'])

    _replace_vector(store.dsn, first['run_id'], learner)
    assert _health(args, first) is None
    second = _ensure(args)
    assert not second['reused'] and second['run_id'] != first['run_id']
    assert _ensure(args)['reused']

    group = scope(second['run_id'], learner)
    driver.edges[group].pop()  # RELATES_TO deleted; the episodic node still exists.
    assert any('Episodic' in node['labels'] for node in driver.nodes[group])
    assert _health(args, second) is None
    third = _ensure(args)
    assert not third['reused'] and third['run_id'] != second['run_id']
    assert adapter.calls == 3
    with index._registry_connect(store.dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT state FROM teachcopilot.personal_graph_indexes WHERE run_id=ANY(%s)',
                    ([first['run_id'], second['run_id']],))
        assert [row[0] for row in cur.fetchall()] == ['invalid', 'invalid']

    # The canonical hash is independent of Neo4j result and label order.
    group = scope(third['run_id'], learner)
    driver.nodes[group].reverse()
    driver.edges[group].reverse()
    driver.nodes[group][1]['labels'].reverse()
    assert _health(args, third) == third['manifest']
    driver.edges[group][0]['props']['fact'] = 'changed synthetic fact'
    assert _health(args, third) is None


def test_other_learner_changes_do_not_invalidate_or_enter_manifest(flow_store):
    archive = Archive(flow_store.dsn)
    archive.initialize()
    driver = GraphDriver()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    first_args = _args(flow_store, 'learner-a-' + uuid4().hex, archive, graph, adapter)
    other_args = _args(flow_store, 'learner-b-' + uuid4().hex, archive, graph, adapter)
    first, other = _ensure(first_args), _ensure(other_args)
    _replace_vector(flow_store.dsn, other['run_id'], other_args['learner_id'])
    driver.nodes[scope(other['run_id'], other_args['learner_id'])][0]['props']['content'] = 'other learner change'
    assert _health(first_args, first) == first['manifest']
    assert _ensure(first_args)['reused']
    assert 'other learner change' not in str(first['manifest'])

    # Even an edge labelled with this group cannot point into a different group.
    own_group = scope(first['run_id'], first_args['learner_id'])
    driver.edges[own_group][0]['target_group'] = scope(other['run_id'], other_args['learner_id'])
    assert _health(first_args, first) is None


def test_neo4j_temporal_point_and_vector_normalization_is_deterministic():
    neo4j_time = pytest.importorskip('neo4j.time')
    neo4j_spatial = pytest.importorskip('neo4j.spatial')
    neo4j_vector = pytest.importorskip('neo4j.vector')
    created = neo4j_time.DateTime(2025, 1, 1, 12, 30, 1, 123456789)
    changed = neo4j_time.DateTime(2025, 1, 1, 12, 30, 1, 123456788)
    point = neo4j_spatial.CartesianPoint((1.0, 2.0))
    vector = neo4j_vector.Vector([0.25, 0.5], 'f32')
    first = index._graph_value({'time': created, 'point': point, 'vector': vector})
    assert index._hash(first) == index._hash(index._graph_value({'vector': vector, 'time': created, 'point': point}))
    assert index._hash(first) != index._hash(index._graph_value({'time': changed, 'point': point, 'vector': vector}))


def test_chat_validation_is_read_only_and_requires_exact_ready_index(flow_store, monkeypatch):
    archive = Archive(flow_store.dsn)
    archive.initialize()
    driver = GraphDriver()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    args = _args(flow_store, 'chat-' + uuid4().hex, archive, graph, adapter)
    built = _ensure(args)
    monkeypatch.setattr(index, 'personal_graph_config_fingerprint', lambda runtime, identities: 'config')
    monkeypatch.setattr(index, '_claim', lambda *args: pytest.fail('chat must never claim an index'))
    monkeypatch.setattr(index, '_invalidate', lambda *args: pytest.fail('chat must never mutate an index'))
    request = dict(dsn=flow_store.dsn, run_id=built['run_id'], learner_id=args['learner_id'],
                   snapshot_fingerprint='snapshot', lessons=[LESSON], runtime=args['runtime'],
                   archive=archive, graph=graph, approved_config_fingerprint='config',
                   approved_manifest=built['manifest'])

    def validate(**overrides):
        return asyncio.run(index.validate_ready_personal_graph_index(**{**request, **overrides}))

    assert validate() == built['manifest']
    for overrides in ({'run_id': 'pre-registry-run'}, {'learner_id': 'other-learner'},
                      {'snapshot_fingerprint': 'changed'}, {'approved_config_fingerprint': 'old-model'},
                      {'approved_manifest': {}}, {'approved_manifest': {'graph_node_count': 3}},
                      {'lessons': [{**LESSON, 'observed_on': '2025-01-02'}]}):
        with pytest.raises(index.PersonalGraphIndexStale, match='обновить'):
            validate(**overrides)

    args['runtime'].drift = True
    with pytest.raises(index.PersonalGraphIndexStale, match='обновить'):
        validate()
    args['runtime'].drift = False
    group = scope(built['run_id'], args['learner_id'])
    removed = driver.edges[group].pop()
    with pytest.raises(index.PersonalGraphIndexStale, match='обновить'):
        validate()
    driver.edges[group].append(removed)
    _replace_vector(flow_store.dsn, built['run_id'], args['learner_id'])
    with pytest.raises(index.PersonalGraphIndexStale, match='обновить'):
        validate()
    with index._registry_connect(flow_store.dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT state FROM teachcopilot.personal_graph_indexes WHERE run_id=%s', (built['run_id'],))
        assert cur.fetchone()[0] == 'ready'


def test_chat_validation_preserves_transport_failures(flow_store, monkeypatch):
    archive = Archive(flow_store.dsn)
    archive.initialize()
    driver = GraphDriver()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    args = _args(flow_store, 'transport-' + uuid4().hex, archive, graph, adapter)
    built = _ensure(args)
    monkeypatch.setattr(index, 'personal_graph_config_fingerprint', lambda runtime, identities: 'config')
    async def unavailable(*args, **kwargs):
        raise ConnectionError('synthetic graph transport failure')
    monkeypatch.setattr(driver, 'execute_query', unavailable)
    with pytest.raises(ConnectionError, match='transport failure'):
        asyncio.run(index.validate_ready_personal_graph_index(dsn=flow_store.dsn, run_id=built['run_id'],
            learner_id=args['learner_id'], snapshot_fingerprint='snapshot', lessons=[LESSON],
            runtime=args['runtime'], archive=archive, graph=graph, approved_config_fingerprint='config',
            approved_manifest=built['manifest']))

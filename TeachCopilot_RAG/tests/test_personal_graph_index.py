import asyncio
from copy import deepcopy
from unittest.mock import patch

import pytest

from local_app import personal_graph_index as index

LESSON = {'lesson_id': 'a', 'observed_on': '2025-01-01', 'turns': [
    {'turn_id': 't', 'role': 'student', 'text': 'hello'}]}


@pytest.fixture(autouse=True)
def fake_source_vector_integrity(monkeypatch):
    monkeypatch.setattr(index, '_source_vector_integrity', lambda *args: (
        ['a:c0001'], {'source_vector_count': 1, 'source_vector_sha256': 'fake-vector-digest'}))


class Archive:
    def __init__(self):
        self.data, self.parts, self.refs = {}, {}, {}
    def put(self, run, learner, lesson):
        self.data[run] = [deepcopy(lesson)]
    def put_chunks(self, run, learner, lesson, parts, vectors):
        self.parts[run] = [{'chunk_id': 'a:c0001', 'lesson_id': 'a', 'position': 0, 'content': deepcopy(parts[0])}]
    def lessons(self, run, learner):
        return self.data.get(run, [])
    def chunks(self, run, learner):
        return self.parts.get(run, [])
    def index_references(self, run, learner, backend):
        return self.refs.get(run, [])


class Runtime:
    def __init__(self): self.calls = 0
    def embed_many(self, texts):
        self.calls += 1
        return [[1] + [0] * 1023 for _ in texts]


class Driver:
    def __init__(self): self.episodes = []
    async def execute_query(self, query, **kwargs):
        group_id = kwargs['group_id']
        if 'labels(n)' in query:
            return ([{'labels': ['Episodic'], 'props': {'uuid': value, 'group_id': group_id}}
                     for value in self.episodes], None, None)
        assert 'type(r)' in query
        return ([], None, None)


class Adapter:
    def __init__(self, archive, driver): self.archive, self.driver, self.calls = archive, driver, 0
    async def ingest(self, run, learner, lesson):
        self.calls += 1
        self.archive.refs[run] = [{'lesson_id': 'a', 'metadata': {'episode_uuids': ['ep']}}]
        self.driver.episodes = ['ep']


class Registry:
    def __init__(self): self.row = None
    def claim(self, dsn, key, job, seconds):
        if self.row: return dict(self.row)
        self.row = {'run_id': 'run', 'state': 'building'}
        return {'run_id': 'run', 'state': 'claimed', 'token': 'token'}
    def finish(self, dsn, run, token, state, manifest):
        self.row = {'run_id': run, 'state': state, 'manifest': manifest}
        return True
    def invalidate(self, dsn, run): self.row = None


def test_build_then_reuse_without_embedding_or_ingest():
    archive, runtime, driver, registry = Archive(), Runtime(), Driver(), Registry()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    args = dict(dsn='ignored', learner_id='l', job_id='j', snapshot_fingerprint='s', lessons=[LESSON],
                runtime=runtime, archive=archive, graph=graph, adapter=adapter, config_fingerprint='c')
    with patch.object(index, 'database_fingerprint', return_value='db'), patch.object(index, '_claim', registry.claim), \
         patch.object(index, '_finish', registry.finish), patch.object(index, '_invalidate', registry.invalidate):
        first = asyncio.run(index.ensure_personal_graph_index(**args))
        args['job_id'] = 'other'
        second = asyncio.run(index.ensure_personal_graph_index(**args))
    assert first['reused'] is False and second['reused'] is True
    assert runtime.calls == adapter.calls == 1


def test_missing_episode_fails_health():
    archive, runtime, driver, registry = Archive(), Runtime(), Driver(), Registry()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    args = dict(dsn='ignored', learner_id='l', job_id='j', snapshot_fingerprint='s', lessons=[LESSON],
                runtime=runtime, archive=archive, graph=graph, adapter=adapter, config_fingerprint='c')
    with patch.object(index, 'database_fingerprint', return_value='db'), patch.object(index, '_claim', registry.claim), \
         patch.object(index, '_finish', registry.finish):
        asyncio.run(index.ensure_personal_graph_index(**args))
        driver.episodes = []
        assert asyncio.run(index._health(archive, graph, 'run', 'l', [LESSON], registry.row['manifest'])) is None
        archive.parts['run'][0]['content']['turns'][0]['text'] = 'changed'
        assert asyncio.run(index._health(archive, graph, 'run', 'l', [LESSON])) is None


def test_snapshot_change_marks_uncertain():
    archive, runtime, driver, registry = Archive(), Runtime(), Driver(), Registry()
    graph = type('Graph', (), {'driver': driver})()
    adapter = Adapter(archive, driver)
    with patch.object(index, 'database_fingerprint', return_value='db'), patch.object(index, '_claim', registry.claim), \
         patch.object(index, '_finish', registry.finish):
        with pytest.raises(ValueError, match='snapshot changed'):
            asyncio.run(index.ensure_personal_graph_index(dsn='x', learner_id='l', job_id='j',
                snapshot_fingerprint='s', lessons=[LESSON], runtime=runtime, archive=archive,
                graph=graph, adapter=adapter, config_fingerprint='c', current_snapshot=lambda: 'new'))
    assert registry.row['state'] == 'uncertain'


def test_ingest_failure_is_uncertain_and_never_marked_ready():
    archive, runtime, driver, registry = Archive(), Runtime(), Driver(), Registry()
    graph = type('Graph', (), {'driver': driver})()
    class FailingAdapter:
        async def ingest(self, run, learner, lesson):
            raise TimeoutError('write outcome unknown')
    with patch.object(index, 'database_fingerprint', return_value='db'), patch.object(index, '_claim', registry.claim), \
         patch.object(index, '_finish', registry.finish):
        with pytest.raises(TimeoutError):
            asyncio.run(index.ensure_personal_graph_index(dsn='x', learner_id='l', job_id='j',
                snapshot_fingerprint='s', lessons=[LESSON], runtime=runtime, archive=archive,
                graph=graph, adapter=FailingAdapter(), config_fingerprint='c'))
    assert registry.row['state'] == 'uncertain'

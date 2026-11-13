"""Exact-snapshot, fail-closed reuse of disposable personal graph indexes."""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit
from uuid import uuid4

from .memory_response_journal import database_fingerprint


_STALE_INDEX_MESSAGE = ('Индекс личной памяти изменился или недоступен. '
                        'Преподавателю нужно обновить и заново утвердить память.')


class PersonalGraphIndexStale(ValueError):
    """The approved archive index must be rebuilt and reviewed again."""
    def __init__(self):
        super().__init__(_STALE_INDEX_MESSAGE)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def resolve_model_identities(runtime):
    """Resolve immutable Ollama model digests; never include endpoint credentials."""
    base = runtime.base_url.rstrip('/')
    endpoint = urlsplit(base)
    if (endpoint.scheme != 'http' or endpoint.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('Model identities require a local Ollama endpoint')
    with urllib.request.urlopen(base + '/api/tags', timeout=10) as response:
        tags = json.load(response)
    entries = tags.get('models')
    if not isinstance(entries, list):
        raise ValueError('Ollama model identities unavailable')
    def find(name):
        aliases = {name, name + ':latest'} if ':' not in name else {name}
        matches = [row for row in entries if row.get('name') in aliases or row.get('model') in aliases]
        digests = {row.get('digest') for row in matches}
        if len(digests) != 1 or not all(isinstance(v, str) and re.fullmatch(r'(?:sha256:)?[0-9a-fA-F]{64}', v) for v in digests):
            raise ValueError('Ollama model digest unavailable or ambiguous')
        return {'name': name, 'digest': next(iter(digests))}
    embedding_name = getattr(getattr(getattr(runtime, 'runtime', None), 'settings', None), 'embedding_model', 'bge-m3')
    return {'chat': find(runtime.model), 'embedding': find(embedding_name)}


def personal_graph_config_fingerprint(runtime, model_identities=None):
    """Hash all local index-producing code, library versions and model identities."""
    identities = model_identities or resolve_model_identities(runtime)
    for key in ('chat', 'embedding'):
        if (not isinstance(identities.get(key), dict)
                or not isinstance(identities[key].get('name'), str) or not identities[key]['name']
                or not isinstance(identities[key].get('digest'), str)
                or not re.fullmatch(r'(?:sha256:)?[0-9a-fA-F]{64}', identities[key]['digest'])):
            raise ValueError('Immutable model digests are required')
    root = Path(__file__).resolve().parents[1]
    files = ['research_memory/personal_memory_lab/backends.py',
             'research_memory/personal_memory_lab/chunks.py',
             'research_memory/personal_memory_lab/archive.py',
             'local_app/personal_graph_memory.py', 'local_app/personal_graph_contract.py',
             'local_app/personal_graph_index.py', 'local_app/runtime.py',
             'local_app/memory_service.py', 'research_memory/clients.py']
    versions = {}
    for package in ('graphiti-core', 'neo4j', 'pydantic'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError('Index library version unavailable: ' + package) from None
    return _hash({'protocol': 'personal-graph-exact-snapshot-v1', 'models': identities,
                  'context': runtime.context, 'chunks': [1800, 6000], 'versions': versions,
                  'code': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}})


def _registry_connect(dsn):
    import psycopg2
    return psycopg2.connect(dsn, connect_timeout=10)


def _claim(dsn, key, job_id, lease_seconds):
    from psycopg2.extras import Json
    token, run_id = uuid4().hex, 'app-index-' + uuid4().hex
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        # Serialize only short registry transactions, never model or graph calls.
        cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (_canonical(key),))
        cur.execute("""UPDATE teachcopilot.personal_graph_indexes SET state='uncertain',
            owner_token=NULL, updated_at=clock_timestamp() WHERE database_identity=%s AND learner_id=%s
            AND snapshot_fingerprint=%s AND lessons_hash=%s AND config_fingerprint=%s
            AND state='building' AND lease_until <= clock_timestamp()""", key)
        cur.execute("""SELECT run_id,state,manifest FROM teachcopilot.personal_graph_indexes
            WHERE database_identity=%s AND learner_id=%s AND snapshot_fingerprint=%s
            AND lessons_hash=%s AND config_fingerprint=%s AND state IN ('building','ready')""", key)
        row = cur.fetchone()
        if row:
            return {'run_id': row[0], 'state': row[1], 'manifest': row[2]}
        cur.execute("""INSERT INTO teachcopilot.personal_graph_indexes
            (run_id,database_identity,learner_id,snapshot_fingerprint,lessons_hash,config_fingerprint,
             state,owner_token,job_id,lease_until,manifest)
            VALUES (%s,%s,%s,%s,%s,%s,'building',%s,%s,clock_timestamp()+(%s * interval '1 second'),%s)""",
            (run_id, *key, token, str(job_id), lease_seconds, Json({})))
    return {'run_id': run_id, 'state': 'claimed', 'token': token}


def _renew(dsn, run_id, token, lease_seconds):
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        # Acquire the row before checking time: an UPDATE's predicate can be
        # evaluated before waiting on a lock, and now() is transaction-scoped.
        cur.execute('SELECT run_id FROM teachcopilot.personal_graph_indexes WHERE run_id=%s FOR UPDATE', (run_id,))
        cur.execute("""UPDATE teachcopilot.personal_graph_indexes SET lease_until=clock_timestamp()+(%s * interval '1 second'),
            updated_at=clock_timestamp() WHERE run_id=%s AND owner_token=%s
            AND state='building' AND lease_until>clock_timestamp()""",
            (lease_seconds, run_id, token))
        return cur.rowcount == 1


def _finish(dsn, run_id, token, state, manifest):
    from psycopg2.extras import Json
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        # Check expiry in a separate statement after any row-lock wait.
        cur.execute('SELECT run_id FROM teachcopilot.personal_graph_indexes WHERE run_id=%s FOR UPDATE', (run_id,))
        cur.execute("""UPDATE teachcopilot.personal_graph_indexes SET state=%s,manifest=%s,
            owner_token=NULL,lease_until=NULL,updated_at=clock_timestamp() WHERE run_id=%s AND owner_token=%s
            AND state='building' AND lease_until>clock_timestamp()""", (state, Json(manifest), run_id, token))
        return cur.rowcount == 1


def _invalidate(dsn, run_id):
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.personal_graph_indexes SET state='invalid',updated_at=now() WHERE run_id=%s AND state='ready'", (run_id,))


def _source_vector_integrity(dsn, run_id, learner_id):
    """Read the stored pgvector values, not the original pre-quantized inputs."""
    from .personal_graph_retrieval import _parse_cached_vector
    if not isinstance(dsn, str) or not dsn:
        raise ValueError('A database is required to verify source vectors')
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT chunk_id, embedding::text FROM nirs_personal_memory.source_chunks
            WHERE run_id=%s AND learner_id=%s""", (run_id, learner_id))
        rows = sorted(cur.fetchall(), key=lambda row: row[0])
    if any(not isinstance(chunk_id, str) or not chunk_id or not isinstance(vector, str)
           for chunk_id, vector in rows):
        raise ValueError('Invalid stored source vector')
    for _, vector in rows:
        _parse_cached_vector(vector)
    return [chunk_id for chunk_id, _ in rows], {
        'source_vector_count': len(rows), 'source_vector_sha256': _hash(rows)}


def _graph_value(value):
    """Convert Bolt property values to deterministic JSON without losing type data."""
    kind = type(value)
    if kind.__module__ == 'neo4j.time' and hasattr(value, 'iso_format'):
        result = {'type': kind.__name__, 'value': value.iso_format()}
        tz = getattr(value, 'tzinfo', None)
        if tz is not None:
            result['timezone'] = str(tz)
        return {'neo4j_temporal': result}
    if kind.__module__ == 'neo4j.spatial' and hasattr(value, 'srid'):
        return {'neo4j_point': {'type': kind.__name__, 'srid': value.srid,
                                'coordinates': [_graph_value(item) for item in value]}}
    if kind.__module__ == 'neo4j.vector' and hasattr(value, 'raw'):
        return {'neo4j_vector': {'dtype': str(value.dtype), 'bytes': value.raw(byteorder='big').hex()}}
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return {'python_temporal': {'type': kind.__name__, 'value': value.isoformat()}}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite graph property')
        return value
    if isinstance(value, (bytes, bytearray)):
        return {'bytes': bytes(value).hex()}
    if isinstance(value, (list, tuple)):
        return [_graph_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _graph_value(item) for key, item in value.items()}
    raise ValueError('Unsupported graph property type')


async def _graph_integrity(graph, group_id):
    """Fingerprint graph objects in one scope, including relation endpoints."""
    node_rows, _, _ = await graph.driver.execute_query(
        'MATCH (n {group_id: $group_id}) RETURN labels(n) AS labels, properties(n) AS props',
        group_id=group_id, routing_='r')
    edge_rows, _, _ = await graph.driver.execute_query(
        """MATCH (source)-[r {group_id: $group_id}]->(target)
        WITH source,r,target,
             coalesce(source.group_id = $group_id AND target.group_id = $group_id, false) AS endpoints_in_scope
        RETURN endpoints_in_scope,
               CASE WHEN endpoints_in_scope THEN source.uuid ELSE null END AS source_uuid,
               CASE WHEN endpoints_in_scope THEN target.uuid ELSE null END AS target_uuid,
               CASE WHEN endpoints_in_scope THEN type(r) ELSE null END AS rel_type,
               CASE WHEN endpoints_in_scope THEN properties(r) ELSE null END AS props""",
        group_id=group_id, routing_='r')
    nodes, edges, node_ids, episodes = [], [], set(), []
    for row in node_rows:
        labels, props = row['labels'], row['props']
        if (not isinstance(labels, (list, tuple)) or not all(isinstance(label, str) for label in labels)
                or not isinstance(props, dict) or props.get('group_id') != group_id
                or not isinstance(props.get('uuid'), str) or not props['uuid'] or props['uuid'] in node_ids):
            raise ValueError('Invalid scoped graph node')
        node_ids.add(props['uuid'])
        if 'Episodic' in labels:
            episodes.append(props['uuid'])
        nodes.append({'labels': sorted(labels), 'props': _graph_value(props)})
    for row in edge_rows:
        props = row['props']
        if (row['endpoints_in_scope'] is not True or row['source_uuid'] not in node_ids
                or row['target_uuid'] not in node_ids
                or not isinstance(row['rel_type'], str) or not row['rel_type']
                or not isinstance(props, dict) or props.get('group_id') != group_id
                or not isinstance(props.get('uuid'), str) or not props['uuid']):
            raise ValueError('Invalid scoped graph edge')
        edges.append({'source_uuid': row['source_uuid'], 'target_uuid': row['target_uuid'],
                      'type': row['rel_type'], 'props': _graph_value(props)})
    nodes.sort(key=_canonical)
    edges.sort(key=_canonical)
    return sorted(episodes), {'graph_node_count': len(nodes), 'graph_edge_count': len(edges),
                              'graph_sha256': _hash({'nodes': nodes, 'edges': edges})}


async def _health(archive, graph, run_id, learner_id, lessons, expected_manifest=None, *, dsn=None):
    from research_memory.personal_memory_lab.backends import scope
    from research_memory.personal_memory_lab.chunks import lesson_chunks
    archived = await asyncio.to_thread(archive.lessons, run_id, learner_id)
    if {x['lesson_id']: _canonical(x) for x in archived} != {x['lesson_id']: _canonical(x) for x in lessons} or len(archived) != len(lessons):
        return None
    expected_chunks = {}
    for lesson in lessons:
        for pos, part in enumerate(lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000)):
            expected_chunks[lesson['lesson_id'] + ':c' + str(pos + 1).zfill(4)] = (lesson['lesson_id'], pos, _canonical(part))
    chunks = await asyncio.to_thread(archive.chunks, run_id, learner_id)
    actual_chunks = {r['chunk_id']: (r['lesson_id'], r['position'], _canonical(r['content'])) for r in chunks}
    if actual_chunks != expected_chunks or len(chunks) != len(expected_chunks):
        return None
    refs = await asyncio.to_thread(archive.index_references, run_id, learner_id, 'graphiti')
    if len(refs) != len(lessons) or {r['lesson_id'] for r in refs} != {x['lesson_id'] for x in lessons}:
        return None
    episodes = []
    for ref in refs:
        values = ref['metadata'].get('episode_uuids')
        if not isinstance(values, list) or not values or not all(isinstance(x, str) and x for x in values):
            return None
        episodes.extend(values)
    if len(set(episodes)) != len(episodes):
        return None
    try:
        vector_ids, vector_manifest = await asyncio.to_thread(
            _source_vector_integrity, dsn or getattr(archive, 'database_url', None), run_id, learner_id)
        found, graph_manifest = await _graph_integrity(graph, scope(run_id, learner_id))
    except ValueError:
        return None
    if (len(vector_ids) != len(expected_chunks) or set(vector_ids) != set(expected_chunks)
            or found != sorted(episodes)):
        return None
    manifest = {'lesson_count': len(lessons), 'chunk_count': len(chunks), 'episode_uuids': sorted(episodes),
                **vector_manifest, **graph_manifest}
    if expected_manifest is not None and manifest != expected_manifest:
        return None
    return manifest


def _ready_index_manifest(dsn, run_id, key):
    """Read the exact registry row without claiming, invalidating or rebuilding it."""
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT manifest FROM teachcopilot.personal_graph_indexes
            WHERE run_id=%s AND database_identity=%s AND learner_id=%s
            AND snapshot_fingerprint=%s AND lessons_hash=%s AND config_fingerprint=%s
            AND state='ready'""", (run_id, *key))
        row = cur.fetchone()
    return row[0] if row else None


def find_ready_personal_graph_index(dsn, learner_id, snapshot_fingerprint, lessons, config_fingerprint):
    """Read-only lookup of the newest ready index for this exact learner scope.

    Returns {'run_id', 'manifest'} when a ready row matches the current database,
    source snapshot, lesson set and config fingerprint; None otherwise. Never
    claims, invalidates or rebuilds anything.
    """
    from psycopg2.extensions import parse_dsn
    params = parse_dsn(dsn)
    addresses = [params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR")]
    if not any(addresses) or any(value and value not in {"127.0.0.1", "localhost", "::1"} for value in addresses):
        raise ValueError("personal graph requires a local PostgreSQL database")
    key = (database_fingerprint(dsn), learner_id, snapshot_fingerprint, _hash(lessons), config_fingerprint)
    with _registry_connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""SELECT run_id, manifest FROM teachcopilot.personal_graph_indexes
            WHERE database_identity=%s AND learner_id=%s AND snapshot_fingerprint=%s
            AND lessons_hash=%s AND config_fingerprint=%s AND state='ready'
            ORDER BY updated_at DESC LIMIT 1""", key)
        row = cur.fetchone()
    return {"run_id": row[0], "manifest": row[1]} if row else None


async def validate_ready_personal_graph_index(*, dsn, run_id, learner_id, snapshot_fingerprint,
                                              lessons, runtime, archive, graph,
                                              approved_config_fingerprint, approved_manifest):
    """Refuse stale approved indexes without changing the registry or the graph."""
    if (not all(isinstance(value, str) and value for value in
                (run_id, learner_id, snapshot_fingerprint, approved_config_fingerprint))
            or not isinstance(lessons, list) or not isinstance(approved_manifest, dict)
            or not approved_manifest or not isinstance(getattr(runtime, 'model_identities', None), dict)
            or not callable(getattr(runtime, 'check_model_identities', None))):
        raise PersonalGraphIndexStale()

    async def check_model():
        try:
            await asyncio.to_thread(runtime.check_model_identities)
        except ValueError as exc:
            raise PersonalGraphIndexStale() from exc

    await check_model()
    current_config = await asyncio.to_thread(personal_graph_config_fingerprint,
                                             runtime, runtime.model_identities)
    if current_config != approved_config_fingerprint:
        raise PersonalGraphIndexStale()
    key = (database_fingerprint(dsn), learner_id, snapshot_fingerprint, _hash(lessons), current_config)
    if await asyncio.to_thread(_ready_index_manifest, dsn, run_id, key) != approved_manifest:
        raise PersonalGraphIndexStale()
    manifest = await _health(archive, graph, run_id, learner_id, lessons, approved_manifest, dsn=dsn)
    if manifest is None or await asyncio.to_thread(_ready_index_manifest, dsn, run_id, key) != approved_manifest:
        raise PersonalGraphIndexStale()
    await check_model()
    return manifest


async def ensure_personal_graph_index(*, dsn, learner_id, job_id, snapshot_fingerprint, lessons,
                                      runtime, archive, graph, adapter, config_fingerprint,
                                      current_snapshot=None, lease_seconds=600):
    """Return an existing verified scope or build a fresh, fenced scope."""
    from research_memory.personal_memory_lab.backends import lesson_text
    from research_memory.personal_memory_lab.chunks import lesson_chunks
    if not all(isinstance(x, str) and x for x in (learner_id, str(job_id), snapshot_fingerprint, config_fingerprint)):
        raise ValueError('Index identity is incomplete')
    if lease_seconds < 3 or not isinstance(lessons, list):
        raise ValueError('Invalid index input')
    if len({x['lesson_id'] for x in lessons}) != len(lessons):
        raise ValueError('Duplicate lessons')
    key = (database_fingerprint(dsn), learner_id, snapshot_fingerprint, _hash(lessons), config_fingerprint)
    async def check_snapshot():
        if current_snapshot is not None:
            value = await asyncio.to_thread(current_snapshot)
            if value != snapshot_fingerprint:
                raise ValueError('source snapshot changed during personal graph indexing')
    while True:
        row = await asyncio.to_thread(_claim, dsn, key, job_id, lease_seconds)
        run_id = row['run_id']
        if row['state'] == 'building':
            await asyncio.sleep(min(2, lease_seconds / 3))
            continue
        if row['state'] == 'ready':
            manifest = await _health(archive, graph, run_id, learner_id, lessons, row['manifest'], dsn=dsn)
            if manifest is None:
                await asyncio.to_thread(_invalidate, dsn, run_id)
                continue
            await check_snapshot()
            return {'run_id': run_id, 'reused': True, 'manifest': manifest}
        break
    token = row['token']
    lost = asyncio.Event()
    async def heartbeat():
        while True:
            await asyncio.sleep(lease_seconds / 3)
            try:
                if not await asyncio.to_thread(_renew, dsn, run_id, token, lease_seconds):
                    lost.set()
                    return
            except Exception:
                lost.set()
                return
    task = asyncio.create_task(heartbeat())
    try:
        for lesson in lessons:
            if lost.is_set():
                raise RuntimeError('personal graph index lease lost')
            await asyncio.to_thread(archive.put, run_id, learner_id, lesson)
            parts = lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000)
            vectors = []
            for start in range(0, len(parts), 16):
                vectors.extend(await asyncio.to_thread(runtime.embed_many, [lesson_text(p) for p in parts[start:start+16]]))
            await asyncio.to_thread(archive.put_chunks, run_id, learner_id, lesson, parts, vectors)
            await adapter.ingest(run_id, learner_id, lesson)
        manifest = await _health(archive, graph, run_id, learner_id, lessons, dsn=dsn)
        if manifest is None:
            raise RuntimeError('personal graph index verification failed')
        await check_snapshot()
        if hasattr(runtime, 'check_model_identities'):
            await asyncio.to_thread(runtime.check_model_identities)
        if lost.is_set() or not await asyncio.to_thread(_finish, dsn, run_id, token, 'ready', manifest):
            raise RuntimeError('personal graph index lease lost')
        return {'run_id': run_id, 'reused': False, 'manifest': manifest}
    except BaseException:
        try:
            await asyncio.to_thread(_finish, dsn, run_id, token, 'uncertain', {})
        except Exception:
            pass
        raise
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

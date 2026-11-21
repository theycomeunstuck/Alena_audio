"""Upgrade synthetic pre-024 history without rewriting its existing records."""
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
from psycopg2.extras import Json

import local_app.store as store_module
from local_app.store import Store


@pytest.fixture
def migration_store():
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN", "")
    params = parse_dsn(dsn) if dsn else {}
    if params.get("host") not in {"localhost", "127.0.0.1", "::1"} or not re.fullmatch(
            r"teachcopilot_memory_verify_[0-9a-f]{12}", params.get("dbname", "")):
        pytest.skip("requires a freshly created disposable migration-verification database")
    name = "teachcopilot_memory_migrate_" + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(dsn, dbname="postgres"))
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
        try:
            yield Store(make_dsn(dsn, dbname=name))
        finally:
            with admin.cursor() as cur:
                cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
                cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    finally:
        admin.close()


def test_024_preserves_existing_conversation_and_old_draft(tmp_path, monkeypatch, migration_store):
    store = migration_store
    full_migrations = store_module.MIGRATIONS
    old_migrations = tmp_path / "old-migrations"
    old_migrations.mkdir()
    # Migrations 025+ arrived after 024; this test exercises exactly the 023 -> 024
    # step, so it applies the chain through 024 only (the full chain was 4 steps).
    through_024 = tmp_path / "through-024"
    through_024.mkdir()
    for path in full_migrations.glob("[0-9][0-9][0-9]_*.sql"):
        if int(path.name[:3]) <= 23:
            shutil.copy(path, old_migrations / path.name)
        if int(path.name[:3]) <= 24:
            shutil.copy(path, through_024 / path.name)
    monkeypatch.setattr(store_module, "MIGRATIONS", old_migrations)
    assert store.migrate() == 23
    assert store.schema_version() == 23

    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card,memory_source_revision)
                       VALUES ('synthetic-migration','Synthetic',4,%s,7)""", (Json({"retained": "card"}),))
        cur.execute("""INSERT INTO teachcopilot.learning_sessions(learner_id,actor_id,topic_id)
                       VALUES ('synthetic-migration','student','synthetic-topic') RETURNING id""")
        session_id = cur.fetchone()["id"]
        cur.execute("""INSERT INTO teachcopilot.memory_sources
                       (id,learner_id,topic_id,learning_session_id,actor_id,author_role,content)
                       VALUES ('synthetic-source','synthetic-migration','synthetic-topic',%s,
                               'student','student','Старая синтетическая реплика')""", (session_id,))
        cur.execute("""INSERT INTO teachcopilot.learning_messages
                       (learning_session_id,learner_id,client_message_id,source_id,content_hash)
                       VALUES (%s,'synthetic-migration','synthetic-turn','synthetic-source','hash')""", (session_id,))
        cur.execute("""INSERT INTO teachcopilot.memory_versions
                       (learner_id,topic_id,base_revision,source_revision,state,context)
                       VALUES ('synthetic-migration','synthetic-topic',1,7,%s,%s) RETURNING id""",
                    (Json({"retained": "state"}), Json({"retained": "context"})))
        version_id = cur.fetchone()["id"]

    def original_rows():
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT learner_id,card,memory_source_revision FROM teachcopilot.learners")
            learners = cur.fetchall()
            cur.execute("SELECT id,content,learning_session_id FROM teachcopilot.memory_sources")
            sources = cur.fetchall()
            cur.execute("SELECT source_id,client_message_id,status FROM teachcopilot.learning_messages")
            messages = cur.fetchall()
            cur.execute("SELECT id,state,context,approved FROM teachcopilot.memory_versions")
            versions = cur.fetchall()
            return learners, sources, messages, versions

    before = original_rows()
    monkeypatch.setattr(store_module, "MIGRATIONS", through_024)
    assert store.migrate() == 1
    assert store.schema_version() == 24
    assert original_rows() == before
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT memory_chat_source_revision FROM teachcopilot.learners WHERE learner_id='synthetic-migration'")
        assert cur.fetchone()["memory_chat_source_revision"] == 0
        cur.execute("SELECT chat_source_revision FROM teachcopilot.memory_versions WHERE id=%s", (version_id,))
        assert cur.fetchone()["chat_source_revision"] is None
    assert store.migrate() == 0

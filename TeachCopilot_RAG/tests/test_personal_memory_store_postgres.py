"""Run only against an explicitly supplied isolated PostgreSQL test database."""
import os
import uuid

import psycopg2
import pytest
from psycopg2.extras import Json

from local_app.store import RevisionConflict, Store
from tests.test_personal_memory_store import OTHER_TOPIC, TOPIC, context_for, state_for


DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")


@pytest.fixture
def database(disposable_store):
    # audit F-017: guarded shared fixture (per-test disposable database).
    # The fixture itself skips when TEACHCOPILOT_TEST_DSN is unset, so importing
    # modules (e.g. test_snapshot_approval_chat.py) no longer ERROR with
    # Store(None) in parse_dsn.
    store = disposable_store
    learners = ["personal-memory-test-" + uuid.uuid4().hex for _ in range(2)]
    with store._connect() as conn, conn.cursor() as cur:
        for learner_id in learners:
            cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
                VALUES (%s,%s,3,%s)""", (learner_id,"Synthetic",Json({"learner_id":learner_id})))
    try:
        yield store, learners
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=ANY(%s)", (learners,))


def draft(store, learner_id):
    store.add_memory_source(learner_id,TOPIC,"Math note","teacher",approved=True)
    original = store.add_memory_source(learner_id,OTHER_TOPIC,"Coding note","teacher",approved=True)
    correction = store.add_memory_source(learner_id,OTHER_TOPIC,"Coding correction","teacher",approved=True,
                                          corrects_source_id=original["source_id"])
    snapshot = store.get_learner_memory(learner_id,TOPIC,include_personal_graph_snapshot=True)
    sources = snapshot["personal_graph_sources"]
    state = state_for(sources,snapshot["source_revision"])
    state["learner_id"] = learner_id
    context = context_for(sources,[original["source_id"],correction["source_id"]])
    version = store.save_memory_version(learner_id,TOPIC,state,base_revision=snapshot["revision"],
                                         source_revision=snapshot["source_revision"],context=context)
    return snapshot,version,original,correction


@pytest.mark.parametrize("full_archive", [False,True])
def test_real_sql_approval_scope_and_cross_topic_correction_freshness(database,full_archive):
    store, learners = database
    learner = learners[0]
    snapshot, version, original, correction = draft(store,learner)
    options = {"full_archive_access":full_archive,
        "expected_archive_fingerprint":snapshot["personal_graph_snapshot_fingerprint"] if full_archive else None}
    approved = store.approve_memory_version(learner,version["id"],"teacher",snapshot["revision"],**options)
    access = approved["context"]["source_access"]
    assert set(access["source_ids"]) == ({s["source_id"] for s in snapshot["personal_graph_sources"]}
        if full_archive else {original["source_id"],correction["source_id"]})
    current = store.get_learner_memory(learner,TOPIC)
    assert not current["versions"][0]["stale"]
    assert "personal_graph_sources" not in current
    assert all(s["topic_id"] == TOPIC for s in current["sources"])
    store.add_memory_source(learner,OTHER_TOPIC,"Later note","teacher",approved=True)
    assert store.get_learner_memory(learner,TOPIC)["versions"][0]["stale"]


def test_real_sql_rejects_foreign_evidence_missing_correction_and_changed_revision(database):
    store, learners = database
    learner, other = learners
    snapshot, version, original, correction = draft(store,learner)
    foreign = store.add_memory_source(other,TOPIC,"Foreign learner","teacher",approved=True)
    context = context_for(snapshot["personal_graph_sources"],[original["source_id"]])
    with pytest.raises(ValueError,match="omit a linked correction"):
        store.save_memory_version(learner,TOPIC,version["state"],base_revision=snapshot["revision"],
                                  source_revision=snapshot["source_revision"],context=context)
    context = context_for(snapshot["personal_graph_sources"],[original["source_id"],correction["source_id"]])
    context["evidence_ids"].append(foreign["source_id"])
    with pytest.raises(ValueError,match="selected source IDs"):
        store.save_memory_version(learner,TOPIC,version["state"],base_revision=snapshot["revision"],
                                  source_revision=snapshot["source_revision"],context=context)
    store.add_memory_source(learner,OTHER_TOPIC,"New source","teacher",approved=True)
    with pytest.raises(RevisionConflict,match="no longer matches"):
        store.approve_memory_version(learner,version["id"],"teacher",snapshot["revision"])


@pytest.mark.parametrize("full_archive", [False,True])
def test_real_sql_active_chat_preserves_grant_until_session_completion(database,full_archive):
    store, learners = database
    learner = learners[0]
    snapshot, version, _, _ = draft(store,learner)
    store.approve_memory_version(learner,version["id"],"teacher",snapshot["revision"],
        full_archive_access=full_archive,
        expected_archive_fingerprint=snapshot["personal_graph_snapshot_fingerprint"] if full_archive else None)
    session = store.create_learning_session(learner,"student",TOPIC)
    store.reserve_chat_message(session["id"],learner,"turn-1","New unfinished turn","student")
    assert not store.get_learner_memory(learner,TOPIC)["versions"][0]["stale"]
    store.complete_chat_message(session["id"],"turn-1",{"answer":"Continue"})
    assert not store.get_learner_memory(learner,TOPIC)["versions"][0]["stale"]
    store.complete_learning_session(session["id"],learner,"student")
    assert store.get_learner_memory(learner,TOPIC)["versions"][0]["stale"]


def test_real_sql_method_constraint_only_accepts_local_next_step(database):
    store, learners = database
    learner = learners[0]
    job = store.enqueue_memory_job(learner,TOPIC,"next_step",method="personal_memory",provider="local")
    assert job["method"] == "personal_memory"
    for kind,provider in (("chat","local"),("memory_update","local"),("next_step","external")):
        with pytest.raises(psycopg2.errors.CheckViolation):
            with store._connect() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO teachcopilot.memory_jobs
                    (learner_id,topic_id,actor_id,kind,method,provider,base_revision,source_revision)
                    VALUES (%s,%s,'teacher',%s,'personal_memory',%s,1,0)""", (learner,TOPIC,kind,provider))

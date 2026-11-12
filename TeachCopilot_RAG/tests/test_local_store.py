"""PostgreSQL integration checks for the isolated local store.

Set TEACHCOPILOT_TEST_DSN to an expendable local database to run these tests.
They use unique IDs and never truncate or drop shared tables.
"""
import os
import uuid
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from psycopg2.extensions import parse_dsn

import pytest

from local_app.store import IndexNotReadyError, RevisionConflict, StaleMaterialSnapshotError, Store
from scripts.manage_local import build_parser, target_dsn_for_database, validate_container_target

DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for local PostgreSQL integration tests")


@pytest.fixture(scope="module")
def store():
    instance = Store(DSN)
    instance.migrate()
    assert instance.ping()
    repo_root=Path(__file__).resolve().parents[2]
    instance.import_repository(repo_root)
    return instance


def sample_card(learner_id):
    path=Path(__file__).resolve().parents[2]/"learner-data"/"learners"/"belov-timur.json"
    card=copy.deepcopy(json.loads(path.read_text(encoding="utf-8-sig")))
    card["learner_id"]=learner_id
    card["pseudonym"]="test"
    return card


def test_card_revision_audit_and_normalized_history(store):
    learner_id = "store-test-" + uuid.uuid4().hex[:12]
    card = sample_card(learner_id)
    assert store.save_card(card, "teacher:test", expected_revision=1)["revision"] == 1
    assert store.get_card(learner_id) == card
    card["learner_model"]["ai_usage"]["reflection_level"] = 0.5
    assert store.save_card(card, "teacher:test", expected_revision=1)["revision"] == 2
    assert store.get_revision(learner_id) == 2
    with pytest.raises(RevisionConflict):
        store.save_card(card, "teacher:test", expected_revision=1)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.card_audit WHERE learner_id=%s", (learner_id,))
        assert cur.fetchone()["n"] == 2
        cur.execute("SELECT count(*) AS n FROM teachcopilot.error_patterns WHERE learner_id=%s", (learner_id,))
        assert cur.fetchone()["n"] == 1
        cur.execute("SELECT count(*) AS n FROM teachcopilot.learner_events WHERE learner_id=%s", (learner_id,))
        assert cur.fetchone()["n"] == 1


def test_atomic_batch_proposal_is_idempotent(store):
    a, b = "store-test-" + uuid.uuid4().hex[:10], "store-test-" + uuid.uuid4().hex[:10]
    ca, cb = sample_card(a), sample_card(b)
    store.save_cards([ca, cb], "teacher:test", {a: 1, b: 1}, "create-" + uuid.uuid4().hex)
    ca["profile"]["pace"] = "slow"
    cb["profile"]["pace"] = "steady"
    proposal_id = "proposal-" + uuid.uuid4().hex
    proposal = store.create_proposal(proposal_id, actor_id="teacher:test", base_revisions={a: 1, b: 1}, proposed_cards={a: ca, b: cb})
    assert len(proposal["cards"]) == 2
    first = store.confirm_proposal(proposal_id, "teacher:test")
    second = store.confirm_proposal(proposal_id, "teacher:test")
    assert first == second
    assert store.get_revision(a) == store.get_revision(b) == 2


def test_material_models_and_filtered_rrf_search(store):
    suffix = uuid.uuid4().hex
    model = "test-model-" + suffix
    math_source = "test/math-" + suffix + ".json"
    chunks = [
        {"id": "chunk-a-" + suffix, "content": "multiplication practice with arrays", "source_file": math_source, "subject_id": "math", "topic_id": "math.g3.numbers", "grade": 3, "embedding": [1.0, 0.0], "raw_snapshot": {"task_id": "a"}},
        {"id": "chunk-b-" + suffix, "content": "python loops and turtle", "source_file": "test/python.json", "subject_id": "programming", "topic_id": "programming.python.loops", "grade": 4, "embedding": [0.0, 1.0]},
    ]
    # Exercise an alternate dimension without relying on a global fixed model size.
    store.upsert_chunks(chunks, embedding_model=model, embedding_dimension=2)
    assert store.embedding_index_status(model)["ready"] is False
    rows = store.hybrid_search("multiplication arrays", [1.0, 0.0], {"subject_id": "math", "source_file": math_source}, 5, model)
    assert rows and rows[0]["id"] == "chunk-a-" + suffix
    assert rows[0]["similarity"] > 0.99
    assert store.material_rows({"source_file": math_source})[0]["raw_snapshot"] == {"task_id": "a"}
    with pytest.raises(ValueError):
        store.upsert_chunks([dict(chunks[0], id="bad-" + suffix, embedding=[1.0, 0.0, 0.0])], embedding_model=model)
    with pytest.raises(Exception):
        store.upsert_chunks([dict(chunks[0],id="orphan-"+suffix,topic_id="math.g99.missing")],embedding_model=model)
    with pytest.raises(IndexNotReadyError):
        store.hybrid_search("multiplication",[1.0,0.0],embedding_model=model)


def test_embedding_refresh_is_snapshot_checked_and_atomic(store):
    suffix=uuid.uuid4().hex
    model="snapshot-model-"+suffix
    source_name="snapshot-refresh-"+suffix+".json"  # unique per run: the rows persist in the test database
    chunks=[
        {"id":"refresh-a-"+suffix,"content":"original A","source_file":source_name,"source_version":"v1","embedding":[1.0,0.0]},
        {"id":"refresh-b-"+suffix,"content":"original B","source_file":source_name,"source_version":"v1","embedding":[0.0,1.0]},
    ]
    store.upsert_chunks(chunks,embedding_model=model,embedding_dimension=2)
    snapshots=store.material_rows({"source_file":source_name})
    for row in snapshots: row["embedding"]=[0.5,0.5]
    stale=dict(snapshots[1],content="changed while embedding")
    with pytest.raises(StaleMaterialSnapshotError):
        store.update_embeddings([snapshots[0],stale],model,2)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding <=> '[1,0]'::vector AS distance FROM teachcopilot.materials WHERE id=%s AND embedding_model=%s",(chunks[0]["id"],model))
        assert cur.fetchone()["distance"] == 0
    assert store.update_embeddings(snapshots,model,2)==2
    changed=dict(chunks[0],content="replacement A",source_version="v2")
    changed.pop("embedding")
    store.upsert_chunks([changed],embedding_model=model)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding,embedding_model,source_version FROM teachcopilot.materials WHERE id=%s",(chunks[0]["id"],))
        row=cur.fetchone()
        assert row["embedding"] is None and row["embedding_model"] is None and row["source_version"] == "v2"


def test_removed_material_membership_is_deactivated_and_history_is_append_only(store,tmp_path):
    learner_id="store-test-"+uuid.uuid4().hex[:12]
    card=sample_card(learner_id)
    first_journal=dict(card["mvp"]["journal"][0],date="2026-01-01",note="first")
    first_points=dict(card["mvp"]["points_ledger"][0],date="2026-01-01",points=2)
    card["mvp"]["journal"]=[first_journal]
    card["mvp"]["points_ledger"]=[first_points]
    store.save_card(card,"history-test")
    card["mvp"]["journal"].append(dict(first_journal,date="2026-01-02",note="second"))
    card["mvp"]["points_ledger"].append(dict(first_points,date="2026-01-02",points=3))
    store.save_card(card,"history-test",expected_revision=1)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.learner_events WHERE learner_id=%s AND event_type='journal'",(learner_id,))
        assert cur.fetchone()["n"] == 2
        cur.execute("SELECT count(*) AS n FROM teachcopilot.points_ledger WHERE learner_id=%s",(learner_id,))
        assert cur.fetchone()["n"] == 2
    card["mvp"]["journal"][0]["note"]="corrected first note"
    card["mvp"]["points_ledger"][0]["points"]=4
    store.save_card(card,"history-test",expected_revision=2)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS total,count(*) FILTER (WHERE active) AS active,count(*) FILTER (WHERE supersedes IS NOT NULL) AS corrections FROM teachcopilot.learner_events WHERE learner_id=%s AND event_type='journal'",(learner_id,))
        assert dict(cur.fetchone()) == {"total":3,"active":2,"corrections":1}
        cur.execute("SELECT count(*) AS total,count(*) FILTER (WHERE active) AS active,count(*) FILTER (WHERE supersedes IS NOT NULL) AS corrections,sum(points) FILTER (WHERE active) AS current_points FROM teachcopilot.points_ledger WHERE learner_id=%s",(learner_id,))
        assert dict(cur.fetchone()) == {"total":3,"active":2,"corrections":1,"current_points":7}
    card["mvp"]["journal"]=card["mvp"]["journal"][:1]
    card["mvp"]["points_ledger"]=card["mvp"]["points_ledger"][:1]
    store.save_card(card,"history-test",expected_revision=3)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS total,count(*) FILTER (WHERE active) AS active FROM teachcopilot.learner_events WHERE learner_id=%s AND event_type='journal'",(learner_id,))
        assert dict(cur.fetchone()) == {"total":3,"active":1}
        cur.execute("SELECT count(*) AS total,count(*) FILTER (WHERE active) AS active,sum(points) FILTER (WHERE active) AS current_points FROM teachcopilot.points_ledger WHERE learner_id=%s",(learner_id,))
        assert dict(cur.fetchone()) == {"total":3,"active":1,"current_points":4}

    root=Path(tmp_path); (root/"knowledge-data").mkdir()
    membership_name="membership-"+uuid.uuid4().hex+".json"  # unique per run: the rows persist in the test database
    source=root/"knowledge-data"/membership_name
    tasks=[{"task_id":"member-a-"+uuid.uuid4().hex,"content":"A"},{"task_id":"member-b-"+uuid.uuid4().hex,"content":"B"}]
    source.write_text(json.dumps({"tasks":tasks}),encoding="utf-8")
    store.import_materials(root)
    source.write_text(json.dumps({"tasks":tasks[:1]}),encoding="utf-8")
    store.import_materials(root)
    rows=store.material_rows({"source_file":"knowledge-data/"+membership_name})
    assert [row["id"] for row in rows] == [tasks[0]["task_id"]]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.materials WHERE source_file=%s AND NOT active",("knowledge-data/"+membership_name,))
        assert cur.fetchone()["n"] == 1


def test_auth_sessions_and_explicit_assignment(store):
    suffix = uuid.uuid4().hex
    learner_id = "store-test-" + suffix[:12]
    store.save_card(sample_card(learner_id), "test")
    account = store.create_account(f"{suffix}@example.invalid", "Test Teacher", "not-a-real-password-hash", "teacher")
    assert not store.authorize_learner_access(account["id"], learner_id)
    store.assign_learner(account["id"], learner_id)
    assert store.authorize_learner_access(account["id"], learner_id)
    student = store.create_student_account(f"student-{suffix}@example.invalid", "Test Student", "pbkdf2-hash", learner_id, account["id"])
    assert student["role"] == "student"
    assert store.authorize_learner_access(student["id"], learner_id)
    token_hash = uuid.uuid4().hex
    store.create_session(account["id"], token_hash, datetime.now(timezone.utc) + timedelta(minutes=3))
    assert store.get_session(token_hash)["account_id"] == account["id"]
    assert store.revoke_session(token_hash)
    assert store.get_session(token_hash) is None
    new_learner="store-test-"+uuid.uuid4().hex[:12]
    created=store.create_assigned_learner(sample_card(new_learner),account["id"])
    assert created["learner_id"] == new_learner
    assert store.authorize_learner_access(account["id"],new_learner)
    with pytest.raises(ValueError):
        store.create_assigned_learner(sample_card(new_learner),account["id"])


def test_import_snapshots_do_not_replace_live_card_edits(store, tmp_path):
    import json
    from pathlib import Path

    root=Path(tmp_path)
    (root/"learner-data"/"learners").mkdir(parents=True)
    (root/"learner-data"/"catalog").mkdir(parents=True)
    (root/"knowledge-data").mkdir()
    learner_id="snapshot-test-"+uuid.uuid4().hex[:10]
    card=sample_card(learner_id)
    (root/"learner-data"/"learners"/(learner_id+".json")).write_text(json.dumps(card),encoding="utf-8")
    catalog_path=Path(__file__).resolve().parents[2]/"learner-data"/"catalog"/"math_g3_g4.json"
    (root/"learner-data"/"catalog"/"math.json").write_text(catalog_path.read_text(encoding="utf-8-sig"),encoding="utf-8")
    programming_catalog=Path(__file__).resolve().parents[2]/"learner-data"/"catalog"/"programming.json"
    (root/"learner-data"/"catalog"/"programming.json").write_text(programming_catalog.read_text(encoding="utf-8-sig"),encoding="utf-8")
    task={"task_id":"snapshot-task-"+uuid.uuid4().hex,"topic_id":"math.g3.numbers","grade":"3","content":"Solve 2 + 2","answer":"4"}
    (root/"knowledge-data"/"tasks.json").write_text(json.dumps({"tasks":[task]}),encoding="utf-8")
    first=store.import_repository(root)
    assert first["cards_imported"] == 1
    edited=store.get_card(learner_id); edited["profile"]["pace"]="edited"
    store.save_card(edited,"teacher:edit",expected_revision=1)
    second=store.import_repository(root)
    assert second["cards_skipped"] == 1 and second["card_conflicts"] == [learner_id]
    assert store.get_card(learner_id)["profile"]["pace"] == "edited"
    rows=store.material_rows({"source_file":"knowledge-data/tasks.json"})
    assert rows[0]["content"] == task["content"]
    assert rows[0]["raw_snapshot"]["answer"] == task["answer"]


def test_repository_import_preserves_expected_source_counts(store):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    first = store.import_repository(repo_root)
    second = store.import_repository(repo_root)
    assert first["cards"] == second["cards"] == 17
    assert first["tasks"] == second["tasks"] == 147
    assert second["cards_imported"] == 0
    assert second["cards_skipped"] == 17
    assert store.import_materials(repo_root) == {"materials": 147}
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.competencies")
        assert cur.fetchone()["n"] > 0
        cur.execute("SELECT count(*) AS n FROM teachcopilot.ai_observations")
        assert cur.fetchone()["n"] > 0
        cur.execute("SELECT content,raw_snapshot FROM teachcopilot.materials WHERE id='div-408-4'")
        row=cur.fetchone()
        assert row and row["raw_snapshot"]["answer"] not in row["content"]


def test_backup_restore_dsn_must_match_local_container_port():
    inspected=json.dumps([{"NetworkSettings":{"Ports":{"5432/tcp":[{"HostIp":"127.0.0.1","HostPort":"5434"}]}}}])
    runner=lambda *args,**kwargs: SimpleNamespace(stdout=inspected)
    assert validate_container_target("postgresql://teachcopilot:secret@127.0.0.1:5434/localdb","teachcopilot_local_db",runner)=="localdb"
    with pytest.raises(ValueError):
        validate_container_target("postgresql://teachcopilot:secret@127.0.0.1:5435/otherdb","teachcopilot_local_db",runner)
    with pytest.raises(ValueError):
        validate_container_target("postgresql://teachcopilot:secret@db.example.invalid:5434/otherdb","teachcopilot_local_db",runner)
    with pytest.raises(ValueError):
        validate_container_target("postgresql://teacher:secret@127.0.0.1:5434/otherdb","teachcopilot_local_db",runner)


def test_restore_target_database_uses_only_configured_dsn_with_new_dbname():
    base="postgresql://teachcopilot:secret@127.0.0.1:5434/currentdb?sslmode=disable"
    derived=target_dsn_for_database("restore_target",base)
    before=parse_dsn(base); after=parse_dsn(derived)
    assert after["dbname"]=="restore_target"
    before.pop("dbname"); after.pop("dbname")
    assert after==before
    args=build_parser().parse_args(["restore","--in","backup.dump","--target-db","restore_target","--yes"])
    assert args.target_db=="restore_target" and args.target_dsn is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["restore","--in","backup.dump","--target-db","restore_target","--target-dsn",base])


def test_curriculum_prefix_is_a_database_filter(store):
    suffix=uuid.uuid4().hex
    model='prefix-model-'+suffix
    source='test/prefix-'+suffix+'.json'
    store.import_repository(__import__('pathlib').Path(__file__).resolve().parents[2])
    store.upsert_chunks([
        {'id':'prefix-python-'+suffix,'content':'python loop','source_file':source,
         'subject_id':'programming','topic_id':'programming.python.loops','embedding':[1.0,0.0]},
        {'id':'prefix-scratch-'+suffix,'content':'scratch loop','source_file':source,
         'subject_id':'programming','topic_id':'programming.scratch.loops','embedding':[1.0,0.0]},
    ],embedding_model=model,embedding_dimension=2)
    rows=store.hybrid_search('loop',[1.0,0.0],
        {'subject_id':'programming','source_file':source,'topic_prefix':'programming.python.'},5,model)
    assert len(rows)==1 and rows[0]['id']=='prefix-python-'+suffix

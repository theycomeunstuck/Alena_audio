"""PostgreSQL memory contracts, using isolated rows in an explicit test database."""
import os
import json
from pathlib import Path
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg2.extras import Json

from local_app.memory_store import MemoryIdempotencyConflict
from local_app.store import RevisionConflict, Store


DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for PostgreSQL integration tests")


@pytest.fixture
def memory_store():
    store = Store(DSN)
    store.migrate()
    learner_ids = ["memory-test-" + uuid.uuid4().hex for _ in range(2)]
    with store._connect() as conn, conn.cursor() as cur:
        for learner_id in learner_ids:
            cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
                VALUES (%s,%s,3,%s)""", (learner_id,"Synthetic Learner",Json({"learner_id":learner_id,"profile":{"full_name":"Synthetic Private Name","school_name":"Test School 190"}})))
    yield store, learner_ids[0], learner_ids[1]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=ANY(%s)", (learner_ids,))


def approved_source(store, learner_id, content="Ученик решил пример; помощь неизвестна."):
    return store.add_memory_source(learner_id,"math.synthetic",content,"teacher:test",approved=True)


def source_context(source):
    return {"method":"tail","topic_id":"math.synthetic","evidence_ids":[source["source_id"]],
            "sources":[{"source_id":source["source_id"],"text":source["content"]}],"text":source["content"]}


def conversation_card(store, learner_id):
    """A valid synthetic card, independent of real learner catalog imports."""
    card = json.loads((Path(__file__).resolve().parents[2] / "learner-data/learners/_TEMPLATE.json").read_text(encoding="utf-8"))
    card.update(learner_id=learner_id, pseudonym="Synthetic", interests=[])
    card.pop("legal_name", None)
    card["rag_context"]["current_goal"]["topic_id"] = "math.synthetic"
    card["rag_context"]["current_topics"] = ["math.synthetic"]
    card["rag_context"]["avoid"] = []
    card["knowledge"] = []
    card["error_patterns"] = []
    card["mvp"]["points_ledger"] = []
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO teachcopilot.catalog_topics(topic_id,subject_id,grade,title,payload)
            VALUES ('math.synthetic','math',3,'Synthetic',%s) ON CONFLICT(topic_id) DO NOTHING""",
            (Json({"id":"math.synthetic","grade":3,"title_ru":"Synthetic","deprecated":False}),))
        cur.execute("UPDATE teachcopilot.learners SET card=%s WHERE learner_id=%s", (Json(card), learner_id))
    return card


def student_interest_version(store, learner_id, text="Я люблю Minecraft.", operation="add", interest="Minecraft"):
    from local_app.conversation_memory import validate_interest_candidates
    session = store.create_learning_session(learner_id,"student","math.synthetic")
    source = store.reserve_chat_message(session["id"],learner_id,"preference",text,"student")
    store.complete_chat_message(session["id"],"preference",{"answer":"Расскажи подробнее."})
    store.complete_learning_session(session["id"],learner_id,"student")
    candidates = validate_interest_candidates(json.dumps({"candidates":[{
        "operation":operation,"interest":interest,"source_id":source["source_id"],"quote":text}]}),
        store.list_learning_messages(session["id"]))
    version = store.save_memory_version(learner_id,"math.synthetic",{"method":"tail"},
        base_revision=store.get_revision(learner_id),context={"interest_candidates":candidates})
    return version, candidates


def test_only_explicit_interest_choices_change_shared_card_and_repeat_is_idempotent(memory_store):
    store, learner, _ = memory_store
    conversation_card(store,learner)
    version, candidates = student_interest_version(store,learner)
    assert store.get_card(learner)["interests"] == []
    selected = [candidates[0]["candidate_id"]]
    accepted = store.approve_memory_version(learner,version["id"],"teacher",1,interest_candidate_ids=selected)
    assert accepted["approved"]
    assert store.get_card(learner)["interests"] == ["Minecraft"]
    assert store.get_revision(learner) == 2
    repeated = store.approve_memory_version(learner,version["id"],"teacher",1,interest_candidate_ids=selected)
    assert repeated["context"]["interest_review"]["candidate_ids"] == selected
    assert store.get_revision(learner) == 2
    with pytest.raises(MemoryIdempotencyConflict):
        store.approve_memory_version(learner,version["id"],"teacher",2,interest_candidate_ids=[])


def test_academic_approval_does_not_silently_accept_interest_or_allow_late_selection(memory_store):
    store, learner, _ = memory_store
    conversation_card(store,learner)
    version, candidates = student_interest_version(store,learner)
    store.approve_memory_version(learner,version["id"],"teacher",1)
    assert store.get_card(learner)["interests"] == []
    assert store.get_revision(learner) == 1
    with pytest.raises(MemoryIdempotencyConflict):
        store.approve_memory_version(learner,version["id"],"teacher",1,
            interest_candidate_ids=[candidates[0]["candidate_id"]])


def test_interest_confirmation_revision_conflict_keeps_card_and_draft_unchanged(memory_store):
    store, learner, _ = memory_store
    card = conversation_card(store,learner)
    version, candidates = student_interest_version(store,learner)
    card["interests"] = ["космос"]
    store.save_card(card,"teacher",1)
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(learner,version["id"],"teacher",2,
            interest_candidate_ids=[candidates[0]["candidate_id"]])
    assert store.get_card(learner)["interests"] == ["космос"]
    assert not store.get_learner_memory(learner)["versions"][0]["approved"]


def test_interest_quote_by_teacher_cannot_become_a_student_preference(memory_store):
    store, learner, _ = memory_store
    conversation_card(store,learner)
    session = store.create_learning_session(learner,"teacher","math.synthetic")
    source = store.reserve_chat_message(session["id"],learner,"teacher-turn","Я люблю Minecraft.","teacher",author_role="teacher")
    store.complete_chat_message(session["id"],"teacher-turn",{"answer":"Понятно."})
    store.complete_learning_session(session["id"],learner,"teacher")
    version = store.save_memory_version(learner,"math.synthetic",{"method":"tail"},base_revision=1,
        context={"interest_candidates":[{"candidate_id":"bad-author","operation":"add","interest":"Minecraft",
            "source_id":source["source_id"],"quote":"Я люблю Minecraft."}]})
    with pytest.raises(ValueError,match="student"):
        store.approve_memory_version(learner,version["id"],"teacher",1,interest_candidate_ids=["bad-author"])
    assert store.get_card(learner)["interests"] == []
    assert not store.get_learner_memory(learner)["versions"][0]["approved"]


def test_interest_avoid_limit_is_clear_and_atomic(memory_store):
    store, learner, _ = memory_store
    card = conversation_card(store,learner)
    card["rag_context"]["avoid"] = ["Старое ограничение 1","Старое ограничение 2","Старое ограничение 3"]
    store.save_card(card,"teacher",1)
    version, candidates = student_interest_version(store,learner,"Не хочу задачи про Minecraft.","avoid")
    with pytest.raises(ValueError,match="Сначала пересмотрите"):
        store.approve_memory_version(learner,version["id"],"teacher",2,
            interest_candidate_ids=[candidates[0]["candidate_id"]])
    assert store.get_card(learner)["rag_context"]["avoid"] == card["rag_context"]["avoid"]
    assert store.get_revision(learner) == 2
    assert not store.get_learner_memory(learner)["versions"][0]["approved"]


def test_orphan_chat_recovery_releases_queue_without_replay(memory_store):
    store, learner, _ = memory_store
    session = store.create_learning_session(learner,"owner","math.synthetic")
    store.reserve_chat_message(session["id"],learner,"orphan","Я начал работу.","owner")
    job = store.enqueue_memory_job(learner,"math.synthetic","chat",{},"owner",
        idempotency_key=f"chat:{session['id']}:orphan")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_messages SET created_at=now()-interval '20 minutes' WHERE learning_session_id=%s",(session["id"],))
    assert store.recover_chat_messages() == 1
    assert store.get_memory_job(job["id"])["status"] == "failed"
    duplicate = store.reserve_chat_message(session["id"],learner,"orphan","Я начал работу.","owner")
    assert not duplicate["created"] and duplicate["status"] == "unknown_outcome"
    assert store.claim_interactive_chat_job(job["id"],"new-owner") is None
    assert store.claim_memory_job("writer") is None


def test_legacy_orphan_admission_expires_and_frees_cap(memory_store):
    store, learner, _ = memory_store
    job = store.enqueue_memory_job(learner,"math.synthetic","chat",{"legacy_chat":True},"owner")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.memory_jobs SET created_at=now()-interval '20 minutes' WHERE id=%s",(job["id"],))
    store.recover_memory_jobs()
    assert store.get_memory_job(job["id"])["status"] == "failed"


def test_queued_interactive_chat_has_priority_over_background_writer(memory_store):
    store, learner, other = memory_store
    background = store.enqueue_memory_job(learner,"math.synthetic","memory_update",{},"teacher")
    chat = store.enqueue_memory_job(other,"math.synthetic","chat",{"legacy_chat":True},"owner")
    assert store.claim_memory_job("background-worker") is None
    active = store.claim_interactive_chat_job(chat["id"],"http-request")
    assert active["id"] == chat["id"]
    assert store.claim_memory_job("background-worker") is None
    store.finish_memory_job(chat["id"],result={},worker_id="http-request")
    assert store.claim_memory_job("background-worker")["id"] == background["id"]


def test_finishing_job_does_not_hold_job_row_while_waiting_for_learner(memory_store):
    store, learner, _ = memory_store
    job = store.enqueue_memory_job(learner,"math.synthetic","next_step")
    store.claim_memory_job("finisher")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with store._connect() as blocker, blocker.cursor() as cur:
            cur.execute("SELECT learner_id FROM teachcopilot.learners WHERE learner_id=%s FOR UPDATE",(learner,))
            future = executor.submit(store.finish_memory_job,job["id"],result={},worker_id="finisher")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with store._connect() as observer, observer.cursor() as check:
                    check.execute("""SELECT count(*) AS n FROM pg_stat_activity WHERE datname=current_database()
                        AND wait_event_type='Lock' AND query LIKE 'SELECT revision,memory_source_revision%%'""")
                    if check.fetchone()["n"]:
                        break
                time.sleep(0.01)
            else:
                raise AssertionError("finisher did not wait for learner lock")
            # Recovery may need this row under the queue advisory lock. Holding
            # it while waiting for learner would create a three-way deadlock.
            with store._connect() as recovery, recovery.cursor() as check:
                check.execute("SELECT id FROM teachcopilot.memory_jobs WHERE id=%s FOR UPDATE NOWAIT",(job["id"],))
                assert check.fetchone()
        assert future.result(timeout=5)["status"] == "completed"


def test_duplicate_message_concurrently_reserves_only_one_request(memory_store):
    store, learner, _ = memory_store
    session = store.create_learning_session(learner,"actor:student","math.synthetic","session-one")
    assert store.create_learning_session(learner,"actor:student","math.synthetic","session-one")["id"] == session["id"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.reserve_chat_message(session["id"],learner,"turn-one","Я получил ответ 12.","actor:student"),range(2)))
    assert sorted(result["created"] for result in results) == [False,True]
    assert len(store.list_learning_messages(session["id"])) == 1
    with pytest.raises(MemoryIdempotencyConflict):
        store.reserve_chat_message(session["id"],learner,"turn-one","Другой ответ.","actor:student")
    answer = {"choices":[{"message":{"role":"assistant","content":"Покажи ход решения."}}]}
    store.complete_chat_message(session["id"],"turn-one",answer)
    store.complete_chat_message(session["id"],"turn-one",answer)
    duplicate = store.reserve_chat_message(session["id"],learner,"turn-one","Я получил ответ 12.","actor:student")
    assert not duplicate["created"] and duplicate["status"] == "completed" and duplicate["response"] == answer
    messages = store.list_learning_messages(session["id"])
    assert [row["role"] for row in messages] == ["user","assistant"]
    assert messages[-1]["content"] == "Покажи ход решения."


def test_unknown_message_outcome_is_retained_without_retry(memory_store):
    store, learner, _ = memory_store
    session = store.create_learning_session(learner,"actor:student","math.synthetic")
    store.reserve_chat_message(session["id"],learner,"turn","Ответ 12.","actor:student")
    store.fail_chat_message(session["id"],"turn","network timeout",unknown_outcome=True)
    duplicate = store.reserve_chat_message(session["id"],learner,"turn","Ответ 12.","actor:student")
    assert not duplicate["created"] and duplicate["status"] == "unknown_outcome"
    with pytest.raises(ValueError):
        store.complete_chat_message(session["id"],"turn","Новая платная попытка")


def test_session_owner_and_learner_boundaries(memory_store):
    store, learner, other = memory_store
    session = store.create_learning_session(learner,"owner","math.synthetic")
    with pytest.raises(PermissionError):
        store.reserve_chat_message(session["id"],learner,"turn","text","other-actor")
    with pytest.raises(KeyError):
        store.reserve_chat_message(session["id"],other,"turn","text","owner")
    assert store.get_learning_session(session["id"],other) is None
    with pytest.raises(KeyError):
        store.complete_learning_session(session["id"],other,"owner")


def test_completed_student_sources_are_visible_assistant_sources_never_evidence(memory_store):
    store, learner, _ = memory_store
    session = store.create_learning_session(learner,"owner","math.synthetic")
    reserved = store.reserve_chat_message(session["id"],learner,"turn","Я ещё не закончил решение.","owner")
    assert not store.get_learner_memory(learner)["sources"]
    with pytest.raises(ValueError):
        store.complete_learning_session(session["id"],learner,"owner")
    store.complete_chat_message(session["id"],"turn",{"answer":"Отлично, ты всё решил!"})
    completed = store.complete_learning_session(session["id"],learner,"owner")
    assert completed["job"]["method"] == "tail"
    assert store.complete_learning_session(session["id"],learner,"owner")["job_id"] == completed["job_id"]
    snapshot = store.get_learner_memory(learner)
    assert len(snapshot["sources"]) == 1
    assert snapshot["sources"][0]["source_id"] == reserved["source_id"]
    assert "не закончил" in snapshot["sources"][0]["text"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM teachcopilot.memory_sources WHERE learning_session_id=%s AND author_role='assistant'", (session["id"],))
        assistant_id = cur.fetchone()["id"]
    with pytest.raises(ValueError,match="assistant"):
        store.save_memory_version(learner,"math.synthetic",{"method":"tail"},base_revision=1,context={"evidence_turn_ids":[assistant_id]})


def test_teacher_chat_is_hidden_while_active_and_maps_to_user(memory_store):
    store, learner, _ = memory_store
    session = store.create_learning_session(learner,"teacher","math.synthetic")
    store.reserve_chat_message(session["id"],learner,"turn","Помощь была неизвестна.","teacher",author_role="teacher")
    assert not store.get_learner_memory(learner)["sources"]
    assert store.list_learning_messages(session["id"])[0]["role"] == "user"
    store.fail_chat_message(session["id"],"turn",unknown_outcome=False)
    store.complete_learning_session(session["id"],learner,"teacher")
    assert store.get_learner_memory(learner)["sources"][0]["author_role"] == "teacher"


def test_version_confirmation_preserves_unknown_conditions_and_requires_current_revision(memory_store):
    store, learner, _ = memory_store
    source = approved_source(store,learner)
    state = {"method":"tail","topic_id":"math.synthetic","unknowns":["help_unknown"],"records":[]}
    version = store.save_memory_version(learner,"math.synthetic",state,base_revision=1,context=source_context(source))
    assert not version["approved"]
    assert store.approve_memory_version(learner,version["id"],"teacher",1)["approved"]
    assert store.approve_memory_version(learner,version["id"],"teacher",1)["state"]["unknowns"] == ["help_unknown"]
    pending = store.save_memory_version(learner,"math.synthetic",state,base_revision=1,context=source_context(source))
    approved_source(store,learner,"Позже ученик выполнил новую самостоятельную попытку.")
    assert store.approve_memory_version(learner,version["id"],"teacher",1)["approved"]
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(learner,pending["id"],"teacher",1)
    assert store.get_learner_memory(learner)["versions"][0]["stale"]


def test_cross_learner_sources_rejected_in_actions_records_and_support_evidence(memory_store):
    store, learner, other = memory_store
    foreign = approved_source(store,other)
    for value in ({"evidence_turn_ids":[foreign["id"]]}, {"support_evidence":{"source_turn_id":foreign["id"],"quote":"text"}}, {"sources":{"x":{"turn_id":foreign["id"]}}}):
        with pytest.raises(ValueError,match="sources"):
            store.save_memory_version(learner,"math.synthetic",value,base_revision=1)
    assert not store.get_learner_memory(learner)["versions"]


def test_job_finish_commits_version_and_action_together_and_keeps_stale_failure(memory_store):
    store, learner, _ = memory_store
    source = approved_source(store,learner)
    job = store.enqueue_memory_job(learner,"math.synthetic","next_step",actor_id="teacher")
    claimed = store.claim_memory_job("worker:test")
    assert claimed["id"] == job["id"]
    action = {"action":"Проверить самостоятельное решение.","evidence_turn_ids":[source["id"]],"review_required":True}
    result = store.finish_memory_job(job["id"],worker_id="worker:test",memory_state={"method":"tail"},next_step=action,context=source_context(source))
    assert result["status"] == "completed" and result["result"]["next_step_id"]
    step = store.get_next_step(result["result"]["next_step_id"],learner)
    assert step["memory_version_id"] == result["result"]["memory_version_id"]
    assert store.get_next_step(step["id"],"wrong-learner") is None
    another = store.enqueue_memory_job(learner,"math.synthetic","next_step",actor_id="teacher")
    assert store.claim_memory_job("worker:test")["id"] == another["id"]
    approved_source(store,learner,"Новая запись после начала генерации.")
    stale = store.finish_memory_job(another["id"],worker_id="worker:test",memory_state={"method":"tail"},next_step=action)
    assert stale["status"] == "failed" and "stale_revision" in stale["error"]
    assert len(store.get_learner_memory(learner)["versions"]) == 1


def test_job_output_reference_failure_rolls_back_both_outputs(memory_store):
    store, learner, other = memory_store
    foreign = approved_source(store,other)
    job = store.enqueue_memory_job(learner,"math.synthetic","next_step",actor_id="teacher")
    assert store.claim_memory_job("worker:test")["id"] == job["id"]
    with pytest.raises(ValueError):
        store.finish_memory_job(job["id"],worker_id="worker:test",memory_state={"method":"tail"},next_step={"evidence_turn_ids":[foreign["id"]]})
    assert not store.get_learner_memory(learner)["versions"]
    store.finish_memory_job(job["id"],worker_id="worker:test",error="invalid generated references")


def test_proposal_sources_group_authorship_and_transaction_rollback(memory_store):
    store, learner, other = memory_store
    metadata = {"memory_sources":{learner:{"topic_id":"math.synthetic","text":"Группа решила задачу.","group_note":True}}}
    with store._connect() as conn, conn.cursor() as cur:
        jobs = store._confirm_memory_proposal(cur,metadata,"teacher",{learner:1})
    assert jobs[0]["method"] == "tail"
    assert "авторство отдельных результатов не установлено" in store.get_learner_memory(learner)["sources"][0]["text"]
    with pytest.raises(ValueError):
        with store._connect() as conn, conn.cursor() as cur:
            store._confirm_memory_proposal(cur,{"memory_sources":{other:{"topic_id":"math.synthetic","text":"not in proposal"}}},"teacher",{learner:1})
    assert not store.get_learner_memory(other)["sources"]
    assert "Synthetic Private Name" in store.learner_known_identifiers(learner)
    assert "Test School 190" in store.learner_known_identifiers(learner)


def test_multiple_confirmed_notes_keep_source_order_and_correction_links(memory_store):
    store, learner, _ = memory_store
    notes = [{"topic_id":"math.synthetic","text":"Записано: ученик решил без помощи."},
             {"topic_id":"math.synthetic","text":"Исправление записи: помощь не фиксировалась; самостоятельность неизвестна."}]
    with store._connect() as conn, conn.cursor() as cur:
        store._confirm_memory_proposal(cur,{"memory_sources":{learner:notes}},"teacher",{learner:1})
    sources = store.get_learner_memory(learner)["sources"]
    assert [row["text"] for row in sources] == [row["text"] for row in notes]
    state = {"method":"structured","records":[
        {"record_id":"past-observation","evidence_turn_ids":[sources[0]["source_id"]]},
        {"record_id":"teacher-correction","corrects_record_id":"past-observation","evidence_turn_ids":[sources[1]["source_id"]],"support_status":"unknown"}]}
    version = store.save_memory_version(learner,"math.synthetic",state,base_revision=1)
    assert version["state"] == state
    assert len(store.get_learner_memory(learner)["sources"]) == 2

"""Explicit teacher corrections preserve literal sources in disposable PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from fastapi.testclient import TestClient
import psycopg2
from psycopg2.extras import Json
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_store import MemoryIdempotencyConflict
from local_app.store import RevisionConflict
from tests.test_memory_api_flow import InspectableRag, flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
OTHER_TOPIC = "math.g4.numbers.division_by_1_2_digit"
CORRECTION = "Исправляю запись: решение показал преподаватель. Самостоятельность неизвестна."


def _source(store, learner=LEARNER, topic=TOPIC, **kwargs):
    return store.add_memory_source(learner,topic,"Ученик решил самостоятельно.","teacher",approved=True,**kwargs)


def _request(store, source, *, key="correction-one", actor="teacher", text=CORRECTION, **overrides):
    options = dict(expected_revision=store.get_revision(LEARNER),
                   expected_source_revision=store.get_learner_source_revision(LEARNER),
                   client_correction_id=key)
    options.update(overrides)
    return store.correct_memory_source(LEARNER,source["source_id"],text,actor,**options)


@pytest.fixture
def correction_api(flow_store):
    store = flow_store
    password = "Synthetic source correction password 123"
    secret = hash_password(password)
    teacher = store.create_account("correction-teacher@synthetic.test","Teacher",secret,"teacher")
    store.assign_learner(teacher["id"],LEARNER)
    student = store.create_student_account("correction-student@synthetic.test","Student",secret,LEARNER,teacher["id"])
    outsider = store.create_account("correction-outsider@synthetic.test","Other teacher",secret,"teacher")
    browser = TestClient(create_app(settings=Settings(),store=store))

    def auth(account):
        response = browser.post("/auth/login",json={"email":account["email"],"password":password})
        assert response.status_code == 200
        return {"Authorization":"Bearer " + response.json()["access_token"]}

    return store,browser,auth(teacher),auth(student),auth(outsider)


def test_api_appends_linked_teacher_source_and_replays_original_job(correction_api):
    store,browser,teacher,_,_ = correction_api
    original = _source(store)
    card_before = deepcopy(store.get_card(LEARNER))
    source_revision = store.get_learner_source_revision(LEARNER)
    payload = {"text":CORRECTION,"expected_revision":store.get_revision(LEARNER),
               "expected_source_revision":source_revision,"client_correction_id":"api-one"}
    url = f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections"
    response = browser.post(url,headers=teacher,json=payload)
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["source"]["corrects_source_id"] == original["source_id"]
    assert result["source"]["author_role"] == "teacher" and result["source"]["approved"] is True
    assert result["topic_id"] == TOPIC
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    assert [row["text"] for row in snapshot["sources"]] == [original["content"],CORRECTION]
    assert snapshot["source_revision"] == source_revision+1
    assert store.get_card(LEARNER) == card_before
    assert browser.post(url,headers=teacher,json=payload).json() == result
    # A replay remains safe after either revision changed, including a completed job.
    store.cancel_memory_job(result["job_id"],"synthetic completed request")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learners SET revision=revision+1 WHERE learner_id=%s",(LEARNER,))
    _source(store)
    assert browser.post(url,headers=teacher,json=payload).json() == result
    job = store.get_memory_job(result["job_id"])
    assert job["payload"] == {"source_id":result["source"]["source_id"]}
    assert job["source_revision"] == source_revision+1


def test_api_rejects_student_foreign_learner_and_unknown_target(correction_api):
    store,browser,teacher,student,outsider = correction_api
    original = _source(store)
    payload = {"text":CORRECTION,"expected_revision":store.get_revision(LEARNER),
               "expected_source_revision":store.get_learner_source_revision(LEARNER),"client_correction_id":"roles"}
    url = f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections"
    assert browser.post(url,headers=student,json=payload).status_code == 403
    assert browser.post(url,headers=outsider,json=payload).status_code == 403
    assert browser.post(url,json=payload).status_code == 401
    assert browser.post(f"/learners/{LEARNER}/memory/sources/unknown/corrections",headers=teacher,json=payload).status_code == 404
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES ('other-correction-learner','Other',4,%s)",
                    (Json({"learner_id":"other-correction-learner"}),))
    foreign = _source(store,learner="other-correction-learner")
    foreign_url = f"/learners/{LEARNER}/memory/sources/{foreign['source_id']}/corrections"
    assert browser.post(foreign_url,headers=teacher,json=payload).status_code == 404
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 1


@pytest.mark.parametrize("change",[
    {"text":""},{"text":"   "},{"text":"x"*2001},
    {"client_correction_id":""},{"client_correction_id":" "},{"client_correction_id":"x"*181},
    {"expected_revision":0},{"expected_source_revision":-1},{"expected_source_revision":True},
    {"expected_source_revision":"1"},{"corrects_source_id":"injected"},{"topic_id":OTHER_TOPIC},
])
def test_api_validates_payload_without_adding_sources(correction_api,change):
    store,browser,teacher,_,_ = correction_api
    original = _source(store)
    payload = {"text":CORRECTION,"expected_revision":store.get_revision(LEARNER),
               "expected_source_revision":store.get_learner_source_revision(LEARNER),"client_correction_id":"invalid"}
    payload.update(change)
    url = f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections"
    assert browser.post(url,headers=teacher,json=payload).status_code == 422
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 1


def test_api_requires_both_current_revisions(correction_api):
    store,browser,teacher,_,_ = correction_api
    original = _source(store)
    payload = {"text":CORRECTION,"expected_revision":store.get_revision(LEARNER),
               "expected_source_revision":store.get_learner_source_revision(LEARNER),"client_correction_id":"stale"}
    url = f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections"
    missing = {key:value for key,value in payload.items() if key != "expected_source_revision"}
    assert browser.post(url,headers=teacher,json=missing).status_code == 422
    assert browser.post(url,headers=teacher,json={**payload,"expected_revision":payload["expected_revision"]+1}).status_code == 409
    _source(store)
    assert browser.post(url,headers=teacher,json=payload).status_code == 409
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 2


def test_reused_client_key_rejects_changed_text_target_or_actor(flow_store):
    store = flow_store
    original,other = _source(store),_source(store)
    first = _request(store,original)
    for target,actor,text in ((original,"teacher","Changed"),(other,"teacher",CORRECTION),(original,"another teacher",CORRECTION)):
        with pytest.raises(MemoryIdempotencyConflict):
            _request(store,target,actor=actor,text=text,expected_revision=0,expected_source_revision=-1)
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 3
    assert store.get_memory_job(first["job_id"])["status"] == "queued"


def test_target_eligibility_and_correction_chain(flow_store):
    store = flow_store
    draft = store.add_memory_source(LEARNER,TOPIC,"Черновик.","teacher")
    with pytest.raises(ValueError,match="eligible"):
        _request(store,draft)
    session = store.create_learning_session(LEARNER,"student",TOPIC)
    student = store.reserve_chat_message(session["id"],LEARNER,"first","ну вроде","student")
    student_source = {"source_id":student["source_id"]}
    with pytest.raises(ValueError,match="eligible"):
        _request(store,student_source)
    store.complete_chat_message(session["id"],"first",{"answer":"Покажи один шаг."})
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM teachcopilot.memory_sources WHERE learning_session_id=%s AND author_role='assistant'",(session["id"],))
        assistant = {"source_id":cur.fetchone()["id"]}
    with pytest.raises(ValueError,match="eligible"):
        _request(store,assistant)
    store.complete_learning_session(session["id"],LEARNER,"student")
    first = _request(store,student_source)
    second = _request(store,first["source"],key="second",text="Уточнение: помощь всё ещё неизвестна.")
    sources = store.get_learner_memory(LEARNER)["sources"]
    assert sources[-1]["corrects_source_id"] == first["source"]["source_id"]
    assert sources[-2]["corrects_source_id"] == student["source_id"]
    assert second["source"]["content"] == "Уточнение: помощь всё ещё неизвестна."


def test_atomic_queue_overflow_rolls_back_correction_and_source_revision(correction_api):
    store,browser,teacher,_,_ = correction_api
    original = _source(store)
    source_revision = store.get_learner_source_revision(LEARNER)
    for index in range(40):
        store.enqueue_memory_job(LEARNER,TOPIC,"memory_update",actor_id="teacher",idempotency_key=f"queue-{index}")
    url = f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections"
    payload = {"text":CORRECTION,"expected_revision":store.get_revision(LEARNER),
               "expected_source_revision":source_revision,"client_correction_id":"full-queue"}
    assert browser.post(url,headers=teacher,json=payload).status_code == 429
    assert store.get_learner_source_revision(LEARNER) == source_revision
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 1
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS count FROM teachcopilot.memory_jobs")
        assert cur.fetchone()["count"] == 40


def test_same_learner_concurrent_corrections_use_source_revision_guard(flow_store):
    store = flow_store
    original = _source(store)
    revision = store.get_revision(LEARNER)
    source_revision = store.get_learner_source_revision(LEARNER)

    def write(key):
        try:
            return _request(store,original,key=key,expected_revision=revision,expected_source_revision=source_revision)
        except RevisionConflict:
            return "revision_conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write,["first","second"]))
    assert sum(isinstance(result,dict) for result in results) == 1
    assert results.count("revision_conflict") == 1
    assert store.get_learner_source_revision(LEARNER) == source_revision+1


def test_same_client_concurrent_corrections_replay_one_job(flow_store):
    store = flow_store
    original = _source(store)
    revision = store.get_revision(LEARNER)
    source_revision = store.get_learner_source_revision(LEARNER)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _request(store,original,expected_revision=revision,
            expected_source_revision=source_revision),range(2)))
    assert results[0] == results[1]
    assert len(store.get_learner_memory(LEARNER)["sources"]) == 2


def test_database_rejects_wrong_topic_role_forward_links_and_mutation(flow_store):
    store = flow_store
    original = _source(store)
    with pytest.raises(ValueError,match="same topic"):
        store.add_memory_source(LEARNER,OTHER_TOPIC,CORRECTION,"teacher",approved=True,
                                corrects_source_id=original["source_id"])
    with pytest.raises(ValueError,match="approved teacher"):
        store.add_memory_source(LEARNER,TOPIC,CORRECTION,"student",author_role="student",approved=True,
                                corrects_source_id=original["source_id"])
    for topic,role,order in ((OTHER_TOPIC,"teacher",None),(TOPIC,"student",None),(TOPIC,"teacher",0)):
        with pytest.raises(psycopg2.IntegrityError):
            with store._connect() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO teachcopilot.memory_sources
                    (learner_id,topic_id,actor_id,content,author_role,approved,corrects_source_id,source_order)
                    VALUES (%s,%s,'teacher',%s,%s,TRUE,%s,COALESCE(%s,nextval('teachcopilot.memory_sources_source_order_seq')))""",
                    (LEARNER,topic,CORRECTION,role,original["source_id"],order))
    correction = _request(store,original)["source"]
    for field,value in (("content","rewritten"),("corrects_source_id",correction["source_id"]),("topic_id",OTHER_TOPIC)):
        with pytest.raises(psycopg2.IntegrityError,match="immutable"):
            with store._connect() as conn, conn.cursor() as cur:
                cur.execute(f"UPDATE teachcopilot.memory_sources SET {field}=%s WHERE id=%s",(value,original["source_id"]))


def test_proposals_preserve_correction_links_and_reject_invalid_target_atomically(flow_store):
    store = flow_store
    original = _source(store)
    entry = {"topic_id":TOPIC,"text":CORRECTION,"corrects_source_id":original["source_id"]}
    with store._connect() as conn, conn.cursor() as cur:
        jobs = store._confirm_memory_proposal(cur,{"memory_sources":{LEARNER:entry}},"teacher",{LEARNER:1})
    assert jobs[0]["payload"]["source_id"] == store.get_learner_memory(LEARNER)["sources"][-1]["source_id"]
    assert store.get_learner_memory(LEARNER)["sources"][-1]["corrects_source_id"] == original["source_id"]
    source_revision = store.get_learner_source_revision(LEARNER)
    with pytest.raises(KeyError):
        with store._connect() as conn, conn.cursor() as cur:
            store._confirm_memory_proposal(cur,{"memory_sources":{LEARNER:[
                {"topic_id":TOPIC,"text":"This must roll back."},
                {**entry,"corrects_source_id":"unknown"}]}},"teacher",{LEARNER:1})
    assert store.get_learner_source_revision(LEARNER) == source_revision
    with pytest.raises(ValueError,match="same topic"):
        with store._connect() as conn, conn.cursor() as cur:
            store._confirm_memory_proposal(cur,{"memory_sources":{LEARNER:{**entry,"topic_id":OTHER_TOPIC}}},"teacher",{LEARNER:1})


def test_persisted_corrects_source_reference_is_checked(flow_store):
    store = flow_store
    with pytest.raises(ValueError,match="unavailable"):
        store.save_memory_version(LEARNER,TOPIC,{"method":"tail"},base_revision=1,
                                  context={"corrects_source_id":"foreign-or-missing"})


@pytest.mark.parametrize("reference",["",1,True,[],{}])
def test_persisted_correction_reference_rejects_malformed_identifier(flow_store,reference):
    with pytest.raises(ValueError,match="nonempty source identifier"):
        flow_store.save_memory_version(LEARNER,TOPIC,{"method":"tail"},base_revision=1,
                                       context={"corrects_source_id":reference})


def test_database_rejects_cross_learner_correction_even_with_identical_topic(flow_store):
    store = flow_store
    original = _source(store)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES ('other-correction-learner','Other',4,%s)",
                    (Json({"learner_id":"other-correction-learner"}),))
    with pytest.raises(psycopg2.IntegrityError):
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.memory_sources
                (learner_id,topic_id,actor_id,content,author_role,approved,corrects_source_id)
                VALUES ('other-correction-learner',%s,'teacher',%s,'teacher',TRUE,%s)""",
                (TOPIC,CORRECTION,original["source_id"]))


def test_confirmed_correction_reaches_chat_projection_with_both_literal_sources(correction_api):
    from local_app.memory_service import MemoryWorker
    from scripts.load_memory import ControlledRuntime

    store,browser,teacher,student,_ = correction_api
    original = _source(store)
    worker = MemoryWorker(store,ControlledRuntime(0))
    before = store.get_learner_memory(LEARNER,TOPIC)
    state,context = worker._prepare({"learner_id":LEARNER,"topic_id":TOPIC,"method":"tail"})
    initial = store.save_memory_version(LEARNER,TOPIC,state,base_revision=before["revision"],
        source_revision=before["source_revision"],context=context)
    store.approve_memory_version(LEARNER,initial["id"],"teacher",before["revision"])
    rag = InspectableRag()
    browser.app.state.rag_service = rag

    def assert_withdrawn_memory_is_withheld():
        # Exercise the compatible chat API without adding another durable source:
        # an active-session message would change the confirmation revision itself.
        response = browser.post("/v1/chat/completions",headers=student,json={
            "messages":[{"role":"user","content":"Продолжим прошлое занятие?"}]})
        assert response.status_code == 200, response.text
        arguments = rag.calls[-1][1]
        assert arguments["memory_projection"] == []
        assert arguments["memory_context"]["text"] == ""
        assert arguments["memory_context"]["sources"] == []
        assert arguments["memory_context"]["version_id"] is None
        assert arguments["memory_context"]["withheld_reason"] == "source_correction_requires_review"

    response = browser.post(f"/learners/{LEARNER}/memory/sources/{original['source_id']}/corrections",
        headers=teacher,json={"text":CORRECTION,"expected_revision":before["revision"],
            "expected_source_revision":before["source_revision"],"client_correction_id":"chat-projection"})
    assert response.status_code == 202, response.text
    correction = response.json()["source"]
    assert_withdrawn_memory_is_withheld()
    completed = worker.run_once()
    assert completed["status"] == "completed", completed.get("error")
    after = store.get_learner_memory(LEARNER,TOPIC)
    version = after["versions"][0]
    assert version["approved"] is False and version["id"] != initial["id"]
    assert {row["source_id"] for row in version["context"]["sources"]} == {
        original["source_id"],correction["source_id"]}
    assert_withdrawn_memory_is_withheld()
    assert browser.post(f"/learners/{LEARNER}/memory/versions/{version['id']}/confirm",
        headers=teacher,json={"expected_revision":after["revision"],"interest_candidate_ids":[]}).status_code == 200
    session = browser.post(f"/learners/{LEARNER}/learning-sessions",headers=student,
                            json={"topic_id":TOPIC}).json()["learning_session_id"]
    response = browser.post("/v1/chat/completions",headers=student,json={
        "learning_session_id":session,"client_message_id":"after-correction",
        "messages":[{"role":"user","content":"Продолжим?"}]})
    assert response.status_code == 200, response.text
    projection = rag.calls[-1][1]["memory_projection"]
    assert {row["source_id"] for row in projection} == {original["source_id"],correction["source_id"]}
    linked = next(row for row in projection if row["source_id"] == correction["source_id"])
    assert linked["corrects_source_id"] == original["source_id"]
    assert linked["quote"] == CORRECTION and linked["role"] == "teacher"
    source = next(row for row in projection if row["source_id"] == original["source_id"])
    assert "corrects_source_id" not in source and source["quote"] == original["content"]

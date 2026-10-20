"""A completed lesson's last assistant turn is context, never learner evidence."""
from copy import deepcopy

import pytest
from psycopg2.extras import Json

from local_app.memory_store import SESSION_END_CONTEXT_CHARS
from local_app.store import RevisionConflict, Store
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
OTHER_TOPIC = "programming.python.variables"
ENDING = "Попробуй записать дробь для четырёх из десяти частей. Напиши только верхнее число."


def _turn(store, session, text="хз", answer=ENDING, key="one"):
    message = store.reserve_chat_message(session["id"], session["learner_id"], key,
                                         text, session["actor_id"])
    store.complete_chat_message(session["id"], key, {"answer": answer})
    rows = store.list_learning_messages(session["id"])
    assistant = next(row for row in rows if row["role"] == "assistant" and row["content"] == answer)
    return message, assistant


def _lesson(store, *, text="хз", answer=ENDING, learner=LEARNER, topic=TOPIC, complete=True):
    session = store.create_learning_session(learner,"student",topic)
    message, assistant = _turn(store,session,text,answer)
    if complete:
        store.complete_learning_session(session["id"],learner,"student")
    return session,message,assistant


def _rows(store, topic=TOPIC):
    return {row["source_id"]:row for row in store.get_learner_memory(LEARNER,topic)["sources"]}


def _expected(session, assistant, text=ENDING):
    return {"assistant_source_id":assistant["source_id"],"session_id":session["id"],
            "role":"assistant","usage":"context_only","text":text,
            "excerpt_mode":"whole","truncated":False}


def _version_inputs(store, source):
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    state = {"method":"tail","topic_id":TOPIC,"sources":{
        source["source_id"]:{"turn_id":source["source_id"],"role":"student",
                             "text":source["text"],"observed_on":"2026-10-20"}}}
    context = {"topic_id":TOPIC,"text":source["text"],"evidence_ids":[source["source_id"]],
               "sources":[{"source_id":source["source_id"],"role":"student","text":source["text"]}],
               "session_end_context":[{"source_id":source["source_id"],**source["session_end_context"]}]}
    return snapshot,state,context


def _save(store, snapshot, state, context):
    return store.save_memory_version(LEARNER,TOPIC,state,context=context,
        base_revision=snapshot["revision"],source_revision=snapshot["source_revision"])


def test_final_instruction_belongs_to_last_student_without_becoming_evidence(flow_store):
    store = flow_store
    session = store.create_learning_session(LEARNER,"student",TOPIC)
    first,_ = _turn(store,session,"Круг на десять частей, четыре закрашены.",
                   "Попробуй подписать дробь.","first")
    last,assistant = _turn(store,session,"хз",ENDING,"last")
    store.complete_learning_session(session["id"],LEARNER,"student")
    rows = _rows(store)
    assert set(rows) == {first["source_id"],last["source_id"]}
    assert "session_end_context" not in rows[first["source_id"]]
    assert rows[last["source_id"]]["session_end_context"] == _expected(session,assistant)
    assert rows[last["source_id"]]["dialogue_context"]["text"] == "Попробуй подписать дробь."
    assert rows[last["source_id"]]["text"] == "хз"
    assert assistant["source_id"] not in rows
    assert all(row["author_role"] != "assistant" for row in rows.values())


def test_only_latest_completion_per_topic_ignores_active_other_topic_and_foreign(flow_store):
    store = flow_store
    first,first_message,_ = _lesson(store,answer="FIRST ENDING",complete=False)
    second,second_message,second_assistant = _lesson(store,answer="SECOND ENDING")
    # Completion order wins over the order in which sources were created.
    store.complete_learning_session(first["id"],LEARNER,"student")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_sessions SET completed_at=now()-interval '1 hour' WHERE id=%s",
                    (second["id"],))
    other,other_message,other_assistant = _lesson(store,topic=OTHER_TOPIC,answer="OTHER ENDING")
    _,active_message,_ = _lesson(store,answer="ACTIVE ENDING",complete=False)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                    ("foreign-ending","Synthetic",Json({"learner_id":"foreign-ending"})))
    _lesson(store,learner="foreign-ending",answer="FOREIGN ENDING")
    rows = _rows(store,None)
    assert rows[first_message["source_id"]]["session_end_context"]["text"] == "FIRST ENDING"
    assert "session_end_context" not in rows[second_message["source_id"]]
    assert rows[other_message["source_id"]]["session_end_context"] == _expected(other,other_assistant,"OTHER ENDING")
    assert active_message["source_id"] not in rows
    assert "FOREIGN ENDING" not in str(rows) and "ACTIVE ENDING" not in str(rows)
    assert other_message["source_id"] not in _rows(store)


def test_completion_tie_uses_session_id_and_empty_latest_does_not_revive_old_ending(flow_store):
    store = flow_store
    first,first_message,_ = _lesson(store,answer="FIRST ENDING")
    second,second_message,_ = _lesson(store,answer="SECOND ENDING")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_sessions SET completed_at=now()-interval '1 hour' WHERE learner_id=%s",
                    (LEARNER,))
    winning_message = first_message if first["id"] > second["id"] else second_message
    rows = _rows(store)
    assert {key for key,row in rows.items() if "session_end_context" in row} == {winning_message["source_id"]}
    empty = store.create_learning_session(LEARNER,"student",TOPIC)
    store.complete_learning_session(empty["id"],LEARNER,"student")
    assert all("session_end_context" not in row for row in _rows(store).values())


@pytest.mark.parametrize("suffix", ["teacher","student","teacher_then_assistant","wrong_topic_assistant","wrong_topic_student"])
def test_intervening_or_wrong_role_and_topic_sources_are_not_skipped(flow_store,suffix):
    store = flow_store
    session,message,_ = _lesson(store,complete=False)
    with store._connect() as conn, conn.cursor() as cur:
        store._memory_learner(cur,LEARNER)
        role = "assistant" if suffix == "wrong_topic_assistant" else "student" if suffix in {"student","wrong_topic_student"} else "teacher"
        store._memory_source(cur,LEARNER,OTHER_TOPIC if suffix.startswith("wrong_topic") else TOPIC,
            "synthetic", "INTERVENING SOURCE",author_role=role,approved=role=="teacher",session_id=session["id"])
        if suffix in {"teacher_then_assistant","wrong_topic_student"}:
            store._memory_source(cur,LEARNER,TOPIC,"assistant","FINAL ASSISTANT",
                                 author_role="assistant",session_id=session["id"])
    store.complete_learning_session(session["id"],LEARNER,"student")
    rows = _rows(store)
    assert message["source_id"] in rows
    assert all("session_end_context" not in row for row in rows.values())


@pytest.mark.parametrize("length", [SESSION_END_CONTEXT_CHARS,SESSION_END_CONTEXT_CHARS+1])
def test_session_end_context_is_whole_or_explicitly_omitted(flow_store,length):
    answer = "я" * length
    session,message,assistant = _lesson(flow_store,answer=answer)
    source = _rows(flow_store)[message["source_id"]]
    if length == SESSION_END_CONTEXT_CHARS:
        assert source["session_end_context"] == _expected(session,assistant,answer)
        assert "session_end_context_omitted" not in source
    else:
        assert "session_end_context" not in source
        assert source["session_end_context_omitted"] == {
            "reason":"exceeds_context_excerpt_limit","limit_chars":SESSION_END_CONTEXT_CHARS}


def test_context_persists_as_draft_then_approved_across_store_restart(flow_store):
    store = flow_store
    _,message,assistant = _lesson(store)
    source = _rows(store)[message["source_id"]]
    snapshot,state,context = _version_inputs(store,source)
    original_knowledge = deepcopy(store.get_card(LEARNER).get("knowledge"))
    saved = _save(store,snapshot,state,context)
    restarted = Store(store.dsn)
    draft = restarted.get_learner_memory(LEARNER,TOPIC)["versions"][0]
    assert draft["id"] == saved["id"] and not draft["approved"]
    assert draft["context"]["session_end_context"] == context["session_end_context"]
    assert assistant["source_id"] not in draft["state"]["sources"]
    restarted.approve_memory_version(LEARNER,saved["id"],"teacher",snapshot["revision"])
    approved = Store(store.dsn).get_learner_memory(LEARNER,TOPIC)["versions"][0]
    assert approved["approved"] and not approved["stale"]
    assert approved["context"]["session_end_context"] == context["session_end_context"]
    assert restarted.get_card(LEARNER).get("knowledge") == original_knowledge


def test_active_next_session_preserves_approval_but_completion_expires_old_context(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    approved = _save(store,snapshot,state,context)
    store.approve_memory_version(LEARNER,approved["id"],"teacher",snapshot["revision"])
    session,_,_ = _lesson(store,answer="NEW ACTIVE INSTRUCTION",complete=False)
    stored = store.get_learner_memory(LEARNER,TOPIC)["versions"][0]
    assert not stored["stale"] and stored["context"] == {**context,"interest_review":stored["context"]["interest_review"]}
    store.complete_learning_session(session["id"],LEARNER,"student")
    assert store.get_learner_memory(LEARNER,TOPIC)["versions"][0]["stale"]
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,store.get_learner_memory(LEARNER,TOPIC),state,context)


def test_forged_metadata_is_rejected_and_assistant_reference_never_unlocks_evidence(flow_store):
    store = flow_store
    _,message,assistant = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    alterations = [{"role":"student"},{"usage":"evidence"},{"text":ENDING+" Invented success."},
                   {"excerpt_mode":"tail"},{"truncated":True},{"truncated":0},
                   {"assistant_source_id":message["source_id"]},{"assistant_source_id":"missing-assistant"},
                   {"session_id":"00000000-0000-0000-0000-000000000000"},
                   {"source_id":assistant["source_id"]},{"text":"я"*(SESSION_END_CONTEXT_CHARS+1)},
                   {"extra":"not allowed"}]
    for change in alterations:
        forged = deepcopy(context)
        forged["session_end_context"][0].update(change)
        with pytest.raises(ValueError):
            _save(store,snapshot,state,forged)
    for invalid in ({},None,"not-a-list"):
        with pytest.raises(ValueError,match="must be a list"):
            _save(store,snapshot,state,{**context,"session_end_context":invalid})
    duplicated = deepcopy(context)
    duplicated["session_end_context"] *= 2
    with pytest.raises(ValueError,match="metadata"):
        _save(store,snapshot,state,duplicated)
    for key in ("evidence_ids","evidence_turn_ids"):
        with pytest.raises(ValueError,match="assistant evidence"):
            _save(store,snapshot,state,{**context,key:[assistant["source_id"]]})
    forged_state = deepcopy(state)
    forged_state["sources"][assistant["source_id"]] = {"turn_id":assistant["source_id"],"role":"student",
                                                       "text":"Solved everything.","observed_on":"2026-10-20"}
    with pytest.raises(ValueError,match="assistant evidence"):
        _save(store,snapshot,forged_state,context)
    assert store.get_learner_memory(LEARNER,TOPIC)["versions"] == []


def test_reference_guard_rejects_foreign_active_and_non_immediate_assistant(flow_store):
    store = flow_store
    _,old_message,_ = _lesson(store,answer="OLD INSTRUCTION")
    old_source = _rows(store)[old_message["source_id"]]
    _,message,assistant = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,snapshot,state,{**context,"session_end_context":[{
            "source_id":old_message["source_id"],**old_source["session_end_context"]}]})
    session,active_message,active_assistant = _lesson(store,complete=False,answer="ACTIVE INSTRUCTION")
    invalid = {"source_id":active_message["source_id"],**_expected(session,active_assistant,"ACTIVE INSTRUCTION")}
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,store.get_learner_memory(LEARNER,TOPIC),state,{**context,"session_end_context":[invalid]})
    store.complete_learning_session(session["id"],LEARNER,"student")
    invalid = {"source_id":active_message["source_id"],**_expected(session,assistant)}
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,store.get_learner_memory(LEARNER,TOPIC),state,{**context,"session_end_context":[invalid]})
    with pytest.raises(ValueError,match="only allowed"):
        _save(store,store.get_learner_memory(LEARNER,TOPIC),state,
              {"assistant_source_id":active_assistant["source_id"]})


def test_valid_closing_context_from_another_topic_cannot_be_saved_in_this_topic(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    other,other_message,other_assistant = _lesson(store,topic=OTHER_TOPIC,answer="OTHER TOPIC INSTRUCTION")
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    other_ending = {"source_id":other_message["source_id"],
                    **_expected(other,other_assistant,"OTHER TOPIC INSTRUCTION")}
    context["session_end_context"] = [other_ending]
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,snapshot,state,context)
    with pytest.raises(ValueError,match="latest completed session"):
        store.save_next_step(LEARNER,TOPIC,{"action":"check"},context=context,
            base_revision=snapshot["revision"],source_revision=snapshot["source_revision"])


def test_wrong_student_anchor_and_foreign_assistant_cannot_be_persisted(flow_store):
    store = flow_store
    session = store.create_learning_session(LEARNER,"student",TOPIC)
    first,_ = _turn(store,session,answer="FIRST INSTRUCTION",key="first")
    last,_ = _turn(store,session,key="last")
    store.complete_learning_session(session["id"],LEARNER,"student")
    snapshot,state,context = _version_inputs(store,_rows(store)[last["source_id"]])
    forged = deepcopy(context)
    forged["session_end_context"][0]["source_id"] = first["source_id"]
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,snapshot,state,forged)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                    ("foreign-ending-reference","Synthetic",Json({"learner_id":"foreign-ending-reference"})))
    foreign,_,assistant = _lesson(store,learner="foreign-ending-reference")
    forged["session_end_context"][0] = {"source_id":last["source_id"],**_expected(foreign,assistant)}
    with pytest.raises(ValueError,match="latest completed session"):
        _save(store,snapshot,state,forged)


def test_new_completed_sources_prevent_old_draft_approval(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    draft = _save(store,snapshot,state,context)
    _lesson(store,answer="ANOTHER LESSON ENDING")
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER,draft["id"],"teacher",snapshot["revision"])


def test_completed_empty_same_topic_session_expires_only_stored_closing_context(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    version = _save(store,snapshot,state,context)
    store.approve_memory_version(LEARNER,version["id"],"teacher",snapshot["revision"])
    before = store.get_learner_memory(LEARNER,TOPIC)
    stored_before = before["versions"][0]
    assert not stored_before["stale"]
    empty = store.create_learning_session(LEARNER,"student",TOPIC)
    store.complete_learning_session(empty["id"],LEARNER,"student")
    for topic_filter in (TOPIC,None):
        after = Store(store.dsn).get_learner_memory(LEARNER,topic_filter)
        stored = next(row for row in after["versions"] if row["id"] == version["id"])
        assert stored["stale"] and stored["approved"]
        assert stored["context"] == stored_before["context"]
        assert stored["state"] == stored_before["state"]
        assert {row["source_id"] for row in after["sources"]} == {row["source_id"] for row in before["sources"]}
        assert all("session_end_context" not in row for row in after["sources"])


def test_active_empty_and_other_topic_completion_preserve_stored_closing_freshness(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    version = _save(store,snapshot,state,context)
    store.approve_memory_version(LEARNER,version["id"],"teacher",snapshot["revision"])
    store.create_learning_session(LEARNER,"student",TOPIC)
    other = store.create_learning_session(LEARNER,"student",OTHER_TOPIC)
    store.complete_learning_session(other["id"],LEARNER,"student")
    for topic_filter in (TOPIC,None):
        stored = store.get_learner_memory(LEARNER,topic_filter)["versions"][0]
        assert stored["id"] == version["id"] and not stored["stale"]
        assert stored["context"]["session_end_context"] == context["session_end_context"]


def test_version_without_selected_closing_context_is_not_enriched_or_expired(flow_store):
    store = flow_store
    _,message,_ = _lesson(store)
    snapshot,state,context = _version_inputs(store,_rows(store)[message["source_id"]])
    context.pop("session_end_context")
    version = _save(store,snapshot,state,context)
    store.approve_memory_version(LEARNER,version["id"],"teacher",snapshot["revision"])
    before = store.get_learner_memory(LEARNER,TOPIC)["versions"][0]
    empty = store.create_learning_session(LEARNER,"student",TOPIC)
    store.complete_learning_session(empty["id"],LEARNER,"student")
    after = Store(store.dsn).get_learner_memory(LEARNER,TOPIC)["versions"][0]
    assert not after["stale"] and "session_end_context" not in after["context"]
    assert after["context"] == before["context"] and after["state"] == before["state"]


def test_explicit_teacher_correction_coexists_with_session_context(flow_store):
    store = flow_store
    _,message,_ = _lesson(store,text="ага")
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    result = store.correct_memory_source(LEARNER,message["source_id"],
        "Это согласие, самостоятельного решения не было.","teacher",
        expected_revision=snapshot["revision"],expected_source_revision=snapshot["source_revision"],
        client_correction_id="ending-correction")
    rows = _rows(store)
    assert rows[message["source_id"]]["session_end_context"]["text"] == ENDING
    assert rows[result["source"]["id"]]["corrects_source_id"] == message["source_id"]
    assert "session_end_context" not in rows[result["source"]["id"]]

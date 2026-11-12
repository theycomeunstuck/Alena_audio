"""The owner columns of get_own_conversation_messages must come from stored rows, not echo the request.

read_own_messages() compares each row's learner_id/actor_id with the requested pair. That comparison only
means something if the query returns the stored owners. Real isolation is in the SQL joins; this guards
the defence-in-depth check against becoming vacuous again.
"""
import inspect
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from local_app.store import Store

DSN = os.getenv("TEACHCOPILOT_TEST_DSN")


def reader_source():
    method = Store.get_own_conversation_messages
    return inspect.getsource(method)


def test_query_does_not_echo_request_parameters_as_owner_columns():
    source = reader_source()
    assert not re.search(r"%\(learner\)s\s*::\s*text\s+AS\s+learner_id", source), "learner_id echoes the request"
    assert not re.search(r"%\(actor\)s\s*::\s*text\s+AS\s+actor_id", source), "actor_id echoes the request"


def test_every_union_branch_selects_stored_owner_columns():
    source = reader_source()
    # three UNION branches (child message, assistant reply, shown check-in), each projecting both columns
    assert len(re.findall(r"AS learner_id", source)) == 3
    assert len(re.findall(r"AS actor_id", source)) == 3


@pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
def test_owner_columns_equal_the_stored_owner_for_child_and_assistant_rows():
    from tests.test_conversation_history_postgres import historical_turn
    from psycopg2.extras import Json

    store = Store(DSN)
    store.migrate()
    learner = "owner-cols-" + uuid4().hex
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,'Synthetic',3,%s)",
                    (learner, Json({"learner_id": learner})))
    account = store.create_account(uuid4().hex + "@synthetic.test", "Synthetic", "unused", "student")
    try:
        session = store.create_learning_session(learner, account["id"], "math.test", event_timezone="Europe/Moscow")
        historical_turn(store, learner, account["id"], session, datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc))
        rows = store.get_own_conversation_messages(learner, account["id"])
        assert {row["role"] for row in rows} == {"user", "assistant"}
        # assistant sources are stored with actor_id='assistant'; the row must still carry the session owner
        assert all(row["learner_id"] == learner and str(row["actor_id"]) == str(account["id"]) for row in rows)
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=%s", (learner,))
            cur.execute("DELETE FROM teachcopilot.accounts WHERE id=%s::uuid", (str(account["id"]),))

"""Persist and verify autonomous division cursors from typed database events."""
from __future__ import annotations

from copy import deepcopy

from . import autonomous_division as division


def _session(cur, learner_id, session_id, actor_id):
    cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE id=%s AND learner_id=%s",
                (session_id, learner_id))
    row = cur.fetchone()
    if row is None:
        raise KeyError(session_id)
    if str(row["actor_id"]) != str(actor_id):
        raise PermissionError("learning session belongs to another actor")
    if row["session_mode"] != "autonomous_division" or row["topic_id"] != division.TOPIC:
        raise ValueError("learning session does not use the autonomous division protocol")
    return dict(row)


def _messages(cur, session_id, learner_id):
    cur.execute("""SELECT m.*,s.content,s.author_role,s.id AS source_id,s.source_order
        FROM teachcopilot.learning_messages m
        JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
        WHERE m.learning_session_id=%s AND m.learner_id=%s
        ORDER BY s.source_order,m.id""", (session_id, learner_id))
    return [dict(row) for row in cur.fetchall()]


def _latest_event(rows, pending_client_message_id=None):
    for row in reversed(rows):
        if row["status"] == "failed":
            continue
        if row["status"] == "pending" and row["client_message_id"] == pending_client_message_id:
            continue
        if row["status"] in {"pending", "unknown_outcome"}:
            return None, row["status"]
        if row["status"] != "completed":
            return None, "invalid_autonomous_chain"
        response = row.get("response")
        if (not isinstance(response, dict) or response.get("learning_protocol") != division.VERSION
                or "_autonomous_division_state" not in response):
            return None, "invalid_autonomous_chain"
        state = response["_autonomous_division_state"]
        if state is None:
            return None, "no_saved_autonomous_step"
        return row, None
    return None, "no_saved_autonomous_step"


def _read_candidate(store, learner_id, topic_id, actor_id, *, learning_session_id=None,
                    pending_client_message_id=None):
    if topic_id != division.TOPIC:
        return {"available": False, "reason": "unsupported_topic", "_state": None}
    with store._connect() as connection, connection.cursor() as cur:
        cur.execute("""SELECT * FROM teachcopilot.learning_sessions
            WHERE learner_id=%s AND actor_id=%s AND session_mode='autonomous_division'
              AND topic_id=%s AND status='active'
            ORDER BY created_at DESC,id DESC LIMIT 1""", (learner_id, str(actor_id), topic_id))
        active = cur.fetchone()
        selected = _session(cur, learner_id, learning_session_id, actor_id) if learning_session_id else (
            dict(active) if active else None)
        if selected and selected["status"] == "completed":
            current_session_id = str(selected["id"])
        elif selected:
            current_session_id = str(selected["id"])
        else:
            current_session_id = None

        selected_rows = _messages(cur, current_session_id, learner_id) if current_session_id else []
        candidate, blocked = _latest_event(selected_rows, pending_client_message_id)
        if blocked in {"pending", "unknown_outcome", "invalid_autonomous_chain"}:
            return {"available": False, "reason": blocked, "_state": None,
                    "active_learning_session_id": current_session_id}
        if candidate is None:
            boundary = selected.get("created_at") if selected else None
            boundary_id = selected.get("id") if selected else None
            cur.execute("""SELECT id FROM teachcopilot.learning_sessions
                WHERE learner_id=%s AND actor_id=%s AND session_mode='autonomous_division'
                  AND topic_id=%s AND status='completed'
                  AND (%s::timestamptz IS NULL OR (created_at,id)<(%s::timestamptz,%s::uuid))
                ORDER BY created_at DESC,id DESC""",
                (learner_id, str(actor_id), topic_id, boundary, boundary, boundary_id))
            for previous in cur.fetchall():
                rows = _messages(cur, str(previous["id"]), learner_id)
                candidate, blocked = _latest_event(rows)
                if blocked == "no_saved_autonomous_step":
                    continue
                if blocked:
                    return {"available": False, "reason": blocked, "_state": None,
                            "active_learning_session_id": current_session_id}
                break
        if candidate is None:
            return {"available": False, "reason": "no_saved_autonomous_step", "_state": None,
                    "active_learning_session_id": current_session_id}
        try:
            state, chain = _verify_chain(store, cur, learner_id, topic_id, actor_id, candidate)
        except (KeyError, TypeError, ValueError, PermissionError):
            return {"available": False, "reason": "invalid_autonomous_chain", "_state": None,
                    "active_learning_session_id": current_session_id}
        if _has_corrected_source(cur, learner_id, chain):
            return {"available": False, "reason": "corrected_autonomous_source", "_state": None,
                    "active_learning_session_id": current_session_id}
        state = division.validate_state(state)
        source_session_id = str(candidate["learning_session_id"])
        resume_required = source_session_id != current_session_id or state["status"] == "paused"
        return {"available": True, "reason": None, "_state": deepcopy(state),
                "source_session_id": source_session_id,
                "source_message_id": candidate["client_message_id"],
                "active_learning_session_id": current_session_id,
                "resume_required": resume_required,
                "prompt": division.render_prompt(state)}


def _verify_chain(store, cur, learner_id, topic_id, actor_id, candidate):
    from .explicit_learning_actions import validate_action

    expected = candidate
    seen = set()
    chain = []
    candidate_state = None
    for _ in range(2048):
        key = (str(expected["learning_session_id"]), expected["client_message_id"])
        if key in seen:
            raise ValueError("cyclic autonomous chain")
        seen.add(key)
        session = _session(cur, learner_id, key[0], actor_id)
        if expected["status"] != "completed" or expected["author_role"] != "student":
            raise ValueError("incomplete autonomous action")
        response = expected.get("response")
        if (not isinstance(response, dict) or response.get("learning_protocol") != division.VERSION
                or response.get("_autonomous_division_state") is None):
            raise ValueError("missing autonomous state")
        current = division.validate_state(response["_autonomous_division_state"])
        if candidate_state is None:
            candidate_state = current
        if current["state_id"] != expected["client_message_id"]:
            raise ValueError("state does not belong to its source action")
        provenance = response.get("_autonomous_division_provenance")
        if not isinstance(provenance, dict) or set(provenance) != {
                "action", "previous_session_id", "previous_message_id", "interrupted"}:
            raise ValueError("missing autonomous provenance")
        raw_action = provenance["action"]
        operation = raw_action.get("operation") if isinstance(raw_action, dict) else None
        if operation == "autonomous_division_clarify":
            if set(raw_action) not in ({"operation"}, {"operation", "expected_state_id"}):
                raise ValueError("invalid clarification action")
            action = dict(raw_action)
            request_text = expected["content"]
        else:
            action, request_text = validate_action(raw_action)
            if action != raw_action or request_text != expected["content"]:
                raise ValueError("autonomous action differs from its source text")
        if current["state_id"] != key[1]:
            raise ValueError("unexpected state identity")
        previous_session_id = provenance["previous_session_id"]
        previous_message_id = provenance["previous_message_id"]
        if (previous_session_id is None) != (previous_message_id is None):
            raise ValueError("incomplete autonomous predecessor")
        previous_state = None
        previous = None
        if previous_session_id is not None:
            _session(cur, learner_id, previous_session_id, actor_id)
            cur.execute("""SELECT m.*,s.content,s.author_role,s.id AS source_id,s.source_order
                FROM teachcopilot.learning_messages m
                JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                WHERE m.learning_session_id=%s AND m.client_message_id=%s AND m.learner_id=%s""",
                (previous_session_id, previous_message_id, learner_id))
            previous = cur.fetchone()
            if previous is None:
                raise ValueError("missing autonomous predecessor")
            previous = dict(previous)
            if (previous["status"] != "completed" or previous["author_role"] != "student"
                    or action.get("expected_state_id") != previous_message_id):
                raise ValueError("invalid autonomous predecessor")
            previous_response = previous.get("response")
            if not isinstance(previous_response, dict):
                raise ValueError("invalid autonomous predecessor response")
            previous_state = previous_response.get("_autonomous_division_state")
            if previous_state is None:
                raise ValueError("autonomous predecessor has no state")
            previous_state = division.validate_state(previous_state)
            if (str(previous_session_id) != key[0]
                    and operation not in {"autonomous_division_resume", "autonomous_division_next", "autonomous_division_switch", "autonomous_division_clarify"}):
                raise ValueError("cross-session link is not an explicit continuation")
        actual, assessment = division.transition(previous_state, action, state_id=key[1])
        if actual != current or actual is None:
            raise ValueError("saved autonomous transition cannot be reproduced")
        if bool(provenance["interrupted"]) != bool(assessment.get("interrupted", False)):
            raise ValueError("incorrect interruption marker")
        expected_public = division.public_state(actual)
        if (response.get("autonomous_division") != expected_public
                or response.get("autonomous_assessment") != {
                    "status": assessment.get("status", "not_assessed"),
                    "competence_assessed": False, "independent_work": "unknown"}
                or response.get("learning_action") != {
                    "operation": operation, "request_text": request_text, "version": division.VERSION}):
            raise ValueError("public autonomous response differs from its saved transition")
        expected_response = division.action_response(actual, assessment, operation, request_text,
                                                     interrupted=bool(assessment.get("interrupted")))
        if (response.get("choices") != expected_response.get("choices")
                or response.get("learning_tool", {}).get("facts") != {}) :
            raise ValueError("autonomous response is not deterministic")
        chain.append(str(expected["source_id"]))
        if operation in {"autonomous_division_start"}:
            if previous_session_id is not None or previous_message_id is not None:
                raise ValueError("autonomous start has an unexpected predecessor")
            break
        if operation in {"autonomous_division_next", "autonomous_division_switch"}:
            if previous is None or previous_state is None:
                raise ValueError("new autonomous example has no verified handoff state")
            # A next or switch action is the root of a fresh example chain. Its
            # immediate handoff is checked above; older example events remain
            # available in history but cannot invalidate this example.
            break
        if operation == "autonomous_division_clarify" and previous_state is None:
            if previous_session_id is not None:
                raise ValueError("empty clarification has a predecessor")
            raise ValueError("empty clarification cannot own a saved state")
        if previous is None:
            raise ValueError("autonomous action has no predecessor")
        expected = previous
    else:
        raise ValueError("autonomous chain exceeds verification bound")
    return candidate_state, chain


def _has_corrected_source(cur, learner_id, source_ids):
    cur.execute("""SELECT 1 FROM teachcopilot.memory_sources
        WHERE learner_id=%s AND corrects_source_id::text=ANY(%s) LIMIT 1""",
        (learner_id, source_ids))
    return cur.fetchone() is not None


def public_context(context):
    state = context.get("_state")
    if not context.get("available"):
        return {"learning_protocol": division.VERSION, "available": False,
                "reason": context.get("reason", "no_saved_autonomous_step")}
    public = division.public_state(state)
    if context.get("resume_required"):
        actions = ["resume", "switch"] if state["status"] != "completed" else ["next"]
    elif state["status"] == "paused":
        actions = ["resume", "switch"]
    elif state["status"] == "completed":
        actions = ["next"]
    else:
        actions = ["answer", "explain", "show_step", "pause", "switch"]
    if state["actions_used"] >= division.MAX_ACTIONS_PER_EXAMPLE:
        actions = ["switch"]
    return {"learning_protocol": division.VERSION, "available": True,
            "state": public, "prompt": context["prompt"],
            "resume_required": bool(context.get("resume_required")),
            "limit_reached": state["actions_used"] >= division.MAX_ACTIONS_PER_EXAMPLE,
            "available_actions": actions}


def autonomous_context(store, learner_id, topic_id, actor_id, *, learning_session_id=None,
                       pending_client_message_id=None):
    return public_context(_read_candidate(store, learner_id, topic_id, actor_id,
        learning_session_id=learning_session_id, pending_client_message_id=pending_client_message_id))


def autonomous_response(store, learner_id, topic_id, actor_id, session_id, message_id,
                        action, request_text):
    from .store import RevisionConflict
    from .explicit_learning_actions import validate_action

    if topic_id != division.TOPIC:
        raise ValueError("Автономное деление доступно только в теме деления.")
    current_context = _read_candidate(store, learner_id, topic_id, actor_id,
        learning_session_id=session_id, pending_client_message_id=message_id)
    if not current_context.get("available"):
        reason = current_context.get("reason")
        if reason in {"pending", "unknown_outcome", "invalid_autonomous_chain", "corrected_autonomous_source"}:
            raise RevisionConflict("autonomous division state is blocked; refresh the current session")
        state = None
        previous_session_id = previous_message_id = None
    else:
        state = current_context["_state"]
        previous_session_id = current_context["source_session_id"]
        previous_message_id = current_context["source_message_id"]
    if action["operation"] == "autonomous_division_clarify":
        action = {"operation": action["operation"], **(
            {"expected_state_id": state["state_id"]} if state is not None else {})}
    else:
        action, canonical_text = validate_action(action)
        if canonical_text != request_text:
            raise ValueError("Автономное действие не совпадает с исходным текстом.")
    if action["operation"] != "autonomous_division_start" and state is None:
        raise RevisionConflict("no autonomous division cursor is available; refresh the current session")
    if action["operation"] == "autonomous_division_start" and state is not None:
        raise RevisionConflict("an unfinished example exists; resume it or explicitly switch examples")
    if state is None:
        previous_session_id = previous_message_id = None
    elif action["operation"] in {"autonomous_division_resume", "autonomous_division_next",
                                  "autonomous_division_switch", "autonomous_division_clarify"}:
        if action.get("expected_state_id") != state["state_id"]:
            raise RevisionConflict("autonomous division cursor changed; refresh the current step")
    else:
        if previous_session_id != str(session_id):
            raise RevisionConflict("resume the saved example before continuing it")
        if action.get("expected_state_id") != state["state_id"]:
            raise RevisionConflict("autonomous division cursor changed; refresh the current step")
    try:
        next_state, assessment = division.transition(state, action, state_id=message_id)
    except ValueError as exc:
        raise RevisionConflict(str(exc)) from exc
    if next_state is None:
        response = division.action_response(None, assessment, action["operation"], request_text)
    else:
        response = division.action_response(next_state, assessment, action["operation"], request_text,
                                           interrupted=bool(assessment.get("interrupted")))
    response["_autonomous_division_state"] = deepcopy(next_state)
    response["_autonomous_division_provenance"] = {
        "action": deepcopy(action),
        "previous_session_id": previous_session_id,
        "previous_message_id": previous_message_id,
        "interrupted": bool(assessment.get("interrupted", False)),
    }
    return response

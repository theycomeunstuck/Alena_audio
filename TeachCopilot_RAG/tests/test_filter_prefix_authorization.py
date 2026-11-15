"""F-014 regression tests: the "[child_id:...]" message prefix must be authorized.

Before the fix, Filter.inlet honored an explicit "[child_id:UUID] question"
prefix UNCONDITIONALLY, so any Open WebUI user could read any child's profile
into their prompt (inlet) and write assessments to that child (outlet);
user_mappings was consulted only when no prefix was present.

The fix (variant a): the prefix is honored only when the requesting user is
authorized for that child_id - the child_ids mapped to their Open WebUI user
id in user_mappings, or the child _resolve_child_id would resolve to anyway
(DEFAULT_CHILD_ID fallback - never new access). An unauthorized prefix logs a
warning, leaves the message content untouched, and falls through to the
normal resolution path; the outlet likewise refuses to write assessments for
an unauthorized prefix child.

Pure-Python fakes; no database, no network.
"""
import copy
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import filter_function as ff  # noqa: E402

OWUI_USER = "owui-user-1"
MY_CHILD = "00000000-0000-0000-0000-00000000000a"
OTHER_CHILD = "00000000-0000-0000-0000-00000000000b"
DEFAULT_CHILD = "00000000-0000-0000-0000-00000000000d"
PREFIX_OTHER = f"[child_id:{OTHER_CHILD}] what is a ray?"
PREFIX_DEFAULT = f"[child_id:{DEFAULT_CHILD}] what is a ray?"


# ---------------------------------------------------------------------------
# Fake DB: only SQL the filter runs itself is the user_mappings lookup
# (get_child_full_profile / search_knowledge / profile_agent are stubbed at
# the ff module level, mirroring tests/test_profile_agent_hygiene.py).
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, mappings):
        self._mappings = mappings  # user_id -> [child_id, ...]
        self.executes = []
        self._rows = []

    def execute(self, sql, params=None):
        norm = " ".join(sql.split())
        self.executes.append((norm, params))
        if "USER_MAPPINGS" not in norm.upper():
            raise AssertionError(f"unhandled SQL in fake cursor: {norm[:120]}")
        user_id = params[0] if params else None
        self._rows = [{"child_id": c} for c in self._mappings.get(user_id, [])]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, mappings):
        self.cursor_obj = _FakeCursor(mappings)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


@pytest.fixture()
def filter_env(monkeypatch):
    """Fake DB plus recorded profile/prompt/RAG/assessment calls."""
    recorded = {"profiles": [], "rag_queries": [], "prompts": [],
                "assessed": [], "updated": []}
    conn = _FakeConn({OWUI_USER: [MY_CHILD]})

    def fake_profile(child_id):
        recorded["profiles"].append(child_id)
        return {"id": child_id, "name": f"kid-{child_id[-1]}"}

    def fake_rag(query):
        recorded["rag_queries"].append(query)
        return []

    def fake_prompt(base, profile, rag, speaker_type="child"):
        recorded["prompts"].append((profile, speaker_type))
        return "SYSTEM"

    def fake_assess(child_id, messages):
        recorded["assessed"].append(child_id)
        return [{"topic": "fractions", "score_delta": 5,
                 "new_status": "learning", "error_tags": []}]

    def fake_update(child_id, assessments, **kw):
        recorded["updated"].append(child_id)
        return len(assessments)

    monkeypatch.setattr(ff, "get_connection", lambda: conn)
    monkeypatch.setattr(ff, "get_child_full_profile", fake_profile)
    monkeypatch.setattr(ff, "search_knowledge", fake_rag)
    monkeypatch.setattr(ff, "build_system_prompt", fake_prompt)
    monkeypatch.setattr(ff, "should_update", lambda n: True)
    monkeypatch.setattr(ff, "already_assessed", lambda *a, **k: False)
    monkeypatch.setattr(ff, "assess_interaction", fake_assess)
    monkeypatch.setattr(ff, "update_profile", fake_update)
    monkeypatch.setattr(ff, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(ff, "SPEAKER_MODE", "child_only")
    monkeypatch.setattr(ff, "DEFAULT_CHILD_ID", DEFAULT_CHILD)
    recorded["conn"] = conn
    return recorded


def _inlet(user_id=None, content=PREFIX_OTHER):
    body = {"messages": [{"role": "user", "content": content}]}
    user = {"id": user_id} if user_id else None
    return body, user


def _warnings(caplog):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and "Unauthorized" in r.getMessage()]


# ---------------------------------------------------------------------------
# Inlet - unauthorized prefixes
# ---------------------------------------------------------------------------

def test_inlet_unmapped_user_with_prefix_gets_no_child_profile(filter_env, caplog):
    """Unmapped user + [child_id:OTHER] prefix: no profile of the prefix child
    may be embedded, the prefix text must not be stripped-and-used, and a
    warning with the child_id must be logged."""
    body, user = _inlet(user_id=None)
    with caplog.at_level(logging.WARNING, logger="pipeline.filter_function"):
        out = ff.Filter().inlet(body, user=user)

    assert filter_env["profiles"] == [DEFAULT_CHILD], \
        "the prefix child's profile must not be loaded - normal resolution only"
    assert out["messages"][1]["content"] == PREFIX_OTHER, \
        "an unauthorized prefix must be left in the message, not stripped-and-used"
    assert filter_env["prompts"] and filter_env["prompts"][0][0]["id"] == DEFAULT_CHILD
    warns = _warnings(caplog)
    assert warns and OTHER_CHILD in warns[0].getMessage(), \
        "an unauthorized prefix must log a warning containing the child_id"


def test_inlet_mapped_user_prefix_for_another_child_is_unauthorized(filter_env, caplog):
    """Mapped user + prefix for ANOTHER child: treated exactly like an
    unmapped user's unauthorized prefix - own child's profile, no strip."""
    body, user = _inlet(user_id=OWUI_USER)
    with caplog.at_level(logging.WARNING, logger="pipeline.filter_function"):
        out = ff.Filter().inlet(body, user=user)

    assert filter_env["profiles"] == [MY_CHILD], \
        "fallback resolution must pick the user's own mapped child, not the prefix child"
    assert out["messages"][1]["content"] == PREFIX_OTHER
    warns = _warnings(caplog)
    assert warns and OTHER_CHILD in warns[0].getMessage()


def test_inlet_default_child_prefix_allowed_only_when_user_would_resolve_there(filter_env, caplog):
    """DEFAULT_CHILD_ID stays addressable via prefix only because the normal
    path would resolve this (unmapped, child_only) user to it anyway."""
    body, user = _inlet(user_id=None, content=PREFIX_DEFAULT)
    with caplog.at_level(logging.WARNING, logger="pipeline.filter_function"):
        out = ff.Filter().inlet(body, user=user)

    assert filter_env["profiles"] == [DEFAULT_CHILD]
    assert out["messages"][1]["content"] == "what is a ray?", \
        "authorized prefix is stripped as before"
    assert not _warnings(caplog), "no unauthorized warning for the default child"


def test_inlet_default_child_prefix_denied_when_user_gets_adult_prompt(filter_env, monkeypatch):
    """In mapped_or_adult mode an unmapped user resolves to no child at all,
    so even the DEFAULT_CHILD_ID prefix must not grant child access."""
    monkeypatch.setattr(ff, "SPEAKER_MODE", "mapped_or_adult")
    body, user = _inlet(user_id=None, content=PREFIX_DEFAULT)
    out = ff.Filter().inlet(body, user=user)

    assert filter_env["profiles"] == [], "no child profile may be loaded"
    assert filter_env["prompts"] and filter_env["prompts"][0][1] == "adult"


# ---------------------------------------------------------------------------
# Inlet - authorized prefix keeps today's byte-identical behavior
# ---------------------------------------------------------------------------

def test_inlet_mapped_user_prefix_for_own_child_unchanged(filter_env, caplog):
    content = f"[child_id:{MY_CHILD}] what is a ray?"
    body, user = _inlet(user_id=OWUI_USER, content=content)
    with caplog.at_level(logging.WARNING, logger="pipeline.filter_function"):
        out = ff.Filter().inlet(body, user=user)

    assert filter_env["profiles"] == [MY_CHILD]
    assert out["messages"][1]["content"] == "what is a ray?", \
        "authorized prefix is stripped exactly as before"
    assert filter_env["rag_queries"] == ["what is a ray?"], \
        "RAG runs on the prefix-stripped query as before"
    assert filter_env["prompts"] and filter_env["prompts"][0][1] == "child"
    assert not _warnings(caplog), "no warning for an authorized prefix"


# ---------------------------------------------------------------------------
# Outlet - no child-scoped writes for unauthorized prefixes
# ---------------------------------------------------------------------------

def _outlet_body(last_user_content):
    """Five user messages (assessment boundary) with the last one carrying
    whatever content the outlet should see."""
    msgs = []
    for i in range(1, 5):
        msgs.append({"role": "user", "content": f"USER-{i}"})
        msgs.append({"role": "assistant", "content": f"A-{i}"})
    msgs.append({"role": "user", "content": last_user_content})
    return {"chat_id": "chat-f014", "messages": msgs}


def test_outlet_unmapped_user_prefix_child_not_written(filter_env):
    body = _outlet_body(PREFIX_OTHER)
    ff.Filter().outlet(copy.deepcopy(body), user=None)

    assert filter_env["assessed"] == [DEFAULT_CHILD], \
        "the outlet must assess the normally-resolved child, not the prefix child"
    assert filter_env["updated"] == [DEFAULT_CHILD]


def test_outlet_mapped_user_prefix_for_another_child_not_written(filter_env):
    body = _outlet_body(PREFIX_OTHER)
    ff.Filter().outlet(copy.deepcopy(body), user={"id": OWUI_USER})

    assert filter_env["assessed"] == [MY_CHILD]
    assert filter_env["updated"] == [MY_CHILD]


def test_outlet_mapped_user_prefix_for_own_child_still_written(filter_env, caplog):
    body = _outlet_body(f"[child_id:{MY_CHILD}] USER-5")
    with caplog.at_level(logging.WARNING, logger="pipeline.filter_function"):
        ff.Filter().outlet(copy.deepcopy(body), user={"id": OWUI_USER})

    assert filter_env["assessed"] == [MY_CHILD]
    assert filter_env["updated"] == [MY_CHILD]
    assert not _warnings(caplog)


def test_outlet_unauthorized_prefix_cannot_write_in_mapped_or_adult_mode(filter_env, monkeypatch):
    """mapped_or_adult + unmapped user: resolution yields no child, so an
    unauthorized prefix must end in no write at all."""
    monkeypatch.setattr(ff, "SPEAKER_MODE", "mapped_or_adult")
    body = _outlet_body(PREFIX_OTHER)
    ff.Filter().outlet(copy.deepcopy(body), user=None)

    assert filter_env["assessed"] == [], "no child-scoped write may happen"
    assert filter_env["updated"] == []

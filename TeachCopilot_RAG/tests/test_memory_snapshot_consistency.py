"""Audit F-060: two independently snapshotted reads could mix pre- and
post-correction content within one response.

- build_conversation_memory read the active-day window and the recent tail in
  two transactions; a message committed in between showed up in one only.
- session_summary_views built the last-session and last-seven-sessions views
  separately; a completion or correction in between made them disagree.
"""
import pytest

import local_app.hierarchical_memory as hm
import local_app.memory_service as ms


class WindowStore:
    def __init__(self):
        self.window_calls = []

    def get_learner_learning_timezone(self, learner_id):
        return "Asia/Tomsk"

    def get_own_conversation_message_windows(self, learner_id, actor_id, windows, **options):
        self.window_calls.append((learner_id, actor_id, windows, options))
        return [[] for _ in windows]

    def get_own_conversation_messages(self, *args, **kwargs):
        pytest.fail("separate window reads must not be used when one snapshot is available")


def test_conversation_memory_reads_both_windows_from_one_snapshot():
    store = WindowStore()
    result = hm.build_conversation_memory(store, "learner-a", "actor-a", context_budget_bytes=6000)
    assert result["available"] is False
    assert len(store.window_calls) == 1
    _, _, windows, options = store.window_calls[0]
    assert windows == [{"limit": None, "active_day_limit": hm.ACTIVE_DAY_LIMIT},
                       {"limit": hm.RECENT_MESSAGE_LIMIT}]
    assert options["learning_timezone"] == "Asia/Tomsk"


def test_windows_share_one_read_only_connection(disposable_store, monkeypatch):
    opened = []
    original = disposable_store._connect

    def counting_connect(*args, **kwargs):
        opened.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(disposable_store, "_connect", counting_connect)
    windows = disposable_store.get_own_conversation_message_windows(
        "learner-x", "actor-x", [{"limit": None, "active_day_limit": 7}, {"limit": 20}])
    assert windows == [[], []]
    assert len(opened) == 1


class SummaryStore:
    _connect = object()


def _view(*sessions):
    return {"sessions": [{"learning_session_id": sid, "input_fingerprint": fp} for sid, fp in sessions]}


def test_summary_views_retry_until_the_latest_session_agrees(monkeypatch):
    answers = iter([
        _view(("s1", "before-correction")),
        _view(("s1", "after-correction"), ("s0", "x")),   # correction landed in between
        _view(("s1", "after-correction")),
        _view(("s1", "after-correction"), ("s0", "x")),
    ])
    monkeypatch.setattr("local_app.session_summaries.get_or_build_summaries",
                        lambda store, learner_id, **kwargs: next(answers))
    views = ms.session_summary_views(SummaryStore(), "learner-a")
    assert views["last_session"]["sessions"][0]["input_fingerprint"] == "after-correction"
    assert views["last_seven_sessions"]["sessions"][0]["input_fingerprint"] == "after-correction"


def test_summary_views_fail_closed_when_the_snapshot_keeps_changing(monkeypatch):
    flip = {"n": 0}

    def changing(store, learner_id, **kwargs):
        flip["n"] += 1
        return _view(("s1", f"v{flip['n']}"))

    monkeypatch.setattr("local_app.session_summaries.get_or_build_summaries", changing)
    with pytest.raises(RuntimeError, match="session_summary_snapshot_changed"):
        ms.session_summary_views(SummaryStore(), "learner-a")


def test_summary_views_with_no_sessions_are_consistent(monkeypatch):
    monkeypatch.setattr("local_app.session_summaries.get_or_build_summaries",
                        lambda store, learner_id, **kwargs: _view())
    views = ms.session_summary_views(SummaryStore(), "learner-a")
    assert views["last_session"]["sessions"] == views["last_seven_sessions"]["sessions"] == []

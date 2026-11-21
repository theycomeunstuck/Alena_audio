"""Audit F-072: the stale-tail availability fallback crashed when a teacher
correction postdated the fallback version, and the response journals (full
child chat payloads) were created with the default umask."""
import os

import pytest

from local_app import chat_response_journal, memory_response_journal
from local_app.memory_service import restore_memory
from tests.test_dialogue_memory_context import TOPIC, prepare, source


def _note(source_id, text, target=None):
    row = source(source_id, text, role="teacher", question=None, session=source_id)
    if target:
        row["corrects_source_id"] = target
    return row


def test_fallback_serves_the_stale_tail_when_a_correction_postdates_it(monkeypatch):
    original = _note("original", "В журнале записано: ученик написал 33.")
    state, _ = prepare([original])
    versions = [{"id": "version-1", "method": "tail", "approved": True, "state": state}]
    memory_class = type(restore_memory(state, None, method="tail", topic_id=TOPIC))

    def saturated(self, lesson):
        raise ValueError("memory_writer_input_exceeds_context")

    monkeypatch.setattr(memory_class, "update", saturated)
    correction = _note("correction", "Исправляю журнал: в работе было 303.", "original")
    new_state, context = prepare([original, correction], versions=versions)

    assert new_state is None  # an old state is never resaved as a fresh version
    assert context["stale"] is True and context["memory_version_id"] == "version-1"
    assert context["update_failure"]["message"] == "memory_writer_input_exceeds_context"
    assert "original" in context["evidence_ids"] and "correction" not in context["evidence_ids"]
    assert {"event": "fallback_correction_after_version", "source_ids": ["correction"]} in context["selection_events"]


@pytest.mark.parametrize("module, cls", [(chat_response_journal, "ChatResponseJournal"),
                                         (memory_response_journal, "MemoryResponseJournal")])
def test_journal_files_are_made_private(tmp_path, monkeypatch, module, cls):
    calls = []
    real_chmod = os.chmod
    monkeypatch.setattr(module.os, "chmod", lambda path, mode: (calls.append((str(path), mode)), real_chmod(path, mode)))
    journal = getattr(module, cls)(tmp_path / "shared" / "journal.sqlite3")
    with journal._connect(write=True):
        pass
    assert (str(tmp_path / "shared" / "journal.sqlite3"), 0o600) in calls

"""CPU-only store contracts for reviewed learner-wide source memory."""
from copy import deepcopy

import pytest

from local_app.memory_store import MemoryIdempotencyConflict, MemoryStoreMixin
from local_app.personal_graph_contract import personal_graph_snapshot_fingerprint
from local_app.personal_memory_contract import PERSONAL_MEMORY_CONTEXT_PROTOCOL, PERSONAL_MEMORY_SELECTION_VERSION
from local_app.store import RevisionConflict


LEARNER = "learner-a"
TOPIC = "math"
OTHER_TOPIC = "programming"


def source(source_id, topic, text, *, corrects=None):
    return {"source_id":source_id, "session_id":None, "author_role":"teacher", "text":text,
            "topic_id":topic, "approved":True, "observed_on":"2026-11-01", "created_at":None,
            "corrects_source_id":corrects}


def context_for(sources, selected):
    selected = set(selected)
    return {"method":"personal_memory", "topic_id":TOPIC,
            "evidence_ids":sorted(selected),
            "sources":[{"source_id":s["source_id"], "role":s["author_role"], "text":s["text"],
                        "topic_id":s["topic_id"],
                        **({"corrects_source_id":s["corrects_source_id"]} if s["corrects_source_id"] else {})}
                       for s in sources if s["source_id"] in selected],
            "text":"reviewed context", "session_end_context":[]}


def state_for(sources, source_revision=4):
    fingerprint = personal_graph_snapshot_fingerprint(sources)
    return {"method":"personal_memory", "learner_id":LEARNER, "topic_id":TOPIC,
            "source_revision":source_revision, "source_ids":sorted(s["source_id"] for s in sources),
            "source_context_protocol":PERSONAL_MEMORY_CONTEXT_PROTOCOL,
            "selection_version":PERSONAL_MEMORY_SELECTION_VERSION, "retrieval_mode":"hybrid",
            "model_identities":{"chat":{"name":"gemma4:12b", "digest":"a" * 64},
                                "embedding":{"name":"bge-m3", "digest":"sha256:" + "b" * 64}},
            "selection_recipe_fingerprint":"c" * 64,
            "archive_snapshot_fingerprint":fingerprint, "source_snapshot_id":"personal-memory-" + fingerprint}


class Cursor:
    def __init__(self, store):
        self.store = store
        self.query = ""
        self.one = None
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, values=None):
        self.query = query
        self.one, self.rows = None, []
        if "SELECT * FROM teachcopilot.memory_versions WHERE id=" in query:
            if values == (self.store.version["id"], LEARNER):
                self.one = deepcopy(self.store.version)
        elif "SELECT * FROM teachcopilot.memory_versions WHERE learner_id=" in query:
            self.rows = [deepcopy(self.store.version)]
        elif "FROM teachcopilot.memory_sources s" in query:
            topic = values[-1] if len(values) == 4 else None
            self.rows = [{**deepcopy(s), "content":s["text"], "preceding_assistant_text":None,
                          "preceding_assistant_truncated":False}
                         for s in self.store.sources if topic is None or s["topic_id"] == topic]
        elif "UPDATE teachcopilot.memory_versions SET approved=TRUE" in query:
            self.store.version.update(approved=True, context=deepcopy(values[1].adapted), base_revision=values[2])
            self.one = deepcopy(self.store.version)
        elif "INSERT INTO teachcopilot.memory_versions" in query:
            self.store.version = {"id":"version-1", "learner_id":values[0], "topic_id":values[1],
                                  "base_revision":values[2], "source_revision":values[3],
                                  "chat_source_revision":values[4], "state":deepcopy(values[5].adapted),
                                  "context":deepcopy(values[6].adapted), "approved":values[7]}
            self.one = deepcopy(self.store.version)
        elif "INSERT INTO teachcopilot.next_steps" in query:
            self.one = {"id":"step-1", "learner_id":values[0], "topic_id":values[1],
                        "memory_version_id":values[6], "method":values[8]}
        else:
            raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return Cursor(self.store)


class Store(MemoryStoreMixin):
    def __init__(self):
        self.sources = [source("math-1",TOPIC,"Original math note"),
                        source("code-1",OTHER_TOPIC,"Original coding note"),
                        source("code-2",OTHER_TOPIC,"Corrected coding note",corrects="code-1")]
        self.learner = {"revision":1, "memory_source_revision":4, "memory_chat_source_revision":0}
        self.version = {"id":"version-1", "learner_id":LEARNER, "topic_id":TOPIC,
                        "base_revision":1, "source_revision":4, "approved":False,
                        "state":state_for(self.sources), "context":context_for(self.sources,["code-1","code-2"])}
        self.reference_scopes = []

    def _connect(self):
        return Connection(self)

    def _memory_learner(self, cur, learner_id, expected_revision=None, lock="UPDATE"):
        assert learner_id == LEARNER
        if expected_revision is not None and expected_revision != self.learner["revision"]:
            raise RevisionConflict("learner revision changed")
        return deepcopy(self.learner)

    def _current_personal_graph_sources(self, cur, learner_id):
        assert learner_id == LEARNER
        return deepcopy(self.sources), 0

    def _latest_session_end_context(self, cur, learner_id, topic_id=None):
        assert learner_id == LEARNER
        return {}

    def _assert_memory_references(self, cur, learner_id, value, *, topic_id=None, all_topics=False):
        self.reference_scopes.append((learner_id,topic_id,all_topics))


@pytest.mark.parametrize("full_archive", [False, True])
def test_approval_records_explicit_access_and_is_idempotent(full_archive):
    store = Store()
    fingerprint = store.version["state"]["archive_snapshot_fingerprint"]
    options = {"full_archive_access":full_archive,
               "expected_archive_fingerprint":fingerprint if full_archive else None}
    approved = store.approve_memory_version(LEARNER,"version-1","teacher",1,**options)
    access = approved["context"]["source_access"]
    assert access["mode"] == ("full_archive" if full_archive else "selected_context")
    assert set(access["source_ids"]) == ({"math-1","code-1","code-2"} if full_archive else {"code-1","code-2"})
    assert access["snapshot_fingerprint"] == (fingerprint if full_archive else None)
    assert store.approve_memory_version(LEARNER,"version-1","teacher",1,**options)["approved"]
    with pytest.raises(MemoryIdempotencyConflict):
        store.approve_memory_version(LEARNER,"version-1","teacher",1,full_archive_access=not full_archive,
                                     expected_archive_fingerprint=fingerprint if not full_archive else None)


@pytest.mark.parametrize("full_archive", [False, True])
def test_approval_rejects_changed_archive_even_when_ids_and_revision_match(full_archive):
    store = Store()
    fingerprint = store.version["state"]["archive_snapshot_fingerprint"]
    store.sources[0]["text"] = "Changed text, same identifier"
    with pytest.raises(RevisionConflict, match="Архив изменился"):
        store.approve_memory_version(LEARNER,"version-1","teacher",1,
            full_archive_access=full_archive, expected_archive_fingerprint=fingerprint if full_archive else None)
    assert not store.version["approved"]


def test_full_archive_approval_requires_the_reviewed_fingerprint():
    store = Store()
    with pytest.raises(RevisionConflict, match="Архив изменился"):
        store.approve_memory_version(LEARNER,"version-1","teacher",1,full_archive_access=True,
                                     expected_archive_fingerprint="a" * 64)
    assert not store.version["approved"]


@pytest.mark.parametrize("change", ["revision", "protocol", "learner", "topic", "snapshot_id", "model",
                                     "retrieval", "selection", "recipe",
                                     "missing_link", "duplicate_evidence", "altered_projection"])
def test_invalid_personal_memory_cannot_be_approved(change):
    store = Store()
    state, context = store.version["state"], store.version["context"]
    if change == "revision":
        store.learner["memory_source_revision"] += 1
    elif change == "protocol":
        state["source_context_protocol"] = "old"
    elif change == "learner":
        state["learner_id"] = "learner-b"
    elif change == "topic":
        state["topic_id"] = OTHER_TOPIC
    elif change == "snapshot_id":
        state["source_snapshot_id"] = "unrelated"
    elif change == "model":
        state["model_identities"]["chat"]["digest"] = "unverified"
    elif change == "retrieval":
        state["retrieval_mode"] = "graph_hybrid"
    elif change == "selection":
        state["selection_version"] = "old"
    elif change == "recipe":
        state["selection_recipe_fingerprint"] = "invalid"
    elif change == "missing_link":
        store.version["context"] = context_for(store.sources,["code-1"])
    elif change == "duplicate_evidence":
        context["evidence_ids"].append(context["evidence_ids"][0])
    else:
        context["sources"][0]["text"] = "Injected text"
    with pytest.raises((RevisionConflict,ValueError)):
        store.approve_memory_version(LEARNER,"version-1","teacher",1)
    assert not store.version["approved"]


def test_draft_save_checks_snapshot_and_accepts_sources_from_other_topics():
    store = Store()
    version = store.save_memory_version(LEARNER,TOPIC,store.version["state"],base_revision=1,
        source_revision=4,context=store.version["context"])
    assert not version["approved"]
    assert store.reference_scopes == [(LEARNER,TOPIC,True)]
    with pytest.raises(ValueError, match="explicit_approval"):
        store.save_memory_version(LEARNER,TOPIC,version["state"],base_revision=1,source_revision=4,
                                  context=version["context"],approved=True)
    store.sources[1]["text"] = "Changed"
    with pytest.raises(RevisionConflict, match="archive changed"):
        store.save_memory_version(LEARNER,TOPIC,version["state"],base_revision=1,source_revision=4,
                                  context=version["context"])


def test_provider_selection_is_accepted_with_correction_closure():
    from local_app.personal_memory import select_ranked_source_context

    store = Store()
    snapshot = {"learner_id":LEARNER, "sources":deepcopy(store.sources)}
    context = select_ranked_source_context(["code-1"],snapshot,LEARNER,TOPIC,
                                            {s["source_id"] for s in store.sources},12000)
    assert set(context["evidence_ids"]) == {"code-1","code-2"}
    version = store.save_memory_version(LEARNER,TOPIC,store.version["state"],base_revision=1,
                                         source_revision=4,context=context)
    approved = store.approve_memory_version(LEARNER,version["id"],"teacher",1)
    assert set(approved["context"]["source_access"]["source_ids"]) == {"code-1","code-2"}


@pytest.mark.parametrize("method,all_topics", [("personal_memory",True),("personal_graph",True),("tail",False)])
def test_next_step_checks_session_context_in_the_method_scope(method,all_topics):
    store = Store()
    store.save_next_step(LEARNER,TOPIC,{"action":"review"},base_revision=1,source_revision=4,
                         context={},method=method)
    assert store.reference_scopes == [(LEARNER,TOPIC,all_topics)]


@pytest.mark.parametrize("change", ["text", "state_revision", "grant", "protocol"])
def test_read_marks_invalid_approval_stale_without_disclosing_learner_wide_sources(change):
    store = Store()
    store.approve_memory_version(LEARNER,"version-1","teacher",1)
    assert not store.get_learner_memory(LEARNER,TOPIC)["versions"][0]["stale"]
    if change == "text":
        store.sources[0]["text"] = "Edited without changing IDs"
    elif change == "state_revision":
        store.version["state"]["source_revision"] += 1
    elif change == "grant":
        store.version["context"]["source_access"]["source_ids"] = ["code-1"]
    else:
        store.version["state"]["source_context_protocol"] = "old"
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    assert snapshot["versions"][0]["stale"]
    assert "personal_graph_sources" not in snapshot
    assert {s["source_id"] for s in snapshot["sources"]} == {"math-1"}


@pytest.mark.parametrize("full_archive", [False,True])
def test_approved_memory_survives_active_turn_but_not_a_changed_archive(full_archive):
    store = Store()
    fingerprint = store.version["state"]["archive_snapshot_fingerprint"]
    options = {"full_archive_access":full_archive,
               "expected_archive_fingerprint":fingerprint if full_archive else None}
    store.approve_memory_version(LEARNER,"version-1","teacher",1,**options)

    # An unfinished chat turn changes the global revision but is not yet in the archive.
    store.learner["memory_source_revision"] += 1
    assert not store.get_learner_memory(LEARNER,TOPIC)["versions"][0]["stale"]
    assert store.approve_memory_version(LEARNER,"version-1","teacher",1,**options)["approved"]

    # Completing that lesson exposes a new eligible source in the archive.
    store.sources.append(source("active-1",OTHER_TOPIC,"Completed turn"))
    store.learner["memory_source_revision"] += 1
    assert store.get_learner_memory(LEARNER,TOPIC)["versions"][0]["stale"]


def test_draft_approval_still_rejects_a_changed_global_revision():
    store = Store()
    store.learner["memory_source_revision"] += 1
    assert store.get_learner_memory(LEARNER,TOPIC)["versions"][0]["stale"]
    with pytest.raises(RevisionConflict,match="no longer matches"):
        store.approve_memory_version(LEARNER,"version-1","teacher",1)


@pytest.mark.parametrize("kind,provider", [("chat","local"), ("memory_update","local"), ("next_step","external")])
def test_enqueue_rejects_personal_memory_outside_local_teacher_next_step(kind,provider):
    with pytest.raises(ValueError, match="personal_memory_is_local_teacher_next_step_only"):
        Store()._enqueue_memory_job(None,LEARNER,TOPIC,kind,method="personal_memory",provider=provider)

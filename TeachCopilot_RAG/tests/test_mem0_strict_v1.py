import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from research_memory.personal_memory_lab.mem0_strict_v1 import (StrictMem0LLM, validate_extraction,
    WriteLatch, guard_mem0_writes, UncertainMem0Write)
from research_memory.personal_memory_lab.backends import Mem0Adapter


@pytest.mark.parametrize("value", [
    {"memory": []}, {"memory": [{"text": "fact"}]},
    {"memory": [{"id": "0", "text": "fact", "attributed_to": "user", "linked_memory_ids": ["1"]}]},
])
def test_valid_shapes(value):
    validate_extraction(value)


@pytest.mark.parametrize("value", [None, [], {}, {"memory": "fact"}, {"memory": ["fact"]},
    {"memory": [{"text": "a"}, "b", []]}, {"memory": [{}]}, {"memory": [{"text": " "}]},
    {"memory": [{"text": 4}]}, {"memory": [{"text": "a", "id": 4}]},
    {"memory": [{"text": "a", "attributed_to": None}]},
    {"memory": [{"text": "a", "linked_memory_ids": [1]}]},
    {"memory": [{"text": "a", "extra": "x"}]}, {"memory": [], "extra": 1}])
def test_invalid_shapes(value):
    with pytest.raises(ValueError):
        validate_extraction(value)


def test_strict_llm_uses_schema_and_keeps_synthetic_raw_only_when_enabled():
    class Runtime:
        model, context = "fake", 100
        def _post(self, route, payload, purpose):
            assert payload["format"]["properties"]["memory"]["items"]["type"] == "object"
            return {"done": True, "message": {"content": '{"memory":["synthetic"]}'}}
    events = []
    llm = StrictMem0LLM(Runtime(), synthetic_event_sink=events.append)
    with pytest.raises(ValueError):
        llm.generate_response([], response_format={"type": "json_object"})
    assert events[0]["raw_response"] == '{"memory":["synthetic"]}'
    assert llm.last_diagnostic == {"stage": "extraction", "code": "invalid_memory_row", "exception_type": "ValueError"}


def test_failure_records_boundaries_and_blocks_retry():
    lesson = {"lesson_id": "lesson", "observed_on": "2026-01-01", "turns": [
        {"turn_id": str(i), "role": "teacher" if i % 2 == 0 else "student", "text": "x" * 450}
        for i in range(12)]}
    class Archive:
        state = None
        def lessons(self, *args): return [lesson]
        def claim(self, *args):
            if self.state: raise RuntimeError("uncertain")
            self.state = "running"
            return True
        def finish(self, *args, metadata=None):
            self.state = "uncertain" if args[-1] else "complete"
            self.metadata = metadata
    class Memory:
        calls = []
        def add(self, text, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 2: raise RuntimeError("postgresql://secret")
            return {"results": []}
    archive, memory = Archive(), Memory()
    memory._strict_write_latch = WriteLatch()
    adapter = Mem0Adapter(memory, archive, extraction="strict_v1")
    with pytest.raises(RuntimeError): adapter.ingest("run", "child", lesson)
    assert len(archive.metadata["attempted_parts"]) == 2
    assert len(archive.metadata["completed_parts"]) == 1
    assert memory.calls[0]["metadata"]["first_turn_id"] == "0"
    assert "secret" not in json.dumps(archive.metadata)
    with pytest.raises(RuntimeError): adapter.ingest("run", "child", lesson)
    assert len(memory.calls) == 2


def test_real_mem0_pipeline_rejects_mixed_rows_before_embedding():
    from mem0.memory.main import Memory
    from mem0.exceptions import LLMError
    class Runtime:
        model, context = "fake", 100
        def _post(self, *args):
            return {"done": True, "message": {"content": '{"memory":[{"text":"a"},"bad",[]]}'}}
    memory = object.__new__(Memory)
    memory.custom_instructions = ""
    memory.db = SimpleNamespace(get_last_messages=lambda *a, **k: [], save_messages=lambda *a, **k: None)
    memory.embedding_model = SimpleNamespace(embed=lambda *a, **k: [0.0] * 3)
    memory.vector_store = SimpleNamespace(search=lambda **k: [])
    memory.llm = StrictMem0LLM(Runtime())
    with pytest.raises(LLMError):
        memory._add_to_vector_store([{"role": "user", "content": "synthetic"}], {},
                                    {"user_id": "synthetic", "run_id": "synthetic"}, True)


def test_real_mem0_pipeline_accepts_valid_row():
    from mem0.memory.main import Memory
    class Runtime:
        model, context = "fake", 100
        def _post(self, *args):
            return {"done": True, "message": {"content": '{"memory":[{"text":"synthetic fact","attributed_to":"user"}]}'}}
    inserted = []
    memory = object.__new__(Memory)
    memory.custom_instructions = ""
    memory.db = SimpleNamespace(get_last_messages=lambda *a, **k: [], save_messages=lambda *a, **k: None,
                                batch_add_history=lambda rows: None)
    memory.embedding_model = SimpleNamespace(embed=lambda *a, **k: [0.0] * 3,
                                             embed_batch=lambda texts, *a: [[0.0] * 3 for _ in texts])
    memory.vector_store = SimpleNamespace(search=lambda **k: [], insert=lambda **k: inserted.append(k))
    memory.llm = StrictMem0LLM(Runtime())
    memory.api_version = "v3"
    with patch("mem0.memory.main.lemmatize_for_bm25", side_effect=lambda value: value), \
         patch("mem0.memory.main.extract_entities_batch", return_value=[[]]), \
         patch("mem0.memory.main.capture_event", return_value=None):
        result = memory._add_to_vector_store([{"role": "user", "content": "synthetic"}], {},
                                             {"user_id": "synthetic", "run_id": "synthetic"}, True)
    assert len(inserted) == 1
    assert result[0]["memory"] == "synthetic fact"


def test_real_mem0_batch_timeout_never_retries_external_write():
    from mem0.memory.main import Memory
    class Runtime:
        model, context = "fake", 100
        def _post(self, *args):
            return {"done": True, "message": {"content": json.dumps(
                {"memory": [{"text": "first fact"}, {"text": "second fact"}]})}}
    calls, history_writes = [], []
    def insert(**kwargs):
        calls.append(len(kwargs["ids"]))
        if len(calls) == 1: raise TimeoutError("unknown batch commit outcome")
    memory = object.__new__(Memory)
    memory.custom_instructions, memory.api_version = "", "v3"
    memory.db = SimpleNamespace(get_last_messages=lambda *a, **k: [],
        save_messages=lambda *a, **k: history_writes.append("save"),
        batch_add_history=lambda *a, **k: history_writes.append("batch"))
    memory.embedding_model = SimpleNamespace(embed=lambda *a, **k: [0.0] * 3,
        embed_batch=lambda texts, *a: [[0.0] * 3 for _ in texts])
    memory.vector_store = SimpleNamespace(search=lambda **k: [], insert=insert)
    memory.llm = StrictMem0LLM(Runtime())
    latch = guard_mem0_writes(memory)
    with patch("mem0.memory.main.lemmatize_for_bm25", side_effect=lambda value: value), \
         patch("mem0.memory.main.extract_entities_batch", return_value=[[], []]), \
         patch("mem0.memory.main.capture_event", return_value=None):
        with pytest.raises(UncertainMem0Write):
            memory._add_to_vector_store([{"role": "user", "content": "synthetic"}], {},
                                       {"user_id": "synthetic", "run_id": "synthetic"}, True)
    assert calls == [2]
    assert history_writes == []
    assert latch.diagnostic() == {"stage": "stateful_write", "operation": "vector_store.insert",
                                  "exception_type": "TimeoutError", "code": "uncertain_write"}
    with pytest.raises(UncertainMem0Write): memory.vector_store.insert(ids=["another"])
    assert calls == [2]


def test_adapter_catches_swallowed_write_failure():
    lesson = {"lesson_id": "lesson", "observed_on": "2026-01-01", "turns": [
        {"turn_id": "t1", "role": "student", "text": "synthetic"}]}
    class Archive:
        def lessons(self, *a): return [lesson]
        def claim(self, *a): return True
        def finish(self, *a, metadata=None): self.args, self.metadata = a, metadata
    class Memory:
        def __init__(self): self._strict_write_latch = WriteLatch()
        def add(self, *a, **k):
            try:
                self._strict_write_latch.call("entity_store.insert", lambda: (_ for _ in ()).throw(TimeoutError()))
            except TimeoutError:
                pass
            return {"results": []}
    archive = Archive()
    with pytest.raises(UncertainMem0Write): Mem0Adapter(Memory(), archive, "strict_v1").ingest("run", "child", lesson)
    assert archive.args[-1] == "UncertainMem0Write"
    assert archive.metadata["completed_parts"] == []
    assert archive.metadata["diagnostic"]["operation"] == "entity_store.insert"


def test_lazy_collection_creation_failure_blocks_following_writes():
    from research_memory.personal_memory_lab.mem0_strict_v1 import GuardedStore
    calls = []
    class Store:
        def _ensure_collection(self):
            calls.append("ensure")
            raise TimeoutError("unknown collection commit outcome")
        def list(self, **kwargs): calls.append("list")
        def insert(self, **kwargs): calls.append("insert")
    latch = WriteLatch()
    store = GuardedStore(Store(), latch, "entity_store")
    with pytest.raises(TimeoutError): store.list()
    with pytest.raises(UncertainMem0Write): store.insert()
    assert calls == ["ensure"]
    assert latch.diagnostic()["operation"] == "entity_store.ensure_collection"

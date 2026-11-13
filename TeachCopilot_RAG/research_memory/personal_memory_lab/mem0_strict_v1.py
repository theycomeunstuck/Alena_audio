"""Opt-in Mem0 2.2.1 extraction response contract, strict_v1.

The schema checks shape only, not the truth or attribution of extracted facts.
Raw model output is sent only to an explicitly supplied synthetic-test sink.
"""
import json

MEMORY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["memory"],
    "properties": {"memory": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text"],
        "properties": {"id": {"type": "string"}, "text": {"type": "string", "minLength": 1},
                       "attributed_to": {"type": "string"},
                       "linked_memory_ids": {"type": "array", "items": {"type": "string"}}}}}}
}


def validate_extraction(value):
    if not isinstance(value, dict) or set(value) != {"memory"} or not isinstance(value["memory"], list):
        raise ValueError("invalid_memory_container")
    for row in value["memory"]:
        if not isinstance(row, dict) or not set(row) <= {"id", "text", "attributed_to", "linked_memory_ids"}:
            raise ValueError("invalid_memory_row")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("invalid_memory_text")
        if any(key in row and not isinstance(row[key], str) for key in ("id", "attributed_to")):
            raise ValueError("invalid_memory_source_field")
        if "linked_memory_ids" in row and (not isinstance(row["linked_memory_ids"], list)
                or not all(isinstance(item, str) for item in row["linked_memory_ids"])):
            raise ValueError("invalid_memory_links")


class StrictMem0LLM:
    def __init__(self, runtime, *, synthetic_event_sink=None):
        self.runtime = runtime
        self.synthetic_event_sink = synthetic_event_sink
        self.last_diagnostic = None

    def generate_response(self, messages, response_format=None, tools=None, **kwargs):
        self.last_diagnostic = None
        if tools:
            raise ValueError("Mem0 tools are not enabled")
        payload = {"model": self.runtime.model, "stream": False, "think": False,
                   "options": {"num_ctx": self.runtime.context, "num_predict": 2048,
                               "temperature": 0, "seed": 42}, "messages": messages}
        if response_format:
            payload["format"] = MEMORY_SCHEMA
        content = None
        try:
            result = self.runtime._post("/api/chat", payload, "mem0_llm")
            content = result.get("message", {}).get("content")
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise ValueError("incomplete_response")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty_response")
            if response_format:
                try:
                    parsed = json.loads(content)
                except (TypeError, ValueError):
                    raise ValueError("invalid_json") from None
                validate_extraction(parsed)
        except Exception as exc:
            # Only controlled error codes are retained, never exception text from external services.
            allowed = {"incomplete_response", "empty_response", "invalid_json", "invalid_memory_container",
                       "invalid_memory_row", "invalid_memory_text", "invalid_memory_source_field", "invalid_memory_links"}
            code = str(exc) if type(exc) is ValueError and str(exc) in allowed else "external_error"
            self.last_diagnostic = {"stage": "extraction", "code": code, "exception_type": type(exc).__name__}
            if self.synthetic_event_sink is not None:
                self.synthetic_event_sink({**self.last_diagnostic, "raw_response": content})
            raise
        if self.synthetic_event_sink is not None:
            self.synthetic_event_sink({"stage": "extraction", "code": "ok", "raw_response": content})
        return content


class UncertainMem0Write(RuntimeError):
    """A stateful Mem0 write failed and its outcome is unknown."""


class WriteLatch:
    def __init__(self):
        self.uncertain = False
        self.operation = None
        self.exception_type = None

    def check(self):
        if self.uncertain:
            raise UncertainMem0Write("Mem0 write outcome is uncertain; rebuild with a new instance")

    def call(self, operation, function, *args, **kwargs):
        self.check()
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.uncertain = True
            self.operation = operation
            self.exception_type = type(exc).__name__
            raise

    def diagnostic(self):
        if not self.uncertain:
            return None
        return {"stage": "stateful_write", "operation": self.operation,
                "exception_type": self.exception_type, "code": "uncertain_write"}


class GuardedStore:
    """Guard Mem0 2.2.1 store and SQLite write entry points with one latch."""
    WRITES = frozenset({"insert", "update", "delete", "delete_col", "reset", "create_col",
                        "save_messages", "batch_add_history", "add_history"})

    def __init__(self, store, latch, name):
        self._store, self._latch, self._name = store, latch, name

    # PGVector 2.2.1 performs lazy collection creation even in these reads.
    ENSURES_COLLECTION = frozenset({"insert", "update", "delete", "reset", "search", "search_batch",
                                    "keyword_search", "get", "list", "col_info"})

    def __getattr__(self, name):
        target = getattr(self._store, name)
        if not callable(target) or (name not in self.WRITES and name not in self.ENSURES_COLLECTION):
            return target
        def guarded(*args, **kwargs):
            ensure = getattr(self._store, "_ensure_collection", None)
            if name in self.ENSURES_COLLECTION and callable(ensure):
                self._latch.call(self._name + ".ensure_collection", ensure)
            if name in self.WRITES:
                return self._latch.call(self._name + "." + name, target, *args, **kwargs)
            return target(*args, **kwargs)
        return guarded


class LazyEntityStore:
    """Keep Mem0 entity store initialization lazy and share the write latch."""
    def __init__(self, memory, latch):
        self._memory, self._latch, self._wrapped = memory, latch, None

    def __getattr__(self, name):
        if self._wrapped is None:
            self._latch.check()
            self._memory._entity_store = None
            try:
                # Creation may itself issue a stateful collection write.
                store = self._latch.call("entity_store.initialize", lambda: self._memory.entity_store)
                self._wrapped = GuardedStore(store, self._latch, "entity_store")
            finally:
                self._memory._entity_store = self
        return getattr(self._wrapped, name)


def guard_mem0_writes(memory):
    latch = WriteLatch()
    memory.vector_store = GuardedStore(memory.vector_store, latch, "vector_store")
    memory.db = GuardedStore(memory.db, latch, "history_db")
    if getattr(memory, "_entity_store", None) is None:
        memory._entity_store = LazyEntityStore(memory, latch)
    else:
        memory._entity_store = GuardedStore(memory._entity_store, latch, "entity_store")
    memory._strict_write_latch = latch
    return latch

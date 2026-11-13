"""Version the teacher-approved experimental graph context contract."""

import hashlib
import json

PERSONAL_GRAPH_CONTEXT_PROTOCOL = "personal-graph-context-v2"


def personal_graph_snapshot_fingerprint(sources):
    """Hash the ordered, eligible source snapshot and its attached contexts.

    The fingerprint covers the source text and metadata used by the graph,
    assistant questions attached to student turns, correction links, and the
    latest completed-session ending attached to a source. Volatile database
    timestamps and display-only fields are intentionally excluded.
    """
    if not isinstance(sources, list):
        raise ValueError("personal graph sources must be a list")

    def context_projection(value, fields):
        if not isinstance(value, dict):
            return None
        return {key: value[key] for key in fields if key in value}

    canonical = []
    for order, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError("personal graph source must be an object")
        source_id = str(source.get("source_id") or source.get("id") or "")
        role = source.get("author_role") or source.get("role")
        text = source.get("text") if isinstance(source.get("text"), str) else source.get("content")
        if not source_id or role not in {"teacher", "student"} or not isinstance(text, str):
            raise ValueError("personal graph source snapshot is incomplete")
        observed = str(source.get("observed_on") or source.get("created_at") or "")[:10]
        canonical.append({
            "order": order,
            "source_id": source_id,
            "session_id": str(source.get("session_id") or source.get("learning_session_id") or ""),
            "role": role,
            "text": text,
            "observed_on": observed,
            "topic_id": str(source.get("topic_id") or "general"),
            "approved": source.get("approved") is True,
            "corrects_source_id": str(source["corrects_source_id"]) if source.get("corrects_source_id") else None,
            "dialogue_context": context_projection(source.get("dialogue_context"),
                ("role", "usage", "text", "excerpt_mode", "truncated")),
            "session_end_context": context_projection(source.get("session_end_context"),
                ("assistant_source_id", "session_id", "role", "usage", "text", "excerpt_mode", "truncated")),
            "session_end_context_omitted": context_projection(source.get("session_end_context_omitted"),
                ("reason", "limit_chars")),
        })
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

# Application-only UTF-8 byte limits, not the research token budget. Leave room
# between selection and the final limit for linked corrections and context.
PERSONAL_GRAPH_SELECTION_BYTE_BUDGET = 10000
PERSONAL_GRAPH_MEMORY_BYTE_BUDGET = 12000
PERSONAL_GRAPH_32K_SELECTION_BYTE_BUDGET = 20000
PERSONAL_GRAPH_32K_MEMORY_BYTE_BUDGET = 24000


def personal_graph_budgets(context_tokens):
    if context_tokens == 16384:
        return PERSONAL_GRAPH_SELECTION_BYTE_BUDGET, PERSONAL_GRAPH_MEMORY_BYTE_BUDGET
    if context_tokens == 32768:
        return PERSONAL_GRAPH_32K_SELECTION_BYTE_BUDGET, PERSONAL_GRAPH_32K_MEMORY_BYTE_BUDGET
    raise ValueError("personal graph requires a 16k or 32k model context")


def approved_memory_fits_budget(method, text, context_tokens=None):
    """Use the same whole-memory limit for generation and query resolution."""
    if not isinstance(text, str):
        return False
    if method == "personal_memory":
        from local_app.personal_memory_contract import personal_memory_budgets
        try:
            _, limit = personal_memory_budgets(32768 if context_tokens is None else context_tokens)
        except ValueError:
            return False
        return len(text.encode("utf-8")) <= limit
    if method == "personal_graph":
        if context_tokens is None:
            limit = PERSONAL_GRAPH_32K_MEMORY_BYTE_BUDGET
        else:
            try:
                _, limit = personal_graph_budgets(context_tokens)
            except ValueError:
                return False
        return len(text.encode("utf-8")) <= limit
    return len(text) <= 4000

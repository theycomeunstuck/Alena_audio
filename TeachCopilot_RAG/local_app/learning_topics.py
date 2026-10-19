"""Explicit curriculum-topic bindings, independent of a child's message text."""

SUBJECTS = frozenset({"math", "programming", "python", "scratch"})


def canonical_topic_subject(topic_id: str | None, catalog: dict) -> str | None:
    """Resolve only existing curriculum entries; unknown legacy IDs stay opaque."""
    if not isinstance(topic_id, str) or topic_id not in catalog:
        return None
    if topic_id.startswith("programming.python."):
        return "python"
    if topic_id.startswith("programming.scratch."):
        return "scratch"
    prefix = topic_id.split(".", 1)[0]
    return prefix if prefix in SUBJECTS else None


def bind_session_subject(topic_id: str | None, requested: object, catalog: dict) -> str | None:
    """A canonical session controls retrieval; metadata can only narrow compatibly."""
    subject = canonical_topic_subject(topic_id, catalog)
    if subject is None:
        return None
    compatible = {subject}
    if subject in {"python", "scratch"}:
        compatible.add("programming")
    if requested is not None and (not isinstance(requested, str) or requested not in compatible):
        raise ValueError("Предмет сообщения не совпадает с темой занятия. Завершите занятие, чтобы выбрать другую тему.")
    return subject

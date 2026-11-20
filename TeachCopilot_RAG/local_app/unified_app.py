"""Application-side glue for the unified memory backend facade.

Client B (application) depends only on the documented facade
``local_app.unified_memory.UnifiedMemoryService(store, runtime, *,
model_call=None)`` and its exceptions. The facade module arrives with the
peer's ready commit; until then every integrated endpoint reports explicit
unavailability (503) instead of a fake success. No competing production
implementation lives here.

Error mapping follows SHARED_CONTRACT.md: UnifiedMemoryConflict → 409,
UnifiedMemoryUnavailable → 503, UnifiedMemoryValidationError → 422,
PermissionError → 403, LookupError → 404. Before the facade module is merged,
the mapping recognizes the contract exception names directly, so injected
test fakes and the merged backend behave identically.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("teachcopilot")

# Job kinds the unified backend owns the storage for; B owns their dispatch.
UNIFIED_JOB_KINDS = ("lesson_generate", "session_process")

# Contract exception names and their HTTP statuses. Builtins keep their
# ordinary meaning; the three Unified* names belong to the facade contract.
CONTRACT_ERROR_STATUS = {
    "UnifiedMemoryConflict": 409,
    "UnifiedMemoryUnavailable": 503,
    "UnifiedMemoryValidationError": 422,
    "PermissionError": 403,
    "LookupError": 404,
}

_FACADE_IMPORT_ERROR: str | None = None


def facade_import_error() -> str | None:
    """The last facade import failure reason, for honest diagnostics."""
    return _FACADE_IMPORT_ERROR


def resolve_unified_memory(store, runtime, *, model_call=None):
    """Build the peer's UnifiedMemoryService, or return None when absent.

    Import stays lazy: importing the application never requires the peer's
    module. The result is not cached so a merged facade is picked up without
    a process restart in tests; production constructs the app once anyway.
    """
    global _FACADE_IMPORT_ERROR
    try:
        from local_app.unified_memory import UnifiedMemoryService
    except ImportError as exc:
        _FACADE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _FACADE_IMPORT_ERROR = None
    return UnifiedMemoryService(store, runtime, model_call=model_call)


def unified_http_error(exc: Exception):
    """Map contract exceptions onto the shared HTTP status codes."""
    from fastapi import HTTPException

    from local_app.store import RevisionConflict
    if isinstance(exc, RevisionConflict):
        raise HTTPException(status_code=409, detail="revision changed; reload before continuing") from exc
    status = CONTRACT_ERROR_STATUS.get(type(exc).__name__)
    if status is not None:
        detail = str(exc) or type(exc).__name__
        raise HTTPException(status_code=status, detail=detail) from exc
    if isinstance(exc, (ValueError, TypeError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc

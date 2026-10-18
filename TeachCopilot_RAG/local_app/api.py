"""Authenticated local API for learner cards, lesson work and local RAG."""
import logging
import secrets
import time
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool

from local_app.auth import hash_password, hash_session_token, new_session_token, session_expiry, verify_password

logger = logging.getLogger("teachcopilot")
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _is_dependency_unavailable(exc: Exception) -> bool:
    try:
        import psycopg2
        import requests
        return isinstance(exc, (psycopg2.Error, requests.RequestException, ConnectionError, TimeoutError))
    except ImportError:
        return isinstance(exc, (ConnectionError, TimeoutError))


def _is_generation_capacity(exc: Exception) -> bool:
    from local_app.generation_scheduler import GenerationCapacityError
    return isinstance(exc, GenerationCapacityError)


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _account_id(account: dict) -> str:
    return str(account.get("account_id") or account.get("id"))


def _role(account: dict) -> str:
    return str(account.get("role", ""))


def _chat_subject(query: str, requested: str | None = None) -> str | None:
    if requested in {"math", "programming", "python", "scratch"}:
        return requested
    lowered = query.casefold()
    if "python" in lowered:
        return "python"
    if "scratch" in lowered or "\u0441\u043a\u0440\u0435\u0442\u0447" in lowered:
        return "scratch"
    if any(word in lowered for word in ("\u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c", "\u043a\u043e\u0434", "\u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c")):
        return "programming"
    return None


def _safe_student_card(card: dict) -> dict:
    """Project only child-safe profile fields; no learner model/history scores."""
    source = card if isinstance(card, dict) else {}
    rag = source.get("rag_context") if isinstance(source.get("rag_context"), dict) else {}
    goal = rag.get("current_goal") if isinstance(rag.get("current_goal"), dict) else {}
    safe_goal = {key: goal[key] for key in ("topic_id", "goal") if goal.get(key)}
    safe_rag = {}
    if safe_goal:
        safe_rag["current_goal"] = safe_goal
    if isinstance(rag.get("current_topics"), list):
        safe_rag["current_topics"] = [str(topic) for topic in rag["current_topics"][:12] if isinstance(topic, (str, int))]
    interests = source.get("interests")
    if not isinstance(interests, list):
        interests = (source.get("profile") or {}).get("interests", []) if isinstance(source.get("profile"), dict) else []
    return {
        key: source[key] for key in ("learner_id", "pseudonym", "grade", "language") if key in source
    } | {
        "interests": [str(value) for value in interests[:8] if isinstance(value, (str, int))] if isinstance(interests, list) else [],
        "rag_context": safe_rag,
    }


def _redact_teacher_fields(value: Any) -> Any:
    blocked = {"legal_name", "journal", "points_ledger", "tutor_notes", "teacher_notes", "teacher_note", "answers", "answer", "answer_key", "expected_answer", "solution", "solutions", "reasoning_content"}
    if isinstance(value, dict):
        return {key: _redact_teacher_fields(item) for key, item in value.items() if str(key).lower() not in blocked}
    if isinstance(value, list):
        return [_redact_teacher_fields(item) for item in value]
    return value


def _validate_learner_card(card: dict, learner_id: str, store) -> None:
    from pipeline.learner_context import load_learner_data_tools
    if not isinstance(card, dict) or card.get("learner_id") != learner_id:
        raise ValueError("card.learner_id must match the requested learner_id")
    tools = load_learner_data_tools()
    findings = tools.validate_card(card, store.catalog(), f"{learner_id}.json")
    errors = [finding.format() for finding in findings if finding.is_error()]
    if errors:
        raise ValueError("learner card did not pass validation: " + "; ".join(errors))


def _safe_student_completion(response: dict) -> dict:
    safe = _redact_teacher_fields(response)
    choices = safe.get("choices", []) if isinstance(safe, dict) else []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            message.pop("reasoning_content", None)
    for key in ("citations", "sources"):
        rows = safe.get(key)
        if isinstance(rows, list):
            safe[key] = [
                {name: row[name] for name in ("id", "source_file", "score", "similarity") if name in row}
                for row in rows if isinstance(row, dict)
            ]
    return safe


def create_app(settings=None, store=None, runtime=None, rag_service=None) -> FastAPI:
    """Build an app with injectable settings, DB store and local model services.

    This factory does not migrate the DB or make network calls. Constructors and
    service requests are deferred until the relevant route is used.
    """
    if settings is None:
        from local_app.config import Settings
        settings = Settings.from_env()
    dsn = _value(settings, "database_url", "database_dsn", "dsn", default="postgresql://localhost/teachcopilot")
    if store is None:
        from local_app.store import Store
        store = Store(dsn)
    if runtime is None or rag_service is None:
        try:
            from local_app.rag import RagService
            from local_app.runtime import OllamaRuntime
            if runtime is None:
                runtime = OllamaRuntime(settings)
            if rag_service is None:
                rag_service = RagService(store=store, settings=settings, runtime=runtime)
        except ImportError:
            # Useful for API-only tests and deployments without model extras.
            runtime = runtime

    app = FastAPI(title="TeachCopilot Local API")
    app.state.settings = settings
    app.state.store = store
    app.state.runtime = runtime
    app.state.rag_service = rag_service

    from local_app.generation_scheduler import GenerationCapacityError

    @app.exception_handler(GenerationCapacityError)
    async def generation_capacity_handler(request, exc):
        return JSONResponse(status_code=429, content={"detail": "generation queue is full; retry later"},
                            headers={"Retry-After": "10"})

    try:
        import psycopg2
    except ImportError:
        psycopg2 = None
    if psycopg2 is not None:
        @app.exception_handler(psycopg2.Error)
        async def database_error_handler(request, exc):
            logger.exception("Database request failed")
            return JSONResponse(status_code=503, content={"detail": "database service is unavailable"})

    def require_user(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer session required", headers={"WWW-Authenticate": "Bearer"})
        token = authorization[7:].strip()
        if not token:
            raise HTTPException(status_code=401, detail="Bearer session required", headers={"WWW-Authenticate": "Bearer"})
        session = store.get_session(hash_session_token(token))
        if not session:
            raise HTTPException(status_code=401, detail="session is invalid or expired", headers={"WWW-Authenticate": "Bearer"})
        account_id = session.get("account_id")
        account = session.get("account") or store.get_account(account_id=account_id)
        if not account:
            raise HTTPException(status_code=401, detail="session account is unavailable")
        return account

    def teacher(user: dict = Depends(require_user)) -> dict:
        if _role(user) != "teacher":
            raise HTTPException(status_code=403, detail="teacher role required")
        return user

    def access(user: dict, learner_id: str) -> None:
        if _role(user) == "student":
            if student_learner_id(user) != learner_id or not store.authorize_learner_access(_account_id(user), learner_id):
                raise HTTPException(status_code=403, detail="learner access denied")
        elif _role(user) == "teacher":
            if not store.authorize_learner_access(_account_id(user), learner_id):
                raise HTTPException(status_code=403, detail="learner access denied")
        else:
            raise HTTPException(status_code=403, detail="role is not allowed")

    def student_learner_id(user: dict) -> str | None:
        if user.get("learner_id"):
            direct = str(user["learner_id"])
            if store.authorize_learner_access(_account_id(user), direct):
                return direct
        for card in store.list_cards():
            candidate = str(card.get("learner_id", ""))
            if candidate and store.authorize_learner_access(_account_id(user), candidate):
                return candidate
        return None

    class LoginRequest(BaseModel):
        email: str = Field(..., min_length=3, max_length=320)
        password: str = Field(..., min_length=1, max_length=1024)

    class LearnerCreateRequest(BaseModel):
        learner_id: str = Field(..., min_length=1, max_length=100)
        card: dict[str, Any]

    class LearnerUpdateRequest(BaseModel):
        expected_revision: int = Field(..., ge=1)
        card: dict[str, Any]

    class LearnerLinkRequest(BaseModel):
        student_email: str = Field(..., min_length=3, max_length=320)

    class StudentCreateRequest(BaseModel):
        email: str = Field(..., min_length=3, max_length=320)
        display_name: str = Field(..., min_length=1, max_length=160)
        password: str = Field(..., min_length=12, max_length=1024)

    class TaskRequest(BaseModel):
        learner_id: str = Field(..., min_length=1)
        topic_id: str | None = None
        count: int | None = Field(default=None, ge=1, le=10)
        mode: Literal["generated", "source_template"] = "generated"
        next_step_id: str | None = Field(default=None, max_length=180)

    class NextStepRequest(BaseModel):
        topic_id: str = Field(..., min_length=1, max_length=180)
        method: Literal["profile_only", "tail", "summary", "structured"] = "tail"
        expected_revision: int | None = Field(default=None, ge=1)

    class LearningSessionRequest(BaseModel):
        topic_id: str | None = Field(default=None, max_length=180)
        client_session_id: str | None = Field(default=None, max_length=180)

    class MemoryConfirmRequest(BaseModel):
        expected_revision: int = Field(..., ge=1)
        interest_candidate_ids: list[str] | None = Field(default=None, max_length=16)

    class UpdateRequest(BaseModel):
        notes: str = Field(..., min_length=1)
        learner_ids: list[str] = Field(..., min_length=1)
        apply: bool = False
        proposal_id: str | None = None
        mode: str | None = None
        confirmed_topics: list[str] | None = None

    class ObservationRequest(BaseModel):
        expected_revision: int = Field(..., ge=1)
        topic_id: str = Field(..., min_length=1, max_length=180)
        task_id: str = Field(..., min_length=1, max_length=180)
        solved: StrictBool
        support_level: Literal["independent", "hint_question", "visual", "joint"]
        assessment_pair_id: str | None = Field(default=None, max_length=180)
        difficulty: Literal["easy", "medium", "hard"] | None = None
        notes: str | None = Field(default=None, max_length=1000)
        observed_on: date | None = None

    class SearchRequest(BaseModel):
        query: str = Field(..., min_length=1)
        learner_id: str | None = None
        child_id: str | None = None
        subject_id: str | None = None
        limit: int = Field(default=5, ge=1, le=50)

    def memory_error(exc: Exception):
        from local_app.store import RevisionConflict
        from local_app.memory_store import MemoryQueueFullError, MemoryIdempotencyConflict
        if isinstance(exc, MemoryIdempotencyConflict):
            raise HTTPException(status_code=409, detail="client_message_id was reused for a different request") from exc
        if isinstance(exc, RevisionConflict):
            raise HTTPException(status_code=409, detail="learner revision changed; reload before continuing") from exc
        if isinstance(exc, MemoryQueueFullError):
            raise HTTPException(status_code=429, detail="memory queue is full; retry later", headers={"Retry-After": "10"}) from exc
        if _is_generation_capacity(exc):
            raise HTTPException(status_code=429, detail="generation queue is full; retry later", headers={"Retry-After": "10"}) from exc
        if isinstance(exc, (KeyError, LookupError)):
            raise HTTPException(status_code=404, detail="memory record not found") from exc
        if isinstance(exc, PermissionError):
            raise HTTPException(status_code=403, detail="memory record belongs to another account") from exc
        if isinstance(exc, (ValueError, TypeError)):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise exc

    def memory_available():
        if not hasattr(store, "get_learner_memory"):
            raise HTTPException(status_code=503, detail="memory storage is unavailable")

    def admit_chat(learner_id: str, topic_id: str, payload: dict, actor_id: str,
                   worker_id: str, idempotency_key: str | None = None):
        if not hasattr(store, "enqueue_memory_job"):
            raise HTTPException(status_code=503, detail="durable chat admission is unavailable")
        job = store.enqueue_memory_job(learner_id, topic_id, "chat", payload, actor_id,
                                       idempotency_key=idempotency_key)
        deadline = time.monotonic() + min(180.0, float(_value(settings, "request_timeout", default=180)))
        while True:
            claimed = store.claim_interactive_chat_job(job["job_id"], worker_id, lease_seconds=600)
            if claimed:
                return claimed
            if time.monotonic() >= deadline:
                store.cancel_memory_job(job["job_id"], "chat_queue_wait_timeout")
                raise HTTPException(status_code=429, detail="chat queue wait expired; send a new message later",
                                    headers={"Retry-After": "10"})
            time.sleep(0.05)

    @app.post("/learners/{learner_id}/next-step", status_code=202)
    def prepare_next_step(learner_id: str, request: NextStepRequest, user: dict = Depends(teacher)):
        access(user, learner_id)
        memory_available()
        try:
            job = store.enqueue_memory_job(learner_id, request.topic_id, "next_step", {},
                _account_id(user), expected_revision=request.expected_revision, method=request.method)
        except Exception as exc:
            memory_error(exc)
        return {"job_id": str(job["job_id"]), "status": job["status"], "learner_id": learner_id}

    @app.get("/memory/jobs/{job_id}")
    def memory_job(job_id: str, user: dict = Depends(require_user)):
        memory_available()
        job = store.get_memory_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="memory job not found")
        access(user, job["learner_id"])
        if _role(user) != "teacher":
            # Student sessions may inspect update progress, never raw teacher
            # sources, extracted records or diagnostics from another workflow.
            if job["kind"] != "memory_update":
                raise HTTPException(status_code=403, detail="teacher role required")
            return {key: job.get(key) for key in ("job_id", "learner_id", "kind", "status")}
        return job

    @app.get("/learners/{learner_id}/memory")
    def learner_memory(learner_id: str, topic_id: str | None = None, user: dict = Depends(teacher)):
        access(user, learner_id)
        memory_available()
        result = store.get_learner_memory(learner_id, topic_id)
        return {**result, "default_method": "tail", "human_reviewed": False}

    @app.post("/learners/{learner_id}/memory/versions/{version_id}/confirm")
    def confirm_memory(learner_id: str, version_id: str, request: MemoryConfirmRequest,
                       user: dict = Depends(teacher)):
        access(user, learner_id)
        memory_available()
        try:
            review_options = ({"interest_candidate_ids": request.interest_candidate_ids}
                              if request.interest_candidate_ids is not None else {})
            version = store.approve_memory_version(learner_id, version_id, _account_id(user),
                                                   request.expected_revision, **review_options)
        except Exception as exc:
            from local_app.memory_store import MemoryIdempotencyConflict
            if isinstance(exc, MemoryIdempotencyConflict):
                raise HTTPException(status_code=409, detail="memory was already confirmed with a different selection") from exc
            memory_error(exc)
        return {"learner_id": learner_id, "version": version, "confirmed": True}

    @app.post("/learners/{learner_id}/learning-sessions", status_code=201)
    def start_learning_session(learner_id: str, request: LearningSessionRequest,
                               user: dict = Depends(require_user)):
        access(user, learner_id)
        memory_available()
        try:
            return store.create_learning_session(learner_id, _account_id(user), request.topic_id or "general",
                                                 client_session_id=request.client_session_id)
        except Exception as exc:
            memory_error(exc)

    @app.post("/learning-sessions/{learning_session_id}/complete", status_code=202)
    def end_learning_session(learning_session_id: str, user: dict = Depends(require_user)):
        memory_available()
        session = store.get_learning_session(learning_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="learning session not found")
        access(user, session["learner_id"])
        if _role(user) == "student" and str(session.get("actor_id")) != _account_id(user):
            raise HTTPException(status_code=403, detail="learning session belongs to another account")
        try:
            return store.complete_learning_session(learning_session_id, session["learner_id"], _account_id(user))
        except Exception as exc:
            memory_error(exc)

    @app.get("/health")
    @app.get("/health/live")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    @app.get("/health/ready")
    def readiness():
        try:
            store.ping()
            from local_app.store import MIGRATIONS
            expected_schema = max((int(path.name.split("_", 1)[0]) for path in MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")), default=0)
            schema = store.schema_version()
            if schema < expected_schema:
                raise RuntimeError("database migrations are incomplete")
            if app.state.rag_service is None or not callable(getattr(app.state.rag_service, "status", None)):
                raise RuntimeError("RAG readiness status is unavailable")
            index_status = app.state.rag_service.status()
            if not index_status.get("ready", False):
                raise RuntimeError("embedding index is incomplete")
            if app.state.runtime is None or not callable(getattr(app.state.runtime, "check_models", None)):
                raise RuntimeError("local model readiness check is unavailable")
            models_status = app.state.runtime.check_models()
            if not models_status.get("ready", False):
                raise RuntimeError("configured local models are unavailable")
            return {"status": "ready", "database": "ok", "schema_version": schema, "index": index_status, "models": models_status}
        except Exception as exc:
            logger.exception("Database readiness check failed")
            raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": "database, index, or local model readiness check failed"}) from exc

    @app.post("/auth/login")
    def login(request: LoginRequest):
        account = store.get_account(email=request.email.strip().lower())
        encoded = account.get("password_hash", "") if account else ""
        if not account or not account.get("active", True) or not verify_password(request.password, encoded):
            raise HTTPException(status_code=401, detail="invalid email or password")
        token = new_session_token()
        store.create_session(_account_id(account), hash_session_token(token), session_expiry(hours=int(_value(settings, "session_hours", default=12))))
        session_hours = int(_value(settings, "session_hours", default=12))
        return {"access_token": token, "token_type": "bearer", "expires_in": session_hours * 3600, "account": {"id": _account_id(account), "email": account.get("email"), "display_name": account.get("display_name"), "role": _role(account), "learner_id": account.get("learner_id")}}

    @app.post("/auth/logout")
    def logout(user: dict = Depends(require_user), authorization: str | None = Header(default=None)):
        token = authorization[7:].strip()
        store.revoke_session(hash_session_token(token))
        return {"ok": True}

    @app.get("/auth/me")
    def current_account(user: dict = Depends(require_user)):
        return {"account": {
            "id": _account_id(user),
            "email": user.get("email"),
            "display_name": user.get("display_name"),
            "role": _role(user),
            "learner_id": user.get("learner_id"),
        }}

    @app.get("/catalog")
    def curriculum_catalog(user: dict = Depends(require_user)):
        rows = []
        for topic_id, raw in sorted(store.catalog().items()):
            entry = raw if isinstance(raw, dict) else {}
            parent = entry.get("parent")
            prerequisites = entry.get("prerequisites")
            if not isinstance(prerequisites, list):
                prerequisites = [parent] if isinstance(parent, str) and parent else []
            rows.append({
                "topic_id": topic_id,
                "title": entry.get("title_ru") or entry.get("title") or topic_id,
                "subject_id": topic_id.split(".", 1)[0],
                "grade": entry.get("grade"),
                "prerequisites": [item for item in prerequisites if isinstance(item, str)],
            })
        return {"topics": rows}

    @app.get("/learners")
    def learners(user: dict = Depends(require_user)):
        if _role(user) == "student":
            learner_id = student_learner_id(user)
            if not learner_id:
                return {"learners": []}
            card = store.get_card(learner_id)
            return {"learners": [{"learner_id": learner_id, "profile": _safe_student_card(card) if card else None}]}
        if _role(user) != "teacher":
            raise HTTPException(status_code=403, detail="role is not allowed")
        cards = store.list_cards()
        rows = [card for card in cards if store.authorize_learner_access(_account_id(user), card.get("learner_id", ""))]
        return {"learners": rows}

    @app.post("/learners", status_code=201)
    def create_learner(request: LearnerCreateRequest, user: dict = Depends(teacher)):
        if request.card.get("learner_id") != request.learner_id:
            raise HTTPException(status_code=422, detail="card.learner_id must match learner_id")
        if store.get_card(request.learner_id) is not None:
            raise HTTPException(status_code=409, detail="learner already exists; existing learners can be assigned only by an operator")
        try:
            _validate_learner_card(request.card, request.learner_id, store)
            card = store.create_assigned_learner(request.card, teacher_id=_account_id(user))
        except ValueError as exc:
            if "already exists" in str(exc).lower():
                raise HTTPException(status_code=409, detail="learner already exists; existing learners can be assigned only by an operator") from exc
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="teacher account is not permitted to create learners") from exc
        return {"learner_id": request.learner_id, "revision": card.get("revision"), "created": True}

    @app.put("/learners/{learner_id}")
    def update_learner(learner_id: str, request: LearnerUpdateRequest, user: dict = Depends(teacher)):
        access(user, learner_id)
        if request.card.get("learner_id") != learner_id:
            raise HTTPException(status_code=422, detail="card.learner_id must match learner_id")
        try:
            _validate_learner_card(request.card, learner_id, store)
            card = store.save_card(request.card, actor_id=_account_id(user), expected_revision=request.expected_revision)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            from local_app.store import RevisionConflict
            if isinstance(exc, RevisionConflict):
                raise HTTPException(status_code=409, detail="learner card revision changed; reload before editing") from exc
            raise
        return {"learner_id": learner_id, "revision": card.get("revision"), "updated": True}

    @app.post("/learners/{learner_id}/student")
    def link_student(learner_id: str, request: LearnerLinkRequest, user: dict = Depends(teacher)):
        access(user, learner_id)
        student = store.get_account(email=request.student_email.strip().lower())
        if not student or _role(student) != "student":
            raise HTTPException(status_code=404, detail="student account not found")
        existing = [
            str(card.get("learner_id")) for card in store.list_cards()
            if card.get("learner_id") and store.authorize_learner_access(_account_id(student), str(card["learner_id"]))
        ]
        if any(existing_id != learner_id for existing_id in existing):
            raise HTTPException(status_code=409, detail="student account is already linked")
        store.assign_learner(_account_id(student), learner_id, relationship="student")
        return {"learner_id": learner_id, "student_account_id": _account_id(student), "linked": True}

    @app.post("/learners/{learner_id}/students", status_code=201)
    def create_student(learner_id: str, request: StudentCreateRequest, user: dict = Depends(teacher)):
        access(user, learner_id)
        if not hasattr(store, "create_student_account"):
            raise HTTPException(status_code=503, detail="atomic student provisioning is unavailable")
        try:
            account = store.create_student_account(
                email=request.email.strip().lower(), display_name=request.display_name.strip(),
                password_hash=hash_password(request.password), learner_id=learner_id,
                teacher_id=_account_id(user),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="learner access denied") from exc
        except Exception as exc:
            logger.exception("Student account provisioning failed")
            raise HTTPException(status_code=409, detail="student account could not be created; check email and learner assignment") from exc
        return {"account": {"id": _account_id(account), "email": account.get("email"), "display_name": account.get("display_name"), "role": "student", "learner_id": learner_id}}

    @app.get("/profile")
    def profile(learner_id: str | None = None, child_id: str | None = None, user: dict = Depends(require_user)):
        target = learner_id or child_id or (student_learner_id(user) if _role(user) == "student" else None)
        if not target:
            raise HTTPException(status_code=422, detail="learner_id is required")
        access(user, target)
        card = store.get_card(target)
        if card is None:
            raise HTTPException(status_code=404, detail="learner not found")
        return {"learner_id": target, "profile": card if _role(user) == "teacher" else _safe_student_card(card)}

    @app.get("/learners/{learner_id}")
    def get_learner(learner_id: str, user: dict = Depends(require_user)):
        access(user, learner_id)
        card = store.get_card(learner_id)
        if card is None:
            raise HTTPException(status_code=404, detail="learner not found")
        return {"learner_id": learner_id, "profile": card if _role(user) == "teacher" else _safe_student_card(card)}

    def _zpd_report(card: dict, catalog: dict) -> list[dict]:
        from pipeline.learner_context import load_learner_data_tools
        zpd = load_learner_data_tools().zpd
        return zpd.zone_report(card, catalog)

    def _observation_history(card: dict, catalog: dict) -> list[dict]:
        rows = []
        competencies = ((card.get("learner_model") or {}).get("competencies") or [])
        for competency in competencies:
            if not isinstance(competency, dict):
                continue
            topic_id = competency.get("topic_id")
            entry = catalog.get(topic_id, {}) if isinstance(topic_id, str) else {}
            for observation in competency.get("history") or []:
                if not isinstance(observation, dict):
                    continue
                support = observation.get("support_level") or observation.get("help_level")
                rows.append({
                    "topic_id": topic_id,
                    "topic_title": entry.get("title_ru", topic_id),
                    "date": observation.get("date"),
                    "solved": observation.get("solved"),
                    "support_level": support,
                    "task_id": observation.get("task_id"),
                    "notes": observation.get("note"),
                })
        rows.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
        return rows[:100]

    def _student_dashboard_topics(report: list[dict]) -> list[dict]:
        labels = {
            "mastered": "Получается самостоятельно",
            "zpd": "Учится с поддержкой",
            "outside": "Нужна подготовительная опора",
            "unknown": "Продолжаем наблюдать",
        }
        rows = []
        for topic in report:
            confirmed = topic.get("confirmed_zone")
            sufficient = bool(topic.get("evidence", {}).get("sufficient"))
            if topic.get("evidence_status") != "confirmed" or not sufficient or not confirmed:
                progress = labels["unknown"]
            else:
                progress = labels.get(confirmed, labels["unknown"])
            rows.append({
                "title": topic.get("title_ru", topic.get("topic_id")),
                "progress": progress,
                "last_observed": topic.get("last_assessed"),
                "stale": topic.get("stale", False),
            })
        return rows

    @app.get("/learners/{learner_id}/dashboard")
    def learner_dashboard(learner_id: str, user: dict = Depends(require_user)):
        access(user, learner_id)
        card = store.get_card(learner_id)
        if card is None:
            raise HTTPException(status_code=404, detail="learner not found")
        catalog = store.catalog()
        report = _zpd_report(card, catalog)
        history = _observation_history(card, catalog)
        revision = store.get_revision(learner_id)
        if _role(user) == "teacher":
            return {
                "learner_id": learner_id,
                "revision": revision,
                "profile": card,
                "zpd": {"topics": report},
                "history": {"observations": history},
            }

        safe_profile = _safe_student_card(card)
        rag = safe_profile.get("rag_context") or {}
        goal = rag.get("current_goal") or {}
        goal_topic = catalog.get(goal.get("topic_id"), {}) if isinstance(goal, dict) else {}
        student_profile = {
            key: safe_profile[key]
            for key in ("learner_id", "pseudonym", "grade", "language", "interests")
            if key in safe_profile
        }
        if goal:
            student_profile["goal"] = {
                "text": goal.get("goal"),
                "topic": goal_topic.get("title_ru", "") if isinstance(goal_topic, dict) else "",
            }
        visible_history = []
        for item in history[:30]:
            support = item.get("support_level")
            outcome = "Наблюдение записано"
            if item.get("solved") is True:
                outcome = "Выполнено самостоятельно" if support == "independent" else "Выполнено с опорой"
            elif item.get("solved") is False:
                outcome = "Попробовали, нужна ещё практика"
            visible_history.append({
                "topic": item.get("topic_title"),
                "date": item.get("date"),
                "outcome": outcome,
            })
        return {
            "learner_id": learner_id,
            "profile": student_profile,
            "zpd": {"topics": _student_dashboard_topics(report)},
            "history": {"observations": visible_history},
        }

    @app.post("/learners/{learner_id}/observations")
    def record_observation(learner_id: str, request: ObservationRequest, user: dict = Depends(teacher)):
        access(user, learner_id)
        if not request.task_id.strip():
            raise HTTPException(status_code=422, detail="task_id cannot be blank")
        card = store.get_card(learner_id)
        if card is None:
            raise HTTPException(status_code=404, detail="learner not found")
        catalog = store.catalog()
        if request.topic_id not in catalog:
            raise HTTPException(status_code=422, detail="topic_id is not in the curriculum catalog")
        tools = __import__("pipeline.learner_context", fromlist=["load_learner_data_tools"])
        zpd = tools.load_learner_data_tools().zpd
        if not callable(getattr(zpd, "record_observation", None)):
            raise HTTPException(status_code=503, detail="evidence-aware ZPD recording is unavailable")
        observed_on = (request.observed_on or date.today()).isoformat()
        try:
            updated = zpd.record_observation(
                card,
                topic_id=request.topic_id,
                support_level=request.support_level,
                solved=request.solved,
                observed_on=observed_on,
                task_id=request.task_id.strip(),
                assessment_pair_id=(request.assessment_pair_id or request.task_id).strip(),
                difficulty=request.difficulty,
                notes=request.notes.strip() if request.notes else None,
            )
            journal_note = (
                f"Наблюдение по задаче {request.task_id.strip()}: "
                f"solved={str(request.solved).lower()}, support={request.support_level}."
            )
            if request.notes and request.notes.strip():
                journal_note += " " + request.notes.strip()
            mvp = updated.setdefault("mvp", {})
            mvp.setdefault("journal", []).append({"date": observed_on, "note": journal_note})
            mvp.setdefault("lessons", []).append({
                "date": observed_on,
                "topic_ids": [request.topic_id],
                "note": f"Задача {request.task_id.strip()}; solved={str(request.solved).lower()}; support={request.support_level}.",
            })
            saved = store.save_card(
                updated, actor_id=_account_id(user), expected_revision=request.expected_revision,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            from local_app.store import RevisionConflict
            if isinstance(exc, RevisionConflict):
                raise HTTPException(status_code=409, detail="learner card revision changed; reload before recording evidence") from exc
            raise
        report = _zpd_report(saved, catalog)
        topic_report = next((row for row in report if row.get("topic_id") == request.topic_id), None)
        return {
            "learner_id": learner_id,
            "revision": saved.get("revision", request.expected_revision + 1),
            "observation": {
                "topic_id": request.topic_id,
                "task_id": request.task_id.strip(),
                "assessment_pair_id": (request.assessment_pair_id or request.task_id).strip(),
                "difficulty": request.difficulty,
                "solved": request.solved,
                "support_level": request.support_level,
                "observed_on": observed_on,
                "notes": request.notes.strip() if request.notes else None,
            },
            "zpd": topic_report,
        }

    @app.post("/rag/search")
    def rag_search(request: SearchRequest, user: dict = Depends(require_user)):
        target = request.learner_id or request.child_id or (student_learner_id(user) if _role(user) == "student" else None)
        if target:
            access(user, target)
            card = store.get_card(target)
            if card is None:
                raise HTTPException(status_code=404, detail="learner not found")
        if app.state.rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        try:
            card = store.get_card(target) if target else None
            rag_context = (card or {}).get("rag_context", {})
            goal = rag_context.get("current_goal") or {}
            topics = list(rag_context.get("current_topics") or [])
            topic_hint = goal.get("topic_id") or (topics[0] if topics else None)
            subject_id = request.subject_id or "math"
            if subject_id not in {"math", "programming", "python", "scratch"}:
                raise HTTPException(status_code=422, detail="unsupported subject_id")
            results = app.state.rag_service.search(query=request.query, subject_id=subject_id, topic_id=topic_hint, grade=(card or {}).get("grade"), limit=request.limit)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("RAG search failed")
            if _is_generation_capacity(exc):
                raise HTTPException(status_code=429, detail="generation queue is full; retry later", headers={"Retry-After": "10"}) from exc
            raise HTTPException(status_code=503, detail="RAG retrieval service is unavailable; check database and embedding health") from exc
        visible_results = results if _role(user) == "teacher" else [
            {key: item[key] for key in ("id", "source_file", "source_version", "score", "similarity", "topic_id", "grade", "subject_id") if key in item}
            for item in results if isinstance(item, dict)
        ]
        return {"query": request.query, "learner_id": target, "child_id": request.child_id, "count": len(visible_results), "results": visible_results}

    @app.post("/task/design")
    def task_design(request: TaskRequest, user: dict = Depends(teacher)):
        access(user, request.learner_id)
        next_step = None
        if request.next_step_id:
            if not hasattr(store, "get_next_step"):
                raise HTTPException(status_code=503, detail="memory storage is unavailable")
            next_step = store.get_next_step(request.next_step_id)
            if not next_step or next_step.get("learner_id") != request.learner_id:
                raise HTTPException(status_code=404, detail="recommendation not found for this learner")
            if next_step.get("base_revision") != store.get_revision(request.learner_id):
                raise HTTPException(status_code=409, detail="recommendation revision changed; prepare another next step")
            if hasattr(store, "get_learner_source_revision") and next_step.get("source_revision") != store.get_learner_source_revision(request.learner_id):
                raise HTTPException(status_code=409, detail="learning sources changed; prepare another next step")
            if request.topic_id and request.topic_id != next_step.get("topic_id"):
                raise HTTPException(status_code=422, detail="recommendation topic does not match the task")
            request.topic_id = next_step.get("topic_id")
        try:
            from pipeline.task_designer import TaskDesignError, design_task_card, render_task_card
            task_count = request.count or (1 if request.mode == "source_template" else 3)
            if request.mode == "source_template":
                from local_app.services import design_source_template
                card = design_source_template(request.learner_id, request.topic_id, store=store, count=task_count)
            else:
                if app.state.runtime is None or not hasattr(app.state.runtime, "complete"):
                    raise HTTPException(status_code=503, detail="local model runtime is unavailable")
                complete = app.state.runtime.complete
                if next_step:
                    original_complete = complete
                    def complete(prompt, system_prompt=""):
                        import json
                        reviewed = next_step.get("result") or next_step.get("recommendation") or {}
                        return original_complete(prompt + "\nRecommendation draft for teacher review: " +
                            json.dumps(reviewed, ensure_ascii=False), system_prompt=system_prompt)
                card = design_task_card(request.learner_id, request.topic_id, count=task_count, store=store, rag_service=app.state.rag_service, complete=complete)
        except Exception as exc:
            from pipeline.learner_context import LearnerContextError
            from pipeline.llm_json import LlmError
            from pipeline.task_designer import TaskDesignError
            if _is_generation_capacity(exc):
                raise HTTPException(status_code=429, detail="generation queue is full; retry later", headers={"Retry-After": "10"}) from exc
            if _is_dependency_unavailable(exc):
                logger.exception("Task design dependency failed")
                raise HTTPException(status_code=503, detail="local database or model service is unavailable") from exc
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=422, detail="task request exceeds local model limits or contains invalid input") from exc
            code = 404 if isinstance(exc, LearnerContextError) else 422 if isinstance(exc, TaskDesignError) else 502 if isinstance(exc, LlmError) else None
            if code is None:
                raise
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        as_dict = card.as_dict()
        hints_origin = None
        if request.mode == "source_template":
            has_hints = any(task.get("hints") for task in as_dict.get("tasks", []))
            used_general = any("General scaffolding was added" in warning for warning in as_dict.get("warnings", []))
            hints_origin = "general_scaffolding" if used_general else "source_authored" if has_hints else "none"
            as_dict["hints_origin"] = hints_origin
        child = render_task_card(card, show_answers=False)
        return {"card": as_dict, "mode": request.mode, "review_required": True, "status": "draft", "next_step_id": request.next_step_id,
                "hints_origin": hints_origin, "text_for_child": child, "text_for_tutor": render_task_card(card, show_answers=True)}

    @app.post("/learner/update")
    def learner_update(request: UpdateRequest, user: dict = Depends(teacher)):
        for learner_id in request.learner_ids:
            access(user, learner_id)
        if request.apply:
            if not request.proposal_id:
                raise HTTPException(status_code=422, detail="proposal_id is required to apply a reviewed lesson proposal")
            if not hasattr(store, "confirm_proposal"):
                raise HTTPException(status_code=503, detail="lesson proposal storage is unavailable")
            proposal = store.get_proposal(request.proposal_id)
            if not proposal:
                raise HTTPException(status_code=404, detail="lesson proposal not found")
            proposal_learners = proposal.get("metadata", {}).get("learner_ids", [])
            for learner_id in proposal_learners:
                access(user, learner_id)
            if set(request.learner_ids) != set(proposal_learners):
                raise HTTPException(status_code=403, detail="proposal learner scope does not match request")
            try:
                result = store.confirm_proposal(request.proposal_id, actor_id=_account_id(user))
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail="lesson proposal belongs to another teacher") from exc
            except Exception as exc:
                from local_app.store import RevisionConflict
                if isinstance(exc, RevisionConflict):
                    raise HTTPException(status_code=409, detail="learner card changed after review; create a new proposal") from exc
                raise
            return {"applied": True, "proposal_id": request.proposal_id, "mode": proposal.get("metadata", {}).get("mode"), "result": result}
        from pipeline.config import MASTERY_MODE, MASTERY_MODES
        from pipeline.card_updater import CardUpdateError, render_update, update_from_notes
        from pipeline.learner_context import LearnerContextError
        from pipeline.llm_json import LlmError
        mode = (request.mode or MASTERY_MODE).strip().lower()
        if mode not in MASTERY_MODES:
            raise HTTPException(status_code=422, detail=f"unknown mode '{mode}'; allowed: {', '.join(MASTERY_MODES)}")
        if app.state.runtime is None or not hasattr(app.state.runtime, "complete"):
            raise HTTPException(status_code=503, detail="local model runtime is unavailable")
        base_revisions = {learner_id: store.get_revision(learner_id) for learner_id in request.learner_ids}
        if any(revision is None for revision in base_revisions.values()):
            raise HTTPException(status_code=404, detail="one or more learners were not found")
        try:
            results = update_from_notes(request.notes, request.learner_ids, apply=False, mode=mode, confirmed_topics=request.confirmed_topics, store=store, complete=app.state.runtime.complete, actor_id=_account_id(user))
        except LearnerContextError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CardUpdateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LlmError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="lesson request exceeds local model limits or contains invalid input") from exc
        except Exception as exc:
            if _is_generation_capacity(exc):
                raise HTTPException(status_code=429, detail="generation queue is full; retry later", headers={"Retry-After": "10"}) from exc
            if _is_dependency_unavailable(exc):
                logger.exception("Lesson update dependency failed")
                raise HTTPException(status_code=503, detail="local database or model service is unavailable") from exc
            raise
        proposal_id = None
        if results:
            if not hasattr(store, "create_proposal"):
                raise HTTPException(status_code=503, detail="lesson proposal storage is unavailable")
            snapshots = {row.learner_id: row.card for row in results}
            revisions = {row.learner_id: base_revisions[row.learner_id] for row in results}
            proposal = store.create_proposal(
                proposal_id=secrets.token_urlsafe(18), actor_id=_account_id(user),
                base_revisions=revisions, proposed_cards=snapshots,
                metadata={"notes": request.notes, "mode": mode, "learner_ids": list(snapshots),
                    "memory_sources": {learner_id: {"topic_id": ((snapshots[learner_id].get("rag_context") or {}).get("current_goal") or {}).get("topic_id") or "general", "text": request.notes,
                        "observed_on": date.today().isoformat(), "group_note": len(snapshots) > 1}
                        for learner_id in snapshots}},
            )
            proposal_id = proposal.get("proposal_id") or proposal.get("id")
        return {"applied": False, "proposal_id": proposal_id, "mode": mode, "updates": [{"learner_id": row.learner_id, "changes": list(row.changes), "pending": list(row.pending), "warnings": list(row.warnings), "summary": render_update(row, applied=False), "memory_records": [{"kind": "teacher_source", "text": request.notes, "support_level": "unknown", "review_required": True, "group_note": len(results) > 1}]} for row in results]}

    @app.post("/lessons/proposals/{proposal_id}/confirm")
    def confirm_lesson(proposal_id: str, user: dict = Depends(teacher)):
        if not hasattr(store, "confirm_proposal"):
            raise HTTPException(status_code=503, detail="lesson proposal storage is unavailable")
        try:
            proposal = store.get_proposal(proposal_id)
            if not proposal:
                raise HTTPException(status_code=404, detail="lesson proposal not found")
            for learner_id in proposal.get("metadata", {}).get("learner_ids", []):
                access(user, learner_id)
            result = store.confirm_proposal(proposal_id, actor_id=_account_id(user))
            return {"proposal_id": proposal_id, "confirmed": True, "result": result}
        except HTTPException:
            raise
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="lesson proposal belongs to another teacher") from exc
        except Exception as exc:
            logger.exception("Lesson confirmation failed")
            raise HTTPException(status_code=409, detail="lesson proposal could not be confirmed; reload or request a new proposal") from exc

    @app.get("/lessons/proposals/{proposal_id}")
    def preview_lesson(proposal_id: str, user: dict = Depends(teacher)):
        proposal = store.get_proposal(proposal_id)
        if not proposal:
            raise HTTPException(status_code=404, detail="lesson proposal not found")
        if str(proposal.get("actor_id")) != _account_id(user):
            raise HTTPException(status_code=403, detail="lesson proposal belongs to another teacher")
        metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
        learner_ids = metadata.get("learner_ids") or ([proposal.get("learner_id")] if proposal.get("learner_id") else [])
        for learner_id in learner_ids:
            access(user, str(learner_id))
        return {
            "proposal_id": proposal.get("proposal_id", proposal_id),
            "status": proposal.get("status", "pending"),
            "created_at": proposal.get("created_at"),
            "metadata": metadata,
            "cards": proposal.get("cards") or ([{
                "learner_id": proposal.get("learner_id"),
                "base_revision": proposal.get("base_revision"),
                "proposed_card": proposal.get("proposed_card"),
                "result_card": proposal.get("result_card"),
            }] if proposal.get("proposed_card") else []),
        }

    @app.post("/materials/import")
    def import_materials(user: dict = Depends(teacher)):
        try:
            from pathlib import Path
            repo_root = Path(__file__).resolve().parents[1]
            result = store.import_materials(repo_root)
            return {"imported": result}
        except Exception as exc:
            logger.exception("Material import failed")
            raise HTTPException(status_code=422, detail="material import failed; inspect server logs") from exc

    @app.post("/materials/index")
    def index_materials(user: dict = Depends(teacher)):
        if app.state.rag_service is None:
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        try:
            return {"indexed": app.state.rag_service.reindex()}
        except Exception as exc:
            logger.exception("Material indexing failed")
            raise HTTPException(status_code=503, detail="material indexing service is unavailable") from exc

    @app.get("/")
    def frontend_index():
        index_file = WEB_ROOT / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="local frontend has not been built")
        return FileResponse(index_file)

    @app.get("/v1/models")
    def openai_models(user: dict = Depends(require_user)):
        return {"object": "list", "data": [{"id": "teachcopilot-rag", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def openai_chat_completions(payload: dict[str, Any], user: dict = Depends(require_user)):
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        target = student_learner_id(user) if _role(user) == "student" else (payload.get("learner_id") or meta.get("learner_id"))
        if not target:
            raise HTTPException(status_code=422, detail="learner_id is required for this account")
        access(user, target)
        if store.get_card(target) is None:
            raise HTTPException(status_code=404, detail="learner not found")
        messages = payload.get("messages")
        explicit = 'learning_action' in payload
        action = None
        if explicit:
            from local_app.explicit_learning_actions import validate_action
            if messages is not None and messages != [] or 'messages' in payload and messages is None:
                raise HTTPException(status_code=422, detail="Учебное действие не совмещается с сообщениями. Используйте поля формы.")
            try:
                action, action_text = validate_action(payload['learning_action'])
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            messages = [{"role": "user", "content": action_text}]
        if not isinstance(messages, list):
            raise HTTPException(status_code=422, detail="messages must be an array")
        # Preserve only chat roles and content. Ignore client model, tools, sampling
        # controls, learner identity overrides and other upstream fields.
        clean_messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages if isinstance(message, dict) and message.get("role") in {"user", "assistant"}]
        if not clean_messages or not any(message["role"] == "user" for message in clean_messages):
            raise HTTPException(status_code=422, detail="at least one user message is required")
        if any(not isinstance(message["content"], str) or not message["content"].strip()
               or len(message["content"]) > 12000 for message in clean_messages):
            raise HTTPException(status_code=422, detail="message content must be nonempty text within 12000 characters")
        learning_session_id = payload.get("learning_session_id") or meta.get("learning_session_id")
        client_message_id = payload.get("client_message_id") or meta.get("client_message_id")
        if explicit and not learning_session_id:
            raise HTTPException(status_code=422, detail="Для учебного действия требуется learning_session_id.")
        reserved = False
        chat_job = None
        chat_worker_id = "chat-" + secrets.token_urlsafe(18)
        call_started = False
        session = None
        if learning_session_id:
            memory_available()
            if not isinstance(client_message_id, str) or not 1 <= len(client_message_id) <= 180:
                raise HTTPException(status_code=422, detail="client_message_id is required for a learning session")
            session = store.get_learning_session(str(learning_session_id), target)
            if not session:
                raise HTTPException(status_code=404, detail="learning session not found for this learner")
            if str(session.get("actor_id")) != _account_id(user):
                raise HTTPException(status_code=403, detail="learning session belongs to another account")
            if explicit:
                from local_app.explicit_learning_actions import validate_subject
                try:
                    validate_subject(action, session.get('topic_id'), meta.get('subject_id'), store.catalog())
                except (ValueError, TypeError) as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            latest_user = next(message for message in reversed(clean_messages) if message["role"] == "user")
            try:
                reservation = store.reserve_chat_message(str(learning_session_id), target,
                    client_message_id, latest_user["content"], _account_id(user), author_role=_role(user))
            except Exception as exc:
                memory_error(exc)
            if not reservation.get("created"):
                if reservation.get("status") == "completed" and isinstance(reservation.get("response"), dict):
                    cached = reservation["response"]
                    from local_app.explicit_learning_actions import VERSION
                    expected_action = {'operation': action['operation'], 'request_text': action_text, 'version': VERSION} if explicit else None
                    if cached.get('learning_action') != expected_action:
                        raise HTTPException(status_code=409, detail="client_message_id was reused for a different request mode")
                    if _role(user) == "student":
                        cached = _safe_student_completion(cached)
                    return JSONResponse(content=cached)
                status_code = 409
                detail = "message is already processing" if reservation.get("status") == "pending" else "previous message outcome is unavailable; automatic retry is disabled"
                raise HTTPException(status_code=status_code, detail=detail)
            reserved = True
            try:
                chat_job = admit_chat(target, session.get("topic_id") or "general",
                    {"learning_session_id": str(learning_session_id), "client_message_id": client_message_id},
                    _account_id(user), chat_worker_id, idempotency_key=f"chat:{learning_session_id}:{client_message_id}")
            except HTTPException:
                store.fail_chat_message(str(learning_session_id), client_message_id,
                                        error="chat admission rejected", unknown_outcome=False)
                raise
            except Exception as exc:
                store.fail_chat_message(str(learning_session_id), client_message_id,
                                        error=type(exc).__name__, unknown_outcome=False)
                memory_error(exc)
            clean_messages = [{"role": row["role"], "content": row["content"]}
                              for row in store.list_learning_messages(str(learning_session_id))]
        else:
            # Legacy clients keep their dialogue contract, while sharing the
            # same persistent waiting cap and local generation admission.
            try:
                chat_job = admit_chat(target, "general", {"legacy_chat": True},
                                       _account_id(user), chat_worker_id)
            except HTTPException:
                raise
            except Exception as exc:
                memory_error(exc)
        if explicit:
            from local_app.explicit_learning_actions import action_response
            try:
                response = action_response(action, action_text, clean_messages)
                response.update(learning_session_id=str(learning_session_id), client_message_id=client_message_id,
                                memory_version_id=None, memory_stale=False)
                finished = store.finish_memory_job(chat_job['job_id'], result={'response_ready': True}, worker_id=chat_worker_id)
                if finished.get('status') != 'completed':
                    raise ValueError('chat context changed or job lease expired during action')
                store.complete_chat_message(str(learning_session_id), client_message_id, response)
            except Exception as exc:
                store.fail_chat_message(str(learning_session_id), client_message_id, error=type(exc).__name__, unknown_outcome=False)
                store.finish_memory_job(chat_job['job_id'], error=type(exc).__name__, worker_id=chat_worker_id, unknown_outcome=False)
                memory_error(exc)
            return JSONResponse(content=_safe_student_completion(response) if _role(user) == 'student' else response)
        if app.state.rag_service is None:
            if reserved:
                store.fail_chat_message(str(learning_session_id), client_message_id,
                                        error="RAG service is unavailable", unknown_outcome=False)
            if chat_job:
                store.finish_memory_job(chat_job["job_id"], error="RAG service is unavailable", worker_id=chat_worker_id)
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
        try:
            card = store.get_card(target)
            safe_profile = card.get("rag_context", {}) if isinstance(card, dict) else {}
            if hasattr(store, "catalog"):
                from pipeline.learner_context import LearnerContextError, load_learner_context
                try:
                    safe_profile = load_learner_context(target, store=store).prompt_text
                except LearnerContextError as exc:
                    raise HTTPException(status_code=404, detail="learner context is unavailable") from exc
            goal = card.get("rag_context", {}).get("current_goal", {}) if isinstance(card, dict) else {}
            topic_hint = goal.get("topic_id") if isinstance(goal, dict) else None
            if session and session.get("topic_id"):
                topic_hint = session["topic_id"]
            memory_context = {"text": "", "sources": []}
            if hasattr(store, "get_learner_memory"):
                from local_app.memory_service import approved_chat_memory
                memory_context = approved_chat_memory(store.get_learner_memory(target, topic_hint), topic_hint)
            latest = next((str(message.get("content", "")) for message in reversed(clean_messages) if message.get("role") == "user"), "").lower()
            subject_id = _chat_subject(latest, meta.get("subject_id"))
            answer_kwargs = dict(profile=safe_profile, subject_id=subject_id, topic_id=topic_hint, grade=card.get("grade") if isinstance(card, dict) else None, limit=int(_value(app.state.settings, "rag_limit", default=3)))
            # Only the real adapter receives server-collected identifiers. Older
            # injected RAG services keep the original call contract.
            from local_app.rag import RagService
            if isinstance(app.state.rag_service, RagService):
                from local_app.conversation_memory import preference_projection
                identifiers = [target, user.get("email"), user.get("display_name"),
                               card.get("legal_name"), card.get("pseudonym")]
                identity_complete = False
                if hasattr(store, "learner_known_identifiers"):
                    identifiers.extend(store.learner_known_identifiers(target))
                    identity_complete = True
                answer_kwargs.update(known_identifiers=[str(value) for value in identifiers if value],
                    identity_context_complete=identity_complete,
                    preference_context=preference_projection(card),
                    memory_context=memory_context,
                    memory_projection=[{"source_id": row["source_id"], "quote": row.get("text", "")}
                                       for row in memory_context.get("sources", [])])
            if chat_job:
                store.start_memory_job_call(chat_job["job_id"], chat_worker_id)
                call_started = True
            response = app.state.rag_service.answer(clean_messages, **answer_kwargs)
            if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
                raise ValueError("chat service returned an invalid response")
            if reserved:
                response.update(learning_session_id=str(learning_session_id), client_message_id=client_message_id,
                                memory_version_id=memory_context.get("version_id"), memory_stale=memory_context.get("stale", False))
            if chat_job:
                finished = store.finish_memory_job(chat_job["job_id"], result={"response_ready": True}, worker_id=chat_worker_id)
                if finished.get("status") != "completed":
                    raise ValueError("chat context changed or job lease expired during generation")
            if reserved:
                store.complete_chat_message(str(learning_session_id), client_message_id, response)
        except HTTPException:
            if reserved:
                store.fail_chat_message(str(learning_session_id), client_message_id, error="chat request rejected", unknown_outcome=call_started)
            if chat_job:
                store.finish_memory_job(chat_job["job_id"], error="chat request rejected", worker_id=chat_worker_id, unknown_outcome=call_started)
            raise
        except Exception as exc:
            capacity = _is_generation_capacity(exc)
            if reserved:
                store.fail_chat_message(str(learning_session_id), client_message_id, error=type(exc).__name__, unknown_outcome=call_started and not capacity)
            if chat_job:
                store.finish_memory_job(chat_job["job_id"], error=type(exc).__name__, worker_id=chat_worker_id, unknown_outcome=call_started and not capacity)
            if capacity:
                raise HTTPException(status_code=429, detail="generation queue is full; retry later", headers={"Retry-After": "10"}) from exc
            logger.exception("Chat/RAG request failed")
            code = 503 if _is_dependency_unavailable(exc) else 502
            detail = "local database or model service is unavailable" if code == 503 else "chat service returned an error"
            raise HTTPException(status_code=code, detail=detail) from exc
        if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
            raise HTTPException(status_code=502, detail="chat service returned an invalid response")
        if _role(user) == "student":
            response = _safe_student_completion(response)
        return JSONResponse(content=response)

    app.mount("/assets", StaticFiles(directory=str(WEB_ROOT), check_dir=False), name="assets")
    return app


app = create_app()

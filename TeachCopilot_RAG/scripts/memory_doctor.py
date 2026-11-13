"""CLI health-check and end-to-end verification for the «Алёна» memory subsystem.

Operational tool, not a blind evaluation. ``--check`` probes a running local
deployment read-only (working database SELECTs only). ``--e2e`` first runs the
check and then exercises the real memory pipeline on a disposable
``nirs_memory_doctor_*`` database with synthetic accounts and fictional
lessons: teacher next-step job -> MemoryWorker -> teacher approval -> student
chat. The real local Ollama models serve every call; no fake embedder is
injected. The working database ``teachcopilot_local`` is never written by the
e2e path, and only the database this tool created is ever dropped.

Output: a human-readable PASS/FAIL/SKIP summary on stdout plus a full JSON
report. DSNs and the Neo4j password are never printed or written. Exit code 0
when no step failed (skips are fine), 2 otherwise.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from urllib.parse import urlsplit
from uuid import uuid4

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

DEFAULT_OUTPUT_DIR = Path("C:/Users/name/Documents/ChatGPT/Claude/teachcopilot-work")
DEFAULT_APP_URL = "http://127.0.0.1:8099"
# The only Neo4j endpoint the product itself uses (research_memory backends).
NEO4J_BOLT_URI = "bolt://127.0.0.1:17687"
NEO4J_BOLT_HOST, NEO4J_BOLT_PORT = "127.0.0.1", 17687
E2E_DB_PREFIX = "nirs_memory_doctor_"
LEARNER_PREFIX = "memory-doctor-"
TOPIC_ID = "math.g4.fractions.numerator_denominator"
CHAT_QUERY = "Какой вопрос помог мне исправить ошибку при сравнении дробей на одном из прошлых занятий?"
APPROVED_MEMORY_TAG = "<approved_memory_untrusted>"
MEMORY_TABLES = (
    "memory_sources", "memory_versions", "memory_jobs", "personal_graph_indexes",
    "personal_graph_source_embedding_cache", "conversation_summary_versions", "session_summary_versions",
)
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
DB_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
ENV_FLAG_MAP = (
    ("personal_memory_enabled", "LOCAL_PERSONAL_MEMORY_ENABLED"),
    ("personal_memory_retrieval_mode", "LOCAL_PERSONAL_MEMORY_RETRIEVAL_MODE"),
    ("memory_graph_channel", "LOCAL_MEMORY_GRAPH_CHANNEL"),
    ("experimental_personal_graph_enabled", "LOCAL_EXPERIMENTAL_PERSONAL_GRAPH"),
    ("context_tokens", "LOCAL_CONTEXT_TOKENS"),
    ("chat_model", "LOCAL_CHAT_MODEL"),
    ("embedding_model", "LOCAL_EMBEDDING_MODEL"),
    ("ollama_url", "OLLAMA_URL"),
)
THREE_SYNTHETIC_LESSONS = (
    {
        "session_key": "fraction-comparison-correction",
        "turns": (
            ("Я сравнил пять восьмых и три четверти и решил, что пять восьмых больше, "
             "потому что пять больше трёх.",
             "Сначала проверь, что целое разделено на равные части. Можно ли сравнивать "
             "только числители? Что изменится, если привести обе дроби к одному знаменателю?"),
            ("Я привёл дроби к знаменателю 40: получилось 25/40 и 30/40, и увидел, что "
             "три четверти больше.",
             "Верно, теперь сравнение опирается на равные части целого."),
        ),
    },
    {
        "session_key": "number-line-review",
        "turns": (
            ("Я отмечал дроби на числовом луче и написал одну вторую правее одной четверти.",
             "Почему одна вторая правее? Сколько равных частей в целом?"),
            ("Потому что половина больше четверти целого, а луч растёт слева направо.",
             "Хорошо, продолжим запись дробей на следующем занятии."),
        ),
    },
    {
        "session_key": "equivalent-fractions-exercise",
        "turns": (
            ("Я сократил четыре восьмых и получил одну вторую, и записал пару равных дробей.",
             "Как ты проверил равенство этой пары?"),
            ("Умножил одну вторую на четыре и получил снова четыре восьмых.",
             "Хорошее завершение упражнения, запишем пару в тетрадь."),
        ),
    },
)


# --------------------------------------------------------------------------- reporting


class StepLog:
    """Ordered check/e2e step results with a stable pass|fail|skip|warn status."""

    def __init__(self):
        self.steps: list[dict] = []

    def add(self, name: str, status: str, detail) -> dict:
        if status not in {"pass", "fail", "skip", "warn"}:
            raise ValueError("invalid step status: " + str(status))
        entry = {"name": name, "status": status, "detail": detail}
        self.steps.append(entry)
        return entry

    def failed(self) -> bool:
        return any(entry["status"] == "fail" for entry in self.steps)

    def counts(self) -> dict:
        result = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for entry in self.steps:
            result[entry["status"]] += 1
        return result


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_report_path() -> Path:
    path = DEFAULT_OUTPUT_DIR / f"memory-doctor-report-{utc_stamp()}.json"
    if path.exists():
        path = DEFAULT_OUTPUT_DIR / f"memory-doctor-report-{utc_stamp()}-{uuid4().hex[:6]}.json"
    return path


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def print_summary(steps: StepLog) -> None:
    for entry in steps.steps:
        marker = entry["status"].upper()
        detail = entry["detail"]
        text = json.dumps(detail, ensure_ascii=True, default=str) if isinstance(detail, dict) else str(detail)
        if len(text) > 220:
            text = text[:217] + "..."
        print(f"[{marker:>4}] {entry['name']}: {text}", flush=True)
    counts = steps.counts()
    print(f"summary: {counts['pass']} passed, {counts['skip']} skipped, "
          f"{counts['warn']} warned, {counts['fail']} failed", flush=True)


# --------------------------------------------------------------------------- DSN safety


def require_loopback_dsn(dsn: str) -> dict:
    from psycopg2.extensions import parse_dsn
    params = parse_dsn(dsn)
    candidates = [params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR")]
    values = [str(value).strip("[]").lower() for value in candidates if value]
    if not values or any(value not in LOOPBACK_HOSTS for value in values):
        raise ValueError("postgresql_must_use_loopback_only")
    return params


def dsn_summary(dsn: str) -> dict:
    """Host/port/database identity only; user and password never leave this function."""
    try:
        params = require_loopback_dsn(dsn)
    except Exception:
        return {"loopback_only": False}
    return {"loopback_only": True, "host": params.get("host"), "port": params.get("port"),
            "dbname": params.get("dbname")}


def require_loopback_app_url(app_url: str) -> str:
    parsed = urlsplit(app_url)
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS or parsed.username \
            or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("app_url_must_be_credential_free_http_loopback")
    return app_url.rstrip("/")


def http_session() -> "requests.Session":
    import requests
    session = requests.Session()
    session.trust_env = False
    return session


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- --check steps


def check_postgresql(settings, steps: StepLog) -> None:
    """Read-only probe of the working database; no migration writes."""
    try:
        require_loopback_dsn(settings.database_url)
    except Exception as exc:
        steps.add("postgresql", "fail", {"reason": type(exc).__name__})
        return
    import psycopg2
    try:
        conn = psycopg2.connect(settings.database_url, connect_timeout=10)
    except Exception as exc:
        steps.add("postgresql", "fail", {"reason": "connect_failed", "error_type": type(exc).__name__,
                                          "target": dsn_summary(settings.database_url)})
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            server = str(cur.fetchone()[0]).split(",")[0].strip()
            cur.execute("SELECT to_regclass(%s)", ("teachcopilot.schema_migrations",))
            migration_table = "teachcopilot.schema_migrations" if cur.fetchone()[0] else None
            if migration_table is None:
                cur.execute("SELECT to_regclass(%s)", ("teachcopilot.schema_version",))
                migration_table = "teachcopilot.schema_version" if cur.fetchone()[0] else None
            schema = {"table": migration_table, "version": None}
            if migration_table:
                cur.execute(f"SELECT COALESCE(max(version), 0) FROM {migration_table}")
                schema["version"] = int(cur.fetchone()[0])
            tables = {}
            for table in MEMORY_TABLES:
                cur.execute("SELECT to_regclass(%s)", (f"teachcopilot.{table}",))
                tables[table] = cur.fetchone()[0] is not None
        missing = [table for table, present in tables.items() if not present]
        detail = {"server": server, "schema": schema, "tables": tables,
                  "target": dsn_summary(settings.database_url), "read_only": True}
        if missing:
            steps.add("postgresql", "fail", {**detail, "reason": "missing_memory_tables", "missing": missing})
        elif schema["version"] is None:
            steps.add("postgresql", "fail", {**detail, "reason": "no_migration_table"})
        else:
            steps.add("postgresql", "pass", detail)
    except Exception as exc:
        steps.add("postgresql", "fail", {"reason": "probe_failed", "error_type": type(exc).__name__,
                                          "target": dsn_summary(settings.database_url)})
    finally:
        conn.close()


def check_neo4j(settings, steps: StepLog, *, graph_strict: bool) -> None:
    """Bolt handshake only when the product's password file exists; skip on failure unless --graph."""
    password_file = Path(settings.personal_graph_password_file)
    if not password_file.is_file():
        steps.add("neo4j", "skip", {"reason": "password_file_unavailable", "graph_strict": graph_strict})
        return
    try:
        import neo4j  # noqa: F401
    except Exception:
        steps.add("neo4j", "fail" if graph_strict else "skip",
                  {"reason": "neo4j_driver_unavailable", "password_file_available": True,
                   "graph_strict": graph_strict})
        return
    try:
        # The password is read and used in place; it is never printed or written.
        password = password_file.read_text(encoding="utf-8").strip()
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_BOLT_URI, auth=("neo4j", password),
                                      connection_timeout=3.0, max_connection_pool_size=1)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
    except Exception as exc:
        steps.add("neo4j", "fail" if graph_strict else "skip",
                  {"reachable": False, "reason": type(exc).__name__, "uri": NEO4J_BOLT_URI,
                   "graph_strict": graph_strict})
        return
    steps.add("neo4j", "pass", {"reachable": True, "uri": NEO4J_BOLT_URI})


def check_ollama(settings, steps: StepLog, *, live: bool) -> bool:
    """Version, configured models (from Settings only), embed probe; chat probe only with --live."""
    base = str(settings.ollama_url).rstrip("/")
    session = http_session()
    version, tags, version_error = None, None, None
    try:
        version = session.get(f"{base}/api/version", timeout=10).json()
        tags = session.get(f"{base}/api/tags", timeout=15).json()
    except Exception as exc:
        version_error = type(exc).__name__
    if not isinstance(tags, dict):
        steps.add("ollama", "fail", {"url": base, "reason": "unreachable_or_invalid_tags",
                                      "error_type": version_error})
        return False
    names = set()
    for entry in tags.get("models") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("model") or "")
            if name:
                names.add(name)

    def model_present(name: str) -> bool:
        return name in names or (":" not in name and f"{name}:latest" in names)

    missing = [name for name in dict.fromkeys((settings.embedding_model, settings.chat_model))
               if not model_present(name)]
    embed_probe = {}
    try:
        response = session.post(f"{base}/api/embed",
                                json={"model": settings.embedding_model,
                                      "input": ["Проверка локальной памяти."]},
                                timeout=max(120.0, float(settings.request_timeout)))
        response.raise_for_status()
        vectors = (response.json() or {}).get("embeddings") or []
        dimensions = len(vectors[0]) if vectors and isinstance(vectors[0], list) else None
        embed_probe = {"ok": dimensions == 1024, "dimensions": dimensions, "expected_dimensions": 1024}
    except Exception as exc:
        embed_probe = {"ok": False, "error_type": type(exc).__name__}
    chat_probe = {"attempted": False}
    if live:
        try:
            started = time.perf_counter()
            response = session.post(f"{base}/api/chat", json={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": "Ответь ровно одним словом: готов."}],
                "stream": False, "think": False,
                "options": {"num_ctx": int(settings.context_tokens), "num_predict": 16,
                            "temperature": 0, "seed": 42},
            }, timeout=float(settings.request_timeout))
            response.raise_for_status()
            content = str((((response.json() or {}).get("message") or {}).get("content")) or "").strip()
            chat_probe = {"attempted": True, "ok": bool(content), "reply_chars": len(content),
                          "elapsed_seconds": round(time.perf_counter() - started, 2)}
        except Exception as exc:
            chat_probe = {"attempted": True, "ok": False, "error_type": type(exc).__name__}
    ok = not missing and bool(embed_probe.get("ok")) and (not live or bool(chat_probe.get("ok")))
    steps.add("ollama", "pass" if ok else "fail", {
        "url": base, "version": str((version or {}).get("version") or "?"),
        "chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
        "models_present": not missing, "missing_models": missing,
        "embed_probe": embed_probe, "chat_probe": chat_probe,
    })
    return ok


def check_env_flags(settings, steps: StepLog) -> None:
    """Echo memory-relevant settings with their effective source (env name + raw value)."""
    fields = []
    for attr, env_name in ENV_FLAG_MAP:
        raw = os.getenv(env_name)
        fields.append({"field": attr, "env": env_name, "env_value": raw,
                       "source": "env" if raw is not None else "default",
                       "effective": getattr(settings, attr)})
    raw_file = os.getenv("LOCAL_PERSONAL_GRAPH_PASSWORD_FILE")
    fields.append({"field": "personal_graph_password_file", "env": "LOCAL_PERSONAL_GRAPH_PASSWORD_FILE",
                   "env_value": raw_file, "source": "env" if raw_file is not None else "default",
                   "available": Path(settings.personal_graph_password_file).is_file()})
    steps.add("env_flags", "pass", {"fields": fields})


def check_running_app(app_url: str, steps: StepLog) -> None:
    """Probe GET /health/ready on a running instance; skip when nothing listens."""
    try:
        base = require_loopback_app_url(app_url)
    except Exception as exc:
        steps.add("running_app", "fail", {"reason": type(exc).__name__})
        return
    try:
        response = http_session().get(f"{base}/health/ready", timeout=8)
    except Exception as exc:
        steps.add("running_app", "skip", {"reason": "unreachable", "error_type": type(exc).__name__,
                                           "url": base})
        return
    body = {}
    if (response.headers.get("content-type") or "").startswith("application/json"):
        try:
            body = response.json()
        except Exception:
            body = {}
    if response.status_code == 200 and body.get("status") == "ready":
        steps.add("running_app", "pass", {"url": base, "status": body.get("status"),
                                           "schema_version": body.get("schema_version"),
                                           "models": body.get("models"),
                                           "index_ready": bool((body.get("index") or {}).get("ready"))})
    else:
        detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
        steps.add("running_app", "warn", {"url": base, "http_status": response.status_code,
                                           "status": body.get("status") or detail.get("status"),
                                           "reason": detail.get("reason")})


# --------------------------------------------------------------------------- --e2e helpers


def _create_disposable_database(base_dsn: str, dbname: str):
    """Create one fresh prefixed database with TEMPLATE0; the caller owns dropping it."""
    import psycopg2
    from psycopg2 import sql
    require_loopback_dsn(base_dsn)
    conn = psycopg2.connect(base_dsn, connect_timeout=10)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(dbname)))
    finally:
        conn.close()


def _drop_disposable_database(base_dsn: str, dbname: str) -> bool:
    """Drop ONLY the disposable database this tool created (prefix asserted first)."""
    import psycopg2
    from psycopg2 import sql
    if not dbname.startswith(E2E_DB_PREFIX) or len(dbname) != len(E2E_DB_PREFIX) + 12 \
            or not DB_NAME_RE.fullmatch(dbname) \
            or any(character not in "0123456789abcdef" for character in dbname[len(E2E_DB_PREFIX):]):
        raise RuntimeError("refusing_to_drop_unexpected_database_name")
    require_loopback_dsn(base_dsn)
    conn = psycopg2.connect(base_dsn, connect_timeout=10)
    dropped = False
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = %s AND pid <> pg_backend_pid()", (dbname,))
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname)))
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            dropped = cur.fetchone() is None
    finally:
        conn.close()
    return dropped


def _import_synthetic_card(store, workdir: Path, learner_id: str) -> None:
    """Copy the static catalog and register one synthetic learner card (no real names)."""
    fixture = workdir / "learner-fixture"
    (fixture / "learner-data" / "learners").mkdir(parents=True, exist_ok=True)
    data_dir = SCRIPT_ROOT.parent / "learner-data"
    shutil.copytree(data_dir / "catalog", fixture / "learner-data" / "catalog")
    card = json.loads((data_dir / "demo" / "learners" / "demo-nika-04.json").read_text(encoding="utf-8-sig"))
    card.update(learner_id=learner_id, pseudonym="Синтетический ученик")
    (fixture / "learner-data" / "learners" / (learner_id + ".json")).write_text(
        json.dumps(card, ensure_ascii=False), encoding="utf-8")
    store.import_repository(fixture)


def _create_synthetic_accounts(store, learner_id: str, nonce: str):
    from local_app.auth import hash_password
    password = "Synthetic-only-" + nonce
    teacher = store.create_account("teacher-" + nonce + "@synthetic.invalid", "Synthetic teacher",
                                   hash_password(password), "teacher")
    store.assign_learner(teacher["id"], learner_id)
    student = store.create_student_account("student-" + nonce + "@synthetic.invalid", "Synthetic student",
                                           hash_password(password), learner_id, teacher["id"])
    return teacher, student, password


def _seed_lessons(store, learner_id: str, student_id: str, nonce: str) -> dict:
    """Three short fictional lessons; each turn pair becomes one student memory source."""
    source_ids = []
    for lesson_index, lesson in enumerate(THREE_SYNTHETIC_LESSONS):
        session = store.create_learning_session(learner_id, student_id, TOPIC_ID,
            client_session_id=f"doctor-{nonce}-{lesson['session_key']}")
        for turn_index, (student_text, assistant_text) in enumerate(lesson["turns"]):
            message_id = f"m-{lesson_index}-{turn_index}"
            reserved = store.reserve_chat_message(session["id"], learner_id, message_id,
                                                  student_text, student_id)
            if reserved.get("created") is False:
                raise RuntimeError("synthetic_message_duplicated")
            source_ids.append(str(reserved["source_id"]))
            store.complete_chat_message(session["id"], message_id,
                {"choices": [{"message": {"role": "assistant", "content": assistant_text}}]})
        store.complete_learning_session(session["id"], learner_id, student_id)
    return {"lesson_count": len(THREE_SYNTHETIC_LESSONS), "student_source_count": len(source_ids),
            "source_ids": source_ids}


def _source_projection(context: dict) -> list[dict]:
    rows = []
    for source in (context or {}).get("sources") or []:
        if isinstance(source, dict):
            rows.append({key: source.get(key) for key in
                         ("source_id", "role", "topic_id", "observed_on", "corrects_source_id")
                         if key in source})
    return rows


class ChatRequestCapture:
    """Record whether outgoing model requests carried the approved-memory block."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = 0
        self.approved_memory_requests = 0
        self._original = None

    def __enter__(self):
        outer, original = self, self.runtime.chat

        def wrapper(messages, *args, **kwargs):
            result = original(messages, *args, **kwargs)
            outer.calls += 1
            try:
                joined = "\n".join(str(row.get("content") or "") for row in (messages or [])
                                   if isinstance(row, dict) and row.get("role") == "system")
                if APPROVED_MEMORY_TAG in joined:
                    outer.approved_memory_requests += 1
            except Exception:
                pass
            return result

        self._original = original
        self.runtime.chat = wrapper
        return self

    def __exit__(self, *exc):
        self.runtime.chat = self._original
        return False


class ObservedRag:
    """Capture the memory context the RAG service actually received."""

    def __new__(cls, *args, **kwargs):
        from local_app.rag import RagService

        class _Observed(RagService):
            def answer(self, messages, *positional, **kwargs):
                self.last_memory_context = kwargs.get("memory_context")
                return super().answer(messages, *positional, **kwargs)

        return _Observed(*args, **kwargs)


def _login(client, email: str, password: str) -> dict:
    response = client.post("/auth/login", json={"email": email, "password": password})
    if response.status_code != 200:
        raise RuntimeError("synthetic_login_failed_" + str(response.status_code))
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def _run_memory_cycle(*, steps: StepLog, cycle: str, method: str, store, runtime, worker, client,
                      learner_id: str, teacher: dict, student: dict, password: str, nonce: str,
                      version_extra: dict | None = None) -> dict:
    """One full teacher-prepare -> approve -> student-chat cycle for one memory method."""
    report: dict = {"method": method, "attempted": True}
    teacher_headers = _login(client, teacher["email"], password)
    queued = client.post(f"/learners/{learner_id}/next-step", headers=teacher_headers,
                         json={"topic_id": TOPIC_ID, "method": method})
    report["queue_http_status"] = queued.status_code
    if queued.status_code != 202:
        steps.add(f"e2e_{cycle}_teacher_job", "fail", {**report, "reason": "teacher_queue_rejected"})
        report["ok"] = False
        return report
    job_id = str(queued.json()["job_id"])
    started = time.perf_counter()
    try:
        finished = worker.run_once()
    except Exception as exc:
        steps.add(f"e2e_{cycle}_teacher_job", "fail",
                  {**report, "reason": "worker_raised", "error_type": type(exc).__name__})
        report["ok"] = False
        return report
    report.update(job_id=job_id, job_status=finished.get("status") if finished else None,
                  elapsed_seconds=round(time.perf_counter() - started, 2),
                  model_calls=len(getattr(worker, "calls", []) or []))
    if not finished or str(finished.get("job_id") or finished.get("id")) != job_id \
            or finished.get("status") != "completed":
        report["job_error"] = str(finished.get("error") or "?")[:400] if finished else "no_job_claimed"
        steps.add(f"e2e_{cycle}_teacher_job", "fail", report)
        report["ok"] = False
        return report
    result = finished.get("result") or {}
    version_id = str(result.get("memory_version_id"))
    report.update(version_id=version_id, teacher_action=result.get("action"),
                  teacher_reason=result.get("reason"))
    steps.add(f"e2e_{cycle}_teacher_job", "pass", report)

    approval = {"version_id": version_id}
    snapshot = client.get(f"/learners/{learner_id}/memory", headers=teacher_headers)
    if snapshot.status_code != 200:
        steps.add(f"e2e_{cycle}_teacher_approval", "fail",
                  {**approval, "reason": "memory_view_failed", "http_status": snapshot.status_code})
        report["ok"] = False
        return report
    confirmed = client.post(f"/learners/{learner_id}/memory/versions/{version_id}/confirm",
                            headers=teacher_headers,
                            json={"expected_revision": snapshot.json()["revision"]})
    if confirmed.status_code != 200:
        steps.add(f"e2e_{cycle}_teacher_approval", "fail",
                  {**approval, "reason": "confirm_rejected", "http_status": confirmed.status_code})
        report["ok"] = False
        return report
    version = confirmed.json().get("version") or {}
    approved_ids = sorted(map(str, ((version.get("context") or {}).get("evidence_ids") or [])))
    approval.update(approved=bool(version.get("approved")), approved_evidence_count=len(approved_ids),
                    approved_evidence_ids=approved_ids)
    steps.add(f"e2e_{cycle}_teacher_approval", "pass", approval)

    student_headers = _login(client, student["email"], password)
    session = store.create_learning_session(learner_id, student["id"], TOPIC_ID,
                                            client_session_id=f"doctor-chat-{cycle}-{nonce}")
    payload = {"learning_session_id": str(session["id"]),
               "client_message_id": f"doctor-question-{cycle}-{nonce}",
               "messages": [{"role": "user", "content": CHAT_QUERY}],
               "metadata": {"memory_method": method}}
    capture = ChatRequestCapture(runtime)
    started = time.perf_counter()
    with capture:
        chat = client.post("/v1/chat/completions", headers=student_headers, json=payload)
    body = chat.json() if (chat.headers.get("content-type") or "").startswith("application/json") else {}
    answer = str((((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
    memory_context = getattr(client.app.state.rag_service, "last_memory_context", None) or {}
    diagnostics = memory_context.get("retrieval_diagnostics") or {}
    graph_channel = diagnostics.get("graph_channel")
    boundary = {
        "approved_evidence_count": len(approved_ids),
        "chat_evidence_ids": sorted(map(str, memory_context.get("evidence_ids") or [])),
    }
    boundary["chat_ids_within_approved_set"] = (
        bool(approved_ids) and set(boundary["chat_evidence_ids"]) <= set(approved_ids))
    chat_report = {
        "method": method, "http_status": chat.status_code,
        "reply": answer or str(body.get("detail") or "")[:400],
        "reply_chars": len(answer),
        "request_contained_approved_memory_untrusted": capture.approved_memory_requests > 0,
        "outgoing_model_calls": capture.calls,
        "memory_version_id": str(body.get("memory_version_id") or "") or None,
        "memory_method": body.get("memory_method"),
        "selected_sources": _source_projection(memory_context),
        "retrieval_diagnostics_graph_channel": graph_channel,
        "graph_channel_used": bool(graph_channel and graph_channel.get("source_ids")),
        "graph_channel_reason": (graph_channel or {}).get("reason")
            if isinstance(graph_channel, dict) else None,
        "approval_boundary": boundary,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    if version_extra:
        chat_report.update(version_extra)
    ok = (chat.status_code == 200 and bool(answer)
          and chat_report["request_contained_approved_memory_untrusted"]
          and boundary["chat_ids_within_approved_set"])
    steps.add(f"e2e_{cycle}_student_chat", "pass" if ok else "fail", chat_report)
    if isinstance(graph_channel, dict) and (chat_report["graph_channel_used"]
                                            or chat_report["graph_channel_reason"]):
        steps.add(f"e2e_{cycle}_graph_channel_report", "pass", {"graph_channel": graph_channel})
    else:
        steps.add(f"e2e_{cycle}_graph_channel_report", "warn",
                  {"reason": "retrieval_diagnostics_graph_channel_absent"})
    report["ok"] = ok
    report["chat"] = chat_report
    return report


def _graph_cycle_preflight(settings) -> str | None:
    """Reason the experimental personal_graph cycle cannot run, or None when it can."""
    if not Path(settings.personal_graph_password_file).is_file():
        return "password_file_unavailable"
    if not port_open(NEO4J_BOLT_HOST, NEO4J_BOLT_PORT, timeout=2.0):
        return "neo4j_unreachable"
    try:
        import graphiti_core  # noqa: F401
        import neo4j  # noqa: F401
    except Exception as exc:
        return f"driver_unavailable:{type(exc).__name__}"
    return None


def run_e2e(base_settings, steps: StepLog, *, graph: bool) -> dict:
    """Disposable-database e2e of the real memory pipeline; drops only its own database."""
    from psycopg2.extensions import make_dsn

    dbname = E2E_DB_PREFIX + uuid4().hex[:12]
    e2e: dict = {"database": dbname, "database_prefix": E2E_DB_PREFIX, "synthetic_only": True,
                 "graph_requested": graph, "dropped": False}
    workdir = Path(tempfile.mkdtemp(prefix="memory-doctor-e2e-"))
    e2e["workdir"] = str(workdir)
    store = None
    client = None
    try:
        _create_disposable_database(base_settings.database_url, dbname)
        dsn = make_dsn(base_settings.database_url, dbname=dbname)
        from local_app.store import Store
        store = Store(dsn)
        applied = store.migrate()
        e2e["dsn_target"] = dsn_summary(dsn)
        e2e["migrations_applied"] = applied
        e2e["schema_version"] = store.schema_version()
        settings = replace(base_settings, database_url=dsn, generation_scheduler_dsn=dsn,
                           request_timeout=300.0,
                           # The project's own memory-path e2e scripts run 16k contexts:
                           # the resolver's fixed prompt needs the complete-memory budget
                           # that an 8k context cannot fit.
                           context_tokens=16384,
                           experimental_personal_graph_enabled=bool(graph),
                           memory_response_journal_path=str(workdir / "memory-responses.sqlite3"),
                           chat_response_journal_path=str(workdir / "chat-responses.sqlite3"))
        steps.add("e2e_disposable_database", "pass",
                  {"database": dbname, "migrations_applied": applied,
                   "schema_version": e2e["schema_version"], "context_tokens": settings.context_tokens,
                   "context_note": "16k complete-memory budget, as in the project's own "
                                   "personal-memory e2e scripts",
                   "experimental_personal_graph_enabled": settings.experimental_personal_graph_enabled,
                   "target": e2e["dsn_target"]})

        from fastapi.testclient import TestClient
        from local_app.answer_adapter import AnswerAdapter
        from local_app.api import create_app
        from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
        from local_app.memory_service import MemoryWorker
        from local_app.runtime import OllamaRuntime

        nonce = uuid4().hex[:12]
        learner_id = LEARNER_PREFIX + nonce
        _import_synthetic_card(store, workdir, learner_id)
        teacher, student, password = _create_synthetic_accounts(store, learner_id, nonce)
        seeded = _seed_lessons(store, learner_id, student["id"], nonce)
        note = store.add_memory_source(learner_id, TOPIC_ID,
            "На занятии ученик сравнил 5/8 и 3/4, привёл дроби к знаменателю 40 и исправил "
            "сравнение после вопроса о равных частях.", teacher["id"], approved=True)
        e2e["seed"] = {"learner_id": learner_id, "topic_id": TOPIC_ID, **seeded,
                       "teacher_note_source_id": str(note["source_id"])}
        steps.add("e2e_seed_synthetic_data", "pass", e2e["seed"])

        runtime = OllamaRuntime(settings)
        rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
        journal = MemoryResponseJournal(settings.memory_response_journal_path,
                                        database_identity=database_fingerprint(dsn))
        worker = MemoryWorker(store, runtime, response_journal=journal)
        app = create_app(settings=settings, store=store, runtime=runtime, rag_service=rag)
        client = TestClient(app)
        e2e["runtime"] = {"chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
                          "context_tokens": settings.context_tokens, "real_ollama": True}

        pm = _run_memory_cycle(steps=steps, cycle="personal_memory", method="personal_memory",
                               store=store, runtime=runtime, worker=worker, client=client,
                               learner_id=learner_id, teacher=teacher, student=student,
                               password=password, nonce=nonce)
        e2e["personal_memory_cycle"] = pm

        if graph:
            reason = _graph_cycle_preflight(settings)
            if reason:
                steps.add("e2e_personal_graph_cycle", "skip", {"reason": reason})
                e2e["personal_graph_cycle"] = {"attempted": False, "skipped_reason": reason}
            else:
                pg = _run_memory_cycle(steps=steps, cycle="personal_graph", method="personal_graph",
                                       store=store, runtime=runtime, worker=worker, client=client,
                                       learner_id=learner_id, teacher=teacher, student=student,
                                       password=password, nonce=nonce + "g",
                                       version_extra={"experimental_personal_graph_enabled": True})
                e2e["personal_graph_cycle"] = pg
                # After the graph cycle the unified personal_memory channel should find
                # the freshly built ready index and report graph_channel used=True.
                try:
                    from local_app.personal_graph_contract import personal_graph_snapshot_fingerprint
                    from local_app.personal_graph_index import (find_ready_personal_graph_index,
                                                                personal_graph_config_fingerprint)
                    from local_app.personal_graph_memory import source_lessons
                    from local_app.personal_memory import PersonalMemoryProvider
                    snapshot = store.get_learner_memory(learner_id, None,
                                                        include_personal_graph_snapshot=True)
                    model = PersonalMemoryProvider(store, runtime)._application_model()
                    lessons, _source_map = source_lessons(
                        {**snapshot, "sources": snapshot["personal_graph_sources"]}, learner_id)
                    fingerprint = personal_graph_snapshot_fingerprint(snapshot["personal_graph_sources"])
                    config_fingerprint = personal_graph_config_fingerprint(model, model.model_identities)
                    index = find_ready_personal_graph_index(store.dsn, learner_id, fingerprint,
                                                            lessons, config_fingerprint)
                    registry = {"ready_index_found": index is not None,
                                "run_id": str(index["run_id"]) if index else None,
                                "manifest_keys": sorted((index.get("manifest") or {}).keys()) if index else []}
                    steps.add("e2e_graph_channel_registry", "pass" if index else "warn",
                              {**registry, "interpretation": "read_only_registry_lookup_only"})
                except Exception as exc:
                    steps.add("e2e_graph_channel_registry", "warn",
                              {"reason": f"registry_check_failed:{type(exc).__name__}"})
        e2e["ok"] = bool((e2e.get("personal_memory_cycle") or {}).get("ok")) and \
            (not graph or (e2e.get("personal_graph_cycle") or {}).get("ok", True) is not False)
        return e2e
    except Exception as exc:
        import traceback
        frames = traceback.extract_tb(exc.__traceback__)
        origin = f"{Path(frames[-1].filename).name}:{frames[-1].name}:{frames[-1].lineno}" if frames else "?"
        steps.add("e2e_failed", "fail", {"error_type": type(exc).__name__,
                                          "message": str(exc)[:400], "origin": origin})
        e2e["ok"] = False
        return e2e
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        try:
            e2e["dropped"] = _drop_disposable_database(base_settings.database_url, dbname)
        except Exception as exc:
            e2e["dropped"] = False
            e2e["drop_error"] = type(exc).__name__
            steps.add("e2e_disposable_database_drop", "warn",
                      {"database": dbname, "reason": f"drop_failed:{type(exc).__name__}"})
        else:
            steps.add("e2e_disposable_database_drop", "pass", {"database": dbname, "dropped": True})
        if e2e.get("ok") and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Health-check and e2e verification for the «Алёна» memory subsystem")
    parser.add_argument("--check", action="store_true",
                        help="run read-only health checks (default when no flag is given)")
    parser.add_argument("--e2e", action="store_true",
                        help="run --check first, then an end-to-end memory cycle on a disposable database")
    parser.add_argument("--graph", action="store_true",
                        help="treat an unreachable Neo4j as a failure and try the experimental "
                             "personal_graph cycle during --e2e")
    parser.add_argument("--live", action="store_true",
                        help="also run a real chat probe against Ollama during --check")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON report path (default: memory-doctor-report-<UTC timestamp>.json "
                             "in the teachcopilot-work directory)")
    parser.add_argument("--app-url", default=DEFAULT_APP_URL,
                        help="running instance to probe during --check (default %(default)s)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Flags are additive: --check is the default; --e2e implies the check first;
    # --graph and --live only tighten or extend the checks they touch.
    run_check = True
    run_e2e_requested = bool(args.e2e)
    try:
        output = args.output if args.output else default_report_path()
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"report already exists: {output}")
        app_url = require_loopback_app_url(args.app_url)
    except Exception as exc:
        print(f"configuration error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2

    from local_app.config import Settings
    settings = Settings.from_env()
    steps = StepLog()
    report = {
        "tool": "memory_doctor", "started_at_utc": utcnow_iso(),
        "mode": {"check": run_check, "e2e": run_e2e_requested,
                 "graph": bool(args.graph), "live": bool(args.live)},
        "working_database": dsn_summary(settings.database_url), "synthetic_only_e2e": True,
        "ok": False,
    }
    if run_check:
        check_postgresql(settings, steps)
        check_neo4j(settings, steps, graph_strict=bool(args.graph))
        check_ollama(settings, steps, live=bool(args.live))
        check_env_flags(settings, steps)
        check_running_app(app_url, steps)
    if run_e2e_requested:
        # Only the steps the e2e actually depends on block it. A failed strict
        # --graph Neo4j probe still fails the overall run (exit 2) but must not
        # prevent the personal_memory cycle from being exercised; its graph
        # part skips on its own preflight instead.
        blocking = [entry["name"] for entry in steps.steps
                    if entry["status"] == "fail"
                    and entry["name"].split("_", 1)[0] in {"postgresql", "ollama"}]
        if blocking:
            steps.add("e2e", "skip", {"reason": "check_steps_failed", "failed_steps": blocking})
            report["e2e"] = {"attempted": False, "skipped_reason": "check_steps_failed",
                             "failed_steps": blocking}
        else:
            report["e2e"] = run_e2e(settings, steps, graph=bool(args.graph))

    report["finished_at_utc"] = utcnow_iso()
    report["steps"] = steps.steps
    counts = steps.counts()
    report["counts"] = counts
    report["ok"] = counts["fail"] == 0
    try:
        write_report(output, report)
    except Exception as exc:
        print(f"could not write report {output}: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2
    print("memory_doctor "
          + ("check+e2e" if run_e2e_requested else "check")
          + (" [graph]" if args.graph else "") + (" [live]" if args.live else ""), flush=True)
    print_summary(steps)
    print(f"report: {output}", flush=True)
    print(f"exit: {0 if report['ok'] else 2}", flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit local settings; importing the API never starts services."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


def require_local_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Ollama must use an HTTP loopback endpoint for local operation")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not contain credentials, query or fragment")
    return value.rstrip("/")


def _autonomous_accounts_from_env() -> tuple[str, ...] | None:
    value = os.getenv("LOCAL_AUTONOMOUS_DIVISION_ACCOUNT_IDS")
    if value is None:
        return None
    try:
        accounts = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Autonomous division account IDs must be a JSON array of canonical UUIDs") from exc
    if not isinstance(accounts, list):
        raise ValueError("Autonomous division account IDs must be a JSON array of canonical UUIDs")
    return tuple(accounts)


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://teachcopilot@127.0.0.1:5434/teachcopilot_local"
    ollama_url: str = "http://127.0.0.1:11435"
    chat_model: str = "gemma4:12b"
    embedding_model: str = "bge-m3"
    # 16384 is the documented minimum for the personal-memory chat path: at 8192
    # the resolver prompt alone leaves no room for prior learner questions
    # (user decision 2026-11-22). LOCAL_CONTEXT_TOKENS overrides it.
    context_tokens: int = 16384
    request_timeout: float = 180.0
    max_output_tokens: int = 1024
    session_hours: int = 12
    # Memory job lease in seconds (LOCAL_JOB_LEASE_SECONDS): the worker claim,
    # the pre-model-call heartbeat and the interactive chat claim all use it.
    job_lease_seconds: int = 600
    rag_limit: int = 3
    minimum_similarity: float = 0.35
    learning_operation_selection_enabled: bool = False
    checked_division_context_enabled: bool = False
    autonomous_division_enabled: bool = False
    autonomous_division_account_ids: tuple[str, ...] | None = field(default=None, repr=False)
    demo_login_enabled: bool = False
    demo_credentials_file: str = str(Path(__file__).resolve().parents[1] / ".local" / "rich-demo-credentials.json")
    generation_scheduler_dsn: str | None = field(default=None, repr=False)
    generation_run_budget_path: str | None = None
    memory_response_journal_path: str = str(Path(__file__).resolve().parents[1] / ".local" / "memory-worker-responses.sqlite3")
    chat_response_journal_path: str | None = None
    personal_memory_enabled: bool = True
    personal_memory_retrieval_mode: str = "hybrid"
    memory_graph_channel: bool = True
    experimental_personal_graph_enabled: bool = False
    personal_graph_retrieval_mode: str = "graph_hybrid"
    personal_graph_password_file: str = str(Path(__file__).resolve().parents[3] / ".local" /
        "neo4j-runtime" / "neo4j-community-5.26.32" / "personal-memory-password.txt")

    def __post_init__(self):
        require_local_url(self.ollama_url)
        if self.personal_memory_retrieval_mode not in {"chronological", "lexical", "vector", "hybrid"}:
            raise ValueError("Unknown personal memory retrieval mode")
        if self.personal_graph_retrieval_mode not in {"chronological", "lexical", "vector", "hybrid", "graph_hybrid"}:
            raise ValueError("Unknown personal graph retrieval mode")
        if self.context_tokens < 2048 or self.max_output_tokens < 128:
            raise ValueError("Invalid context/output token limits")
        if self.request_timeout <= 0 or not 1 <= self.session_hours <= 168:
            raise ValueError("Invalid timeout/session lifetime")
        if not 60 <= self.job_lease_seconds <= 3600:
            raise ValueError("Invalid job lease seconds")
        if self.experimental_personal_graph_enabled:
            if self.context_tokens not in (16384, 32768):
                raise ValueError("Experimental personal graph requires a 16k or 32k context")
            if self.chat_model != "gemma4:12b" or self.embedding_model.split(":")[0] != "bge-m3":
                raise ValueError("Experimental personal graph requires local Gemma and BGE-M3")
        accounts = self.autonomous_division_account_ids
        if accounts is not None:
            if not isinstance(accounts, tuple):
                raise ValueError("Autonomous division account IDs must be a tuple of canonical UUIDs")
            try:
                valid = all(isinstance(account, str) and str(UUID(account)) == account for account in accounts)
            except (ValueError, TypeError, AttributeError):
                valid = False
            if not valid or len(set(accounts)) != len(accounts):
                raise ValueError("Autonomous division account IDs must be unique canonical UUIDs")

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".local.env", override=False)
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            chat_model=os.getenv("LOCAL_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", cls.embedding_model),
            context_tokens=int(os.getenv("LOCAL_CONTEXT_TOKENS", str(cls.context_tokens))),
            request_timeout=float(os.getenv("LOCAL_REQUEST_TIMEOUT", "180")),
            max_output_tokens=int(os.getenv("LOCAL_MAX_OUTPUT_TOKENS", "1024")),
            session_hours=int(os.getenv("LOCAL_SESSION_HOURS", "12")),
            job_lease_seconds=int(os.getenv("LOCAL_JOB_LEASE_SECONDS", "600")),
            minimum_similarity=float(os.getenv("LOCAL_MIN_SIMILARITY", "0.35")),
            learning_operation_selection_enabled=os.getenv('LOCAL_LEARNING_OPERATION_SELECTION', '0') == '1',
            checked_division_context_enabled=os.getenv('LOCAL_CHECKED_DIVISION_CONTEXT', '0') == '1',
            autonomous_division_enabled=os.getenv('LOCAL_AUTONOMOUS_DIVISION_ENABLED', '0') == '1',
            autonomous_division_account_ids=_autonomous_accounts_from_env(),
            demo_login_enabled=os.getenv('LOCAL_DEMO_LOGIN', '0') == '1',
            demo_credentials_file=os.getenv('LOCAL_DEMO_CREDENTIALS_FILE', cls.demo_credentials_file),
            generation_scheduler_dsn=os.getenv('LOCAL_GENERATION_SCHEDULER_DSN') or None,
            generation_run_budget_path=os.getenv('LOCAL_GENERATION_RUN_BUDGET_PATH') or None,
            memory_response_journal_path=os.getenv('LOCAL_MEMORY_RESPONSE_JOURNAL', cls.memory_response_journal_path),
            chat_response_journal_path=os.getenv('LOCAL_CHAT_RESPONSE_JOURNAL', str(
                Path(__file__).resolve().parents[1] / '.local' / 'chat-responses.sqlite3')),
            personal_memory_enabled=os.getenv("LOCAL_PERSONAL_MEMORY_ENABLED", "1") == "1",
            personal_memory_retrieval_mode=os.getenv("LOCAL_PERSONAL_MEMORY_RETRIEVAL_MODE", "hybrid"),
            memory_graph_channel=os.getenv("LOCAL_MEMORY_GRAPH_CHANNEL", "1") == "1",
            experimental_personal_graph_enabled=os.getenv("LOCAL_EXPERIMENTAL_PERSONAL_GRAPH", "0") == "1",
            personal_graph_retrieval_mode=os.getenv("LOCAL_PERSONAL_GRAPH_RETRIEVAL_MODE", "graph_hybrid"),
            personal_graph_password_file=os.getenv("LOCAL_PERSONAL_GRAPH_PASSWORD_FILE", cls.personal_graph_password_file),
        )

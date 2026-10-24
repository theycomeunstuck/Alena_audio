"""Explicit local settings; importing the API never starts services."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def require_local_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Ollama must use an HTTP loopback endpoint for local operation")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not contain credentials, query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://teachcopilot@127.0.0.1:5434/teachcopilot_local"
    ollama_url: str = "http://127.0.0.1:11435"
    chat_model: str = "gemma4:12b"
    embedding_model: str = "bge-m3"
    context_tokens: int = 8192
    request_timeout: float = 180.0
    max_output_tokens: int = 1024
    session_hours: int = 12
    rag_limit: int = 3
    minimum_similarity: float = 0.35
    learning_operation_selection_enabled: bool = False
    checked_division_context_enabled: bool = False
    autonomous_division_enabled: bool = False
    demo_login_enabled: bool = False
    demo_credentials_file: str = str(Path(__file__).resolve().parents[1] / ".local" / "rich-demo-credentials.json")
    generation_scheduler_dsn: str | None = field(default=None, repr=False)
    generation_run_budget_path: str | None = None
    memory_response_journal_path: str = str(Path(__file__).resolve().parents[1] / ".local" / "memory-worker-responses.sqlite3")
    chat_response_journal_path: str | None = None

    def __post_init__(self):
        require_local_url(self.ollama_url)
        if self.context_tokens < 2048 or self.max_output_tokens < 128:
            raise ValueError("Invalid context/output token limits")
        if self.request_timeout <= 0 or not 1 <= self.session_hours <= 168:
            raise ValueError("Invalid timeout/session lifetime")

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".local.env", override=False)
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            chat_model=os.getenv("LOCAL_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", cls.embedding_model),
            context_tokens=int(os.getenv("LOCAL_CONTEXT_TOKENS", "8192")),
            request_timeout=float(os.getenv("LOCAL_REQUEST_TIMEOUT", "180")),
            max_output_tokens=int(os.getenv("LOCAL_MAX_OUTPUT_TOKENS", "1024")),
            session_hours=int(os.getenv("LOCAL_SESSION_HOURS", "12")),
            minimum_similarity=float(os.getenv("LOCAL_MIN_SIMILARITY", "0.35")),
            learning_operation_selection_enabled=os.getenv('LOCAL_LEARNING_OPERATION_SELECTION', '0') == '1',
            checked_division_context_enabled=os.getenv('LOCAL_CHECKED_DIVISION_CONTEXT', '0') == '1',
            autonomous_division_enabled=os.getenv('LOCAL_AUTONOMOUS_DIVISION_ENABLED', '0') == '1',
            demo_login_enabled=os.getenv('LOCAL_DEMO_LOGIN', '0') == '1',
            demo_credentials_file=os.getenv('LOCAL_DEMO_CREDENTIALS_FILE', cls.demo_credentials_file),
            generation_scheduler_dsn=os.getenv('LOCAL_GENERATION_SCHEDULER_DSN') or None,
            generation_run_budget_path=os.getenv('LOCAL_GENERATION_RUN_BUDGET_PATH') or None,
            memory_response_journal_path=os.getenv('LOCAL_MEMORY_RESPONSE_JOURNAL', cls.memory_response_journal_path),
            chat_response_journal_path=os.getenv('LOCAL_CHAT_RESPONSE_JOURNAL', str(
                Path(__file__).resolve().parents[1] / '.local' / 'chat-responses.sqlite3')),
        )

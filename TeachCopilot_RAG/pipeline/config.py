# pipeline/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL     = os.getenv("DATABASE_URL", "postgresql://localhost/teachcopilot")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SCORE    = float(os.getenv("RAG_MIN_SCORE", "0.5"))
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_CHILD_ID = os.getenv("DEFAULT_CHILD_ID", "")

# JSON learner cards live alongside the RAG project in this repository.  An
# explicit path is useful when the service and data are deployed separately.
_RAG_ROOT = Path(__file__).resolve().parent.parent
LEARNER_DATA_DIR = Path(
    os.getenv("TEACHCOPILOT_LEARNER_DATA_DIR", str(_RAG_ROOT.parent / "learner-data"))
).expanduser().resolve()

# Runtime profile:
# - cpu_debug: local development on machines without a GPU; favors CPU and small models.
# - production: production can override every model/device through env.
RUNTIME_PROFILE = os.getenv("TEACHCOPILOT_RUNTIME_PROFILE", "cpu_debug").strip().lower()

# Embedding model: try multilingual (better for Russian), fallback to MiniLM
_EMBEDDING_PREFERRED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBEDDING_FALLBACK  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "auto")

# Device for embedding model: cpu_debug defaults to CPU. Production may use "auto" or "cuda".
_DEFAULT_EMBEDDING_DEVICE = "cpu" if RUNTIME_PROFILE == "cpu_debug" else "auto"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _DEFAULT_EMBEDDING_DEVICE).strip().lower()

# Main chat proxy. If CHAT_MODEL is empty, server.py forwards the caller's model
# unchanged and lets the upstream OpenAI-compatible runtime choose its default.
CHAT_API_BASE_URL = os.getenv("TEACHCOPILOT_CHAT_API_BASE_URL", os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
CHAT_MODEL = os.getenv("TEACHCOPILOT_CHAT_MODEL", os.getenv("LMSTUDIO_MODEL", "")).strip()
CHAT_REQUEST_TIMEOUT_SEC = float(os.getenv("TEACHCOPILOT_CHAT_TIMEOUT_SEC", "180"))

# Streaming remains off by default. "diagnostic" keeps the non-stream response but
# logs when Open WebUI asked for stream=True.
STREAM_MODE = os.getenv("TEACHCOPILOT_STREAM_MODE", "off").strip().lower()
DEBUG_RAG = _env_bool("TEACHCOPILOT_DEBUG_RAG", False)

# Speaker mode:
# - child_only: every unknown user gets the default child prompt/profile.
# - mapped_or_adult: mapped users are children; unknown users get adult mode.
SPEAKER_MODE = os.getenv("TEACHCOPILOT_SPEAKER_MODE", "child_only").strip().lower()

# Profile Agent — uses SEPARATE LLM endpoint (not Open WebUI proxy)
PROFILE_LLM_API_URL = os.getenv("PROFILE_LLM_API_URL", CHAT_API_BASE_URL).rstrip("/")
PROFILE_LLM_MODEL   = os.getenv("PROFILE_LLM_MODEL", CHAT_MODEL).strip()
PROFILE_UPDATE_INTERVAL = int(os.getenv("PROFILE_UPDATE_INTERVAL", "5"))

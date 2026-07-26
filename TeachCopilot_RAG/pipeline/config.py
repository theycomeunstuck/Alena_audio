# pipeline/config.py
import logging
import os
from pathlib import Path

_HERE = Path(__file__).resolve()


def _read_env_files() -> None:
    """Minimal stdlib reader for ``.env``, used when python-dotenv is missing.

    The ZPD and task-design modules are deliberately runnable on a bare
    interpreter, so configuration must not depend on an installed package.
    Existing environment variables always win: a value exported in the shell is
    a deliberate override of the file.
    """
    candidates = [
        _HERE.parent.parent / ".env",         # TeachCopilot_RAG/.env
        _HERE.parent.parent.parent / ".env",  # корень проекта
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit on a bare interpreter
    _read_env_files()

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

# --- LLM endpoint -----------------------------------------------------------
# Presets so a provider can be selected by name instead of remembering URLs.
# Everything is OpenAI-compatible: the same code talks to LM Studio and to a
# cloud API, only the base URL, the model and the API key change.
PROVIDER_PRESETS = {
    "lmstudio":   ("http://127.0.0.1:1234/v1", ""),
    "deepseek":   ("https://api.deepseek.com/v1", "deepseek-chat"),
    "openai":     ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", ""),
    "groq":       ("https://api.groq.com/openai/v1", ""),
    "together":   ("https://api.together.xyz/v1", ""),
    "mistral":    ("https://api.mistral.ai/v1", ""),
}

AI_PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
_preset_url, _preset_model = PROVIDER_PRESETS.get(AI_PROVIDER, ("http://127.0.0.1:1234/v1", ""))

# Main chat endpoint. If CHAT_MODEL is empty, server.py forwards the caller's model
# unchanged and lets the upstream OpenAI-compatible runtime choose its default.
CHAT_API_BASE_URL = os.getenv(
    "TEACHCOPILOT_CHAT_API_BASE_URL", os.getenv("LMSTUDIO_BASE_URL", _preset_url)
).rstrip("/")
CHAT_MODEL = os.getenv("TEACHCOPILOT_CHAT_MODEL", os.getenv("LMSTUDIO_MODEL", _preset_model)).strip()
CHAT_REQUEST_TIMEOUT_SEC = float(os.getenv("TEACHCOPILOT_CHAT_TIMEOUT_SEC", "180"))

# Ключ отправляется в заголовке Authorization. Локальные рантаймы его не
# требуют, облачные — требуют. Значение никогда не логируется и не печатается.
CHAT_API_KEY = (
    os.getenv("TEACHCOPILOT_API_KEY")
    or os.getenv("AI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
).strip()

# Как просить у модели строгий JSON:
#   auto        — пробуем json_object и молча откатываемся, если рантайм не умеет;
#   json_object — принудительно (режим JSON у OpenAI-совместимых API);
#   json_schema — строгая схема, поддерживается не везде;
#   off         — только текстом промпта.
JSON_MODE = os.getenv("TEACHCOPILOT_JSON_MODE", "auto").strip().lower()

# Один повторный запрос с конкретной претензией, если ответ нарушил контракт.
# Слабые модели со второй попытки обычно попадают, а стоит это одну генерацию.
MAX_CONTRACT_RETRIES = int(os.getenv("TEACHCOPILOT_CONTRACT_RETRIES", "1"))

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

# --- ЗБР stage 4: individual task design ------------------------------------
# How many tasks one individual card holds. The lesson deck shows three.
TASK_CARD_TASK_COUNT = int(os.getenv("TEACHCOPILOT_TASK_COUNT", "3"))
# Low temperature: the task must follow the curriculum, not be creative about it.
# The storyline gets its variety from the child's interests, not from sampling.
TASK_DESIGN_TEMPERATURE = float(os.getenv("TEACHCOPILOT_TASK_TEMPERATURE", "0.4"))
TASK_DESIGN_TIMEOUT_SEC = float(os.getenv("TEACHCOPILOT_TASK_TIMEOUT_SEC", "180"))

# How mastery/independence get into the card after a lesson:
# - auto_ema        (default): code recomputes them from the observed help level.
# - tutor_confirmed: the same numbers are only proposed; the tutor confirms per topic.
MASTERY_MODE = os.getenv("TEACHCOPILOT_MASTERY_MODE", "auto_ema").strip().lower()
MASTERY_MODES = ("auto_ema", "tutor_confirmed")

# Сколько учеников уходит в модель за один запрос при разборе итогов урока.
# Чем больше ростер, тем выше шанс, что модель припишет ошибку одного ребёнка
# другому, и тем ближе промпт к пределу контекста: 17 учеников — это уже ~4000
# токенов на входе, что не влезает в типовое окно 4096.
ROSTER_BATCH_SIZE = int(os.getenv("TEACHCOPILOT_ROSTER_BATCH", "5"))

# Порог, после которого промпт считается опасно большим для локальной модели.
# Предупреждение лучше, чем молча обрезанный на середине JSON.
PROMPT_WARN_CHARS = int(os.getenv("TEACHCOPILOT_PROMPT_WARN_CHARS", "12000"))

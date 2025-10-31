# app/settings.py
import os
from pathlib import Path
from core.config import TTS_CKPT_PATH, VOCAB_FILE, device

# ---- Определяем корень проекта ----
def _find_project_root(start: Path) -> Path:
    """Поднимаемся вверх в поисках маркеров корня (pyproject.toml / .git)."""
    for p in [*start.parents, start]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start.parents[1]  # запасной вариант: на два уровня вверх

# Файл settings.py обычно в app/, поднимаемся от него
PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

# Единый STORAGE
# Можно переопределить через ENV (например, для docker/CI):
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_ROOT / "storage")).resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Папка эмбеддингов
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", STORAGE_DIR / "embeddings")).resolve()
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Папка клонов голосов
VOICES_DIR = Path(os.getenv("VOICES_DIR", STORAGE_DIR / "voices_TTS")).resolve()
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Остальные TTS-параметры
# Set environment variable if not already set
if not os.getenv("F5TTS_CKPT_PATH"):
    os.environ["F5TTS_CKPT_PATH"] = TTS_CKPT_PATH
    print(f"🔧 Set F5TTS_CKPT_PATH environment variable to: {TTS_CKPT_PATH}")
if not os.getenv("VOCAB_FILE_PATH"):
    os.environ["VOCAB_FILE_PATH"] = VOCAB_FILE

VOCAB_FILE_PATH = os.getenv("VOCAB_FILE_PATH", VOCAB_FILE)


F5TTS_CKPT_PATH = os.getenv("F5TTS_CKPT_PATH", TTS_CKPT_PATH) #path to model.pt (.spt)
F5TTS_VOCODER_NAME = os.getenv("F5TTS_VOCODER_NAME", "vocos")
F5TTS_VOCODER_CKPT = os.getenv("F5TTS_VOCODER_CKPT", "")
DEVICE = os.getenv("DEVICE", device)
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
TTS_MAX_SECONDS = int(os.getenv("TTS_MAX_SECONDS", "25"))
TTS_NFE_STEPS   = int(os.getenv("TTS_NFE_STEPS", "16"))


#ASR
ASR_COMPUTE_TYPE = 'bfloat16'
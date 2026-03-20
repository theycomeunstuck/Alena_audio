# app/settings.py
import os
from pathlib import Path
from core.config import device



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





DEVICE = os.getenv("DEVICE", device)
TTS_MAX_SECONDS = int(os.getenv("TTS_MAX_SECONDS", "45"))


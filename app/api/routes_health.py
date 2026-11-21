import os
import shutil

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import settings

router = APIRouter(prefix="/health", tags=["Health"])

# Audit F-063: /health used to be a static "ok". It stays a liveness probe
# ("status" is always "ok" while the process serves requests) and now reports
# component readiness from cheap checks only – no model is ever built here.
TTS_MODEL_DIR = settings.PROJECT_ROOT / "pretrained_models" / "Qwen3-TTS-12Hz-1.7B-Base"


def _storage_writable() -> bool:
    return settings.STORAGE_DIR.is_dir() and os.access(settings.STORAGE_DIR, os.W_OK)


def _models_loaded() -> dict:
    """Which lazy model singletons are already in memory (never builds one)."""
    import core.config as cfg
    from app.services import embeddings_utils
    return {"asr": cfg._asr_model is not None,
            "noise": cfg._noise_Model is not None,
            "speaker_encoder": embeddings_utils._ENCODER is not None}


def _report() -> dict:
    components = {
        "storage_writable": _storage_writable(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "default_voice": (settings.VOICES_DIR / "_default" / "reference.wav").is_file(),
        "tts_model": TTS_MODEL_DIR.is_dir(),
        "models_loaded": _models_loaded(),
    }
    features = {"tts": components["ffmpeg"] and components["default_voice"] and components["tts_model"]}
    return {"status": "ok", "ready": components["storage_writable"], "components": components, "features": features}


@router.get("")
def health():
    return _report()


@router.get("/ready")
def ready():
    report = _report()
    return JSONResponse(report, status_code=200 if report["ready"] else 503)

# app/services/transcription_cache.py
from __future__ import annotations
from pathlib import Path
import json, hashlib, time
from typing import Optional
from core.ASR.transcriber import AsrTranscriber
from app.settings import STORAGE_DIR

# JSON cache will live here:
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = STORAGE_DIR / "transcriptions_TTS_cache.json"

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"by_sha1": {}, "by_path": {}, "updated_at": int(time.time())}

def _save_cache(d: dict) -> None:
    d["updated_at"] = int(time.time())
    CACHE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _sha1_of_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def transcribe_with_cache(audio_path: Path, *, language: Optional[str]=None) -> str:
    """Transcribe audio using standard OpenAI Whisper with a persistent JSON cache.
    Cache keys: file SHA1 (robust if path moves) and absolute path.
    """
    p = Path(audio_path).resolve()
    cache = _load_cache()
    sha1 = _sha1_of_file(p)
    # Try hash, then path
    text = cache.get("by_sha1", {}).get(sha1) or cache.get("by_path", {}).get(str(p))
    if text:
        return text

    #todo: RUAccent to do ударения (phonema)
    transcriber = AsrTranscriber(language=language)
    text = transcriber.transcribe(p)

    # Normalize simple spaces
    text = (text or "").strip().replace("  ", " ")

    cache.setdefault("by_sha1", {})[sha1] = text
    cache.setdefault("by_path", {})[str(p)] = text
    _save_cache(cache)
    return text

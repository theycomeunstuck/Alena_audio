# app/api/routes_TTS.py
from __future__ import annotations

from f5_tts.infer.infer_cli import ref_audio
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import io, pathlib, tempfile

from app import settings
from app.models.audio_models import TtsIn, CloneOut
from core.TTS import VoiceStore, TtsEngine

router = APIRouter(tags=["TTS"])

# Инициализация зависимостей
store = VoiceStore(settings.VOICES_DIR)
engine = TtsEngine()
# убедимся, что есть базовый голос
try:
    store.ensure_reference_wav("_default")
except FileNotFoundError as e:  # Даём понятную ошибку при первом старте без "storage/voices/_default/reference.wav"
    raise FileNotFoundError(f"{e}")


@router.post("/tts/clone", summary="Добавить голос в БД для клонирования из аудиофайла", response_model=dict)
async def clone_voice(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Не передан файл")
    suffix = pathlib.Path(file.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        raise HTTPException(status_code=400, detail="Поддерживаются WAV/MP3/FLAC/M4A/OGG")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = pathlib.Path(tmp.name)

    meta = store.clone_from_upload(tmp_path, sample_rate=settings.TTS_SAMPLE_RATE)

    return {"voice_id": meta.voice_id}


@router.post(
    "/tts",
    responses={
        200: {"content": {"audio/wav": {}, "audio/mpeg": {}, "audio/ogg": {}}, "description": "Синтезированная речь"},
        400: {"description": "Ошибка валидации / пустой текст / неверный формат"},
        404: {"description": "Не найден voice_id"},
        503: {"description": "Превышен лимит генерации (25с)"},
    },
    summary="Синтез речи",
)
async def tts(req: TtsIn):
    # Validate text is not empty
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    vid = req.voice_id or "_default"
    if not store.exists(vid):
        raise HTTPException(status_code=404, detail=f"Голос '{vid}' не найден")

    ref_audio = store.ensure_reference_wav(vid)
    meta = store.read_meta(vid)
    audio = await engine.synth(text=req.text.strip(), ref_audio=ref_audio, out_format=req.format,
                               ref_text=meta.ref_text, vid=vid, stress=req.stress)

    mt = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}[req.format]



    return Response(
        content=audio,
        media_type=mt,
        headers={
            "Content-Disposition": f'attachment; filename="tts.{req.format}"'
        }
    )




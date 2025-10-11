# app/api/routes_TTS.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
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
    store.ensure_default("_default")
except FileNotFoundError as e: # Даём понятную ошибку при первом старте без "storage/voices/_default/reference.wav"
    raise FileNotFoundError(f"{e}")
asr = AsrTranscriber()

@router.post("/voice/clone", summary="Клонировать голос из аудиофайла", response_model=dict)
async def clone_voice(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Не передан файл")
    suffix = pathlib.Path(file.filename).suffix.lower()
    if suffix not in {".wav",".mp3",".flac",".m4a",".ogg"}:
        raise HTTPException(status_code=400, detail="Поддерживаются WAV/MP3/FLAC/M4A/OGG")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = pathlib.Path(tmp.name)

    meta = store.clone_from_upload(tmp_path, sample_rate=settings.TTS_SAMPLE_RATE)

    # Автотранскрипция референса
    if settings.ASR_ENABLE_ON_CLONE:
        ref_path = store.reference_wav(meta.voice_id)
        try:
            ref_text = asr.transcribe(ref_path)
            if ref_text:
                store.update_meta(meta.voice_id, ref_text=ref_text)
        except Exception as e:
            # не валим запрос — просто без ref_text
            pass

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
    vid = req.voice_id or "_default"
    if not store.exists(vid):
        raise HTTPException(status_code=404, detail=f"Голос '{vid}' не найден")

    ref = store.reference_wav(vid)
    audio = await engine.synth(req.text, ref, req.format)

    mt = {"wav":"audio/wav","mp3":"audio/mpeg","ogg":"audio/ogg"}[req.format]
    return StreamingResponse(io.BytesIO(audio), media_type=mt)

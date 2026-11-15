from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
from typing import Optional
import asyncio  # audit F-010: offload blocking model calls off the event loop
from app.settings import STORAGE_DIR
from app.services.audio_service import AudioService
from app.services.speaker_service import unique_temp_file, cleanup_temp_file
from app.models.audio_models import EnhanceResponse, TranscribeResponse

router = APIRouter(prefix="/audio", tags=["Audio"])
svc = AudioService(STORAGE_DIR)

@router.post("/enhance",
             response_model=EnhanceResponse
             )
async def enhance(file: UploadFile = File(...)):
    # audit F-021b/F-026: unique per-request temp file instead of the
    # predictable shared upload_ name, deleted in finally after use.
    in_path = unique_temp_file(STORAGE_DIR, file.filename, prefix="upload_")
    try:
        in_path.write_bytes(await file.read())
        # audit F-010: SepFormer separate_batch blocks -- run it off the loop.
        out_path = await asyncio.to_thread(svc.enhance_file, in_path)
        return {"output_filename": out_path.name}
    finally:
        cleanup_temp_file(in_path)

@router.post("/transcribe",
             response_model=TranscribeResponse,
             response_model_exclude_none=True,  # скрыть raw, если None
             )
async def transcribe(file: UploadFile = File(...),
                     language: str = "ru",
                     verbose: bool = Query(False, description="Вернуть подробный объект raw")):
    # audit F-021b/F-026: unique per-request temp file, deleted in finally.
    in_path = unique_temp_file(STORAGE_DIR, file.filename, prefix="upload_")
    try:
        in_path.write_bytes(await file.read())
        # audit F-010: Whisper transcribe blocks -- run it off the loop.
        result = await asyncio.to_thread(svc.transcribe_file, in_path, language=language)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        cleanup_temp_file(in_path)

    text = result.get("text", "")
    raw  = result.get("raw") if verbose else None
    return TranscribeResponse(text=text, raw=raw)

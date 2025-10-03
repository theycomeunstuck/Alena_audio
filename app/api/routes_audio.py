from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
from app.settings import STORAGE_DIR
from app.services.audio_service import AudioService
from app.models.audio_models import EnhanceResponse, TranscribeResponse

router = APIRouter(prefix="/audio", tags=["Audio"])
svc = AudioService(STORAGE_DIR)

@router.post("/enhance",
             response_model=EnhanceResponse
             )
async def enhance(file: UploadFile = File(...)):
    in_path = STORAGE_DIR / f"upload_{file.filename}"
    in_path.write_bytes(await file.read())
    out_path = svc.enhance_file(in_path)
    return {"output_filename": out_path.name}

@router.post("/transcribe",
             response_model=TranscribeResponse,
             response_model_exclude_none=True,  # скрыть raw, если None
             )
async def transcribe(file: UploadFile = File(...),
                     language: str = "ru",
                     verbose: bool = Query(False, description="Вернуть подробный объект raw")):
    in_path = STORAGE_DIR / f"upload_{file.filename}"
    in_path.write_bytes(await file.read())
    try:
        result = svc.transcribe_file(in_path, language=language)
    except ValueError as e:
        raise HTTPException(400, str(e))

    text = result.get("text", "")
    raw  = result.get("raw") if verbose else None
    return TranscribeResponse(text=text, raw=raw)

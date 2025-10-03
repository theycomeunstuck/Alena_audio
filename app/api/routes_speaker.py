from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
from app.settings import STORAGE_DIR
from app.services.speaker_service import SpeakerService
from app.models.speaker_models import VerifyResponse, TrainMicResponse

router = APIRouter(prefix="/speaker", tags=["Speaker"])
svc = SpeakerService(STORAGE_DIR)

@router.post("/verify", response_model=VerifyResponse)
async def verify(
    probe: UploadFile = File(...),
    reference: UploadFile | None = File(None),
):
    probe_path = STORAGE_DIR / f"probe_{probe.filename}"
    probe_path.write_bytes(await probe.read())
    #todo: [!] добавить поддержку npy | вероятно это не имеет смысл - поддержка npy в api.
    # ref_path = STORAGE_DIR / f"ru_sample.wav" #todo: Пока что реализовано verify_speaker только для одного пользователя. и только для wav...
    ref_path = None #todo: Пока что реализовано verify_speaker только для одного пользователя. и только для wav...
    if reference is not None: # Если получили второй аргумент по API
        ref_path = STORAGE_DIR / f"ref_{reference.filename}"
        ref_path.write_bytes(await reference.read())
    try:
        result = svc.verify_files(probe_path, ref_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result

@router.post("/train/microphone", response_model=TrainMicResponse)
def train_microphone(
    user_id: str = Query("default", description="Идентификатор пользователя"),
    duration: float = Query(None, description="Длительность записи, сек (по умолчанию из конфигурации)"),
):
    """Записывает образец голоса с локального микрофона (вроде как того, что на машине, держащем API)
    и сохраняет эмбеддинг."""
    try:
        res = svc.train_from_microphone(user_id=user_id, duration=duration or None)
        return {"status": "ok", "message": f"Сохранено wav, npy: {res['wavPath']}"}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"internal error: {e.__class__.__name__}")


from fastapi import APIRouter, UploadFile, File, HTTPException
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
def train_microphone():
    # это использование микрофона на машине, где крутится API
    res = svc.train_from_microphone()
    return res

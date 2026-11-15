#app/api/routes_speaker.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
from typing import Optional

import core.config
from app.settings import STORAGE_DIR
from app.services.speaker_service import SpeakerService, validate_resource_id, unique_temp_file, cleanup_temp_file
from app.services.multi_speaker_matcher import get_global_matcher
from app.services.audio_utils import load_and_resample
from app.models.speaker_models import VerifyResponse, TrainMicResponse, MultiVerifyResponse, MultiVerifyMatch, \
    RegistryVerifyResponse, RegistryReloadResposne
from core.config import TRAIN_USER_VOICE_S, sim_threshold as _sim_threshold

router = APIRouter(prefix="/speaker", tags=["Speaker"])
svc = SpeakerService(STORAGE_DIR)
matcher = get_global_matcher()

@router.post("/verify", response_model=VerifyResponse, summary="Сравнение образца с эталоном") #todo: rename and refuse using this url. need to be /verify_solo (around this like)
async def verify(
    probe: UploadFile = File(...),
    reference: UploadFile | None = File(None),
):
    probe_path: Optional[Path] = None
    ref_path: Optional[Path] = None
    try:
        # audit F-021b/F-026: unique per-request temp files instead of the
        # predictable shared probe_/ref_ names, deleted in finally below.
        probe_path = unique_temp_file(STORAGE_DIR, probe.filename, prefix="probe_")
        probe_path.write_bytes(await probe.read())
        if reference is not None: # Если получили второй аргумент по API
            ref_path = unique_temp_file(STORAGE_DIR, reference.filename, prefix="ref_")
            ref_path.write_bytes(await reference.read())

        result = svc.verify_files(probe_path, ref_path) # ожидается {'score': float, 'decision': bool}
        if not isinstance(result, dict) or "score" not in result or "decision" not in result:
            raise ValueError("invalid response from speaker service (app/api/routes_speaker.py)")
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e.__class__.__name__}: {e}")
    finally:
        cleanup_temp_file(probe_path)
        cleanup_temp_file(ref_path)

@router.post("/verify_registry", response_model=RegistryVerifyResponse, summary="Проверка образца по всему реестру зарегистрированных голосов")
async def verify_registry(
    probe: UploadFile = File(...),
    sim_threshold: float = Query(_sim_threshold, ge=0.0, le=1.0, description="Порог бинарного решения в [0..1]"),
    top_k: int = Query(5, ge=1, le=50, description="Диагностически вернуть Top-K совпадений. По умолчанию 5"),
):
    """
    Перебирает **все** зарегистрированные в реестре голоса (VOICES_DIR + EMBEDDINGS_DIR), предзагруженные в RAM,
    и возвращает бинарное решение и лучший матч. Для диагностики можно запросить `top_k`.
    """
    try:
        buf = await probe.read()
        # Для бинарного решения достаточно best из top-1, но для прозрачности собираем top_k
        audio = load_and_resample(buf)
        result = matcher.match_probe_array(audio, top_k=top_k)
        best = result[0] if result else None
        decision = bool(best and best["score"] >= float(sim_threshold))
        best_model = MultiVerifyMatch(**best) if decision and best is not None else None
        return {"decision": decision, "best": best_model, "threshold": float(sim_threshold), "matches": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e.__class__.__name__}: {e}")

@router.post("/verify/topk", response_model=MultiVerifyResponse, summary="Top-K совпадений по всему реестру")
async def verify_topk(
    probe: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50, description="Сколько лучших совпадений вернуть"),
):
    try:
        buf = await probe.read()
        audio = load_and_resample(buf)
        matches = matcher.match_probe_array(audio, top_k=top_k)
        return MultiVerifyResponse(count=len(matches), matches=[MultiVerifyMatch(**m) for m in matches])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e.__class__.__name__}: {e}")

@router.post("/registry/reload", response_model=RegistryReloadResposne, summary="Перечитать реестр голосов с диска и обновить RAM")
def registry_reload(
    flag: bool = Query(False, description="Вернуть список пользователей? По умолчанию False")
) -> dict:
    try:
        id_list, count = matcher.reload(flag)
        if flag:
            return {"id_list": id_list, "count": int(count)}
        return {"count": int(count)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e.__class__.__name__}: {e}")


@router.post("/train/microphone", response_model=TrainMicResponse)
def train_microphone(
    user_id: str = Query("_default", description="Идентификатор пользователя. Если не задан, то генерируется случайный (uuid4)"),
    duration: float = Query(TRAIN_USER_VOICE_S, le=30, description=f"Длительность записи, сек (по умолчанию из конфигурации: {TRAIN_USER_VOICE_S}, максимум 30)"),  # audit F-025b: верхняя граница
):
    """Записывает образец голоса с локального микрофона (вроде как того, что на машине, держащем API)
    сохраняет эмбеддинг, обновляет бд."""
    try:
        validate_resource_id(user_id, "user_id")  # audit F-021
        res = svc.train_from_microphone(user_id=user_id, duration=duration)
        matcher.reload() #обновляем бд
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"internal error: {e.__class__.__name__}\n {e}")



@router.post("/train/file", response_model=TrainMicResponse)
async def train_file(
    probe: UploadFile = File(..., description="Файл, который будет переведён в эмбеддинг"),
    user_id: str = Query("_default", description="Идентификатор пользователя. Если не задан, то генерируется случайный (uuid4)"),
):
    """Записывает образец голоса с локального микрофона (вроде как того, что на машине, держащем API)
    сохраняет эмбеддинг, обновляет бд."""
    try:
        validate_resource_id(user_id, "user_id")  # audit F-021: до декодирования контента
        buf = await probe.read()
        audio = load_and_resample(buf)
        res = svc.train_from_file(probe_wav=audio, user_id=user_id)
        matcher.reload() #обновляем бд
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"internal error: {e.__class__.__name__}")


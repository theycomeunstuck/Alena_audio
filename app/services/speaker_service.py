from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from app.services.audio_utils import load_and_resample
from core.audio_enhancement import Audio_Enhancement

_ENCODER: Optional[EncoderClassifier] = None


# Поднимем один раз энкодер для референс-проверки
def _get_encoder() -> EncoderClassifier:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        _ENCODER.eval()
    return _ENCODER

def _to_tensor_1d(x: np.ndarray) -> torch.Tensor:
    if not isinstance(x, np.ndarray):
        raise ValueError("ожидался np.ndarray")
    if x.ndim != 1:
        raise ValueError(f"ожидался 1D массив, получено shape={x.shape}")
    if np.allclose(x, 0.0, atol=1e-7):
        raise ValueError("пустой/нулевой сигнал")
    return torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)  # [1,T]

def _embed_sb(x: np.ndarray) -> torch.Tensor:
    enc = _get_encoder()
    wav = _to_tensor_1d(x)  # [1,T]
    with torch.no_grad():
        emb = enc.encode_batch(wav)  # [1,1,D] или [1,D]
        emb = emb.squeeze()  # [D]
        emb = F.normalize(emb, p=2, dim=-1)
    return emb

class SpeakerService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)



    def verify_files(self, probe_wav: Path, reference_wav: Optional[Path] = None) -> Dict[str, Any]:
        # 1) грузим оба файла в 1D float32 @16k
        probe = load_and_resample(str(probe_wav))
        ref = load_and_resample(str(reference_wav)) if reference_wav else None

        # safety-лог (в консоль uvicorn) #todo: убрать принт app/api/speaker_service.py
        print(f"[verify_files] probe len={len(probe)} ref len={len(ref) if ref is not None else 'None'} "
              f"rms_probe={float(np.sqrt(np.mean(probe**2))):.6f} "
              f"rms_ref={(float(np.sqrt(np.mean(ref**2))) if ref is not None else 'None')}")

        # 2) старый путь (твой Audio_Enhancement) — СЧИТАЕМ, но не верим слепо
        try:
            enhancer = Audio_Enhancement(probe, ref)
            sim = enhancer.speech_verification()
        except Exception as e:
            sim = f"speech_verification_exception:{e.__class__.__name__}:{e}"

        # распакуем старый sim в (score, decision)
        ae_score: Optional[float] = None
        ae_decision: Optional[bool] = None
        if isinstance(sim, (list, tuple)) and len(sim) == 2:
            ae_score, ae_decision = float(sim[0]), bool(sim[1])
        elif isinstance(sim, (float, int)):
            ae_score = float(sim)
            ae_decision = bool(ae_score >= 0.5)
        else:
            # что угодно (str/None/torch.Tensor/np.ndarray) — покажем как raw
            pass

        # 3) Эталонный путь: честно считаем эмбеддинги speechbrain и косинус
        try:
            if ref is None:
                raise ValueError("audio_ref не задан для верификации")

            emb_p = _embed_sb(probe)
            emb_r = _embed_sb(ref)
            sb_score = float(F.cosine_similarity(emb_p.unsqueeze(0), emb_r.unsqueeze(0)).item())
            sb_decision = bool(sb_score >= 0.65)  # пример порога
        except Exception as e:
            # Если тут ошибка — критично: вернём 400/500 в роуте
            raise

        # 4) Вернём И то, и другое в debug-режиме
        debug = os.getenv("DEBUG_SPEAKER", "0") == "1"
        result = {
            "score": sb_score,
            "decision": sb_decision,
        }
        if debug:
            # все подробности для отладки
            result["raw"] = {
                "ae_sim_type": type(sim).__name__,
                "ae_score": ae_score,
                "ae_decision": ae_decision,
                "sb_score": sb_score,
                "probe_len": int(len(probe)),
                "ref_len": int(len(ref) if ref is not None else 0),
                "rms_probe": float(np.sqrt(np.mean(probe**2))),
                "rms_ref": float(np.sqrt(np.mean(ref**2))) if ref is not None else None,
            }
        return result

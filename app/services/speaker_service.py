from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf

from app.services.audio_utils import load_and_resample
from core.audio_enhancement import Audio_Enhancement
from core.config import SAMPLE_RATE
from core.train_speaker import train_user_voice


class SpeakerService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # === Верификация: probe vs reference (оба файла) ===
    # Используем логику Audio_Enhancement.speech_verification()
    def verify_files(self,
                     probe_wav: Path,
                     reference_wav: Optional[Path] = None) -> Dict[str, Any]:
        probe = load_and_resample(str(probe_wav))
        ref = load_and_resample(str(reference_wav)) if reference_wav else None

        enhancer = Audio_Enhancement(probe, ref)


        sim = enhancer.speech_verification()   # На случай, если возвращается одно число — интерпретируем порог на фронте.


        if isinstance(sim, (list, tuple)) and len(sim) == 2:
            score, decision = float(sim[0]), bool(sim[1])
        else:
            score, decision = float(sim), bool(sim >= 0.5)  # базовый порог, если не вернули bool

        return {"score": score, "decision": decision}

    # === Тренировка эталона как (через микрофон) ===
    def train_from_microphone(self) -> Dict[str, Any]:
        train_user_voice()
        return {"status": "ok", "message": "Эталон записан и сохранён (reference.npy/reference.wav)"}

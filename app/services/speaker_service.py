#app/services/speaker_service.py
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
import os
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from app.services.audio_utils import load_and_resample
from core.audio_capture import record_audio
from core.audio_enhancement import Audio_Enhancement
from core.config import TRAIN_USER_VOICE_S, EMBEDDINGS_DIR, SAMPLE_RATE

_ENCODER: Optional[EncoderClassifier] = None

# todo: убрать весь пайплайн файла в core

# Поднимем один раз энкодер для референс-проверки
def _get_encoder() -> EncoderClassifier:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(
                (Path(__file__).resolve().parents[2] / "pretrained_models" / "SpeechBrain" / "spkrec-ecapa-voxceleb")),
        ).eval()
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

    def _project_root(self) -> Path:
        # .../app/services/speaker_service.py -> project root (на 2 уровня вверх от app/)
        return Path(__file__).resolve().parents[2]

    def _default_ref_candidates(self) -> list[Path]: #todo: заменить всю функцию на перебор пользователей
        root = self._project_root()
        return [
            self.storage_dir / "misha_20sec.wav",
            self.storage_dir / "ref_misha_20sec.wav",
            root / "core" / "misha_20sec.wav",
            root / "tests" / "samples" / "misha_20sec.wav",
        ]

    def _find_default_ref(self) -> Optional[Path]:
        for p in self._default_ref_candidates():
            if p.is_file():
                return p
        return None

    def verify_files(self, probe_wav: Path, reference_wav: Optional[Path] = None) -> Dict[str, Any]:
        # 1) грузим оба файла в 1D float32 @ 16k
        probe = load_and_resample(str(probe_wav))
        ref = load_and_resample(str(reference_wav)) if reference_wav else None

        try:
            # если reference не передан — ищем надёжный дефолт
            if ref is None:
                default_ref_path = self._find_default_ref()
                if default_ref_path is None:
                    searched = self._default_ref_candidates()
                    hint = "\n".join(f"- {p}" for p in searched)
                    raise ValueError(
                        "reference не задан и не найден дефолтный образец. "
                        "Положите файл по одному из путей:\n" + hint
                    )
                ref = load_and_resample(str(default_ref_path))

            emb_p = _embed_sb(probe)
            emb_r = _embed_sb(ref)
            sb_score = float(F.cosine_similarity(emb_p.unsqueeze(0), emb_r.unsqueeze(0)).item())
            sb_decision = bool(sb_score >= 0.65)  # пример порога
        except Exception as e:
            raise Exception(f"app/service/speaker_service:: verify_files()\n{e}")

        result = {
            "score": sb_score,
            "decision": sb_decision,
        }
        return result

    def train_from_microphone(self, user_id: str = "default", duration: float = TRAIN_USER_VOICE_S) -> Dict[str, Any]:
        """
        Записывает голос с локального микрофона API-хоста, извлекает эмбеддинг и
        сохраняет в EMBEDDINGS_DIR/<user_id>.npy
        """
        audio = record_audio(duration=duration)  # 1D float32 @ SAMPLE_RATE
        if audio.ndim != 1 or audio.size < int(0.5 * SAMPLE_RATE):
            raise ValueError("Слишком короткая запись — повторите попытку")


        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        out_wav_path = EMBEDDINGS_DIR / f"{user_id}.wav"
        try:
            # wav = torch.from_numpy(audio).reshape(1, -1)  # [C=1, N]
            torchaudio.save(str(out_wav_path), src=torch.from_numpy(audio).unsqueeze(0),  # [1,T]
                            sample_rate=SAMPLE_RATE, format="wav",
                            encoding="PCM_S", bits_per_sample=16)  # стандартный 16-бит PCM

        except Exception as e:
            print(e)
            raise e

        emb = _embed_sb(audio).detach().cpu().numpy().astype(np.float32)

        out_npy_path = EMBEDDINGS_DIR / f"{user_id}.npy"

        np.save(out_npy_path, emb)
        return {"status": "ok", "wavPath": str(out_wav_path), "npyPath": str(out_npy_path)}


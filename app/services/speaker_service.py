#app/services/speaker_service.py
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
import os, re, tempfile, uuid
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from app.services.audio_utils import load_and_resample
from app.services.embeddings_utils import embed_speechbrain, get_encoder
from core.audio_capture import record_audio
from core.audio_enhancement import Audio_Enhancement
from core.audio_utils import normalize_rms
from core.config import TRAIN_USER_VOICE_S, EMBEDDINGS_DIR, SAMPLE_RATE, sim_threshold, EMBEDDINGS_WAV_DIR

_ENCODER: Optional[EncoderClassifier] = None


# --- Audit F-021 / F-021b / F-026 helpers: strict resource ids and per-request temp files ---

# Anything interpolated into a filesystem path (EMBEDDINGS_DIR/<id>.npy,
# EMBEDDINGS_WAV_DIR/<id>.wav, VOICES_DIR/<id>/...) must match this pattern;
# anything else enabled traversal writes and registry poisoning (audit F-021).
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_resource_id(value: str, kind: str = "id") -> str:
    """Reject ids that could traverse directories or poison the registry.

    Allowed: 1..64 characters of [A-Za-z0-9_-] ("_default" included). Anything
    else (separators, dots, spaces, empty, too long) raises ValueError, which
    the routes map to HTTP 400.
    """
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {kind}: must be 1-64 characters of [A-Za-z0-9_-]")
    return value


def unique_temp_file(directory: Path, original_name: Optional[str], prefix: str = "upload_") -> Path:
    """Create a unique per-request temp file inside `directory` (audit F-021b).

    The name comes from tempfile (unpredictable, collision-free); only a safe
    suffix is carried over from the original upload name, so it can never
    contain separators or traversal segments.
    """
    suffix = Path(original_name or "").suffix.lower()
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix):
        suffix = ""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(directory))
    os.close(fd)
    return Path(name)


def cleanup_temp_file(path: Optional[Path]) -> None:
    """Best-effort removal of a per-request temp file (audit F-026)."""
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


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
            self.storage_dir / "_default" / "reference.wav",
            root / "tests" / "samples" / "ru_sample.wav",
            root / "tests" / "samples" / "reference.wav",
            root / "storage" / "_default" / "reference.wav",
            self.storage_dir / "reference.wav",
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
                    hint = ";  ".join(f"- {p}" for p in searched)
                    raise ValueError(
                        "reference не задан и не найден дефолтный образец. "
                        "Положите файл по одному из путей: " + hint
                    )
                ref = load_and_resample(str(default_ref_path))

            emb_p = embed_speechbrain(probe)
            emb_r = embed_speechbrain(ref)


            sb_score = float(F.cosine_similarity(emb_p.unsqueeze(0), emb_r.unsqueeze(0)).item())
            sb_decision = bool(sb_score >= sim_threshold)  # пример порога
        except Exception as e:
            raise Exception(f"app/service/speaker_service:: verify_files()\n{e}")

        result = {
            "score": sb_score,
            "decision": sb_decision,
        }
        return result


    def train_from_microphone(self, user_id: str = "_default", duration: float = TRAIN_USER_VOICE_S) -> Dict[str, Any]:
        """
        Записывает голос с локального микрофона API-хоста, извлекает эмбеддинг и
        сохраняет в EMBEDDINGS_DIR/<user_id>.npy
        """
        validate_resource_id(user_id, "user_id")  # audit F-021
        if user_id == "_default":
            user_id = str(uuid.uuid4().hex)

        audio = record_audio(duration=duration)  # 1D float32 @ SAMPLE_RATE
        if audio.ndim != 1 or audio.size < int(3 * SAMPLE_RATE):
            raise ValueError("Слишком короткая запись — повторите попытку")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_WAV_DIR.mkdir(parents=True, exist_ok=True)
        out_npy_path = EMBEDDINGS_DIR / f"{user_id}.npy"
        out_wav_path = EMBEDDINGS_WAV_DIR / f"{user_id}.wav"

        audio = Audio_Enhancement(audio).noise_suppression() # Tensor. Noise suppresion + rms_normalize; cpu

        try:
            torchaudio.save(str(out_wav_path), src=audio.unsqueeze(0),  # [1,T]
                            sample_rate=SAMPLE_RATE, format="wav",
                            encoding="PCM_S", bits_per_sample=16)  # стандартный 16-бит PCM
        except Exception as e:
            print(e); raise

        emb = embed_speechbrain(audio) # to(device)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12).float()

        emb = emb.detach().cpu()  # если уже float32 — отлично
        if emb.dtype != torch.float32:
            emb = emb.to(torch.float32)  # привести один раз в torch
        np.save(out_npy_path, emb.numpy())


        return {"user_id": user_id, "wav_path": str(out_wav_path), "npy_path": str(out_npy_path)}

    def train_from_file(self, probe_wav: np.ndarray, user_id: str = "_default") -> Dict[str, Any]:

        validate_resource_id(user_id, "user_id")  # audit F-021
        if user_id == "_default":
            user_id = str(uuid.uuid4().hex)

        audio = load_and_resample(probe_wav)

        if audio.ndim != 1 or audio.size < int(3 * SAMPLE_RATE):
            raise ValueError("Слишком короткая запись — повторите попытку")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_WAV_DIR.mkdir(parents=True, exist_ok=True)
        out_npy_path = EMBEDDINGS_DIR / f"{user_id}.npy"
        out_wav_path = EMBEDDINGS_WAV_DIR / f"{user_id}.wav"

        audio = Audio_Enhancement(audio).noise_suppression() #noise suppresion + rms_normalize; cpu

        try:
            torchaudio.save(str(out_wav_path), src=audio.unsqueeze(0),  # [1,T]
                            sample_rate=SAMPLE_RATE, format="wav",
                            encoding="PCM_S", bits_per_sample=16)  # стандартный 16-бит PCM
        except Exception as e:
            raise e

        emb = embed_speechbrain(audio) # to(device)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12).float() # L2-norm

        emb = emb.detach()
        np.save(
            out_npy_path,
            emb.cpu().numpy().astype(np.float32)
        )



        return {"user_id": user_id, "wav_path": str(out_wav_path), "npy_path": str(out_npy_path)}
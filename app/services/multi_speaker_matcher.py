# app/services/multi_speaker_matcher.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
import numpy as np
import torch

from app.services.audio_utils import load_and_resample
from app.services.embeddings_utils import embed_speechbrain
from core.config import VOICES_DIR, EMBEDDINGS_DIR, SAMPLE_RATE, sim_threshold as DEFAULT_SIM_THRESHOLD


def _cos_to01(x: torch.Tensor) -> torch.Tensor:
    """
    Преобразует косинусное сходство из диапазона [-1..1] в [0..1].
    """
    return (torch.clamp(x, -1.0, 1.0) + 1.0) * 0.5


class MultiSpeakerMatcher:
    """
    Держит в RAM набор референс-аудио пользователей + их эмбеддинги.
    Ищет best-match и top-k по входному аудиофрагменту.
    Источники данных:
      - VOICES_DIR/<user_id>/reference.wav  (исторический формат; совместимость с тестами)
      - EMBEDDINGS_DIR/<user_id>.wav        (новый формат реестра)
      - EMBEDDINGS_DIR/<user_id>.npy        (эмбеддинг; опционально, ускоряет загрузку)
    """
    def __init__(self, voices_dir: Path | str = VOICES_DIR, embeddings_dir: Path | str = EMBEDDINGS_DIR, sample_rate: int = SAMPLE_RATE):
        self.voices_dir = Path(voices_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.sample_rate = int(sample_rate)

        self._user_ids: List[str] = []
        self._paths: List[Path] = []
        self._wav_arrays: List[np.ndarray] = []  # исходные mono float32 [-1..1]
        self._embs: torch.Tensor = torch.zeros((0, 256), dtype=torch.float32)  # (N, D)
        self._lock = threading.RLock()


    def _collect_voice_candidates(self) -> List[Tuple[str, Path]]:
        """
        Возвращает список (user_id, path_to_reference_wav) из VOICES_DIR и EMBEDDINGS_DIR.
        """
        pairs: List[Tuple[str, Path]] = []

        # 1) VOICES_DIR/<uid>/reference.wav
        if self.voices_dir.exists():
            for sub in sorted(self.voices_dir.iterdir()):
                if not sub.is_dir():
                    continue
                uid = sub.name
                ref = sub / "reference.wav"
                if ref.exists():
                    pairs.append((uid, ref))

        # 2) EMBEDDINGS_DIR/<uid>.wav
        if self.embeddings_dir.exists():
            for wav in sorted(self.embeddings_dir.glob("*.wav")):
                uid = wav.stem
                pairs.append((uid, wav))

        return pairs

    def _try_load_embedding_npy(self, uid: str) -> Optional[torch.Tensor]:
        """
        Если есть EMBEDDINGS_DIR/<uid>.npy — грузим оттуда нормализованный эмбеддинг.
        """
        npy = self.embeddings_dir / f"{uid}.npy"
        if npy.exists():
            try:
                arr = np.load(npy)
                t = torch.from_numpy(arr).float()
                if t.ndim != 1:
                    t = t.view(-1)
                # предполагаем, что он уже L2-нормализован; но на всякий случай нормализуем ещё раз
                t = torch.nn.functional.normalize(t, p=2, dim=-1)
                return t
            except Exception:
                return None
        return None


    def reload(self) -> int:
        """
        Перечитать все пользователи из диска и поместить их аудио + эмбеддинги в RAM.
        Возвращает количество пользователей.
        """
        with self._lock:
            user_ids: List[str] = []
            paths: List[Path] = []
            wavs: List[np.ndarray] = []
            embs: List[torch.Tensor] = []

            candidates = self._collect_voice_candidates()
            for uid, ref in candidates:
                try:
                    # Загружаем аудио (моно, float32 @ SAMPLE_RATE)
                    wav = load_and_resample(ref)


                    if wav.size < int(0.3 * self.sample_rate): # если отрезок больше чем 0.3 сек
                        # короткие/битые пропускаем
                        print(f"[MultiSpeakerMatcher] skip {uid}: too short")
                        continue

                    # Попробуем взять готовый эмбеддинг из .npy, иначе посчитаем
                    emb = self._try_load_embedding_npy(uid)
                    if emb is None:
                        emb = embed_speechbrain(wav)  # torch.Tensor [D]

                except Exception as e:
                    # защищаем пайплайн от битых файлов
                    print(f"[MultiSpeakerMatcher] skip {uid}: {e}")
                    continue

                user_ids.append(uid)
                paths.append(ref)
                wavs.append(wav.astype(np.float32, copy=False))  # держим в RAM
                embs.append(emb.view(1, -1))  # [1,D]

            # Консолидируем
            if embs:
                self._embs = torch.cat(embs, dim=0).contiguous()  # [N,D]
            else:
                self._embs = torch.zeros((0, 256), dtype=torch.float32)
            self._user_ids = user_ids
            self._paths = paths
            self._wav_arrays = wavs
            return len(self._user_ids)

    def registry_size(self) -> int:
        with self._lock:
            return len(self._user_ids)

    def match_probe_array(self, audio: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Возвращает top-k совпадений по сходству в [0..1].
        """
        with self._lock:
            if self._embs.shape[0] == 0:
                return []
            # эмбеддинг пробы
            probe_emb = embed_speechbrain(np.asarray(audio, dtype=np.float32))
            # косинусное сходство со всеми референсами
            sims = torch.matmul(self._embs, probe_emb.view(-1, 1)).squeeze(1)  # [N]
            scores = _cos_to01(sims)  # [N]
            k = min(int(top_k), scores.numel())
            topk_val, topk_idx = torch.topk(scores, k)
            matches: List[Dict[str, Any]] = []
            for v, idx in zip(topk_val.tolist(), topk_idx.tolist()):
                matches.append({
                    "user_id": self._user_ids[idx],
                    "score": float(v),
                    "ref_path": str(self._paths[idx]),
                })
            return matches

    def match_probe_file(self, path: Path, top_k: int = 5) -> List[Dict[str, Any]]:
        audio = load_and_resample(path, self.sample_rate)
        return self.match_probe_array(audio, top_k=top_k)

    def binary_decision(self, audio: np.ndarray, threshold: float = DEFAULT_SIM_THRESHOLD) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Бинарное решение и лучший матч.
        """
        matches = self.match_probe_array(audio, top_k=1)
        if not matches:
            return False, None
        best = matches[0]
        return (best["score"] >= float(threshold)), best



_GLOBAL_MATCHER: Optional[MultiSpeakerMatcher] = None
_SINGLETON_LOCK = threading.RLock()

def get_global_matcher() -> MultiSpeakerMatcher:
    """
    Возвращает глобальный матчинг-объект.
    Если он ещё не создан — создаём и сразу делаем .reload(), чтобы в RAM
    появились все эмбеддинги пользователей.
    """
    global _GLOBAL_MATCHER
    with _SINGLETON_LOCK:
        if _GLOBAL_MATCHER is None:
            print("⚙️ Initializing global MultiSpeakerMatcher()")
            _GLOBAL_MATCHER = MultiSpeakerMatcher(VOICES_DIR, EMBEDDINGS_DIR)
            _GLOBAL_MATCHER.reload()
        return _GLOBAL_MATCHER

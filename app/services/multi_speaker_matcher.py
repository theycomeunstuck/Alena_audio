# app/services/multi_speaker_matcher.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
import numpy as np
import torch

from app.services.audio_utils import load_and_resample
from app.services.embeddings_utils import embed_speechbrain
from core.config import EMBEDDINGS_DIR, SAMPLE_RATE, sim_threshold as DEFAULT_SIM_THRESHOLD, TARGET_DBFS
from core.audio_utils import normalize_rms


def _cos_to01(x: torch.Tensor) -> torch.Tensor:
    """Косинус [-1..1] → [0..1]."""
    return (torch.clamp(x, -1.0, 1.0) + 1.0) * 0.5

# [30.10.25] todo: нужно бы обезопасить device, чтобы было на одном устройстве.
# then i need to check websocket multiply user same time. gpt says its cant
class MultiSpeakerMatcher:
    """
    Реестр спикеров хранится только в EMBEDDINGS_DIR как <user_id>.npy.
    Каждый .npy — L2-нормированный эмбеддинг (1D float, любая длина D).
    Объект держит в RAM:
      - self._user_ids: порядок пользователей
      - self._emb_paths: пути к соответствующим .npy
      - self._embs: матрицу эмбеддингов [N, D]
    """

    def __init__(self, embeddings_dir: Path | str = EMBEDDINGS_DIR, sample_rate: int = SAMPLE_RATE):
        self.embeddings_dir = Path(embeddings_dir)
        self.sample_rate = int(sample_rate)

        self._user_ids: List[str] = []
        self._emb_paths: List[Path] = []
        self._embs: torch.Tensor = torch.empty((0, 0), dtype=torch.float32)  # [N, D], D задастся при reload()
        self._lock = threading.RLock()


    def _list_embedding_files(self) -> List[Tuple[str, Path]]:
        """Сканирует embeddings_dir и возвращает [(uid, path_to_npy)]."""
        pairs: List[Tuple[str, Path]] = []
        if self.embeddings_dir.exists():
            for npy in sorted(self.embeddings_dir.glob("*.npy")):
                uid = npy.stem
                pairs.append((uid, npy))
        return pairs

    @staticmethod
    def _load_embedding_npy(npy_path: Path) -> torch.Tensor:
        """
        Грузит <uid>.npy, приводит к 1D float32 и L2-нормализует.
        Бросает исключение, если файл битый.
        """
        arr = np.load(npy_path)
        t = torch.from_numpy(arr).float()
        if t.ndim != 1:
            t = t.view(-1)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)
        return t


    def reload(self) -> int:
        """
        Перечитать все *.npy из embeddings_dir и собрать RAM-индекс.
        Возвращает количество пользователей.
        """
        candidates = self._list_embedding_files()

        new_user_ids: List[str] = []
        new_paths: List[Path] = []
        new_embs: List[torch.Tensor] = []

        for uid, npy_path in candidates:
            try:
                emb = self._load_embedding_npy(npy_path)  # [D]
            except Exception as e:
                print(f"[MultiSpeakerMatcher] skip {uid}: can't load {npy_path.name} ({e})")
                continue

            new_user_ids.append(uid)
            new_paths.append(npy_path)
            new_embs.append(emb.view(1, -1).contiguous())  # [1, D]

        new_embs_tensor = torch.cat(new_embs, dim=0) if new_embs else torch.empty((0, 0), dtype=torch.float32)

        # быстрый swap под коротким локом
        with self._lock:
            self._user_ids = new_user_ids
            self._emb_paths = new_paths
            self._embs = new_embs_tensor

        return len(self._user_ids)

    def registry_size(self) -> int:
        with self._lock:
            return len(self._user_ids)

    def match_probe_array(self, audio: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Считает эмбеддинг пробы и возвращает top-k совпадений по сходству в [0..1],
        вместе с путями к .npy (ref_path).
        """
        with self._lock:
            if self._embs.shape[0] == 0:
                return []
            embs = self._embs
            user_ids = list(self._user_ids)
            emb_paths = list(self._emb_paths)

        # вне лока — тяжёлое вычисление эмбеддинга
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=1)  # привести к моно на всякий
        a = normalize_rms(a)

        with torch.no_grad():
            probe_emb = embed_speechbrain(a)                       # [D]
            probe_emb = torch.nn.functional.normalize(probe_emb, p=2, dim=-1, eps=1e-12).float()

            sims = torch.matmul(embs, probe_emb.view(-1, 1)).squeeze(1)  # [N]
            scores = _cos_to01(sims)
            k = max(1, min(int(top_k), scores.numel()))
            topk_val, topk_idx = torch.topk(scores, k)

        return [
            {"user_id": user_ids[i], "score": float(v), "ref_path": str(emb_paths[i])} # [DEBUG] todo: ref_path передаётся для отладки. позже он не нужен
            for v, i in zip(topk_val.tolist(), topk_idx.tolist())
        ]

    def match_probe_file(self, path: Path | str, top_k: int = 5) -> List[Dict[str, Any]]:
        audio = load_and_resample(Path(path))
        return self.match_probe_array(audio, top_k=top_k)

    def binary_decision(self, audio: np.ndarray, threshold: float = DEFAULT_SIM_THRESHOLD) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Бинарное решение и лучший матч (по top-1).
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
    При первом вызове создаёт его и делает .reload() (реестр из *.npy).
    """
    global _GLOBAL_MATCHER
    with _SINGLETON_LOCK:
        if _GLOBAL_MATCHER is None:
            print("⚙️ Initializing global MultiSpeakerMatcher() [npy-only]")
            _GLOBAL_MATCHER = MultiSpeakerMatcher(EMBEDDINGS_DIR)
            _GLOBAL_MATCHER.reload()
        return _GLOBAL_MATCHER

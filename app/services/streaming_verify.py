# app/services/streaming_verify.py
from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
import numpy as np
from app.services.audio_utils import (
    SAMPLE_RATE,
    ensure_float_mono_16k_from_pcm16,
)
from app.services.multi_speaker_matcher import get_global_matcher

class StreamingVerifySession:
    """
    Держит скользящий буфер последнего аудио (до ~8 сек @16кГц) и
    позволяет вычислять best-match/top-k по всем зарегистрированным голосам.
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels_hint: int = 1, inactivity_sec: float | None = 120.0):
        self.SAMPLE_RATE = SAMPLE_RATE
        self._src_sr = int(sample_rate)
        self._channels = int(channels_hint)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.inactivity_sec = inactivity_sec
        self._last_audio_t = time.time()

    def ingest_pcm16_chunk(self, pcm_chunk: bytes) -> None:
        x16k = ensure_float_mono_16k_from_pcm16(pcm_chunk, src_sr=self._src_sr, channels=self._channels)
        # x16k = ensure_float_mono_16k_from_pcm16(pcm_chunk, src_sr=self._src_sr, channels=self._channels)
        if x16k.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, x16k]).astype(np.float32, copy=False)
        # keep last 8s
        max_len = int(8 * self.SAMPLE_RATE)
        if self.buffer.size > max_len:
            self.buffer = self.buffer[-max_len:]
        self._last_audio_t = time.time()

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)

    def inactive_timed_out(self) -> bool:
        if self.inactivity_sec is None:
            return False
        return (time.time() - self._last_audio_t) >= float(self.inactivity_sec)

    def _has_min_audio(self) -> bool:
        # Минимально 300 мс речи, иначе эмбеддинг нерепрезентативен
        return self.buffer.size >= int(0.3 * self.SAMPLE_RATE)

    def current_best(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Топ-K совпадений (оценка [0..1]) по всей накопленной речи."""
        if not self._has_min_audio():
            return []
        matcher = get_global_matcher()
        return matcher.match_probe_array(self.buffer, top_k=top_k)

    def current_best_binary(self, threshold: float, top_k: int = 5) -> Dict[str, Any]:
        """
        Бинарная верификация: совпал ли кто-то из реестра при пороге threshold ([0..1]).
        Возвращает {'decision': bool, 'threshold': float, 'best': {user_id, score, ref_path}|None, 'matches': [...]}
        'matches' включён для диагностики/логирования; на клиент можно не показывать.
        """
        if not self._has_min_audio():
            return {"decision": False, "threshold": float(threshold), "best": None, "matches": []}
        matcher = get_global_matcher()
        matches = matcher.match_probe_array(self.buffer, top_k=top_k)
        best = matches[0] if matches else None
        decision = bool(best and best["score"] >= float(threshold))
        return {"decision": decision, "threshold": float(threshold), "best": best if decision else None, "matches": matches}

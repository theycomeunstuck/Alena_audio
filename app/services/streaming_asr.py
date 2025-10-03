# app/services/streaming_asr.py
from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple
from app.services.audio_utils import (
    SAMPLE_RATE, #target_rate
    pcm16_bytes_to_float1d,
    ensure_float_mono_16k_from_pcm16,
)
from core.config import ASR_WINDOW_SEC, ASR_EMIT_SEC, asr_model

class StreamingASRSession:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        language: str = "ru",
        window_sec: float = ASR_WINDOW_SEC,
        emit_sec: float = ASR_EMIT_SEC,
        assume_pcm16_mono_16k: bool = True,
        channels_hint: int = 1,
        inactivity_sec: float | None = None,  # авто-final если нет аудио дольше, например, 2.5с
    ):
        self.language = language
        self.window_sec = float(window_sec)
        self.emit_sec = float(emit_sec)
        self.SAMPLE_RATE = SAMPLE_RATE
        self.assume_pcm16_mono_16k = assume_pcm16_mono_16k
        self.channels_hint = channels_hint
        self.buffer = np.zeros(0, dtype=np.float32)
        self._last_emit_t = 0.0
        self._src_sr = int(sample_rate)
        self._last_audio_t = time.time()
        self._inactivity_sec = inactivity_sec

    # ------ жизненный цикл фразы ------
    def reset(self) -> None:
        """Очистить буфер и таймеры — начать новую фразу в том же сокете."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self._last_emit_t = 0.0
        self._last_audio_t = time.time()

    # ------ приём аудио ------
    def _append_audio(self, x16k: np.ndarray) -> None:
        if x16k.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, x16k]).astype(np.float32, copy=False)
        max_len = int(self.window_sec * self.SAMPLE_RATE)
        if self.buffer.size > max_len:
            self.buffer = self.buffer[-max_len:]
        self._last_audio_t = time.time()

    def ingest_pcm16_chunk(self, pcm_chunk: bytes) -> None:
        if self.assume_pcm16_mono_16k and self._src_sr == self.SAMPLE_RATE and self.channels_hint == 1:
            x = pcm16_bytes_to_float1d(pcm_chunk)
        else:
            x = ensure_float_mono_16k_from_pcm16(pcm_chunk, src_sr=self._src_sr, channels=self.channels_hint)
        self._append_audio(x)

    # ------ инференс ------

    def _transcribe_partial(self):
        # Пустой буфер — возвращаем пустой результат без вызова модели
        if self.buffer.size == 0:
            return "", {"segments": [], "language": self.language, "text": ""}

        # self.buffer — np.ndarray 1D float32; не вызывать как функцию!
        audio = self.buffer.astype(np.float32, copy=False)
        res = asr_model.transcribe(audio, language=self.language)
        txt = (res.get("text") or "").strip()
        return txt, res

    def _transcribe_final(self):
        # Финальный транскрипт должен быть корректным даже для пустого буфера
        if self.buffer.size == 0:
            txt, res = "", {"segments": [], "language": self.language, "text": ""}
            return txt, res

        audio = self.buffer.astype(np.float32, copy=False)
        res = asr_model.transcribe(audio, language=self.language)
        txt = (res.get("text") or "").strip()
        return txt, res

    def maybe_emit(self) -> Optional[Tuple[str, dict]]:
        now = time.time()
        if now - self._last_emit_t < self.emit_sec:
            return None
        self._last_emit_t = now
        return self._transcribe_partial()

    def finalize(self) -> Tuple[str, dict]:
        return self._transcribe_partial()

    def inactive_timed_out(self) -> bool:
        """True, если включён авто-final и мы молчим дольше порога."""
        if self._inactivity_sec is None:
            return False
        return (time.time() - self._last_audio_t) >= self._inactivity_sec

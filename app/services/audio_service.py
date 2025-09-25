from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch, torchaudio

from app.services.audio_utils import load_and_resample, ensure_pcm16_mono_16k

from core.audio_enhancement import Audio_Enhancement
from core.config import SAMPLE_RATE, ASR_LANGUAGE, ASR_WINDOW_SEC, ASR_EMIT_SEC, asr_model


class AudioService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # === Препроцессинг аудио (шумоподавление + нормализация) ===
    def enhance_file(self, input_wav: Path) -> Path:
        # 1) приводим к 1D float32 mono @ 16k
        audio = load_and_resample(str(input_wav))  # -> 1D np.float32

        # 2) шумоподавление (старый модуль)
        enh = Audio_Enhancement(audio)
        enhanced = np.asarray(enh.noise_suppression(), dtype=np.float32).squeeze()

        # 3) сохраняем корректно: 2D Tensor [C,N], PCM16 WAV
        wav = torch.from_numpy(enhanced).reshape(1, -1)
        out_path = self.storage_dir / f"{input_wav.stem}_enhanced.wav"
        torchaudio.save(
            str(out_path),
            wav,
            SAMPLE_RATE,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )
        return out_path

    # === ASR по файлу ===

    def transcribe_file(self, input_wav: Path, language: str = "ru") -> Dict[str, Any]:
        audio = load_and_resample(str(input_wav), target_sr=SAMPLE_RATE)
        result = asr_model.transcribe(audio, language=language)
        text = (result.get("text") or "").strip()
        return {"text": text, "raw": result}



class StreamingASRSession:
    """
    Буферизатор для потокового ASR:
      - накапливает аудио в float32 [-1;1],
      - держит «скользящее окно» последних window_sec,
      - решает, когда выдавать partial (каждые emit_sec).
    """
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        language: str = ASR_LANGUAGE,
        window_sec: float = ASR_WINDOW_SEC,  # окно анализа для частичных гипотез
        emit_sec: float = ASR_EMIT_SEC,    # частота выдачи partial
    ):
        self.sample_rate = sample_rate
        self.language = language
        self.window_sec = window_sec
        self.emit_sec = emit_sec

        self._chunks: List[np.ndarray] = []
        self._frames_since_emit: int = 0

    def add_pcm16(self, chunk: bytes) -> int:
        """Принимает raw PCM16 mono LE, добавляет в буфер, возвращает число добавленных сэмплов."""
        pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        self._chunks.append(pcm)
        frames = pcm.shape[0]
        self._frames_since_emit += frames

        # Укорачиваем до окна window_sec
        total = self.get_audio()
        max_frames = int(self.window_sec * self.sample_rate)
        if total.shape[0] > max_frames:
            total = total[-max_frames:]
            self._chunks = [total]

        return frames

    def should_emit(self) -> bool:
        """Пора ли отдать partial-гипотезу."""
        return self._frames_since_emit >= int(self.emit_sec * self.sample_rate)

    def reset_emit_counter(self):
        self._frames_since_emit = 0

    def get_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(self._chunks, axis=0)

    def transcribe(self) -> Dict[str, Any]:
        """
        Синхронный вызов asr_model.transcribe на текущем окне (выполняется из threadpool).
        Возвращает {"text": str, "raw": dict}.
        """
        audio = self.get_audio()
        if audio.size == 0:
            return {"text": "", "raw": {}}
        result = asr_model.transcribe(audio, language=self.language)
        text = (result.get("text") or "").strip()
        return {"text": text, "raw": result}

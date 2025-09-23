from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import soundfile as sf

from app.services.audio_utils import load_and_resample

from core.audio_enhancement import Audio_Enhancement
from core.config import SAMPLE_RATE, ASR_LANGUAGE, ASR_WINDOW_SEC, ASR_EMIT_SEC, asr_model


class AudioService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # === Препроцессинг аудио (шумоподавление + нормализация) ===
    def enhance_file(self, input_wav: Path) -> Path:
        audio = load_and_resample(str(input_wav))  # -> 1D np.float32
        enh = Audio_Enhancement(audio)
        enhanced = enh.noise_suppression()  # ожидается numpy-массив
        out_path = self.storage_dir / f"{input_wav.stem}_enhanced.wav"
        sf.write(str(out_path), np.asarray(enhanced).squeeze(), SAMPLE_RATE)
        return out_path

    # === ASR по файлу ===
    def transcribe_file(self, input_wav: Path, language: str = "ru") -> Dict[str, Any]:
        audio, sr = sf.read(str(input_wav), dtype="float32", always_2d=False)
        pcm = ensure_pcm16_mono_16k("tests/samples/ru_sample.wav")
        if sr != SAMPLE_RATE:
            raise ValueError(f"Ожидается SAMPLE_RATE={SAMPLE_RATE}, получено {sr}")
        audio = np.asarray(audio).squeeze().astype(np.float32)

        result = asr_model.transcribe(audio, language=language)  # ваш объект из core.config
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

# core/asr/transcriber.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from app import settings

class AsrTranscriber:
    def __init__(
        self,
        model_name: str = settings.ASR_MODEL,
        device: str = settings.ASR_DEVICE,
        compute_type: str = settings.ASR_COMPUTE_TYPE,
        language: Optional[str] = settings.ASR_LANGUAGE or None,
        max_chars: int = settings.REF_TEXT_MAX_CHARS,
    ):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language
        self.max_chars = max_chars

    def transcribe(self, audio_path: Path) -> str:
        # Faster-Whisper возвращает генератор сегментов
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,  # None => autodetect
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        parts = []
        for seg in segments:
            parts.append(seg.text.strip())
            if sum(len(p) for p in parts) >= self.max_chars:
                break
        text = " ".join(parts).strip()
        # лёгкая нормализация
        text = text.replace("  ", " ")
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0]  # не резать слово посередине
        return text

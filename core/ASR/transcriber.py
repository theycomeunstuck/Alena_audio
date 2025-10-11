# core/asr/transcriber.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import whisper
from core.config import WHISPER_MODEL, device, ASR_LANGUAGE

class AsrTranscriber:
    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        device: str = device,
        language: Optional[str] = ASR_LANGUAGE or None,
        max_chars=None,
    ):
        self.model = whisper.load_model(model_name, device=device)
        self.language = language
        self.max_chars = max_chars

    def transcribe(self, audio_path: Path) -> str:
        # Standard OpenAI Whisper transcription
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,  # None => autodetect
        )
        text = result.get("text", "").strip()
        
        # лёгкая нормализация
        text = text.replace("  ", " ")
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0]  # не резать слово посередине
        return text

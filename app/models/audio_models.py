from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Literal, Tuple, Union

class EnhanceResponse(BaseModel):
    output_filename: str

class WhisperSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

class WhisperRaw(BaseModel):
    text: str
    segments: List[WhisperSegment] = Field(default_factory=list)
    language: str

class TranscribeResponse(BaseModel):
    text: str
    raw: Optional[WhisperRaw] = Field(
        default=None,
        description="Присутствует только при verbose=true",
    )

class TtsIn(BaseModel):
    text: str = Field(None, description="Текст для синтеза")
    voice_id: Optional[str] = Field(None, description="Идентификатор голоса; если не задан, используется базовый (_default)")
    format: Literal["wav","mp3","ogg"] = Field("wav", description="Формат выходного аудио")
    stress: bool = Field(True, description="Использовать модуль для автоматической расстановки ударений, например: я заб+ыл закр+ыть зам+ок от з+амка (silero-stress)")

class CloneOut(BaseModel):
    voice_id: str

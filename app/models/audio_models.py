from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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
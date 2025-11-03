from pydantic import BaseModel, Field
from typing import Optional, List

class VerifyResponse(BaseModel):
    score: float
    decision: bool

class TrainMicResponse(BaseModel):
    user_id: str = Field(..., description="uuid4 Пользователя")
    wav_path: str = Field(..., description="Путь к референс-файлу .wav")
    npy_path: str = Field(..., description="Путь к эмбеддинг-файлу .npy")


class MultiVerifyMatch(BaseModel):
    user_id: str = Field(..., description="ID пользователя/голоса")
    score: float = Field(..., description="Схожесть в диапазоне [0..1] (ближе к 1 — лучше)")
    ref_path: str = Field(..., description="Путь к референс-файлу WAV")

class MultiVerifyResponse(BaseModel):
    count: int
    matches: list[MultiVerifyMatch]

class RegistryVerifyResponse(BaseModel):
    threshold: float = Field(..., description="Порог бинарного решения в [0..1]")
    decision: bool = Field(..., description="Есть ли совпадение с кем-то из реестра при заданном пороге")
    best: Optional[MultiVerifyMatch] = Field(None, description="Лучшее совпадение, если decision=true")
    matches: Optional[List[MultiVerifyMatch]] = Field(default=None, description="Диагностический Top-K список совпадений")

class RegistryReloadResposne(BaseModel):
        id_list: Optional[List] = Field(..., description="Список пользователей, которые  есть в базе данных")
        count: int = Field(..., description="Количество пользователей в базе данных")


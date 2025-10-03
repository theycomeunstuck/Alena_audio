# tests/conftest.py
import sys, os, io
from pathlib import Path


# Корень проекта = родительская папка над tests/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pytest
import torchaudio, torch

from fastapi.testclient import TestClient

from app.main import app
from core import config





@pytest.fixture(scope="session")
def client():
    # Синхронный тестовый клиент Starlette/FASTAPI
    return TestClient(app)


def make_sine(sr: int, duration: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return x


def wav_bytes_from_np(x: np.ndarray, sr: int) -> bytes:
    """Создаёт WAV (RIFF) из 1D float32 numpy через torchaudio.save в память."""
    tensor = torch.from_numpy(x).unsqueeze(0)  # [1, N]
    buf = io.BytesIO()
    torchaudio.save(buf, tensor, sr, format="wav")
    return buf.getvalue()


def pcm16_from_np(x: np.ndarray) -> bytes:
    """Преобразует float32 [-1,1] -> PCM16 little-endian bytes."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


@pytest.fixture()
def mock_asr_transcribe(monkeypatch):
    """
    Подменяем asr_model.transcribe, чтобы не тянуть реальную модель.
    Возвращает текст, зависящий от длины входа -> удобно проверять partial.
    """
    def fake_transcribe(audio: np.ndarray, language: str = "ru"):
        n = int(audio.shape[0])
        # «текст» зависит от длины, но стабилен
        return {"text": f"len={n}"}

    monkeypatch.setattr(config.asr_model, "transcribe", fake_transcribe)
    return fake_transcribe


@pytest.fixture()
def mock_enhancement(monkeypatch):
    """
    Подменяем Audio_Enhancement, чтобы не гонять шумодав и спикер-верификацию.
    """
    import app.services.audio_service as audio_service_mod
    import core.audio_enhancement as core_enh

    class DummyEnh:
        def __init__(self, audio, ref=None):
            self.audio = np.asarray(audio, dtype=np.float32)
            self.ref = None if ref is None else np.asarray(ref, dtype=np.float32)

        def noise_suppression(self):
            # имитируем «улучшение» — просто возвращаем как есть
            return self.audio

        def speech_verification(self):
            # стабильная «верификация»
            return (0.91, True)

    # Патчим и место, откуда вызывает сервис
    monkeypatch.setattr(core_enh, "Audio_Enhancement", DummyEnh)
    monkeypatch.setattr(audio_service_mod, "Audio_Enhancement", DummyEnh)
    return DummyEnh

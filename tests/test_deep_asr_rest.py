# tests/test_deep_asr_rest.py
import os, pytest, io
import numpy as np
import soundfile as sf
import torch

from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from core.config import SAMPLE_RATE


pytestmark = pytest.mark.deep
client = TestClient(app)


def wav_bytes_from_tensor(x: torch.Tensor, sr: int) -> bytes:
    # F-020: written with soundfile, like the product IO (no torchcodec needed).
    buf = io.BytesIO()
    sf.write(buf, x.detach().cpu().numpy().T, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()

@pytest.mark.deep
def test_transcribe_real_model_minimal():
    """
    НЕ мок: реальный asr_model.transcribe.
    Тест корректности протокола и устойчивости: допускаем пустой текст.
    """

    sample_path = Path(__file__).resolve().parent / "samples" / "ru_sample.wav"
    assert sample_path.exists(), "Положи файл с речью в tests/samples/ru_sample.wav"

    files = {"file": ("ru_sample.wav", sample_path.read_bytes(), "audio/wav")}

    r = client.post("/audio/transcribe", files=files, params={"language": "ru"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "text" in data or ["text", "raw"] in data # verbose flag depended
    assert isinstance(data["text"], str)  # текст может быть пустым — не валимся



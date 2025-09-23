# tests/test_deep_speaker_verify.py
import os, io, pytest
import torch, torchaudio
import numpy as np

from fastapi.testclient import TestClient
from app.main import app
from core.config import SAMPLE_RATE
from pathlib import Path


client = TestClient(app)
pytestmark = pytest.mark.deep

def wav_bytes_from_tensor(x: torch.Tensor, sr: int) -> bytes:
    buf = io.BytesIO()
    torchaudio.save(buf, x, sr, format="wav")
    return buf.getvalue()

def test_speaker_verify_real():
    sr = SAMPLE_RATE
    t = torch.linspace(0, 0.6, int(sr*0.6), dtype=torch.float32)[:-1]
    # небольшой шум вместо речи — цель: проверить устойчивость пайплайна, а не качество
    x = 0.02 * torch.randn_like(t)
    probe = wav_bytes_from_tensor(x.unsqueeze(0), sr)

    files = {"probe": ("probe.wav", probe, "audio/wav")}
             # "reference": ("reference.wav", x, "audio/wav")}
    r = client.post("/speaker/verify", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "score" in data and "decision" in data
    assert isinstance(data["decision"], bool)

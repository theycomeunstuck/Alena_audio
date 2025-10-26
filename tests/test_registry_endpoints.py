# tests/test_registry_endpoints.py
import io
import numpy as np
import torch
import torchaudio
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from core.config import VOICES_DIR, EMBEDDINGS_DIR, SAMPLE_RATE

client = TestClient(app)

def _sine(sr: int, seconds: float, freq: float) -> torch.Tensor:
    t = torch.arange(int(sr*seconds)) / sr
    return 0.1 * torch.sin(2*np.pi*freq * t)

def test_registry_reload_and_topk(tmp_path: Path, monkeypatch):
    sr = SAMPLE_RATE

    # создадим 2 пользователя в VOICES_DIR
    (VOICES_DIR / "u1").mkdir(parents=True, exist_ok=True)
    (VOICES_DIR / "u2").mkdir(parents=True, exist_ok=True)
    v1 = _sine(sr, 8.25, 220.0)
    v2 = _sine(sr, 10.25, 440.0)
    torchaudio.save(str(VOICES_DIR / "u1" / "reference.wav"), v1.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16, format="wav")
    torchaudio.save(str(VOICES_DIR / "u2" / "reference.wav"), v2.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16, format="wav")

    # перезагрузка реестра
    r = client.post("/speaker/registry/reload")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
    assert r.json()["count"] >= 2

    # probe ~220 Hz -> ближе к u1
    buf = io.BytesIO()
    torchaudio.save(buf, v1.unsqueeze(0), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    r2 = client.post("/speaker/verify/topk?top_k=2", files={"probe": ("p.wav", buf.getvalue(), "audio/wav")})
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert data["count"] == 2
    assert data["matches"][0]["user_id"] in ("u1","u2")

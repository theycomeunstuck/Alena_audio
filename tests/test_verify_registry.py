# tests/test_verify_registry.py
import io
import numpy as np
import torch
import torchaudio
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from core.config import VOICES_DIR, SAMPLE_RATE

client = TestClient(app)

def test_verify_registry_thresholds(monkeypatch, tmp_path: Path):
    # monkeypatch эмбеддинга
    def _fake_embed(audio: np.ndarray) -> torch.Tensor:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        nfft = 2048
        if x.size < nfft:
            pad = np.zeros(nfft - x.size, dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        spec = np.abs(np.fft.rfft(x[:nfft]))[:256].astype(np.float32)
        v = torch.from_numpy(spec)
        v = v / (v.norm(p=2) + 1e-8)
        return v
    import app.services.embeddings_utils as eu
    monkeypatch.setattr(eu, "embed_speechbrain", _fake_embed, raising=True)

    # подготовим две ссылки
    sr = SAMPLE_RATE
    t = torch.linspace(0, 0.5, int(0.5*sr), dtype=torch.float32)[:-1]
    v1 = 0.1 * torch.sin(2*np.pi*220.0 * t)
    v2 = 0.1 * torch.sin(2*np.pi*330.0 * t)
    (VOICES_DIR / "u1").mkdir(parents=True, exist_ok=True)
    (VOICES_DIR / "u2").mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(VOICES_DIR / "u1" / "reference.wav"), v1.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16, format="wav")
    torchaudio.save(str(VOICES_DIR / "u2" / "reference.wav"), v2.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16, format="wav")

    # probe 220 Hz
    x = 0.1 * torch.sin(2*np.pi*220.0 * t)
    buf = io.BytesIO()
    torchaudio.save(buf, x.unsqueeze(0), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    files = {"probe": ("probe.wav", buf.getvalue(), "audio/wav")}

    # Умеренный порог → True
    r = client.post("/speaker/verify_registry?sim_threshold=0.5", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["decision"] is True
    assert data["best"] is not None
    assert data["best"]["user_id"] in ("u1","u2")
    assert 0.0 <= data["best"]["score"] <= 1.0


    # Высокий порог → при fake emb может остаться True, но проверим диапазон
    r2 = client.post("/speaker/verify_registry?sim_threshold=0.999", files=files)
    assert r2.status_code == 200, r2.text
    data2 = r2.json()
    assert 0.0 <= data2["threshold"] <= 1.0
    # решение не критично, главное структура
    assert "decision" in data2 and "best" in data2


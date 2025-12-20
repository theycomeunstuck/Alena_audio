# tests/test_multi_speaker_verify.py
import io, json
import numpy as np
import torch, torchaudio
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from core.config import VOICES_DIR, SAMPLE_RATE
from fastapi.websockets import WebSocketDisconnect

client = TestClient(app)

def _write_wav(path: Path, x: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), x, sr, encoding="PCM_S", bits_per_sample=16, format="wav")

def _bytes_from_np1d(x: np.ndarray, sr: int) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)
    return x16.tobytes()

def _fake_embed(audio: np.ndarray) -> torch.Tensor:
    """Дешёвый детерминированный эмбеддинг по спектру, L2-нормированный."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    # простой спектральный вектор фиксированной длины
    nfft = 2048
    if x.size < nfft:
        pad = np.zeros(nfft - x.size, dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    spec = np.abs(np.fft.rfft(x[:nfft]))[:256].astype(np.float32)  # 256 фичей
    v = torch.from_numpy(spec)
    v = v / (v.norm(p=2) + 1e-8)
    return v

def setup_module(module):
    # Monkeypatch embed_speechbrain ДО импортов матчера
    import app.services.embeddings_utils as eu
    eu.embed_speechbrain = _fake_embed  # type: ignore

    # Prepare two dummy voices in storage/voices
    sr = SAMPLE_RATE
    t = torch.linspace(0, 0.5, int(0.5*sr), dtype=torch.float32)[:-1]
    v1 = 0.1 * torch.sin(2*np.pi*220.0 * t)
    v2 = 0.1 * torch.sin(2*np.pi*330.0 * t)
    _write_wav(VOICES_DIR / "u1" / "reference.wav", v1.unsqueeze(0), sr)
    _write_wav(VOICES_DIR / "u2" / "reference.wav", v2.unsqueeze(0), sr)

def test_rest_verify_all_returns_matches():
    # Probe tone 220 Hz → ближе к u1
    sr = SAMPLE_RATE
    t = np.linspace(0, 0.5, int(0.5*sr), endpoint=False, dtype=np.float32)
    x = 0.1 * np.sin(2*np.pi*220.0 * t).astype(np.float32)
    buf = io.BytesIO()
    torchaudio.save(buf, torch.from_numpy(x).unsqueeze(0), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    files = {"probe": ("probe.wav", buf.getvalue(), "audio/wav")}
    r = client.post("/speaker/verify/topk?top_k=2", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "matches" in data and isinstance(data["matches"], list)
    assert data["count"] == len(data["matches"])
    assert all(0.0 <= m["score"] <= 1.0 for m in data["matches"])

def test_ws_verify_basic_flow():
    # Open WS and send a short chunk, then flush
    def recv_until_any(ws, types=("partial", "final"), max_msgs=20):
        for _ in range(max_msgs):
            msg = ws.receive_json()
            if msg.get("type") in types:
                return msg
        raise AssertionError("Did not receive partial/final within limit")

    sr = SAMPLE_RATE
    with client.websocket_connect(f"/ws/speaker/verify?sample_rate={sr}&channels=1&top_k=2") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        x = np.random.uniform(-0.1, 0.1, int(0.2*sr)).astype(np.float32)  # 200ms random
        ws.send_bytes(_bytes_from_np1d(x, sr))
        ws.send_text(json.dumps({"event": "flush"}))

        ws.send_text(json.dumps({"event": "flush"}))
        msg = recv_until_any(ws)
        assert msg["type"] in ("partial", "final")
        assert "matches" in msg
        assert all(0.0 <= m["score"] <= 1.0 for m in msg["matches"])

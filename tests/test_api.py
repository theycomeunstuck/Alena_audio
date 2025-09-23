# tests/test_api.py
import io
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.config import SAMPLE_RATE

#pytest -v tests/
client = TestClient(app)

def generate_wav(sr=SAMPLE_RATE, duration=1.0, freq=440.0):
    """Генерируем простую синусоиду как wav-байты."""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    buf.seek(0)
    return buf

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_file_upload_and_download(tmp_path):
    wav_buf = generate_wav()
    files = {"file": ("test.wav", wav_buf, "audio/wav")}
    r = client.post("/files/upload", files=files)
    assert r.status_code == 200
    fname = r.json()["filename"]

    r2 = client.get(f"/files/download/{fname}")
    assert r2.status_code == 200
    assert r2.content.startswith(b"RIFF")  # заголовок WAV

def test_audio_enhance(tmp_path):
    wav_buf = generate_wav()
    files = {"file": ("test.wav", wav_buf, "audio/wav")}
    r = client.post("/audio/enhance", files=files)
    assert r.status_code == 200
    assert "output_filename" in r.json()

def test_audio_transcribe_mock(monkeypatch):
    # Заглушим asr_model.transcribe
    def fake_transcribe(audio, language="ru"):
        return {"text": "hello world"}
    from core import config
    monkeypatch.setattr(config.asr_model, "transcribe", fake_transcribe)

    wav_buf = generate_wav()
    files = {"file": ("test.wav", wav_buf, "audio/wav")}
    r = client.post("/audio/transcribe", files=files)
    assert r.status_code == 200
    assert r.json()["text"] == "hello world"

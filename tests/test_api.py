# tests/test_api.py
import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from core.config import SAMPLE_RATE

client = TestClient(app)


def generate_wav(sr=SAMPLE_RATE, duration=1.0, freq=440.0):
    """Генерируем простую синусоиду как wav-байты (PCM16 mono через модуль wave).

    torchaudio.save в этом окружении не работает (torchcodec не может загрузить
    свою DLL — см. tests/test_voices_cache.py). Upload/download routes не
    декодируют содержимое, поэтому тесту torchaudio не нужен: пишем валидный
    WAV стандартным модулем wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_file_upload_and_download(tmp_path, monkeypatch):
    # routes_files читает settings.STORAGE_DIR на каждый вызов — патчим его,
    # чтобы тест не писал в настоящий storage/.
    monkeypatch.setattr("app.settings.STORAGE_DIR", tmp_path, raising=False)

    wav_buf = generate_wav()
    files = {"file": ("test.wav", wav_buf, "audio/wav")}
    r = client.post("/files/upload", files=files)
    assert r.status_code == 200
    fname = r.json()["filename"]
    assert fname == "test.wav"

    r2 = client.get(f"/files/download/{fname}")
    assert r2.status_code == 200
    assert r2.content.startswith(b"RIFF")  # заголовок WAV
    assert r2.content == wav_buf.getvalue()  # содержимое дошло без изменений


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

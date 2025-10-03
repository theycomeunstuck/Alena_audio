# tests/test_endpoints_rest.py
import numpy as np
import io

from core.config import SAMPLE_RATE
from tests.conftest import make_sine, wav_bytes_from_np


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_and_download(client):
    x = make_sine(SAMPLE_RATE, 0.1)
    files = {"file": ("simple.wav", wav_bytes_from_np(x, SAMPLE_RATE), "audio/wav")}
    r_up = client.post("/files/upload", files=files)
    assert r_up.status_code == 200
    fname = r_up.json()["filename"]

    r_down = client.get(f"/files/download/{fname}")
    assert r_down.status_code == 200
    assert r_down.content[:4] == b"RIFF"  # WAV заголовок


def test_audio_enhance(client, mock_enhancement):
    x = make_sine(SAMPLE_RATE, 0.1)
    files = {"file": ("a.wav", wav_bytes_from_np(x, SAMPLE_RATE), "audio/wav")}
    r = client.post("/audio/enhance", files=files)
    assert r.status_code == 200
    assert "output_filename" in r.json()
    # можем скачать и убедиться, что это WAV
    out = r.json()["output_filename"]
    r2 = client.get(f"/files/download/{out}")
    assert r2.status_code == 200
    assert r2.content[:4] == b"RIFF"


def test_audio_transcribe(client, mock_asr_transcribe):
    x = make_sine(SAMPLE_RATE, 0.15)
    files = {"file": ("b.wav", wav_bytes_from_np(x, SAMPLE_RATE), "audio/wav")}
    r = client.post("/audio/transcribe", files=files, params={"language": "ru"})
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert data["text"].startswith("len=")


def test_speaker_verify(client, mock_enhancement):
    # probe и reference обе подаются, verify вернёт заглушку (0.91, True)
    x = make_sine(SAMPLE_RATE, 0.12, freq=300)
    y = make_sine(SAMPLE_RATE, 0.12, freq=500)
    files = {
        "probe": ("p.wav", wav_bytes_from_np(x, SAMPLE_RATE), "audio/wav"),
        "reference": ("r.wav", wav_bytes_from_np(y, SAMPLE_RATE), "audio/wav"),
    }
    r = client.post("/speaker/verify", files=files)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["score"] <= 1.0
    assert isinstance(data["decision"], bool)

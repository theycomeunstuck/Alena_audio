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


def test_speaker_verify(client, monkeypatch, tmp_path):
    """Route contract for /speaker/verify with the service seam faked.

    The old version relied on the conftest Audio_Enhancement fake, but
    SpeakerService.verify_files never uses Audio_Enhancement — it decodes the
    uploads and embeds them via speechbrain (ECAPA). That made the test load a
    real model in a healthy environment and die in the broken torchaudio
    decoder here. The route's own contract lives at the service seam: temp
    files are written per request and the {score, decision} dict passes
    through the VerifyResponse model.
    """
    import app.api.routes_speaker as rs

    # probe/reference temp files must not land in the real storage tree
    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path, raising=False)

    def fake_verify_files(probe_wav, reference_wav=None):
        from pathlib import Path

        assert Path(probe_wav).read_bytes() == b"PROBE-BYTES", (
            "the service must read this request's probe upload")
        assert reference_wav is not None and Path(reference_wav).read_bytes() == b"REF-BYTES", (
            "the service must receive this request's reference upload")
        return {"score": 0.91, "decision": True}

    monkeypatch.setattr(rs.svc, "verify_files", fake_verify_files, raising=True)

    files = {
        "probe": ("p.wav", b"PROBE-BYTES", "audio/wav"),
        "reference": ("r.wav", b"REF-BYTES", "audio/wav"),
    }
    r = client.post("/speaker/verify", files=files)
    assert r.status_code == 200
    data = r.json()
    # exact passthrough of the service dict (score/decision) through the route
    assert data["score"] == 0.91
    assert data["decision"] is True
    assert list(tmp_path.iterdir()) == [], "probe/reference temp files must be cleaned up"

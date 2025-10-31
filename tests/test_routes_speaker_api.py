import io
import numpy as np
import pytest

import app.api.routes_speaker as rs


def test_verify_ok_and_errors(client, monkeypatch):
    # Успешный ответ сервиса
    monkeypatch.setattr(rs.svc, "verify_files", lambda probe, ref: {"score": 0.87, "decision": True}, raising=True)

    wav = io.BytesIO(b"RIFF....WAVEfmt ")  # контент неважен — мы мокаем сервис
    files = {
        "probe": ("p.wav", wav.getvalue(), "audio/wav"),
        "reference": ("r.wav", wav.getvalue(), "audio/wav"),
    }
    r = client.post("/speaker/verify", files=files)
    assert r.status_code == 200
    assert r.json() == {"score": 0.87, "decision": True}

    # Ошибка «нет дефолтного референса» → 400
    def _raise_valerr(probe, ref): raise ValueError("no default reference")
    monkeypatch.setattr(rs.svc, "verify_files", _raise_valerr, raising=True)
    r2 = client.post("/speaker/verify", files={"probe": ("p.wav", wav.getvalue(), "audio/wav")})
    assert r2.status_code == 400

    # Прочая ошибка → 500
    def _raise_any(probe, ref): raise RuntimeError("boom")
    monkeypatch.setattr(rs.svc, "verify_files", _raise_any, raising=True)
    r3 = client.post("/speaker/verify", files={"probe": ("p.wav", wav.getvalue(), "audio/wav")})
    assert r3.status_code == 500
# (роуты вокруг speaker_service: обработка успешного кейса и ошибок)


def test_train_microphone_returns_paths(client, monkeypatch, tmp_path):
    ret = {"user_id": "u1", "wav_path": str(tmp_path / "u1.wav"), "npy_path": str(tmp_path / "u1.npy")}
    monkeypatch.setattr(rs.svc, "train_from_microphone", lambda user_id, duration: ret, raising=True)
    r = client.post("/speaker/train/microphone?user_id=u1&duration=1.5")
    assert r.status_code == 200
    js = r.json()
    assert js["user_id"] == "u1" and "wav_path" in js and "npy_path" in js
# (роут train_from_microphone)



#todo: сделать проверку GPU if available + test WebSocket
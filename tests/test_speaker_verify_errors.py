# Audit F-065: an all-silence probe (no embedding) and a missing default
# reference are client errors (400), not 500s; ValueError reaches the route.
import io

import numpy as np
import pytest

import app.api.routes_speaker as rs
import app.services.speaker_service as speaker_service
from tests.test_routes_speaker_api import _patch_route_globals

WAV = io.BytesIO(b"RIFF....WAVEfmt ").getvalue()


@pytest.fixture
def silent_service(monkeypatch, tmp_path):
    storage = _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(speaker_service, "load_and_resample", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(speaker_service, "embed_speechbrain", lambda audio: None)
    return storage


def test_silent_probe_is_a_value_error(silent_service, tmp_path):
    probe = tmp_path / "p.wav"
    probe.write_bytes(WAV)
    with pytest.raises(ValueError):
        rs.svc.verify_files(probe, probe)


def test_silent_probe_maps_to_400(client, silent_service):
    files = {"probe": ("p.wav", WAV, "audio/wav"), "reference": ("r.wav", WAV, "audio/wav")}
    response = client.post("/speaker/verify", files=files)
    assert response.status_code == 400, response.text


def test_missing_default_reference_maps_to_400(client, silent_service, monkeypatch):
    monkeypatch.setattr(rs.svc, "_find_default_ref", lambda: None)
    response = client.post("/speaker/verify", files={"probe": ("p.wav", WAV, "audio/wav")})
    assert response.status_code == 400, response.text


def test_unexpected_embedding_failure_stays_a_generic_500(client, silent_service, monkeypatch):
    def broken(audio):
        raise RuntimeError("encoder crashed")
    monkeypatch.setattr(speaker_service, "embed_speechbrain", broken)
    files = {"probe": ("p.wav", WAV, "audio/wav"), "reference": ("r.wav", WAV, "audio/wav")}
    response = client.post("/speaker/verify", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "internal error"

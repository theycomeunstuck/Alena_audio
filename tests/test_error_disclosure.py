# Audit F-066: API errors must not echo raw exception text or server paths,
# and the torch pickle-loading warning must not be silenced.
import io
import re
from pathlib import Path

import pytest

import app.api.routes_speaker as rs
from tests.test_routes_speaker_api import _patch_route_globals

SECRET = r"C:\srv\secret\models\ecapa.ckpt"
WAV = io.BytesIO(b"RIFF....WAVEfmt ").getvalue()


def _boom(*args, **kwargs):
    raise RuntimeError(f"failed to open {SECRET}")


def _assert_generic_500(response):
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "internal error"
    assert "secret" not in detail and "RuntimeError" not in detail


def test_verify_internal_error_hides_exception_text(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(rs.svc, "verify_files", _boom, raising=True)
    _assert_generic_500(client.post("/speaker/verify", files={"probe": ("p.wav", WAV, "audio/wav")}))


@pytest.mark.parametrize("path", ["/speaker/verify_registry", "/speaker/verify/topk"])
def test_registry_matching_internal_error_hides_exception_text(client, monkeypatch, tmp_path, path):
    _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(rs, "load_and_resample", lambda buf: [0.0], raising=True)
    monkeypatch.setattr(rs.matcher, "match_probe_array", _boom, raising=True)
    _assert_generic_500(client.post(path, files={"probe": ("p.wav", WAV, "audio/wav")}))


def test_registry_reload_internal_error_hides_exception_text(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(rs.matcher, "reload", _boom, raising=True)
    _assert_generic_500(client.post("/speaker/registry/reload"))


def test_train_microphone_internal_error_hides_exception_text(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(rs.svc, "train_from_microphone", _boom, raising=True)
    _assert_generic_500(client.post("/speaker/train/microphone?user_id=u1&duration=1"))


def test_train_responses_return_storage_relative_paths(client, monkeypatch, tmp_path):
    storage = _patch_route_globals(monkeypatch, tmp_path)
    ret = {"user_id": "u1", "wav_path": str(storage / "voices" / "u1.wav"),
           "npy_path": str(storage / "embeddings" / "u1.npy")}
    monkeypatch.setattr(rs.svc, "train_from_microphone", lambda user_id, duration: ret, raising=True)
    monkeypatch.setattr(rs.matcher, "reload", lambda *a, **k: ([], 0), raising=True)
    body = client.post("/speaker/train/microphone?user_id=u1&duration=1").json()
    assert body["wav_path"] == "voices/u1.wav"
    assert body["npy_path"] == "embeddings/u1.npy"
    assert str(tmp_path) not in str(body)


def test_train_response_outside_storage_keeps_only_the_file_name(client, monkeypatch, tmp_path):
    _patch_route_globals(monkeypatch, tmp_path)
    outside = tmp_path / "elsewhere"
    ret = {"user_id": "u1", "wav_path": str(outside / "u1.wav"), "npy_path": str(outside / "u1.npy")}
    monkeypatch.setattr(rs.svc, "train_from_microphone", lambda user_id, duration: ret, raising=True)
    monkeypatch.setattr(rs.matcher, "reload", lambda *a, **k: ([], 0), raising=True)
    body = client.post("/speaker/train/microphone?user_id=u1&duration=1").json()
    assert (body["wav_path"], body["npy_path"]) == ("u1.wav", "u1.npy")


def test_config_does_not_silence_torch_pickle_warning():
    source = (Path(__file__).resolve().parents[1] / "core" / "config.py").read_text(encoding="utf-8")
    ignores = [line for line in source.splitlines() if "filterwarnings" in line and "ignore" in line]
    assert not [line for line in ignores if re.search(r"weights_only", line)]


def test_registry_matches_name_the_embedding_file_without_a_server_path(monkeypatch, tmp_path):
    import numpy as np
    import torch
    import app.services.multi_speaker_matcher as msm

    np.save(tmp_path / "alena.npy", np.asarray([1.0, 0.0], dtype=np.float32))

    class NoOpEnhancement:
        def __init__(self, audio):
            self.audio = audio

        def noise_suppression(self):
            return self.audio

    monkeypatch.setattr(msm, "CFG_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(msm, "Audio_Enhancement", NoOpEnhancement, raising=True)
    monkeypatch.setattr(msm, "embed_speechbrain", lambda audio: torch.tensor([1.0, 0.0]), raising=True)
    matcher = msm.MultiSpeakerMatcher(embeddings_dir=tmp_path)
    matcher.reload()
    match = matcher.match_probe_array(np.zeros(16000, dtype=np.float32), top_k=1)[0]
    assert match["ref_path"] == "alena.npy"
    assert str(tmp_path) not in str(match)


def test_missing_reference_hint_lists_no_absolute_paths(monkeypatch, tmp_path):
    import numpy as np
    import app.services.speaker_service as speaker_service

    storage = _patch_route_globals(monkeypatch, tmp_path)
    monkeypatch.setattr(speaker_service, "load_and_resample", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(rs.svc, "_find_default_ref", lambda: None)
    probe = tmp_path / "p.wav"
    probe.write_bytes(WAV)
    with pytest.raises(ValueError) as caught:
        rs.svc.verify_files(probe)
    message = str(caught.value)
    root = str(Path(__file__).resolve().parents[1])
    assert root not in message and str(storage) not in message
    assert "STORAGE_DIR/_default/reference.wav" in message


def test_audio_enhancement_has_no_dead_speech_verification():
    """Audit F-067: Audio_Enhancement.speech_verification compared an ECAPA
    embedding with a raw waveform and had no live callers; the live speaker
    check is app/services/speaker_service.verify_files."""
    from core.audio_enhancement import Audio_Enhancement
    assert not hasattr(Audio_Enhancement, "speech_verification")

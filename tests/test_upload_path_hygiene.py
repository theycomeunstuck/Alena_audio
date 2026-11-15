# Regression tests for audit findings F-021, F-021b/F-026 and F-025b.
#
#   F-021        – user_id / voice_id reach filesystem paths unvalidated
#                  (traversal writes outside embeddings/voices dirs, registry
#                  poisoning, existence oracle).
#   F-021b/F-026 – probe/upload temp files use predictable shared names and are
#                  never deleted (cross-request mixing, unbounded disk growth);
#                  /tts/clone NamedTemporaryFile(delete=False) is never removed.
#   F-025b       – /speaker/train/microphone accepts an unbounded `duration`,
#                  letting an unauthenticated caller pin a threadpool worker.
#
# Route tests patch the route module's STORAGE_DIR into tmp_path so nothing is
# written into the real storage tree, and the service methods are mocked so no
# model inference (and no microphone access) ever happens in these tests.
import inspect
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routes_speaker as rs
import app.api.routes_audio as ra
import app.api.routes_TTS as rt
from app.main import app
from app.services.speaker_service import SpeakerService


@pytest.fixture()
def client_soft():
    # Does not raise on unhandled server exceptions: the pre-fix behaviour of
    # some endpoints here is a raw 500, which must surface as a response.
    return TestClient(app, raise_server_exceptions=False)


def _bomb(msg="must not be reached"):
    def _f(*args, **kwargs):
        raise AssertionError(msg)
    return _f


BAD_USER_IDS = [
    "../escape_uid",
    "..\\escape_uid",
    "sub/dir",
    "..",
    ".",
    "has space",
    "dot.in.id",
    "id$inject",
    "a" * 65,   # longer than the 64-character cap
    "",         # empty must be rejected too
]

BAD_VOICE_IDS = [
    "../escape",
    "..\\escape",
    "sub/dir",
    "..",
    "has space",
    "dot.in.id",
    "a" * 65,
]


# --------------------------------------------------------------------- F-021
@pytest.mark.parametrize("bad_id", BAD_USER_IDS)
def test_train_file_rejects_unsafe_user_id(client_soft, monkeypatch, tmp_path, bad_id):
    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path, raising=False)
    # The route must reject the id BEFORE touching the upload content.
    monkeypatch.setattr(rs, "load_and_resample", lambda buf: np.zeros(16, dtype=np.float32), raising=True)
    monkeypatch.setattr(rs.svc, "train_from_file",
                        _bomb("train_from_file must not be called with an unsafe user_id"), raising=True)
    r = client_soft.post("/speaker/train/file",
                         files={"probe": ("p.wav", b"JUNK", "audio/wav")},
                         params={"user_id": bad_id})
    assert r.status_code == 400, f"user_id {bad_id!r} must be rejected with 400, got {r.status_code}: {r.text[:120]}"


@pytest.mark.parametrize("bad_id", BAD_USER_IDS)
def test_train_microphone_rejects_unsafe_user_id(client_soft, monkeypatch, bad_id):
    # The service (and therefore the microphone) must never be reached.
    monkeypatch.setattr(rs.svc, "train_from_microphone",
                        _bomb("train_from_microphone must not be called with an unsafe user_id"), raising=True)
    r = client_soft.post("/speaker/train/microphone", params={"user_id": bad_id, "duration": 1})
    assert r.status_code == 400, f"user_id {bad_id!r} must be rejected with 400, got {r.status_code}: {r.text[:120]}"


def test_speaker_service_train_from_file_rejects_unsafe_user_id(tmp_path):
    # Service-boundary guard: the same strict validation inside SpeakerService.
    svc = SpeakerService(tmp_path)
    with pytest.raises(ValueError, match="user_id"):
        svc.train_from_file(np.zeros(16, dtype=np.float32), user_id="../escape_uid")


def test_speaker_service_train_from_microphone_rejects_unsafe_user_id(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.speaker_service.record_audio",
                        _bomb("record_audio must never run for an unsafe user_id"))
    svc = SpeakerService(tmp_path)
    with pytest.raises(ValueError, match="user_id"):
        svc.train_from_microphone(user_id="..\\escape_uid")


def test_train_file_accepts_safe_user_id(client, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(rs, "load_and_resample", lambda buf: np.zeros(16, dtype=np.float32), raising=True)
    got = {}

    def fake_train(probe_wav, user_id):
        got["user_id"] = user_id
        return {"user_id": user_id, "wav_path": "w.wav", "npy_path": "n.npy"}

    monkeypatch.setattr(rs.svc, "train_from_file", fake_train, raising=True)
    r = client.post("/speaker/train/file",
                    files={"probe": ("p.wav", b"JUNK", "audio/wav")},
                    params={"user_id": "teacher_1"})
    assert r.status_code == 200
    assert got["user_id"] == "teacher_1"
    assert r.json()["user_id"] == "teacher_1"


def test_resource_id_validation_allows_default_and_max_len():
    from app.services.speaker_service import validate_resource_id
    assert validate_resource_id("_default", "user_id") == "_default"
    assert validate_resource_id("u" * 64, "user_id") == "u" * 64
    assert validate_resource_id("voice-9_X", "voice_id") == "voice-9_X"


@pytest.mark.parametrize("bad_vid", BAD_VOICE_IDS)
def test_tts_rejects_unsafe_voice_id(client_soft, bad_vid):
    r = client_soft.post("/tts", json={"text": "hi", "voice_id": bad_vid})
    assert r.status_code == 400, f"voice_id {bad_vid!r} must be rejected with 400, got {r.status_code}: {r.text[:120]}"


def test_tts_missing_but_wellformed_voice_id_still_404(client):
    # A syntactically valid, unknown voice must keep its old 404 semantics.
    r = client.post("/tts", json={"text": "hi", "voice_id": "no_such_voice_xyz"})
    assert r.status_code == 404


# ------------------------------------------------------------- F-021b / F-026
def test_speaker_verify_uses_unique_temp_files_and_cleans_up(client, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path, raising=False)
    seen = []

    def fake_verify(probe_wav, reference_wav=None):
        p = Path(probe_wav)
        ref = Path(reference_wav) if reference_wav is not None else None
        assert p.exists(), "probe temp file must exist while the service reads it"
        assert p.read_bytes() == b"PROBE-BYTES", "the service must read this request's probe, not another's"
        assert ref is not None and ref.exists(), "reference temp file must exist while the service reads it"
        seen.append((p, ref))
        return {"score": 0.0, "decision": False}

    monkeypatch.setattr(rs.svc, "verify_files", fake_verify, raising=True)

    files = {
        "probe": ("p.wav", b"PROBE-BYTES", "audio/wav"),
        "reference": ("r.wav", b"REF-BYTES", "audio/wav"),
    }
    assert client.post("/speaker/verify", files=files).status_code == 200
    assert client.post("/speaker/verify", files=files).status_code == 200

    assert len(seen) == 2
    probe_paths = [p for p, _ in seen]
    assert len({str(p) for p in probe_paths}) == 2, "probe temp file name must be unique per request"
    for p, ref in seen:
        assert p.name not in ("probe_p.wav", "p.wav"), "probe temp name must not be the predictable shared name"
        assert ref.name not in ("ref_r.wav", "r.wav"), "reference temp name must not be the predictable shared name"
        assert not p.exists() and not ref.exists(), "temp files must be deleted after the request"
    assert list(tmp_path.iterdir()) == [], "no probe/ref temp files may be left behind in STORAGE_DIR"


def test_speaker_verify_cleans_up_on_service_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "STORAGE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(rs.svc, "verify_files",
                        lambda probe_wav, reference_wav=None: (_ for _ in ()).throw(ValueError("no default reference")),
                        raising=True)
    r = client.post("/speaker/verify", files={"probe": ("p.wav", b"x", "audio/wav")})
    assert r.status_code == 400
    assert list(tmp_path.iterdir()) == [], "temp files must be deleted even when the service fails"


def test_audio_enhance_uses_unique_temp_file_and_cleans_up(client, monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "STORAGE_DIR", tmp_path, raising=False)
    seen = []

    def fake_enhance(in_path):
        p = Path(in_path)
        assert p.exists() and p.read_bytes() == b"UPLOAD-BYTES"
        seen.append(p)
        return tmp_path / "out_enhanced.wav"

    monkeypatch.setattr(ra.svc, "enhance_file", fake_enhance, raising=True)
    r = client.post("/audio/enhance", files={"file": ("a.wav", b"UPLOAD-BYTES", "audio/wav")})
    assert r.status_code == 200
    assert len(seen) == 1
    assert seen[0].name != "upload_a.wav", "upload temp name must not be the predictable shared name"
    assert not seen[0].exists(), "upload temp file must be deleted after the request"
    assert list(tmp_path.iterdir()) == []


def test_audio_transcribe_uses_unique_temp_file_and_cleans_up(client, monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "STORAGE_DIR", tmp_path, raising=False)
    seen = []

    def fake_transcribe(in_path, language="ru"):
        p = Path(in_path)
        assert p.exists() and p.read_bytes() == b"UPLOAD-BYTES"
        seen.append(p)
        return {"text": "len=3", "raw": {}}

    monkeypatch.setattr(ra.svc, "transcribe_file", fake_transcribe, raising=True)
    r = client.post("/audio/transcribe", files={"file": ("b.wav", b"UPLOAD-BYTES", "audio/wav")})
    assert r.status_code == 200
    assert r.json()["text"] == "len=3"
    assert len(seen) == 1
    assert seen[0].name != "upload_b.wav", "upload temp name must not be the predictable shared name"
    assert not seen[0].exists(), "upload temp file must be deleted after the request"
    assert list(tmp_path.iterdir()) == []


def test_audio_transcribe_cleans_up_on_value_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "STORAGE_DIR", tmp_path, raising=False)

    def fake_transcribe(in_path, language="ru"):
        raise ValueError("bad audio")

    monkeypatch.setattr(ra.svc, "transcribe_file", fake_transcribe, raising=True)
    r = client.post("/audio/transcribe", files={"file": ("b.wav", b"x", "audio/wav")})
    assert r.status_code == 400
    assert list(tmp_path.iterdir()) == [], "upload temp file must be deleted even when the service fails"


def test_tts_clone_temp_file_removed_after_call(client, monkeypatch):
    seen = []

    class _Meta:
        voice_id = "clonedvoice"

    def fake_clone(up_path, sample_rate):
        p = Path(up_path)
        assert p.exists() and p.read_bytes() == b"RIFF-CLONE"
        seen.append(p)
        return _Meta()

    monkeypatch.setattr(rt.store, "clone_from_upload", fake_clone, raising=True)
    r = client.post("/tts/clone", files={"file": ("x.wav", b"RIFF-CLONE", "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"voice_id": "clonedvoice"}
    assert len(seen) == 1
    assert not seen[0].exists(), "/tts/clone temp file must be removed after the call"


def test_tts_clone_temp_file_removed_on_error(client_soft, monkeypatch):
    seen = []

    def fake_clone(up_path, sample_rate):
        seen.append(Path(up_path))
        raise RuntimeError("clone backend down")

    monkeypatch.setattr(rt.store, "clone_from_upload", fake_clone, raising=True)
    r = client_soft.post("/tts/clone", files={"file": ("x.wav", b"RIFF-CLONE", "audio/wav")})
    assert r.status_code == 500
    assert seen and not seen[0].exists(), "/tts/clone temp file must be removed even when cloning fails"


# --------------------------------------------------------------------- F-025b
def _query_le_bound(param) -> "float | None":
    """Return the Query `le` bound; new fastapi/pydantic keep it in metadata."""
    import annotated_types
    q = param.default
    for c in getattr(q, "metadata", []) or []:
        if isinstance(c, annotated_types.Le):
            return c.le
    return getattr(q, "le", None)  # older fastapi/pydantic store it as attribute


def test_train_microphone_duration_has_upper_bound_in_signature():
    q = inspect.signature(rs.train_microphone).parameters["duration"]
    assert _query_le_bound(q) == 30, "duration must be capped with Query le=30"


@pytest.mark.parametrize("duration", [30, 10, 0.5])
def test_train_microphone_accepts_durations_within_cap(client, monkeypatch, duration):
    monkeypatch.setattr(rs.svc, "train_from_microphone",
                        lambda user_id, duration: {"user_id": user_id, "wav_path": "w.wav", "npy_path": "n.npy"},
                        raising=True)
    r = client.post("/speaker/train/microphone", params={"duration": duration})
    assert r.status_code == 200


def test_train_microphone_default_duration_still_allowed(client, monkeypatch):
    monkeypatch.setattr(rs.svc, "train_from_microphone",
                        lambda user_id, duration: {"user_id": user_id, "wav_path": "w.wav", "npy_path": "n.npy"},
                        raising=True)
    r = client.post("/speaker/train/microphone")
    assert r.status_code == 200


def test_train_microphone_rejects_duration_above_cap(client_soft, monkeypatch):
    monkeypatch.setattr(rs.svc, "train_from_microphone",
                        _bomb("service must not be reached with a duration above the cap"), raising=True)
    r = client_soft.post("/speaker/train/microphone", params={"duration": 31})
    assert r.status_code == 422, "duration above the cap must fail FastAPI validation (422)"
    r2 = client_soft.post("/speaker/train/microphone", params={"duration": 3600})
    assert r2.status_code == 422, "hours-long recording requests must be rejected outright"

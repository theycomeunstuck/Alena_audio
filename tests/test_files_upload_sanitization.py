# Tests for /files/upload filename sanitization (audit F-001).
# routes_files has no module-level STORAGE_DIR binding: the handlers read
# app.settings.STORAGE_DIR on every call, so patching the settings attribute
# is what redirects the tests away from the real storage tree.
import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.STORAGE_DIR", tmp_path, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "..\\escape.txt", "sub/dir/x.txt", "sub\\dir\\x.txt", ".."],
)
def test_upload_rejects_traversal_names(storage, name):
    r = client.post("/files/upload", files={"file": (name, io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 400, f"filename {name!r} must be rejected"


def _raw_upload(filename: bytes) -> "TestClient":
    # httpx normalizes client-side, so absolute Windows filenames must be sent
    # as a hand-built multipart body to reach the server unsanitized.
    body = (
        b"--BOUNDARY\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        b"x\r\n"
        b"--BOUNDARY--\r\n"
    )
    return client.post(
        "/files/upload",
        content=body,
        headers={"Content-Type": "multipart/form-data; boundary=BOUNDARY"},
    )


@pytest.mark.parametrize(
    "name",
    [b"..\\..\\evil.txt", b"/etc/passwd"],
)
def test_upload_rejects_absolute_and_escaped_names(storage, name):
    r = _raw_upload(name)
    assert r.status_code == 400, f"filename {name!r} must be rejected"


def test_containment_rejects_absolute_paths(storage):
    # The multipart layer normalizes some absolute names before the handler,
    # so the containment check is exercised directly. It must raise a 400
    # HTTPException, not any other error type.
    from app.api import routes_files

    for name in (r"C:\Windows\evil.txt", "../escape.txt"):
        with pytest.raises(HTTPException) as exc_info:
            routes_files._contained_path(name)
        assert exc_info.value.status_code == 400, (
            f"{name!r} must map to HTTP 400, got {exc_info.value.status_code}")


def test_upload_accepts_bare_name_and_roundtrips(storage):
    # Positive path with a payload that never needs the (broken) torchaudio
    # decoder: upload writes raw bytes, download serves them back unchanged.
    r = client.post("/files/upload", files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")})
    assert r.status_code == 200
    assert r.json() == {"filename": "hello.txt"}
    assert (storage / "hello.txt").read_bytes() == b"hello world"

    r2 = client.get("/files/download/hello.txt")
    assert r2.status_code == 200
    assert r2.content == b"hello world"


def test_upload_empty_filename_rejected(storage):
    r = _raw_upload(b"")
    assert r.status_code == 400

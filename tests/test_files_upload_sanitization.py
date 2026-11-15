# Tests for /files/upload filename sanitization (audit F-001).
# The storage-dir attributes are patched in both places the code has used,
# so the tests never touch the real storage tree.
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.STORAGE_DIR", tmp_path, raising=False)
    monkeypatch.setattr("app.api.routes_files.STORAGE_DIR", tmp_path, raising=False)
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
    # so the containment check is exercised directly.
    from app.api import routes_files

    with pytest.raises(Exception):  # HTTPException 400
        routes_files._contained_path(r"C:\Windows\evil.txt")
    with pytest.raises(Exception):
        routes_files._contained_path("../escape.txt")


def test_upload_empty_filename_rejected(storage):
    r = _raw_upload(b"")
    assert r.status_code == 400

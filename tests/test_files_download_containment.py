# Tests for /files/download containment (audit F-007).
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture()
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.settings.STORAGE_DIR", tmp_path, raising=False)
    monkeypatch.setattr("app.api.routes_files.STORAGE_DIR", tmp_path, raising=False)
    (tmp_path / "ok.txt").write_bytes(b"data")
    return tmp_path


def test_download_plain_name_serves_file(storage):
    r = client.get("/files/download/ok.txt")
    assert r.status_code == 200
    assert r.content == b"data"


def test_download_missing_file_404(storage):
    r = client.get("/files/download/nope.txt")
    assert r.status_code == 404


@pytest.mark.parametrize("name", ["..%5Cescape.txt", "..\\ok.txt", "..%5C..%5Cescape.txt"])
def test_download_rejects_traversal(storage, name):
    r = client.get(f"/files/download/{name}")
    assert r.status_code == 400, f"{name!r} must not be served"

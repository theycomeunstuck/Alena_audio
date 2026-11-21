# Audit F-063: GET /health was a static "ok" with no component readiness.
# /health stays a liveness probe (status "ok") and now reports components;
# /health/ready answers 503 when a hard requirement (writable storage) fails.
# Neither endpoint may build a model.
import pytest

import app.api.routes_health as health
import core.config as cfg


@pytest.fixture
def no_model_builds(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("health check must not build a model")
    for name in ("_build_asr_model", "_build_noise_model", "_build_speech_verification_model"):
        monkeypatch.setattr(cfg, name, forbidden)


def test_health_stays_ok_and_reports_components(client, no_model_builds):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    components = body["components"]
    assert set(components) >= {"storage_writable", "ffmpeg", "default_voice", "tts_model", "models_loaded"}
    assert isinstance(body["ready"], bool)
    assert set(components["models_loaded"]) == {"asr", "noise", "speaker_encoder"}
    assert all(isinstance(value, bool) for value in components["models_loaded"].values())


def test_ready_is_200_with_writable_storage(client, monkeypatch, no_model_builds):
    monkeypatch.setattr(health, "_storage_writable", lambda: True)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_ready_is_503_without_writable_storage(client, monkeypatch, no_model_builds):
    monkeypatch.setattr(health, "_storage_writable", lambda: False)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["components"]["storage_writable"] is False


def test_tts_feature_needs_ffmpeg_default_voice_and_model(client, monkeypatch, no_model_builds):
    monkeypatch.setattr(health, "_storage_writable", lambda: True)
    monkeypatch.setattr(health.shutil, "which", lambda name: None)
    body = client.get("/health").json()
    assert body["components"]["ffmpeg"] is False
    assert body["features"]["tts"] is False
    assert body["ready"] is True  # TTS is a feature, not a liveness/readiness gate

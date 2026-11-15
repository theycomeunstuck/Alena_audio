# Tests for TTS settings contract (audit F-005).
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_settings_define_tts_sample_rate():
    from app import settings

    assert isinstance(settings.TTS_SAMPLE_RATE, int)
    assert settings.TTS_SAMPLE_RATE > 0


def test_tts_sample_rate_env_override(monkeypatch):
    import app.settings as settings

    monkeypatch.setenv("TTS_SAMPLE_RATE", "44100")
    reloaded = importlib.reload(settings)
    try:
        assert reloaded.TTS_SAMPLE_RATE == 44100
    finally:
        monkeypatch.delenv("TTS_SAMPLE_RATE")
        importlib.reload(settings)


def test_tts_clone_passes_configured_sample_rate_to_store(monkeypatch, client):
    """The /tts/clone route must forward app.settings.TTS_SAMPLE_RATE into
    VoiceStore.clone_from_upload as the reference.wav resample rate.

    The old test only checked hasattr(settings, "TTS_SAMPLE_RATE") — it passed
    even if the route never read the setting. This drives the real route
    handler with a fake store and pins the actual data flow.
    """
    import app.api.routes_TTS as routes_tts
    from app import settings

    seen = {}

    class _FakeStore:
        def clone_from_upload(self, up_path, sample_rate=24000, language=None):
            seen["sample_rate"] = sample_rate
            seen["up_path"] = Path(up_path)
            # the route deletes the temp file in its finally block, so the
            # content can only be observed while the fake store is running
            seen["up_bytes"] = Path(up_path).read_bytes()
            return SimpleNamespace(voice_id="sr-check")

    monkeypatch.setattr(routes_tts, "store", _FakeStore())
    monkeypatch.setattr(settings, "TTS_SAMPLE_RATE", 12345)

    r = client.post("/tts/clone", files={"file": ("x.wav", b"RIFF-CLONE", "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json() == {"voice_id": "sr-check"}
    assert seen["sample_rate"] == 12345, (
        "clone_from_upload must receive the configured TTS_SAMPLE_RATE")
    assert seen["up_bytes"] == b"RIFF-CLONE", (
        "the uploaded content must reach the store through the per-request temp file")
    assert not seen["up_path"].exists(), "the clone temp file must be removed after the request"

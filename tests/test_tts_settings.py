# Tests for TTS settings contract (audit F-005).
import importlib

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


def test_routes_tts_clone_uses_defined_setting():
    # The clone route reads settings.TTS_SAMPLE_RATE; it must exist by import time.
    import app.api.routes_TTS as routes_tts  # noqa: F401
    from app import settings

    assert hasattr(settings, "TTS_SAMPLE_RATE")

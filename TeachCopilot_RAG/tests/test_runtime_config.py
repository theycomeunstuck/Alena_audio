import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_config_reloads(monkeypatch):
    """Audit F-080: importlib.reload keeps the test's env values in the module
    objects after monkeypatch restores os.environ, and the reload's
    load_dotenv() adds .env values to the process environment. Restore the
    environment, reload both modules from it, and restore it once more."""
    saved = dict(os.environ)
    yield
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)
    import pipeline.config as config
    import pipeline.prompt_builder as prompt_builder
    importlib.reload(config)
    importlib.reload(prompt_builder)
    os.environ.clear()
    os.environ.update(saved)


def test_cpu_debug_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setenv("TEACHCOPILOT_RUNTIME_PROFILE", "cpu_debug")

    import pipeline.config as config

    config = importlib.reload(config)
    assert config.RUNTIME_PROFILE == "cpu_debug"
    assert config.EMBEDDING_DEVICE == "cpu"


def test_chat_model_can_be_left_to_caller(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    monkeypatch.setenv("TEACHCOPILOT_CHAT_MODEL", "")

    import pipeline.config as config

    config = importlib.reload(config)
    assert config.CHAT_MODEL == ""


def test_adult_prompt_is_config_gated(monkeypatch):
    monkeypatch.setenv("TEACHCOPILOT_SPEAKER_MODE", "mapped_or_adult")

    import pipeline.config as config
    import pipeline.prompt_builder as prompt_builder

    importlib.reload(config)
    prompt_builder = importlib.reload(prompt_builder)

    prompt = prompt_builder.build_system_prompt(
        "child prompt",
        {"name": "Миша", "knowledge": []},
        [],
        speaker_type="adult",
    )

    assert "режиме педагога" in prompt
    assert "Ученик: Миша" in prompt


def test_reloaded_config_does_not_leak_into_later_tests():
    """Audit F-080: the reloads above used to leave pipeline.config and
    pipeline.prompt_builder with the test's env values for the rest of the
    session (test order dependent). The autouse fixture reloads them with the
    restored environment."""
    import os

    import pipeline.config as config
    import pipeline.prompt_builder as prompt_builder

    expected = os.getenv("TEACHCOPILOT_SPEAKER_MODE", "child_only").strip().lower()
    assert config.SPEAKER_MODE == expected
    assert prompt_builder.SPEAKER_MODE == expected

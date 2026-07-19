import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

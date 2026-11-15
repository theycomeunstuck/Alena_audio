# Tests for the TTS engine runtime resolution (audit F-006).
import ast
from pathlib import Path

import torch


def test_runtime_resolution_cpu():
    from core.TTS import tts_engine

    rt = tts_engine._tts_runtime("cpu")
    assert rt["device_map"] == "cpu"
    assert rt["dtype"] == torch.float32
    assert rt["attn_implementation"] == "sdpa"


def test_runtime_resolution_cuda():
    from core.TTS import tts_engine

    rt = tts_engine._tts_runtime("cuda:0")
    assert rt["device_map"] == "cuda:0"
    assert rt["dtype"] == torch.bfloat16


def test_attn_implementation_env_override(monkeypatch):
    from core.TTS import tts_engine

    monkeypatch.setenv("TTS_ATTN_IMPLEMENTATION", "flash_attention_2")
    assert tts_engine._tts_runtime("cuda:0")["attn_implementation"] == "flash_attention_2"


def test_settings_device_is_default():
    from core.TTS import tts_engine

    rt = tts_engine._tts_runtime()
    assert rt["device_map"] is not None and rt["device_map"] != ""


def test_no_hardcoded_cuda_or_flash_attention_literals():
    src = Path(__file__).resolve().parents[1] / "core" / "TTS" / "tts_engine.py"
    text = src.read_text(encoding="utf-8")
    assert '"cuda:0"' not in text, "device must come from settings, not a literal"
    assert '"flash_attention_2"' not in text, "attention must come from settings/env, not a literal"
    tree = ast.parse(text)
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "_tts_runtime" for n in tree.body
    ), "runtime resolution helper must exist"

# tests/test_deep_TTS.py
# Audit F-067: the previous end-to-end test targeted the removed F5-TTS CLI and
# was commented out. This deep test drives the current Qwen3-TTS engine for
# real. It is deselected by default (pytest.ini: -m "not deep") and skips
# unless the real runtime, the local model and ffmpeg are all present.
import asyncio
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "pretrained_models" / "Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE = ROOT / "tests" / "samples" / "ru_sample.wav"


@pytest.mark.deep
def test_tts_deep_real_qwen_model(monkeypatch):
    qwen_tts = pytest.importorskip("qwen_tts")
    if "stub" in (qwen_tts.__doc__ or "").lower():
        pytest.skip("qwen_tts is a test stub – the real Qwen3-TTS runtime is not installed")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"Qwen3-TTS model directory is missing: {MODEL_DIR.relative_to(ROOT)}")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not on PATH")

    monkeypatch.chdir(ROOT)  # the engine loads the model by a project-relative path
    from core.TTS.tts_engine import get_tts_engine

    audio = asyncio.run(get_tts_engine().synth(
        text="Привет! Это проверка синтеза речи.", ref_audio=REFERENCE,
        ref_text="", vid="_default", out_format="wav"))
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000

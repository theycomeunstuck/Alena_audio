# The app imports qwen_tts at core/TTS/tts_engine.py:13; both requirement
# manifests must name it (audit F-002). The files are UTF-16 encoded.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements_text(name: str) -> str:
    data = (ROOT / name).read_bytes()
    encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return data.decode(encoding)


def test_gpu_requirements_name_qwen_tts():
    assert "qwen" in _requirements_text("requirements_GPU_noTTS_api.txt").lower()


def test_cpu_requirements_name_qwen_tts():
    assert "qwen" in _requirements_text("requirements_cpu_noTTS_api.txt").lower()

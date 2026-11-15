# The app imports qwen_tts at core/TTS/tts_engine.py:13; both requirement
# manifests must name it (audit F-002). The files are UTF-16 encoded.
# The torch pins must stay identical across both manifests (audit F-020
# follow-up) so the GPU and CPU installs cannot drift apart silently.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements_text(name: str) -> str:
    data = (ROOT / name).read_bytes()
    encoding = "utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return data.decode(encoding)


def _pinned_version(text: str, package: str):
    """Return the version from the package's `package==<version>` line, or None."""
    prefix = package.lower() + "=="
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix):]
    return None


def test_gpu_requirements_name_qwen_tts():
    assert "qwen" in _requirements_text("requirements_GPU_noTTS_api.txt").lower()


def test_cpu_requirements_name_qwen_tts():
    assert "qwen" in _requirements_text("requirements_cpu_noTTS_api.txt").lower()


def test_torch_pin_matches_between_gpu_and_cpu_requirements():
    gpu = _pinned_version(_requirements_text("requirements_GPU_noTTS_api.txt"), "torch")
    cpu = _pinned_version(_requirements_text("requirements_cpu_noTTS_api.txt"), "torch")
    assert gpu is not None, "requirements_GPU_noTTS_api.txt must contain a torch== pin"
    assert cpu is not None, "requirements_cpu_noTTS_api.txt must contain a torch== pin"
    assert gpu == cpu, (
        "torch pin drifted between manifests: "
        f"requirements_GPU_noTTS_api.txt pins {gpu!r} but "
        f"requirements_cpu_noTTS_api.txt pins {cpu!r}"
    )

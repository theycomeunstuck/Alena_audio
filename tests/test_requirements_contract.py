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


def test_torchaudio_pin_matches_between_gpu_and_cpu_requirements():
    """torchaudio is imported by the app.main chain (audio_utils, audio_service,
    speaker_service, audio_enhancement) - the pin must exist and match across
    both manifests (verification round 2: it was missing while four modules
    import it at module level)."""
    gpu = _pinned_version(_requirements_text("requirements_GPU_noTTS_api.txt"), "torchaudio")
    cpu = _pinned_version(_requirements_text("requirements_cpu_noTTS_api.txt"), "torchaudio")
    assert gpu is not None, "requirements_GPU_noTTS_api.txt must contain a torchaudio== pin"
    assert cpu is not None, "requirements_cpu_noTTS_api.txt must contain a torchaudio== pin"
    assert gpu == cpu, (
        "torchaudio pin drifted between manifests: "
        f"requirements_GPU_noTTS_api.txt pins {gpu!r} but "
        f"requirements_cpu_noTTS_api.txt pins {cpu!r}"
    )


def _version_tuple(version: str):
    return tuple(int(part) for part in version.split("+", 1)[0].split(".")[:2])


def test_speechbrain_pin_imports_with_the_pinned_torchaudio():
    """speechbrain 1.0.x calls torchaudio.list_audio_backends() at import time;
    torchaudio 2.9 removed it, so `import app.main` (speaker_service imports
    speechbrain) failed with the round-2 torchaudio 2.11.1 pin. speechbrain
    1.1 dropped that call. Both manifests must pin a compatible pair."""
    for name in ("requirements_GPU_noTTS_api.txt", "requirements_cpu_noTTS_api.txt"):
        text = _requirements_text(name)
        speechbrain = _pinned_version(text, "speechbrain")
        torchaudio = _pinned_version(text, "torchaudio")
        assert speechbrain is not None, f"{name} must contain a speechbrain== pin"
        assert torchaudio is not None, f"{name} must contain a torchaudio== pin"
        if _version_tuple(torchaudio) >= (2, 9):
            assert _version_tuple(speechbrain) >= (1, 1), (
                f"{name}: speechbrain {speechbrain} imports torchaudio.list_audio_backends, "
                f"which torchaudio {torchaudio} no longer provides"
            )

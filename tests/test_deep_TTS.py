#tests/test_deep_TTS.py
import io
import os
import shutil
import wave
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

def _which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)

def _write_silence_wav(path: Path, sr: int = 24000, dur_ms: int = 300):
    frames = int(sr * dur_ms / 1000)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames)

@pytest.mark.deep
def test_tts_deep_real_model(tmp_path, monkeypatch):
    """
    Глубокий e2e-тест: вызываем НАСТОЯЩИЙ движок.
    Скип, если окружение не готово.
    """
    # --- проверки окружения ---
    ckpt = os.environ.get("F5TTS_CKPT_PATH", "")
    if not ckpt or not Path(ckpt).exists():
        pytest.skip("F5TTS_CKPT_PATH не задан или файл отсутствует — пропускаем deepTTS.")
    if not _which("f5-tts_infer-cli"):
        pytest.skip("CLI f5-tts_infer-cli не найден в PATH — пропускаем deepTTS.")
    if not _which("ffmpeg"):
        pytest.skip("FFmpeg не найден — пропускаем deepTTS.")

    # --- перенаправляем VOICES_DIR в tmp ---
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOICES_DIR", str(voices_dir))
    # лучше на CPU для предсказуемости: (или оставьте 'auto')
    monkeypatch.setenv("DEVICE", os.environ.get("DEVICE", "cpu"))

    # --- создаём базовый _default/reference.wav ---
    default_dir = voices_dir / "_default"
    default_dir.mkdir(parents=True, exist_ok=True)
    ref = default_dir / "reference.wav"
    _write_silence_wav(ref, sr=int(os.environ.get("TTS_SAMPLE_RATE", "24000")), dur_ms=400)

    # --- импортируем приложение ПОСЛЕ подготовки окружения ---
    app_module = importlib.import_module("app.main")
    app = getattr(app_module, "app")
    client = TestClient(app)

    # --- 1) clone: отправляем валидный WAV из памяти ---
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * int(24000 * 0.2))
    buf.seek(0)
    r = client.post("/voice/clone", files={"file": ("speaker.wav", buf, "audio/wav")})
    assert r.status_code == 200, r.text
    voice_id = r.json().get("voice_id")
    assert voice_id and len(voice_id) > 10

    # --- 2) tts: базовый голос (без voice_id), короткий текст ---
    r = client.post("/tts", json={"text": "Привет!"})
    assert r.status_code == 200, r.text
    ctype = r.headers.get("content-type", "")
    assert ctype.startswith("audio/wav")
    body = r.content
    assert body[:4] == b"RIFF"  # WAV-хедер

    # --- 3) tts: с указанным voice_id и другим форматом ---
    r = client.post("/tts", json={"text": "Проверка клонированного голоса.", "voice_id": voice_id, "format": "mp3"})
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("audio/mpeg")
    assert len(r.content) > 1000  # должно быть не пусто

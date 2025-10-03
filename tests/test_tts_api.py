# tests/test_tts_api.py
import os
import io
import wave
import importlib
from pathlib import Path

from fastapi.testclient import TestClient

# эти переменные должны быть выставлены до импорта app.main
# чтобы routes_TTS успел корректно инициализировать store/engine.
def _prepare_env_and_default(tmp_path: Path, sample_rate: int = 24000) -> Path:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Перенаправим роутер на временное хранилище
    os.environ["VOICES_DIR"] = str(voices_dir)
    # Чтобы TtsEngine.__init__ не падал из-за отсутствия ckpt
    os.environ.setdefault("F5TTS_CKPT_PATH", "")

    # Создаём базовый голос _default/reference.wav (валидный WAV с тишиной)
    default_dir = voices_dir / "_default"
    default_dir.mkdir(parents=True, exist_ok=True)

    ref = default_dir / "reference.wav"
    _write_silence_wav(ref, sr=sample_rate, dur_ms=200)
    return voices_dir


def _write_silence_wav(path: Path, sr: int = 24000, dur_ms: int = 200) -> None:
    frames = int(sr * dur_ms / 1000)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames)


def _make_client_and_patch_engine():
    # Импортируем приложение после подготовки окружения
    app_module = importlib.import_module("app.main")
    app = getattr(app_module, "app")

    # Подменим движок синтеза в роутере на фейковый
    r = importlib.import_module("app.api.routes_TTS")
    class _FakeEngine:
        async def synth(self, text, ref, fmt):
            # Возвращаем минимальные валидные «байты аудио» для ответа
            # (для проверки протокола достаточно заголовка RIFF + немного данных)
            return b"RIFF" + b"\x00" * 256

    r.engine = _FakeEngine()
    # Возвращаем TestClient
    return TestClient(app)


def test_clone_and_tts_default(tmp_path):
    # Подготовить окружение и базовый голос
    _prepare_env_and_default(tmp_path, sample_rate=24000)

    client = _make_client_and_patch_engine()

    # 1) Клонирование — отправляем валидный WAV
    wav_buf = io.BytesIO()
    _write_silence_wav(Path("dummy.wav"), sr=24000, dur_ms=100)  # только для формы
    # Пересоздадим заново, но прямо в память:
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * int(24000 * 0.1))
    wav_buf.seek(0)

    r = client.post("/voice/clone", files={"file": ("a.wav", wav_buf, "audio/wav")})
    assert r.status_code == 200, r.text
    assert "voice_id" in r.json()
    voice_id = r.json()["voice_id"]
    assert voice_id and len(voice_id) > 10

    # 2) TTS с базовым голосом (без voice_id)
    r = client.post("/tts", json={"text": "Привет!"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")

    # 3) TTS с указанным voice_id и форматом mp3
    r = client.post("/tts", json={"text": "Ещё раз.", "voice_id": voice_id, "format": "mp3"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_errors(tmp_path):
    # Подготовить окружение и базовый голос
    _prepare_env_and_default(tmp_path, sample_rate=24000)

    client = _make_client_and_patch_engine()

    # пустой текст -> 400
    r = client.post("/tts", json={"text": ""})
    assert r.status_code == 400

    # несуществующий voice_id -> 404
    r = client.post("/tts", json={"text": "ok", "voice_id": "no-such-id"})
    assert r.status_code == 404

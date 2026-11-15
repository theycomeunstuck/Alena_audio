# tests/test_tts_api.py
import importlib
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from core.TTS import VoiceStore
from core.TTS.voices import VoiceMeta


def _write_silence_wav_bytes(sr: int = 24000, dur_ms: int = 200) -> bytes:
    """Минимальный валидный WAV (тишина, 16-bit PCM mono) в памяти."""
    import io
    frames = int(sr * dur_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _make_tts_client(monkeypatch, tmp_path: Path):
    """Импортируем приложение и подменяем ровно то, что читают роуты TTS.

    POST /tts при каждом запросе вызывает get_tts_engine() (module-level биндинг
    в app.api.routes_TTS) и работает с module-level store, который при импорте
    собран из app.settings.VOICES_DIR. Оба биндинга патчим на tmp-хранилище,
    чтобы тест не трогал настоящий storage/.
    """
    app_module = importlib.import_module("app.main")
    app = getattr(app_module, "app")
    r = importlib.import_module("app.api.routes_TTS")

    store = VoiceStore(tmp_path / "voices_TTS")
    (store.root / "_default").mkdir(parents=True, exist_ok=True)
    (store.root / "_default" / "reference.wav").write_bytes(_write_silence_wav_bytes())
    monkeypatch.setattr(r, "store", store)

    class _FakeEngine:
        # Сигнатура совпадает с реальным TtsEngine.synth (routes_TTS вызывает
        # synth(text=..., ref_audio=..., out_format=..., ref_text=..., vid=...)).
        def __init__(self):
            self.calls = []

        async def synth(self, text, ref_audio, ref_text, vid, out_format="wav"):
            self.calls.append({"text": text, "ref_audio": ref_audio,
                               "ref_text": ref_text, "vid": vid, "out_format": out_format})
            # Возвращаем минимальные валидные «байты аудио» для ответа
            # (для проверки протокола достаточно заголовка RIFF + немного данных)
            return b"RIFF" + b"\x00" * 256

    engine = _FakeEngine()
    monkeypatch.setattr(r, "get_tts_engine", lambda: engine)
    return TestClient(app), engine


def test_tts_default_voice_positive_path(monkeypatch, tmp_path):
    client, engine = _make_tts_client(monkeypatch, tmp_path)

    r = client.post("/tts", json={"text": "Привет!"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    # фейковый движок действительно использован роутом, а не построен заново
    assert len(engine.calls) == 1
    assert engine.calls[0]["text"] == "Привет!"
    assert engine.calls[0]["vid"] == "_default"


def test_tts_explicit_voice_id_media_type(monkeypatch, tmp_path):
    client, engine = _make_tts_client(monkeypatch, tmp_path)
    # Засеиваем второй голос напрямую в tmp-хранилище (без тяжёлого /tts/clone).
    import app.api.routes_TTS as routes_tts
    store = routes_tts.store
    (store.root / "u1").mkdir(parents=True, exist_ok=True)
    (store.root / "u1" / "reference.wav").write_bytes(_write_silence_wav_bytes())
    store.write_meta(VoiceMeta(voice_id="u1", sr=24000, orig_file="a.wav", ref_text=""))

    r = client.post("/tts", json={"text": "Ещё раз.", "voice_id": "u1", "format": "mp3"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/mpeg")
    assert engine.calls[0]["vid"] == "u1"
    assert engine.calls[0]["out_format"] == "mp3"


def test_errors(monkeypatch, tmp_path):
    client, engine = _make_tts_client(monkeypatch, tmp_path)

    # пустой текст -> 400
    r = client.post("/tts", json={"text": ""})
    assert r.status_code == 400
    assert engine.calls == []  # до синтеза дело не дошло

    # несуществующий voice_id -> 404
    r = client.post("/tts", json={"text": "ok", "voice_id": "no-such-id"})
    assert r.status_code == 404

    # невалидный voice_id (обход каталога) -> 400 через validate_resource_id
    r = client.post("/tts", json={"text": "ok", "voice_id": "../escape"})
    assert r.status_code == 400
    assert engine.calls == []

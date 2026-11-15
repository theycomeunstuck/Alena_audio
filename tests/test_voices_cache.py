# tests/test_voices_cache.py
import json
import wave
import numpy as np
from pathlib import Path
from core.TTS.voices import VoiceStore
from core.ASR.transcriber import AsrTranscriber

def _make_wav(tmp_path: Path, seconds: float = 0.3, sr: int = 24000) -> Path:
    # Записываем WAV стандартным модулем wave: torchaudio в этом окружении не
    # работает (torchcodec не может загрузить свою DLL), а pydub, которым
    # clone_from_upload читает файл, такой WAV читает без проблем.
    t = np.arange(int(sr * seconds)) / sr
    x = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    x16 = (x * 32767.0).astype("<i2")
    p = tmp_path / "in.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())
    return p

def test_clone_transcription_cache(tmp_path: Path, monkeypatch):
    # перенаcтраиваем хранилище голосов на tmp
    store = VoiceStore(tmp_path / "voices_TTS")
    wav = _make_wav(tmp_path)

    calls = {"n": 0}

    def fake_init(self, *a, **kw):
        # сохраняем контракт конструктора (language/max_chars), но без загрузки весов Whisper
        self.language = kw.get("language")
        self.max_chars = None
        self.model = None

    def fake_transcribe(self, path):
        calls["n"] += 1
        return "привет мир"

    monkeypatch.setattr(AsrTranscriber, "__init__", fake_init)
    monkeypatch.setattr(AsrTranscriber, "transcribe", fake_transcribe)

    # первый клон → вызов транскрипции
    m1 = store.clone_from_upload(wav, sample_rate=24000, language="ru")
    # Ударения больше не расставляются: акцентор удалён намеренно (см. комментарии
    # в core/TTS/voices.py) — meta хранит сырую транскрипцию, а '+' срезает сам
    # движок синтеза перед генерацией.
    assert m1.ref_text == "привет мир"
    assert calls["n"] == 1

    # второй клон того же файла → берём из кэша (в VOICES_DIR/whisper_transcriptions.json)
    m2 = store.clone_from_upload(wav, sample_rate=24000, language="ru")
    assert m2.ref_text == "привет мир"
    assert calls["n"] == 1  # не увеличился — использовали кэш

    # проверим, что файл кэша существует и валиден
    cache_file = store._cache_path()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "by_sha1" in data and isinstance(data["by_sha1"], dict)

# tests/test_voices_cache.py
import io, uuid
import json
import numpy as np
import torch, torchaudio
from pathlib import Path
from core.TTS.voices import VoiceStore
from core.ASR.transcriber import AsrTranscriber

def _make_wav(tmp_path: Path, seconds: float = 0.3, sr: int = 24000) -> Path:
    t = torch.arange(int(sr*seconds)) / sr
    x = 0.1 * torch.sin(2*np.pi*440.0 * t)
    p = tmp_path / "in.wav"
    torchaudio.save(str(p), x.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16, format="wav")
    return p

def test_clone_transcription_cache(tmp_path: Path, monkeypatch):
    # перенаcтраиваем хранилище голосов на tmp
    store = VoiceStore(tmp_path / "voices_TTS")
    wav = _make_wav(tmp_path)

    calls = {"n": 0}
    orig_init = AsrTranscriber.__init__
    orig_transcribe = AsrTranscriber.transcribe

    def fake_init(self, *a, **kw):
        orig_init(self, *a, **kw)

    def fake_transcribe(self, path):
        calls["n"] += 1
        return "привет мир"

    monkeypatch.setattr(AsrTranscriber, "transcribe", fake_transcribe)

    # первый клон → вызов транскрипции
    m1 = store.clone_from_upload(wav, sample_rate=24000, language="ru")
    assert m1.ref_text == "прив+ет м+ир" # был добавлен стрессер. здесь никак не чекается работа расстановщика ударений
    assert calls["n"] == 1

    # второй клон того же файла → берём из кэша (в VOICES_DIR/whisper_transcriptions.json)
    m2 = store.clone_from_upload(wav, sample_rate=24000, language="ru")
    assert m2.ref_text == "прив+ет м+ир"
    assert calls["n"] == 1  # не увеличился — использовали кэш

    # проверим, что файл кэша существует и валиден
    cache_file = store._cache_path()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "by_sha1" in data and isinstance(data["by_sha1"], dict)

#tests/test_transcription_meta.py

import json
from pathlib import Path

import pytest

from core.TTS.voices import VoiceStore


class DummyAudio:
    def set_channels(self, n):
        return self
    def set_frame_rate(self, sr):
        return self
    def export(self, path, format):
        # запишем что-то, чтобы sha1 посчитался и файл существовал
        Path(path).write_bytes(b"WAVDATA")


class DummyAudioSegment:
    @classmethod
    def from_file(cls, up_path):
        return DummyAudio()


class DummyTranscriber:
    def __init__(self, language=None):
        pass
    def transcribe(self, wav_path):
        return "Тестовая фраза. И, да, — это точно тестовая фраза."


@pytest.fixture(autouse=True)
def patch_audio_and_asr(monkeypatch):
    # Подменяем внутри core.TTS.voices то, что модуль уже импортировал
    import core.TTS.voices as voices_mod
    monkeypatch.setattr(voices_mod, "AudioSegment", DummyAudioSegment)
    monkeypatch.setattr(voices_mod, "AsrTranscriber", DummyTranscriber)
    yield


def test_clone_creates_meta_json_with_expected_format_and_content(tmp_path: Path):
    # Arrange: входной "аудио"-файл
    up = tmp_path / "speaker_ru.mp3"
    up.write_bytes(b"\x00\x01FAKE-MP3")

    store = VoiceStore(tmp_path)

    # Act: клонируем голос
    meta = store.clone_from_upload(up_path=up, sample_rate=24000)

    # Assert: reference.wav создан
    ref = tmp_path / meta.voice_id / "reference.wav"
    assert ref.exists(), "reference.wav должен быть создан"

    # Assert: meta.json создан
    meta_path = tmp_path / meta.voice_id / "meta.json"
    assert meta_path.exists(), "meta.json должен быть создан"

    # Проверка содержимого и форматирования
    raw = meta_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Данные
    assert data["voice_id"] == meta.voice_id
    assert data["sr"] == 24000
    assert data["orig_file"] == "speaker_ru.mp3"
    assert data["ref_text"] == "Здравствуйте! Это тестовая фраза."

    # Форматирование (dump с indent=2): проверим, что есть отступы в 2 пробела
    # (жёстко к точному порядку полей не привязываемся)
    lines_with_indent2 = [line for line in raw.splitlines()[1:] if line.startswith("  ")]
    assert lines_with_indent2, "Ожидались строки с отступом в два пробела (indent=2)"


def test_read_meta_reads_back_same_content(tmp_path: Path):
    # Arrange: создаём структуру вручную
    store = VoiceStore(tmp_path)
    vid = "abc123"
    vdir = tmp_path / vid
    vdir.mkdir(parents=True, exist_ok=True)

    meta_path = vdir / "meta.json"
    meta_json = {
        "voice_id": vid,
        "sr": 24000,
        "orig_file": "speaker_ru.mp3",
        "ref_text": "Тестовая фраза. И, да, — это точно тестовая фраза."
    }
    meta_path.write_text(json.dumps(meta_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Act
    meta = store.read_meta(vid)

    # Assert
    assert meta.voice_id == vid
    assert meta.sr == 24000
    assert meta.orig_file == "speaker_ru.mp3"
    assert meta.ref_text == "Тестовая фраза. И, да, — это точно тестовая фраза."

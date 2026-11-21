# Audit F-064: speaker embeddings are written atomically, and concurrent
# VoiceStore transcription-cache updates do not lose each other's entries.
import json
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

import app.services.speaker_service as speaker_service
from core.TTS.voices import VoiceStore


def test_failed_embedding_write_keeps_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "u1.npy"
    np.save(target, np.array([1.0, 0.0], dtype=np.float32))

    real_save = np.save

    def torn_save(file, array, *args, **kwargs):
        file.write(b"\x93NUMPY partial")  # a crash halfway through the write
        raise OSError("disk full")

    monkeypatch.setattr(speaker_service.np, "save", torn_save)
    with pytest.raises(OSError):
        speaker_service.atomic_save_npy(target, np.array([0.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(speaker_service.np, "save", real_save)

    assert np.load(target).tolist() == [1.0, 0.0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u1.npy"]


def test_embedding_write_replaces_the_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "u1.npy"
    speaker_service.atomic_save_npy(target, np.array([0.6, 0.8], dtype=np.float32))
    assert np.load(target).tolist() == pytest.approx([0.6, 0.8])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u1.npy"]


def _wav(path: Path, freq: float) -> Path:
    sr = 16000
    t = np.arange(int(sr * 0.1)) / sr
    data = (0.1 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


def test_concurrent_transcriptions_keep_both_cache_entries(tmp_path, monkeypatch):
    store = VoiceStore(tmp_path / "voices")
    slow, fast = _wav(tmp_path / "slow.wav", 220.0), _wav(tmp_path / "fast.wav", 440.0)
    slow_started, fast_done = threading.Event(), threading.Event()

    class Transcriber:
        def transcribe(self, path):
            if Path(path).name == "slow.wav":
                slow_started.set()
                assert fast_done.wait(5)  # the other update lands while this one is in flight
                return "slow text"
            return "fast text"

    monkeypatch.setattr(store, "_get_transcriber", lambda language=None: Transcriber())

    worker = threading.Thread(target=store._transcribe_with_local_cache, args=(slow,))
    worker.start()
    assert slow_started.wait(5)
    store._transcribe_with_local_cache(fast)
    fast_done.set()
    worker.join(5)

    cache = json.loads((tmp_path / "voices" / "whisper_transcriptions.json").read_text(encoding="utf-8"))
    assert sorted(cache["by_sha1"].values()) == ["fast text", "slow text"]
    assert not [p.name for p in (tmp_path / "voices").iterdir() if p.name.endswith(".tmp")]

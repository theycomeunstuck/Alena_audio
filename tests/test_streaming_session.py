# tests/test_streaming_session.py
import time

import numpy as np

from app.services.audio_service import StreamingASRSession


def test_session_accumulates_and_trims():
    sr = 16000
    sess = StreamingASRSession(sample_rate=sr, language="ru", window_sec=0.1, emit_sec=0.05)

    # 60 ms звука -> не должно превысить окно
    chunk1 = np.zeros(int(0.06 * sr), dtype=np.float32)
    bytes1 = (chunk1 * 32767.0).astype("<i2").tobytes()
    sess.add_pcm16(bytes1)
    assert 0 < sess.get_audio().shape[0] <= int(0.1 * sr)

    # Ещё 100 ms -> общий размер должен обрезаться до окна ~ 0.1s
    chunk2 = np.zeros(int(0.1 * sr), dtype=np.float32)
    sess.add_pcm16((chunk2 * 32767.0).astype("<i2").tobytes())
    assert int(0.09 * sr) <= sess.get_audio().shape[0] <= int(0.11 * sr)


def test_session_should_emit_and_reset():
    sr = 8000
    sess = StreamingASRSession(sample_rate=sr, language="ru", window_sec=0.2, emit_sec=0.05)

    # добавим 40 ms — ещё рано
    chunk = np.zeros(int(0.04 * sr), dtype=np.float32)
    sess.add_pcm16((chunk * 32767.0).astype("<i2").tobytes())
    assert not sess.should_emit()

    # добавим ещё 20 ms — пора
    chunk2 = np.zeros(int(0.02 * sr), dtype=np.float32)
    sess.add_pcm16((chunk2 * 32767.0).astype("<i2").tobytes())
    assert sess.should_emit()

    sess.reset_emit_counter()
    assert not sess.should_emit()


# ---------------------------------------------------------------------------
# The REAL session behind /ws/asr: routes_ws drives
# app.services.streaming_asr.StreamingASRSession, not the legacy class above.
# These tests pin its buffer/inactivity/emit contracts with the model seam
# faked, so no Whisper weights are loaded.
# ---------------------------------------------------------------------------

def _fake_asr(calls):
    class _FakeASR:
        def transcribe(self, audio, language=None, **kw):
            calls["n"] += 1
            return {"text": f"len={int(len(audio))}", "segments": [], "language": language or "ru"}

    return _FakeASR()


def test_production_session_trims_window_and_gates_inactivity(monkeypatch):
    import app.services.streaming_asr as sas

    monkeypatch.setattr(sas, "asr_model", _fake_asr({"n": 0}))
    sess = sas.StreamingASRSession(sample_rate=16000, language="ru",
                                   window_sec=0.5, emit_sec=0.05, inactivity_sec=2.0)

    # 1.0 s of PCM16 (16000 samples) must be trimmed to the 0.5 s window
    sess.ingest_pcm16_chunk(np.zeros(16000, dtype="<i2").tobytes())
    assert sess.buffer.size == 8000

    # fresh audio -> not timed out; stale audio -> timed out
    assert sess.inactive_timed_out() is False
    sess._last_audio_t -= 3.0
    assert sess.inactive_timed_out() is True

    # no inactivity configured -> never times out
    sess2 = sas.StreamingASRSession(sample_rate=16000, inactivity_sec=None)
    sess2.ingest_pcm16_chunk(np.zeros(160, dtype="<i2").tobytes())
    sess2._last_audio_t -= 3600.0
    assert sess2.inactive_timed_out() is False


def test_production_session_emit_gate_and_empty_buffer(monkeypatch):
    import app.services.streaming_asr as sas

    calls = {"n": 0}
    monkeypatch.setattr(sas, "asr_model", _fake_asr(calls))
    sess = sas.StreamingASRSession(sample_rate=16000, language="ru", window_sec=8.0, emit_sec=2.0)

    sess.ingest_pcm16_chunk(np.zeros(1600, dtype="<i2").tobytes())

    # just emitted -> gated by emit_sec
    sess._last_emit_t = time.time()
    assert sess.maybe_emit() is None
    assert calls["n"] == 0

    # emit_sec elapsed -> emits the current buffer through the model seam
    sess._last_emit_t = time.time() - 10.0
    out = sess.maybe_emit()
    assert out is not None
    text, raw = out
    assert text == "len=1600"
    assert raw["language"] == "ru"
    assert calls["n"] == 1

    # empty buffer -> ("", raw) WITHOUT touching the model (the background
    # emitter ticks several times per second and must not call Whisper)
    sess.reset()
    sess._last_emit_t = time.time() - 10.0
    empty_text, empty_raw = sess.maybe_emit()
    assert empty_text == ""
    assert calls["n"] == 1, "empty-buffer transcription must not call the model"

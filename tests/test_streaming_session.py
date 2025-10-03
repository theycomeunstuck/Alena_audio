# tests/test_streaming_session.py
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

import numpy as np
import pytest

import app.services.streaming_verify as sv


@pytest.fixture()
def patched_resampler(monkeypatch):
    # Быстрый конвертер PCM16->float32 16кГц без реального ресемплинга
    def ensure_float_mono_16k_from_pcm16(pcm_chunk: bytes, src_sr: int, channels: int):
        x = np.frombuffer(pcm_chunk, dtype="<i2").astype(np.float32) / 32767.0
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        return x
    monkeypatch.setattr(sv, "ensure_float_mono_16k_from_pcm16", ensure_float_mono_16k_from_pcm16, raising=True)
    return ensure_float_mono_16k_from_pcm16


@pytest.fixture()
def patched_matcher(monkeypatch):
    class DummyMatcher:
        def match_probe_array(self, audio, top_k=5):
            # стабильный список совпадений
            return [{"user_id": f"u{i}", "score": 0.8 - 0.01*i, "ref_path": f"/ref/{i}"} for i in range(top_k)]
    monkeypatch.setattr(sv, "get_global_matcher", lambda: DummyMatcher(), raising=True)
    return DummyMatcher()


def test_buffer_trim_and_min_duration_gate(patched_resampler, patched_matcher):
    sess = sv.StreamingVerifySession(sample_rate=16000, channels_hint=1, inactivity_sec=None)

    # Кусок на 10с -> в буфере останется 8с
    ten_sec = (np.zeros(16000 * 10, dtype="<i2")).tobytes()
    sess.ingest_pcm16_chunk(ten_sec)
    assert sess.buffer.size == 16000 * 8

    # До 300 мс — матчинга нет
    sess.reset()
    short = (np.zeros(int(0.25 * 16000), dtype="<i2")).tobytes()
    sess.ingest_pcm16_chunk(short)
    assert sess.current_best(top_k=3) == []
    out = sess.current_best_binary(threshold=0.5, top_k=3)
    assert set(out.keys()) == {"decision", "threshold", "best", "matches"}
    assert out["decision"] is False and out["matches"] == []

    # 350 мс — уже считаем
    extra = (np.zeros(int(0.10 * 16000), dtype="<i2")).tobytes()
    sess.ingest_pcm16_chunk(extra)
    res = sess.current_best(top_k=3)
    assert len(res) == 3 and res[0]["user_id"] == "u0"

    out2 = sess.current_best_binary(threshold=0.7, top_k=3)
    assert out2["decision"] is True and out2["best"]["user_id"] == "u0"
# (обрезка буфера, 300мс-гейт, стабильная форма ответа)
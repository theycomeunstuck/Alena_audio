import json
import time
import numpy as np
import pytest

import app.services.streaming_verify as sv
from starlette.websockets import WebSocketDisconnect

@pytest.fixture()
def patch_sv_for_ws(monkeypatch):
    # Быстрый ресемплер
    def ensure_float_mono_16k_from_pcm16(pcm_chunk: bytes, src_sr: int, channels: int):
        x = np.frombuffer(pcm_chunk, dtype="<i2").astype(np.float32) / 32767.0
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        return x
    monkeypatch.setattr(sv, "ensure_float_mono_16k_from_pcm16", ensure_float_mono_16k_from_pcm16, raising=True)

    # Стабильный матчер
    class DummyMatcher:
        def match_probe_array(self, audio, top_k=5):
            return [{"user_id": f"u{i}", "score": 0.8 - 0.01*i, "ref_path": f"/ref/{i}"} for i in range(top_k)]
    monkeypatch.setattr(sv, "get_global_matcher", lambda: DummyMatcher(), raising=True)



def _recv_until(ws, predicate, max_msgs=10, timeout=2.0):
    start = time.time()
    msgs = []

    while len(msgs) < max_msgs and (time.time() - start) < timeout:
        try:
            msg = ws.receive_json()
        except WebSocketDisconnect:
            # сокет закрыт — это НОРМАЛЬНО
            break

        msgs.append(msg)
        if predicate(msg):
            return msg, msgs

    raise AssertionError(f"Expected message not received, got: {msgs}")

def test_ready_and_metadata(client, patch_sv_for_ws):
    with client.websocket_connect("/ws/speaker/verify?emit_interval_ms=1000&top_k=3") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        # Метаданные safe_send добавлены
        for k in ("session_id", "version", "ts_ms"):
            assert k in ready
        # Параметры сессии отражены
        assert set(("sample_rate", "channels", "sim_threshold", "emit_interval_ms", "top_k")) <= set(ready.keys())
# (ready + safe_send метаданные)

def test_errors_and_flow_partial_final(client, patch_sv_for_ws):
    with client.websocket_connect("/ws/speaker/verify?emit_interval_ms=1000&top_k=3") as ws:
        ws.receive_json()  # ready

        # bad_frame_size
        ws.send_bytes(b"\x00")  # не кратно 2*channels
        err, _ = _recv_until(ws, lambda m: m.get("type") == "error" and m.get("code") == "bad_frame_size")
        assert err["detail"]["bytes_per_frame"] == 2

        # bad_json
        ws.send_text("not a json")
        err, _ = _recv_until(ws, lambda m: m.get("type") == "error" and m.get("code") == "bad_json")

        # unknown_event
        ws.send_text(json.dumps({"event": "weird"}))
        err, _ = _recv_until(ws, lambda m: m.get("type") == "error" and m.get("code") == "unknown_event")

        # Заливаем 0.4c PCM16 и делаем flush -> partial
        good = (np.zeros(int(0.4 * 16000), dtype="<i2")).tobytes()
        ws.send_bytes(good)
        ws.send_text(json.dumps({"event": "flush"}))
        partial, _ = _recv_until(ws, lambda m: m.get("type") == "partial")
        assert set(partial.keys()) >= {"decision", "threshold", "best", "matches", "type"}

        # stop -> final
        ws.send_text(json.dumps({"event": "stop"}))
        final, _ = _recv_until(ws, lambda m: m.get("type") == "final")
        assert final["reason"] == "client_stop"
# (валидации, flush→partial, stop→final)


def test_final_by_inactivity(client, patch_sv_for_ws):
    # inactivity_sec=0.0 → финал сразу после любой обработки сообщения
    with client.websocket_connect("/ws/speaker/verify?inactivity_sec=0.0&emit_interval_ms=1000") as ws:
        ws.receive_json()  # ready
        ws.send_text(json.dumps({"event": "noop"}))  # вызовет unknown_event, затем inactivity break
        final, _ = _recv_until(ws, lambda m: m.get("type") == "final")
        assert final["reason"] == "inactivity"
# (финал по неактивности)
# tests/test_websocket_asr.py
import json
import numpy as np
from tests.conftest import make_sine, pcm16_from_np
from core.config import SAMPLE_RATE


def test_ws_asr_basic_flow(client, mock_asr_transcribe):
    with client.websocket_connect(f"/ws/asr?language=ru&sample_rate={SAMPLE_RATE}") as ws:
        # сервер готов
        msg = ws.receive_json()
        assert msg["type"] == "ready"

        # отправим 100 ms звука
        x = make_sine(SAMPLE_RATE, 0.1)
        ws.send_bytes(pcm16_from_np(x))

        # попросим flush -> должен прийти partial
        ws.send_text(json.dumps({"event": "flush"}))
        msg2 = ws.receive_json()
        assert msg2["type"] == "partial"
        assert msg2["text"].startswith("len=")

        # стоп -> финал и закрытие
        ws.send_text(json.dumps({"event": "stop"}))
        msg3 = ws.receive_json()
        assert msg3["type"] == "final"
        assert msg3["text"].startswith("len=")


def test_ws_asr_handles_invalid_json(client):
    with client.websocket_connect(f"/ws/asr?language=ru&sample_rate={SAMPLE_RATE}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "ready"

        # отправим битый JSON
        ws.send_text("{broken json")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Invalid JSON" in err["detail"]


def test_ws_asr_multiple_chunks_no_crash(client, mock_asr_transcribe):
    with client.websocket_connect(f"/ws/asr?language=ru&sample_rate={SAMPLE_RATE}") as ws:
        ws.receive_json()  # ready

        # шлём много маленьких кусков; сервер должен иногда отдавать partial
        got_partial = False
        for _ in range(10):
            x = np.random.uniform(-0.1, 0.1, int(0.02 * SAMPLE_RATE)).astype(np.float32)
            ws.send_bytes(pcm16_from_np(x))
            # не требуем ответ каждый раз; но через flush проверим
        ws.send_text(json.dumps({"event": "flush"}))
        msg = ws.receive_json()
        assert msg["type"] == "partial"
        got_partial = True

        assert got_partial

        # стоп
        ws.send_text(json.dumps({"event": "stop"}))
        final = ws.receive_json()
        assert final["type"] == "final"

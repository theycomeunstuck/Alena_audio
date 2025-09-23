# tests/test_deep_asr_ws.py
import json, struct, time, pytest
from fastapi.testclient import TestClient
from app.main import app

pytestmark = pytest.mark.deep
client = TestClient(app)

def pcm16_tone(seconds: float, freq: float = 440.0, sr: int = 16000):
    import math
    N = int(seconds * sr)
    data = bytearray()
    for n in range(N):
        v = int(0.2 * 32767 * math.sin(2*math.pi*freq*n/sr))
        data += int.to_bytes(v, 2, "little", signed=True)
    return bytes(data)

def test_ws_asr_deep_end_to_end():
    with client.websocket_connect("/ws/asr?language=ru&sample_rate=16000") as ws:
        ws.receive_json()  # ready

        tone = pcm16_tone(2.0)
        chunk = 32000  # ~1 сек
        for i in range(0, len(tone), chunk):
            ws.send_bytes(tone[i:i+chunk])

        # запросим partial — не проверяем содержимое
        ws.send_text(json.dumps({"event": "flush"}))
        _ = ws.receive_json()

        # запрашиваем финал
        ws.send_text(json.dumps({"event": "stop"}))

        # читаем, пока не придёт final (может прилететь 1-2 partial перед ним)
        deadline = time.time() + 5.0  # таймаут 5с
        final = None
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg.get("type") == "final":
                final = msg
                break
        assert final is not None, "Не дождались 'final' в ответ на 'stop' за 5 секунд"
        assert "text" in final

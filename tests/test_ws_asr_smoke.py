# tests/test_ws_asr_smoke.py
import json
import struct
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)

def pcm16_silence(seconds: float, sr: int = 16000):
    n = int(seconds * sr)
    # 16-bit little-endian нули
    return struct.pack("<" + "h"*n, *([0]*n))

def test_ws_asr_protocol_smoke(monkeypatch):
    # Если хотите вообще исключить модель — подмените функцию транскрипции,
    # чтобы WS-слой отработал независимо от Whisper:
    # monkeypatch.setattr("app.services.audio_service.asr_model.transcribe",
    #                     lambda audio, language=None, fp16=False: {"text": "", "raw": {}})

    with client.websocket_connect("/ws/asr?language=ru&sample_rate=16000") as ws:
        # ready
        msg = ws.receive_json()
        assert msg.get("type") == "ready"
        assert msg.get("sample_rate") == 16000

        # отправим немного тишины
        ws.send_bytes(pcm16_silence(0.5))

        # запросим partial
        ws.send_text(json.dumps({"event": "flush"}))
        partial = ws.receive_json()
        assert partial.get("type") in ("partial", "final", "empty", "ok")

        # запросим final
        ws.send_text(json.dumps({"event": "stop"}))
        final = ws.receive_json()
        assert final.get("type") == "final"

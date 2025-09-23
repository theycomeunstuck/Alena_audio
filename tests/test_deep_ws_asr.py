# tests/test_deep_ws_asr.py
import json
import numpy as np
import pytest

from fastapi.testclient import TestClient
from app.main import app
from core.config import SAMPLE_RATE

from tests.conftest import pcm16_from_np


pytestmark = pytest.mark.deep
client = TestClient(app)


def test_ws_asr_real_models_accepts_and_responds():
    with client.websocket_connect(f"/ws/asr?language=ru&sample_rate={SAMPLE_RATE}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        # немного «нейтрального» сигнала, просто чтобы проверить устойчивость стрима
        chunk = np.random.uniform(-0.05, 0.05, int(0.5 * SAMPLE_RATE)).astype(np.float32)
        ws.send_bytes(pcm16_from_np(chunk))

        # попросим partial
        ws.send_text(json.dumps({"event": "flush"}))
        part = ws.receive_json() # partial (может быть пустым — это нормально)
        assert part["type"] == "partial"
        assert "text" in part

        # стоп → финал
        ws.send_text(json.dumps({"event": "stop"}))
        final = ws.receive_json()
        assert final["type"] == "final"
        assert "text" in final

# tests/test_ws_asr_smoke.py
"""Protocol smoke test for /ws/asr through the real FastAPI app.

The route drives the production StreamingASRSession; only the model seam
(app.services.streaming_asr.asr_model - the binding the session actually
reads) is replaced, so no Whisper weights are ever loaded.

Ordering: the background emitter sends automatic partials, so its frames can
interleave with the flush/stop/reset answers. The documented protocol echoes
utt_id on flush/stop/reset answers while background partials never carry one,
so the test correlates on utt_id and drains interleaved frames with a bounded
receive loop (which also makes the test hang-proof: the emitter keeps the
frame flow alive, and a route crash surfaces as WebSocketDisconnect instead
of a silent timeout).

The previous version of this test ran the real Whisper (the commented hint
even pointed at the wrong seam, app.services.audio_service.asr_model) and read
exactly one frame per command: the emitter's first-tick partial and the real
model's failures raced the flush/stop answers, so the test saw an "error"
frame where it expected "final".
"""
import json
import struct

from fastapi.testclient import TestClient

import app.services.streaming_asr as streaming_asr
from app.main import app

client = TestClient(app)


def pcm16_silence(seconds: float, sr: int = 16000):
    n = int(seconds * sr)
    # 16-bit little-endian silence
    return struct.pack("<" + "h" * n, *([0] * n))


class _FakeASR:
    """Same seam fake as tests/test_ws_asr_resilience.py: the text depends on
    the buffer length, so partial/final content is deterministic without
    loading any weights."""

    def transcribe(self, audio, language=None, **kw):
        return {"text": f"len={int(len(audio))}", "segments": [], "language": language or "ru"}


def _recv_msg(ws, pred, max_msgs=10):
    """Receive until pred matches, draining interleaved background frames."""
    for _ in range(max_msgs):
        msg = ws.receive_json()
        if pred(msg):
            return msg
    raise AssertionError(f"expected protocol message not received within {max_msgs} frames")


def test_ws_asr_protocol_smoke(monkeypatch):
    monkeypatch.setattr(streaming_asr, "asr_model", _FakeASR())

    with client.websocket_connect("/ws/asr?language=ru&sample_rate=16000") as ws:
        # 1) ready greeting with the documented session parameters
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000
        assert ready["language"] == "ru"
        assert ready["window_sec"] == 8.0
        assert ready["emit_sec"] == 2.0
        assert ready["inactivity_sec"] is None

        # 2) flush answers with a partial echoing utt_id
        #    (0.5 s @ 16 kHz = 8000 samples; the fake reports len=<samples>)
        ws.send_bytes(pcm16_silence(0.5))
        ws.send_text(json.dumps({"event": "flush", "utt_id": "u1"}))
        partial = _recv_msg(ws, lambda m: m.get("type") == "partial" and m.get("utt_id") == "u1")
        assert partial["text"] == "len=8000"

        # 3) stop answers with a final for the same (not yet reset) buffer,
        #    echoing utt_id
        ws.send_text(json.dumps({"event": "stop", "utt_id": "u2"}))
        final = _recv_msg(ws, lambda m: m.get("type") == "final" and m.get("utt_id") == "u2")
        assert final["text"] == "len=8000"

        # 4) long-lived socket: after stop the buffer is reset but the session
        #    still answers flush; an empty buffer answers without a model call.
        ws.send_text(json.dumps({"event": "flush", "utt_id": "u3"}))
        partial2 = _recv_msg(ws, lambda m: m.get("type") == "partial" and m.get("utt_id") == "u3")
        assert partial2["text"] == ""

        # 5) reset -> "ok" with the reset detail and the echoed utt_id
        ws.send_text(json.dumps({"event": "reset", "utt_id": "u4"}))
        ok = _recv_msg(ws, lambda m: m.get("type") == "ok" and m.get("utt_id") == "u4")
        assert ok["detail"] == "reset"

        # 6) unknown event -> error notice naming the event
        ws.send_text(json.dumps({"event": "definitely-not-an-event"}))
        err = _recv_msg(ws, lambda m: m.get("type") == "error")
        assert "definitely-not-an-event" in err["detail"]

        # 7) broken JSON -> error notice and the session survives
        #    (a subsequent flush is still answered)
        ws.send_text("{broken json")
        _recv_msg(ws, lambda m: m.get("type") == "error" and "Invalid JSON" in m.get("detail", ""))
        ws.send_text(json.dumps({"event": "flush", "utt_id": "u5"}))
        partial3 = _recv_msg(ws, lambda m: m.get("type") == "partial" and m.get("utt_id") == "u5")
        assert partial3["text"] == ""

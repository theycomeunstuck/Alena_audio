# Regression tests for REG-3: /ws/asr flush, stop and auto-final transcription
# must run off the event loop (asyncio.to_thread), like the REST routes fixed
# under F-010. Same harness as tests/test_ws_asr_resilience.py.
import asyncio
import json
import struct
import math
import time
from contextlib import suppress

from starlette.websockets import WebSocketDisconnect, WebSocketState

import app.api.routes_ws as routes_ws
import app.services.streaming_asr as streaming_asr


def pcm16(seconds, sr=16000, freq=200.0):
    n = int(sr * seconds)
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr))) for i in range(n)
    )


class FakeWS:
    def __init__(self):
        self.inbox = asyncio.Queue()
        self.sent = []
        self.accepted = False
        self.closed = False
        self.client_state = WebSocketState.CONNECTED

    async def accept(self):
        self.accepted = True

    async def send_json(self, obj, mode=None):
        self.sent.append(dict(obj))

    async def receive(self):
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code=1000):
        self.closed = True


class SlowASR:
    """Transcriber whose blocking call stalls long enough to starve a busy loop."""

    def __init__(self, stall=0.3):
        self.stall = stall
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        time.sleep(self.stall)
        # The real transcriber returns a result dict (see StreamingASRSession).
        return {"text": "замедленный текст", "segments": [], "language": "ru"}


async def _wait_for(pred, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


def test_flush_transcription_does_not_starve_the_event_loop(monkeypatch):
    slow = SlowASR(stall=0.3)
    monkeypatch.setattr(streaming_asr, "asr_model", slow)

    async def scenario():
        ws = FakeWS()
        task = asyncio.create_task(routes_ws.ws_asr(
            ws=ws, language="ru", sample_rate=16000, channels=1,
            window_sec=8.0, emit_sec=0.2, inactivity_sec=None))
        ticks = 0
        try:
            assert await _wait_for(lambda: any(o.get("type") == "ready" for o in ws.sent), 3)
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.2)})

            async def heartbeat():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.05)
                    ticks += 1

            hb = asyncio.create_task(heartbeat())
            before = ticks
            await ws.inbox.put({"type": "websocket.receive",
                                "text": json.dumps({"event": "flush", "utt_id": "u1"})})
            got = await _wait_for(lambda: any(o.get("type") == "partial" and o.get("text") for o in ws.sent), 4)
            hb.cancel()
            with suppress(asyncio.CancelledError):
                await hb
            assert got, "flush must still answer"
            advanced = ticks - before
            # 0.3 s stall on the loop would allow at most 1 tick; offloaded, the
            # loop keeps ticking (~6 ticks at 0.05 s).
            assert advanced >= 3, (
                f"event loop starved during flush transcription: only {advanced} heartbeat ticks "
                "in ~0.3 s - the Whisper call still runs on the loop (REG-3)")
        finally:
            await ws.inbox.put(WebSocketDisconnect(code=1000))
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())


def test_stop_and_autofinal_transcription_is_offloaded(monkeypatch):
    slow = SlowASR(stall=0.3)
    monkeypatch.setattr(streaming_asr, "asr_model", slow)

    async def scenario():
        ws = FakeWS()
        task = asyncio.create_task(routes_ws.ws_asr(
            ws=ws, language="ru", sample_rate=16000, channels=1,
            window_sec=8.0, emit_sec=0.2, inactivity_sec=None))
        ticks = 0
        try:
            assert await _wait_for(lambda: any(o.get("type") == "ready" for o in ws.sent), 3)
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.2)})

            async def heartbeat():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.05)
                    ticks += 1

            hb = asyncio.create_task(heartbeat())
            before = ticks
            await ws.inbox.put({"type": "websocket.receive",
                                "text": json.dumps({"event": "stop", "utt_id": "u1"})})
            got = await _wait_for(lambda: any(o.get("type") == "final" for o in ws.sent), 4)
            hb.cancel()
            with suppress(asyncio.CancelledError):
                await hb
            assert got, "stop must still answer with a final"
            advanced = ticks - before
            assert advanced >= 3, (
                f"event loop starved during stop transcription: only {advanced} ticks (REG-3)")
        finally:
            await ws.inbox.put(WebSocketDisconnect(code=1000))
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())


def test_bg_loop_finalize_is_offloaded():
    # Source-level pin: the auto-final inside the background emitter must run
    # via asyncio.to_thread as well.
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "api" / "routes_ws.py").read_text(encoding="utf-8")
    assert "to_thread(sess.finalize" in src or "to_thread(sess._transcribe" in src, (
        "bg_loop finalize must be offloaded (REG-3)")

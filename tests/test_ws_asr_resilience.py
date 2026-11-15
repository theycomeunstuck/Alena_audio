# tests/test_ws_asr_resilience.py
"""Regression tests for audit findings F-018a/F-018b/F-018c and F-025a in app/api/routes_ws.py.

The tests drive the REAL route handlers (routes_ws.ws_asr, routes_ws.ws_speaker_verify)
with a fake WebSocket transport - the same technique as the verification repro
.audit/20261115-0002/repro/V3/cl_ws_repro.py - so the route logic, ticker and background
emitter under test are the production code. Only the model seams are faked:

  - app.services.streaming_asr.asr_model is replaced by a stub transcriber so no
    Whisper inference runs;
  - routes_ws.StreamingVerifySession is replaced by a counting fake so no
    ECAPA/matcher inference runs.

Covered findings:
  - F-018a: the /ws/asr background emitter must keep working after a "stop"
    (socket stays open per the documented long-lived-socket contract);
  - F-018b: /ws/speaker/verify inactivity auto-STOP must fire on true silence,
    not only after a received message;
  - F-018c: one odd-length PCM frame on /ws/asr must produce a "bad_frame_size"
    notice and must not end the session;
  - F-025a: window_sec must have an upper bound (le=60) while keeping the 8.0 default.
"""
import asyncio
import inspect
import json
import math
import struct
import time
from contextlib import suppress

import annotated_types as at

import pytest
from starlette.websockets import WebSocketDisconnect, WebSocketState

import app.api.routes_ws as routes_ws
import app.services.streaming_asr as streaming_asr


# --------------------------------------------------------------------------- helpers

def pcm16(seconds, sr=16000, freq=200.0):
    """PCM16 little-endian sine bytes, like the audit repro uses."""
    n = int(sr * seconds)
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr))) for i in range(n)
    )


class FakeWS:
    """Minimal fake WebSocket transport (mirrors the audit repro)."""

    def __init__(self):
        self.inbox = asyncio.Queue()
        self.sent = []
        self.accepted = False
        self.closed = False
        self.close_code = None
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
        self.close_code = code
        # Mimic the post-close state: after the server closed, sends are no-ops
        # (safe_send in the verify handler relies on client_state).
        self.client_state = WebSocketState.DISCONNECTED


async def wait_for(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def frames(ws, since=0):
    return ws.sent[since:]


class _FakeASR:
    """Mimics the whisper transcribe interface used by StreamingASRSession."""

    def transcribe(self, audio, language=None, **kw):
        return {"text": f"len={int(len(audio))}", "segments": [], "language": language or "ru"}


class _FakeVerifySession:
    """Same interface as StreamingVerifySession; counts inactivity checks.

    Lets the test observe WHEN the route consults inactivity without any model
    inference (same trick as the audit repro's CountingFakeVerifySession).
    """

    instances = []

    def __init__(self, sample_rate=16000, channels_hint=1, inactivity_sec=120.0):
        self.inactivity_sec = inactivity_sec
        self._last_audio_t = time.time()
        self.ingest_count = 0
        self.inactivity_checks = 0
        _FakeVerifySession.instances.append(self)

    def ingest_pcm16_chunk(self, b):
        self.ingest_count += 1
        self._last_audio_t = time.time()

    def inactive_timed_out(self):
        self.inactivity_checks += 1
        if self.inactivity_sec is None:
            return False
        return (time.time() - self._last_audio_t) >= float(self.inactivity_sec)

    def current_best_binary(self, threshold, top_k=5):
        return {"decision": False, "threshold": float(threshold), "best": None, "matches": []}

    def reset(self):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------- F-018a

def test_asr_emitter_restarts_after_stop(monkeypatch):
    """F-018a: after an explicit "stop" the socket stays open and the background
    emitter must be restarted - the client keeps receiving automatic partials and
    inactivity auto-finals for the rest of the connection."""
    monkeypatch.setattr(streaming_asr, "asr_model", _FakeASR())

    async def scenario():
        ws = FakeWS()
        task = asyncio.create_task(routes_ws.ws_asr(
            ws=ws, language="ru", sample_rate=16000, channels=1,
            window_sec=8.0, emit_sec=0.2, inactivity_sec=0.8))
        try:
            assert await wait_for(lambda: any(o.get("type") == "ready" for o in ws.sent), 3)

            # Pre-stop sanity: the emitter produces a partial, then an auto-final on silence.
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.3)})
            assert await wait_for(
                lambda: any(o.get("type") == "partial" and o.get("text", "").startswith("len=")
                            for o in ws.sent), 3)
            assert await wait_for(lambda: any(o.get("type") == "final" for o in ws.sent), 4)

            # Explicit stop -> one final, buffer reset, socket stays open.
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.3)})
            await asyncio.sleep(0.05)
            n_before_stop = len(ws.sent)
            await ws.inbox.put({"type": "websocket.receive",
                                "text": json.dumps({"event": "stop", "utt_id": "u2"})})
            assert await wait_for(
                lambda: any(o.get("type") == "final" for o in frames(ws, n_before_stop)), 4)

            # Post-stop: the session must keep emitting partials and auto-finals.
            after_stop = len(ws.sent)
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.3)})
            got_partial = await wait_for(
                lambda: any(o.get("type") == "partial" and o.get("text", "").startswith("len=")
                            for o in frames(ws, after_stop)), 3)
            got_autofinal = await wait_for(
                lambda: any(o.get("type") == "final" for o in frames(ws, after_stop)), 4)

            # Manual flush is still answered.
            n_before_flush = len(ws.sent)
            await ws.inbox.put({"type": "websocket.receive",
                                "text": json.dumps({"event": "flush", "utt_id": "u3"})})
            got_flush = await wait_for(
                lambda: any(o.get("type") == "partial" for o in frames(ws, n_before_flush)), 3)

            assert got_partial, "no automatic partial after stop - emitter not restarted (F-018a)"
            assert got_autofinal, "no inactivity auto-final after stop - emitter not restarted (F-018a)"
            assert got_flush, "manual flush not answered after stop"
        finally:
            await ws.inbox.put(WebSocketDisconnect(code=1000))
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)
        assert task.done() and not task.cancelled()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- F-018b

def test_verify_inactivity_fires_on_true_silence(monkeypatch):
    """F-018b: /ws/speaker/verify auto-STOP must fire while the client is simply
    silent (no messages at all), not only right after some received message."""
    monkeypatch.setattr(routes_ws, "StreamingVerifySession", _FakeVerifySession)

    async def scenario():
        ws = FakeWS()
        task = asyncio.create_task(routes_ws.ws_speaker_verify(
            ws=ws, sample_rate=16000, channels=1, top_k=5,
            inactivity_sec=0.6, sim_threshold=0.65, emit_interval_ms=200))
        try:
            assert await wait_for(lambda: any(o.get("type") == "ready" for o in ws.sent), 3)

            # Speak a little; wait until the receive loop actually processed the
            # frame (its post-message inactivity check runs at least once).
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.3)})
            assert await wait_for(lambda: any(o.get("type") == "partial" for o in ws.sent), 3)
            inst = _FakeVerifySession.instances[-1]
            assert await wait_for(lambda: inst.inactivity_checks >= 1, 3)

            # Go silent: no messages at all for 2.5 s (~4x inactivity_sec 0.6).
            since = len(ws.sent)
            await asyncio.sleep(2.5)
            during = frames(ws, since)
            finals = [o for o in during if o.get("type") == "final"]
            partials = [o for o in during if o.get("type") == "partial"]

            await ws.inbox.put({"type": "websocket.disconnect", "code": 1000})
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)

            assert len(partials) >= 1, "ticker stopped emitting partials during silence"
            assert len(finals) == 1 and finals[0].get("reason") == "inactivity", (
                f"auto-STOP did not fire on true silence (F-018b): finals={finals!r}")
        finally:
            await ws.inbox.put({"type": "websocket.disconnect", "code": 1000})
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)
        assert task.done() and not task.cancelled()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- F-018c

def test_asr_odd_frame_rejected_and_session_continues(monkeypatch):
    """F-018c: one odd-length PCM frame must produce the same kind of
    "bad_frame_size" notice /ws/speaker/verify sends and must NOT end the session."""
    monkeypatch.setattr(streaming_asr, "asr_model", _FakeASR())

    async def scenario():
        ws = FakeWS()
        task = asyncio.create_task(routes_ws.ws_asr(
            ws=ws, language="ru", sample_rate=16000, channels=1,
            window_sec=8.0, emit_sec=0.2, inactivity_sec=None))
        try:
            assert await wait_for(lambda: any(o.get("type") == "ready" for o in ws.sent), 3)

            # One odd-length frame (3 bytes cannot be whole int16 samples).
            await ws.inbox.put({"type": "websocket.receive", "bytes": b"\x01\x02\x03"})
            got_bad = await wait_for(
                lambda: any(o.get("type") == "error" and o.get("code") == "bad_frame_size"
                            for o in ws.sent), 3)

            # The session continues: an even frame followed by flush is answered
            # and the server has not closed the socket.
            await ws.inbox.put({"type": "websocket.receive", "bytes": pcm16(0.3)})
            await ws.inbox.put({"type": "websocket.receive",
                                "text": json.dumps({"event": "flush", "utt_id": "u9"})})
            got_flush = await wait_for(
                lambda: any(o.get("type") == "partial" and o.get("text", "").startswith("len=")
                            for o in ws.sent), 3)

            assert got_bad, "no bad_frame_size notice for an odd PCM frame (F-018c)"
            assert got_flush, "session did not survive one odd PCM frame (F-018c)"
            assert not ws.closed, "server closed the socket after one odd PCM frame (F-018c)"
            assert not any("server exception" in (o.get("detail") or "") for o in ws.sent), (
                "odd frame leaked into a server exception (F-018c)")
        finally:
            await ws.inbox.put(WebSocketDisconnect(code=1000))
            with suppress(Exception):
                await asyncio.wait_for(task, timeout=5)
        assert task.done() and not task.cancelled()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- F-025a

def test_window_sec_has_upper_bound(client):
    """F-025a: window_sec keeps the 8.0 default but is bounded above (le=60)."""
    q = inspect.signature(routes_ws.ws_asr).parameters["window_sec"].default
    # FastAPI >= 0.100 stores numeric constraints as annotated_types metadata.
    lower = [m.gt for m in q.metadata if isinstance(m, at.Gt)]
    upper = [m.le for m in q.metadata if isinstance(m, at.Le)]
    assert q.default == 8.0
    assert lower == [0]
    assert upper == [60], f"window_sec upper bound missing (F-025a): metadata={q.metadata!r}"

    # Live validation through the real app: beyond the bound is rejected (1008).
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/asr?window_sec=100&emit_sec=0.2"):
            pass
    assert exc_info.value.code == 1008

    # The boundary itself is still accepted and reported in the ready frame.
    with client.websocket_connect("/ws/asr?window_sec=60&emit_sec=0.2") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["window_sec"] == 60.0

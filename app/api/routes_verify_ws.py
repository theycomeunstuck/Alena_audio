# app/api/routes_verify_ws.py
from __future__ import annotations
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.services.audio_utils import SAMPLE_RATE
from app.services.streaming_verify import StreamingVerifySession
from core.config import sim_threshold as _sim_threshold
router = APIRouter()

@router.websocket("/ws/speaker/verify")
async def ws_speaker_verify(
    ws: WebSocket,
    sample_rate: int = Query(SAMPLE_RATE, ge=8000, le=48000, description="Частота входящих PCM16 кадров"),
    channels: int = Query(1, ge=1, le=2, description="Число каналов входящих PCM16 кадров"),
    top_k: int = Query(5, ge=1, le=50, description="Количество лучших совпадений в диагностике"),
    inactivity_sec: float | None = Query(120.0, description="Авто-STOP при молчании (сек)"),
    sim_threshold: float = Query(_sim_threshold, ge=0.0, le=1.0, description="Порог совпадения по [0..1]"),
    emit_interval_ms: int = Query(500, ge=50, le=5000, description="Интервал авто-partial, мс"),
):
    """
    Постоянная верификация:
    - Клиент присылает PCM16-байты (любого размера), сервер буферизует.
    - Раз в emit_interval_ms сервер сам отсылает partial с {'decision','threshold','best','matches'}.
    - При молчании дольше inactivity_sec сервер шлёт final и закрывает соединение.
    - Управляющие сообщения: {"event":"flush"} → принудительный partial, {"event":"stop"} → финал.
    """
    await ws.accept()
    sess = StreamingVerifySession(sample_rate=sample_rate, channels_hint=channels, inactivity_sec=inactivity_sec)
    await ws.send_json({
        "type": "ready",
        "sample_rate": sample_rate,
        "channels": channels,
        "sim_threshold": sim_threshold,
        "emit_interval_ms": emit_interval_ms
    })

    interval = max(0.05, emit_interval_ms / 1000.0)

    try:
        while True:
            try:
                # ждём данные или текст не дольше interval секунд
                msg = await asyncio.wait_for(ws.receive(), timeout=interval)
                if "bytes" in msg and msg["bytes"] is not None:
                    sess.ingest_pcm16_chunk(msg["bytes"])
                elif "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        payload = {}
                    event = payload.get("event")
                    if event == "flush":
                        res = sess.current_best_binary(sim_threshold, top_k=top_k)
                        await ws.send_json({"type": "partial", **res})
                    elif event == "stop":
                        res = sess.current_best_binary(sim_threshold, top_k=top_k)
                        await ws.send_json({"type": "final", **res})
                        break
            except asyncio.TimeoutError:
                if ws.client_state.name != "DISCONNECTED":
                    res = sess.current_best_binary(sim_threshold, top_k=top_k)
                    await ws.send_json({"type": "partial", **res})

            if ws.client_state.name == "DISCONNECTED":
                break

            if sess.inactive_timed_out():
                if ws.client_state.name != "DISCONNECTED":
                    res = sess.current_best_binary(sim_threshold, top_k=top_k)
                    await ws.send_json({"type": "final", **res, "reason": "inactivity"})
                break

            # Авто-финал по молчанию
            if sess.inactive_timed_out():
                res = sess.current_best_binary(sim_threshold, top_k=top_k)
                await ws.send_json({"type": "final", **res, "reason": "inactivity"})
                break

    except WebSocketDisconnect:
        # Клиент закрыл соединение; отправить финал уже нельзя
        return

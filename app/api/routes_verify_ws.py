# app/api/routes_verify_ws.py
from __future__ import annotations
import json
import uuid
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState
from app.services.audio_utils import SAMPLE_RATE
from app.services.streaming_verify import StreamingVerifySession
from core.config import sim_threshold as _sim_threshold

router = APIRouter()

@router.websocket("/ws/speaker/verify")
async def ws_speaker_verify(
    ws: WebSocket,
    sample_rate: int = Query(SAMPLE_RATE, ge=8000, le=48000, description="Частота входящих PCM16 кадров"),
    channels: int = Query(1, ge=1, le=2, description="Число каналов входящих PCM16"),
    top_k: int = Query(5, ge=1, le=50, description="Количество лучших совпадений в диагностике"),
    inactivity_sec: float | None = Query(120.0, description="Авто-STOP при молчании (сек)"),
    sim_threshold: float = Query(_sim_threshold, ge=0.0, le=1.0, description="Порог совпадения [0..1]"),
    emit_interval_ms: int = Query(500, ge=50, le=5000, description="Интервал авто-partial, мс"),
):
    await ws.accept()

    session_id = uuid.uuid4().hex
    version = 1
    interval = max(0.05, emit_interval_ms / 1000.0)

    sess = StreamingVerifySession(
        sample_rate=sample_rate,
        channels_hint=channels,
        inactivity_sec=inactivity_sec
    )

    async def safe_send(obj: dict) -> None:
        if ws.client_state == WebSocketState.CONNECTED:
            # добавим метаданные в каждый ответ
            obj.setdefault("session_id", session_id)
            obj.setdefault("version", version)
            obj.setdefault("ts_ms", int(time.time() * 1000))
            try:
                await ws.send_json(obj)
            except RuntimeError:
                # гонка при закрытии сокета — просто игнорим
                pass

    await safe_send({
        "type": "ready",
        "sample_rate": sample_rate,
        "channels": channels,
        "sim_threshold": sim_threshold,
        "emit_interval_ms": emit_interval_ms,
        "top_k": top_k,
    })

    stop_reason: str | None = None
    send_lock = asyncio.Lock()  # один отправитель за раз (тикер/receive)

    dirty = asyncio.Event()
    dirty.set()  # считаем один раз сразу после ready
    inflight = False  # защита от параллельных расчётов

    async def compute_and_send(type: str):
        nonlocal inflight
        if inflight:
            return
        inflight = True
        try:
            # выносим cpu/gpu-вычисления из event-loop
            res = await asyncio.to_thread(sess.current_best_binary, sim_threshold, top_k=top_k)
            async with send_lock:
                await safe_send({"type": type, **res})
        finally:
            inflight = False

    async def send_partial():
        await compute_and_send("partial")

    async def send_final(reason: str | None):
        res = await asyncio.to_thread(sess.current_best_binary, sim_threshold, top_k=top_k)
        async with send_lock:
            await safe_send({"type": "final", **res, "reason": reason})

    # фон-таск, который шлёт partial по таймеру
    async def ticker():
        try:
            while True:
                try: # ждём новый звук, но не реже чем раз в interval
                    await asyncio.wait_for(dirty.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
                dirty.clear()
                await compute_and_send("partial")
        except asyncio.CancelledError:
            pass

    tick_task = asyncio.create_task(ticker())

    try:
        while True:
            msg = await ws.receive()

            # Явная обработка типов событий от Starlette
            if msg.get("type") == "websocket.disconnect":
                stop_reason = "client_disconnect"
                break

            b = msg.get("bytes")
            t = msg.get("text")

            if b is not None:
                # Валидация pcm16
                frame = len(b)
                bytes_per_frame = 2 * channels
                if frame % bytes_per_frame != 0:
                    async with send_lock:
                        await safe_send({
                            "type": "error",
                            "code": "bad_frame_size",
                            "detail": {"got_bytes": frame, "bytes_per_frame": bytes_per_frame}
                        })
                    # можно либо продолжить, либо отбросить «хвост»; сейчас просто продолжим
                    continue

                try:
                    sess.ingest_pcm16_chunk(b)
                    dirty.set()
                except Exception as e:
                    async with send_lock:
                        await safe_send({"type": "error", "code": "ingest_failed", "detail": str(e)})
                    continue

            elif t is not None:
                try:
                    payload = json.loads(t)
                except json.JSONDecodeError:
                    async with send_lock:
                        await safe_send({"type": "error", "code": "bad_json"})
                    continue

                event = payload.get("event")
                if event == "flush":
                    await send_partial()
                elif event == "stop":
                    stop_reason = "client_stop"
                    break
                else:
                    async with send_lock:
                        await safe_send({"type": "error", "code": "unknown_event", "detail": event})

            # Авто-финал по молчанию
            if sess.inactive_timed_out():
                stop_reason = "inactivity"
                break

    except WebSocketDisconnect:
        stop_reason = "client_disconnect"

    finally:
        # останавливаем тикер
        tick_task.cancel()
        try:
            await tick_task
        except asyncio.CancelledError:
            pass

        # финал и закрытие сокета
        if ws.client_state == WebSocketState.CONNECTED:
            await send_final(stop_reason)
            try:
                await ws.close(code=1000)
            except Exception:
                pass

        # подчистка сессии (если есть close)
        if hasattr(sess, "close"):
            try:
                sess.close()
            except Exception:
                pass
        else:
            sess.reset()

# app/api/routes_ws.py
from __future__ import annotations
import asyncio, json, time
from contextlib import suppress
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.services.streaming_asr import StreamingASRSession
from app.services.audio_utils import SAMPLE_RATE

router = APIRouter()

@router.websocket("/ws/asr")
async def ws_asr(
    ws: WebSocket,
    language: str = Query("ru", description="Код языка"),
    sample_rate: int = Query(SAMPLE_RATE, ge=8000, le=48000, description="Частота входящих PCM16 кадров"),
    channels: int = Query(1, ge=1, le=2, description="Число каналов входящих PCM16 кадров"),
    window_sec: float = Query(8.0, gt=0, description="Скользящее окно для буфера, сек. то есть окно, с которым мы работаем; храним последние window_sec сек."),
    emit_sec: float = Query(2.0, gt=0, description="Интервал эмиссии partial, сек. Частота обращения клиента к серверу "),
    inactivity_sec: Optional[float] = Query(None, description="Авто-final при тишине float сек; empty = off"),
):
    """
    ДОЛГОЖИВУЩИЙ сокет:
      - бинарные кадры = PCM16 (любая частота/каналы) → приводим к mono 16k
      - 'flush'  -> прислать partial
      - 'stop'   -> прислать final и СБРОСИТЬ буфер (сокет остаётся открытым)
      - 'reset'  -> принудительно очистить буфер (без final)
    Поддерживает utt_id для корреляции: пришлёте в 'flush/stop/reset' — вернём тем же в ответе.
    """
    await ws.accept()

    # Сессия живёт весь коннект
    assume_direct = (sample_rate == SAMPLE_RATE and channels == 1)
    sess = StreamingASRSession(
        sample_rate=sample_rate,
        language=language,
        window_sec=window_sec,
        emit_sec=emit_sec,
        assume_pcm16_mono_16k=assume_direct,
        channels_hint=channels,
        inactivity_sec=inactivity_sec,
    )

    # Приветствие
    await ws.send_json({
        "type": "ready",
        "sample_rate": sample_rate,
        "channels": channels,
        "language": language,
        "window_sec": window_sec,
        "emit_sec": emit_sec,
        "inactivity_sec": inactivity_sec,
    })

    # Фоновая эмиссия partial + авто-final по тишине
    stop_bg = False
    async def bg_loop():
        try:
            while not stop_bg:
                await asyncio.sleep(max(0.1, min(sess.emit_sec, 0.5)))  # частая «тикалка», но не шумная
                # авто-emission по emit_sec
                out = sess.maybe_emit()
                if out is not None:
                    text, _raw = out
                    await ws.send_json({"type": "partial", "text": text})
                # авто-final при длительной тишине
                if sess.inactive_timed_out():
                    text, _raw = sess.finalize()
                    await ws.send_json({"type": "final", "text": text})
                    sess.reset()  # готов к следующей фразе (сокет остаётся)
        except asyncio.CancelledError:
            return
        except Exception as e:
            with suppress(Exception):
                await ws.send_json({"type": "error", "detail": f"bg_loop: {e.__class__.__name__}"})

    bg_task = asyncio.create_task(bg_loop())

    try:
        while True:
            msg = await ws.receive()

            # БИНАРНЫЕ КАДРЫ: PCM16
            data = msg.get("bytes")
            if data is not None:
                # приведём к mono float32@16k + положим в буфер
                sess.ingest_pcm16_chunk(data)
                continue

            # ТЕКСТОВЫЕ КАДРЫ: JSON-команды
            text = msg.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "detail": "Invalid JSON: text frames must be JSON"})
                    continue

                event = payload.get("event")
                utt_id = payload.get("utt_id")  # опционально

                if event == "flush":
                    txt, _raw = sess._transcribe_partial()
                    out = {"type": "partial", "text": txt}
                    if utt_id is not None:
                        out["utt_id"] = utt_id
                    await ws.send_json(out)

                elif event == "stop":
                    # 1) жёстко останавливаем фонового эмиттера, чтобы не прилетел 'partial' поверх final
                    try:
                        bg_task.cancel()
                        try:
                            await bg_task
                        except asyncio.CancelledError: pass
                    except NameError: pass
                        # если по какой-то причине bg_task не создан — просто идём дальше


                    # 2) финализируем и отправляем 'final' (даже если текст пуст)
                    try:
                        txt, _raw = sess._transcribe_final()
                    except Exception:
                        txt = ""
                    out = {"type": "final", "text": txt}
                    if utt_id is not None:
                        out["utt_id"] = utt_id
                    await ws.send_json(out)

                    # 3) очистка буфера после отправки final
                    try:
                        sess.reset()
                    except Exception:
                        pass

                elif event == "reset":
                    sess.reset()
                    out = {"type": "ok", "detail": "reset"}
                    if utt_id is not None:
                        out["utt_id"] = utt_id
                    await ws.send_json(out)

                else:
                    await ws.send_json({"type": "error", "detail": f"unknown event: {event!r}"})
                continue

            # Иное (пустой фрейм и т.п.)
            await ws.send_json({"type": "error", "detail": "unsupported frame"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with suppress(Exception):
            await ws.send_json({"type": "error", "detail": f"server exception: {e.__class__.__name__}"})
    finally:
        if not bg_task.done():
            bg_task.cancel()
            with suppress(asyncio.CancelledError):
                await bg_task
        with suppress(Exception):
            await ws.close()

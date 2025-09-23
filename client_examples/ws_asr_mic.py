# scripts/ws_asr_mic.py
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets

# Наши хелперы
from app.services.audio_utils import ensure_float_mono_16k_from_pcm16, SAMPLE_RATE

WS_URL = "ws://127.0.0.1:8000/ws/asr?language=ru&sample_rate=16000&channels=1&window_sec=8&emit_sec=2&inactivity_sec=3.0"
print("нужно нажимать enter с задержкой. иначе байты не успевают долететь до сервера и теряются") #todo: described into the this print. packets race

# Для Windows + Python 3.10/11 иногда нужно SelectorPolicy (если где-то используются add_reader)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

def make_mic_stream(queue: "asyncio.Queue[bytes]", sr: int = 16000, channels: int = 1, block_ms: int = 50):
    """
    Возвращает sounddevice.RawInputStream, который складывает PCM16-чанки в очередь.
    block_ms=50 → ~800 байт на канал при 16kHz/16-bit (реально 16000 * 0.05 * 2 * channels).
    """
    blocksize = int(sr * block_ms / 1000)

    def callback(indata, frames, time_info, status):
        if status:
            # можно залогировать статус, но не падать
            # print("mic status:", status, file=sys.stderr)
            pass
        # indata — bytes-like (int16) если RawInputStream
        try:
            queue.put_nowait(bytes(indata))
        except asyncio.QueueFull:
            # если продюсер опережает потребителя — сбросим самый старый (или просто молча пропустим)
            try:
                _ = queue.get_nowait()
                queue.put_nowait(bytes(indata))
            except Exception:
                pass

    stream = sd.RawInputStream(
        samplerate=sr,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        callback=callback,
    )
    return stream

async def ws_sender(ws, queue: "asyncio.Queue[bytes]", src_sr: int, channels: int):
    """
    Читает PCM16-байты из очереди, при необходимости приводит к mono@16k и шлёт в сокет.
    """
    while True:
        pcm = await queue.get()
        if src_sr == SAMPLE_RATE and channels == 1:
            # уже подходящий формат — отправляем как есть
            await ws.send(pcm)
        else:
            # если вдруг микрофон не 16k/не моно — приведём
            x = ensure_float_mono_16k_from_pcm16(pcm, src_sr=src_sr, channels=channels)  # 1D float32@16k
            x = np.clip(x, -1.0, 1.0)

            pcm16 = (x * 32767.0).astype(np.int16, copy=False).tobytes()           # float -> PCM16 bytes
            await ws.send(pcm16)

async def interactive_loop(ws):
    """
    Простой REPL: 
    - микрофон пишет в WS постоянно,
    - по Enter → stop → ждём final → reset → следующая фраза.
    """
    print("Подключено. Нажми Enter, если закончил фразу. Ctrl+C для выхода.")
    utt_counter = 0
    while True:
        # ждём Enter от пользователя (не блокируя WS)
        await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        utt_counter += 1
        utt_id = f"utt{utt_counter}"

        # стоп — получаем финал
        await ws.send(json.dumps({"event": "stop", "utt_id": utt_id}))

        # читаем, пока не final (допускаем несколько partial перед ним)
        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)
            except Exception:
                # игнорируем бинарные кадры (мы сами их шлём)
                continue
            t = data.get("type")
            if t == "partial":
                print("partial:", data.get("text",""))
                pass
            elif t == "final":
                print(f"[{utt_id}] FINAL:", data.get("text", ""))
                break
            elif t == "error":
                print("WS error:", data.get("detail"))
                break

        # сбросить буфер перед следующей фразой
        await ws.send(json.dumps({"event": "reset", "utt_id": utt_id}))
        # можно прочитать ответ ok, но не обязательно

async def main():
    # Подключаемся к сокету
    async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
        ready = await ws.recv()
        print("WS:", ready)

        # Очередь обмена между аудио-коллбэком и WS
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # Микрофон: подними stream и отдельную задачу-отправителя
        mic_sr = 16000      # если твой микрофон другой, поставь реальную частоту (например, 48000)
        mic_channels = 1    # 1 или 2 — что даёт устройство
        stream = make_mic_stream(q, sr=mic_sr, channels=mic_channels, block_ms=50)

        # Запускаем стрим в sync-контексте, а асинхронно — таск отправки в WS + интерактив
        sender_task = asyncio.create_task(ws_sender(ws, q, src_sr=mic_sr, channels=mic_channels))
        try:
            with stream:
                await interactive_loop(ws)
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

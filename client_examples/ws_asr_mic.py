from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# Наши хелперы
from app.services.audio_utils import ensure_float_mono_16k_from_pcm16, SAMPLE_RATE

# Вариант без server-side auto_partials: просим partial сами через flush
WS_URL = (
    "ws://127.0.0.1:8000/ws/asr"
    "?language=ru&sample_rate=16000&channels=1&window_sec=8&emit_sec=2&auto_partials=false"
)
CLIENT_EMIT_SEC = 0.7  # как часто слать {"event":"flush"} для промежуточных текстов

print("Подсказка: после окончания фразы нажмите Enter. Во время речи partial будут печататься автоматически.")

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
            pcm16 = (x * 32767.0).astype(np.int16, copy=False).tobytes()  # float -> PCM16 bytes
            await ws.send(pcm16)


class IncrementalAggregator:
    """
    Простая «сшивка»: на каждом partial добавляем к аккумулятору только хвост,
    которого ещё не было (по максимальному суффикс/префикс перекрытию).
    Так можно переживать ограничение window_sec на сервере.
    """
    def __init__(self) -> None:
        self._acc: str = ""
        self._last_printed: str = ""

    @staticmethod
    def _stitch(prev: str, new: str) -> str:
        if not new:
            return prev
        # убираем начальные пробелы у новой порции, иначе получаются двойные пробелы
        new = new.lstrip()
        if not prev:
            return new
        # ищем максимальное k: prev.endswith(new[:k])
        maxk = min(len(prev), len(new))
        k = 0
        for i in range(maxk, -1, -1):
            if prev.endswith(new[:i]):
                k = i
                break
        tail = new[k:]
        # если нужно — вставим пробел между словами
        if tail and (not prev.endswith(" ")) and (not tail.startswith(" ")):
            tail = " " + tail
        return prev + tail

    def feed_partial(self, s: str) -> str:
        self._acc = self._stitch(self._acc, s)
        return self._acc

    def finalize(self) -> str:
        out = self._acc.strip()
        self._acc = ""
        self._last_printed = ""
        return out


async def ws_receiver(ws, final_queue: "asyncio.Queue[str]", agg: IncrementalAggregator):
    """
    Постоянно читаем сообщения от сервера и:
    - печатаем «partial: ...» с агрегированным текстом;
    - кладём финальный агрегированный текст в final_queue.
    """
    while True:
        msg = await ws.recv()
        if isinstance(msg, (bytes, bytearray)):
            # бинарные фреймы нам неинтересны — это мы же и шлём
            continue
        try:
            data = json.loads(msg)
        except Exception:
            continue

        t = data.get("type")
        if t == "partial":
            text = data.get("text", "") or ""
            combined = agg.feed_partial(text)
            print("partial:", combined)
        elif t == "final":
            text = data.get("text", "") or ""
            combined = agg.feed_partial(text)  # на всякий случай «доклеим» последние символы
            final_text = agg.finalize()
            print("FINAL:", final_text)
            await final_queue.put(final_text)
        elif t == "error":
            detail = data.get("detail", "")
            print("WS error:", detail)
        elif t == "ready":
            # можно распарсить и показать параметры
            print("WS READY:", data)
        elif t == "ok":
            # ответ на reset — можно игнорировать
            pass
        else:
            # неизвестные типы пропускаем
            pass


async def ws_flusher(ws, interval: float, stop_event: asyncio.Event):
    """
    Периодически просим у сервера промежуточную расшифровку.
    Это надёжнее, чем auto_partials на сервере (меньше гонок).
    """
    try:
        while not stop_event.is_set():
            await asyncio.sleep(interval)
            await ws.send(json.dumps({"event": "flush"}))
    except asyncio.CancelledError:
        return


async def interactive_loop(ws):
    """
    REPL:
      - микрофон пишет в WS постоянно,
      - параллельно flusher шлёт flush,
      - при Enter: stop → ждём final из receiver → reset.
    """
    print("Подключено. Нажми Enter, если закончил фразу. Ctrl+C для выхода.")
    utt_counter = 0

    final_queue: asyncio.Queue[str] = asyncio.Queue()
    agg = IncrementalAggregator()

    # стартуем постоянный приём сообщений
    receiver_task = asyncio.create_task(ws_receiver(ws, final_queue, agg))

    # flusher для partial
    stop_flusher = asyncio.Event()
    flusher_task = asyncio.create_task(ws_flusher(ws, CLIENT_EMIT_SEC, stop_flusher))

    try:
        while True:
            # ждём Enter от пользователя (не блокируя WS)
            await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            utt_counter += 1
            utt_id = f"utt{utt_counter}"

            # Запрашиваем финальную расшифровку
            await ws.send(json.dumps({"event": "stop", "utt_id": utt_id}))

            # Ждём финальный агрегированный текст от receiver
            final_text = await final_queue.get()
            print(f"[{utt_id}] FINAL_AGG:", final_text)

            # Сбросить буфер на сервере и в агрегаторе уже сделан (агрегатор финализирован),
            # но команде reset всё равно быть — для чистоты протокола:
            await ws.send(json.dumps({"event": "reset", "utt_id": utt_id}))
    finally:
        stop_flusher.set()
        flusher_task.cancel()
        receiver_task.cancel()
        for t in (flusher_task, receiver_task):
            try:
                await t
            except asyncio.CancelledError:
                pass


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

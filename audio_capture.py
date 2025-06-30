import sounddevice as sd
import numpy as np
import asyncio
from typing import Callable, Optional

SAMPLE_RATE: int = 16000  # Частота дискретизации, Hz


def record_noise_profile(duration: float = 2.0) -> np.ndarray:
    """
    Запись фонового шума длительностью duration секунд.
    Возвращает float32-массив нормализованного шума в диапазоне [-1..1].
    """
    # Запись int16
    raw_int16 = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype=np.int16)
    sd.wait()
    # Конвертация в float32 [-1..1]
    noise = raw_int16.flatten().astype(np.float32) / 32768.0
    return noise


def record_audio(duration: float) -> np.ndarray:
    """
    Запись аудио длительностью duration секунд.
    Возвращает float32-массив нормализованного сигнала в диапазоне [-1..1].
    """
    raw_int16 = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype=np.int16)
    sd.wait()
    audio = raw_int16.flatten().astype(np.float32) / 32768.0
    return audio


async def record_audio_with_countdown(
        duration: float,
        countdown_callback: Optional[Callable[[int], None]] = None
) -> np.ndarray:
    """
    Асинхронная запись аудио с выводом обратного отсчёта.
    countdown_callback(remaining_seconds) вызывается каждую секунду.
    Возвращает numpy-массив float32 нормализованного аудио.
    """
    loop = asyncio.get_running_loop()
    # Запуск блокирующей записи в пуле потоков
    record_task = loop.run_in_executor(None, record_audio, duration)

    async def _countdown():
        remaining = int(duration)
        while remaining > 0:
            if countdown_callback:
                countdown_callback(remaining)
            await asyncio.sleep(1)
            remaining -= 1

    # Параллельно выполняем запись и таймер
    await asyncio.gather(_countdown(), record_task)
    return record_task.result()


def stream_audio(
        callback: Callable[[np.ndarray], None],
        block_duration: float = 1.0
) -> None:
    """
    Стриминг аудио из микрофона.
    callback(chunk) вызывается для каждого блока длительностью block_duration.
    chunk - float32-массив нормализованного аудио.
    """
    blocksize = int(block_duration * SAMPLE_RATE)

    def _audio_callback(indata, frames, time, status):
        # indata: ndarray shape (frames, channels), dtype=int16
        audio = indata[:, 0].astype(np.float32) / 32768.0
        callback(audio)

    with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=blocksize,
            dtype='int16',
            channels=1,
            callback=_audio_callback
    ):
        # Блокируем основной поток, чтобы стрим работал непрерывно
        threading.Event().wait()

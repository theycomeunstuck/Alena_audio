import sounddevice as sd
import numpy as np

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



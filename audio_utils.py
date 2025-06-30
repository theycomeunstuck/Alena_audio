# audio_utils.py
import numpy as np
import sounddevice as sd
import noisereduce as nr
import soundfile as sf
from config import SAMPLE_RATE, NOISE_DURATION, TARGET_DBFS

def normalize_rms(y: np.ndarray, target_dBFS: float = TARGET_DBFS) -> np.ndarray:
    rms = np.sqrt(np.mean(y**2))
    target = 10**(target_dBFS / 20)
    scale = (target / (rms + 1e-9)) if rms > 0 else 1.0
    return np.clip(y * scale, -1.0, 1.0)

def record_noise_profile() -> np.ndarray:
    print(f"[1] Запись {NOISE_DURATION}s фонового шума…")
    buf = sd.rec(int(NOISE_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=1, dtype=np.int16)
    sd.wait()
    noise = (buf.flatten().astype(np.float32) / 32768.0)
    return noise

def record_audio(duration: float) -> np.ndarray:
    audio_int16 = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                         channels=1, dtype=np.int16)
    sd.wait()
    return audio_int16.flatten().astype(np.float32) / 32768.0

def reduce_and_normalize(y: np.ndarray, noise_prof: np.ndarray) -> np.ndarray:
    clean = nr.reduce_noise(
        y=y, y_noise=noise_prof, sr=SAMPLE_RATE,
        stationary=False, prop_decrease=0.85,
        n_fft=512, win_length=512, hop_length=256
    )
    return normalize_rms(clean)

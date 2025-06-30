# train_speaker.py
import os
import numpy as np
import soundfile as sf
from config import REFERENCE_FILE, REFERENCE_FILE_WAV, NOISE_DURATION, SPK_WINDOW_S, STEP_S, SAMPLE_RATE, device
from audio_utils import record_audio, normalize_rms, reduce_and_normalize
from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize


def train_user_voice():
    print("[TRAIN] Запись эталона 15с...")
    audio = record_audio(15)

    # профиль шума и очистка
    noise_prof = audio[:NOISE_DURATION * SAMPLE_RATE]
    clean = reduce_and_normalize(audio, noise_prof)

    # сохранение WAV
    sf.write(REFERENCE_FILE_WAV, clean, SAMPLE_RATE)
    print(f"[TRAIN] Чистая запись сохранена как '{REFERENCE_FILE_WAV}'")

    # разбивка на сегменты и извлечение эмбеддингов
    win = SPK_WINDOW_S * SAMPLE_RATE
    step = STEP_S * SAMPLE_RATE
    encoder = VoiceEncoder(device=device)
    embeds = []
    for start in range(0, len(clean) - win + 1, step):
        seg = clean[start:start + win]
        embeds.append(encoder.embed_utterance(seg))

    if not embeds:
        raise RuntimeError("Не удалось получить сегментов для энролмента")

    avg_emb = normalize(np.mean(np.vstack(embeds), axis=0).reshape(1, -1))[0]
    np.save(REFERENCE_FILE, avg_emb)
    print(f"[TRAIN] Эталонный вектор сохранён в '{REFERENCE_FILE}'")


# TODO: [asr] спокойная музыка, аплодисменты. Может быть это игнорить или пропадёт, если добавить speechbrain?
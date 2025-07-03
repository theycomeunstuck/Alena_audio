# train_speaker.py
import os
import numpy as np
import soundfile as sf
import torch
from audio_enhancement import noise_suppresion_SB, to_tensor
from config import REFERENCE_FILE, REFERENCE_FILE_WAV, TRAIN_USER_VOICE_S, SPK_WINDOW_S, STEP_S, SAMPLE_RATE, device, speech_verification_model
from audio_utils import record_audio, normalize_rms
from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize


def train_user_voice(): #todo: добавить audio enhancement ! SB
    try:
        # сохраняет эмбеддинг
        print(f"[TRAIN] Запись эталона {TRAIN_USER_VOICE_S}с...")
        audio = record_audio(TRAIN_USER_VOICE_S).astype(np.float32)
        clean = noise_suppresion_SB(audio) #np.ndarray

        # сохранение WAV
        sf.write(REFERENCE_FILE_WAV, clean, SAMPLE_RATE)
        print(f"[TRAIN] Чистая запись сохранена как '{REFERENCE_FILE_WAV}'")

        # разбивка на сегменты и извлечение эмбеддингов

        wav_t = to_tensor(clean)

        with torch.no_grad():
            emb = speech_verification_model.encode_batch(wav_t)  # [1, D]
            emb = emb.squeeze(0)  # [D]

        # 6. L2-нормализация
        emb = emb / emb.norm(p=2)
        print(emb.shape)
        np.save(REFERENCE_FILE, emb)
        print(f"[TRAIN] Эталонный вектор сохранён в '{REFERENCE_FILE}'")
        # todo: Возвращение в код и продолжение работы, когда мы добавили какого-то пользователя в бд

    except Exception:
        print(f"Ошибка в файле {__file__}")
        raise



# train_speaker.py
import numpy as np
import soundfile as sf
import torch
from core.audio_enhancement import Audio_Enhancement, to_tensor
from core.config import REFERENCE_FILE, REFERENCE_FILE_WAV, TRAIN_USER_VOICE_S, SAMPLE_RATE, speech_verification_model
from core.audio_capture import record_audio



def train_user_voice():
    try:
        print(f"[TRAIN] Запись эталона {TRAIN_USER_VOICE_S}с...")
        audio = record_audio(TRAIN_USER_VOICE_S).astype(np.float32)
        clean = Audio_Enhancement(audio).noise_suppression() # normalized np.ndarray

        sf.write(REFERENCE_FILE_WAV, clean, SAMPLE_RATE)
        print(f"[TRAIN] Чистая запись сохранена как '{REFERENCE_FILE_WAV}'")

        # разбивка на сегменты и извлечение эмбеддингов
        wav_t = to_tensor(clean)
        with torch.no_grad():
            emb = speech_verification_model.encode_batch(wav_t)  # [1, D]
            emb = emb.squeeze(0)  # [D]

        # 6. L2-нормализация
        emb = emb / emb.norm(p=2)

        if emb.shape != 192:
            target_dim = min(emb.shape[1], 192)
            emb = emb[:, :target_dim]

        np.save(REFERENCE_FILE, emb)
        print(f"[TRAIN] Эталонный вектор сохранён в '{REFERENCE_FILE}'")

        # todo: Возвращение в код и продолжение работы, когда мы добавили какого-то пользователя в бд

    except Exception:
        print(f"Ошибка в файле {__file__}")
        raise



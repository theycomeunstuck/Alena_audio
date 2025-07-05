# verify.py
import os
import numpy as np
import webrtcvad
import sounddevice as sd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from audio_enhancement import Audio_Enhancement, to_tensor
from config import (REFERENCE_FILE, SAMPLE_RATE, FRAME_MS, sim_threshold,
                    VAD_AGGR_MODE, SPK_WINDOW_S, STEP_S,
                    MIN_VOICE_RATIO, MAX_ASR_FAILURES, device)
from audio_utils import  normalize_rms
from resemblyzer import VoiceEncoder
from config import asr_model


def verify_speaker():
    if not os.path.exists(REFERENCE_FILE):
        print("Нет файла-референса. Сначала вызовите train_user_voice()")
        return False

    ref_emb = normalize(np.load(REFERENCE_FILE).reshape(1, -1))


    vad = webrtcvad.Vad(VAD_AGGR_MODE) # vad
    frame_size = int(FRAME_MS * SAMPLE_RATE / 1000) * 2

    spk_buf = np.zeros(0, dtype=np.float32)
    win_samples = SPK_WINDOW_S * SAMPLE_RATE
    step_samples = STEP_S * SAMPLE_RATE
    asr_fail = 0
    counter = 0
    enhancer = Audio_Enhancement(np.zeros(1), None)
    verifier = None
    print("[VERIFY] Стриминг, Ctrl+C чтобы выйти")
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=step_samples,
                           dtype=np.float32, channels=1) as stream:
        try:
            while True:
                data, _ = stream.read(step_samples)

                # VAD
                voiced, total = [], 0
                int16_frame = (data * 32767).astype(np.int16)
                for i in range(0, len(int16_frame), frame_size):
                    frame_i16  = int16_frame[i:i+frame_size]
                    if len(frame_i16)<frame_size: break
                    total += 1
                    if vad.is_speech(frame_i16 , SAMPLE_RATE):
                        voiced.append(data[i:i+frame_size].squeeze()) # original fl32

                if total == 0 or (len(voiced)/total) < MIN_VOICE_RATIO:
                    asr_fail += 1
                    if asr_fail >= MAX_ASR_FAILURES:
                        spk_buf = np.zeros(0, dtype=np.float32)
                        asr_fail = 0
                    continue
                asr_fail = 0

                # очистка и накопление
                audio = np.concatenate(voiced) #float32
                spk_buf = np.concatenate([spk_buf, audio])[-win_samples:]
                enhancer.audio = to_tensor(spk_buf, pad_to_min=True)
                # верификация
                if len(spk_buf) >= win_samples:
                    clean_audio = enhancer.noise_suppression().cpu().numpy().squeeze()

                    # Инициализация или обновление verifier
                    if verifier is None:
                        verifier = Audio_Enhancement(clean_audio, ref_emb)
                    else:
                        verifier.audio = to_tensor(clean_audio, pad_to_min=True)
                        verifier.audio_ref = to_tensor(ref_emb, pad_to_min=True)

                    sim = verifier.speech_verification()

                    # import soundfile as sf
                    # sf.write(f"debug_wav/clean_audio{counter}.wav", clean_audio, SAMPLE_RATE)
                    # counter += 1

                    result = asr_model.transcribe(clean_audio, language="ru")
                    text = result["text"].strip().lower()
                    print(f"[VERIFY]: similarity = {sim:.3f}")
                    print(f"[ASR]: {text}")

                    if sim > sim_threshold and "стоп" in text:
                        print(">>> Команда СТОП получена. Завершаю.")
                        break
                    elif sim > sim_threshold:
                        print(">>> Speaker VERIFIED!")
        except KeyboardInterrupt:
            print("\n[VERIFY] Остановлено пользователем")
    return False
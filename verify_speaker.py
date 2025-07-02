# verify.py
import os
import numpy as np
import webrtcvad
import sounddevice as sd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from audio_enhancement import noise_suppresion_SB
from config import (REFERENCE_FILE, SAMPLE_RATE, FRAME_MS, VAD_AGGR_MODE, SPK_WINDOW_S, STEP_S, MIN_VOICE_RATIO, MAX_ASR_FAILURES, device)
from audio_utils import record_noise_profile, normalize_rms, reduce_and_normalize
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
    encoder = VoiceEncoder(device=device)
    asr_fail = 0

    print("[VERIFY] Стриминг, Ctrl+C чтобы выйти")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=step_samples,
                           dtype=np.int16, channels=1) as stream:
        try:
            while True:
                data, _ = stream.read(step_samples)
                raw = bytes(data)

                # VAD
                voiced, total = bytearray(), 0
                for i in range(0, len(raw), frame_size):
                    frame = raw[i:i+frame_size]
                    if len(frame)<frame_size: break
                    total += 1
                    if vad.is_speech(frame, SAMPLE_RATE):
                        voiced.extend(frame)

                if total == 0 or (len(voiced)/total) < MIN_VOICE_RATIO:
                    asr_fail += 1
                    if asr_fail >= MAX_ASR_FAILURES:
                        spk_buf = np.zeros(0, dtype=np.float32)
                        asr_fail = 0
                    continue
                asr_fail = 0

                # очистка и накопление
                audio = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32)/32768.0
                spk_buf = np.concatenate([spk_buf, audio])[-win_samples:]

                # верификация
                if len(spk_buf) >= win_samples:
                    clean_audio = noise_suppresion_SB(spk_buf)
                    emb = normalize(encoder.embed_utterance(clean_audio).reshape(1, -1))
                    sim = cosine_similarity(ref_emb, emb)[0,0]

                    result = asr_model.transcribe(clean_audio, language="ru")
                    text = result["text"].strip().lower()
                    print(f"[VERIFY] similarity = {sim:.3f}")
                    print(f"[ASR] {text}")

                    if sim > 0.75 and "стоп" in text: # todo: переделать под sb, очень высоко оценивает чужие голоса
                        print(">>> Команда СТОП получена. Завершаю.")
                        break
                    elif sim > 0.75:
                        print(">>> Speaker VERIFIED!")
        except KeyboardInterrupt:
            print("\n[VERIFY] Остановлено пользователем")
    return False
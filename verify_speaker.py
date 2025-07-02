# verify.py
import os
import numpy as np
import webrtcvad
import sounddevice as sd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from audio_enhancement import noise_suppresion_SB, speech_separation_SB, speech_verification_SB
from config import (REFERENCE_FILE, SAMPLE_RATE, FRAME_MS,
                    VAD_AGGR_MODE, SPK_WINDOW_S, STEP_S,
                    MIN_VOICE_RATIO, MAX_ASR_FAILURES, device)
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
    counter = 0 #todo:
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
                    if len(frame_i16 )<frame_size: break
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
                # audio = np.frombuffer(bytes(voiced), dtype=np.int16).astype(np.float32)/32768.0
                audio = np.concatenate(voiced)
                spk_buf = np.concatenate([spk_buf, audio])[-win_samples:]

                # верификация
                if len(spk_buf) >= win_samples:
                    #todo: сейчас опция 1
                    # todo: Проверить как будет работать, если: 1) очистка, speaker; 2) speaker; очистка    ?
                    import soundfile as sf
                    sf.write(f"debug_wav/fully_naked_{counter}.wav", spk_buf, SAMPLE_RATE)
                    counter += 1

                    # TODO: ДО СЮДА С АУДИО ВСЁ НОРМ
                    #  КОПАЙ НА ШУМЫ, ВИДИМО

                    # TODO: 2 SPEAKER RECOGNIZED ALWAYS
                    clean_audio = noise_suppresion_SB(spk_buf)
                    speech_separation = speech_separation_SB(clean_audio)
                    n_spk = speech_separation.size(1)
                    print(f"{__file__}: Распознано {n_spk} speaker`ов")
                    for speaker in range(n_spk): #speaker 1 - speakers
                        source = speech_separation[:, speaker] # Речь {speaker} пользователя
                        # clean_audio = noise_suppresion_SB(speech_separation[source])

                        emb = normalize(encoder.embed_utterance(source).reshape(1, -1))
                        sim = cosine_similarity(ref_emb, emb)[0,0]

                        result = asr_model.transcribe(source, language="ru")
                        text = result["text"].strip().lower()
                        print(f"[VERIFY]. Speaker {speaker + 1}: similarity = {sim:.3f}")
                        print(f"[ASR] Speaker {speaker + 1}: {text}")

                        if sim > 0.75 and "стоп" in text: # todo: переделать под sb, очень высоко оценивает чужие голоса
                            print(">>> Команда СТОП получена. Завершаю.")
                            break
                        elif sim > 0.75:
                            print(">>> Speaker VERIFIED!")
        except KeyboardInterrupt:
            print("\n[VERIFY] Остановлено пользователем")
    return False
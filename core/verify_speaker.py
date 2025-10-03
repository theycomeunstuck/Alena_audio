# verify.py
import os
import numpy as np
import webrtcvad
import sounddevice as sd
from sklearn.preprocessing import normalize
from core.audio_enhancement import Audio_Enhancement, to_tensor, handle_exceptions
from core.config import (REFERENCE_FILE, SAMPLE_RATE, FRAME_MS, sim_threshold,
                    VAD_AGGR_MODE, SPK_WINDOW_S, STEP_S,
                    MIN_VOICE_RATIO, MAX_ASR_FAILURES, asr_model)



class Speaker_Processing:

    @handle_exceptions
    def __init__(self):

        self.enhancer = Audio_Enhancement(np.zeros(1), None)
        self.verifier = None


    @handle_exceptions
    def verify_speaker(self):
        if not os.path.exists(REFERENCE_FILE):
            print("Нет файла-референса. Сначала вызовите train_user_voice()")
            return False
        self.ref_emb = normalize(np.load(REFERENCE_FILE).reshape(1, -1)) # todo: переписать. Оно должно стоять не здесь, и не в init: потому что в будущем будет сортировка по пользователям.

        vad = webrtcvad.Vad(VAD_AGGR_MODE) # vad
        frame_size = int(FRAME_MS * SAMPLE_RATE / 1000) * 2
        spk_buf = np.zeros(0, dtype=np.float32)
        win_samples = SPK_WINDOW_S * SAMPLE_RATE
        step_samples = STEP_S * SAMPLE_RATE
        asr_fail = 0
        print("[VERIFY] Стриминг, Ctrl+C чтобы выйти")
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=step_samples,
                               dtype=np.float32, channels=1) as stream:
            try:
                while True:
                    data, _ = stream.read(step_samples)
                    '''
                    todo:  спрятать vad под функцию . а ещё лучше - под другой класс, который будет отвечать за стриминг
                    todo: переписать vad. даёт пройти посторонним звукам, не обрезает их. из-за этого asr может выдать
                    аплодисменты, "продолжение следует" и "субтитры сделал dimatorzok" и прочее
                    '''
                    # VAD. 
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
                    self.enhancer.audio = to_tensor(spk_buf, pad_to_min=True)
                    # верификация
                    if len(spk_buf) >= win_samples:
                        clean_audio = spk_buf
                        self.audio_processing(clean_audio)

            except KeyboardInterrupt:
                print("\n[VERIFY] Остановлено пользователем")
        return False

    def audio_processing(self, speaker_audio, speaker=0):
        # Инициализация или обновление verifier
        if self.verifier is None:
            self.verifier = Audio_Enhancement(speaker_audio, self.ref_emb)
        else:
            self.verifier.audio = to_tensor(speaker_audio, pad_to_min=True)
            self.verifier.audio_ref = to_tensor(self.ref_emb, pad_to_min=True) # todo: в будущем вынести проверку референсов в отдельную функцию

        sim = self.verifier.speech_verification()

        result = asr_model.transcribe(speaker_audio, language="ru")
        text = result["text"].strip().lower()
        print(f"[VERIFY]: similarity = {sim:.3f}")

        _ = ""
        if speaker != 0: _ = f"({speaker}) " # Обработка логики, когда это n-ый speaker
        print(f"[ASR]: {_}{text}")


        if sim >= sim_threshold and "стоп" in text:
            print(">>> Команда СТОП получена. Завершаю.")
            raise KeyboardInterrupt
            # return "break" #
        elif sim > sim_threshold:
            print(">>> Speaker VERIFIED!")
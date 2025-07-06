# verify.py
import os
import numpy as np
import webrtcvad
import sounddevice as sd

from sklearn.preprocessing import normalize
from audio_enhancement import Audio_Enhancement, to_tensor, to_numpy, handle_exceptions
from config import (REFERENCE_FILE, SAMPLE_RATE, FRAME_MS, sim_threshold,
                    VAD_AGGR_MODE, SPK_WINDOW_S, STEP_S,
                    MIN_VOICE_RATIO, MAX_ASR_FAILURES, diarization_pipeline)
from config import asr_model

import soundfile as sf


class Stream_Processing:

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGR_MODE) # vad
        self.frame_size = int(FRAME_MS * SAMPLE_RATE / 1000) * 2
        self.spk_buf = np.zeros(0, dtype=np.float32)
        self.win_samples = SPK_WINDOW_S * SAMPLE_RATE
        self.step_samples = STEP_S * SAMPLE_RATE
        self.asr_fail = 0
        self.enhancer = Audio_Enhancement(np.zeros(1), None)


    def streaming(self):
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=self.step_samples,
                                dtype=np.float32, channels=1) as stream:
                while True:
                    data, _ = stream.read(self.step_samples)
                    '''
                    todo:  спрятать vad под функцию . а ещё лучше - под другой класс, который будет отвечать за стриминг
                    todo: переписать vad. даёт пройти посторонним звукам, не обрезает их. из-за этого asr может выдать
                    аплодисменты, "продолжение следует" и "субтитры сделал dimatorzok" и прочее
                    '''
                    # VAD.
                    voiced, total = [], 0
                    int16_frame = (data * 32767).astype(np.int16)
                    for i in range(0, len(int16_frame), self.frame_size):
                        frame_i16 = int16_frame[i:i + self.frame_size]
                        if len(frame_i16) < self.frame_size: break
                        total += 1
                        if self.vad.is_speech(frame_i16, SAMPLE_RATE):
                            voiced.append(data[i:i + self.frame_size].squeeze())  # original fl32
                    if total == 0 or (len(voiced) / total) < MIN_VOICE_RATIO:
                        self.asr_fail += 1
                        if self.asr_fail >= MAX_ASR_FAILURES:
                            self.spk_buf = np.zeros(0, dtype=np.float32)
                            self.asr_fail = 0
                        continue
                    self.asr_fail = 0

                    # очистка и накопление
                    audio = np.concatenate(voiced)  # float32
                    self.spk_buf = np.concatenate([self.spk_buf, audio])[-self.win_samples:]
                    yield self.spk_buf
        except Exception as e:
            print(f"[ERROR] Ошибка в методе '{__file__}': {e}")

class Speaker_Processing:

    def __init__(self):
        # self.verifier = None
        self.ref_emb = normalize(np.load(REFERENCE_FILE).reshape(1, -1))
        self.enhancer = Audio_Enhancement(np.zeros(1), self.ref_emb)

        self.data_stream = Stream_Processing()

        self.win_samples = SPK_WINDOW_S * SAMPLE_RATE
        self.step_samples = STEP_S * SAMPLE_RATE

        self.counter = 0

    #@handle_exceptions #todo: нужен ли он здесь?
    def verify_speaker(self):
        if not os.path.exists(REFERENCE_FILE):
            print("Нет файла-референса. Сначала вызовите train_user_voice()")
            return False


        print("[VERIFY] Стриминг, Ctrl+C чтобы выйти")
        cmd = ''
        for spk_buf in self.data_stream.streaming():

            self.enhancer.audio = to_tensor(spk_buf, pad_to_min=True)
            if len(spk_buf) < self.win_samples: continue

            cnt_spk = len(set((diarization_pipeline({
                'waveform': self.enhancer.audio,
                'sample_rate': SAMPLE_RATE
            }).labels())))
            print(f'speaker counter: {cnt_spk}')

            if cnt_spk == 1:
                audio = self.enhancer.audio
                cmd = self.audio_processing(audio)
            elif cnt_spk == 0: continue
            else:
                speech_separation = self.enhancer.speech_separation()
                for Speaker_id in range(cnt_spk):
                    speaker_audio = speech_separation[Speaker_id]  # audio пользователя {speaker}
                    cmd = self.audio_processing(speaker_audio, Speaker_id+1)

            if cmd == "break":
                print("Было сказано слово СТОП. Завершение программы")
                break


    def audio_processing(self, speaker_audio, speaker=0):
        # Инициализация или обновление verifier
        # if self.verifier is None:
        #     self.verifier = Audio_Enhancement(speaker_audio, self.ref_emb)
        # else:
        #     self.verifier.audio = to_tensor(speaker_audio, pad_to_min=True).squeeze()
        #     self.verifier.audio_ref = to_tensor(self.ref_emb, pad_to_min=True) # todo: может быть и не надо сувать в to_tensor. единожды. а. | Хочется, можно, нужно поставить в init, чтобы считало всего один раз

        self.enhancer.audio = to_tensor(speaker_audio, pad_to_min=True).squeeze()

        sim = self.enhancer.speech_verification()

        result = asr_model.transcribe(self.enhancer.audio, language="ru")
        text = result["text"].strip().lower()
        print(f"[VERIFY]: similarity = {sim:.3f}")

        _ = ""
        if speaker != 0: _ = f"({speaker}) "
        print(f"[ASR]: {_}{text}")


        if sim >= sim_threshold and "стоп" in text:
            print(">>> Команда СТОП получена. Завершаю.")
            return "break" # todo: как мне сделать break?..
        elif sim > sim_threshold:
            print(">>> Speaker VERIFIED!")

        sf.write(f"debug_wav/clean_audio{self.counter}.wav", to_numpy(self.enhancer.audio), SAMPLE_RATE)
        self.counter += 1
        return None
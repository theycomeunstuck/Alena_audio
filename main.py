# main.py
import os
from config import REFERENCE_FILE
from train_speaker import train_user_voice
from verify_speaker import Speaker_Processing

if __name__ == "__main__":
    if not os.path.exists(REFERENCE_FILE):
        train_user_voice()
    else:
        sp = Speaker_Processing()
        sp.verify_speaker()



# todo: add voice enhancement (Echo) to train_speaker.py

# TODO: [asr] спокойная музыка, аплодисменты.
#  Может быть это игнорить или пропадёт, если добавить speechbrain?

# TOOD [TASK]:
# TOOD [TASK]: Оценка метрик. Я не помню что это значит. Просто анализ? то есть отчётная часть
# TOOD [TASK]: Voice separation
# TODO [TASK]: fully tts


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



# TOOD [TASK]: Voice separation



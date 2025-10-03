# main.py
import os
from core.config import REFERENCE_FILE
from core.train_speaker import train_user_voice
from core.verify_speaker import Speaker_Processing

if __name__ == "__main__":
    if not os.path.exists(REFERENCE_FILE):
        train_user_voice()
    else:
        sp = Speaker_Processing()
        sp.verify_speaker()



# TOOD [TASK]: Voice separation



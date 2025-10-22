#client_examples/call_api_TTS.py
import requests

BASE = "http://127.0.0.1:8000"

# 1) Клонирование
with open("speaker.mp3", "rb") as f:
    r = requests.post(f"{BASE}/tts/clone", files={"file": ("speaker.mp3", f, "audio/mpeg")})
print("voice_id:", r.json()["voice_id"])

# 2) TTS c голосом по умолчанию (WAV)
r = requests.post(f"{BASE}/tts", json={"text": "Здравствуйте! Это проверка синтеза."})
open("out_default.wav", "wb").write(r.content)

# 3) TTS с выбранным голосом (MP3)
vid = r.json().get("voice_id", "")  # todo: вставить склонированный голос
payload = {"text": "Это мой клонированный голос.", "voice_id": vid, "format": "mp3"}
r = requests.post(f"{BASE}/tts", json=payload)
open("out_clone.mp3", "wb").write(r.content)

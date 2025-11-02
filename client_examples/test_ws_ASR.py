# client_examples/test_ws_ASR.py
import asyncio, json, wave, websockets
from pathlib import Path
from app.services.audio_utils import load_and_resample

WS_URL = "ws://127.0.0.1:8000/ws/asr?language=ru&sample_rate=16000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAV_PATH = PROJECT_ROOT / "tests/samples/ru_sample.wav"  # 16k mono PCM16; относительный путь от корня проекта


def wav_to_pcm16(path: Path | str) -> bytes:
    p = Path(path)  # на случай, если пришла строка
    with open(p, "rb") as fh:
        with wave.open(fh, "rb") as w:
            assert w.getnchannels() == 1, "нужен mono"
            assert w.getsampwidth() == 2, "нужен 16-bit PCM"
            assert w.getframerate() == 16000, "нужен 16kHz"
            return w.readframes(w.getnframes())

async def main():
    async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
        print("server:", await ws.recv())  # ready
        pcm = wav_to_pcm16(WAV_PATH)
        chunk = 32000  # ~1с при 16k/mono/16-bit
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i+chunk])
            await asyncio.sleep(0.05)

        await ws.send(json.dumps({"event":"flush"}))
        print("partial:", await ws.recv())

        await ws.send(json.dumps({"event":"stop"}))
        # может прилететь ещё один partial — читаем до final
        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)
            except Exception:
                print("binary?", len(msg)); continue
            print("rx:", data)
            if data.get("type") == "final":
                break

asyncio.run(main())

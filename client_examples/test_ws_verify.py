#client_examples/test_ws_verify
import asyncio, websockets, json, numpy as np, struct
from core.config import SAMPLE_RATE, sim_threshold
URI=f"ws://localhost:8000/ws/speaker/verify?sample_rate={SAMPLE_RATE}&channels=1&top_k=2&sim_threshold={sim_threshold}&emit_interval_ms=500"
def float_to_pcm16_bytes(x):
    x=np.clip(x,-1,1); x=(x*32767).astype(np.int16); return x.tobytes()
async def main():
    async with websockets.connect(URI) as ws:
        print("READY:", await ws.recv())
        # 1 сек шума
        noise=np.random.uniform(-0.1,0.1,SAMPLE_RATE).astype(np.float32)
        await ws.send(float_to_pcm16_bytes(noise))
        # ждём несколько partial
        for _ in range(4):
            msg = await ws.recv()
            print("MSG:", msg)
        # стоп
        await ws.send(json.dumps({"event":"stop"}))
        print("FINAL:", await ws.recv())
asyncio.run(main())

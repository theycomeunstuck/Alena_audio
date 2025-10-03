from fastapi import FastAPI
from app.api import routes_health, routes_files, routes_audio, routes_speaker, routes_ws, routes_TTS

app = FastAPI(
    title="Audio Core API",
    description="REST + WebSocket API: шумоподавление, ASR, верификация спикера, тренировка эталона, TTS.",
    version="1.0.1",
)

app.include_router(routes_health.router)
app.include_router(routes_files.router)
app.include_router(routes_audio.router)
app.include_router(routes_speaker.router)
app.include_router(routes_ws.router)
app.include_router(routes_TTS.router)



# app — веб-слой (FastAPI)

Транспорт: REST + WebSocket. Веб-слой **не меняет** поведение `core/`, а только адаптирует под HTTP/WS.

## Структура
- `main.py` — сборка FastAPI и подключение роутеров.
- `api/`
  - `routes_health.py` — `GET /health`
  - `routes_files.py` — `POST /files/upload`, `GET /files/download/{filename}`
  - `routes_audio.py` — `POST /audio/enhance`, `POST /audio/transcribe?language=...`
  - `routes_speaker.py` — `POST /speaker/verify` (поддерживает `probe+reference.wav`, **либо** `probe+reference_npy`, **либо** fallback на `core/reference.npy`), `POST /speaker/train/microphone`
  - `routes_ws.py` — `WS /ws/asr?language=..&sample_rate=..` (PCM16 mono, события `flush`/`stop`)
- `services/`
  - `audio_service.py` — работа с файлами и стримингом ASR (буфер, окно/emit, вызов `asr_model.transcribe`)
  - `audio_utils.py` — `load_and_resample(path) -> 1D np.float32 @ 16kHz` (torchaudio)
  - `speaker_service.py` — верификация: через `Audio_Enhancement` (wav-wav) и `verify_with_ref_embedding` (wav-npy)
- `models/` — Pydantic-схемы ответов
- `storage/` — временные файлы загрузок

## Документация
Смотри `./index.html` (открой в браузере) + Swagger `/docs`.


## Запуск
```bash
uvicorn app.main:app --reload
# Swagger: http://127.0.0.1:8000/docs

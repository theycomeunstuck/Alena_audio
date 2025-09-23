
# core/ — доменная логика

- `audio_enhancement.py` — `Audio_Enhancement`: шумоподавление и `speech_verification()` (косинус эмбеддингов ECAPA).
- `verify_speaker.py`, `train_speaker.py` — процесс верификации/тренировки эталона (микрофонный сценарий).
- `audio_utils.py`, `audio_capture.py` — низкоуровневые операции с аудио.
- `config.py` — единая конфигурация:
  - `SAMPLE_RATE` (16_000), `device`, пути `MODELS_DIR`/кэши HuggingFace/torch,
  - параметры partial стриминга: `ASR_WINDOW_SEC`, `ASR_EMIT_SEC`,
  - опции верификации (порог, длительность записи эталона и т.п.),
  - инициализация моделей (без прогрева): Whisper (`asr_model`), SpeechBrain (`speech_verification_model`)

## Контракт с веб-слоем
- Веб-слой подает **только** готовые 1D `np.float32` на вход в `Audio_Enhancement`/Whisper.
- Ничего не меняем внутри `core/` без причины — это «источник истины».








---
## used stack:
pytorch, whisper, f5_tts, speechbrain.

Путь до последней модели TTS на сервере: E:\PycharmProjects\AI-Teach\Speech\tts\model.safetensors

Без в safetensors: E:\PycharmProjects\AI-Teach\Speech\libs\F5-TTS\ckpts\f5-tts_ru_en

Файла для запуска tts из кода не сделан.

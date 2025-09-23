# client_examples/ — примеры клиентов

Примеры того, как теперь обращаться к функционалу **только через API**, а не через прямые вызовы Python:

- `call_api_audio.py` — примеры `enhance` и `transcribe` (HTTP).
- `call_api_speaker.py` — `verify`/`train` (HTTP).
- `ws_asr_mic.py` — потоковый ASR через WebSocket (микрофон → partial/final).

Смотри также `../docs/index.html` для детальной спецификации.

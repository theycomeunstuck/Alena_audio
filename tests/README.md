# tests/ — интеграционные и юнит-тесты (pytest)

## Запуск:
### Быстрый тест
```bash
pytest -q -s
```

### Глубокий тест
```bash 
pytest -q -m "deep"
```

### Всё подряд (рекомендуется)
```bash 
pytest -v -s -m "deep or not deep"
```


## Что проверяем:

  - корректный ресэмплинг и сведение в моно (audio_utils),  
  
  - REST эндпоинты (upload/download/enhance/transcribe/verify),  
  
  - WebSocket /ws/asr (принимает, отдает partial/final, не ломается),  
  
  - накопление и обрезку буфера StreamingASRSession,  
  
  - устойчивость к невалидному JSON и странным входам.  
  
  - В тяжёлых местах используются моки (ASR/Enhancement), чтобы тесты были быстрыми и воспроизводимыми.


---
Если видишь «unknown mark: deep» — проверь, что запускаешь из корня (где лежит pytest.ini).
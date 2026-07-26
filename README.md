<!-- README.md -->
# Alena / TeachCopilot

Две части проекта:

- **Audio Core API** — распознавание речи, TTS, идентификация говорящего.
  👉 [Открыть документацию (docs/index.md)](./docs/index.md)
- **TeachCopilot** — ИИ-репетитор: карточки учеников, RAG и подготовка
  индивидуальных заданий по Выготскому (4 этап ЗБР).

## TeachCopilot: с чего начать

| Кому | Куда |
|---|---|
| Что сделано и в каком состоянии | [SUMMARY.md](SUMMARY.md) |
| Что проверить на сервере перед использованием | [TeachCopilot_RAG/docs/SERVER_CHECKLIST.md](TeachCopilot_RAG/docs/SERVER_CHECKLIST.md) |
| Репетитору: какие команды запускать до урока, после и для отчёта | [TeachCopilot_RAG/docs/HOWTO.md](TeachCopilot_RAG/docs/HOWTO.md) |
| Разработчику: правила ЗБР, контракты, устройство 4 этапа | [TeachCopilot_RAG/docs/ZPD_STAGE4.md](TeachCopilot_RAG/docs/ZPD_STAGE4.md) |
| Тому, кто продолжает работу: что не проверено и что делать дальше | [HANDOFF.md](HANDOFF.md) |
| Как заполнять карточки учеников | [learner-data/MANUAL.md](learner-data/MANUAL.md) |
| Вся документация RAG | [TeachCopilot_RAG/docs/README.md](TeachCopilot_RAG/docs/README.md) |

Быстрая проверка, что всё живо (не требует ни БД, ни LLM):

```bash
python learner-data/tools/validate.py --strict
python learner-data/tools/selftest.py
cd TeachCopilot_RAG && python scripts/check_material_coverage.py
```

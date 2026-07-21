# JSON / JSONB в векторной базе

`scripts/ingest.py --mode json` превращает каждую запись JSON в embedding и
сохраняет исходный объект без потерь в `knowledge_base.metadata` (`JSONB`).
Поддерживаются JSON-массивы и объекты с `records`, `items`, `tasks` или `data`.

```json
{"tasks":[{"task_id":"div-408-4","topic_id":"math.g4.numbers.division_by_1_2_digit","grade":"4","content":"Раздели 408 на 4 столбиком.","answer":"102","tags":["деление"]}]}
```

```bash
uv run python scripts/apply_schema.py
uv run python scripts/ingest.py --mode json --file tasks.json
```

Embedding строится по `content` (либо `question`/`text`) и `answer` (либо
`solution`). Для связи с профилями заполняйте те же `topic_id` и `grade`, что в
`learner-data/catalog/`.

# Работа с `learner-data` — JSON-first

## Что хранится сейчас

- Карточки учеников — обычные файлы `learner-data/learners/<learner_id>.json`.
- Каталог учебных тем — `learner-data/catalog/math_g3_g4.json`.
- Это **JSON**, не JSONB. JSONB — тип PostgreSQL и в JSON-first режиме не
  используется.

`legal_name` хранится только для закрытого педагогического интерфейса и состоит
из `first_name`, `last_name`, `patronymic`. Отчество может быть `""` или `"-"`.
Ни это поле, ни журнал, баллы, интересы и полная learner model не отправляются
в модель.

## Проверить данные

```bash
cd Alena_audio
python learner-data/tools/validate.py --strict
python learner-data/tools/selftest.py
```

Полностью заполненные варианты карточек и инструкция создания нового ученика:
[`../../learner-data/EXAMPLES.md`](../../learner-data/EXAMPLES.md).

## Получить безопасный контекст для одного ученика

```bash
python learner-data/tools/build_context.py ivanov-ivan
```

Пример результата:

```text
<child_safe_profile>
Имя_для_обращения: Иван
Школьный_уровень: 4 класс
Язык: russian
</child_safe_profile>

<learner_rag_context>
Текущая_цель: Деление на однозначное и двузначное число [math.g4.numbers.division_by_1_2_digit] — закрепить деление столбиком с проверкой нулей и остатка
...
</learner_rag_context>
```

В application flow сервер сначала получает проверенный `learner_id`, затем
вызывает этот builder и передаёт в LLM только его результат:

```text
authenticated session → learner_id=ivanov-ivan → build_context.py
question → JSON vector index + topic_id/grade
compact learner context + matching material → LLM answer
```

## Пример учебного JSON-файла

Пока учебные материалы также остаются файлами. Для каждого фрагмента используйте
те же `topic_id` и `grade`, что и в каталоге:

```json
{
  "task_id": "div-408-4",
  "topic_id": "math.g4.numbers.division_by_1_2_digit",
  "grade": "4",
  "content": "Раздели 408 на 4 столбиком.",
  "answer": "102",
  "difficulty": "easy",
  "tags": ["деление", "столбик"]
}
```

Полная пошаговая инструкция именно для учебных JSON — от добавления объекта до
поиска и передачи результата модели: [../knowledge-data/README.md](../knowledge-data/README.md).

## Построить локальную векторную базу из JSON

Для JSON-first режима есть файловый MVP: `scripts/json_rag.py`. Он использует
модель embedding, но хранит индекс и metadata в обычном JSON-файле — без
PostgreSQL, Docker и JSONB.

```bash
cd TeachCopilot_RAG
uv run python scripts/json_rag.py build \
  --source knowledge-data/math_g4_division_demo.json \
  --index .local-rag/math-g4.json
```

Пример результата:

```json
{"indexed": 2, "index": ".local-rag/math-g4.json"}
```

## Найти материал для вопроса

```bash
uv run python scripts/json_rag.py search \
  --index .local-rag/math-g4.json \
  --query "Как разделить 408 на 4 и не забыть ноль?" \
  --topic-id math.g4.numbers.division_by_1_2_digit \
  --grade 4
```

Пример результата:

```json
[
  {
    "score": 0.74,
    "content": "Раздели 408 на 4 столбиком.\nТеги: деление, столбик, ноль в частном\n408 : 4 = 102...",
    "metadata": {
      "task_id": "div-408-4",
      "topic_id": "math.g4.numbers.division_by_1_2_digit",
      "grade": "4",
      "difficulty": "easy"
    }
  }
]
```

`score` зависит от модели и поэтому меняется между окружениями. Не делайте
`topic_id` жёстким фильтром для каждого вопроса: сначала пробуйте его как
предпочтение, затем при нулевом результате повторите поиск без фильтра.

## Добавить новый учебный JSON

1. Добавьте объект с `task_id`, `topic_id`, `grade`, `content`, `answer` и
   `tags` в JSON-файл учебных материалов.
2. Убедитесь, что `topic_id` существует в `learner-data/catalog/`.
3. Повторите команду `build` — индекс намеренно строится заново, чтобы данные и
   embeddings не расходились.
4. Запустите `search` на 2–3 реальных вопросах ребёнка и проверьте metadata
   результата.

Текущий `pipeline/rag.py` с PostgreSQL/pgvector остаётся legacy-кодом и не
нужен для описанного JSON-first CLI-пути.

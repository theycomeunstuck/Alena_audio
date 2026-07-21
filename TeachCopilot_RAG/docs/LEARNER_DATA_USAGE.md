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

## Получить безопасный контекст для одного ученика

```bash
python learner-data/tools/build_context.py volk-08
```

Пример результата:

```text
<child_safe_profile>
Имя_для_обращения: Волк
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
authenticated session → learner_id=volk-08 → build_context.py
question → JSON учебных материалов по topic_id/grade
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

## Важное ограничение

В коде всё ещё есть старый PostgreSQL/pgvector retrieval-путь. Он не является
текущей JSON-first архитектурой и не должен подниматься для работы только с
карточками. Следующая реализация должна добавить файловый локальный векторный
индекс поверх учебных JSON-файлов (например, FAISS или Chroma) и заменить этот
legacy-путь без изменения формата `learner-data`.

# JSON-материалы для локального RAG

Здесь лежат учебные материалы. Один JSON-файл содержит массив записей или
объект с массивом `tasks` (также поддерживаются `records`, `items`, `data`).
Это не карточки учеников: профили находятся в `../../learner-data/learners/`.

## Что уже есть

| Файл | Темы |
|---|---|
| `math_g3_numbers.json` | столбик с переходом, таблица умножения, внетабличные умножение и деление, порядок действий, простые уравнения |
| `math_g3_quantities.json` | масса (грамм, килограмм), время (час, минута, календарь) |
| `math_g3_geometry_lines_demo.json` | точка, прямая, луч, отрезок; длина; периметр многоугольника |
| `math_g3_geometry_triangles.json` | виды треугольников |
| `math_g3_fractions.json` | доли, сравнение долей |
| `math_g3_word_problems.json` | кратное сравнение; цена, количество, стоимость |
| `math_g4_numbers.json` | многозначные числа: разряды и классы |
| `math_g4_division_demo.json` | деление на однозначное и двузначное число |
| `math_g4_geometry.json` | углы, площадь и периметр составных фигур, куб и параллелепипед |
| `math_g4_fractions.json` | дробь, числитель и знаменатель; нахождение части числа |
| `math_g4_quantities.json` | центнер и тонна |
| `math_g4_word_problems.json` | составные задачи, цена/количество/стоимость, работа и производительность |

В каждой теме есть хотя бы одна задача на её типичную ошибку из профилей
учеников — генератор 4 этапа должен уметь бить точно в неё.

## Проверить покрытие

```bash
cd TeachCopilot_RAG
python scripts/check_material_coverage.py
```

Команда сравнивает темы, на которых реально стоят ученики, с содержимым этой
папки. Пробел означает, что задание для такого ребёнка будет собираться без
опоры на программу. Заодно проверяются `topic_id` (должны быть из каталога) и
совпадение класса. То же самое закреплено тестом
`tests/test_material_coverage.py`, включая правило «не меньше двух задач на
тему».

## Формат одной записи

```json
{
  "task_id": "div-408-4",
  "topic_id": "math.g4.numbers.division_by_1_2_digit",
  "grade": "4",
  "content": "Раздели 408 на 4 столбиком.",
  "answer": "408 : 4 = 102. Ноль в частном записываем обязательно.",
  "difficulty": "easy",
  "tags": ["деление", "столбик", "ноль в частном"]
}
```

Обязательные на практике поля: `task_id`, `topic_id`, `grade`, `content` и
`answer`. `topic_id` должен совпадать с записью из
`../../learner-data/catalog/math_g3_g4.json`; только так профиль ученика и
учебный материал связываются между собой.

## Добавить новый материал

1. Откройте или создайте файл, например `knowledge-data/math_g4_division.json`.
2. Добавьте объект задачи в массив `tasks`.
3. Сохраните JSON в UTF-8.
4. Пересоберите индекс. Индекс строится заново намеренно: так embeddings всегда
   соответствуют актуальному тексту.

```bash
cd Alena_audio/TeachCopilot_RAG
uv run python scripts/json_rag.py build \
  --source knowledge-data/math_g4_division.json \
  --index .local-rag/math-g4-division.json
```

Пример успешного ответа:

```json
{"indexed": 12, "index": ".local-rag/math-g4-division.json"}
```

`.local-rag/` — локальная генерируемая папка, в git её не добавляем.

## Найти материал по вопросу

```bash
uv run python scripts/json_rag.py search \
  --index .local-rag/math-g4-division.json \
  --query "Как разделить 408 на 4 и не потерять ноль?" \
  --topic-id math.g4.numbers.division_by_1_2_digit \
  --grade 4
```

Результат — JSON-массив. Его первый элемент — самый похожий учебный фрагмент:

```json
[
  {
    "score": 0.74,
    "content": "Раздели 408 на 4 столбиком...",
    "metadata": {
      "task_id": "div-408-4",
      "topic_id": "math.g4.numbers.division_by_1_2_digit",
      "grade": "4",
      "difficulty": "easy"
    }
  }
]
```

`score` — относительная близость запроса и материала: он зависит от embedding
модели. Для вопроса вне текущей темы сначала ищите с `topic_id`/`grade`, а при
пустом результате повторите поиск без этих фильтров:

```bash
uv run python scripts/json_rag.py search \
  --index .local-rag/math-g4-division.json \
  --query "Что такое угол?"
```

## Как соединить поиск с учеником

```bash
cd ..
python learner-data/tools/build_context.py ivanov-ivan
```

Передайте модели только вывод этой команды плюс `content` 1–3 найденных
материалов. Не передавайте модели полный JSON профиля, `legal_name`, журнал или
баллы.

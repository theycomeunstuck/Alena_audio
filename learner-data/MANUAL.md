# Manual: JSON-профили учеников и локальный RAG

Этот manual описывает текущую рабочую схему без PostgreSQL: карточки учеников,
учебные материалы и локальный векторный индекс — файлы JSON.

## 1. Что хранится где

```text
learner-data/
├── learners/<learner_id>.json       # карточки учеников
├── catalog/math_g3_g4.json          # допустимые topic_id
├── benchmarks/rag_behavior_cases.json
├── EXAMPLES.md                       # готовые заполненные примеры
└── tools/                            # validate.py и build_context.py

TeachCopilot_RAG/
├── knowledge-data/*.json             # учебные задания и фрагменты
├── scripts/json_rag.py               # build/search локального индекса
└── .local-rag/*.json                 # сгенерированные embeddings, не коммитятся
```

`learner-data` и `knowledge-data` — разные наборы данных. Никогда не
индексируйте `learners/*.json` как учебный материал.

## 2. Какие данные получает LLM

В модель передаётся обезличенная педагогическая персонализация:

- интересы;
- текущая цель, сильные и западающие темы;
- повторяющиеся ошибки;
- баллы за прогресс и последние удачи;
- ЗБР, эффективные стратегии и ограничения подачи материала.

В модель **не** передаются `legal_name.last_name`, `legal_name.patronymic`,
контакты, адрес, медицинские сведения или другие прямые идентификаторы.

`pseudonym` используется в LLM как имя для обращения. В production он должен
быть нейтральным alias, например `Ученик-01`; настоящее ФИО храните только в
`legal_name`. Имена в репозитории вымышленные и допустимы только как demo.

## 3. Создать карточку нового ученика

Возьмите наиболее похожую полностью заполненную карточку:

```bash
cd Alena_audio
cp learner-data/learners/smirnova-alina.json learner-data/learners/student-01.json
```

Измените **одновременно** имя файла и `learner_id`. Для production используйте
нейтральный pseudonym:

```json
{
  "learner_id": "student-01",
  "pseudonym": "Ученик-01",
  "legal_name": {
    "first_name": "Иван",
    "last_name": "Иванов",
    "patronymic": "Иванович"
  },
  "grade": "4"
}
```

Для отсутствующего отчества допустимы:

```json
"patronymic": ""
```

или:

```json
"patronymic": "-"
```

После этого заполните все блоки карточки:

| Блок | Что в нём хранить | Попадает в LLM |
|---|---|---|
| `rag_context` | текущая цель, приоритеты, способ объяснения | да |
| `profile` | темп, самостоятельность, мотивация | косвенно через контекст |
| `interests` | 1–3 безопасных интереса | да |
| `knowledge` | сильные, изучаемые, западающие темы | да |
| `error_patterns` | повторяющиеся учебные ошибки | да |
| `mvp.points_ledger` | события прогресса с баллами | да, как агрегат и удачи |
| `learner_model.competencies` | карта компетенций: mastery, самостоятельность, опоры | да, только по теме задания |
| `learner_model.zpd` | производная проекция карты компетенций | да |
| `legal_name` | настоящее ФИО для закрытого интерфейса | нет |

Используйте только `topic_id` из `catalog/math_g3_g4.json`. Готовые разные
профили: [EXAMPLES.md](EXAMPLES.md).

### Карта компетенций и ЗБР

Зона ближайшего развития **не заполняется руками**. Источник правды —
`learner_model.competencies`; блок `learner_model.zpd` из него выводится:

```
mastery >= 0.85 и independence >= 0.80  ->  освоено (делает сам)
mastery <  0.30                          ->  пока за пределами ЗБР
иначе                                    ->  в ЗБР (делает с опорой)
```

```bash
python learner-data/tools/zpd.py student-01           # какая зона и почему
python learner-data/tools/zpd.py student-01 --write   # перезаписать блок zpd
```

Если тема должна быть «за пределами ЗБР», дайте ей запись в карте компетенций с
низкой `mastery` — иначе валидатор предупредит, что зона не выводится. Тему,
которую ещё не проходили, в блок `zpd` не вносят: для неё достаточно
`knowledge[].status = not_started`.

`mastery` и `independence` обычно не выставляют вручную: их пересчитывает код по
уровню помощи, который потребовался ребёнку. Подробности и лестница подсказок —
[TeachCopilot_RAG/docs/ZPD_STAGE4.md](../TeachCopilot_RAG/docs/ZPD_STAGE4.md).

## 4. Проверить карточки и увидеть LLM-контекст

```bash
python learner-data/tools/validate.py --strict
python learner-data/tools/selftest.py
python learner-data/tools/build_context.py student-01
```

Пример безопасного результата:

```text
<child_safe_profile>
Имя_для_обращения: Ученик-01
Школьный_уровень: 4 класс
</child_safe_profile>

<learner_personalization>
Интересы: рисование, настольные игры
Западающие_темы:
- Деление на однозначное и двузначное число [...]
Баллы_за_прогресс: 18
Последние_успехи:
- за самостоятельную проверку решения
ЗБР_сейчас:
- Деление на однозначное и двузначное число [...]
</learner_personalization>
```

Полную карточку для закрытого педагогического инструмента можно вывести так:

```bash
python learner-data/tools/build_context.py student-01 --format json
```

Её никогда нельзя передавать модели целиком.

## 5. Добавить учебный материал для RAG

Создайте файл в `TeachCopilot_RAG/knowledge-data/`, например
`math_g4_division.json`:

```json
{
  "tasks": [
    {
      "task_id": "div-408-4",
      "topic_id": "math.g4.numbers.division_by_1_2_digit",
      "grade": "4",
      "content": "Раздели 408 на 4 столбиком.",
      "answer": "408 : 4 = 102. После первой цифры обязательно записываем ноль в частном.",
      "difficulty": "easy",
      "tags": ["деление", "столбик", "ноль в частном"]
    }
  ]
}
```

`topic_id` и `grade` связывают материал с профилем ученика. При каждом
изменении файла пересобирайте индекс.

## 6. Построить и использовать локальный JSON RAG

```bash
cd TeachCopilot_RAG
uv run python scripts/json_rag.py build \
  --source knowledge-data/math_g4_division.json \
  --index .local-rag/math-g4-division.json
```

Ожидаемый ответ:

```json
{"indexed": 1, "index": ".local-rag/math-g4-division.json"}
```

Поиск:

```bash
uv run python scripts/json_rag.py search \
  --index .local-rag/math-g4-division.json \
  --query "Как разделить 408 на 4 и не потерять ноль?" \
  --topic-id math.g4.numbers.division_by_1_2_digit \
  --grade 4
```

Результат содержит текст фрагмента, оригинальную metadata и `score` похожести.
Если поиск с `topic_id`/`grade` пуст, повторите его без фильтров: ребёнок мог
спросить новую тему.

## 7. Итоговый flow одного вопроса

```text
Авторизованная сессия → learner_id
learner_id → build_context.py → обезличенная персонализация
Вопрос → json_rag.py search → 1–3 учебных фрагмента
Контекст ученика + фрагменты + вопрос → LLM
```

Перед запуском проверяйте обе стороны:

```bash
cd Alena_audio
python learner-data/tools/validate.py --strict
python learner-data/tools/selftest.py
cd TeachCopilot_RAG
uv run ruff check scripts/json_rag.py
uv run pytest tests -q
```

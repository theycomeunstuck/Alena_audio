# Алёна – devops-руководство по памяти (14 ноября 2026 года)

Рабочее дерево: репозиторий `Alena_audio`, приложение – `TeachCopilot_RAG`. Руководство описывает локальный контур памяти: как он устроен, чем настраивается, как запускается и проверяется. Перед переносом на другую машину сверься с актуальным кодом и с `docs/ALENA_MEMORY_STATUS_2026_11_13.md` – часть работ может оставаться незакоммиченной.

## 1. Карта памяти

Четыре контура поиска, у каждого своя роль и своя реализация.

**(a) Материалы (product RAG).** `Store.hybrid_search` в `local_app/store.py:630` – гибрид pgvector и полнотекстового поиска PostgreSQL. Векторная нога – косинусная близость (`embedding <=> %s::vector`, similarity = 1 − расстояние), текстовая – `ts_rank` по `plainto_tsquery('russian')` и `plainto_tsquery('simple')`. Слияние – reciprocal rank fusion с k = 60, финальный тай-брейк по `id`, поэтому порядок детерминирован. Если проиндексированы не все активные материалы под запрошенную модель эмбеддингов, поиск поднимает `IndexNotReadyError` вместо тихой неполной выдачи.

**(b) Personal memory.** Гибридный режим – round-robin слияние векторной ноги (BGE-M3, 1024 измерения) и лексической по подтверждённым исходным урокам. Ранжирование выполняет `rank_source_candidates` из `local_app/personal_graph_retrieval.py`; `local_app/personal_memory.py` подключает его вместе с кешем эмбеддингов источников (`SourceEmbeddingCache`). В поиск попадают только одобренные преподавателем источники.

**(c) Диалоговый поиск (conversation retrieval).** `local_app/conversation_retrieval.py`, две формулы: `legacy` – поставляется и работает по умолчанию (cosine + 0,04 × доля точных терминов), `rrf` – опция (reciprocal rank fusion косинуса и стеммированного BM25). Переключение – параметр `fusion` вызывающего кода; env-переключателя нет.

**(d) Personal graph (эксперимент).** Graphiti поверх Neo4j на `bolt://127.0.0.1:17687` (`research_memory/personal_memory_lab/backends.py`). Реестр индексов – таблица `teachcopilot.personal_graph_indexes` (миграция `016_personal_graph_index_registry.sql`), кеш эмбеддингов графа – миграция `017_personal_graph_embedding_cache.sql`. По умолчанию приложение Graphiti и LangGraph не импортирует. Эксперимент валиден только с контекстом 16384 или 32768 токенов и моделями `gemma4:12b` + `bge-m3` (проверка в `local_app/config.py`). Факты из графа НИКОГДА не попадают в текст ответа: находки графа только указывают на канонические исходные ответы, которые и цитируются.

**(e) Объединение каналов [внедрено 14 ноября, коммит a2825bd].** Граф – автоматический канал внутри personal_memory hybrid: флаг `LOCAL_MEMORY_GRAPH_CHANNEL` (по умолчанию включён) добавляет графовые попадания третьей ногой round-robin, когда готовый индекс соответствует точному снимку; без запущенного Neo4j или без готового индекса канал мягко деградирует с причиной в diagnostics (`channel_disabled`, `no_ready_index`, `password_file_unavailable` и др.). Канал только потребляет готовый индекс и никогда не строит его; факты графа в текст ответа не попадают. Индекс строится явным teacher-заданием personal_graph (требует Neo4j и контекст 16k/32k).

## 2. Env-флаги

Все значения читает `Settings.from_env()` из `local_app/config.py`; файл `.local.env` в корне `TeachCopilot_RAG` загружается с `override=False` (реальные переменные окружения сильнее файла). Память-относящиеся переменные:

| Переменная | По умолчанию | Эффект |
|---|---|---|
| `DATABASE_URL` | `postgresql://teachcopilot@127.0.0.1:5434/teachcopilot_local` | Рабочая локальная БД (Docker-контейнер, порт 5434 → 5432). Для тестов не используется – см. раздел 6. |
| `OLLAMA_URL` | `http://127.0.0.1:11435` | Только HTTP-loopback без креденшелов, query и fragment (`require_local_url` иначе бросает исключение). |
| `LOCAL_CHAT_MODEL` | `gemma4:12b` | Чат-модель Ollama. Для экспериментального графа – строго `gemma4:12b`. |
| `LOCAL_EMBEDDING_MODEL` | `bge-m3` | Модель эмбеддингов (BGE-M3, 1024). Для графа – семейство `bge-m3`. |
| `LOCAL_CONTEXT_TOKENS` | `8192` | Контекст модели; минимум 2048. Экспериментальный граф требует 16384 или 32768. |
| `LOCAL_MIN_SIMILARITY` | `0.35` | Минимальная семантическая похожесть в диалоговом поиске. |
| `LOCAL_PERSONAL_MEMORY_ENABLED` | `1` | `0` – исключает старый диалог, личный архив, сводки, персональные предпочтения и подтверждённую память из контекста нового ответа. Текущее сообщение и ответ записываются в историю, владелец её видит. |
| `LOCAL_PERSONAL_MEMORY_RETRIEVAL_MODE` | `hybrid` | Один из `chronological`, `lexical`, `vector`, `hybrid`; иное значение – ошибка при старте. |
| `LOCAL_EXPERIMENTAL_PERSONAL_GRAPH` | `0` | `1` – включает экспериментальный графовый метод. Требует контекст 16k/32k и `gemma4:12b` + `bge-m3`, иначе конфиг невалиден. |
| `LOCAL_PERSONAL_GRAPH_RETRIEVAL_MODE` | `graph_hybrid` | Один из `chronological`, `lexical`, `vector`, `hybrid`, `graph_hybrid`. |
| `LOCAL_PERSONAL_GRAPH_PASSWORD_FILE` | `C:/Users/name/PycharmProjects/misc/teachCopilot/.local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt` | Файл с паролем Neo4j (пользователь `neo4j`). Не коммитить и не печатать. |
| `LOCAL_MEMORY_GRAPH_CHANNEL` | `1` | Автоматический графовый канал внутри personal_memory hybrid; без Neo4j или без готового индекса канал мягко отключается с причиной в diagnostics. |

Сопутствующие (не память, но тот же config): `LOCAL_REQUEST_TIMEOUT` (180), `LOCAL_MAX_OUTPUT_TOKENS` (1024), `LOCAL_SESSION_HOURS` (12), `LOCAL_MEMORY_RESPONSE_JOURNAL`, `LOCAL_CHAT_RESPONSE_JOURNAL`, `LOCAL_GENERATION_SCHEDULER_DSN`, `LOCAL_DEMO_LOGIN`.

## 3. Установка и запуск

### 3.1 Окружение (uv)

Python 3.12 и `uv`; источники зависимостей – `pyproject.toml` и `uv.lock` приложения. Все команды – из каталога `TeachCopilot_RAG`.

```powershell
cd C:\Users\name\PycharmProjects\misc\teachCopilot\Alena_audio\TeachCopilot_RAG
uv sync --locked
```

Новая зависимость – только `uv add <пакет>` (для разработки `uv add --dev <пакет>`) с сохранением `pyproject.toml` и `uv.lock`; не заменять это установкой через системный `pip`. Команды проекта запускать через `uv run --locked`. `torch` уже в зависимостях (CPU-индекс PyTorch) – отсутствие его в отдельном окружении не повод добавлять заново.

### 3.2 PostgreSQL в Docker

`compose.local.yml` – образ `pgvector/pgvector:0.8.7-pg18`, контейнер `teachcopilot_local_db`, порт `127.0.0.1:5434 → 5432`, пароль берётся из `.local.env` (`POSTGRES_PASSWORD`), данные – в named volume `local_database`.

```powershell
docker compose --env-file .local.env -f compose.local.yml up -d --wait
```

Если `.local.env` ещё нет – создаст его `scripts/setup_local.ps1` (генерирует пароль, пишет `DATABASE_URL`, `OLLAMA_URL`, `LOCAL_CHAT_MODEL`, `LOCAL_EMBEDDING_MODEL`, `LOCAL_CONTEXT_TOKENS`), создаёт `.venv-local`, ставит зависимости из `requirements-local.lock.txt`, применяет миграции и импортирует данные:

```powershell
powershell -File scripts\setup_local.ps1
```

### 3.3 Старт одним скриптом

```powershell
powershell -File scripts\start_local.ps1
```

Опционально `param([int]$ApiPort = 8099)`. Скрипт: поднимает Docker-БД (`up -d --wait`), применяет миграции (`scripts/manage_local.py migrate`), готовит Ollama (`OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_NO_CLOUD=1`, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=2`; runtime – `.local\ollama\ollama.exe`, иначе из PATH; запускает `ollama serve`, если порт ещё молчит) и ждёт готовности до 30 секунд. Затем стартует memory worker (`scripts/memory_worker.py`, скрытый процесс, логи `.local\memory-worker-<guid>.stdout.log` и `.stderr.log`) и API `uvicorn server:app` на `127.0.0.1:8099`. При выходе скрипта worker останавливается.

Миграции отдельно (подкоманды `manage_local.py`: `migrate`, `import`, `bootstrap`, `export`, `backup`, `restore`):

```powershell
uv run --locked python scripts\manage_local.py migrate
```

### 3.4 Ollama

Локальный Ollama на `http://127.0.0.1:11435`, модели `gemma4:12b` (чат) и `bge-m3` (эмбеддинги). 13 ноября Ollama обновлён с 0.24.0 до 0.40.16 – до обновления `gemma4:12b` не загружался. Проверка живости:

```powershell
Invoke-RestMethod http://127.0.0.1:11435/api/version
ollama list
```

### 3.5 Neo4j

Экспериментальный граф использует Neo4j Community на `bolt://127.0.0.1:17687` (только loopback), пользователь `neo4j`, пароль – из файла по пути `LOCAL_PERSONAL_GRAPH_PASSWORD_FILE` (по умолчанию `.local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt`; рядом с дистрибутивом лежит JDK 21). Без запущенного Neo4j продукт работает: графовый метод просто недоступен, а после объединения 14 ноября канал деградирует мягко.

## 4. Проверки

### 4.1 Обычные тесты

```powershell
uv run --locked python -m pytest
```

Конфиг `pyproject.toml` собирает только `tests/`. Маркер `deep` (корневой `pytest.ini` репозитория) помечает долгие реальные интеграционные тесты с моделями; по умолчанию они исключены (`addopts = -m "not deep"`). Полные серийные прогоны выполнялись на свежих одноразовых базах: 2392 теста. Тесты с PostgreSQL – только на одноразовой базе (раздел 6).

### 4.2 Одноразовые базы (disposable-DB launcher)

Лаунчер `C:/Users/name/Documents/ChatGPT/Claude/teachcopilot-work/codex_disposable_db_suite_20261113.py`: читает `DATABASE_URL` из `.local.env` только чтобы дойти до локального PostgreSQL-сервера, создаёт случайную базу `teachcopilot_memory_verify_<12 hex>`, направляет на неё оба DSN (`TEACHCOPILOT_TEST_DSN` и `DATABASE_URL`), запускает `python -m pytest <targets>` из `TeachCopilot_RAG` и удаляет только свою базу (с проверкой префикса и длины имени). Аргументы – цели pytest (по умолчанию `tests`):

```powershell
python C:\Users\name\Documents\ChatGPT\Claude\teachcopilot-work\codex_disposable_db_suite_20261113.py tests
```

Тот же префикс `teachcopilot_memory_verify_` использует `scripts/verify_memory.py` – он сам создаёт одноразовую базу и передаёт её через `TEACHCOPILOT_TEST_DSN`. Нагрузочный раннер `scripts/load_memory.py` защищён `require_disposable_dsn` – принимает DSN только вида `teachcopilot_memory_verify_` + 12 hex-символов; провайдер заменён `ControlledRuntime` (инъекция задержек, сохранение всех промптов), случайный HTTP-вызов роняет процесс до отправки. Для графовых интеграционных проверок `scripts/create_personal_graph_test_db.py` создаёт базу `nirs_personal_graph_it_<12 hex>` с миграциями:

```powershell
uv run --locked python scripts\create_personal_graph_test_db.py --name-file <путь-к-файлу-имени>
```

Смоук `scripts/smoke_personal_graph_api.py` («Real API, PostgreSQL, Neo4j and local Gemma with synthetic accounts and data») поднимает реальный API на этой одноразовой базе с синтетическими аккаунтами и учеником.

### 4.3 Live-тесты и memory_doctor

```powershell
uv run --locked python -m pytest -m live
```

Live-маркер требует запущенного Ollama; модель берётся из переменной `LOCAL_CHAT_MODEL` окружения, а не хардкодом. В дереве на 14 ноября маркер ещё не зарегистрирован – сверься с актуальным кодом.

Скрипт диагностики `scripts/memory_doctor.py`: `--check` (PostgreSQL только на чтение, таблицы памяти, Ollama-модели, embed-проба, env-флаги, `/health/ready` работающего приложения), `--live` (плюс живой вызов модели), `--e2e` (сквозной цикл personal_memory на одноразовой базе `nirs_memory_doctor_*`: задание → воркер → одобрение → чат с реальной моделью, JSON-отчёт), `--graph` (расширяет e2e циклом personal_graph при доступном Neo4j; иначе skip). Пример: `uv run --locked python scripts/memory_doctor.py --check --live`. Отчёты пишутся в `teachcopilot-work/` и в git не публикуются.

### 4.4 Диагностика памяти в интерфейсе (только учитель)

В рабочем месте учителя – панели памяти: «Основания следующего шага – Память занятий» и «Мои беседы» с кнопкой «Обновить». По каждой версии памяти видны метод, дата, признак `stale` («требует обновления») и статус подтверждения («подтверждена» / «черновик»); устаревший разрешённый снимок помощник не использует. Это функциональная диагностика, не числовой дашборд.

## 4b. Context tokens: the 16k floor (English note)

**Set `LOCAL_CONTEXT_TOKENS=32768` for the unified memory path.** The
application accepts smaller values, but the personal-memory chat path
breaks below 16384:

- the complete-memory resolver prompt alone is larger than the byte
  budget the app derives as `(context_tokens - max_output_tokens) * 2`;
  at `LOCAL_CONTEXT_TOKENS=8192` a real chat returns HTTP 413 (or 502
  through some routes) — verified independently twice on 2026-11-14;
- the graph channel requires `context_tokens` of 16384 or 32768
  (`ApplicationModel` refuses anything else), and the index config
  fingerprint binds an index to the context it was built with. After
  changing `LOCAL_CONTEXT_TOKENS`, previously approved memory versions
  go stale once (fail-closed by design) and the teacher must
  re-approve them in the app.

Current deployment: `.local.env` sets `LOCAL_CONTEXT_TOKENS=32768`
(since 2026-11-14). If you change it, restart the app
(`scripts/start_local.ps1`) and verify `/health/ready`, then re-approve
memory versions as prompted.

## 5. Troubleshooting

**HTTP 409 при работе с памятью – штатная реверификация.** После новых сообщений привязанного ученика прежний снимок памяти нужно заново подтвердить преподавателем (в UI – предупреждение и явное подтверждение). Новые источники не включаются в ранее разрешённую область; если изменились карточка, исправления или другие источники – требуется новая проверка. API возвращает 409 в том числе при повторном подтверждении с другим выбором (`memory was already confirmed with a different selection`) и при изменении ревизии (`learner revision changed; reload before continuing`).

**`PersonalGraphIndexStale`.** Поиск отказывается использовать устаревший подтверждённый графовый индекс, не меняя ни реестр `teachcopilot.personal_graph_indexes`, ни сам граф (`local_app/personal_graph_index.py`). Лечение – построить новую версию индекса и подтвердить её; старые версии без сохранённой базовой ревизии консервативно считаются устаревшими.

**«unknown model architecture» в Ollama.** Значит Ollama старше, чем блобы модели: 0.24.0 не знал всех архитектур новых блобов. Обнови Ollama (13 ноября обновлён до 0.40.16 через winget), после чего `gemma4:12b` загружается и генерирует.

**Режим рассуждений gemma4.** Модель может вернуть «thinking»-поле; приложение отключает рассуждения параметром `think: false` (`local_app/runtime.py`). Если видишь служебные рассуждения в ответе – проверь, что запрос идёт через `OllamaRuntime`, а не напрямую.

**Playwright не находит браузер.** Ставь Chromium через uv и используй один и тот же `PLAYWRIGHT_BROWSERS_PATH` при установке и запуске; часть тестов по умолчанию ищет браузер в `TeachCopilot_RAG/.local/ms-playwright`, установка в общий кеш сама по себе путь не меняет.

```powershell
uv run --locked python -m playwright install chromium
```

**`neo4j.time` – необязательная зависимость.** В полных прогонах допустим один пропуск теста из-за отсутствующего `neo4j.time` – это ожидаемое поведение, не регрессия.

**`WSAEADDRINUSE` (10048) на `127.0.0.1:5434` в живых браузерных тестах.** Приложение открывает новое соединение на каждую операцию; всплеск тысяч коротких соединений исчерпывает эфемерные порты Windows (диапазон 49152–65535, часть занята Hyper-V). Это среда тестов, не логика приложения и не сбой PostgreSQL (на сервере в момент сбоя – 10 соединений из 100; изолированные прогоны зелёные). Возможный фикс – расширение динамического диапазона `netsh int ipv4 set dynamicport tcp start=1025 num=64510` (права администратора, обратимо) – применяется только по решению пользователя. Диагностика – `teachcopilot-work/503-diag-20261113-run1` и `-run2`.

## 6. Правила безопасности

- Тесты с PostgreSQL – только на одноразовых базах `nirs_*` / `teachcopilot_memory_verify_*`. Рабочая база `teachcopilot_local` для тестов не используется никогда; перед любой очисткой проверяй точный объект и сохраняй данные пользователя.
- Никаких реальных детских данных в тестах, отчётах и на публичном сайте – только синтетические аккаунты и ученики (как в `smoke_personal_graph_api.py`: «Синтетический ученик», `@synthetic.test`).
- Секреты – пароль Neo4j, содержимое `.local.env`, токены, журналы ответов – не коммитить, не печатать в отчёты и не публиковать. Локальные резервные копии, тестовые журналы и учётные данные не публиковать.
- Результаты синтетических тестов не представляют проверку качества на реальных детях и не доказывают, что поиск всегда находит нужное или что модель корректно его использует. НИРС (`research_memory/`) и продуктовые изменения вести раздельно.
- Применение схемы 024 к рабочей базе – не запуск новой версии приложения; запущенный процесс проверяй отдельно и не считай его автоматически обновлённым.

## Сверка

Прогонять сверху вниз; каждый шаг либо зелёный, либо ссылается на раздел 5.

```powershell
cd C:\Users\name\PycharmProjects\misc\teachCopilot\Alena_audio\TeachCopilot_RAG
```

1. Окружение воспроизводится: `uv sync --locked` завершается без ошибок.

```powershell
uv sync --locked
```

2. Docker Desktop запущен; БД поднимается и здорова.

```powershell
docker compose --env-file .local.env -f compose.local.yml up -d --wait
```

3. Ollama жив на 11435, модели `gemma4:12b` и `bge-m3` на месте.

```powershell
Invoke-RestMethod http://127.0.0.1:11435/api/version
ollama list
```

4. Схема БД актуальна (миграции 001–024 применены).

```powershell
uv run --locked python scripts\manage_local.py migrate
```

5. Обычные тесты зелёные (без `-m live`, без `deep`).

```powershell
uv run --locked python -m pytest
```

6. Для PostgreSQL-тестов создана одноразовая база `teachcopilot_memory_verify_*` (или `nirs_personal_graph_it_*` для графа) – и только она.

```powershell
python C:\Users\name\Documents\ChatGPT\Claude\teachcopilot-work\codex_disposable_db_suite_20261113.py tests
```

7. Live-тесты – при запущенном Ollama: `uv run --locked python -m pytest -m live`; модель берётся из `LOCAL_CHAT_MODEL`/`LOCAL_EMBEDDING_MODEL`, без запущенного Ollama тесты пропускаются.

```powershell
uv run --locked python -m pytest -m live
```

8. Диагностика памяти – `memory_doctor.py --check`, затем `--e2e`; при графовом эксперименте – `--graph` и `--live`.

```powershell
uv run --locked python scripts\memory_doctor.py --check
```

9. Весь стек поднимается одним скриптом; API отвечает на 8099.

```powershell
powershell -File scripts\start_local.ps1
```

10. В UI учителя панели «Память занятий» и «Мои беседы» открываются; версии памяти показывают метод и статус подтверждения; 409 при устаревании – штатно, подтвердить снимок учителем.

11. Neo4j (если включён графовый эксперимент): доступен на `bolt://127.0.0.1:17687`, пароль читается из файла `LOCAL_PERSONAL_GRAPH_PASSWORD_FILE`, пароль нигде не напечатан и не закоммичен.

12. Итог сверки: ни один шаг не обращался к рабочей базе `teachcopilot_local`, секреты не публиковались, синтетические результаты не приписаны реальным детям.

# Покрытие проверки последовательных занятий

Карта подготовлена 22 октября 2026 года от исходного коммита `40b86e5`, дополнена результатами полных проверок 23 октября по местному времени. Здесь перечислены существующие проверки, результаты полных прогонов и отдельного чтения сохранённых данных.

## Команда заключительной проверки

Из `TeachCopilot_RAG` после последних продуктовых изменений:

```powershell
.venv-local/Scripts/python.exe scripts/verify_local.py --ui
```

Скрипт создаёт собственную случайно названную PostgreSQL БД `teachcopilot_verify_*`, передаёт её тестам через `TEACHCOPILOT_TEST_DSN` и удаляет только эту БД после выполнения. Он запускает весь `tests/`, затем `../learner-data/tools/selftest.py`. Отчёт сохраняется в `.local/verification/latest.json`, подробности – в `.local/verification/pytest.log` и `.local/verification/learner-policy.log`. Эти текущие журналы нужно скопировать в новый каталог пакета после прогона.

При исходном аудите `pytest --collect-only -q tests` собирал 1386 тестов, включая 99 браузерных проверок в 11 модулях. В `selftest.py` объявлены 52 проверки карточки. Новые тесты пакета увеличили общий набор; фактические результаты приведены ниже. Проверка браузера не считается выполненной при пропуске из-за отсутствующего Chromium.

## Фактические полные прогоны

| Пакет | Коммит в начале и конце | Pytest | Проверки карточки | Пропуски | Полное время |
| --- | --- | ---: | ---: | ---: | ---: |
| `full01` | `5fded36` | 1435 пройдено | 52 пройдено | 0 | 784,156 с |
| `full02` – после восстановления и единственной инженерной итерации | `129d90c` | 1458 пройдено | 52 пройдено | 0 | 646,297 с |

Оба вызова `scripts/verify_local.py --ui` завершились с кодом 0, включая все 99 браузерных проверок. В каждом pytest есть одно предупреждение об устаревающем использовании `httpx` в Starlette; ошибок и пропусков нет. В `full02` дополнительные тесты покрывают восстановление, однократное разрешённое расширение срока и выбор синтетического интереса.

Исходные отчёты и журналы каждого прогона скопированы в отдельные каталоги `.local/continuation_audit_20261022/verification/full01/` и `full02/`. В них лежат `start.json`, `finished.json`, `driver.log`, `latest.json`, `pytest.log` и `learner-policy.log`. `finished.json` фиксирует SHA256 трёх файлов результата и коммит после проверки. Эти локальные файлы не добавляются в Git.

Для каждого тестового subprocess отдельно установлены `OLLAMA_URL=http://127.0.0.1:9`, пустой `TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG`, пустые `LOCAL_GENERATION_SCHEDULER_DSN` и `LOCAL_GENERATION_RUN_BUDGET_PATH`. Настоящие модели и внешний провайдер тестами не вызывались; новых оплачиваемых или локальных модельных генераций – 0. Окружение и процессы выполняющегося пилота не изменялись. PostgreSQL используется только для создаваемых тестами временных БД; учебные записи рабочей БД не изменялись.

## Соответствие согласованным условиям

| Условие | Существующие проверки | Существенная граница |
| --- | --- | --- |
| Изоляция учеников, владельца и тем | `test_memory_store.py`, `test_memory_load.py::test_real_worker_keeps_learner_sources_out_of_other_actual_prompts`, `test_memory_cross_topic_flow.py`, `test_memory_answer_projection.py`, `test_learning_topic_binding.py` | Проверяются фактические сохранённые источники и переданные аргументы; ответы модели в этих тестах управляемые. |
| Повтор сообщения без новой генерации | `test_memory_store.py::test_duplicate_message_concurrently_reserves_only_one_request`, `test_memory_service.py`, `test_verified_learning_api.py::test_replayed_message_returns_same_response_without_another_model_call_or_source`, `test_explicit_learning_actions_postgres.py` | Отдельно покрыты одновременный повтор и повтор после пересоздания приложения. |
| Конфликт ревизий и атомарное подтверждение | `test_memory_store.py::test_interest_confirmation_revision_conflict_keeps_card_and_draft_unchanged`, `test_memory_freshness.py`, `test_memory_source_corrections.py` | Проверяется сохранение карточки и черновика при конфликте, включая одновременные исправления. |
| Pending и неизвестный исход без автоматического повторения | `test_memory_load.py::test_real_worker_restart_requeues_only_before_call_and_preserves_unknown_outcome`, `test_memory_generation.py`, `test_memory_store.py`, `test_current_learning_session_api.py` | До вызова разрешено восстановление очереди; после начатого вызова неизвестный исход сохраняется. |
| Исправление источника и неизвестные условия помощи | `test_memory_source_corrections.py`, `test_memory_source_correction_context.py`, `test_memory_retrieval.py`, `test_memory_cross_topic_flow.py::test_correction_stays_draft_then_survives_restart_with_unknown_help` | Исходная запись остаётся неизменной; исправление связано с ней и не становится новым успехом ученика. |
| Состояние после перезапуска | `test_guided_restart.py`, `test_current_learning_session_api.py::test_current_survives_new_store_and_remains_read_only_with_expired_pending`, `test_learning_session_archive_api.py::test_archive_reopens_with_a_fresh_read_only_store_and_never_mutates_durable_rows` | Первый тест перезапускает собственный процесс API с запрещёнными вызовами модели. Перезапуск реальных API и worker с `gemma4:12b` относится к новому конечному runner. |
| Свободная беседа не продвигает проверенный шаг и не подтверждает компетенцию | `test_guided_division_api.py`, `test_guided_division_ui.py`, `test_chat_guided_return_ui.py`, `test_memory_api_flow.py`, `test_zpd_evidence.py` | Возобновление требует явного действия; интерес и единичный успех не доказывают освоение. |
| Реальное HTTP-тело внешнего ответа, текущая реплика и цитаты | `test_answer_adapter.py`, `test_memory_answer_projection.py` | Тестовый transport проверяет фактическое тело до отправки. Шесть новых оплачиваемых ответов и шесть локальных переходов фиксируются отдельным интеграционным пакетом. |
| Конечность и изоляция нового runner | `test_product_sessions_runner.py`, `test_product_sessions_recovery.py`, `test_generation_run_budget.py` | Проверяются сухой режим без БД и моделей, синтетические карточки, пределы исходного и зависимого наборов, единый счётчик между процессами, срок, отдельный учёт embeddings и сохранение неизвестного исхода при восстановлении. |
| Новый бюджет и разрешённый журнал v2 | `test_memory_budget_v5.py`, `test_external_answers_v2_runner.py` | Проверяются сохранение прежних резервов, отказ от переноса старого остатка, обязательный receipt и совпадение конфигурации, шесть внешних вызовов, локальный переход до оплаты и отсутствие повторения неизвестного оплаченного исхода. |

## Браузерные проверки, необходимые для передачи

| Поведение | Модуль и ключевые сценарии | Собрано |
| --- | --- | ---: |
| Восстановление чата и продолжение того же занятия | `test_current_learning_session_ui.py` – перезагрузка с исходными сообщениями, тема Python, блокировка при pending/unknown, чтение без повторной отправки, поздние ответы старого аккаунта | 13 |
| Архив, обе пагинации и сохранение черновика | `test_learning_session_archive_ui.py` – `test_archive_does_not_change_the_active_topic_draft_mode_or_session`, `test_history_pagination_prepends_older_turns_in_order_without_duplicate_requests`, `test_completed_session_list_loads_the_next_page_without_changing_chat`, ошибки и смена аккаунта | 11 |
| Явный возврат к сохранённому примеру из обычного чата | `test_chat_guided_return_ui.py` – сохранённый шаг после перезагрузки и после беседы, подтверждённый возврат между занятиями, двойное нажатие, изменившаяся закладка, запрет при pending | 18 |
| Пошаговый режим | `test_guided_division_ui.py` – нулевой шаг, пауза, перезагрузка, возврат после беседы и после подтверждения преподавателем | 15 |
| Исправление и сохранение черновика преподавателя | `test_memory_corrections_ui.py` – конфликт ревизий, неизвестный исход, устаревшие ответы при смене ученика или темы, отслеживание задания без повторной отправки | 17 |

Все 99 браузерных проверок используют реальный FastAPI и статический интерфейс с синтетическими управляемыми хранилищем и ответами. Они проверяют поведение интерфейса, но отдельно не доказывают качество `gemma4:12b` или долговечность PostgreSQL. Для этих свойств нужны новый конечный runner и перечисленные проверки БД. Новые дублирующие тесты по этой карте не добавлены.

## Сохранность и исходные данные НИРС

Канонический manifest – `.local/research_memory/final_v4_20261018/freeze.json`: 13 файлов кода, 7 промптов и 9 входных файлов, всего 29. Сохранён отдельный baseline вне замороженного опыта:

```powershell
.venv-local/Scripts/python.exe .local/continuation_audit_20261022/preservation_baseline.py --verify
```

Он проверяет 29 замороженных файлов, все 673 исходных runtime-файла `final_v4_20261018` и 7 исходных посторонних файлов. Дополнительные файлы внутри замороженного runtime также вызывают отказ. Manifest `.local/continuation_audit_20261022/preservation_baseline.json` имеет SHA256 `4a3400b9ef15e6cc808e9225a1c04dc82b5ee28575eb4e4c425b812fb2ff3b97`.

После заключительного `full02` проверка прошла: 29 + 673 + 7 файлов совпадают, изменённых, отсутствующих и добавленных внутри замороженного runtime нет. Результат сохранён в `.local/continuation_audit_20261022/verification/full02/preservation_check.json`. Исходные продуктовые HTTP-артефакты этой проверкой не изменялись.

Чтение сохранённого опыта подтвердило 128 уникальных строк со статусом содержательной оценки `not_assessed`: 94 технически принятых ответа и 34 отказа. Все 94 принятых ответа проходят прежнюю строгую схему действия и допустимость видимых ID. Есть 16 историй, в каждой два запроса и четыре метода. Старый слепой пакет сохраняет 20 мест по пяти заранее выбранным историям; шесть мест содержат технический отказ вместо ответа. Названия методов находятся в отдельном private-файле. Это технический аудит, содержательные оценки он не создаёт.

Артефакты бесплатного аудита – `.local/continuation_audit_20261022/inputs_and_collection.json` и `frozen_input_audit/`. Их воспроизведение:

```powershell
.venv-local/Scripts/python.exe .local/continuation_audit_20261022/audit_inputs_and_collection.py
.venv-local/Scripts/python.exe tools/analyze_frozen_memory_results.py --output .local/continuation_audit_20261022/frozen_input_audit_repeat
```

Новая калибровка и её закрытые метки при подготовке этой карты не читались. Модель и внешний API не вызывались.

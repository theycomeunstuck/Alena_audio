# Разрешённое восстановление окна локального пакета

23 октября 2026 года по времени Asia/Tomsk пользователь разрешил: «Разрешить ещё 60 минут, общий предел 160 сохранить». Это разрешение относится к операционному восстановлению после прерывания сессии по лимиту использования. Оно не является инженерной итерацией продукта, не разрешает повторение запросов с неизвестным исходом и не меняет денежные лимиты внешнего API.

Исходный журнал – `.local/product_sessions_v1/initial01/local-generation-budget.sqlite3`. Начало исходного окна – `2026-10-22T15:31:46.745992+00:00`; исходное окончание – `2026-10-22T17:01:46.745992+00:00`. Отдельная запись разрешения – `.local/product_sessions_v1/recovery_authorization_20261023.json`. Она содержит точную цитату пользователя, дату записи, область разрешения, путь к журналу и исходные границы окна. Журнал и запись разрешения являются runtime-артефактами и не добавляются в Git.

`GenerationRunBudget.extend_authorized(path)` добавляет одну строку в `authorized_extensions`. Окно начинается при явном применении разрешения и заканчивается через 3600 секунд. Повторное применение запрещено, включая применение другой записи разрешения. Исходные `run.started_at`, `run.deadline_at`, `run.max_seconds=5400` и `run.max_generations=160` сохраняются. Все строки вызовов, embeddings, ошибки и неизвестные исходы сохраняются. UPDATE и DELETE строк таблицы продлений запрещены триггерами SQLite.

`snapshot.deadline_at` после применения показывает действующее окончание окна; `snapshot.original_deadline_at` и `snapshot.authorized_extensions` сохраняют историю. Повторное `initialize(max_calls=160, max_seconds=5400)` проверяет исходную конфигурацию и не начинает новое окно. Новые настройки runtime и записи в рабочей PostgreSQL для продления не нужны.

Проверка разрешения без изменения журнала и без запуска нового окна:

```powershell
.venv-local/Scripts/python.exe scripts/extend_local_generation_run.py --ledger .local/product_sessions_v1/initial01/local-generation-budget.sqlite3 --authorization .local/product_sessions_v1/recovery_authorization_20261023.json
```

После готовности кода восстановления оператор один раз применяет разрешение:

```powershell
.venv-local/Scripts/python.exe scripts/extend_local_generation_run.py --ledger .local/product_sessions_v1/initial01/local-generation-budget.sqlite3 --authorization .local/product_sessions_v1/recovery_authorization_20261023.json --execute
```

На момент подготовки изменения разрешение не применялось. Бесплатные проверки покрывают ровно одно продление, сохранность 30 генераций, 10 embeddings и одного неизвестного исхода, исходные настройки, запрет изменения записи продления, неизменность общего лимита 160 и отказ по окончании дополнительного окна.

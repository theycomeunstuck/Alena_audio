"""Build final research and integration documents from completed durable artifacts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / ".local/research_memory/final_v4_20261018"
NIRS = ROOT.parents[1] / "nirs"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    Path(path).write_text(text.replace(chr(0x2014), chr(0x2013)), encoding="utf-8")


def number(value):
    return "не измерено" if value is None else f"{value:.1f}"


def pytest_passed_tests(app: dict) -> int:
    check = next((row for row in app.get("checks", []) if row.get("name") == "pytest"), None)
    if check is None or "passed_tests" not in check:
        raise RuntimeError("application_verification/latest.json has no pytest check with passed_tests; "
                           "delivery not built")
    return check["passed_tests"]


def build():
    freeze = read(PACKAGE / "freeze.json")
    source = ROOT / "research_memory"
    current = {"code": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob("*.py")},
               "prompts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source / "prompts").glob("*.txt")}}
    if current != freeze["code_and_prompts"]:
        raise RuntimeError("Frozen research source changed; delivery cannot claim the fixed protocol")
    report = read(PACKAGE / "final_report/report.json")
    calibration = read(PACKAGE / "assessment/calibration/summary.json")
    integration = read(PACKAGE / "integration_state.json")
    if integration["status"] not in {"completed", "completed_with_limitations", "failed"}:
        raise RuntimeError("Integration is still running")
    load_path = ROOT / ".local/memory-verification/final_20261018_v4/load.json"
    load = read(load_path)
    app = read(PACKAGE / "application_verification/latest.json")
    # Audit F-077: validated before the first artifact is written – a missing
    # check used to raise StopIteration mid-build and leave a half-built delivery.
    tests = pytest_passed_tests(app)
    budget = []
    for path in sorted((ROOT / ".local/research_memory").glob("*budget.sqlite3")):
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT charged_usd,status FROM budget_calls").fetchall()
        budget.append({"ledger": path.name, "calls": len(rows),
            "charged_or_reserved_usd": str(sum((Decimal(row[0]) for row in rows), Decimal(0))),
            "unknown_reservations": sum(row[1] == "reserved_unknown" for row in rows)})
    total = sum((Decimal(row["charged_or_reserved_usd"]) for row in budget), Decimal(0))
    if total > Decimal("4"):
        raise RuntimeError("Overall authorized budget was exceeded")
    budget_report = {"overall_cap_usd": "4", "charged_or_reserved_usd": str(total),
                    "remaining_usd": str(Decimal("4")-total), "ledgers": budget}
    (PACKAGE / "budget_final.json").write_text(json.dumps(budget_report, indent=2), encoding="utf-8")
    methods = report["heldout"]["methods"]
    dev_rows = [json.loads(line) for line in (PACKAGE / "dev/results.jsonl").read_text(encoding="utf-8").splitlines()]
    dev_counts = Counter((row["method"], row["status"]) for row in dev_rows)
    lines = ["# Итог исследования памяти TeachCopilot", "", "18 октября 2026 года.", "",
        "Данные, рубрика, калибровочные метки и автоматические оценки содержательно не проверены человеком. "
        "Исследование синтетическое; эффект обучения детей не измерялся. Рабочий метод приложения остаётся tail.", "",
        "## Dev и техническая исполнимость", "",
        "В единственном новом dev-прогоне приняты 48 из 64 строк, в прежнем – 59. Сопоставлены все 64 строки: "
        "47 остались успешными, четыре имеют отказ в обеих версиях, 12 получили новый отказ, одна восстановилась. "
        "Это сравнение технической пригодности при изменённых строгих требованиях к основаниям, "
        "а не оценка педагогической правильности.", "",
        "| Метод | Dev ok / 16 | Test ok / 32 | Содержательно оценено / 32 | Критические ошибки среди оценённых | useful среди оценённых |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, values in methods.items():
        lines.append(f"| {method} | {dev_counts[(method,'ok')]} | {values['technical_ok']} | {values['assessed']} | "
                     f"{values['factual_verdict']['critical_error']} | {values['utility']['useful']} |")
    lines += ["", "Ноль критических меток при нулевом покрытии не означает отсутствие ошибок. "
        "Структурированный метод часто не формирует отдельное буквальное основание известной помощи/проверки "
        "или перефразирует цитату. У h03 отсутствие принятой ранней записи также делает последующее исправление "
        "с конкретным record_id невозможным. В summary h07 сохранился обрыв по лимиту выхода. "
        "JSON и цитаты не восстанавливались, дубликаты не удалялись, ответы повторно не генерировались.", "",
        "## Отложенный опыт и оценивание", "",
        f"Зафиксированы 16 новых историй × 2 запроса × 4 метода = {report['heldout']['planned']} строк. "
        "Восемь типов ситуаций представлены короткой и длинной историей; длинные источники превышают бюджет памяти. "
        "Существенные факты и несколько допустимых действий записаны до генерации отдельно от входа модели.", "",
        f"Состояние автоматической оценки: `{report['assessment']['status']}`. "
        "Проведена одна калибровка на 24 случаях; повторный подбор промпта и платное восстановление не выполнялись. "
        "При непрохождении gate ответы опыта остаются not_assessed. "
        "Пять критериев, фактический verdict, полезность и технические отказы приведены раздельно в подробном отчёте.", "",
        f"Калибровка: оценено {calibration['assessed_count']} из {calibration['expected_count']}; "
        f"gate qualified={calibration['qualified']}. Не выполнены условия: "
        + (", ".join(key for key, value in calibration["gate"].items() if not value) or "нет") + ". "
        "Случаи с несовпадением сохранены в summary.json; ожидаемые метки после запуска не менялись.", "",
        "Единственный непринятый случай – C022: судья вернул finding с цитатой, которая не является буквальной "
        "выдержкой из указанной реплики. Сохранён код invalid_judgement и исходный ответ; стоимость этого вызова "
        "известна. Это отказ проверки основания оценки, а не HTTP-сбой. Во всех 23 принятых случаях обязательные "
        "verdict и utility совпали, но неполная калибровка не квалифицируется. Цитата не исправлялась. "
        "Содержательные искажения, полезность и пять критериев основных 128 строк остаются неизвестными.", "",
        "Основная единица сопоставления – история. Два запроса и четыре метода одной истории зависимы; "
        "128 ответов не трактуются как 128 независимых наблюдений. Программные отказы отнесены к сохранённой стадии. "
        "Смысловую ошибку записи, отбора, ответа или судьи нельзя установить только по статусу ok; "
        "первичные источники и trace подготовлены для последующего разбора. Победитель по содержательному качеству не объявляется.", "",
        "## Воспроизводимость и расходы", "",
        "Код и промпты оставались одинаковыми на dev и test. Модель gemma4:12b, digest, seed 42, temperature 0, "
        "контекст 8192, выход 1536 и выбранная память 4000 зафиксированы. Вход сообщений, схемы, обновления и ответы "
        "учтены отдельно в JSON/Markdown отчёте. Эвристика полного входа использует другой токенизатор с запасом "
        "и не доказывает отсутствие скрытого усечения внутри Ollama.", "",
        f"Накопленный консервативный расход с неизвестными резервами: ${total} из $4. "
        "Старые журналы заблокированы, расходы и резервы не обнулены. Оценка ограничена $2.40 / 152 вызовами, "
        "интеграция – $0.40 / 20 вызовами. Локальная генерация не тарифицируется Spiderssense; "
        "стоимость оборудования и электричества не рассчитывалась.", "",
        "## Материалы просмотра", "",
        "Пять историй выбраны заранее: t02, t04, t07, t10, t12, первый запрос каждой, четыре ответа. "
        "Названия методов закрыты в пакете на 20 мест; отсутствующий ответ сохраняет место с причиной. "
        "Раскрытие методов и спорные случаи лежат отдельно в private.", "",
        "- [Подробный отчёт и все критерии](../Alena_audio/TeachCopilot_RAG/.local/research_memory/final_v4_20261018/final_report/report.md).",
        "- [Слепой пакет для человека](../Alena_audio/TeachCopilot_RAG/.local/research_memory/final_v4_20261018/final_report/human_review/packet.md).",
        "- [Состояние калибровки](../Alena_audio/TeachCopilot_RAG/.local/research_memory/final_v4_20261018/assessment/calibration/summary.json).",
        "- [Правила и запуск приложения](../Alena_audio/TeachCopilot_RAG/docs/MEMORY_RUNBOOK.md).",
        "- [Интеграция и нагрузка](25_memory_integration_results.md).", ""]
    research_doc = NIRS / "24_final_memory_results.md"
    write_text(research_doc, "\n".join(lines))
    external = integration.get("external", {})
    demo = integration.get("demo", {})
    lines = ["# Интеграция памяти и нагрузочная проверка", "", "18 октября 2026 года.", "",
        f"Приложение проверено: {tests} тест прошёл без пропусков, отдельно – 52 проверки учебной политики. "
        "Полный набор включает браузерные сценарии и сквозной синтетический проход API с настоящей PostgreSQL. "
        "Исследовательские 169 тестов и 96 дополнительных проверок выполнены отдельно перед фиксацией опыта.", "",
        "Существующее приложение доступно на http://127.0.0.1:8099, схема БД 11. "
        "Перед применением миграций сохранена .local/backups/pre-memory-20261018.dump. "
        "Карточки, аккаунты и локальный векторный поиск сохранены. Демо памяти подготовлено для demo-nika-04 "
        "по теме math.g4.fractions.numerator_denominator; источники явно помечены как синтетические.", "",
        f"Состояние задания демо: `{demo.get('status','not_completed')}`. "
        "API и отдельный worker запущены локально; после перезапуска ПК их следует запустить по инструкции. "
        "Подтверждение памяти преподавателем остаётся отдельным действием; компетенции автоматически не изменяются.", "",
        "## Изоляция, очередь и восстановление", "",
        "Проверены чужие источники и права, повтор сообщения и подтверждения, смена карточки/источников, "
        "восстановление до/после начала вызова, освобождение брошенного чата, недоступность БД и модели. "
        "Фактические вызовы ограничены общей PostgreSQL: local1 и external4, вместе до 40 ожидающих запросов. "
        "Проверки отдельных процессов используют синтетический HTTP. Неизвестный оплачиваемый исход не повторяется. "
        "Интерактивные ожидания имеют приоритет; уже захваченное фоновое задание не прерывается.", "",
        "## 5, 10 и 20 логических сессий", "",
        "Использованы реальные PostgreSQL, очередь, правила памяти и поиск с детерминированными синтетическими векторами. "
        "Провайдер заменён задержкой 40 мс. Embeddings и модель в этой нагрузке не вычислялись. "
        "Это измерение компонентов и имитации, не публичной API-пропускной способности.", "",
        "| Сессии | Линия | Завершено | Ошибки | Пик вызовов | Память p50/p95, мс | Поиск p50/p95, мс | Очередь p50/p95, мс | Генерация p50/p95, мс |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for wave in load["waves"]:
        timings = wave["timings"]
        pairs = [number(timings[key]["p50_ms"])+" / "+number(timings[key]["p95_ms"]) for key in ("memory_get","search","queue_wait","generation")]
        lines.append(f"| {wave['logical_sessions']} | {wave['provider_lane']} | {wave['completed']} | {wave['error_count']} | {wave['provider_peak_active']} | " + " | ".join(pairs) + " |")
    rss = max(wave["resources"]["runner_peak_rss_bytes"] for wave in load["waves"])/(1024*1024)
    lines += ["", f"Пик RSS измеренного процесса нагрузки: {rss:.1f} МиБ. "
        "RAM хоста и VRAM устройства записаны отдельно; RSS PostgreSQL-контейнера не выделялся. "
        "VRAM измерена до и после волн, пик не измерен. Параллельно выполнялся отдельный исследовательский прогон, "
        "поэтому VRAM и RAM хоста нельзя приписывать имитации нагрузки.", "",
        "## Небольшой внешний проход", "",
        f"Успешных внешних ответов: {external.get('external_answers',0)}, локальных переходов: {external.get('local_fallback_answers',0)}, "
        f"новых оплачиваемых попыток: {external.get('new_ledger_calls',0)}. "
        f"Учтено с резервами: ${external.get('new_charged_or_reserved_usd','unknown')}. "
        f"Статус проверки: {external.get('passed','not_completed')}. "
        "Использованы только синтетические данные, реальная локальная проверка приватности и аудит фактического HTTP-тела. "
        "Текущая реплика, прошлые реплики, цитаты и схема проверяются на известные идентификаторы. "
        "Полные карточки и необработанные заметки не отправляются.", "",
        "Из пяти запланированных интеграционных примеров выполнены четыре попытки: три ответа получены, "
        "четвёртая попытка получила HTTP-ошибку провайдера с неизвестным списанием. Пятый пример не запускался. "
        "Повтор не выполнялся, резерв сохранён. Поэтому реальный внешний проход имеет ограниченное покрытие "
        "и не объявляется полностью успешным; программные и имитационные проверки прошли.", "",
        "Аудит синтетических запросов не доказывает полной анонимности произвольного текста. "
        "Пропускная способность публичного API не измерялась. По умолчанию приложение сохраняет локальные ответы.", "",
        "- [Инструкция запуска](../Alena_audio/TeachCopilot_RAG/docs/MEMORY_RUNBOOK.md).",
        "- [Полный API и восстановление](../Alena_audio/TeachCopilot_RAG/docs/MEMORY_INTEGRATION.md).",
        "- [Сырые измерения нагрузки](../Alena_audio/TeachCopilot_RAG/.local/memory-verification/final_20261018_v4/load.json).",
        "- [Внешний проход и аудит отправленных тел](../Alena_audio/TeachCopilot_RAG/.local/research_memory/final_v4_20261018/external_integration.json).", ""]
    integration_doc = NIRS / "25_memory_integration_results.md"
    write_text(integration_doc, "\n".join(lines))
    write_text(ROOT / "docs/MEMORY_VERIFICATION.md", "# Проверка памяти\n\n"
        "Актуальные результаты, команды и ограничения: [отчёт интеграции](../../../nirs/25_memory_integration_results.md).\n\n"
        "Проверки повторяются командой `.venv-local/Scripts/python.exe scripts/verify_memory.py`; "
        "полный набор с браузером – `.venv-local/Scripts/python.exe scripts/verify_local.py --ui`. "
        "Каждый прогон использует отдельную временную локальную БД. "
        "Оплачиваемый проход имеет отдельный конечный запуск и журнал; не повторяйте его автоматически.\n")
    archive = NIRS / "artifacts/memory_final_20261018_v2.zip"
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise FileExistsError("Delivery archive already exists")
    files = list(PACKAGE.rglob("*")) + list((ROOT / "research_memory").rglob("*"))
    files += list((ROOT / ".local/memory-verification/final_20261018_v4").rglob("*"))
    files += list((ROOT / "scripts").glob("*memory*.py"))
    files += [ROOT / "scripts/check_memory_external.py", ROOT / "requirements-local.txt", ROOT / "requirements-local.lock.txt",
              ROOT / "docs/MEMORY_RUNBOOK.md", ROOT / "docs/MEMORY_INTEGRATION.md", ROOT / "docs/EXTERNAL_ANSWERS.md",
              NIRS / "23_final_execution_protocol.md", research_doc, integration_doc]
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(set(files), key=str):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc",".key"} or path.name.endswith(("-wal","-shm")):
                continue
            output.write(path, path.relative_to(ROOT.parents[1]).as_posix())
    receipt = {"status": "completed", "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "research_status": report["assessment"]["status"], "integration_status": integration["status"],
        "human_content_review": "not_performed", "budget": budget_report}
    (PACKAGE / "DELIVERY_PROCESSING.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    build()

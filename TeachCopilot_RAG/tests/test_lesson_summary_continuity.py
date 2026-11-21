"""Two assigned lessons and ordinary recall against a disposable PostgreSQL database."""
from dataclasses import replace
import json
import multiprocessing
import re

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from local_app.store import Store
from local_app.unified_memory import UnifiedMemoryService
from tests.test_unified_memory_postgres import LEARNER, OTHER_LEARNER, accounts, unified_store


TOPIC = "math.g4.numbers.division_by_1_2_digit"
FIRST = ("В примере с маяком я разделила 1200 на 10 и получила 12: потеряла ноль. "
         "В следующий раз хочу потренироваться делить на 10, чтобы не терять ноль. "
         "Можно сегодня одну задачу про воздушные шары?")
SECOND = "В примере с мостом я разделила 450 на 10 и получила 45."
RECALL = ("Вспомни мой первый урок с маяком про деление на 10. "
          "Процитируй дословно мою первую фразу о том, что у меня получилось.")
CONTEXT_PREFIX = "Предыдущий завершённый урок (JSON): "
SCRIPTED_LESSON = """# Урок: Деление на 10, 20 минут
## Цель
Сохранять нужные нули в частном.
## Разминка
Сколько десятков в 1200?
## Объяснение
В 1200 помещается 120 десятков.
## Практика
1. 1200 : 10
2. 450 : 10
3. 600 : 10
## Проверка
Сколько получится при делении 800 на 10?
## Ответы
120, 45, 60, 80
"""


class RecordingRuntime(OllamaRuntime):
    """Observe the final Ollama requests and optionally replace the transport."""

    def __init__(self, settings, *, live, recall_mode="full_quote"):
        super().__init__(settings)
        self.live = live
        # Scripted answer to the recall question: the full source text or a paraphrase.
        self.recall_mode = recall_mode
        self.requests = []

    def _available_models(self):
        if self.live:
            return super()._available_models()
        return {name: {"name": name, "digest": "synthetic"} for name in
                (self.settings.chat_model, self.settings.embedding_model)}

    def _post(self, path, payload):
        if path == "/api/chat":
            self.requests.append(payload)
        if self.live:
            return super()._post(path, payload)
        if path == "/api/embed":
            return {"embeddings": [[1.0] + [0.0] * 1023 for _ in payload["input"]]}
        assert path == "/api/chat"
        messages = payload["messages"]
        system, latest = messages[0]["content"], messages[-1]["content"]
        if system.startswith("Ты опытный педагог"):
            answer = SCRIPTED_LESSON
        elif system.startswith("Ты доброжелательный учебный помощник"):
            answer = "Проверь частное умножением.\n[Этап: практика]"
        elif system.startswith("You summarize ONE"):
            wrong = "маяком" in latest
            answer = {"studied": ["Деление на 10."],
                      "achieved": ["Получила 12 вместо 120." if wrong else "Получила 45."],
                      "repeat": ["Нули в частном."] if wrong else [],
                      "next_step": ["Закрепить деление на 10, сохраняя нужный ноль."
                                    if wrong else "Проверить частное умножением."]}
        elif system.startswith("You merge per-chunk"):
            answer = json.loads(latest)["chunk_summaries_untrusted"][0]
        elif system.startswith("You extract durable"):
            answer = {"changes": [], "candidates": []}
        elif system.startswith("You classify and resolve"):
            answer = {"in_scope": False, "uses_prior_context": False, "search_query": "",
                      "personal_memory_recall": True}
        elif "<own_conversation_retrieval_untrusted>" in system and FIRST in system:
            answer = {"full_quote": "«" + FIRST + "»",
                      "first_sentence": "«" + FIRST.split(". ", 1)[0] + ".»",
                      "paraphrase": "Ты рассказывала про деление на 10 и потерянный ноль."}[self.recall_mode]
        else:
            answer = "В найденных репликах этого нет."
        return {"message": {"content": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)},
                "prompt_eval_count": 500, "eval_count": 50, "done": True}


def request(client, method, path, status, **kwargs):
    response = client.request(method, path, **kwargs)
    assert response.status_code == status, response.text
    return response.json()


def login(client, account, password):
    body = request(client, "POST", "/auth/login", 200,
                   json={"email": account["email"], "password": password})
    return {"Authorization": "Bearer " + body["access_token"]}


def chat(client, headers, session_id, message_id, content, *, guided=False):
    return request(client, "POST", "/v1/chat/completions", 200, headers=headers, json={
        "learning_session_id": session_id, "client_message_id": message_id,
        "messages": [{"role": "user", "content": content}],
        "metadata": {"lesson_mode": "guided"} if guided else {}})


def make_app(settings, store, runtime, trace=None):
    return create_app(settings=settings, store=store, runtime=runtime, memory_trace_sink=trace,
                      rag_service=RagService(store=store, settings=settings, runtime=runtime))


def run_job(worker, *, kind, job_id=None, session_id=None):
    while (job := worker.run_once()) is not None:
        if job["kind"] != kind:
            continue
        assert not job_id or job["job_id"] == job_id, job
        assert not session_id or job["payload"]["learning_session_id"] == session_id, job
        assert job["status"] == "completed", job.get("error")
        return job["result"]
    pytest.fail(f"Worker did not process {kind}")


def generate(client, headers, worker, runtime, learner, number):
    first_call = len(runtime.requests)
    queued = request(client, "POST", f"/learners/{learner}/lessons", 202, headers=headers, json={
        "topic_id": TOPIC, "minutes": 20, "client_request_id": f"generate-{number}"})
    lesson = run_job(worker, kind="lesson_generate", job_id=queued["job_id"])
    calls = [item for item in runtime.requests[first_call:]
             if item["messages"][0]["content"].startswith("Ты опытный педагог")]
    assert calls and all(item["options"]["num_ctx"] == runtime.context_tokens for item in calls)
    prompts = [item["messages"][-1]["content"].splitlines() for item in calls]
    return lesson, prompts


def assign_and_start(client, teacher, student, lesson, number):
    path = f"/learners/{LEARNER}/lessons/{lesson['lesson_id']}"
    lesson_doc = request(client, "GET", path, 200, headers=teacher)
    assert "## Ответы" in lesson_doc["lesson_text"]
    request(client, "POST", path + "/confirm", 200, headers=teacher,
            json={"expected_revision": lesson_doc["revision"], "request_id": f"confirm-{number}"})
    assignment = request(client, "POST", path + "/assignments", 201, headers=teacher,
                         json={"expected_lesson_revision": lesson_doc["revision"],
                               "request_id": f"assign-{number}"})
    assert assignment["status"] == "pending"
    assert not assignment["learning_session_id"]
    started = request(client, "POST", f"/learners/{LEARNER}/lesson-assignments/{assignment['assignment_id']}/start",
                      200, headers=student, json={"request_id": f"start-{number}"})
    return started["learning_session_id"]


def finish_and_read(client, teacher, student, service, teacher_id, worker, session_id, original):
    store = service.store
    path = f"/learning-sessions/{session_id}/complete"
    ended = request(client, "POST", path, 202, headers=student)
    assert ended["status"] == "completed"
    assert ended["session_processing_status"] == "queued"
    assert service.previous_lesson_context(LEARNER, teacher_id) is None

    processed = run_job(worker, kind="session_process", session_id=session_id)
    result = request(client, "GET", f"/learners/{LEARNER}/learning-sessions/{session_id}/processing-result",
                     200, headers=teacher)
    assert result["status"] == "completed" and not result["provisional"], result
    assert result["run_id"] == processed["run_id"]
    replay = request(client, "POST", path, 202, headers=student)
    assert (replay["completed_at"], replay["job_id"]) == (ended["completed_at"], ended["job_id"])
    assert worker.run_once() is None
    versions = store.list_unified_summary_versions(LEARNER, session_id)
    assert len(versions) == 1 and str(versions[0]["id"]) == result["summary"]["summary_id"]
    assert versions[0]["text"] == result["summary"]["text"]
    sources = store.unified_session_snapshot(LEARNER, session_id)["sources"]
    source = next(row for row in sources if row["author_role"] == "student")
    assert source["text"] == original
    assert any(row["author_role"] == "assistant" and row["text"] for row in sources)
    assert source["source_id"] in {ref["source_id"] for ref in result["summary"]["source_refs"]}
    return result["summary"], source["source_id"]


def recall_after_restart(settings, live, student_account, other_account, password, old_source_id, old_session,
                         recall_mode="full_quote"):
    store = Store(settings.database_url)
    runtime = RecordingRuntime(settings, live=live, recall_mode=recall_mode)
    events = []
    with TestClient(make_app(settings, store, runtime, events.append)) as client:
        # The other learner goes first so isolation is checked whatever happens to the quote.
        for learner, account in ((OTHER_LEARNER, other_account), (LEARNER, student_account)):
            events.clear()
            runtime.requests.clear()
            headers = login(client, account, password)
            if learner == OTHER_LEARNER:
                request(client, "GET", f"/learners/{LEARNER}/learning-sessions/{old_session}/processing-result",
                        403, headers=headers)
            started = request(client, "POST", f"/learners/{learner}/learning-sessions", 201,
                              headers=headers, json={"topic_id": TOPIC, "client_session_id": "after-restart"})
            reply = chat(client, headers, started["learning_session_id"], "recall", RECALL)
            answer = reply["choices"][0]["message"]["content"]
            answer_events = [event for event in events if event["phase"] == "answer_prompt"]
            if learner == LEARNER:
                assert len(answer_events) == 1, events
                final = answer_events[0]
                assert final["route"] == "personal_recall", events
                assert old_source_id in final["delivered_source_ids"]
                assert any(FIRST in row["content"] for row in runtime.requests[-1]["messages"])
                assert FIRST.split(". ", 1)[0] in answer, answer
                check = final.get("quote_check")
                assert check, final
                assert check["status"] in {"model_verbatim", "inserted"}, check
                assert check["selected_source_id"] == old_source_id, check
                assert (check["match"] is None) == (check["status"] == "inserted"), check
                if check["status"] == "model_verbatim":
                    assert check["match"] in {"full", "sentence"}, check
                if not live:
                    status, match = {"full_quote": ("model_verbatim", "full"),
                                     "first_sentence": ("model_verbatim", "sentence"),
                                     "paraphrase": ("inserted", None)}[recall_mode]
                    assert (check["status"], check["match"]) == (status, match), check
                print(f"quote_check={check['status']} match={check['match']}")
            else:
                assert all(old_source_id not in event["delivered_source_ids"] for event in answer_events)
                assert all(FIRST not in row["content"] for call in runtime.requests for row in call["messages"])
                assert FIRST.split(". ", 1)[0] not in answer, answer


@pytest.mark.parametrize("live, recall_mode", [
    (False, "full_quote"), (False, "first_sentence"), (False, "paraphrase"),
    pytest.param(True, "full_quote", marks=pytest.mark.live)],
    ids=["scripted-full", "scripted-sentence", "scripted-paraphrase", "live"])
def test_two_assigned_lessons_and_restart(unified_store, accounts, live, recall_mode, tmp_path):
    store = unified_store
    configured = Settings.from_env()
    context = configured.context_tokens if live else 16384
    assert context in (16384, 32768), "Set LOCAL_CONTEXT_TOKENS to 16384 or 32768"
    settings = replace(configured, database_url=store.dsn, context_tokens=context,
                       generation_scheduler_dsn=None, generation_run_budget_path=None,
                       personal_memory_enabled=True, experimental_personal_graph_enabled=False,
                       memory_response_journal_path=str(tmp_path / "worker.sqlite3"),
                       chat_response_journal_path=str(tmp_path / "chat.sqlite3"))
    runtime = RecordingRuntime(settings, live=live)
    if live:
        assert runtime.check_models()["ready"]
    service = UnifiedMemoryService(store, runtime)
    worker = MemoryWorker(store, runtime)
    teacher_id = accounts.teacher["id"]
    before = service.fast_memory(LEARNER, teacher_id)
    other = store.create_student_account("continuity-other@synthetic.test", "Other student",
                                         hash_password(accounts.password), OTHER_LEARNER, teacher_id)
    with TestClient(make_app(settings, store, runtime)) as client:
        teacher = login(client, accounts.teacher, accounts.password)
        student = login(client, accounts.student, accounts.password)
        first_lesson, first_prompts = generate(client, teacher, worker, runtime, LEARNER, 1)
        assert all(not any(line.startswith(CONTEXT_PREFIX) for line in prompt) for prompt in first_prompts)
        first = assign_and_start(client, teacher, student, first_lesson, 1)
        chat(client, student, first, "wrong-division", FIRST, guided=True)
        first_summary, first_source_id = finish_and_read(client, teacher, student, service,
                                                        teacher_id, worker, first, FIRST)
        context = service.previous_lesson_context(LEARNER, teacher_id)
        assert context and re.search(r"\b(?:10|десять|десяти|десятью)\b", context["next_step"], re.I), context
        summary_text = first_summary["text"]
        # The review item may name the corrected exercise. The full summary
        # must preserve the actual error or the lost zero.
        assert re.search(r"н[оу]л", summary_text, re.I) or {"12", "1200"} <= set(re.findall(r"\d+", summary_text))
        assert context["next_step"] in summary_text
        assert context.get("review_focus") and context["review_focus"] in summary_text
        assert context["source_ref"] == {"kind": "unified_summary_version", "session_id": first,
                                         "lesson_id": first_lesson["lesson_id"],
                                         "summary_id": first_summary["summary_id"]}
        assert context["completed_on"]
        with pytest.raises(PermissionError):
            service.previous_lesson_context(LEARNER, other["id"])

        second_lesson, second_prompts = generate(client, teacher, worker, runtime, LEARNER, 2)
        assert second_lesson["previous_lesson_summary_ref"] == context["source_ref"]
        for prompt in second_prompts:
            assert "аквариумы" in prompt[0]
            line = next(line for line in prompt if line.startswith(CONTEXT_PREFIX))
            assert json.loads(line.removeprefix(CONTEXT_PREFIX)) == context
        second = assign_and_start(client, teacher, student, second_lesson, 2)
        chat(client, student, second, "correct-division", SECOND, guided=True)
        second_summary, _ = finish_and_read(client, teacher, student, service, teacher_id, worker, second, SECOND)
        assert second_summary["summary_id"] != first_summary["summary_id"]
        assert len(store.list_unified_summary_versions(LEARNER, first)) == 1
        assert store.list_unified_summary_versions(LEARNER, first)[0]["text"] == first_summary["text"]
        after = service.fast_memory(LEARNER, teacher_id)
        assert after["personal"] == before["personal"]
        assert after["learning"]["confirmed_observations"] == before["learning"]["confirmed_observations"]

        isolated_lesson, isolated_prompts = generate(client, teacher, worker, runtime, OTHER_LEARNER, 3)
        assert isolated_lesson["previous_lesson_summary_ref"] is None
        assert all(not any(line.startswith(CONTEXT_PREFIX) for line in prompt) for prompt in isolated_prompts)

    # A fresh interpreter rebuilds Store, API, RAG and Ollama runtime before recall.
    process = multiprocessing.get_context("spawn").Process(target=recall_after_restart, args=(
        settings, live, accounts.student, other, accounts.password, first_source_id, first, recall_mode))
    process.start()
    process.join(timeout=420 if live else 30)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Ordinary chat timed out after restart")
    assert process.exitcode == 0

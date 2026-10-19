"""Unknown chat outcomes must stop this finite demonstration without retry."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import requests

from scripts import run_natural_child_demo as demo


def test_timeout_preserves_attempt_and_stops_without_retry_or_confirmation(tmp_path, monkeypatch):
    from local_app.config import Settings
    from local_app import store as store_module

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, *args):
            pass

        def fetchone(self):
            return None

    card = {"learner_id": demo.LEARNER, "pseudonym": "ДЕМО – Ника",
            "learner_model": {"engagement": {"record_type": "synthetic_demo"}}}
    store = SimpleNamespace(
        get_card=lambda learner: card,
        catalog=lambda: {demo.TOPIC: {"title_ru": "Числитель и знаменатель"}},
        _connect=lambda: nullcontext(SimpleNamespace(cursor=lambda: Cursor())),
        get_learner_memory=lambda learner: {"sources": [], "versions": []},
    )
    monkeypatch.setattr(Settings, "from_env", lambda: SimpleNamespace(database_url="unused"))
    monkeypatch.setattr(store_module, "Store", lambda dsn: store)
    monkeypatch.setattr(demo, "attest_local_server", lambda origin: {"external_answer_config_absent": True})
    calls = []

    class Client:
        def __init__(self):
            self.headers = {}

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs.get("json")))
            if url.endswith("/health/ready"):
                result = {"status": "ready"}
            elif url.endswith("/auth/demo/student"):
                result = {"access_token": "private-demo-token",
                          "account": {"learner_id": None, "role": "student"}}
            elif url.endswith("/learners"):
                result = {"learners": [{"learner_id": demo.LEARNER}]}
            elif url.endswith("/learning-sessions"):
                result = {"learning_session_id": "synthetic-session"}
            elif url.endswith("/v1/chat/completions"):
                raise requests.Timeout("synthetic unknown outcome")
            elif url.endswith("/auth/logout"):
                result = {"ok": True}
            else:
                pytest.fail("Unexpected mutation: " + url)
            return SimpleNamespace(status_code=200, json=lambda: result)

        def close(self):
            pass

    monkeypatch.setattr(requests, "Session", Client)
    output = tmp_path / "receipt"
    with pytest.raises(requests.Timeout):
        demo.execute("http://127.0.0.1:8099", output, wait_seconds=1)
    assert len([row for row in calls if row[1].endswith("/v1/chat/completions")]) == 1
    assert not any("/complete" in row[1] or "/confirm" in row[1] for row in calls)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted_without_retry"
    assert receipt["turns"] == []
    assert (output / "turn-01-request.json").exists()
    assert not (output / "turn-01-response.json").exists()
    assert all("private-demo-token" not in item.read_text(encoding="utf-8") for item in output.iterdir())


def test_dry_run_does_not_connect_to_the_app_or_database(monkeypatch, capsys):
    monkeypatch.setattr(demo, "execute", lambda *args, **kwargs: pytest.fail("Dry run attempted a mutation"))
    monkeypatch.setattr("sys.argv", ["run_natural_child_demo.py"])
    assert demo.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["student_messages"][-3:] == ["хз", "ну вроде. а можно про кометы?", "ага"]


@pytest.mark.parametrize("origin", ["https://127.0.0.1", "http://example.test", "http://127.0.0.1@elsewhere.test", "http://127.0.0.1/?api_key=secret"])
def test_external_origins_are_rejected_before_authentication(origin):
    with pytest.raises(ValueError):
        demo.local_origin(origin)

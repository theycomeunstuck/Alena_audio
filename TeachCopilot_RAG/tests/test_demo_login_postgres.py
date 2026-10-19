"""Demo buttons authenticate against actual PostgreSQL summary/full records."""
from dataclasses import replace

from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from tests.test_demo_login import DEMO_PASSWORD, LOCAL_HEADERS, demo_credentials
from tests.test_local_api import FakeRag, FakeRuntime
from tests.test_memory_api_flow import flow_store  # noqa: F401


def test_demo_login_uses_full_synthetic_profiles_and_persists_real_sessions(flow_store, tmp_path):
    store = flow_store
    teacher = store.create_account(
        "demo.rich.teacher@local.test", "Synthetic demo teacher", hash_password(DEMO_PASSWORD),
        "teacher", {"synthetic_demo": True},
    )
    store.assign_learner(teacher["id"], "demo-nika-04")
    store.create_student_account(
        "demo.nika04@local.test", "Synthetic demo student", hash_password(DEMO_PASSWORD),
        "demo-nika-04", teacher["id"],
    )
    assert "learner_model" not in store.list_cards()[0]
    assert store.get_card("demo-nika-04")["learner_model"]["engagement"]["record_type"] == "synthetic_demo"
    settings = replace(Settings(), demo_login_enabled=True,
                       demo_credentials_file=str(demo_credentials(tmp_path / "private-demo.json")))
    app = create_app(settings=settings, store=store, runtime=FakeRuntime(), rag_service=FakeRag())
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as browser:
        assert browser.get("/auth/demo").json() == {"enabled": True, "roles": ["teacher", "student"]}
        for role in ("teacher", "student"):
            response = browser.post(f"/auth/demo/{role}", json={}, headers=LOCAL_HEADERS)
            assert response.status_code == 200, response.text
            auth = {"Authorization": "Bearer " + response.json()["access_token"]}
            assert browser.get("/auth/me", headers=auth).json()["account"]["role"] == role
            learners = browser.get("/learners", headers=auth).json()["learners"]
            assert [row["learner_id"] for row in learners] == ["demo-nika-04"]
            assert browser.post("/auth/logout", headers=auth).status_code == 200
            assert browser.get("/auth/me", headers=auth).status_code == 401

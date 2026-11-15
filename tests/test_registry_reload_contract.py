# Test for the registry-reload response contract (audit F-009).
def test_registry_reload_response_validates_without_id_list():
    from app.models.speaker_models import RegistryReloadResposne

    m = RegistryReloadResposne(count=0)
    assert m.id_list is None


def test_registry_reload_response_with_id_list():
    from app.models.speaker_models import RegistryReloadResposne

    m = RegistryReloadResposne(id_list=["a", "b"], count=2)
    assert m.id_list == ["a", "b"]
    assert m.count == 2


def test_registry_reload_route_response_contract(client, monkeypatch):
    """The model contract above is wired into POST /speaker/registry/reload:
    the route forwards the flag to matcher.reload and only includes id_list
    when the client asked for it. Driven with a fake matcher so no real
    embeddings are read."""
    import app.api.routes_speaker as rs

    class _FakeMatcher:
        def __init__(self):
            self.flags = []

        def reload(self, flag=False):
            self.flags.append(flag)
            if flag:
                return (["u1", "u2"], 2)
            return ([], 2)  # same tuple shape as MultiSpeakerMatcher.reload

    fake = _FakeMatcher()
    monkeypatch.setattr(rs, "matcher", fake, raising=True)

    r = client.post("/speaker/registry/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert body["id_list"] is None, "flag=False must not leak the id list (F-009)"

    r2 = client.post("/speaker/registry/reload?flag=true")
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["count"] == 2
    assert body2["id_list"] == ["u1", "u2"]

    assert fake.flags == [False, True], "the route must forward the flag to matcher.reload"

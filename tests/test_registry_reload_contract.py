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

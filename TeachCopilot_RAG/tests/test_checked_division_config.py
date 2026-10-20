"""The unsuccessful arithmetic scaffold remains an explicit experiment."""
import pytest

from local_app.config import Settings


def test_checked_division_context_is_disabled_by_default():
    assert Settings().checked_division_context_enabled is False


@pytest.mark.parametrize('value,expected', [(None, False), ('0', False), ('1', True), ('true', False)])
def test_checked_division_environment_requires_explicit_one(monkeypatch, value, expected):
    # Do not depend on a developer's private .local.env or unrelated settings.
    monkeypatch.setattr('dotenv.load_dotenv', lambda *args, **kwargs: None)
    for name in ('OLLAMA_URL', 'LOCAL_CONTEXT_TOKENS', 'LOCAL_REQUEST_TIMEOUT',
                 'LOCAL_MAX_OUTPUT_TOKENS', 'LOCAL_SESSION_HOURS', 'LOCAL_MIN_SIMILARITY'):
        monkeypatch.delenv(name, raising=False)
    if value is None:
        monkeypatch.delenv('LOCAL_CHECKED_DIVISION_CONTEXT', raising=False)
    else:
        monkeypatch.setenv('LOCAL_CHECKED_DIVISION_CONTEXT', value)
    assert Settings.from_env().checked_division_context_enabled is expected

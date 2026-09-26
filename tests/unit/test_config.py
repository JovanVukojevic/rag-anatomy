from urllib.parse import unquote, urlsplit

import pytest
from pydantic import ValidationError

from rag_anatomy.config import DatabaseSettings, OpenAISettings

_DATABASE_ENV = {
    "POSTGRES_USER": "rag",
    "POSTGRES_PASSWORD": "p@ss:/w rd",
    "POSTGRES_DB": "rag_anatomy",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
}


@pytest.fixture
def database_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in _DATABASE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return monkeypatch


@pytest.mark.parametrize("missing", sorted(_DATABASE_ENV))
def test_missing_database_variable_fails_fast(
    database_env: pytest.MonkeyPatch, missing: str
) -> None:
    database_env.delenv(missing)
    with pytest.raises(ValidationError):
        DatabaseSettings(_env_file=None)


def test_dsn_escapes_credentials(database_env: pytest.MonkeyPatch) -> None:
    dsn = urlsplit(DatabaseSettings(_env_file=None).dsn)
    assert dsn.scheme == "postgresql"
    assert unquote(dsn.username or "") == "rag"
    assert unquote(dsn.password or "") == "p@ss:/w rd"
    assert (dsn.hostname, dsn.port, dsn.path) == ("127.0.0.1", 5432, "/rag_anatomy")


def test_password_is_not_in_repr(database_env: pytest.MonkeyPatch) -> None:
    assert "p@ss" not in repr(DatabaseSettings(_env_file=None))


def test_database_settings_do_not_need_openai_key(
    database_env: pytest.MonkeyPatch,
) -> None:
    DatabaseSettings(_env_file=None)
    with pytest.raises(ValidationError):
        OpenAISettings(_env_file=None)


def test_empty_openai_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValidationError):
        OpenAISettings(_env_file=None)

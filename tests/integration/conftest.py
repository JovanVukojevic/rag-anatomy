import pytest

from rag_anatomy.adapters.driven.postgres import EMBEDDING_DIMENSIONS
from rag_anatomy.adapters.driven.postgres.store import Pool
from rag_anatomy.config import DatabaseSettings
from tests.fakes import RERANKER_MODEL, RERANKER_REVISION


@pytest.fixture
def openai_env(
    database: DatabaseSettings, pg_pool: Pool, monkeypatch: pytest.MonkeyPatch
) -> pytest.MonkeyPatch:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    return monkeypatch


@pytest.fixture
def reranker_env(openai_env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    openai_env.setenv("RERANKER_ENABLED", "true")
    openai_env.setenv("RERANKER_CANDIDATE_POOL", "20")
    openai_env.setenv("RERANKER_HOST", "tei.test")
    openai_env.setenv("RERANKER_PORT", "80")
    openai_env.setenv("RERANKER_MODEL", RERANKER_MODEL)
    openai_env.setenv("RERANKER_REVISION", RERANKER_REVISION)
    return openai_env

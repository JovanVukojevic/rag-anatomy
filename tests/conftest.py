import os
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import HttpWaitStrategy

from rag_anatomy.adapters.driven.postgres import PgVectorStore, connection_pool
from rag_anatomy.adapters.driven.postgres.store import Pool
from rag_anatomy.config import DatabaseSettings, RerankerServerSettings

_ROOT = Path(__file__).parent.parent
_IMAGE = re.compile(r"^\s*image:\s*(\S+)\s*$", re.MULTILINE)


def _compose_image(repository: str) -> str:
    images = _IMAGE.findall((_ROOT / "docker-compose.yml").read_text())
    for image in images:
        if image.startswith(f"{repository}:"):
            return str(image)
    raise LookupError(f"docker-compose.yml declares no {repository} image")


@pytest.fixture(scope="session")
def database() -> Iterator[DatabaseSettings]:
    with (
        PostgresContainer(
            _compose_image("pgvector/pgvector"), driver=None
        ) as container,
        pytest.MonkeyPatch.context() as env,
    ):
        env.setenv("POSTGRES_USER", container.username)
        env.setenv("POSTGRES_PASSWORD", container.password)
        env.setenv("POSTGRES_DB", container.dbname)
        env.setenv("POSTGRES_HOST", container.get_container_host_ip())
        env.setenv("POSTGRES_PORT", str(container.get_exposed_port(container.port)))
        migrations = Config(toml_file=_ROOT / "pyproject.toml")
        command.upgrade(migrations, "head")
        command.downgrade(migrations, "base")
        command.upgrade(migrations, "head")
        settings = DatabaseSettings()
        # Tiny test tables would be seq-scanned, leaving HNSW behaviour (ef_search,
        # dead tuples, iterative scans, tie order) untested.
        with psycopg.connect(settings.dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("ALTER DATABASE {} SET enable_seqscan = off").format(
                    sql.Identifier(settings.db)
                )
            )
        yield settings


@pytest.fixture
async def pg_pool(database: DatabaseSettings) -> AsyncIterator[Pool]:
    async with connection_pool(database.dsn, max_size=4, timeout=10) as pool:
        async with pool.connection() as conn:
            await conn.execute("TRUNCATE documents, embedding_space CASCADE")
        yield pool


@pytest.fixture
def pg_store(pg_pool: Pool) -> PgVectorStore:
    return PgVectorStore(pg_pool)


@pytest.fixture(scope="session")
def reranker_server() -> Iterator[RerankerServerSettings]:
    pinned = RerankerServerSettings(_env_file=_ROOT / ".env.example")
    cache = _ROOT / ".cache" / "tei"
    cache.mkdir(parents=True, exist_ok=True)
    container = (
        DockerContainer(_compose_image("ghcr.io/huggingface/text-embeddings-inference"))
        .with_command(
            [
                f"--model-id={pinned.model}",
                f"--revision={pinned.revision}",
                "--max-batch-tokens=2048",
                "--tokenization-workers=2",
                "--max-client-batch-size=4",
            ]
        )
        .with_volume_mapping(cache, "/data", "rw")
        .with_exposed_ports(80)
        .with_kwargs(
            user=f"{os.getuid()}:{os.getgid()}", mem_limit="6g", memswap_limit="6g"
        )
        .waiting_for(HttpWaitStrategy(80, "/health").with_startup_timeout(900))
    )
    with container:
        yield pinned.model_copy(
            update={
                "host": container.get_container_host_ip(),
                "port": container.get_exposed_port(80),
            }
        )

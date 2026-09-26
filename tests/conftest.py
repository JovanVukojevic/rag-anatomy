import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from testcontainers.community.postgres import PostgresContainer

from rag_anatomy.adapters.driven.postgres import PgVectorStore, connection_pool
from rag_anatomy.adapters.driven.postgres.store import Pool
from rag_anatomy.config import DatabaseSettings

_ROOT = Path(__file__).parent.parent
_IMAGE = re.compile(r"^\s*image:\s*(\S+)\s*$", re.MULTILINE)


def _compose_image() -> str:
    match = _IMAGE.search((_ROOT / "docker-compose.yml").read_text())
    if match is None:
        raise LookupError("docker-compose.yml declares no image")
    return match.group(1)


@pytest.fixture(scope="session")
def database() -> Iterator[DatabaseSettings]:
    with (
        PostgresContainer(_compose_image(), driver=None) as container,
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
        yield DatabaseSettings()


@pytest.fixture
async def pg_pool(database: DatabaseSettings) -> AsyncIterator[Pool]:
    async with connection_pool(database.dsn, max_size=4, timeout=10) as pool:
        async with pool.connection() as conn:
            await conn.execute("TRUNCATE documents CASCADE")
        yield pool


@pytest.fixture
def pg_store(pg_pool: Pool) -> PgVectorStore:
    return PgVectorStore(pg_pool)
